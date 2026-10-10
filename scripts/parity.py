"""Parity with the training repository: the same audio from the same files, seeds and settings.

Runs the released v3.2 files through both packages, ``fast_tr_drifting`` and the reference ``drifting_tts``
(github.com/kadirnar/drifting-tts at commit a51553a: put a checkout on ``PYTHONPATH``, or install it with its
dependencies), and compares every mel and waveform:

* ``sentence``: ``Synthesizer.from_pretrained("v3.2")`` whole sentences (``mels`` and ``__call__``, eager);
* ``stream``: the single-request fast path (CUDA graphs): streaming vocoder, whole-sentence DiT;
* ``chunked`` / ``chunked-philox``: the streaming DiT on frame windows (``Chunking()``, R = 64), torch / counter-based
  noise, CUDA graphs;
* ``batched-fp32``: ``stream_batched`` of ``--batch`` requests with ``Chunking(noise="philox")`` and ``Serving()``;
* ``batched-fast``: the same with ``Serving.fast()`` (fp16 + ``torch.compile``).

Every voice runs with the release's vocoder (``vocos-v2``); ``male`` and ``female`` also with ``vocos-v2-balanced``
(``sentence``). Request ``i`` of the text set uses seed ``i``, T = 0.3, alpha = 2 and the release's pauses.

Each package runs in a process of its own (``--worker``) and saves its outputs; this process compares them. Two
packages compiled in one process share ``torch.compile``'s kernel and autotuning caches, which can change the fp16
rounding of the compiled passes. ``--floor`` runs the new package a second time for ``batched-fast``: the run-to-run
variation of the compiled passes. Needs a CUDA GPU (about 3 GB).

    PYTHONPATH=.:/path/to/drifting-tts python scripts/parity.py --num 24 --out parity.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import torch

CHECKS = ["sentence", "stream", "chunked", "chunked-philox", "batched-fp32", "batched-fast"]


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# --- worker: one package, every output saved -----------------------------------------------------------------------

@torch.no_grad()
def worker(args) -> None:
    from fast_tr_drifting.evalsets import load_texts
    from fast_tr_drifting.hub import RELEASES, VOCODERS, hub_file

    texts = [r["text"] for r in load_texts(args.texts)][: args.num]
    rel = RELEASES["v3.2"]
    files = {f: hub_file(f) for f in (rel["model"], "prosody_drift_v3.2.pt", *VOCODERS.values())}
    if args.worker == "ref":
        from drifting_tts import batched
        from drifting_tts.chunked import Chunking
        from drifting_tts.synthesize import Synthesizer

        # the same files, passed explicitly (the reference's from_pretrained reads the latest Hub revision)
        def build(**kw):
            return Synthesizer(files[rel["model"]], "cuda", vocoder=files["vocos_v2.pt"],
                               prosody=files["prosody_drift_v3.2.pt"], prosody_durations="regressor", pause="punct",
                               **kw)

        balanced = files["vocos_v2_balanced.pt"]
    else:
        from fast_tr_drifting import batched
        from fast_tr_drifting.chunked import Chunking
        from fast_tr_drifting.synthesize import Synthesizer

        def build(**kw):
            return Synthesizer.from_pretrained("v3.2", "cuda", **kw)

        balanced = "vocos-v2-balanced"
    kw = dict(cfg_scale=args.cfg, temperature=args.temperature)
    akw = kw | {"pause": "punct"}  # stream_batched's own default pause differs between the packages
    out, t0 = {"files": {f: sha256(p)[:16] for f, p in files.items()}}, time.time()
    if "sentence" in args.checks:
        synth = build()
        synths = {"vocos-v2": synth, "vocos-v2-balanced": synth.variant(vocoder=balanced)}
        for voice in args.voices:
            for name in ["vocos-v2"] + (["vocos-v2-balanced"] if voice != "studio" else []):
                s, res = synths[name], []
                for i, text in enumerate(texts):
                    mel = torch.cat(s.mels(text, voice, seed=i, **kw), -1).cpu() if name == "vocos-v2" else None
                    res.append({"mel": mel, "wav": s(text, voice, seed=i, **akw)[0]})
                out[f"sentence/{voice}/{name}"] = res
        del synth, synths
        torch.cuda.empty_cache()
    fast = [c for c in args.checks if c != "sentence"]
    if fast:
        synth = build(fast=True)
        for voice in args.voices:
            for check in fast:
                res = []
                if check in ("stream", "chunked", "chunked-philox"):
                    ch = {"stream": None, "chunked": Chunking(), "chunked-philox": Chunking(noise="philox")}[check]
                    for i, text in enumerate(texts):
                        pieces = list(synth.stream(text, voice, seed=i, chunked=ch, **akw))
                        res.append({"wav": torch.cat(pieces), "pieces": [p.numel() for p in pieces]})
                else:
                    sv = batched.Serving.fast() if check == "batched-fast" else batched.Serving()
                    for s in range(0, len(texts), args.batch):
                        idx = list(range(s, min(s + args.batch, len(texts))))
                        got = {i: [] for i in range(len(idx))}
                        for rnd in batched.stream_batched(synth, [texts[i] for i in idx], speaker=voice, seeds=idx,
                                                          chunked=Chunking(noise="philox"), serving=sv, **akw):
                            for j, piece in rnd:
                                got[j].append(piece)
                        res += [{"wav": torch.cat(got[j]), "pieces": [p.numel() for p in got[j]]}
                                for j in range(len(idx))]
                out[f"{check}/{voice}/vocos-v2"] = res
    out["env"] = {"package": sys.modules[Synthesizer.__module__].__file__, "gpu": torch.cuda.get_device_name(),
                  "torch": torch.__version__, "seconds": round(time.time() - t0, 1),
                  "peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
    torch.save(out, args.dump)


# --- comparison -----------------------------------------------------------------------------------------------------

_LOGMEL = None


def compare(ref: torch.Tensor, new: torch.Tensor, audio: bool) -> dict:
    """Shape equality, bit identity, max |difference|, SNR and (audio) the mean log-mel distance in dB."""
    ref, new = ref.float().flatten(), new.float().flatten()
    out = {"same_shape": ref.shape == new.shape}
    if not out["same_shape"]:
        return out | {"bitexact": False}
    diff = (new - ref).abs()
    err = diff.double().pow(2).sum().item()
    out |= {"bitexact": bool(torch.equal(ref, new)), "max_abs": diff.max().item() if diff.numel() else 0.0,
            "snr_db": math.inf if err == 0 else 10 * math.log10(ref.double().pow(2).sum().item() / err)}
    if audio:
        global _LOGMEL
        if _LOGMEL is None:
            from fast_tr_drifting.audio import make_logmel

            _LOGMEL = make_logmel()
        a, b = (_LOGMEL(x[None]) for x in (ref, new))
        out["logmel_db"] = float((a - b).abs().mean() * 20 / math.log(10))
    return out


def summarize(rows: list[dict]) -> dict:
    out = {"items": len(rows)}
    if "pieces_equal" in rows[0]:
        out["pieces_equal"] = sum(r["pieces_equal"] for r in rows)
    for kind in ("mel", "wav"):
        rs = [r[kind] for r in rows if r.get(kind)]
        if not rs:
            continue
        same = [r for r in rs if r["same_shape"]]
        out[kind] = {"same_shape": len(same), "bitexact": sum(r["bitexact"] for r in rs),
                     "max_abs": max((r["max_abs"] for r in same), default=None),
                     "min_snr_db": min((r["snr_db"] for r in same), default=None)}
        if kind == "wav" and same:
            d = sorted(r["logmel_db"] for r in same)
            out[kind] |= {"logmel_db_median": d[len(d) // 2], "logmel_db_max": d[-1]}
    return out


def compare_dumps(a: dict, b: dict) -> dict:
    checks = {}
    for name, rows_a in a.items():
        if name in ("files", "env") or name not in b:
            continue
        rows = []
        for i, (x, y) in enumerate(zip(rows_a, b[name])):
            row = {"i": i, "wav": compare(x["wav"], y["wav"], True)}
            if x.get("mel") is not None:
                row["mel"] = compare(x["mel"], y["mel"], False)
            if "pieces" in x:
                row["pieces_equal"] = x["pieces"] == y["pieces"]
            rows.append(row)
        checks[name] = summarize(rows) | {"rows": rows}
    return checks


def table(checks: dict) -> str:
    lines = ["| check | items | mel frames equal / bit-exact | samples equal / bit-exact | max abs diff | min SNR (dB) "
             "| log-mel distance median / max (dB) |", "|---|---|---|---|---|---|---|"]
    for name, s in checks.items():
        mel, wav = s.get("mel"), s["wav"]
        snr = wav["min_snr_db"]
        mel_s = f"{mel['same_shape']} / {mel['bitexact']}" if mel else "-"
        lines.append(f"| {name} | {s['items']} | {mel_s} | {wav['same_shape']} / {wav['bitexact']} | "
                     f"{wav['max_abs']:.2g} | {'inf' if snr == math.inf else f'{snr:.1f}'} | "
                     f"{wav.get('logmel_db_median', 0):.4f} / {wav.get('logmel_db_max', 0):.4f} |")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--num", type=int, default=24, help="Freya-TR-Eval sentences (the first ones)")
    p.add_argument("--texts", default="freyavoice/freya-tr-eval")
    p.add_argument("--voices", nargs="+", default=["studio", "male", "female"])
    p.add_argument("--batch", type=int, default=8, help="requests per batch of the batched checks")
    p.add_argument("--checks", nargs="+", default=CHECKS, choices=CHECKS)
    p.add_argument("--temperature", type=float, default=0.3)
    p.add_argument("--cfg", type=float, default=2.0)
    p.add_argument("--floor", action="store_true", help="also run the new package twice for batched-fast")
    p.add_argument("--out", default=None, help="write the results as JSON")
    p.add_argument("--worker", choices=["ref", "new"], help=argparse.SUPPRESS)
    p.add_argument("--dump", help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.worker:
        return worker(args)

    common = ["--num", str(args.num), "--texts", args.texts, "--batch", str(args.batch), "--temperature",
              str(args.temperature), "--cfg", str(args.cfg), "--voices", *args.voices]
    dumps = {}
    with tempfile.TemporaryDirectory() as tmp:
        runs = [("ref", args.checks), ("new", args.checks)]
        if args.floor and "batched-fast" in args.checks:
            runs.append(("new2", ["batched-fast"]))
        for name, checks in runs:
            path = os.path.join(tmp, f"{name}.pt")
            cmd = [sys.executable, __file__, "--worker", name.rstrip("2"), "--dump", path, "--checks", *checks, *common]
            print(f"running {name} ...", flush=True)
            subprocess.run(cmd, check=True)
            dumps[name] = torch.load(path, weights_only=False)
    ref, new = dumps["ref"], dumps["new"]
    if ref["files"] != new["files"]:
        raise SystemExit(f"the packages read different files: {ref['files']} vs {new['files']}")
    res = {"files": new["files"], "env": {"ref": ref["env"], "new": new["env"]},
           "settings": {"num": args.num, "voices": args.voices, "batch": args.batch, "temperature": args.temperature,
                        "cfg": args.cfg, "seeds": "sentence index", "pause": "punct"},
           "checks": compare_dumps(ref, new)}
    if "new2" in dumps:
        res["floor"] = compare_dumps(new, dumps["new2"])
    print("\nfast_tr_drifting against drifting_tts:\n" + table(res["checks"]))
    if "floor" in res:
        print("\nfast_tr_drifting against itself (a second process):\n" + table(res["floor"]))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
