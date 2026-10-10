"""Write the files of a release from training outputs: sanitise, verify, scan (no upload).

Writes ``<out>/vocos_v2.pt`` (``--vocoder``), ``<out>/prosody_drift_v3.2.pt`` (``--prosody``) and, when given,
``<out>/drifting_tts_v3.2.pt`` (``--model``), with only what the loaders read (:mod:`fast_tr_drifting.train.publish`).
Each file is reloaded on the CPU and checked against its source (identical weights / outputs), then scanned for
absolute paths and for the data root's private names (speaker, show and dataset names: counted, never printed).
``MANIFEST.json`` lists sizes and SHA-256 (it names the sources: keep it local).

With ``--data`` and both ``--prosody`` and ``--vocoder``, the edge silence of the release's generated sentences
(``--pause-edges`` held-out sentences per voice) is measured and stored in the prosody checkpoint (``pause_edges``):
``pause="punct"`` then inserts the measured pause minus the edges this pipeline leaves.

    python scripts/export_release.py --out release --model runs/acoustic_v3_add_voice/model_ema.pt \\
        --prosody runs/prosody_drift/prosody_ema.pt --vocoder runs/vocos_v2/vocos_ft_160000.pt --data data/train

Then ``FAST_TR_DRIFTING_HUB_DIR=release fast-tr-drifting say ...`` runs the release from these files exactly as users
get it from the Hub.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

from fast_tr_drifting.models.prosody_net import ProsodyPredictor, export_checkpoint
from fast_tr_drifting.models.tts import load_tts
from fast_tr_drifting.text import text_to_ids
from fast_tr_drifting.train.publish import export_tts, export_vocos, private_terms, scan

TEXTS = ["merhaba, bu cümle tek adımda üretildi.", "yarın saat on dörtte toplantı var mı?"]
VOICE_IDS = (722, 389, 323)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def check_prosody(src: str, dst: Path, tts) -> None:
    a, b = ProsodyPredictor.load(src, "cpu", tts=tts), ProsodyPredictor.load(dst, "cpu", tts=tts)
    assert a.temperature == b.temperature and a.duration_scales == b.duration_scales, "metadata changed"
    n_spk = tts.encoder.spk.num_embeddings
    for text in TEXTS:
        ids = torch.tensor([text_to_ids(text, normalized=True)])
        for spk in [s for s in VOICE_IDS if s < n_spk] or [0]:
            out = []
            for p in (a, b):
                g = torch.Generator().manual_seed(spk)
                out.append(p.predict(tts, ids, torch.tensor([ids.shape[1]]), torch.tensor([spk]), p.temperature or 1.0,
                                     p.duration_scales.get(spk, 1.0), generator=g))
            assert all(torch.equal(x, y) for x, y in zip(*out)), f"prosody differs: {text!r}, speaker {spk}"


def check_vocoder(src: str, dst: Path) -> None:
    from fast_tr_drifting.vocoder import load_vocoder

    a, b = load_vocoder(src, "cpu"), load_vocoder(str(dst), "cpu")
    mel = torch.randn(1, 100, 64, generator=torch.Generator().manual_seed(0)) - 5
    assert torch.equal(a(mel), b(mel)), "vocoder output differs"


def check_tts(src: str, dst: Path) -> None:
    (a, _, sa), (b, _, sb) = load_tts(src), load_tts(dst)
    assert sa == sb and a.duration_scales == b.duration_scales and a.temperature == b.temperature
    assert all(torch.equal(x, y) for x, y in zip(a.state_dict().values(), b.state_dict().values()))


def measure_pause_edges(tts: str, prosody: Path, vocoder: Path, data: str, num: int, device: str) -> dict[int, float]:
    """Leading + trailing silence of ``num`` generated held-out sentences per voice, with this release's parts."""
    from fast_tr_drifting.pauses import PAUSES
    from fast_tr_drifting.synthesize import Synthesizer
    from fast_tr_drifting.train.prosody import edge_silence, edge_texts

    synth = Synthesizer(tts, device, vocoder=str(vocoder), prosody=str(prosody))
    edges = {}
    for v in VOICE_IDS:
        r = edge_silence(synth, edge_texts(data, v, num), v)
        edges[v] = round(r["lead_mean"] + r["trail_mean"], 3)
        print(f"pause edge, speaker {v}: {edges[v]:.3f} s over {r['n']} sentences (the regressors' table: "
              f"{PAUSES.get(v, PAUSES['base'])['edge']:.3f} s)")
    return edges


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", required=True, help="staging directory (laid out like the Hub repo)")
    p.add_argument("--model", required=True,
                   help="the release's acoustic model (model_ema.pt, calibrated); written as drifting_tts_v3.2.pt "
                        "unless --keep-model")
    p.add_argument("--keep-model", action="store_true", help="only read --model (the prosody fingerprint and check)")
    p.add_argument("--prosody", help="train-prosody checkpoint (prosody_ema.pt) -> prosody_drift_v3.2.pt")
    p.add_argument("--vocoder", help="finetune-vocoder Vocos checkpoint (vocos_ft_<step>.pt) -> vocos_v2.pt")
    p.add_argument("--vocoder-name", default="vocos_v2.pt", help="e.g. vocos_v2_balanced.pt for the balanced one")
    p.add_argument("--data", default=None, help="prepared data root: its private names must not appear, and the "
                                                "held-out sentences of the pause edges")
    p.add_argument("--scan", nargs="*", default=[], help="also scan these files (e.g. already published ones)")
    p.add_argument("--pause-edges", type=int, default=200, help="sentences per voice for the edge silence (0: skip)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    terms = private_terms(args.data) if args.data else {}
    manifest, ok = {}, True

    def done(name: str, src: str | None) -> None:
        path = out / name
        manifest[name] = {"bytes": path.stat().st_size, "sha256": sha256(path), "source": src}
        print(f"{name}: {path.stat().st_size / 2**20:.1f} MB, sha256 {manifest[name]['sha256'][:16]}..., "
              f"verified against its source")

    tts_path = args.model
    if not args.keep_model:
        torch.save(export_tts(torch.load(args.model, map_location="cpu", weights_only=False)),
                   out / "drifting_tts_v3.2.pt")
        check_tts(args.model, out / "drifting_tts_v3.2.pt")
        done("drifting_tts_v3.2.pt", args.model)
        tts_path = str(out / "drifting_tts_v3.2.pt")
    if args.vocoder:
        name = args.vocoder_name
        torch.save(export_vocos(torch.load(args.vocoder, map_location="cpu", weights_only=False)), out / name)
        check_vocoder(args.vocoder, out / name)
        done(name, args.vocoder)
    if args.prosody:
        tts = load_tts(tts_path)[0]
        ck = export_checkpoint(torch.load(args.prosody, map_location="cpu", weights_only=False), tts)
        dst = out / "prosody_drift_v3.2.pt"
        torch.save(ck, dst)
        if args.data and args.vocoder and args.pause_edges:
            ck["pause_edges"] = measure_pause_edges(tts_path, dst, out / args.vocoder_name, args.data,
                                                    args.pause_edges, args.device)
            torch.save(ck, dst)
        check_prosody(args.prosody, dst, tts)
        done("prosody_drift_v3.2.pt", args.prosody)

    print("scan (hits per category; 0 = clean):")
    for path in [out / n for n in manifest] + [Path(x) for x in args.scan]:
        hits = scan(path, terms)
        ok &= path.name not in manifest or not any(hits.values())
        print(f"  {path.name}: " + ", ".join(f"{k} {v}" for k, v in hits.items()))
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=1))
    if not ok:
        sys.exit("a staged file contains private strings: do not upload it")


if __name__ == "__main__":
    main()
