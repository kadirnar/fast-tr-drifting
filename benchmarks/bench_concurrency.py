"""Time to first audio (TTFA) when N streaming requests arrive at the same instant (docs/LATENCY.md).

Strategies (``--strategy``):

- ``fifo``: one request after another, each streamed to its end with the single-request fast path
  (``Synthesizer(fast=True).stream``: CUDA graphs), as a naive server; request k waits for the k - 1 before it;
- ``batched``: all N requests in one batch (:func:`fast_tr_drifting.batched.stream_batched`, fp32 ``Serving()``): one
  padded pass of the text encoder and prosody predictor, the first DiT window and the first vocoder window of every
  request, so every request gets its first piece after one round; then one batch per round with the next window;
- ``batched-fast``: the same with ``Serving.fast()`` (four length buckets, fp16, ``torch.compile``; the first batch
  compiles, about half a minute per pass, and is not timed);
- ``microbatch``: groups of ``--micro-batch`` requests in arrival order (``Serving()``), each streamed to its end.

By default every strategy streams the DiT on frame windows with counter-based noise (``Chunking(noise="philox")``),
so request ``i`` (seed ``i``) sounds the same under every strategy; ``--dit whole`` generates whole sentences.
The requests are Freya-TR-Eval sentences in order, every fourth request a paragraph of three sentences. Each
(strategy, N) runs once to warm up, then ``--repeats`` timed runs; TTFA percentiles pool all requests of all runs,
the other columns are medians over the runs. N = 1 times ``--single-runs`` single requests one at a time.

    python benchmarks/bench_concurrency.py --requests 1 64 128 256 --strategy fifo batched batched-fast
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from fast_tr_drifting.audio import SAMPLE_RATE
from fast_tr_drifting.batched import Serving, stream_batched
from fast_tr_drifting.chunked import Chunking
from fast_tr_drifting.evalsets import FREYA, request_mix
from fast_tr_drifting.synthesize import Synthesizer
from fast_tr_drifting.text import normalize, split_sentences


def environment() -> dict:
    def sh(*cmd: str) -> str:
        try:
            return subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return ""

    cpu = next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                if line.startswith("model name")), platform.processor()) if Path("/proc/cpuinfo").exists() else ""
    return {"gpu": torch.cuda.get_device_name(), "gpu_memory_gb": round(torch.cuda.mem_get_info()[1] / 2**30, 1),
            "driver": sh("nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"),
            "cuda_runtime": torch.version.cuda, "cudnn": torch.backends.cudnn.version(), "torch": torch.__version__,
            "python": platform.python_version(), "cpu": cpu, "torch_threads": torch.get_num_threads(),
            "other_gpu_processes": [x for x in sh("nvidia-smi", "--query-compute-apps=pid,used_memory",
                                                  "--format=csv,noheader").splitlines()
                                    if x and x.split(",")[0].strip() != str(os.getpid())]}


def run_fifo(synth, texts: list[str], kw: dict) -> list[list[tuple[float, int]]]:
    """Each request streamed to its end, one after another: ``[(time on the host, samples), ...]`` per request."""
    events = [[] for _ in texts]
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i, text in enumerate(texts):
        for piece in synth.stream(text, seed=kw["seeds"][i], **kw["stream"]):
            events[i].append((time.perf_counter() - t0, piece.numel()))
    return events


def run_batched(synth, texts: list[str], kw: dict, serving: Serving,
                group: int | None = None) -> list[list[tuple[float, int]]]:
    """All requests in one batch (``group=None``) or in groups of ``group`` in arrival order."""
    events = [[] for _ in texts]
    group = group or len(texts)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for s in range(0, len(texts), group):
        idx = list(range(s, min(s + group, len(texts))))
        for out in stream_batched(synth, [texts[i] for i in idx], seeds=[kw["seeds"][i] for i in idx],
                                  serving=serving, **kw["stream"]):
            t = time.perf_counter() - t0
            for j, piece in out:
                events[idx[j]].append((t, piece.numel()))
    return events


def playback(ev: list[tuple[float, int]]) -> tuple[float, float]:
    """A client that plays a request from its first piece on: the seconds it waits for audio (stalls) and the least
    audio it still had buffered when a later piece arrived (negative: a stall)."""
    play, total, least = ev[0][0] + ev[0][1] / SAMPLE_RATE, 0.0, float("inf")
    for t, n in ev[1:]:
        if n:
            least = min(least, play - t)
        if t > play:
            total, play = total + t - play, t
        play += n / SAMPLE_RATE
    return total, least


def summarize(runs: list[list[list[tuple[float, int]]]], peaks: list[float]) -> dict:
    """Per-request TTFA percentiles pooled over the runs; per-run aggregates as the median over the runs."""
    ttfa = np.array([ev[0][0] for run in runs for ev in run]) * 1000
    play = np.array([playback(ev) for run in runs for ev in run]) * 1000
    stalls, buffered = play[:, 0], play[:, 1]

    def per_run(f) -> float:
        return float(np.median([f(run) for run in runs]))

    def audio(run) -> float:
        return sum(n for ev in run for _, n in ev) / SAMPLE_RATE

    def done(run) -> float:
        return max(ev[-1][0] for ev in run)

    out = {"runs": len(runs), "requests": len(runs[0]),
           "ttfa_ms": {"p50": np.median(ttfa), "p90": np.percentile(ttfa, 90), "p99": np.percentile(ttfa, 99),
                       "max": ttfa.max(), "mean": ttfa.mean()},
           "all_done_ms": per_run(lambda r: done(r) * 1000), "audio_s": per_run(audio),
           "throughput": per_run(lambda r: audio(r) / done(r)),
           "stalled_share": float((stalls > 0).mean()), "stall_ms_max": float(stalls.max()),
           "buffer_ms_min": float(buffered.min()) if np.isfinite(buffered).any() else None, "peak_gb": max(peaks)}
    out["ttfa_ms"] = {k: round(float(v), 2) for k, v in out["ttfa_ms"].items()}
    return {k: round(v, 3) if isinstance(v, float) else v for k, v in out.items()}


def measure(fn, repeats: int) -> dict:
    fn()  # warm-up: allocator, kernels of these shapes (and the compilation of Serving.fast())
    runs, peaks = [], []
    for _ in range(repeats):
        base = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        runs.append(fn())
        peaks.append(torch.cuda.max_memory_allocated() / 2**30)
    res = summarize(runs, peaks)
    res["resident_gb"] = round(base / 2**30, 3)
    res["ttfa_ms_first_run"] = [round(ev[0][0] * 1000, 2) for ev in runs[0]]  # in arrival order
    return res


def measure_single(fn, texts: list[str], runs: int) -> dict:
    """N = 1: ``runs`` single requests (the first ones of the mix), one at a time on an otherwise idle GPU."""
    one, peaks = [], []
    for warm in (True, False):
        for j in range(min(runs, 10) if warm else runs):
            torch.cuda.reset_peak_memory_stats()
            ev = fn([texts[j]], [j])
            if not warm:
                one.append(ev)
                peaks.append(torch.cuda.max_memory_allocated() / 2**30)
    res = summarize(one, peaks)
    for k in ("all_done_ms", "audio_s", "throughput"):
        res.pop(k)
    audio = [sum(n for _, n in ev[0]) / SAMPLE_RATE for ev in one]
    total = [ev[0][-1][0] for ev in one]
    res["rtf_p50"] = round(float(np.median(np.array(total) / np.array(audio))), 4)
    return res


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--release", default="v3.2")
    p.add_argument("--vocoder", default=None, help="default: the release's (vocos-v2)")
    p.add_argument("--requests", type=int, nargs="+", default=[1, 64, 128, 256])
    p.add_argument("--strategy", nargs="+", choices=["fifo", "batched", "batched-fast", "microbatch"],
                   default=["fifo", "batched", "batched-fast"])
    p.add_argument("--micro-batch", type=int, nargs="+", default=[32], help="group sizes of microbatch")
    p.add_argument("--dit", choices=["chunked", "whole"], default="chunked")
    p.add_argument("--noise", choices=["torch", "philox"], default="philox", help="chunked: the noise scheme")
    p.add_argument("--texts", default=FREYA, help="HF dataset id, .jsonl or .txt (default: Freya-TR-Eval)")
    p.add_argument("--paragraph-every", type=int, default=4, help="every k-th request is a paragraph (0: none)")
    p.add_argument("--paragraph-sentences", type=int, default=3)
    p.add_argument("--repeats", type=int, default=5, help="timed runs per (strategy, N > 1), after one warm-up")
    p.add_argument("--single-runs", type=int, default=100, help="N = 1: single requests timed, after 10 warm-ups")
    p.add_argument("--voice", default="studio")
    p.add_argument("--temperature", type=float, default=0.3)
    p.add_argument("--cfg", type=float, default=2.0)
    p.add_argument("--pause", type=float, default=0.0, help="seconds between a paragraph's sentences")
    p.add_argument("--out", default=None, help="write the results as JSON")
    args = p.parse_args()
    env = environment()  # before loading: other processes on the GPU
    t = time.perf_counter()
    synth = Synthesizer.from_pretrained(args.release, "cuda", fast=True,
                                        **({"vocoder": args.vocoder} if args.vocoder else {}))
    load_s = time.perf_counter() - t
    chunked = Chunking(noise=args.noise) if args.dit == "chunked" else None
    n_max = max(max(args.requests), args.single_runs)
    texts = request_mix(args.texts, n_max, args.paragraph_every, args.paragraph_sentences)
    kw = {"seeds": list(range(n_max)),
          "stream": dict(speaker=args.voice, cfg_scale=args.cfg, temperature=args.temperature, pause=args.pause,
                         chunked=chunked)}
    mix = texts[: max(args.requests)]
    res = {"env": env, "config": {**{k: v for k, v in vars(args).items() if k != "out"},
                                  "vocoder_name": synth.vocoder.name, "load_s": round(load_s, 1)},
           "mix": {"requests": len(mix), "sentences": sum(len(split_sentences(normalize(x))) for x in mix),
                   "paragraphs": sum(len(split_sentences(normalize(x))) > 1 for x in mix)},
           "results": {}}
    print(json.dumps(res["env"]), json.dumps(res["config"]), json.dumps(res["mix"]), sep="\n")
    print(f"{'strategy':16s} {'N':>4s} {'TTFA p50':>9s} {'p90':>8s} {'p99':>8s} {'max':>8s} {'all done':>9s} "
          f"{'audio/s':>8s} {'stalled':>7s} {'buffer':>7s} {'peak GB':>7s}")
    servings = {"batched": Serving(), "batched-fast": Serving.fast(), "microbatch": Serving()}
    for n in args.requests:
        for name in args.strategy:
            for m in args.micro_batch if name == "microbatch" else [None]:
                if name == "microbatch" and m >= n:
                    continue  # one group: the batched strategy
                label = f"microbatch-{m}" if m else name
                if name == "fifo":
                    def fn(ts, seeds, name=name):
                        return run_fifo(synth, ts, {"seeds": seeds, "stream": kw["stream"]})
                else:
                    def fn(ts, seeds, name=name, m=m):
                        return run_batched(synth, ts, {"seeds": seeds, "stream": kw["stream"]}, servings[name], m)
                if n == 1:
                    r = measure_single(fn, texts, args.single_runs)
                else:
                    r = measure(lambda fn=fn, n=n: fn(texts[:n], kw["seeds"][:n]), args.repeats)
                res["results"].setdefault(label, {})[str(n)] = r
                tt = r["ttfa_ms"]
                done = r.get("all_done_ms", 0.0)
                thr = r.get("throughput", 1 / r["rtf_p50"] if r.get("rtf_p50") else 0.0)
                buf = r["buffer_ms_min"] if r["buffer_ms_min"] is not None else float("nan")
                print(f"{label:16s} {n:4d} {tt['p50']:7.1f}ms {tt['p90']:6.1f}ms {tt['p99']:6.1f}ms {tt['max']:6.1f}ms "
                      f"{done:7.1f}ms {thr:8.1f} {r['stalled_share']:7.2f} {buf:7.0f} {r['peak_gb']:7.2f}", flush=True)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(res, f, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
