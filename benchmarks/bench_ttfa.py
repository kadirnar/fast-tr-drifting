"""Time to first audio (TTFA) of one request: a short sentence, a long sentence and a 4-sentence paragraph.

TTFA runs from the call with the input text to the first audio piece on the host (text normalisation, sentence split
and the copy from the GPU included; network, audio encoding and playback not). ``--mode``:

- ``fast`` (default): ``Synthesizer(fast=True).stream``: CUDA graphs for the text encoder + prosody predictor, the
  DiT (over frame buckets) and the vocoder windows;
- ``stream``: the same without CUDA graphs;
- ``sentence``: each sentence vocoded whole (``Synthesizer.__call__``), TTFA grows with the first sentence.

``--dit chunked`` (default) streams the DiT too, on frame windows (``Chunking()``: R = 64, L = 64, 256-frame chunks,
crossfade 32); ``--dit whole`` generates each sentence's mel in one pass. 100 runs per input (seeds 0-99) after one
cold call and 10 warm-up paragraphs; RTF is the median total time over the audio duration.

    python benchmarks/bench_ttfa.py                       # v3.2, fast, streaming DiT
    python benchmarks/bench_ttfa.py --dit whole --out ttfa_whole.json
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict

import numpy as np
import torch

from fast_tr_drifting.audio import SAMPLE_RATE
from fast_tr_drifting.chunked import Chunking
from fast_tr_drifting.synthesize import Synthesizer, split_sentences
from fast_tr_drifting.text import normalize, text_to_ids

TEXTS = {
    "short sentence": "Merhaba, nasılsınız?",
    "long sentence": "İstanbul Boğazı'nın iki yakası, 1973 yılında açılan köprüyle birbirine bağlandı.",
    "4-sentence paragraph": ("Toplantı yarın saat 14:30'da, 2. katta başlayacak. "
                             "Lütfen raporlarınızı yanınızda getirin. "
                             "Prof. Dr. Ayşe Yılmaz, 250 TL'lik bağışın tamamının öğrencilere ayrılacağını söyledi. "
                             "Yapay zekâ modelleri her geçen gün daha hızlı ve daha verimli hâle geliyor."),
}


@torch.no_grad()
def run_sentence(synth: Synthesizer, text: str, seed: int, speaker, temperature: float, cfg: float,
                 chunked=None) -> dict:
    """Sentence by sentence, each vocoded whole: TTFA is the first sentence's audio on the host."""
    dev, sync = synth.device, torch.cuda.synchronize
    g = torch.Generator(device=dev).manual_seed(seed)
    spk, tempo = synth._speaker(speaker)
    voc = synth.vocoder_for(speaker)
    sync()
    t0 = time.perf_counter()
    marks, samples = {}, 0
    for k, sentence in enumerate(split_sentences(normalize(text))):
        text_to_ids(sentence, normalized=True)
        mel = synth._mel(sentence, spk, g, cfg, temperature, tempo)
        wav = voc(synth.stats.denormalize(mel))[0].cpu()
        samples += wav.numel()
        if k == 0:
            marks["ttfa"] = time.perf_counter() - t0
            marks["first_audio_s"] = wav.numel() / SAMPLE_RATE
    marks["total"] = time.perf_counter() - t0
    marks["audio_s"] = samples / SAMPLE_RATE
    return marks


@torch.no_grad()
def run_stream(synth: Synthesizer, text: str, seed: int, speaker, temperature: float, cfg: float,
               chunked=None) -> dict:
    """:meth:`Synthesizer.stream`: TTFA is the arrival of its first piece on the host."""
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    marks, samples = {}, 0
    for piece in synth.stream(text, speaker=speaker, cfg_scale=cfg, temperature=temperature, seed=seed, pause=0.0,
                              chunked=chunked):
        if not marks:
            marks["ttfa"] = time.perf_counter() - t0
            marks["first_audio_s"] = piece.numel() / SAMPLE_RATE
        samples += piece.numel()
    marks["total"] = time.perf_counter() - t0
    marks["audio_s"] = samples / SAMPLE_RATE
    return marks


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--release", default="v3.2")
    p.add_argument("--model", default=None, help="an acoustic model checkpoint instead of the release's")
    p.add_argument("--vocoder", default=None, help="vocos-v2 | vocos-v2-balanced | per-voice | a checkpoint")
    p.add_argument("--mode", choices=["sentence", "stream", "fast"], default="fast")
    p.add_argument("--dit", choices=["chunked", "whole"], default="chunked",
                   help="stream / fast: the DiT on frame windows (chunked) or each sentence in one pass")
    p.add_argument("--noise", choices=["torch", "philox"], default="torch", help="chunked: the noise scheme")
    p.add_argument("--chunk-right", type=int, default=64, help="chunked: the DiT's lookahead in frames")
    p.add_argument("--prosody-durations", choices=["sampled", "regressor"], default=None,
                   help="sampled: the prosody predictor samples the rhythm too (default: the release's)")
    p.add_argument("--prosody-duration-temperature", type=float, default=None)
    p.add_argument("--compile", action="store_true", help="fast: also torch.compile the DiT")
    p.add_argument("--tf32", action="store_true", help="fast: TF32 matmuls (changes the output slightly)")
    p.add_argument("--runs", type=int, default=100)
    p.add_argument("--voice", default="studio")
    p.add_argument("--temperature", type=float, default=0.3)
    p.add_argument("--cfg", type=float, default=2.0)
    p.add_argument("--out", default=None, help="write the results as JSON")
    args = p.parse_args()
    if args.mode == "sentence" and args.dit == "chunked":
        args.dit = "whole"  # a sentence vocoded whole needs its whole mel
    over = {k: v for k, v in (("model", args.model), ("vocoder", args.vocoder),
                              ("prosody_durations", args.prosody_durations)) if v is not None}
    t = time.perf_counter()
    synth = Synthesizer.from_pretrained(args.release, "cuda", fast=args.mode == "fast", compile=args.compile,
                                        tf32=args.tf32, prosody_duration_temperature=args.prosody_duration_temperature,
                                        **over)
    load_s = time.perf_counter() - t
    chunked = Chunking(right=args.chunk_right, noise=args.noise) if args.dit == "chunked" else None
    fn = run_sentence if args.mode == "sentence" else run_stream
    kw = dict(speaker=args.voice, temperature=args.temperature, cfg=args.cfg, chunked=chunked)

    cold = fn(synth, TEXTS["long sentence"], 0, **kw)  # first call after loading: CUDA / cuDNN initialisation
    for _ in range(10):
        fn(synth, TEXTS["4-sentence paragraph"], 1, **kw)
    res = {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "release": args.release,
           "mode": args.mode, "dit": args.dit, "chunking": None if chunked is None else asdict(chunked),
           "compile": args.compile, "tf32": args.tf32, "vocoder": synth.vocoder.name, "voice": args.voice,
           "prosody_durations": synth.prosody_durations, "graphed": synth.acoustic is not None,
           "load_s": round(load_s, 1), "cold_ttfa_ms": round(1000 * cold["ttfa"], 1), "rows": {}}
    print(f"{res['gpu']}, mode {args.mode}, DiT {args.dit}, vocoder {res['vocoder']}, voice {args.voice}: "
          f"load {load_s:.1f} s, cold first call TTFA {res['cold_ttfa_ms']} ms")
    print(f"{'input':22s} {'TTFA p50':>9s} {'p90':>7s} {'p99':>7s} {'1st audio':>9s} {'total':>8s} {'RTF':>7s}")
    for name, text in TEXTS.items():
        runs = [fn(synth, text, seed, **kw) for seed in range(args.runs)]

        def get(k: str, runs: list = runs) -> np.ndarray:
            return np.array([r[k] for r in runs])

        row = {"ttfa_ms_p50": 1000 * np.median(get("ttfa")), "ttfa_ms_p90": 1000 * np.percentile(get("ttfa"), 90),
               "ttfa_ms_p99": 1000 * np.percentile(get("ttfa"), 99), "first_audio_s": get("first_audio_s").mean(),
               "total_ms": 1000 * np.median(get("total")), "audio_s": get("audio_s").mean()}
        row["rtf"] = row["total_ms"] / 1000 / row["audio_s"]
        res["rows"][name] = {k: round(float(v), 4 if k == "rtf" else 2) for k, v in row.items()}
        print(f"{name:22s} {row['ttfa_ms_p50']:7.1f}ms {row['ttfa_ms_p90']:5.1f}ms {row['ttfa_ms_p99']:5.1f}ms "
              f"{row['first_audio_s']:8.2f}s {row['total_ms']:6.1f}ms {row['rtf']:7.4f}")
    res["peak_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
