# Latency: one request, and 64 / 128 / 256 requests at once

How long a listener waits for the first audio of release v3.2 on one RTX 5090, alone and when many requests arrive
together, measured with this package's `benchmarks/` on 10 October 2026 on an otherwise idle GPU.

| requests at the same instant | one after another (FIFO), first audio p50 / last | batched, `Serving()` fp32 | **batched, `Serving.fast()`** |
|---|---|---|---|
| 1 (fast path, CUDA graphs) | **5.8 / 6.2 / 6.1 ms** (short / long / paragraph, p50) | – | – |
| 64 | 645 ms / 1.28 s | 37 ms | **18 ms** (max 19) |
| 128 | 1.25 s / 2.47 s | 67 ms | **29 ms** (max 30) |
| 256 | 2.51 s / 4.70 s | 131 ms | **52 ms** (max 54) |

- **One request:** the first 0.34 s of audio is on the host 5.8–6.2 ms after the call, whatever the length of the
  text: the DiT streams on frame windows and every pass runs as a CUDA graph. p99 is at most 0.6 ms above the median.
- **Many requests:** with `stream_batched` every request gets its first piece after the first round. With
  `Serving.fast()` the 256 requests' audio (1,340 s) is all done after 0.29 s: 4,680 s of audio per second.
- **No gaps:** in every configuration each later piece arrived while the earlier audio was still playing (at least
  94 ms of audio still buffered; 256 ms with `Serving.fast()` at N = 256).
- **They reproduce the training repository's measurements** ([below](#reproduced)).

## Setup

- **System:** `Synthesizer.from_pretrained("v3.2", "cuda", fast=True)`: v3.1's acoustic weights, the `drift` prosody
  predictor sampling the token pitch, Vocos v2 (`drifting_tts_v3.2.pt`, `prosody_drift_v3.2.pt`, `vocos_v2.pt`; sha256
  7940508e…, 2d3733e7…, a9e99783…), studio voice, T = 0.3, α = 2, fp32 as shipped (cuDNN may pick TF32 convolution
  kernels, as PyTorch does by default).
- **Streaming DiT:** `Chunking()`: the first window 128 frames, then 256-frame chunks with 64 frames of context on
  each side and a 32-frame crossfade ([HOW_IT_WORKS.md](HOW_IT_WORKS.md#streaming-the-dit-frame-windows)). The
  N-request runs use counter-based noise (`Chunking(noise="philox")`), so request *i* (seed *i*) sounds the same under
  every strategy.
- **Machine:** NVIDIA GeForce RTX 5090 (32 GB, driver 580.173.02, CUDA 13.0), CUDA runtime 12.8, cuDNN 9.19,
  PyTorch 2.11.0+cu128, Triton 3.6, Python 3.12, AMD EPYC 9135 with a 7.7-CPU container quota and 2 torch threads.
  `nvidia-smi` showed no other process on the GPU.

## Definitions

- **TTFA (time to first audio):** from the call with the input text to the first audio piece on the host: text
  normalisation, sentence splitting and the copy from the GPU included; network, audio encoding and playback not
  (the server example below measures those too). A piece is float32 PCM at 24 kHz; the first one is 0.34 s long.
- **One request** (`benchmarks/bench_ttfa.py`): the model warm (one cold call, then 10 warm-up paragraphs), 100 runs
  per input with seeds 0–99. A short sentence (1.3 s of audio), a long sentence (6.0 s), a 4-sentence paragraph
  (17.4 s). **RTF** is the median total time divided by the audio duration.
- **N requests** (`benchmarks/bench_concurrency.py`): all N arrive at the same instant; each request's TTFA runs from
  that instant to its own first piece, so queueing is included.
  - **Mix:** Freya-TR-Eval sentences in order, every fourth request a paragraph of three sentences: for N = 256, 385
    sentences and 1,340 s of audio (5.2 s per request).
  - **Runs:** one warm-up run (it also compiles `Serving.fast()`), then 5 timed runs. TTFA percentiles pool the
    5 × N requests; the other columns are medians over the runs.
  - **all done:** when the last request's last piece arrived. **audio/s:** seconds of audio delivered per second.
  - **buffered:** a client plays each request from its first piece on; the least audio it still had buffered when a
    later piece arrived (below 0: a stall).
  - **peak GB:** `torch.cuda.max_memory_allocated()`, including the 0.68 GB that stays resident (weights and CUDA
    graphs).
- **Strategies:** `fifo`: one request after another through the single-request fast path, as a naive server;
  `batched`: all N requests through `stream_batched` with `Serving()` (fp32, no compilation); `batched-fast`: the same
  with `Serving.fast()` (four length buckets, fp16 text encoder / prosody network / DiT / vocoder, all compiled).

## One request

`benchmarks/bench_ttfa.py`, 100 runs per input; p50 / p90 / p99.

| DiT | short sentence | long sentence | 4-sentence paragraph | RTF short / long / paragraph |
|---|---|---|---|---|
| **streaming (`Chunking()`, the default)** | **5.8 / 5.9 / 6.3 ms** | **6.2 / 6.3 / 6.4 ms** | **6.1 / 6.4 / 6.7 ms** | 0.0051 / 0.0027 / 0.0030 |
| streaming, counter-based noise | 6.1 / 6.1 / 6.3 ms | 6.5 / 6.5 / 6.6 ms | 6.4 / 6.4 / 6.4 ms | 0.0053 / 0.0028 / 0.0031 |
| whole sentence (`--dit whole`) | 5.8 / 5.8 / 5.9 ms | 7.4 / 7.4 / 7.5 ms | 7.1 / 7.1 / 7.2 ms | 0.0050 / 0.0017 / 0.0020 |

- **TTFA no longer grows with the sentence** with the streaming DiT. A short sentence is one window, so it costs what
  it did.
- **The cost is total GPU time:** window overlaps raise it for long inputs (the long sentence takes 16 ms in total
  instead of 10). So a FIFO queue grows faster than with whole sentences.
- **Loading** takes about 2 s (the CUDA graphs are captured then); the first call after loading takes 24–57 ms
  (380 ms with the Triton noise kernel's first compilation). Peak memory: 0.69 GB.
- **Without CUDA graphs** (`--mode stream`) the first audio takes about 15 ms (14.9 ms in the training
  repository's measurement).

## N requests at once

`benchmarks/bench_concurrency.py --requests 1 64 128 256 --strategy fifo batched batched-fast`; streaming DiT with
counter-based noise. Times in ms.

| N | strategy | TTFA p50 | p90 | p99 | max | all done | audio/s | buffered | peak GB |
|---|---|---|---|---|---|---|---|---|---|
| 1 | fifo (fast path) | 6.4 | 6.8 | 7.2 | 7.2 | – | – | 336 | 0.69 |
| 1 | batched | 13.3 | 16.8 | 17.7 | 21.3 | – | – | 332 | 0.69 |
| 64 | fifo | 645 | 1,126 | 1,238 | 1,275 | 1,258 | 305 | 336 | 0.69 |
| 64 | batched | 36.7 | 36.7 | 36.7 | 36.7 | 236 | 1,624 | 284 | 1.22 |
| 64 | **batched-fast** | **17.9** | 18.7 | 18.7 | **18.7** | 101 | **3,814** | 323 | 1.15 |
| 128 | fifo | 1,251 | 2,240 | 2,456 | 2,471 | 2,499 | 305 | 337 | 0.69 |
| 128 | batched | 66.6 | 66.8 | 66.8 | 66.8 | 424 | 1,800 | 220 | 1.77 |
| 128 | **batched-fast** | **29.1** | 30.2 | 30.2 | **30.2** | 171 | **4,446** | 298 | 1.61 |
| 256 | fifo | 2,507 | 4,224 | 4,643 | 4,701 | 4,702 | 285 | 336 | 0.69 |
| 256 | batched | 130.8 | 131.4 | 131.4 | 131.4 | 752 | 1,783 | 94 | 2.86 |
| 256 | **batched-fast** | **52.0** | 53.6 | 53.6 | **53.6** | 286 | **4,679** | 256 | 2.54 |

- **FIFO is a queue:** request *k* starts speaking after about 20 ms × *k* (a request of this mix takes about 20 ms
  of GPU time with the streaming DiT), and the GPU runs batch-1 work all the time: about 300 s of audio per second
  whatever N is.
- **Batched times are flat:** p50 ≈ max, because every request gets its first piece in the same round. The round
  grows linearly with N, so it is limited by computation, not by kernel launches.
- **A lone request should take the fast path:** batched at N = 1 costs 13 ms against 6 ms. The server example does
  this. (`Serving.fast()` at N = 1 is not shown: a batch of one row compiles its own specialisation of the passes,
  and in this run one of them was compiled during the timed requests.)
- **Where the time goes at N = 256 with `Serving.fast()`** (first round, from the training repository's profile):
  text frontend 10 ms of CPU time overlapping the GPU, text encoder + prosody predictor 18 ms of GPU time in four
  length groups, alignment and noise under 1 ms, the first DiT window (256 × 128 frames) 17 ms, the first vocoder
  window (256 × 64 frames) 4.5 ms.

## Reproduced

The same measurements in the training repository's `docs/LATENCY.md`, same machine, earlier the same day:

| | this package | training repository |
|---|---|---|
| one request, streaming DiT (p50, short / long / paragraph) | 5.8 / 6.2 / 6.1 ms | 5.8 / 6.3 / 6.1 ms |
| one request, streaming DiT, counter-based noise | 6.1 / 6.5 / 6.4 ms | 6.1 / 6.5 / 6.4 ms |
| one request, whole-sentence DiT | 5.8 / 7.4 / 7.1 ms | 5.8 / 7.4 / 7.0 ms |
| `Serving.fast()`, first audio p50 / max at N = 64 / 128 / 256 | 18 / 19, 29 / 30, 52 / 54 ms | 18 / 19, 29 / 30, 52 / 53 ms |
| `Serving.fast()`, all done at N = 64 / 128 / 256 | 101 / 171 / 286 ms | 100 / 168 / 281 ms |
| `Serving.fast()`, audio/s at N = 256 | 4,679 | 4,772 |
| `Serving()` fp32, counter-based noise, N = 64 / 128 / 256 | 37 / 67 / 131 ms | 38 / 68 / 131 ms |

FIFO with the streaming DiT gives 645 / 1,275 ms (p50 / max) at N = 64 here; the training repository's docs report
566 / 1,099 ms from an earlier run, but its script gives 636 / 1,234 ms on the machine today, the same as this
package (635 / 1,232 ms in a back-to-back run). FIFO is limited by the host (one Python loop per window), so it
depends most on the CPU's load.

## Quality

The streaming DiT and the batched serving path passed a quality gate on Freya-TR-Eval (all 495 sentences, studio
voice, Whisper large-v3 on 8 kHz audio, UTMOSv2 and DNSMOS on the full band), measured in the training repository.
Δ is the paired bootstrap difference with its 95% interval.

| system | WER | CER | UTMOSv2 | DNSMOS | Δ WER (pp) | Δ UTMOSv2 | against |
|---|---|---|---|---|---|---|---|
| streaming vocoder, whole-sentence DiT (reference) | 1.36% | 0.27% | 3.021 | 3.346 | | | |
| **streaming DiT, R = 64** (`Chunking()`) | 1.25% | 0.25% | 3.020 | 3.345 | −0.10 [−0.26, +0.05] | −0.001 [−0.007, +0.005] | reference |
| streaming DiT, R = 32 | 1.43% | 0.29% | 3.012 | 3.347 | +0.08 [−0.15, +0.33] | −0.009 [−0.020, +0.002] | reference |
| streaming DiT, R = 128 | 1.38% | 0.27% | 3.019 | 3.346 | +0.03 [−0.13, +0.18] | −0.002 [−0.007, +0.003] | reference |
| streaming DiT, R = 64, counter-based noise | 1.48% | 0.28% | 3.023 | 3.346 | +0.12 [−0.26, +0.52] | +0.002 [−0.020, +0.023] | reference (other draws) |
| batched (64), streaming DiT, fp32 | 1.25% | 0.25% | 3.020 | 3.339 | +0.00 [−0.08, +0.08] | +0.000 [−0.002, +0.003] | streaming DiT, R = 64 |
| **batched (256), `Serving.fast()`, counter-based noise** | 1.46% | 0.28% | 3.024 | 3.338 | −0.03 [−0.08, +0.00] | +0.001 [−0.002, +0.004] | R = 64, counter-based noise |

- **R = 64 is the smallest lookahead within noise;** `Chunking()` uses it.
- **DNSMOS:** every batched run gives 3.338–3.339 against 3.345–3.346 one request at a time (−0.007; median per
  sentence −0.006). The training repository's whole-sentence batched path gives the same 3.339, so batching causes
  it, not the streaming DiT; UTMOSv2 does not show it, and the cause was not found.
- **Frame counts:** `Serving.fast()` gives 12 of the 495 sentences one frame more or less than its fp32
  single-request counterpart (half precision moves a few durations across a rounding boundary).
- **Voices:** the gate covered the studio voice; the fp16 options were not checked on the male and female voices.

## Memory

| configuration | peak GB (`max_memory_allocated`) |
|---|---|
| resident after loading (weights + CUDA graphs) | 0.68 |
| one request, or FIFO at any N | 0.69 |
| `Serving()` fp32, N = 64 / 128 / 256 | 1.22 / 1.77 / 2.86 |
| `Serving.fast()`, N = 64 / 128 / 256 | 1.15 / 1.61 / 2.54 |

`nvidia-smi` shows more: it also counts the CUDA context and the allocator's cache.

## Caveats

- **Same-instant arrival** is the worst case for FIFO and the best case for batching. Requests arriving over time
  need continuous batching (new requests joining the next round), which is not implemented; the server example
  batches what arrives within a few milliseconds of an idle GPU.
- **Compilation:** `Serving.fast()` compiles the DiT, the text pass and the vocoder on the first batch of a process
  (about half a minute each); a batch of another size reuses them (dynamic shapes), a batch of one row compiles once
  more. Warm a server up before serving (`examples/server.py` does).
- **One voice per batch:** `stream_batched` takes one voice; the server groups requests by voice.
- **This mix only:** throughput depends on the mix (5.2 s of audio per request). Punctuation pauses (`pause="punct"`)
  add silence that costs no GPU time.

## Reproduce

```bash
python benchmarks/bench_ttfa.py --out ttfa.json                       # one request, streaming DiT
python benchmarks/bench_ttfa.py --noise philox --out ttfa_philox.json
python benchmarks/bench_ttfa.py --dit whole --out ttfa_whole.json
python benchmarks/bench_concurrency.py --requests 1 64 128 256 --strategy fifo batched batched-fast \
    --out concurrency.json
```

`bench_concurrency.py` writes the environment (including the GPU processes seen at its start), the configuration and
every request's TTFA of the first timed run to its JSON. `--strategy microbatch --micro-batch 16 32` adds groups in
arrival order, `--dit whole` whole-sentence DiT passes.
