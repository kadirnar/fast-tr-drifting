# How it works

```
text ─► normaliser ─► text encoder ─► prosody predictor ─► DriftDiT (1 pass, frame windows) ─► Vocos (windows) ─► PCM
          (CPU)        durations        token pitch             mel                           audio pieces
```

This page covers the inference side: the model, the single-request fast path, the streaming windows, the batched
serving path and the noise scheme that ties them together. Training is in [TRAINING.md](TRAINING.md), the measured
latencies in [LATENCY.md](LATENCY.md).

## The model

| part | parameters | what it does |
|---|---|---|
| Turkish frontend | – | lower-cases with Turkish rules, reads numbers, dates, times, units, currencies and abbreviations aloud, splits sentences; characters are the symbols (Turkish spelling is close to phonemic) |
| text encoder | 7.2 M | 6 transformer layers over characters (with a blank between them); a prior mean per token, log-durations, a speaker embedding |
| token pitch | 0.4 M | a regressor of the normalised log-F0 of each token, embedded into the text features (FastPitch) |
| prosody predictor | 8.1 M | samples the token pitch (and, opt-in, the durations) from noise in one pass; trained with drifting |
| DriftDiT | 60.0 M | a 12-layer diffusion-transformer-style network that maps noise plus the frame-aligned text condition to a 100-band log-mel **in one pass** |
| Vocos v2 | 13.5 M | ConvNeXt backbone and an inverse-STFT head: log-mel to 24 kHz audio |

**One step.** The DiT is trained with the drifting objective (Deng et al., 2026): its samples are pulled towards the
recordings and pushed away from each other in the feature space of a frozen mel autoencoder, with Kyutai's learned
kernel temperature. Training moves the generator's output distribution; inference is a single network evaluation,
with no diffusion or flow steps. Classifier-free guidance is learned during training (the guidance scale α is an
input of the network), so α = 2 costs nothing at inference.

**Two sources of randomness.** The DiT's Gaussian noise (scaled by the temperature, 0.3) and 32 categorical style
codes. With the prosody predictor's noise they all come from the request's seed, in a fixed order, so a seed gives
the same audio every time.

**Sampled intonation.** The DiT was trained with the recordings' durations and token pitch, so the deterministic
regressors can be replaced at inference. Release v3.2 samples the token pitch with a small drifting-trained network
(prosody temperature 0.5): each seed reads a sentence with another natural tune. The durations stay the regressor's:
sampled durations cost intelligibility for the voices with little data (`prosody_durations="sampled"` is an opt-in
for the studio voice).

**Pauses.** Sentences are generated one by one and joined with the pause the training data has after their final
mark (`.`, `?`, `,`), minus the silence the model already leaves at a sentence's edges, measured per voice
(`fast_tr_drifting/pauses.py`).

## One request: CUDA graphs

`Synthesizer(fast=True)` (`fast_tr_drifting/fast.py`) captures CUDA graphs once, at load time (about 2 s):

- **Text side:** the text encoder, the pitch regressor and the prosody predictor in one graph per token bucket (a
  multiple of 32 tokens). The prosody noise is drawn outside the graph, in the eager order, and copied in.
- **DiT:** one graph per frame bucket (a multiple of 64 frames, up to 1,664). The padding is masked, so the output
  equals the eager model's.
- **Vocoder:** one graph per window size (below).

The graphs replay with the request's inputs. A graph replay launches the whole pass at once, which removes the
Python and kernel-launch overhead that dominates batch-1 inference of these small networks (eager streaming: 15 ms
to the first audio; graphs: 6 ms).

## Streaming the vocoder

`stream_vocoder` vocodes the mel in windows: the first 32 frames (0.34 s of audio) with 32 frames of context on the
right, then 256-frame windows with 32 frames of context on each side. Each window keeps only its own frames. Vocos's
receptive field fits within 32 frames, so the pieces join into the whole-sentence audio at the fp32 noise floor
(86–88 dB SNR on the longest Freya-TR-Eval sentences). The two window sizes are fixed, so both run as CUDA graphs.

## Streaming the DiT (frame windows)

Without it, a sentence's whole mel is generated before its first audio piece, so the time to first audio grows with
the sentence. With `chunked=Chunking()` (`fast_tr_drifting/chunked.py`):

```
frames:   0        64       128               320      384                576      640
window 0: [===commit===|..lookahead..)                                                   run [0, 128),   commit [0, 64)
window 1: [..left..|========== commit 256 ===========|..lookahead..)                     run [0, 384),   commit [64, 320)
window 2:                            [..left..|========== commit 256 ===========|..la..) run [256, 640), commit [320, 576)
                                          ^ the first 32 committed frames blend with the previous lookahead
```

- **Text side once per sentence.** The text encoder and the prosody predictor run over the whole sentence, so the
  durations, the token pitch and the aligned condition are known up front. The DiT's noise and style codes are drawn
  once per sentence.
- **First window:** frames [0, 128), committing [0, 64): the 32 frames of the vocoder's first piece and its 32
  frames of context; the other 64 are the DiT's lookahead (`right`).
- **Later windows** commit the next 256 frames with 64 frames of context on each side (`left`, `right`). Their first
  32 committed frames are blended linearly, in log-mel space, with the previous window's lookahead (`crossfade`).
- **DiT window k commits exactly the frames vocoder window k needs**, so a streaming round runs one DiT window and
  one vocoder window.
- **A sentence of at most 128 frames (1.4 s) is one window:** bit for bit the whole-sentence generation.
- **Why this suits the model:** the DiT was trained on 256-frame crops with the condition cropped alike. Windows of
  128 and 384 frames are closer to that than whole sentences of up to about 1,500 frames.
- **Cost and quality:** the DiT runs over 1.5 times the frames because the windows overlap. The chunked mel is
  another sample of the same model, 0.4 dB from the whole-sentence one on average, and scores the same on
  Freya-TR-Eval (WER 1.25% against 1.36%, UTMOSv2 3.020 against 3.021; [LATENCY.md](LATENCY.md#quality)). Without
  lookahead and crossfade, the joins show up as seams (+4.8% spectral flux at the joins); with them they cannot be
  told from the rest.

## Many requests: batched serving

`stream_batched(synth, texts, ...)` (`fast_tr_drifting/batched.py`) streams many requests of one voice together, in
rounds:

1. **First round:** the text frontend of every request, one padded pass of the text encoder and the prosody
   predictor over every request's first sentence, the alignment, the noise, then the first DiT window of every
   request as one batch and their first vocoder windows as one batch. Every request has its first piece after this
   round.
2. **Later rounds:** one more DiT window and one more vocoder window per unfinished request, each as one batch. A
   request whose sentence ends joins the round's text-side batch with its next sentence.
3. **Masking:** each row of a padded batch is computed as if alone. The DiT's attention masks the padded frames.
   The batched Vocos zeroes the padding before every convolution and leaves it out of the inverse STFT
   (`vocode_masked`). So each request's audio equals the single-request path's with the same seed, up to float
   rounding (tested on tiny models to 1e-5).

`Serving` adds speed options to the batched passes, all off by default; `Serving.fast()` is the measured
configuration:

| option | effect at N = 256 (first round) |
|---|---|
| `buckets=4` (groups of ≥ 64 rows) | the text pass runs per length group (less padding); the next group's CPU frontend overlaps the GPU work of the one before |
| `text_dtype` / `prosody_dtype="fp16"`, `compile_text` | text encoder + prosody predictor 55.5 → 25.9 ms; durations are still rounded from the fp32 regressor |
| `dit_dtype="fp16"`, `compile` | the first DiT window (256 × 128 frames) 52.8 → 16.5 ms: fused RMSNorm, adaLN modulation, SwiGLU, QK-norm + RoPE |
| `vocoder_dtype="fp16"`, `compile_vocoder` | the first vocoder window (256 × 64 frames) 14.1 → 4.5 ms; the inverse STFT stays fp32 |
| `pipeline=True` (not in `fast()`) | yields each length group's first pieces as soon as they are ready: lower median, higher maximum |

Not kept because they did not help: CUDA graphs of the compiled passes (the GPU is the bottleneck at these batch
sizes), `max-autotune`, frontend worker processes, TF32 in the text encoder (it moves a duration).

The fp16 passes change the output slightly (log-mel distance 0.06 dB against the single-request path, median over
32 requests; mel SNR 70 dB for the fp16 DiT). They passed the Freya-495 quality gate on the studio voice (WER 1.46%
against 1.48%, UTMOSv2 3.024 against 3.023 for the same draws in fp32, one at a time).

## Counter-based noise

With `torch.Generator`, each request draws its noise call by call (four calls per sentence), in order, which costs a
thousand small kernel launches for 256 requests and ties the noise to the order of the calls.
`Chunking(noise="philox")` (`fast_tr_drifting/noise.py`) makes every element of a request's noise a function of
`(seed, sentence index, stream, element index)`:

- **Generator:** Philox4x32-10 (the generator of cuRAND and Triton). The key is the seed; the counter is
  `(element // 4, 0, stream, sentence)`; element `i` takes output word `i % 4`.
- **Normals:** pairs of 24-bit uniforms through Box–Muller. **Integers** (the style codes): a word mod 64.
- **Streams:** the prosody predictor's token noise and global noise, the DiT's noise, the style codes.
- **Frame-major layout:** the DiT's noise is indexed `frame * n_mels + channel`, so a window's frames are drawn
  without the rest of the sentence.
- **One kernel per stream and batch:** a Triton kernel fills a whole batch in one launch (per-request noise: 4.9 ms of
  host time and 1,024 calls at N = 256, then 0.1 ms and 4 launches). Without Triton (or on the CPU) the same
  arithmetic runs in PyTorch; the two agree to 2e-6.
- **The same audio alone or in any batch.** A request's noise does not depend on the batch, its padding or the other
  requests, so a request sounds the same through the single-request fast path or in any batch
  (`tests/test_noise.py`). The draws differ from the default `torch.Generator` draws of the same seed: another
  sample, scored the same on Freya-495.

## Determinism and parity

The package is a port of the inference code of the training repository
([kadirnar/drifting-tts](https://github.com/kadirnar/drifting-tts) at a51553a), with its Vocos network vendored.
`scripts/parity.py` runs both on the released files and compares every output ([PARITY.md](PARITY.md)): the eager,
CUDA-graph, streaming-DiT and fp32 batched paths are bit-identical; the compiled fp16 batched path
(`Serving.fast()`) agrees at 143 dB SNR (max 2.4e-7), the same as two runs of this package.
