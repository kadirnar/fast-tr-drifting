# fast-tr-drifting: one-step Turkish TTS, streaming and batched

> **Türkçe özet.** `fast-tr-drifting`, Türkçe metni tek bir ağ geçişiyle (difüzyon adımı olmadan) konuşmaya çeviren
> *drifting* tabanlı TTS modelinin (sürüm v3.2) hızlı çıkarım ve sunucu paketidir. Tonlama, yine drifting ile eğitilmiş
> küçük bir prozodi modelinden örneklenir; ses Vocos v2 ile üretilir. Akustik model (DiT) kare pencerelerinde akış
> halinde çalışır: RTX 5090'da tek bir istekte ilk ses **6 ms** içinde gelir, aynı anda gelen 256 isteğin hepsi ilk
> seslerini **52 ms** içinde alır. Eğitim kodu da pakette: v3.2'yi baştan sona yeniden üretmek için
> [docs/TRAINING.md](docs/TRAINING.md). Çıktılar eğitim deposuyla bit düzeyinde aynıdır ([docs/PARITY.md](docs/PARITY.md)).

A lean inference and serving package for the Turkish TTS model trained with *[Generative Modeling via
Drifting](https://arxiv.org/abs/2602.04770)* (Deng et al., 2026), release v3.2, plus the code that trained it.

- **One step.** The DiT turns noise plus the aligned text into a mel spectrogram in a single forward pass: no diffusion
  or flow steps. It was trained with the drifting objective and the learned kernel temperature of
  [Kyutai's Pocket TTS](https://kyutai.org/blog/2026-09-28-pocket-tts-drifting/); classifier-free guidance is learned,
  so it costs nothing at inference.
- **Sampled intonation.** A small prosody model, also trained with drifting, samples the pitch of every character, so
  each seed reads a sentence with another natural tune. The rhythm stays the duration regressor's.
- **Vocos v2** (13.5 M) turns the mels into 24 kHz audio, vendored here (no `vocos` / `torchaudio` dependency). The
  male and female voices can use `vocos-v2-balanced`, trained further on speaker-balanced batches.
- **Streaming DiT.** The DiT runs on frame windows of each sentence and the vocoder on matching windows, so the
  first 0.34 s of audio does not wait for the whole sentence: 6 ms for any length, all passes as CUDA graphs.
- **Batched serving.** `stream_batched` streams many requests together in rounds; `Serving.fast()` adds length
  buckets, fp16 and `torch.compile`, and a Triton Philox kernel draws every request's noise in one launch: 256
  requests arriving at once all get their first audio after 52 ms.
- **Same audio as the training repository**, bit for bit on every deterministic path
  ([docs/PARITY.md](docs/PARITY.md)).

| | |
|---|---|
| **first audio, one request** (RTX 5090) | **5.8 / 6.2 / 6.1 ms** for a short sentence / a long sentence / a paragraph |
| **first audio, N requests at once** | **18 / 29 / 52 ms** for N = 64 / 128 / 256 (`Serving.fast()`), up to 4,680 s of audio per second |
| **quality** ([Freya-TR-Eval](https://huggingface.co/datasets/freyavoice/freya-tr-eval), 495 sentences) | studio voice WER 1.33%, UTMOSv2 3.02 ([below](#quality)) |
| **size** | 67.7 M acoustic model + 8.1 M prosody model + 13.5 M vocoder |
| **voices** | `studio` (default, the clearest), `male`, `female` |

## Install

```bash
pip install "fast-tr-drifting @ git+https://github.com/kadirnar/fast-tr-drifting"   # a CUDA build of PyTorch
pip install "fast-tr-drifting[serve] @ git+https://github.com/kadirnar/fast-tr-drifting"   # + the server example
```

The dependencies are `torch`, `numpy`, `huggingface_hub` and `soundfile`; Triton (for the noise kernel) comes with
the CUDA builds of PyTorch on Linux, and a PyTorch fallback runs elsewhere. The weights (about 400 MB) are downloaded
once from [Vyvo/drifting-tts-tr](https://huggingface.co/Vyvo/drifting-tts-tr), pinned to the revision of release v3.2.

## Quick start

```python
import soundfile as sf
from fast_tr_drifting import Synthesizer

tts = Synthesizer.from_pretrained("v3.2", "cuda")   # acoustic model, prosody predictor, Vocos v2, punctuation pauses
wav, info = tts("Merhaba, bu cümle tek adımda üretildi. Nasıl buldunuz?", speaker="studio", seed=0)
sf.write("merhaba.wav", wav.numpy(), 24000)
```

- `speaker`: `"studio"`, `"male"` or `"female"` (or a training speaker ID).
- `seed`: another seed gives another intonation for the same text.
- `temperature` (default 0.3, the checkpoint's) and `cfg_scale` (default 2): the recommended settings.
- `prosody_temperature` (default 0.5): how freely the pitch is sampled.
- `length_scale` > 1 speaks slower; `pause`: `"punct"` (default: by the final punctuation, measured per voice) or
  seconds.
- `Synthesizer.from_pretrained("v3.2", vocoder="per-voice")`: `vocos-v2` for the studio voice, `vocos-v2-balanced`
  for the male and female voices (higher UTMOSv2, [below](#quality)). `tts.variant(vocoder="vocos-v2-balanced")`
  shares the acoustic model.
- `prosody_durations="sampled"` (opt-in): the prosody model samples the rhythm too; for the studio voice, with
  `prosody_duration_temperature=0.3`.
- Numbers, dates, times, units, currencies and common abbreviations are read out in Turkish.

From the command line:

```bash
fast-tr-drifting say "Merhaba, nasılsınız?" -o merhaba.wav
fast-tr-drifting say "Merhaba! Bu ses parça parça geliyor." -o out.wav --stream --chunked --voice female
```

## Streaming

```python
from fast_tr_drifting import Chunking, Synthesizer

tts = Synthesizer.from_pretrained("v3.2", "cuda", fast=True)   # CUDA graphs, captured once (about 2 s)
for piece in tts.stream("İstanbul Boğazı'nın iki yakası, 1973 yılında açılan köprüyle birbirine bağlandı.",
                        speaker="studio", seed=0, chunked=Chunking()):
    play(piece)   # float32 CPU tensor at 24 kHz; the first piece is 0.34 s, the next ones 2.7 s
```

`fast=True` gives the same audio as without it. `chunked=Chunking()` streams the DiT on frame windows: another
sample of the same model, which scored the same on Freya-TR-Eval. Without it, each sentence's mel is made in one
pass and only the vocoder streams (first audio 7 ms for a long sentence). How the windows work:
[docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md).

## Serving many requests

```python
from fast_tr_drifting import Chunking, Serving, Synthesizer, stream_batched

tts = Synthesizer.from_pretrained("v3.2", "cuda", fast=True)
texts = ["Merhaba!", "Yarın hava nasıl olacak?", "Toplantı saat 14:30'da başlıyor."]
for pieces in stream_batched(tts, texts, speaker="studio", seeds=[0, 1, 2], chunked=Chunking(noise="philox"),
                             serving=Serving.fast()):
    for i, piece in pieces:   # request index, the next float32 piece of its audio
        send(i, piece)
```

Every request has its first piece after the first round; each later round brings the next piece of every request.
With counter-based noise (`noise="philox"`) a request sounds the same alone (`tts.stream`) or in any batch.
`Serving.fast()` compiles its passes on the first batch of a process (about 1.5 minutes): warm up before serving.

**Server example.** `examples/server.py` is a FastAPI server (WebSocket and HTTP chunked streaming of 16-bit PCM)
with dynamic micro-batching: one GPU worker waits a few milliseconds for more requests whenever it is free, sends a
lone request through the CUDA-graph fast path and a group through `stream_batched`, and streams each round's pieces
back to their clients. `examples/client_load.py` opens N WebSocket clients that send at the same instant:

```bash
python examples/server.py --port 8000          # warms up (compiles) first
python examples/client_load.py --clients 1 64 256 --url ws://127.0.0.1:8000/tts
```

Measured on the RTX 5090 with the defaults above (`--window-ms 4`, `Serving.fast()`, per-voice vocoder) and the
clients on the same machine (`--repeats 3`, after one warm-up run; studio voice, the Freya-TR-Eval mix). The
client-side first audio includes the batching window, the WebSocket and the PCM16 conversion:

| clients at once | first audio at the client: p50 / p90 / max | server-side first piece (p50) | all audio done | stalls |
|---|---|---|---|---|
| 1 | 12.1–13.5 / 12.1–13.5 / 12.1–13.5 ms | 11.5–12.7 ms | 21–23 ms | 0 |
| 64 | 26.5–27.6 / 27.3–28.0 / 27.4–28.2 ms | 24.3–25.5 ms | 125–130 ms | 0 |
| 256 | 71.3–85.3 / 76.2–90.0 / 77.9–92.4 ms | 59.4–75.4 ms | 344–379 ms | 0 |

The ranges are the three timed runs. One client pays the 4 ms batching window (the in-process fast path is 6 ms);
at 256 clients the server-side numbers also include collecting the requests (a 4 ms window, at most 30 ms) and
are 7–23 ms above the in-process 52 ms of `benchmarks/bench_concurrency.py`.

## Latency

RTX 5090, release v3.2, studio voice, T = 0.3, α = 2, measured with `benchmarks/` on an otherwise idle GPU
([docs/LATENCY.md](docs/LATENCY.md): setup, definitions, the training repository's numbers they reproduce).

**One request** (`benchmarks/bench_ttfa.py`: CUDA graphs, 100 runs per input; p50 / p90 / p99):

| DiT | short sentence (1.3 s) | long sentence (6.0 s) | paragraph (4 sentences, 17.4 s) | RTF short / long / paragraph |
|---|---|---|---|---|
| **streaming (`Chunking()`)** | **5.8 / 5.9 / 6.3 ms** | **6.2 / 6.3 / 6.4 ms** | **6.1 / 6.4 / 6.7 ms** | 0.0051 / 0.0027 / 0.0030 |
| streaming, counter-based noise | 6.1 / 6.1 / 6.3 ms | 6.5 / 6.5 / 6.6 ms | 6.4 / 6.4 / 6.4 ms | 0.0053 / 0.0028 / 0.0031 |
| whole sentence | 5.8 / 5.8 / 5.9 ms | 7.4 / 7.4 / 7.5 ms | 7.1 / 7.1 / 7.2 ms | 0.0050 / 0.0017 / 0.0020 |

**N requests arriving at the same instant** (`benchmarks/bench_concurrency.py`: Freya-TR-Eval sentences, every fourth
request a paragraph of three, 5.2 s of audio per request on average; streaming DiT with counter-based noise; 5 runs):

| N | one after another (FIFO): first audio p50 / max | batched, `Serving()` fp32: p50 / max | **batched, `Serving.fast()`: p50 / max** | `Serving.fast()`: all done, audio/s |
|---|---|---|---|---|
| 1 | 6.4 / 7.2 ms (fast path) | 13.3 / 21.3 ms | – | – |
| 64 | 645 / 1,275 ms | 36.7 / 36.7 ms | **17.9 / 18.7 ms** | 101 ms, 3,814 s/s |
| 128 | 1,251 / 2,471 ms | 66.6 / 66.8 ms | **29.1 / 30.2 ms** | 171 ms, 4,446 s/s |
| 256 | 2,507 / 4,701 ms | 130.8 / 131.4 ms | **52.0 / 53.6 ms** | 286 ms, 4,679 s/s |

No configuration stalled: every later piece arrived while the earlier audio was still playing. Peak GPU memory is
0.69 GB for one request and 2.5 GB for 256 requests with `Serving.fast()`.

## Quality

[Freya-TR-Eval](https://huggingface.co/datasets/freyavoice/freya-tr-eval) (495 everyday Turkish sentences the model
never saw; seed = sentence index, T = 0.3, α = 2; Whisper large-v3 on 8 kHz audio as in the FreyaTTS report, UTMOSv2
on the full band), measured in the training repository:

| voice | vocoder | WER | UTMOSv2 |
|---|---|---|---|
| studio | `vocos-v2` | **1.33%** | **3.021** |
| male | `vocos-v2-balanced` | 2.43% | 2.944 |
| male | `vocos-v2` | 2.28% | 2.896 |
| female | `vocos-v2-balanced` | 3.91% | 2.749 |
| female | `vocos-v2` | 3.99% | 2.722 |
| *for reference:* Piper (tr, dfki), MMS-TTS (tr) | | 3.76%, 6.26% | |

**The streaming paths passed a quality gate** (all 495 sentences, studio voice; Δ: paired bootstrap 95% interval):

| system | WER | UTMOSv2 | DNSMOS | against |
|---|---|---|---|---|
| whole-sentence DiT, streaming vocoder | 1.36% | 3.021 | 3.346 | |
| streaming DiT (`Chunking()`, R = 64) | 1.25% | 3.020 | 3.345 | ΔWER −0.10 pp [−0.26, +0.05], ΔUTMOSv2 −0.001 [−0.007, +0.005] |
| streaming DiT, counter-based noise | 1.48% | 3.023 | 3.346 | other draws of the same model |
| batched (256), `Serving.fast()`, counter-based noise | 1.46% | 3.024 | 3.338 | ΔWER −0.03 pp [−0.08, +0.00], ΔUTMOSv2 +0.001 [−0.002, +0.004] against the line above |

The training repository's `benchmark` command is ported (`fast-tr-drifting benchmark`, the `eval` extra) and gives
the same transcripts and scores on the same audio.

## Training

`fast_tr_drifting.train` reproduces v3.2 end to end: data preparation from a Hugging Face parquet dataset, data
scoring, the 2-D Mel-MAE feature encoder, the acoustic model with the drifting loss (the official and Kyutai's
learned-temperature fields), duration calibration, the drift prosody predictor, the two Vocos recipes and the
speaker-balanced continuation, and the evaluation (`benchmark`, `evaluate`, `prosody`). The recipe, the configs and
the measured cost on an RTX 5090 (the acoustic model: 13.6 h for 150k steps) are in
[docs/TRAINING.md](docs/TRAINING.md).

```bash
pip install -e ".[train,eval]"
fast-tr-drifting prepare --dataset <hf-dataset-id> --out data/train --f0 --save-audio --dev-size 200
fast-tr-drifting train-mae --config configs/mae2d.yaml --workdir runs/mae2d
fast-tr-drifting train --config configs/acoustic_v3.yaml --workdir runs/acoustic_v3
```

Experiments, BigVGAN fine-tuning, audio-VAE latent spaces, MLX / Swift and ONNX / WebGPU live in the training
repository, [kadirnar/drifting-tts](https://github.com/kadirnar/drifting-tts).

## Repository

| | |
|---|---|
| `fast_tr_drifting/` | `synthesize.py` (Synthesizer, `say`), `fast.py` (CUDA graphs), `chunked.py` (streaming DiT), `batched.py` (batched serving), `noise.py` (counter-based noise, Triton), `vocoder.py` (vendored Vocos), `models/`, `text.py` (Turkish frontend), `hub.py` (the release table) |
| `fast_tr_drifting/train/` | training and evaluation ([docs/TRAINING.md](docs/TRAINING.md)) |
| `benchmarks/` | `bench_ttfa.py`, `bench_concurrency.py` ([docs/LATENCY.md](docs/LATENCY.md)) |
| `examples/` | `server.py`, `client_load.py` |
| `scripts/` | `parity.py` ([docs/PARITY.md](docs/PARITY.md)), `export_release.py` |
| `configs/` | the training configs of the release |
| `docs/` | [HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md), [LATENCY.md](docs/LATENCY.md), [PARITY.md](docs/PARITY.md), [TRAINING.md](docs/TRAINING.md) |

Tests: `pytest` (CPU, tiny random models; the GPU tests run when CUDA is available). The parity test against the
training repository runs when `drifting_tts` is importable.

## Limitations

- **The studio voice is the best one.** The male and female voices have 59 and 19 minutes of training data and are
  less intelligible (Freya WER 2.4% and 3.9% against 1.3%).
- **Sampled rhythm is for the studio voice only** (`prosody_durations="sampled"`): for the voices with little data it
  costs words (Freya WER 5.78% male, 11.28% female).
- **Compilation:** `Serving.fast()` compiles on the first batch of a process (about 30 s per pass, three passes; a
  batch of one row compiles once more). Warm up before serving.
- **DNSMOS −0.007 in the batched paths** (3.338–3.339 against 3.345–3.346 one request at a time), fp32 or fp16,
  whole-sentence or streaming DiT; UTMOSv2 and WER do not move, and the cause was not found.
- **One voice per batch;** no continuous batching (requests that arrive while a batch streams wait for the next one).
- **Three voices,** no voice cloning; each sentence is generated without the context of its neighbours.
- **Naturalness** is measured with automatic scores (UTMOSv2, DNSMOS), not by listeners.

## Credits and citation

- **Drifting:** Deng, Li, Li, Du, He. *Generative Modeling via Drifting.* arXiv
  [2602.04770](https://arxiv.org/abs/2602.04770), 2026.
- **Learned kernel temperature:** Kyutai, *Pocket TTS with drifting* (blog, 28 September 2026,
  <https://kyutai.org/blog/2026-09-28-pocket-tts-drifting/>) and [pocket-tts](https://github.com/kyutai-labs/pocket-tts).
- **Vocoder:** Siuzdak. *Vocos: Closing the gap between time-domain and Fourier-based neural vocoders for
  high-quality audio synthesis.* arXiv [2306.00814](https://arxiv.org/abs/2306.00814), 2023; the code is vendored
  from [gemelo-ai/vocos](https://github.com/gemelo-ai/vocos) (MIT), the fine-tunes start from
  [charactr/vocos-mel-24khz](https://huggingface.co/charactr/vocos-mel-24khz).
- **Evaluation set:** [Freya-TR-Eval](https://huggingface.co/datasets/freyavoice/freya-tr-eval) (CC-BY-4.0) and the
  FreyaTTS report's protocol (arXiv 2607.09530).
- **Training repository:** [kadirnar/drifting-tts](https://github.com/kadirnar/drifting-tts). **Models:**
  [Vyvo/drifting-tts-tr](https://huggingface.co/Vyvo/drifting-tts-tr).

```bibtex
@article{deng2026drifting,
  title   = {Generative Modeling via Drifting},
  author  = {Deng, Mingyang and Li, He and Li, Tianhong and Du, Yilun and He, Kaiming},
  journal = {arXiv preprint arXiv:2602.04770},
  year    = {2026}
}
@article{siuzdak2023vocos,
  title   = {Vocos: Closing the gap between time-domain and Fourier-based neural vocoders for high-quality audio
             synthesis},
  author  = {Siuzdak, Hubert},
  journal = {arXiv preprint arXiv:2306.00814},
  year    = {2023}
}
```

## License

The code is MIT ([LICENSE](LICENSE)); the vendored Vocos network is MIT (Copyright (c) 2023 Charactr Inc.). The model
weights on [Vyvo/drifting-tts-tr](https://huggingface.co/Vyvo/drifting-tts-tr) are CC-BY-4.0, per the model repo.
