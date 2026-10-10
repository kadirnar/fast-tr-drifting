# Parity with the training repository

This package ports the inference code of [kadirnar/drifting-tts](https://github.com/kadirnar/drifting-tts) (commit
a51553a, the code behind release v3.2) and vendors the Vocos network. `scripts/parity.py` checks that it gives the
same audio: it runs the released files through both packages, each in a process of its own, with the same seeds and
settings, and compares every mel and waveform.

- **Files:** `drifting_tts_v3.2.pt`, `prosody_drift_v3.2.pt`, `vocos_v2.pt`, `vocos_v2_balanced.pt` (sha256 7940508e…,
  2d3733e7…, a9e99783…, 74198538…), the same files on both sides.
- **Texts:** the first 24 Freya-TR-Eval sentences (one of them has two sentences), seed = sentence index, T = 0.3,
  α = 2, the release's punctuation pauses, all three voices.
- **Machine:** RTX 5090, PyTorch 2.11.0+cu128, 10 October 2026.

| check | path | per voice | studio | male | female |
|---|---|---|---|---|---|
| `sentence` | `from_pretrained("v3.2")`: `mels` and `__call__`, eager, Vocos v2 | mels equal / audio equal | 24 / 24 bit-exact | 24 / 24 bit-exact | 24 / 24 bit-exact |
| `sentence` | the same with `vocos-v2-balanced` | audio equal | – | 24 bit-exact | 24 bit-exact |
| `stream` | single-request fast path (CUDA graphs), streaming vocoder | pieces and audio | 24 bit-exact | 24 bit-exact | 24 bit-exact |
| `chunked` | the streaming DiT (`Chunking()`), CUDA graphs | pieces and audio | 24 bit-exact | 24 bit-exact | 24 bit-exact |
| `chunked-philox` | the same with counter-based noise | pieces and audio | 24 bit-exact | 24 bit-exact | 24 bit-exact |
| `batched-fp32` | `stream_batched`, 3 batches of 8, `Serving()`, streaming DiT, counter-based noise | audio | 24 bit-exact | 24 bit-exact | 24 bit-exact |
| `batched-fast` | the same with `Serving.fast()` (fp16, `torch.compile`) | audio: min SNR, max abs diff | 142.5 dB, 1.8e-7 | 142.7 dB, 1.8e-7 | 142.6 dB, 1.8e-7 |
| `batched-fast`, this package against itself (a second process) | | | 142.7 dB, 1.8e-7 | 142.6 dB, 1.8e-7 | 142.7 dB, 2.4e-7 |

- **Bit-exact** everywhere the code path is deterministic: every mel frame and every waveform sample is the same
  (frame counts, piece sizes and values).
- **`Serving.fast()`** is not bit-exact even against itself: two runs of the compiled fp16 passes differ by up to
  2.4e-7 (143 dB SNR, log-mel distance below 3e-5 dB). The two packages differ by the same amount. The frame counts
  and piece sizes are equal.
- **One process per package.** Two packages compiled in one process share `torch.compile`'s kernel and autotuning
  caches. In that setting, a batch of 2 requests came out up to 0.15 apart in the waveform (0.05 dB log-mel distance,
  the size of fp16 rounding in Vocos's output), in either order of the packages, while a second copy of the same
  package matched the first exactly. Each package in its own process: 1.2e-7. `scripts/parity.py` therefore runs
  them separately.
- **Tiny models on the CPU:** `tests/test_reference_parity.py` makes the same comparison on random tiny models
  (whole sentences, the streaming vocoder, the streaming DiT with both noise schemes, the batched path, sampled and
  regressor durations), bit for bit. It runs when `drifting_tts` (a51553a or later) and `vocos` are importable.
- **The vendored Vocos** gives the same audio as the `vocos` package with the same weights (`tests/test_vocoder.py`,
  CPU, and the `sentence` rows above on the GPU).

Reproduce (about 5 minutes on an RTX 5090, compilation included; under 1.1 GB):

```bash
git clone https://github.com/kadirnar/drifting-tts && git -C drifting-tts checkout a51553a
PYTHONPATH=.:drifting-tts python scripts/parity.py --num 24 --floor --out parity.json
```

The reference needs its own imports (`numba`, `pyworld`, `vocos`, `einops`, `pyyaml`, `tqdm`, …; its
`pyproject.toml`).
