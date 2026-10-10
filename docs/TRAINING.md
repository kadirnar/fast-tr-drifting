# Training: reproduce release v3.2 end to end

`fast_tr_drifting.train` holds everything that produced the released model: data preparation and scoring, the Mel-MAE
feature encoder, the acoustic model with the drifting loss, duration calibration, the drift prosody predictor, the
Vocos fine-tunes and the evaluation that gave the published numbers. Inference needs none of it.

```bash
pip install -e ".[train,eval,score,dev]"     # + a CUDA build of PyTorch; UTMOSv2 pulls torchvision: install it first
fast-tr-drifting <command> -h               # every command below takes a config and key=value overrides
```

All costs below were measured on one RTX 5090 (32 GB) in a container with a 7.7-CPU quota; keep `OMP_NUM_THREADS`
and `NUMBA_NUM_THREADS` at 4 or below there. The configs are those of the release (`configs/`).

## The pipeline

| # | stage | command | config | output | cost (RTX 5090) |
|---|---|---|---|---|---|
| 1 | data | `prepare` | – | `data/train` (BigVGAN log-mels, F0, audio) | CPU, `--workers` processes |
| 2 | data scores | `score` | – | `data/train/scores.jsonl` | Whisper large-v3, DNSMOS, ECAPA |
| 3 | feature encoder | `train-mae` | `mae2d.yaml` | `runs/mae2d/mae_ema.pt` | 60k steps, 10–35 it/s, ~1.5 h |
| 4 | acoustic model (v3) | `train` | `acoustic_v3.yaml` | `runs/acoustic_v3/model_ema.pt` | 150k steps, 3.08 it/s, 13.6 h |
| 5 | studio voice (v3.1 = v3.2's acoustic weights) | `prepare`, `merge-data`, `train` | `acoustic_v3_add_voice.yaml` | `runs/acoustic_v3_add_voice/model_ema.pt` | 30k steps, 2.7 h |
| 6 | duration factors | `calibrate-durations` | – | stored in the checkpoint | seconds |
| 7 | prosody targets | `prosody-cache` | – | `runs/prosody_cache/targets.pt` | ~2 min |
| 8 | prosody predictor | `train-prosody` | `prosody_drift.yaml` | `runs/prosody_drift/prosody_ema.pt` | 12k steps, 5–10 it/s, ~6 GB (shared GPU) |
| 9 | Vocos v1 (`vocos-ft`) | `finetune-vocoder` | `vocos_v1.yaml` | `runs/vocos_v1/vocos_ft.pt` | 40k steps, ~4.5 it/s |
| 10 | Vocos v2 (`vocos-v2`) | `finetune-vocoder` | `vocos_v2.yaml` | `runs/vocos_v2/vocos_ft_160000.pt` | 160k steps, ~4.5 it/s, 6.5 GB |
| 11 | Vocos v2 balanced (`vocos-v2-balanced`) | `finetune-vocoder` | `vocos_v2_balance.yaml` | `runs/vocos_v2_balance/vocos_ft_20000.pt` | 2 h 19 min for 30k steps (shared GPU), 6.7 GB |
| 12 | release files | `scripts/export_release.py` | – | `drifting_tts_v3.2.pt`, `prosody_drift_v3.2.pt`, `vocos_v2*.pt` | minutes |
| 13 | evaluation | `benchmark`, `evaluate`, `prosody` | – | the tables below | Whisper large-v3 + UTMOSv2: ~6 GB |

```bash
# 1-2. data: any Hugging Face parquet dataset with `audio` and `text` columns (`speaker`, `quality_score` optional)
fast-tr-drifting prepare --dataset <hf-dataset-id> --out data/train --f0 --save-audio --dev-size 200
fast-tr-drifting score --data data/train
fast-tr-drifting score --data data/train --summary --config configs/acoustic_v3.yaml   # what the filters keep

# 3-4. feature encoder and the acoustic model
fast-tr-drifting train-mae --config configs/mae2d.yaml --workdir runs/mae2d train.steps=60000
fast-tr-drifting train --config configs/acoustic_v3.yaml --workdir runs/acoustic_v3

# 5-6. a second dataset with the new voice, merged after the base corpus (its training utterances listed twice)
fast-tr-drifting prepare --dataset <hf-dataset-id> --out data/new_voice --f0 --save-audio \
    --speaker-name new_voice --val-size 100 --dev-size 100 --val-max-seconds 16
fast-tr-drifting merge-data --out data/train_new_voice data/train data/new_voice --repeat 1 2
fast-tr-drifting train --config configs/acoustic_v3_add_voice.yaml --workdir runs/acoustic_v3_add_voice
fast-tr-drifting calibrate-durations --model runs/acoustic_v3_add_voice/model_ema.pt --data data/train_new_voice \
    --temperature 0.3 --speakers studio male female

# 7-8. the drift prosody predictor (pitch sampled at prosody temperature 0.5 in the release)
M=runs/acoustic_v3_add_voice/model_ema.pt
fast-tr-drifting prosody-cache --model $M --data data/train_new_voice --out runs/prosody_cache/targets.pt
fast-tr-drifting train-prosody --workdir runs/prosody_drift tts=$M cache=runs/prosody_cache/targets.pt \
    train.steps=12000
fast-tr-drifting train-prosody --workdir runs/prosody_drift --calibrate-only tts=$M \
    cache=runs/prosody_cache/targets.pt calibrate.temperature=0.5

# 9-11. Vocos: the first recipe, the second one continued from it, then speaker-balanced batches
fast-tr-drifting finetune-vocoder --config configs/vocos_v1.yaml --workdir runs/vocos_v1 tts.path=$M
fast-tr-drifting finetune-vocoder --config configs/vocos_v2.yaml --workdir runs/vocos_v2 tts.path=$M
fast-tr-drifting finetune-vocoder --config configs/vocos_v2_balance.yaml --workdir runs/vocos_v2_balance tts.path=$M

# 12. the release files (sanitised, verified, scanned; the prosody file gets the pause edges of this pipeline)
python scripts/export_release.py --out release --model $M --prosody runs/prosody_drift/prosody_ema.pt \
    --vocoder runs/vocos_v2/vocos_ft_160000.pt --data data/train_new_voice
python scripts/export_release.py --out release --model $M --keep-model --pause-edges 0 \
    --vocoder runs/vocos_v2_balance/vocos_ft_20000.pt --vocoder-name vocos_v2_balanced.pt

# 13. evaluate them as users get them
export FAST_TR_DRIFTING_HUB_DIR=release
fast-tr-drifting benchmark --release v3.2 --speaker studio --out outputs/freya_studio      # Freya-TR-Eval, all 495
fast-tr-drifting benchmark --release v3.2 --speaker male --vocoder vocos-v2-balanced --out outputs/freya_male
fast-tr-drifting evaluate --model $M --data data/train_new_voice --split val --temperature 0.3 --cfg 2
fast-tr-drifting prosody --data data/train_new_voice --speaker 722 --systems recording predicted release \
    --release-prosody drift --release-prosody-durations regressor --release-vocoder vocos-v2
```

- **Data format:** `prepare` reads the parquet files of a Hugging Face dataset (or local ones with
  `--parquet-glob`): `audio` (bytes of any format soundfile reads) and `text`; `speaker` and `quality_score` are
  optional. It writes 100-band BigVGAN-style log-mels at 24 kHz / hop 256 (93.75 frames per second), WORLD F0 on the
  same frames (`--f0`) and the 24 kHz audio (`--save-audio`, needed by the vocoder fine-tunes and `score`).
- **Splits:** `val` (reported results) and `dev` (tuning) are drawn from seen speakers with at least 20 utterances;
  only `train` is filtered by `data.filters`.
- **Filters** (`acoustic_v3.yaml`): DNSMOS OVRL ≥ 3, speaker similarity ≥ 0.5, 8–20 letters per second, bandwidth ≥
  10 kHz, Whisper CER ≤ 15%. An utterance without scores passes.
- **The released model's data:** a multi-speaker Turkish corpus of about 93 h (722 speakers), merged with 34 h of one
  studio voice (speaker 722) for v3.1. The `male` and `female` voices (speakers 389 and 323) have 59 and 19 minutes.
- **Resuming:** every trainer resumes from `<workdir>/last.pt`. `train.init_from` starts another run from a
  checkpoint (the acoustic model: its weights, a grown speaker table and its learned temperature; Vocos: generator,
  discriminators and their optimiser states).
- **Checking a run:** `evaluate --harmonic` compares the harmonic contrast, global variance and level of generated
  mels with the recordings under the ground-truth alignment and pitch, in seconds and without judges.

## The acoustic model

One step (`fast_tr_drifting/train/train_tts.py`):

1. The text encoder gives hidden states, a prior mean and log-durations; monotonic alignment search (MAS) aligns the
   prior to the target mel, which gives the prior NLL, the duration loss and the token pitch targets (FastPitch).
2. The same random 256-frame window (2.7 s) is cut from the target mel and from the aligned condition.
3. For each of 16 conditions (utterances) a CFG scale α is drawn (density ∝ α^-3 on [1, 4], 10% α = 1) and G = 16
   samples are generated in one pass of the DriftDiT (60 M parameters).
4. Positives: the target window and 7 slightly perturbed views. Negatives: the other samples of the same condition,
   and 16 real windows of *other* utterances from an unconditional bank, weighted by (α − 1)(G − 1) / N: classifier-free
   guidance learned during training, free at inference.
5. The drifting loss is computed in the multi-scale feature space of the frozen 2-D Mel-MAE (80 feature maps with the
   block features, spectral detail and raw-mel features), averaged over maps, plus the prior, duration and pitch
   losses. AdamW (0.9, 0.95), LR 2e-4 constant after 5k warm-up steps, EMA 0.999, bf16, compiled generator.

| choice | evidence (10k-step pilots of the training repository) |
|---|---|
| Kyutai's released learned-temperature field (`drift.mode: kyutai`) | harmonic contrast 0.98 vs 0.93 of the recordings, CER 15.4% vs 17.7% against the paper's fixed temperatures |
| the paper's block features (`every_k_block: 2`), CFG α ∈ [1, 4] | 48 → 80 feature maps, as in the official code |
| G = N = 16 samples / negatives per condition, a 60 M generator | larger per-condition sets beat more conditions at a fixed budget (paper, Table 2) |
| data filtered on measured quality (`score`) | Whisper CER, DNSMOS, bandwidth, speaker purity, speaking rate |
| location subsampling of large feature maps (`max_locations: 1024`), a compiled generator | 3.1 it/s |

What to watch: the drift loss is constant by construction. The useful curves are the distance of the samples' mean
to the target (`centroid_mse`), the spread across samples (`across_sample_std`, stable: no collapse), the learned
kernel temperature (`tau_mean`: 1.0 → 0.072 over the v3 run; Kyutai reports ≈ 0.056) and `p_data`, the kernel mass on
the data. The released v3 checkpoint's harmonic contrast against the recordings is 0.99 / 1.02 / 1.02 (low / mid /
high band; the first recipe without the 2-D Mel-MAE and pitch: 0.56 / 0.32 / 0.44).

**Adding a voice** (v3 → v3.1, `acoustic_v3_add_voice.yaml`): the corpus merged with the new voice, 30k steps at LR
1e-4, the speaker table grown by one row (initialised at the mean of the others), the learned temperature resumed;
the new voice was about 40% of the batches. The existing voices kept their quality (Freya WER male 1.92% → 1.74%,
female 3.78% → 3.02%, partly from the per-voice duration factors). `calibrate-durations` then stores the median
recorded / predicted length per voice (studio 0.935, male 1.150, female 1.194) and the preferred noise temperature
(0.3, chosen on `dev`).

## The drift prosody predictor (v3.2's intonation)

The DiT was trained with ground-truth durations and token pitch, so the deterministic duration and pitch regressors
can be replaced at inference without retraining it. `ProsodyNet` (8.1 M parameters: four RoPE transformer layers,
d 256) reads the frozen encoder states, the speaker embedding, the regressors' own predictions and noise (16 channels
per token, 32 global), and predicts a residual log-duration and token pitch plus a voicing logit per token.

- **Targets** (`prosody-cache`): MAS durations and token pitch of every training utterance under the frozen model.
- **Objective** (`train-prosody`, `net.kind: drift`): 16 samples per utterance, the recording's prosody as the one
  positive, Kyutai's drifting loss at one learned temperature on nine feature maps of the sequence (tokens, first
  differences, windows of 3 / 9 / 27 and 8 / 32 tokens, words, the utterance). AdamW (0.9, 0.999), LR 3e-4, gradient
  clip 3, EMA 0.999. `mse` and `flow` (flow matching) are the baselines on the same backbone.
- **Calibration** (`--calibrate-only`): the preferred prosody temperature (0.5 in the release), per-voice duration
  factors for sampled durations, and optionally a separate temperature for the letters' durations
  (`calibrate.duration_temperature`) and a rhythm table (`calibrate.rhythm`: a voice samples another speaker's
  rhythm, e.g. `{389: 722, 323: 722}`).
- **Why the release samples only the pitch** (`prosody_durations="regressor"`): on Freya-495 with Vocos v2, sampled
  durations gave WER 1.89% (studio), 5.78% (male) and 11.28% (female) against 1.33%, 2.28% and 3.99% with the
  pitch only. The voices with little data cannot render the sampled short sounds on new text; the studio voice can,
  and with the letters at prosody temperature 0.3 its sampled rhythm is as intelligible as the release (WER 1.40%
  against 1.36% over three seed sets). Sampling the pitch raises the F0 spread on held-out studio sentences from 3.21
  semitones (v3.1) to 3.53 (recordings: 3.68).
- **Pause edges:** `scripts/export_release.py --data` measures the leading and trailing silence the release leaves
  around a generated sentence (200 held-out sentences per voice) and stores it in the prosody file, so that
  `pause="punct"` inserts the corpus's pause at each mark minus that edge.

## Vocos

The vocoder is fine-tuned on the acoustic model's own mels under the ground-truth alignment (80% of the batches;
the rest are recorded mels), 64-frame segments, with the pretrained `charactr/vocos-mel-24khz` as the start. Its
ISTFT head switches to `same` padding, so frame *i* is centred on sample *i*·256 + 128 as BigVGAN's mels are.

| recipe | config | what changes | result (Freya, v3.1 mels unless noted) |
|---|---|---|---|
| v1 (`vocos-ft`) | `vocos_v1.yaml` | fresh MPD + MRD, hinge, feature matching, 45 × log-mel L1, 2k discriminator warm-up steps, 40k steps | Freya-100 UTMOSv2 2.627 |
| v2 (`vocos-v2`) | `vocos_v2.yaml` | continues v1: MRD × 1, feature matching × 2, BigVGAN-v2's 7-scale mel loss × 15, an instantaneous-frequency loss × 20 against the recording, cosine LR 5e-5 → 5e-6, 160k steps | Freya-495 UTMOSv2 3.017 (BigVGAN-v2-ft 2.935), WER 1.18% |
| v2 balanced (`vocos-v2-balanced`) | `vocos_v2_balance.yaml` | continues v2 on speaker-balanced batches (studio 30%, male 8%, female 6%, the others by audio^0.5), 20k steps | with v3.2: male UTMOSv2 2.896 → 2.944, female 2.722 → 2.749 |

The instantaneous-frequency loss compares the frame-to-frame phase advance of every STFT bin with the recording's: it
lowered the F0 error and the gross pitch errors that set v1 apart from BigVGAN. The snapshot is chosen on Freya-100
and copy-synthesis (`train.keep_snapshots` keeps one every 5k steps).

## Results of the release

Freya-TR-Eval (495 sentences, seed = sentence index, T = 0.3, α = 2, Whisper large-v3 on 8 kHz audio, UTMOSv2 on the
full band), reproduced with step 13:

| voice | vocoder | WER | CER | UTMOSv2 |
|---|---|---|---|---|
| studio | `vocos-v2` | 1.33% | 0.27% | 3.021 |
| male | `vocos-v2-balanced` | 2.43% | – | 2.944 |
| female | `vocos-v2-balanced` | 3.91% | – | 2.749 |
| studio, v3.1 (regressors, BigVGAN-v2-ft) | – | 1.23% | 0.24% | 2.935 |

## Design notes

- **Drifting** (Deng et al., 2026, arXiv 2602.04770). A generator maps noise to a sample in one pass. Training moves
  the generated distribution toward the data with an anti-symmetric drifting field — samples are attracted to real
  data and repelled by each other — computed in a frozen feature space, and regresses the generator's output onto
  its drifted, stop-gradient copy. The field vanishes when the two distributions match. There is no denoising and
  no distillation, so inference is one network evaluation.
- **One target per condition.** Each text has exactly one recording, so the drift problems are built per condition:
  the 16 samples of a text compete, and the target window with its perturbed views attracts them, one problem per
  (condition, feature location). MAS aligns text and mel, so generated and target frames line up and per-location
  features compare like with like.
- **Kyutai's learned temperature** (Pocket TTS with drifting, 2026). The kernel temperature is learned as a
  data-vs-siblings classifier (it starts at 1.0 and anneals), distances are normalised per row, and the field keeps
  the official two-sided product form, which balances attraction and repulsion for any temperature. The blog's
  simplified pseudo-code (`drift.mode: learned`, kept for reproducibility) diverges with one positive per condition:
  while the data holds less than half of the kernel mass, its field pushes samples apart.
- **Pitch-only sampling.** The prosody predictor is trained with the same drifting objective, so each seed reads a
  sentence with another natural tune. Only its pitch is used by default because the sampled rhythm costs words for
  the voices with little data (above).
- **Streaming.** The DiT was trained on 256-frame windows, so at inference it can also run on frame windows of a
  sentence (the first one 128 frames, then 384 with 64 frames of context on each side and a 32-frame crossfade):
  the first audio no longer waits for the whole sentence, with Freya-495 quality unchanged. See
  [HOW_IT_WORKS.md](HOW_IT_WORKS.md).

## Not in this package

The training repository, [kadirnar/drifting-tts](https://github.com/kadirnar/drifting-tts), also has: BigVGAN
fine-tuning (v3.1's vocoder), the GAN-free drift vocoder, the 1-D Mel-MAE, audio-VAE latent spaces (DAC-VAE, VoxCPM2),
the SSL adversarial fine-tuning, word and sentence features for the prosody predictor (BERTurk, ELECTRA), the
experiment log, MLX / Swift, ONNX / WebGPU and the vocoder comparison scripts.
