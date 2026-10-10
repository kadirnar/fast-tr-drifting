"""The acoustic-model trainer (fast_tr_drifting.train.train_tts), duration calibration and Vocos fine-tuning on tiny
models, CPU. Needs the ``train`` extra."""

import pytest
import torch

from .helpers import CONFIGS, TINY, prepare, tiny_mae, tiny_train_args, train_extras

train_extras()

from fast_tr_drifting.cli import main  # noqa: E402
from fast_tr_drifting.models.tts import DriftingTTS, load_tts  # noqa: E402
from fast_tr_drifting.train.config import Config, load_config  # noqa: E402
from fast_tr_drifting.train.train_tts import (  # noqa: E402
    CropBank,
    build_taus,
    feature_keys,
    grow_speakers,
    log_samples,
    sample_cfg,
    training_step,
)
from fast_tr_drifting.train.utils import save_checkpoint  # noqa: E402

TINY_MODEL = {"text": {"d": 16, "heads": 2, "layers": 1, "ffn": 32, "dropout": 0.0, "spk_dim": 8},
              "gen": {"hidden": 32, "depth": 1, "heads": 2, "patch": 2, "mlp_ratio": 2.0, "n_registers": 2,
                      "noise_classes": 4, "noise_coords": 2, "num_steps": 1}}


def test_sample_cfg_range():
    a = sample_cfg(1000, 1.0, 3.0, 3.0, 0.2, "cpu")
    assert a.min() >= 1.0 and a.max() <= 3.0
    assert (a == 1.0).float().mean() > 0.1  # no-CFG fraction
    assert a.median() < 2.0  # power law favours small scales


def test_train_end_to_end_cpu(tmp_path):
    data = prepare(tmp_path)
    tiny_mae(tmp_path / "mae.pt")
    work = tmp_path / "run"
    main(tiny_train_args(work, data, tmp_path / "mae.pt", "train.cpu=true", "train.steps=2", "train.save_every=2"))
    model, cfg, stats = load_tts(work / "model_ema.pt")
    assert stats["backend"] == "bigvgan" and cfg["model"]["gen"]["hidden"] == 32
    mel, y_len = model.synthesize(torch.tensor([[1, 5, 1, 6, 1]]), torch.tensor([5]), torch.tensor([0]))
    assert mel.shape[1] == 100 and mel.shape[-1] == y_len[0] and torch.isfinite(mel).all()


def test_train_kyutai_mode_cpu(tmp_path):
    """drift.mode: kyutai: one global raw tau, trained with the model's LR, checkpointed and resumed."""
    data = prepare(tmp_path)
    mae = tiny_mae(tmp_path / "mae.pt")
    work = tmp_path / "run"
    args = tiny_train_args(work, data, tmp_path / "mae.pt", "train.cpu=true", "train.save_every=1")
    main(args + ["drift.mode=kyutai", "train.steps=2"])
    ck = torch.load(work / "last.pt", weights_only=False)
    assert list(ck["taus"]) == ["tau"] and 0 < abs(ck["taus"]["tau"].item() - 1.0) < 1e-2 and ck["log_taus"] is None
    main(args + ["drift.mode=kyutai", "train.steps=3"])  # resumes, including tau
    assert torch.load(work / "last.pt", weights_only=False)["step"] == 3
    with pytest.raises(ValueError, match="drift.mode"):
        main(args + ["drift.mode=kyutia", "train.steps=4"])
    # the export keeps the learned tau and fine-tuning from it starts there, not at tau_init
    tau = torch.load(work / "model_ema.pt", weights_only=False)["taus"]["tau"].item()
    ft = tiny_train_args(tmp_path / "ft", data, tmp_path / "mae.pt", "train.cpu=true", "train.save_every=1")
    main(ft + ["drift.mode=kyutai", "drift.kyutai.tau_init=10", "train.steps=1",
               f"train.init_from={work / 'model_ema.pt'}"])
    assert abs(torch.load(tmp_path / "ft" / "last.pt", weights_only=False)["taus"]["tau"].item() - tau) < 1e-2

    cfg = load_config(CONFIGS / "acoustic.yaml", ["drift.kyutai.per_feature_tau=true", "drift.kyutai.tau_init=10"])
    taus = build_taus(mae, cfg, "cpu")
    assert set(taus) == set(feature_keys(mae, cfg, "cpu")) and len(taus) > 1
    assert all(t.item() == 10.0 for t in taus.values())


def test_calibrate_durations_roundtrip(tmp_path):
    """Calibration stores duration_scale in the checkpoint and load_tts exposes it."""
    test_train_end_to_end_cpu(tmp_path)
    model_path = tmp_path / "run" / "model_ema.pt"
    main(["calibrate-durations", "--model", str(model_path), "--num", "5", "--device", "cpu"])
    model, _, _ = load_tts(model_path)
    assert model.duration_scale > 0 and model.duration_scale != 1.0


def test_resume_restores_bank_and_rng_and_keeps_duration_scale(tmp_path):
    test_calibrate_durations_roundtrip(tmp_path)  # run/: last.pt at step 2 (batches of 3), calibrated model_ema.pt
    work, data, mae_path = tmp_path / "run", tmp_path / "prep", tmp_path / "mae.pt"
    ck = torch.load(work / "last.pt", weights_only=False)
    assert ck["bank"]["count"] == ck["bank"]["ptr"] == 6 and {"python", "numpy", "torch"} <= set(ck["rng"])
    scale = torch.load(work / "model_ema.pt", weights_only=False)["duration_scale"]
    main(tiny_train_args(work, data, mae_path, "train.cpu=true", "train.steps=3"))
    ck = torch.load(work / "last.pt", weights_only=False)
    assert ck["step"] == 3 and ck["bank"]["count"] == 9  # restored bank plus one batch
    assert torch.load(work / "model_ema.pt", weights_only=False)["duration_scale"] == scale
    del ck["bank"], ck["rng"]  # checkpoints without them still resume (with an empty bank)
    save_checkpoint(work / "last.pt", **ck)
    main(tiny_train_args(work, data, mae_path, "train.cpu=true", "train.steps=4"))
    assert torch.load(work / "last.pt", weights_only=False)["bank"]["count"] == 3


def test_crop_bank_push_matches_per_item_loop():
    def loop_push(data, ptr, count, x):  # the straightforward implementation
        for i in range(x.shape[0]):
            data[ptr] = x[i]
            ptr = (ptr + 1) % data.shape[0]
            count = min(count + 1, data.shape[0])
        return ptr, count

    torch.manual_seed(0)
    bank = CropBank(7, 3, 4, "cpu")
    ref, ptr, count = bank.data.clone(), 0, 0
    for n in (3, 2, 5, 7, 1, 16, 4, 0):
        x = torch.randn(n, 3, 4)
        bank.push(x)
        ptr, count = loop_push(ref, ptr, count, x)
        assert torch.equal(bank.data, ref) and (bank.ptr, bank.count) == (ptr, count)
    restored = CropBank(7, 3, 4, "cpu")
    restored.load_state_dict(bank.state_dict())
    assert torch.equal(restored.data, bank.data) and (restored.ptr, restored.count) == (bank.ptr, bank.count)


def test_uncond_negatives_are_drawn_before_the_batch_enters_the_bank():
    from fast_tr_drifting.train.mae import MelMAE2d

    events = []

    class Bank(CropBank):
        def sample(self, n):
            events.append(("sample", self.count))
            return super().sample(n)

        def push(self, x):
            events.append(("push", self.count))
            super().push(x)

    torch.manual_seed(0)
    cfg = load_config(CONFIGS / "acoustic.yaml", TINY)
    model = DriftingTTS(cfg.model, num_speakers=2)
    mae = MelMAE2d(n_mels=100, base_channels=8, layers=(1, 1, 1, 1)).eval().requires_grad_(False)
    batch = {"text": torch.randint(2, 39, (3, 20)), "text_len": torch.tensor([20, 18, 15]),
             "mel": torch.randn(3, 100, 80), "mel_len": torch.tensor([80, 75, 70]), "spk": torch.tensor([0, 1, 0])}
    bank = Bank(16, 100, 64, "cpu")
    for _ in range(2):  # the first step (empty bank) takes its negatives from the other conditions of the batch
        loss, metrics, _ = training_step(model, mae, batch, bank, cfg, "cpu")
        assert torch.isfinite(loss) and isinstance(metrics["rollout_step"], float)
    assert events == [("push", 0), ("sample", 3), ("push", 3)]


def test_log_samples_restores_mode_and_uses_temperature(tmp_path, monkeypatch):
    from fast_tr_drifting.audio import MelStats

    model = DriftingTTS(load_config(CONFIGS / "acoustic.yaml", TINY).model, num_speakers=2)
    temperatures, synthesize = [], model.synthesize

    def spy(*args, **kw):
        temperatures.append(kw["temperature"])
        return synthesize(*args, **kw)

    class DS(list):
        stats = MelStats(0.0, 1.0)

    class Writer:
        def add_audio(self, *args, **kw):
            pass

    monkeypatch.setattr(model, "synthesize", spy)
    ds = DS([{"text": torch.tensor([1, 5, 1, 6, 1]), "spk": 0}])
    vocoder = lambda mel: torch.zeros(1, 256 * mel.shape[-1])
    for training in (True, False):
        model.train(training)
        log_samples(model, ds, vocoder, Writer(), 7, tmp_path, "cpu")
        assert model.training == training
    assert temperatures == [0.5] * 4 and len(list(tmp_path.glob("step7_*.wav"))) == 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_train_compiled_generator_cuda(tmp_path):
    """train.compile: the compiled generator trains, checkpoints keep plain keys, the EMA export loads."""
    data = prepare(tmp_path)
    tiny_mae(tmp_path / "mae.pt")
    work = tmp_path / "run"
    main(tiny_train_args(work, data, tmp_path / "mae.pt", "train.compile=true", "train.steps=3"))
    ck = torch.load(work / "last.pt", weights_only=False)
    assert not any("_orig_mod" in k for k in ck["model"]) and ck["model"].keys() == ck["ema"].keys()
    model, _, _ = load_tts(work / "model_ema.pt", "cuda")
    mel, _ = model.synthesize(torch.tensor([[1, 5, 1, 6, 1]], device="cuda"), torch.tensor([5], device="cuda"),
                              torch.tensor([0], device="cuda"))
    assert torch.isfinite(mel).all()


def test_train_with_spectral_detail_and_init_from(tmp_path):
    test_train_end_to_end_cpu(tmp_path)  # produces run/model_ema.pt and prep/
    work = tmp_path / "run2"
    main(tiny_train_args(work, tmp_path / "prep", tmp_path / "mae.pt", "train.cpu=true", "train.steps=1",
                         "train.save_every=1", "drift.spectral_detail=true", "drift.every_k_block=2",
                         f"train.init_from={tmp_path / 'run' / 'model_ema.pt'}"))
    assert (work / "model_ema.pt").exists()


def test_train_with_pitch_conditioning_cpu(tmp_path):
    import json

    pytest.importorskip("pyworld")
    data = prepare(tmp_path, 12, "--f0")
    stats = json.loads((data / "stats.json").read_text())
    assert stats["voiced_frames"] > 0 and 4.5 < stats["lf0_mean"] < 6.5  # sine tones of 200-310 Hz
    tiny_mae(tmp_path / "mae.pt")
    work = tmp_path / "run"
    main(tiny_train_args(work, data, tmp_path / "mae.pt", "model.pitch.enabled=true", "train.cpu=true",
                         "train.steps=2", "train.save_every=2"))
    model, _, _ = load_tts(work / "model_ema.pt")
    assert model.pitch_enabled and abs(model.lf0_stats[0].item() - stats["lf0_mean"]) < 1e-4
    args = (torch.tensor([[1, 5, 1, 6, 1]]), torch.tensor([5]), torch.tensor([0]))
    g = torch.Generator().manual_seed(0)
    mel, _ = model.synthesize(*args, generator=g)
    g = torch.Generator().manual_seed(0)
    mel_up, _ = model.synthesize(*args, generator=g, pitch=torch.full((1, 1, 5), 1.0))  # a given token pitch
    assert mel.shape == mel_up.shape and not torch.allclose(mel, mel_up)


def test_multistep_drifting_cpu(tmp_path):
    """num_steps > 1: on-policy rollout during training, K-step (or fewer) generation at inference."""
    data = prepare(tmp_path)
    tiny_mae(tmp_path / "mae.pt")
    work = tmp_path / "run"
    torch.manual_seed(1)
    main(tiny_train_args(work, data, tmp_path / "mae.pt", "model.gen.num_steps=3", "train.cpu=true",
                         "train.steps=4", "train.save_every=4"))
    model, _, _ = load_tts(work / "model_ema.pt")
    assert model.generator.num_steps == 3
    args = (torch.tensor([[1, 5, 1, 6, 1]]), torch.tensor([5]), torch.tensor([0]))
    outs = {}
    for steps in (1, 3):
        g = torch.Generator().manual_seed(0)
        outs[steps], _ = model.synthesize(*args, generator=g, steps=steps)
    assert outs[1].shape == outs[3].shape and not torch.allclose(outs[1], outs[3])


def test_rollout_with_one_step_equals_generate():
    model = DriftingTTS(Config(TINY_MODEL), num_speakers=2).eval()
    z, cond = torch.randn(2, 100, 12), torch.randn(2, 116, 12)
    spk, alpha, nl = torch.tensor([0, 1]), torch.ones(2), torch.zeros(2, 2, dtype=torch.long)
    a = model.rollout(z, cond, spk, alpha, 1, noise_labels=nl)
    b = model.generate(z, cond, spk, alpha, noise_labels=nl)
    assert torch.allclose(a, b)


TINY_VOCOS = """
feature_extractor:
  class_path: vocos.feature_extractors.MelSpectrogramFeatures
  init_args: {sample_rate: 24000, n_fft: 1024, hop_length: 256, n_mels: 100, padding: center}
backbone:
  class_path: vocos.models.VocosBackbone
  init_args: {input_channels: 100, dim: 16, intermediate_dim: 32, num_layers: 1}
head:
  class_path: vocos.heads.ISTFTHead
  init_args: {dim: 16, n_fft: 1024, hop_length: 256, padding: center}
"""


def test_finetune_vocoder_cpu_and_the_export_loads_in_the_core(tmp_path):
    """A tiny Vocos fine-tune on GTA mels; the exported vocos_ft.pt runs in the vendored inference Vocos with the
    output of the ``vocos`` package's model."""
    pytest.importorskip("vocos")
    from fast_tr_drifting.vocoder import load_vocoder

    data = prepare(tmp_path, 12, "--save-audio")
    tiny_mae(tmp_path / "mae.pt")
    main(tiny_train_args(tmp_path / "tts", data, tmp_path / "mae.pt", "train.cpu=true", "train.steps=1",
                         "train.save_every=1"))
    (tmp_path / "tiny_vocos.yaml").write_text(TINY_VOCOS)
    work = tmp_path / "voc"
    main(["finetune-vocoder", "--config", str(CONFIGS / "vocos_v1.yaml"), "--workdir", str(work),
          f"data.root={data}", "data.min_quality=0", f"tts.path={tmp_path / 'tts' / 'model_ema.pt'}",
          f"vocoder.init={tmp_path / 'tiny_vocos.yaml'}", "train.cpu=true", "train.steps=2", "train.batch_size=2",
          "train.num_workers=0", "train.log_every=1", "train.save_every=2", "train.disc_warmup_steps=1",
          "train.segment_frames=33", "train.gta_prob=1.0", "train.sample_every=0"])
    ck = torch.load(work / "vocos_ft.pt", weights_only=False)
    assert (ck["mel"], ck["head_padding"]) == ("bigvgan", "same")
    voc = load_vocoder(str(work / "vocos_ft.pt"), "cpu")
    mel = torch.randn(1, 100, 10) - 5
    wav = voc(mel)
    assert wav.shape == (1, 10 * 256) and torch.isfinite(wav).all()
    # the same weights in the vocos package (the trainer's model) give the same waveform
    from fast_tr_drifting.train.finetune_vocoder import build_vocos

    ref = build_vocos(str(tmp_path / "tiny_vocos.yaml"))
    ref.head.istft.padding = "same"
    ref.load_state_dict(ck["vocos"])
    with torch.no_grad():
        h = ref.head.out(ref.backbone(mel)).transpose(1, 2)
    from fast_tr_drifting.vocoder import ola_istft

    mag, p = h.chunk(2, dim=1)
    expected = ola_istft(torch.exp(mag).clip(max=1e2) * (torch.cos(p) + 1j * torch.sin(p)), ref.head.istft.window)
    torch.testing.assert_close(wav, expected.clamp(-1, 1), rtol=0, atol=1e-6)


def test_grow_speakers_keeps_old_voices_and_adds_new_rows():
    old, new = DriftingTTS(Config(TINY_MODEL), num_speakers=3), DriftingTTS(Config(TINY_MODEL), num_speakers=4)
    state = grow_speakers(old.state_dict(), new.state_dict())
    new.load_state_dict(state)
    for name in ("encoder.spk.weight", "generator.spk.weight"):
        w_old, w_new = old.state_dict()[name], new.state_dict()[name]
        assert torch.equal(w_new[:3], w_old) and torch.allclose(w_new[3], w_old.mean(0))
