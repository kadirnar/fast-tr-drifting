"""The drift prosody predictor: its training features and loss, sampling (incl. the duration row and borrowed
rhythm), and prosody-cache -> train-prosody end to end on tiny models (CPU)."""

import numpy as np
import pytest
import torch

pytest.importorskip("yaml")

from fast_tr_drifting.models.prosody_net import (  # noqa: E402
    ProsodyNet,
    ProsodyPredictor,
    ProsodyStats,
    boundary_tokens,
    edge_tokens,
)
from fast_tr_drifting.text import text_to_ids  # noqa: E402
from fast_tr_drifting.train.prosody_features import prosody_features, word_index  # noqa: E402
from fast_tr_drifting.train.train_prosody import drift_maps_loss, interpolate_unvoiced  # noqa: E402


def _batch(texts):
    ids = [torch.tensor(text_to_ids(t, normalized=True)) for t in texts]
    N = max(len(i) for i in ids)
    out = torch.zeros(len(ids), N, dtype=torch.long)
    for k, i in enumerate(ids):
        out[k, : len(i)] = i
    lens = torch.tensor([len(i) for i in ids])
    return out, lens, torch.arange(N)[None] < lens[:, None]


def test_features_ignore_padding_and_mark_valid_locations():
    ids, lens, mask = _batch(["bugün hava çok güzel.", "merhaba."])
    B, N, S = ids.shape[0], ids.shape[1], 3
    stats = ProsodyStats()
    y = torch.randn(B, S, 2, N)
    feats, valid = prosody_features(y, mask, word_index(ids), stats)
    assert set(feats) == {"tok", "d1", "avg3", "avg9", "avg27", "win8", "win32", "word", "utt"}
    for k, f in feats.items():
        assert f.shape[:3] == (B, S, valid[k].shape[1]) and torch.isfinite(f).all(), k
    assert valid["tok"][1].sum() == lens[1] and valid["word"][1].sum() == 1  # "merhaba." is one word
    y2 = y.clone()
    y2[1, :, :, lens[1]:] = 100.0  # padding must not leak into any valid location
    feats2, _ = prosody_features(y2, mask, word_index(ids), stats)
    for k in feats:
        assert torch.allclose(feats[k][valid[k][:, None].expand(-1, S, -1)],
                              feats2[k][valid[k][:, None].expand(-1, S, -1)], atol=1e-5), k


def test_drift_maps_loss_pulls_samples_towards_the_positive():
    torch.manual_seed(0)
    ids, lens, mask = _batch(["bir iki üç dört beş altı yedi."])
    stats = ProsodyStats()
    N = ids.shape[1]
    pos = torch.zeros(1, 1, 2, N)
    x = (torch.randn(1, 8, 2, N) * 0.5 + 2.0).requires_grad_(True)
    opt = torch.optim.Adam([x], lr=0.05)
    tau = torch.tensor(1.0, requires_grad=True)
    first = None
    for _ in range(60):
        fg, valid = prosody_features(x, mask, word_index(ids), stats)
        fp, _ = prosody_features(pos, mask, word_index(ids), stats)
        loss, tau_loss, _ = drift_maps_loss(fg, fp, valid, tau)
        opt.zero_grad()
        (loss + tau_loss).backward()
        opt.step()
        dist = x.detach().mean(1).abs().mean().item()
        first = dist if first is None else first
    assert dist < 0.5 * first


def test_interpolate_unvoiced_holds_edges():
    p = np.array([0, 1.0, 0, 0, 3.0, 0], dtype=np.float32)
    v = np.array([0, 1, 0, 0, 1, 0], dtype=bool)
    assert np.allclose(interpolate_unvoiced(p, v), [1, 1, 5 / 3, 7 / 3, 3, 3])


def _tiny_pitch_tts():
    from fast_tr_drifting.models.tts import DriftingTTS

    cfg = {"text": {"d": 16, "heads": 2, "layers": 1, "ffn": 32, "dropout": 0.0, "spk_dim": 8},
           "gen": {"hidden": 32, "depth": 1, "heads": 2, "patch": 2, "mlp_ratio": 2.0, "n_registers": 2,
                   "noise_classes": 4, "noise_coords": 2, "num_steps": 1},
           "pitch": {"enabled": True}}
    torch.manual_seed(0)
    return DriftingTTS(cfg, num_speakers=2).eval()


def test_predict_feeds_the_generator_and_the_seed_determines_it():
    tts = _tiny_pitch_tts()
    pred = ProsodyPredictor({"kind": "drift", "d": 32, "layers": 1, "heads": 2, "ffn": 64, "noise_tok": 4,
                             "noise_glob": 4, "out_init": 1.0}, cond_dim=16 + 8 + 2).eval()
    ids, lens, _ = _batch(["merhaba dünya."])
    spk = torch.tensor([1])

    def run(seed, temperature=1.0):
        g = torch.Generator().manual_seed(seed)
        frames, pitch = pred.predict(tts, ids, lens, spk, temperature, generator=g)
        mel, y_len = tts.synthesize(ids, lens, spk, generator=g, durations=frames, pitch=pitch)
        return frames, pitch, mel, y_len

    frames, pitch, mel, y_len = run(0)
    assert frames.min() >= 1 and frames.dtype.is_floating_point and torch.equal(frames, frames.round())
    assert int(y_len) == int(frames.sum()) == mel.shape[-1]
    assert torch.equal(run(0)[2], mel)  # one seed fixes prosody and generator noise
    assert not torch.equal(run(1)[0], frames) or not torch.equal(run(1)[1], pitch)
    f0, p0, _, _ = run(5, temperature=0.0)  # zero temperature: deterministic prosody
    f1, p1, _, _ = run(6, temperature=0.0)
    assert torch.equal(f0, f1) and torch.equal(p0, p1)


def test_mse_and_flow_nets_run():
    for kind in ("mse", "flow"):
        net = ProsodyNet(10, kind=kind, d=16, layers=1, heads=2, ffn=32)
        mask = torch.ones(2, 1, 7)
        kw = {"x_t": torch.randn(2, 2, 7), "t": torch.rand(2)} if kind == "flow" else {}
        assert net(torch.randn(2, 10, 7), mask, **kw).shape == (2, 3, 7)


def test_cache_and_train_end_to_end_cpu(tmp_path):
    """prepare --f0 -> a 1-step pitch TTS -> prosody-cache -> train-prosody (drift, flow) -> calibrate -> predict."""
    import pytest

    from .helpers import CONFIGS, fake_parquet, tiny_mae, tiny_train_args, train_extras

    train_extras()
    pytest.importorskip("pyworld")
    from fast_tr_drifting.cli import main

    fake_parquet(tmp_path / "x.parquet", n=12)
    data = tmp_path / "prep"
    main(["prepare", "--parquet-glob", str(tmp_path / "*.parquet"), "--out", str(data), "--workers", "1",
          "--val-size", "2", "--no-trim", "--f0", "--val-max-seconds", "30"])
    tiny_mae(tmp_path / "mae.pt")
    main(tiny_train_args(tmp_path / "tts", data, tmp_path / "mae.pt", "train.cpu=true", "train.steps=1",
                         "model.pitch.enabled=true"))
    tts_path, cache = tmp_path / "tts" / "model_ema.pt", tmp_path / "cache.pt"
    main(["prosody-cache", "--model", str(tts_path), "--data", str(data), "--out", str(cache), "--num-workers", "0",
          "--device", "cpu", "--splits", "train", "val"])
    c = torch.load(cache, weights_only=False)
    assert all(int(c["tokens"]["dur"][u["start"]: u["start"] + u["n"]].sum()) == u["frames"] for u in c["utts"])
    tiny = ["net.d=16", "net.layers=1", "net.heads=2", "net.ffn=32", "net.noise_tok=2", "net.noise_glob=2",
            "drift.gen_per_cond=3", "train.batch_size=2", "train.steps=2", "train.log_every=1", "train.warmup=1",
            "train.eval_every=2", "train.save_every=2", "train.cpu=true", "eval_temperatures=[1.0]",
            "calibrate.enabled=false"]
    from fast_tr_drifting.models.tts import load_tts

    tts, _, _ = load_tts(tts_path)
    ids, lens, _ = _batch(["merhaba dünya."])
    for kind in ("drift", "flow"):
        work = tmp_path / kind
        # the cache has no dev split: evaluate on val
        main(["train-prosody", "--config", str(CONFIGS / "prosody_drift.yaml"), "--workdir", str(work),
              f"tts={tts_path}", f"cache={cache}", f"net.kind={kind}", *tiny])
        pred = ProsodyPredictor.load(work / "prosody_ema.pt", tts=tts)
        frames, pitch = pred.predict(tts, ids, lens, torch.tensor([0]), generator=torch.Generator().manual_seed(0))
        assert frames.shape == ids.shape and pitch.shape == (1, 1, ids.shape[1]) and frames.min() >= 1
    # the release's operating point: preferred temperatures, the duration temperature and a rhythm table
    main(["train-prosody", "--config", str(CONFIGS / "prosody_drift.yaml"), "--workdir", str(tmp_path / "drift"),
          "--calibrate-only", f"tts={tts_path}", f"cache={cache}", *tiny, "calibrate.temperature=0.5",
          "calibrate.duration_temperature=0.3", "calibrate.rhythm={1: 0}"])
    ck = torch.load(tmp_path / "drift" / "prosody_ema.pt", weights_only=False)
    assert ck["temperature"] == 0.5 and ck["duration_temperature"] == 0.3 and ck["rhythm"] == {1: 0}
    pred = ProsodyPredictor.load(tmp_path / "drift" / "prosody_ema.pt", tts=tts)
    assert pred.temperature == 0.5 and pred.duration_temperature == 0.3 and pred.rhythm == {1: 0}


def test_durations_at_their_own_temperature_and_with_a_borrowed_rhythm():
    """A second row of the sampler's batch gives the log-durations: the same noise draw at the duration temperature
    for the letters (the pauses keep the call's), or for another speaker (all of them); the pitch stays the one-row
    sample's."""
    tts = _tiny_pitch_tts()
    pred = ProsodyPredictor({"kind": "drift", "d": 32, "layers": 1, "heads": 2, "ffn": 64, "noise_tok": 4,
                             "noise_glob": 4, "out_init": 1.0}, cond_dim=16 + 8 + 2).eval()
    ids, lens, _ = _batch(["merhaba dünya, bugün nasılsın."])

    def run(seed, spk=1, **kw):
        g = torch.Generator().manual_seed(seed)
        return pred.predict(tts, ids, lens, torch.tensor([spk]), 0.7, generator=g, **kw)

    frames, pitch = run(0)
    same_f, same_p = run(0, duration_temperature=0.7)  # two rows, equal temperatures: the one-row sample
    torch.testing.assert_close(same_f, frames)
    torch.testing.assert_close(same_p, pitch)
    cold = [run(s, duration_temperature=0.0) for s in (0, 1)]
    torch.testing.assert_close(cold[0][1], pitch)  # the pitch keeps the call's temperature and noise
    gap = boundary_tokens(ids)
    assert gap.any() and (~gap).any()
    assert torch.equal(cold[0][0][~gap], cold[1][0][~gap])  # zero temperature: the same letters for every seed
    torch.testing.assert_close(cold[0][0][gap], frames[gap])  # the pauses keep the call's temperature
    assert not torch.equal(cold[0][1], cold[1][1])
    borrowed_f, borrowed_p = run(0, duration_speaker=torch.tensor([0]))
    torch.testing.assert_close(borrowed_p, pitch)  # pitch of speaker 1
    torch.testing.assert_close(borrowed_f, run(0, spk=0)[0])  # all durations as speaker 0's
    edged_f, _ = run(0, duration_speaker=torch.tensor([0]), edge_scale=1.3)  # the sentence's edges: speaker 1's own
    edges = edge_tokens(lens, ids.shape[1])
    _, _, logw, _ = tts.encoder(ids, lens, torch.tensor([1]))
    torch.testing.assert_close(edged_f[edges], torch.ceil(torch.exp(logw[:, 0]) * 1.3)[edges])
    torch.testing.assert_close(edged_f[~edges], borrowed_f[~edges])
    assert edges.sum() == 5


def _tiny_pred(**net):
    cfg = {"kind": "drift", "d": 32, "layers": 1, "heads": 2, "ffn": 64, "noise_tok": 4, "noise_glob": 4,
           "out_init": 1.0, **net}
    return ProsodyPredictor(cfg, cond_dim=16 + 8 + 2).eval()


def test_predict_is_the_condition_one_draw_frames_and_pitch():
    """``predict`` without options: the condition, one draw from the generator, frames and pitch; a duration row at
    the call's temperature or an unused letter floor change nothing."""
    base = _tiny_pred()
    tts = _tiny_pitch_tts()
    ids, lens, _ = _batch(["ali dün mü geldi? evet, geldi."])
    spk = torch.tensor([1])

    def run(pred, **kw):
        return pred.predict(tts, ids, lens, spk, 0.5, 1.2, generator=torch.Generator().manual_seed(5), **kw)

    h, _, logw, x_mask = tts.encoder(ids, lens, spk)
    cond, b0 = ProsodyPredictor.condition(tts, h, x_mask, spk, logw, base.stats)
    ref = base.frames_and_pitch(*base._draw(cond, b0, x_mask, 0.5, torch.Generator().manual_seed(5), None), x_mask, 1.2)
    out = run(base)
    assert all(torch.equal(a, b) for a, b in zip(out, ref))
    assert all(torch.equal(a, b) for a, b in zip(run(base, min_letter_frames=0, rel_letter_floor=0), out))
    torch.testing.assert_close(run(base, duration_temperature=0.5)[0], out[0])


def test_word_or_sentence_feature_checkpoints_are_refused(tmp_path):
    import pytest

    pred = _tiny_pred()
    ck = {"ema": pred.net.state_dict(), "stats": pred.stats.state_dict(), "cond_dim": 26,
          "net_cfg": {**pred.net_cfg, "sent_dim": 19, "ctx_pitch_only": True}}
    torch.save(ck, tmp_path / "p.pt")
    with pytest.raises(ValueError, match="word / sentence features"):
        ProsodyPredictor.load(tmp_path / "p.pt")
    torch.save({**ck, "net_cfg": {**pred.net_cfg, "word_dim": 0, "sent_dim": 0, "ctx_pitch_only": False}},
               tmp_path / "q.pt")  # a phase-2 config with the options off loads
    assert ProsodyPredictor.load(tmp_path / "q.pt").kind == "drift"


def test_floor_letters_raises_short_letters_only():
    from fast_tr_drifting.models.prosody_net import floor_letters

    ids, _, mask = _batch(["ab c."])  # tokens: _ a _ b _ ' ' _ c _ . _
    frames = torch.tensor([[1.0, 1, 1, 4, 1, 2, 1, 1, 0, 3, 1]])
    out = floor_letters(frames, ids, min_frames=3)
    assert out[0].tolist() == [1, 2, 1, 4, 1, 2, 1, 3, 0, 3, 1]  # a: 1+1 -> 3, c: 1+0 -> 3; blanks, space, '.' kept
    ref = torch.full_like(frames, 4.0)
    assert floor_letters(frames, ids, rel=0.5, ref_frames=ref)[0, 3].item() == 4  # b: 4+1 >= round(0.5 * 8)


def test_boundary_tokens():
    ids, _, _ = _batch(["ab cd, ef."])  # _ a _ b _ ' ' _ c _ d _ , _ ' ' _ e _ f _ . _
    assert boundary_tokens(ids)[0].int().tolist() == [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1]
