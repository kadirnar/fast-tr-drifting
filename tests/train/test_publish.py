"""Release files (fast_tr_drifting.train.publish, scripts/export_release.py): keep what the loaders read, drop
training state and local paths, scan for private names. CPU, tiny models."""

import json
import warnings

import pytest
import torch

pytest.importorskip("yaml")

from fast_tr_drifting.models.prosody_net import (  # noqa: E402
    CHECKPOINT_KEYS,
    ProsodyPredictor,
    export_checkpoint,
    tts_fingerprint,
)
from fast_tr_drifting.models.tts import DriftingTTS, load_tts  # noqa: E402
from fast_tr_drifting.train.publish import export_tts, export_vocos, pickle_strings, private_terms, scan  # noqa: E402

MODEL = {"text": {"d": 16, "heads": 2, "layers": 1, "ffn": 32, "dropout": 0.0, "spk_dim": 8},
         "gen": {"hidden": 32, "depth": 1, "heads": 2, "patch": 2, "mlp_ratio": 2.0, "n_registers": 2,
                 "noise_classes": 4, "noise_coords": 2, "num_steps": 1},
         "pitch": {"enabled": True}}
NET = {"kind": "drift", "d": 32, "layers": 1, "heads": 2, "ffn": 64, "noise_tok": 4, "noise_glob": 4, "out_init": 1.0}


def _tts_file(path, seed: int = 0, root: str = "/home/user/data/private_corpus") -> str:
    torch.manual_seed(seed)
    tts = DriftingTTS(MODEL, num_speakers=3)
    torch.save({"ema": tts.state_dict(), "config": {"data": {"root": root}, "model": MODEL,
                                                    "mae": {"path": "/home/user/runs/mae2d/mae_ema.pt"}},
                "num_speakers": 3, "stats": {"mean": -5.0, "std": 2.0, "backend": "bigvgan"},
                "duration_scales": {2: 1.1}, "temperature": 0.3, "opt": {"lr": 1.0}, "taus": [0.1]}, path)
    return str(path)


def _prosody_ck(temperature: float = 0.5) -> dict:
    torch.manual_seed(1)
    pred = ProsodyPredictor(NET, cond_dim=16 + 8 + 2)
    return {"ema": pred.net.state_dict(), "stats": pred.stats.state_dict(), "net_cfg": NET, "cond_dim": 26,
            "duration_scales": {"2": 1.05}, "temperature": temperature, "flow_steps": 8, "step": 10,
            "tts": "/home/user/runs/release/x.pt", "config": {"cache": "/home/user/runs/prosody_cache/targets.pt"}}


def test_prosody_export_keeps_what_load_needs_and_fingerprints_the_encoder(tmp_path):
    tts = load_tts(_tts_file(tmp_path / "tts.pt"))[0]
    out = export_checkpoint(_prosody_ck(), tts)
    assert set(out) <= set(CHECKPOINT_KEYS) and "config" not in out and "tts" not in out and "step" not in out
    assert out["duration_scales"] == {2: 1.05} and out["tts_fingerprint"] == tts_fingerprint(tts)
    torch.save(out, tmp_path / "p.pt")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ProsodyPredictor.load(tmp_path / "p.pt", tts=tts)  # same encoder: no warning
    other = load_tts(_tts_file(tmp_path / "tts2.pt", seed=5))[0]
    with pytest.warns(UserWarning, match="another acoustic model"):
        ProsodyPredictor.load(tmp_path / "p.pt", tts=other)


def test_publish_exports_and_scan(tmp_path):
    torch.save(_prosody_ck(), tmp_path / "raw.pt")
    terms = {"names": ["private_corpus", "abc"]}  # terms under 4 characters are ignored
    hits = scan(tmp_path / "raw.pt", terms)
    assert hits["paths"] >= 2  # 'tts' and 'config.cache' hold absolute paths
    raw = torch.load(_tts_file(tmp_path / "tts.pt"), weights_only=False)
    assert scan(tmp_path / "tts.pt", terms)["names"] >= 1  # data.root names the corpus
    clean = export_tts(raw)
    assert "opt" not in clean and "taus" not in clean and clean["config"]["data"]["root"] == "data/train"
    assert clean["config"]["mae"]["path"] == "runs/mae2d/mae_ema.pt" and clean["duration_scales"] == {2: 1.1}
    torch.save(clean, tmp_path / "clean.pt")
    assert scan(tmp_path / "clean.pt", terms) == {"paths": 0, "names": 0}
    assert "ema" in pickle_strings(tmp_path / "clean.pt")
    assert load_tts(tmp_path / "clean.pt")[0].temperature == 0.3  # the core loads it
    with pytest.raises(ValueError, match="Hub repo id"):
        export_vocos({"vocos": {}, "init": "/home/user/runs/vocos.yaml"})
    v = export_vocos({"vocos": {"w": torch.ones(2)}, "init": "charactr/vocos-mel-24khz", "mel": "bigvgan",
                      "head_padding": "same", "step": 5, "opt": {}, "config": {"workdir": "/home/user/x"}})
    assert set(v) == {"vocos", "init", "mel", "head_padding", "step"}


def test_private_terms_from_a_data_root(tmp_path):
    root = tmp_path / "my_corpus"
    root.mkdir()
    (root / "speakers.json").write_text(json.dumps({"spk_alpha": 0, "spk_beta": 1}))
    (root / "index.jsonl").write_text(json.dumps({"show_name": "Some Show", "spk_id": 0}) + "\n")
    t = private_terms(root)
    assert t["speaker names"] == ["spk_alpha", "spk_beta"] and t["source names"] == ["Some Show"]
    assert "my_corpus" in t["data root"]


def test_export_release_script(tmp_path):
    """scripts/export_release.py on tiny files: sanitised acoustic model and prosody predictor, verified, scanned."""
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "export_release", Path(__file__).resolve().parents[2] / "scripts" / "export_release.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    tts = _tts_file(tmp_path / "model_ema.pt")
    torch.save(_prosody_ck(), tmp_path / "prosody_ema.pt")
    mod.main(["--out", str(tmp_path / "release"), "--model", tts, "--prosody", str(tmp_path / "prosody_ema.pt"),
              "--device", "cpu"])
    manifest = json.loads((tmp_path / "release" / "MANIFEST.json").read_text())
    assert set(manifest) == {"drifting_tts_v3.2.pt", "prosody_drift_v3.2.pt"}
    for name in manifest:
        assert scan(tmp_path / "release" / name)["paths"] == 0
