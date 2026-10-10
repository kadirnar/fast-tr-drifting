"""Shared fixtures of the training tests: a fake parquet dataset, a tiny Mel-MAE and tiny model overrides."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
CONFIGS = REPO / "configs"

# a tiny acoustic model and drift problem (CPU, seconds per step)
TINY = ["drift.crop_frames=64", "drift.gen_per_cond=3", "drift.pos_views=2", "drift.uncond_per_cond=2",
        "drift.uncond_bank=16", "model.gen.hidden=32", "model.gen.depth=1", "model.gen.heads=2",
        "model.gen.noise_coords=2", "model.text.d=16", "model.text.layers=1", "model.text.ffn=32",
        "model.text.spk_dim=8"]
TINY_MAE = ["model.base_channels=8", "model.layers=[1,1,1,1]"]


def train_extras() -> None:
    """Skip a test when the ``train`` extra is missing."""
    for name in ("numba", "pyarrow", "torchaudio", "yaml", "soundfile", "tqdm"):
        pytest.importorskip(name)


def wav_bytes(seconds: float, sr: int = 44_100, freq: float = 220.0) -> bytes:
    import soundfile as sf

    t = np.arange(int(seconds * sr)) / sr
    wav = 0.3 * np.sin(2 * np.pi * freq * t)[:, None].repeat(2, 1)  # stereo
    buf = io.BytesIO()
    sf.write(buf, wav, sr, format="FLAC")
    return buf.getvalue()


def fake_parquet(path, n: int = 30) -> None:
    """``n`` sine tones (1-4 s, 200-490 Hz) with Turkish transcripts, 3 speakers and quality scores."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = {
        "audio": [{"bytes": wav_bytes(1.0 + 0.1 * i, freq=200 + 10 * i), "path": f"{i}.flac"} for i in range(n)],
        "text": [f"Merhaba dünya {i}." for i in range(n)],
        "duration_seconds": [1.0 + 0.1 * i for i in range(n)],
        "speaker": [f"spk_{i % 3}" for i in range(n)],
        "quality_score": [50.0 + i for i in range(n)],
    }
    pq.write_table(pa.table(rows), path)


def prepare(tmp_path, n: int = 12, *flags: str) -> Path:
    """``fake_parquet`` -> ``prepare`` into ``tmp_path / "prep"``."""
    from fast_tr_drifting.cli import main

    fake_parquet(tmp_path / "x.parquet", n=n)
    data = tmp_path / "prep"
    main(["prepare", "--parquet-glob", str(tmp_path / "*.parquet"), "--out", str(data), "--workers", "1",
          "--val-size", "0", "--no-trim", *flags])
    return data


def tiny_mae(path):
    """A tiny 2-D Mel-MAE checkpoint at ``path`` (as ``train-mae`` writes it); returns the model."""
    from fast_tr_drifting.train.config import load_config
    from fast_tr_drifting.train.mae import MelMAE2d
    from fast_tr_drifting.train.utils import save_checkpoint

    cfg = load_config(CONFIGS / "mae2d.yaml", TINY_MAE)
    mae = MelMAE2d(n_mels=100, base_channels=8, layers=(1, 1, 1, 1))
    save_checkpoint(path, ema=mae.state_dict(), config=cfg.to_dict(), num_classes=0)
    return mae


def tiny_train_args(work, data, mae_path, *overrides) -> list[str]:
    return ["train", "--config", str(CONFIGS / "acoustic.yaml"), "--workdir", str(work), f"data.root={data}",
            "data.min_quality=0", f"mae.path={mae_path}", "train.batch_size=3", "train.num_workers=0",
            "train.log_every=1", "train.sample_every=0", "train.warmup=1", *TINY, *overrides]
