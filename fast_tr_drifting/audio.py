"""Audio constants, mel normalisation and the BigVGAN-style log-mel front end (24 kHz, 100 bands, hop 256).

Inference needs only the constants and :class:`MelStats`. The front end, F0 extraction and waveform preparation are
used by training and evaluation; they import ``torchaudio`` / ``pyworld`` lazily (the ``train`` extra).
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import torch

SAMPLE_RATE = 24_000
N_FFT = 1024
HOP_LENGTH = 256
N_MELS = 100
FRAME_RATE = SAMPLE_RATE / HOP_LENGTH  # 93.75 frames / s

BACKENDS = ("bigvgan",)  # the mel front end of every released model


class MelStats:
    """Normalisation ``(x - mean) / std`` of the log-mels (one scalar each, stored in the checkpoint)."""

    def __init__(self, mean: float, std: float):
        self.mean, self.std = float(mean), float(std)

    @classmethod
    def load(cls, root: str | Path) -> MelStats:
        s = json.loads((Path(root) / "stats.json").read_text())
        return cls(s["mean"], s["std"])

    def normalize(self, mel: torch.Tensor) -> torch.Tensor:
        return (mel - self.mean) / self.std

    def denormalize(self, mel: torch.Tensor) -> torch.Tensor:
        return mel * self.std + self.mean

    def to_dict(self) -> dict:
        return {"mean": self.mean, "std": self.std}


class BigVGANLogMel(torch.nn.Module):
    """Mel front end of ``nvidia/bigvgan_v2_24khz_100band_256x``: Slaney mel filters (librosa defaults),
    reflect-padded uncentred STFT, magnitude ``sqrt(|X|^2 + 1e-9)``, ``log(clamp(mel, 1e-5))``."""

    def __init__(self):
        import torchaudio

        super().__init__()
        fb = torchaudio.functional.melscale_fbanks(N_FFT // 2 + 1, 0.0, SAMPLE_RATE / 2, N_MELS, SAMPLE_RATE,
                                                   norm="slaney", mel_scale="slaney")
        self.register_buffer("fb", fb.T.contiguous(), persistent=False)
        self.register_buffer("window", torch.hann_window(N_FFT), persistent=False)

    @torch.no_grad()
    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """``[..., samples] -> [..., n_mels, samples // hop]``."""
        lead = wav.shape[:-1]
        y = wav.reshape(-1, wav.shape[-1])
        pad = (N_FFT - HOP_LENGTH) // 2
        y = torch.nn.functional.pad(y[:, None], (pad, pad), mode="reflect")[:, 0]
        spec = torch.stft(y, N_FFT, HOP_LENGTH, N_FFT, self.window, center=False, return_complex=True)
        mag = torch.sqrt(spec.real**2 + spec.imag**2 + 1e-9)
        mel = torch.log(torch.matmul(self.fb, mag).clamp_min(1e-5))
        return mel.reshape(*lead, N_MELS, mel.shape[-1])


def make_logmel(backend: str = "bigvgan") -> torch.nn.Module:
    """The mel front end of the released models (``bigvgan``)."""
    if backend not in BACKENDS:
        raise ValueError(f"unknown backend {backend!r}, expected one of {BACKENDS}")
    return BigVGANLogMel()


def decode_audio(data: bytes) -> tuple[np.ndarray, int]:
    """Decode encoded audio bytes (wav/flac/mp3/...) into a mono float32 array."""
    import soundfile as sf

    wav, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    return wav.mean(axis=1), sr


def trim_silence(wav: np.ndarray, sr: int, top_db: float = 40.0, margin_s: float = 0.1) -> np.ndarray:
    """Trim leading / trailing frames quieter than ``top_db`` below the peak frame energy."""
    frame = int(0.02 * sr)
    if len(wav) < 2 * frame:
        return wav
    n = len(wav) // frame
    energy = 10 * np.log10(np.mean(wav[: n * frame].reshape(n, frame) ** 2, axis=1) + 1e-10)
    voiced = np.nonzero(energy > energy.max() - top_db)[0]
    if len(voiced) == 0:
        return wav
    margin = int(margin_s * sr)
    start = max(0, voiced[0] * frame - margin)
    end = min(len(wav), (voiced[-1] + 1) * frame + margin)
    return wav[start:end]


def normalize_loudness(wav: np.ndarray, target_dbfs: float = -20.0, peak: float = 0.95) -> np.ndarray:
    """RMS-normalise to ``target_dbfs`` while keeping the peak below ``peak``."""
    rms = np.sqrt(np.mean(wav**2) + 1e-12)
    gain = 10 ** (target_dbfs / 20) / rms
    gain = min(gain, peak / (np.abs(wav).max() + 1e-12))
    return (wav * gain).astype(np.float32)


def prepare_waveform(data: bytes, trim: bool = True) -> torch.Tensor:
    """Bytes -> mono, trimmed, loudness-normalised 24 kHz float tensor ``[samples]``."""
    wav, sr = decode_audio(data)
    if trim:
        wav = trim_silence(wav, sr)
    wav = normalize_loudness(wav)
    out = torch.from_numpy(wav)
    if sr != SAMPLE_RATE:
        import torchaudio

        out = torchaudio.functional.resample(out, sr, SAMPLE_RATE)
    return out


# centre of mel frame i, in hops past sample i * hop: BigVGAN pads (n_fft - hop) / 2 and frames without centring, so
# its frame i covers samples [i * hop - 384, i * hop + 640)
FRAME_CENTRE = {"bigvgan": 0.5}


def extract_f0(wav: torch.Tensor | np.ndarray, frames: int, sample_rate: int = SAMPLE_RATE, method: str = "dio",
               backend: str = "bigvgan") -> np.ndarray:
    """WORLD F0 in Hz at the centres of the ``backend``'s mel frames, 0 for unvoiced frames, length ``frames``.

    ``method``: ``dio`` (+ stonemask, fast) or ``harvest`` (about 15x slower, fewer voicing / octave errors on
    noisy speech).
    """
    period = 1000.0 * HOP_LENGTH / sample_rate / 2  # half-hop grid: frame centres sit on it
    f0 = world_f0(wav, sample_rate, period, method)[round(2 * FRAME_CENTRE[backend])::2]
    out = np.zeros(frames, dtype=np.float32)
    n = min(frames, len(f0))
    out[:n] = f0[:n]
    return out


def world_f0(wav: torch.Tensor | np.ndarray, sample_rate: int, frame_period: float, method: str = "dio",
             allowed_range: float = 0.1) -> np.ndarray:
    """WORLD F0 in Hz (60-800 Hz, 0: unvoiced) at ``k * frame_period`` ms, ``k = 0, 1, ...`` (:func:`extract_f0`).
    ``allowed_range``: dio's voicing threshold (larger: more frames voiced)."""
    import pyworld

    x = np.asarray(wav, dtype=np.float64)
    if method == "harvest":
        f0, _ = pyworld.harvest(x, sample_rate, f0_floor=60.0, f0_ceil=800.0, frame_period=frame_period)
    elif method == "dio":
        f0, t = pyworld.dio(x, sample_rate, f0_floor=60.0, f0_ceil=800.0, frame_period=frame_period,
                            allowed_range=allowed_range)
        f0 = pyworld.stonemask(x, f0, t, sample_rate)
    else:
        raise ValueError(f"unknown F0 method {method!r}")
    return f0


def to_pcm16(wav: torch.Tensor | np.ndarray) -> bytes:
    """Float waveform in [-1, 1] -> little-endian 16-bit PCM bytes (for streaming to clients)."""
    x = wav.detach().cpu().numpy() if isinstance(wav, torch.Tensor) else np.asarray(wav)
    return (np.clip(x, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()


def save_wav(path: str | Path, wav: torch.Tensor | np.ndarray, sample_rate: int = SAMPLE_RATE) -> None:
    import soundfile as sf

    x = wav.detach().cpu().numpy() if isinstance(wav, torch.Tensor) else np.asarray(wav)
    sf.write(str(path), x, sample_rate)
