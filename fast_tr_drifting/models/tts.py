"""The drifting TTS model: text encoder (+ durations and token pitch) and the one-step DriftDiT generator."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import torch
from torch import Tensor, nn

from ..alignment import sequence_mask
from ..text import SYMBOLS
from .generator import DriftDiT
from .text_encoder import DurationPredictor, TextEncoder, durations_to_alignment, expand, frames_to_alignment


class DriftingTTS(nn.Module):
    """``cfg``: the ``model`` section of a training config (a mapping with ``text``, ``gen`` and ``pitch``)."""

    def __init__(self, cfg: Mapping, num_speakers: int, n_mels: int = 100):
        super().__init__()
        t, g = cfg["text"], cfg["gen"]
        self.encoder = TextEncoder(len(SYMBOLS), n_mels=n_mels, d=t["d"], heads=t["heads"], layers=t["layers"],
                                   ffn=t["ffn"], dropout=t.get("dropout", 0.1), num_speakers=num_speakers,
                                   spk_dim=t["spk_dim"])
        self.generator = DriftDiT(n_mels=n_mels, cond_channels=n_mels + t["d"], hidden=g["hidden"], depth=g["depth"],
                                  heads=g["heads"], patch=g["patch"], mlp_ratio=g["mlp_ratio"],
                                  n_registers=g["n_registers"], num_speakers=num_speakers,
                                  noise_classes=g["noise_classes"], noise_coords=g["noise_coords"],
                                  num_steps=int(g.get("num_steps", 1)))
        # predict a residual over the aligned prior mean: a non-constant, well-conditioned output at init
        self.residual_prior = bool(g.get("residual_prior", True))
        self.n_mels = n_mels
        # FastPitch-style pitch conditioning: token-level normalised log-F0 added to the text features
        self.pitch_enabled = bool((cfg.get("pitch") or {}).get("enabled", False))
        if self.pitch_enabled:
            self.pitch_predictor = DurationPredictor(t["d"] + t["spk_dim"], dropout=t.get("dropout", 0.1))
            self.pitch_emb = nn.Conv1d(1, t["d"], 3, padding=1)
            self.register_buffer("lf0_stats", torch.tensor([5.0, 0.35]))  # log-F0 mean / std, set from the data
        # set by load_tts from the checkpoint (``calibrate-durations``)
        self.duration_scale: float = 1.0
        self.duration_scales: dict[int, float] = {}
        self.temperature: float | None = None

    def pitch_condition(self, h: Tensor, x_mask: Tensor, spk: Tensor,
                        pitch: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """Add a pitch embedding to the token features: the given token pitch (ground truth in training, a sampled
        one at inference), else the predicted one. Returns the conditioned features and the predicted pitch."""
        s = self.encoder.spk(spk)[:, :, None].expand(-1, -1, h.shape[-1])
        pred = self.pitch_predictor(torch.cat([h.detach(), s], 1), x_mask)
        p = pred if pitch is None else pitch
        return h + self.pitch_emb(p) * x_mask, pred

    def generate(self, z: Tensor, cond: Tensor, spk: Tensor, cfg_scale: Tensor, mask: Tensor | None = None,
                 noise_labels: Tensor | None = None, step: int = 0) -> Tensor:
        """One generator evaluation on an aligned condition (see :meth:`frame_condition`)."""
        x = self.generator(z, cond, spk, cfg_scale, mask=mask, noise_labels=noise_labels, step=step)
        return x + cond[:, : self.n_mels] if self.residual_prior else x

    def rollout(self, z: Tensor, cond: Tensor, spk: Tensor, cfg_scale: Tensor, steps: int,
                generator: torch.Generator | None = None, **kw) -> Tensor:
        """``x_0 = z``, ``x_{k+1} = G(x_k, k)`` for ``k < steps`` (the released model: ``steps = 1``, one pass).

        Unless given, one style code per trajectory is drawn from ``generator`` (the RNG of ``z``), so a seeded
        generator fully determines the output."""
        if kw.get("noise_labels") is None:
            kw["noise_labels"] = self.style_codes(spk.shape[0], spk.device, generator)
        x = z
        for k in range(steps):
            x = self.generate(x, cond, spk, cfg_scale, step=k, **kw)
        return x

    @staticmethod
    def frame_condition(h: Tensor, mu: Tensor, attn: Tensor) -> Tensor:
        """Aligned generator condition ``[B, n_mels + d, T]``: prior mean and text features."""
        return torch.cat([expand(mu, attn), expand(h, attn)], 1)

    @torch.no_grad()
    def aligned_condition(self, text: Tensor, text_len: Tensor, spk: Tensor, length_scale: float = 1.0,
                          durations: Tensor | None = None, pitch: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """The text half of :meth:`synthesize`: the generator's aligned condition ``[B, n_mels + d, T]`` and the mel
        lengths ``[B]`` (same arguments)."""
        h, mu, logw, x_mask = self.encoder(text, text_len, spk)
        if self.pitch_enabled:
            h, _ = self.pitch_condition(h, x_mask, spk, pitch)
        if durations is None:
            attn, y_len = durations_to_alignment(logw, x_mask, length_scale)
        else:
            attn, y_len = frames_to_alignment(durations, x_mask)
        return self.frame_condition(h, mu, attn), y_len

    def style_codes(self, batch: int, device, generator: torch.Generator | None = None) -> Tensor:
        """One style code per trajectory ``[B, noise_coords]``, drawn as :meth:`rollout` draws them."""
        net = self.generator
        return torch.randint(0, net.noise_classes, (batch, max(1, net.noise_coords)), device=device,
                             generator=generator)

    @torch.no_grad()
    def synthesize(self, text: Tensor, text_len: Tensor, spk: Tensor, cfg_scale: float = 1.0,
                   temperature: float = 1.0, length_scale: float = 1.0, noise_labels: Tensor | None = None,
                   generator: torch.Generator | None = None, steps: int | None = None,
                   durations: Tensor | None = None, pitch: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """Generate mels with ``steps`` generator evaluations (default: the trained number, 1 = one-step).

        ``z`` and the style codes both come from ``generator``. Returns normalised mels and lengths. Prosody
        overrides: ``durations`` (frames per token ``[B, N]``, used as they are, without ``length_scale``) and
        ``pitch`` (normalised token log-F0 ``[B, 1, N]``) replace the predicted ones."""
        cond, y_len = self.aligned_condition(text, text_len, spk, length_scale, durations, pitch)
        z = torch.randn(cond.shape[0], self.n_mels, cond.shape[-1], device=cond.device, generator=generator)
        mask = sequence_mask(y_len, cond.shape[-1])
        alpha = torch.full((text.shape[0],), float(cfg_scale), device=text.device)
        steps = self.generator.num_steps if steps is None else steps
        mel = self.rollout(z * temperature, cond, spk, alpha, steps, generator=generator, mask=mask,
                           noise_labels=noise_labels)
        return mel * mask[:, None], y_len


def load_tts(path: str | Path, device: str = "cpu") -> tuple[DriftingTTS, dict, dict]:
    """An exported acoustic model (``model_ema.pt``, ``drifting_tts_v3.2.pt``): the model in eval mode, the
    checkpoint's config (a plain dict; only ``config["model"]`` is read here) and the mel statistics (``{"mean",
    "std", "backend"}``). The calibration stored by ``calibrate-durations`` (``duration_scale``, per-voice
    ``duration_scales``, the preferred noise ``temperature``) is set on the model."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    model = DriftingTTS(cfg["model"], num_speakers=ck["num_speakers"], n_mels=ck.get("n_mels", 100))
    model.load_state_dict(ck["ema"])
    model.duration_scale = float(ck.get("duration_scale", 1.0))
    model.duration_scales = {int(k): float(v) for k, v in (ck.get("duration_scales") or {}).items()}
    model.temperature = ck.get("temperature")
    stats = ck.get("stats")
    if stats is None:
        raise ValueError(f"{path}: no mel statistics ('stats') in the checkpoint")
    if stats.get("backend", "bigvgan") != "bigvgan":
        raise ValueError(f"{path}: a model trained on {stats.get('backend')!r} features; this package runs models "
                         "trained on BigVGAN-style log-mels (the released ones)")
    return model.to(device).eval(), cfg, stats
