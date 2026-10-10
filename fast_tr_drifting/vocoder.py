"""The Vocos vocoder of the release, vendored for inference: log-mel ``[B, 100, T]`` -> 24 kHz audio ``[B, T * 256]``.

The network is Vocos (Siuzdak, 2023, arXiv 2306.00814; https://github.com/gemelo-ai/vocos, MIT License, Copyright (c)
2023 Charactr Inc.): a ConvNeXt backbone and an inverse-STFT head. Only the inference path is kept here, so neither
the ``vocos`` package nor ``torchaudio`` / ``encodec`` are needed. The fine-tunes of this model (``vocos-v2``,
``vocos-v2-balanced``) read BigVGAN-style mels and use a ``same``-padded head: frame ``i`` is centred on sample
``i * 256 + 128`` and ``T`` frames give ``T * 256`` samples. The module and parameter names follow the ``vocos``
package, so its checkpoints load as they are (``fast_tr_drifting.train.finetune_vocoder`` writes them).
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .alignment import sequence_mask
from .audio import HOP_LENGTH, N_FFT

STREAM_CONTEXT = 32  # frames of context on each side of a streaming window: the fp32 noise floor (> 86 dB SNR)


class ConvNeXtBlock(nn.Module):
    """Vocos's ConvNeXt block (no adaptive norm): depthwise conv, LayerNorm, pointwise MLP, layer scale, residual."""

    def __init__(self, dim: int, intermediate_dim: int, layer_scale_init_value: float):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, intermediate_dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(intermediate_dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.dwconv(x)
        x = x.transpose(1, 2)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = self.gamma * x
        x = x.transpose(1, 2)
        return residual + x


class VocosBackbone(nn.Module):
    """``[B, C, T]`` mel -> ``[B, T, dim]`` features."""

    def __init__(self, input_channels: int = 100, dim: int = 512, intermediate_dim: int = 1536, num_layers: int = 8):
        super().__init__()
        self.embed = nn.Conv1d(input_channels, dim, kernel_size=7, padding=3)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.convnext = nn.ModuleList([ConvNeXtBlock(dim, intermediate_dim, 1 / num_layers)
                                       for _ in range(num_layers)])
        self.final_layer_norm = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, x: Tensor) -> Tensor:
        x = self.embed(x)
        x = self.norm(x.transpose(1, 2)).transpose(1, 2)
        for block in self.convnext:
            x = block(x)
        return self.final_layer_norm(x.transpose(1, 2))


class _ISTFT(nn.Module):
    """Holds the synthesis window (``head.istft.window`` in the checkpoints) and the framing."""

    def __init__(self, n_fft: int = N_FFT, padding: str = "same"):
        super().__init__()
        self.padding = padding
        self.register_buffer("window", torch.hann_window(n_fft))


class ISTFTHead(nn.Module):
    """Linear layer to log-magnitude and phase of ``n_fft // 2 + 1`` bins (inverse STFT in :func:`ola_istft`)."""

    def __init__(self, dim: int = 512, n_fft: int = N_FFT):
        super().__init__()
        self.out = nn.Linear(dim, n_fft + 2)
        self.istft = _ISTFT(n_fft)


class Vocos(nn.Module):
    def __init__(self, input_channels: int = 100, dim: int = 512, intermediate_dim: int = 1536, num_layers: int = 8,
                 n_fft: int = N_FFT):
        super().__init__()
        self.backbone = VocosBackbone(input_channels, dim, intermediate_dim, num_layers)
        self.head = ISTFTHead(dim, n_fft)

    def forward(self, mel: Tensor) -> Tensor:
        """``[B, 100, T]`` log-mel -> ``[B, T * hop]`` waveform (unclamped)."""
        n = mel.shape[-1] * HOP_LENGTH
        h = self.head.out(self.backbone(mel)).transpose(1, 2)
        mag, p = h.chunk(2, dim=1)
        spec = torch.exp(mag).clip(max=1e2) * (torch.cos(p) + 1j * torch.sin(p))
        return ola_istft(spec, self.head.istft.window)[..., :n]

    @classmethod
    def from_state_dict(cls, state: dict) -> Vocos:
        """A Vocos whose sizes are read from a ``vocos``-package state dict (its mel front end is left out)."""
        dim, mels, _ = state["backbone.embed.weight"].shape
        layers = len({k.split(".")[2] for k in state if k.startswith("backbone.convnext.")})
        model = cls(mels, dim, state["backbone.convnext.0.pwconv1.weight"].shape[0], layers,
                    state["head.out.weight"].shape[0] - 2)
        model.load_state_dict({k: v for k, v in state.items() if not k.startswith("feature_extractor.")})
        return model


def ola_istft(spec: Tensor, window: Tensor) -> Tensor:
    """Least-squares inverse STFT ``[B, n_fft // 2 + 1, T]`` -> ``[B, T * hop]`` (overlap-add, divided by the squared
    window envelope), BigVGAN's framing (frame ``i`` centred on sample ``i * hop + hop / 2``), with no host
    synchronisation, so that it can run inside a CUDA graph."""
    t = spec.shape[-1]
    size, pad = (t - 1) * HOP_LENGTH + N_FFT, (N_FFT - HOP_LENGTH) // 2
    frames = torch.fft.irfft(spec, N_FFT, dim=1) * window[:, None]
    y = F.fold(frames, (1, size), (1, N_FFT), stride=(1, HOP_LENGTH))[:, 0, 0]
    env = F.fold(window.square()[None, :, None].expand(1, -1, t), (1, size), (1, N_FFT), stride=(1, HOP_LENGTH))
    return (y / env[0, 0, 0].clamp_min(1e-11))[:, pad: size - pad]


class Vocoder:
    """Unnormalised log-mel ``[B, 100, T]`` -> waveform ``[B, T * 256]`` at 24 kHz, clamped to [-1, 1].

    ``name``: the registry name or path; ``context``: frames of context for streaming windows
    (:func:`fast_tr_drifting.fast.stream_vocoder`)."""

    def __init__(self, model: Vocos, device: str = "cuda", name: str = "", context: int = STREAM_CONTEXT):
        self.model, self.device, self.name, self.context = model.to(device).eval(), device, name, context

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    @torch.no_grad()
    def __call__(self, log_mel: Tensor) -> Tensor:
        return self.model(log_mel.to(self.device).float()).clamp(-1, 1)


def load_vocoder(spec: str | Path = "vocos-v2", device: str = "cuda") -> Vocoder:
    """A vocoder name of :data:`fast_tr_drifting.hub.VOCODERS` (``vocos-v2``, ``vocos-v2-balanced``; downloaded) or a
    Vocos checkpoint (``{"vocos": state dict, "mel": "bigvgan", "head_padding": "same", ...}``, as written by
    ``fast-tr-drifting finetune-vocoder``)."""
    from .hub import resolve_vocoder

    path = resolve_vocoder(spec)
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if "vocos" not in ck:
        raise ValueError(f"{path}: not a Vocos checkpoint (keys {sorted(ck)[:8]})")
    if ck.get("mel", "vocos") != "bigvgan" or ck.get("head_padding") != "same":
        raise ValueError(f"{path}: a Vocos for {ck.get('mel', 'vocos')!r} mels with a {ck.get('head_padding')!r} "
                         "head; this package runs Vocos fine-tunes on BigVGAN-style mels (head_padding 'same')")
    if ck.get("noise_channels"):
        raise ValueError(f"{path}: a drift-trained vocoder with input noise is not supported here")
    return Vocoder(Vocos.from_state_dict(ck["vocos"]), device, str(spec))


def _masked_vocos(model: Vocos, x: Tensor, m: Tensor) -> Tensor:
    """Vocos on ``x`` ``[B, C, T]`` with the valid-frame mask ``m`` ``[B, 1, T]``: ``[B, T * hop]``, each row as
    vocoded alone (the padding is zeroed before every convolution and left out of the inverse STFT)."""
    bb, head = model.backbone, model.head
    h = bb.embed(x * m)  # each convolution sees zeros after the row's end, as its own padding
    h = bb.norm(h.transpose(1, 2)).transpose(1, 2)
    for block in bb.convnext:
        h = block(h * m)
    h = head.out(bb.final_layer_norm(h.transpose(1, 2))).transpose(1, 2).float()  # the inverse STFT in fp32
    mag, p = h.chunk(2, dim=1)
    spec = torch.exp(mag).clip(max=1e2) * (torch.cos(p) + 1j * torch.sin(p)) * m  # padded frames add nothing
    t = x.shape[-1]
    size, pad = (t - 1) * HOP_LENGTH + N_FFT, (N_FFT - HOP_LENGTH) // 2
    window = head.istft.window
    frames = torch.fft.irfft(spec, N_FFT, dim=1) * window[:, None]
    y = F.fold(frames, (1, size), (1, N_FFT), stride=(1, HOP_LENGTH))[:, 0, 0]
    env = F.fold(window.square()[None, :, None] * m, (1, size), (1, N_FFT), stride=(1, HOP_LENGTH))[:, 0, 0]
    return (y / env.clamp_min(1e-11))[:, pad: size - pad]


@torch.no_grad()
def vocode_masked(vocoder: Vocoder, mel: Tensor, lengths: Tensor, compile: bool = False) -> Tensor:
    """Unnormalised log-mels ``[B, n_mels, T]`` whose rows have ``lengths`` frames -> waveforms ``[B, T * hop]``:
    row ``b`` equals ``vocoder(mel[b:b+1, :, :lengths[b]])`` on its first ``lengths[b] * hop`` samples (zeros after),
    in one batch. ``compile``: the batch through ``torch.compile`` (float rounding only)."""
    from .batched import _compiled

    T = mel.shape[-1]
    m = sequence_mask(lengths.to(mel.device), T)[:, None].float()
    fn = _compiled(vocoder.model, "masked_vocos", _masked_vocos) if compile else _masked_vocos
    wav = fn(vocoder.model, mel.to(vocoder.device).float(), m)
    keep = sequence_mask(lengths.to(mel.device) * HOP_LENGTH, T * HOP_LENGTH)
    return wav.clamp(-1, 1) * keep
