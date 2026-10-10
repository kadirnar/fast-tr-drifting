"""Stochastic prosody predictor: noise + text features + speaker -> per-token durations and pitch in one pass .

The generator of a pitch-conditioned :class:`~fast_tr_drifting.models.tts.DriftingTTS` is trained on ground-truth MAS
durations and ground-truth token pitch, so its deterministic duration / pitch regressors can be replaced at inference
by a sampler without retraining it. :class:`ProsodyNet` maps the frozen encoder states ``h``, the speaker embedding,
the regressors' own predictions and noise to three channels per token:

* ``ld``: the log-duration in frames (MAS durations are >= 1 frame, so ``log d >= 0``; inference rounds ``exp(ld)``);
* ``p``: a continuous token pitch contour (normalised log-F0, unvoiced tokens interpolated from their neighbours);
* a voicing logit: unvoiced tokens get pitch 0, as in :func:`~fast_tr_drifting.models.text_encoder.token_pitch`.

``ld`` and ``p`` are standardised (:class:`ProsodyStats`) and predicted as a residual over the regressors'
predictions. Three training objectives share the network (``kind``):

* ``drift``: per-token and global noise in, one pass; drifting loss on multi-scale feature maps of the sequence
  (:func:`fast_tr_drifting.train.prosody_features.prosody_features`), one positive (the recording's prosody) and
  ``G`` samples per utterance (``train-prosody``);
* ``mse``: no noise, regression (the deterministic status quo with the same backbone);
* ``flow``: conditional flow matching on the residual, a few Euler steps at inference.
"""

from __future__ import annotations

import math
import warnings

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..text import BLANK_ID, PUNCTUATION, SYMBOL_TO_ID, SYMBOLS
from .text_encoder import EncoderLayer

KINDS = ("drift", "mse", "flow")
# what ProsodyPredictor.load reads: a published checkpoint keeps only these (export_checkpoint)
CHECKPOINT_KEYS = ("ema", "stats", "net_cfg", "cond_dim", "duration_scales", "temperature", "flow_steps",
                   "tts_fingerprint", "pause_edges", "duration_temperature", "rhythm")
SPACE_ID = SYMBOL_TO_ID[" "]
PUNCT_IDS = [SYMBOL_TO_ID[c] for c in PUNCTUATION if c != " "]


class ProsodyStats(nn.Module):
    """Standardisation of the two channels and of the word / utterance summary features (set from the data; the
    summaries are training features, :mod:`fast_tr_drifting.train.prosody_features`)."""

    def __init__(self):
        super().__init__()
        self.register_buffer("seq", torch.tensor([[0.0, 1.0], [0.0, 1.0]]))  # (ld, p) x (mean, std)
        self.register_buffer("word", torch.zeros(2, 4) + torch.tensor([[0.0], [1.0]]))  # mean / std of 4 features
        self.register_buffer("utt", torch.zeros(2, 6) + torch.tensor([[0.0], [1.0]]))  # mean / std of 6 features

    def norm(self, ld: Tensor, p: Tensor) -> Tensor:
        """``[..., N]`` log-durations and pitch -> standardised ``[..., 2, N]``."""
        s = self.seq
        return torch.stack([(ld - s[0, 0]) / s[0, 1], (p - s[1, 0]) / s[1, 1]], -2)

    def denorm(self, y: Tensor) -> tuple[Tensor, Tensor]:
        s = self.seq
        return y[..., 0, :] * s[0, 1] + s[0, 0], y[..., 1, :] * s[1, 1] + s[1, 0]


def boundary_tokens(ids: Tensor) -> Tensor:
    """Tokens between words ``[..., N]`` bool: spaces, punctuation and the blanks next to them. A pause between words
    lives in their durations; the letters' durations (a character and the blank after it) are the articulation."""
    mark = ids == SPACE_ID
    for i in PUNCT_IDS:
        mark = mark | (ids == i)
    left, right = F.pad(mark, (1, 0))[..., :-1], F.pad(mark, (0, 1))[..., 1:]
    return mark | ((ids == BLANK_ID) & (left | right))


def edge_tokens(lengths: Tensor, n: int, first: int = 3, last: int = 2) -> Tensor:
    """The sentence's edges ``[B, n]`` bool: its ``first`` tokens (the leading blank, the first character and its blank)
    and its ``last`` valid ones (the final punctuation and blank)."""
    pos = torch.arange(n, device=lengths.device)[None]
    return (pos < first) | ((pos >= lengths[:, None] - last) & (pos < lengths[:, None]))


def floor_letters(frames: Tensor, ids: Tensor, min_frames: float = 0.0, rel: float = 0.0,
                  ref_frames: Tensor | None = None) -> Tensor:
    """Raise short letters: a letter (a character token and the blank after it) gets at least ``min_frames`` frames
    and at least ``rel`` times its ``ref_frames`` (e.g. the regressors' durations, same shape as ``frames``); the
    missing frames go to the character token. ``frames`` / ``ids`` ``[B, N]``."""
    table = torch.tensor([len(c) == 1 and c.isalpha() for c in SYMBOLS], device=ids.device)
    letter = table[ids]
    total = frames + F.pad(frames[:, 1:], (0, 1))
    need = torch.full_like(frames, float(min_frames))
    if rel > 0 and ref_frames is not None:
        need = torch.maximum(need, torch.round(rel * (ref_frames + F.pad(ref_frames[:, 1:], (0, 1)))))
    return frames + (need - total).clamp_min(0) * letter


def _time_embedding(t: Tensor, dim: int = 64) -> Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(1000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    ang = t.float()[:, None] * 1000.0 * freqs[None]
    return torch.cat([ang.sin(), ang.cos()], -1)


class ProsodyNet(nn.Module):
    """Non-causal transformer over tokens; see the module docstring. ``cond`` = ``[h; speaker; regressors]``."""

    def __init__(self, cond_dim: int, kind: str = "drift", d: int = 256, layers: int = 4, heads: int = 4,
                 ffn: int = 1024, dropout: float = 0.0, noise_tok: int = 16, noise_glob: int = 32,
                 out_init: float = 0.1):
        super().__init__()
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
        self.kind, self.noise_tok, self.noise_glob, self.cond_dim = kind, noise_tok, noise_glob, cond_dim
        c_in = cond_dim + (noise_tok if kind == "drift" else 0) + (2 if kind == "flow" else 0)
        self.inp = nn.Conv1d(c_in, d, 1)
        g_dim = noise_glob if kind == "drift" else 64 if kind == "flow" else 0
        self.glob = nn.ModuleList([nn.Linear(g_dim, d) for _ in range(layers + 1)]) if g_dim else None
        self.prenet = nn.ModuleList([nn.Conv1d(d, d, 5, padding=2) for _ in range(2)])
        self.prenet_norms = nn.ModuleList([nn.LayerNorm(d) for _ in range(2)])
        self.layers = nn.ModuleList([EncoderLayer(d, heads, ffn, dropout) for _ in range(layers)])
        self.norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, 3)
        with torch.no_grad():
            self.out.weight.mul_(out_init)
            self.out.bias.zero_()

    def forward(self, cond: Tensor, mask: Tensor, z_tok: Tensor | None = None, z_glob: Tensor | None = None,
                x_t: Tensor | None = None, t: Tensor | None = None) -> Tensor:
        """``cond`` ``[B, C, N]``, ``mask`` ``[B, 1, N]`` -> ``[B, 3, N]``: residual (ld, p) or velocity; voicing."""
        parts = [cond]
        g = None
        if self.kind == "drift":
            parts.append(z_tok)
            g = z_glob
        elif self.kind == "flow":
            parts.append(x_t)
            g = _time_embedding(t)
        m = mask.transpose(1, 2)  # [B, N, 1]
        x = self.inp(torch.cat(parts, 1) * mask).transpose(1, 2)
        if g is not None:
            x = x + self.glob[0](g)[:, None]
        for conv, norm in zip(self.prenet, self.prenet_norms):
            x = x + norm(F.gelu(conv((x * m).transpose(1, 2)).transpose(1, 2)))
        x = x * m
        for i, layer in enumerate(self.layers):
            if g is not None:
                x = x + self.glob[i + 1](g)[:, None] * m
            x = layer(x, m)
        return (self.out(self.norm(x)) * m).transpose(1, 2)


class ProsodyPredictor(nn.Module):
    """A trained :class:`ProsodyNet` with its statistics: samples durations and token pitch for a TTS model."""

    def __init__(self, net_cfg: dict, cond_dim: int):
        super().__init__()
        self.net_cfg = dict(net_cfg)
        self.net = ProsodyNet(cond_dim, **self.net_cfg)
        self.stats = ProsodyStats()
        self.duration_scales: dict[int, float] = {}
        self.temperature: float | None = None  # preferred prosody temperature (train-prosody --calibrate-only)
        # preferred noise temperature of the sampled letters' durations (None: the prosody temperature; the pauses keep
        # the prosody temperature either way); train-prosody --calibrate-only calibrate.duration_temperature
        self.duration_temperature: float | None = None
        # voice -> speaker whose rhythm its sampled durations borrow (duration_scales then holds that rhythm's factor)
        self.rhythm: dict[int, int] = {}
        # leading + trailing silence of its generated sentences per voice (PausePolicy.for_voice(edge=...))
        self.pause_edges: dict[int, float] = {}
        self.flow_steps = 8

    @property
    def kind(self) -> str:
        return self.net.kind

    @staticmethod
    def condition(tts, h: Tensor, x_mask: Tensor, spk: Tensor, logw_det: Tensor,
                  stats: ProsodyStats) -> tuple[Tensor, Tensor]:
        """``[h; speaker embedding; standardised regressor predictions]`` ``[B, C, N]`` and the standardised
        regressor predictions ``[B, 2, N]`` (the base the network's residual is added to)."""
        s = tts.encoder.spk(spk)[:, :, None].expand(-1, -1, h.shape[-1])
        pitch_det = tts.pitch_predictor(torch.cat([h, s], 1), x_mask)
        base = stats.norm(logw_det[:, 0], pitch_det[:, 0]) * x_mask
        return torch.cat([h, s, base], 1) * x_mask, base

    def sample(self, cond: Tensor, base: Tensor, mask: Tensor, temperature: float = 1.0,
               generator: torch.Generator | None = None, steps: int | None = None, spread: float = 1.0,
               mean_samples: int = 16, duration_temperature: float | None = None,
               duration_cond: tuple[Tensor, Tensor] | None = None,
               duration_mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """Standardised ``(ld, p)`` ``[B, 2, N]`` and the voicing logit ``[B, N]``; noise from ``generator``.

        ``temperature`` scales the input noise. ``spread`` != 1 (output-space temperature) draws ``mean_samples``
        samples in one batch and returns ``mean + spread * (sample - mean)`` for the first one: below 1 it trades
        expressiveness and seed diversity for per-token accuracy (the noise temperature of the drift sampler mostly
        changes the diversity between seeds, not the spread within an utterance).

        The log-durations can be drawn apart from the pitch: at their own noise temperature
        (``duration_temperature``) and/or for another condition (``duration_cond``: the ``(cond, base)`` of
        :meth:`condition` for another speaker, whose rhythm is borrowed). Given either, the same unit noise runs as a
        second batch row, and the log-duration channel comes from it (:meth:`_draw`), for the tokens of
        ``duration_mask`` ``[B, N]`` only if given (e.g. the letters: the pauses keep the first row's); without, the
        sample is the one-pass one."""
        dur = None
        if duration_temperature is not None or duration_cond is not None:
            dc, db = duration_cond if duration_cond is not None else (cond, base)
            dur = (temperature if duration_temperature is None else duration_temperature, dc, db, duration_mask)
        if spread == 1.0 or self.net.kind == "mse":
            return self._draw(cond, base, mask, temperature, generator, steps, dur)
        K, B = mean_samples, cond.shape[0]
        if dur is not None:
            dur = (dur[0], dur[1].repeat(K, 1, 1), dur[2].repeat(K, 1, 1),
                   None if duration_mask is None else duration_mask.repeat(K, 1))
        y, v = self._draw(cond.repeat(K, 1, 1), base.repeat(K, 1, 1), mask.repeat(K, 1, 1), temperature, generator,
                          steps, dur)
        y = y.view(K, B, *y.shape[1:])
        mean = y.mean(0)
        return (mean + spread * (y[0] - mean)) * mask, v[:B]

    def _draw(self, cond: Tensor, base: Tensor, mask: Tensor, temperature: float,
              generator: torch.Generator | None, steps: int | None,
              durations: tuple[float, Tensor, Tensor, Tensor | None] | None = None) -> tuple[Tensor, Tensor]:
        """One sample per row. ``durations``: ``(temperature, cond, base, mask)`` of a second row that gives the
        log-durations (of the tokens in ``mask`` ``[B, N]``, or all); the noise drawn for the first row is reused
        there (scaled by its own temperature), so both rows run as one batch of ``2 B`` and the draws from
        ``generator`` do not change."""
        B, _, N = cond.shape
        net, dev = self.net, cond.device
        two = durations is not None
        if two:
            td, dc, db, dmask = durations
            cond, base, mask = torch.cat([cond, dc]), torch.cat([base, db]), torch.cat([mask, mask])

        def scaled(z: Tensor) -> Tensor:
            return torch.cat([z * temperature, z * td]) if two else z * temperature

        if net.kind == "drift":
            z_tok = torch.randn(B, net.noise_tok, N, device=dev, generator=generator)
            z_glob = torch.randn(B, net.noise_glob, device=dev, generator=generator)
            out = net(cond, mask, scaled(z_tok), scaled(z_glob))
            y = (base + out[:, :2]) * mask
        elif net.kind == "mse":
            out = net(cond, mask)
            y = (base + out[:, :2]) * mask
        else:
            steps = steps or self.flow_steps
            x = scaled(torch.randn(B, 2, N, device=dev, generator=generator)) * mask
            for k in range(steps):
                t = torch.full((x.shape[0],), k / steps, device=dev)
                out = net(cond, mask, x_t=x, t=t)
                x = (x + out[:, :2] / steps) * mask
            y = (base + x) * mask
        if two:  # log-durations from the second row, pitch and voicing from the first
            ld = y[B:, :1] if dmask is None else torch.where(dmask[:, None], y[B:, :1], y[:B, :1])
            y = torch.cat([ld, y[:B, 1:]], 1)
        return y, out[:B, 2]

    def frames_and_pitch(self, y: Tensor, voiced_logit: Tensor, mask: Tensor,
                         length_scale: float | Tensor = 1.0) -> tuple[Tensor, Tensor]:
        """Standardised samples -> integer frames ``[B, N]`` (rounded, >= 1) and token pitch ``[B, 1, N]``."""
        ld, p = self.stats.denorm(y)
        frames = torch.round(torch.exp(ld) * length_scale).clamp_min(1) * mask[:, 0]
        pitch = torch.where(voiced_logit > 0, p, torch.zeros_like(p)) * mask[:, 0]
        return frames, pitch[:, None]

    @torch.no_grad()
    def predict(self, tts, text: Tensor, text_len: Tensor, spk: Tensor, temperature: float = 1.0,
                length_scale: float | Tensor = 1.0, generator: torch.Generator | None = None,
                spread: float = 1.0, duration_temperature: float | None = None,
                duration_speaker: Tensor | None = None, edge_scale: float | Tensor | None = None,
                min_letter_frames: float = 0.0, rel_letter_floor: float = 0.0) -> tuple[Tensor, Tensor]:
        """Sample frames per token ``[B, N]`` and token pitch ``[B, 1, N]`` for ``tts`` (its frozen encoder and
        regressors give the condition); pass them to :meth:`DriftingTTS.synthesize` as ``durations`` / ``pitch``.
        ``temperature`` / ``spread``: see :meth:`sample`. ``duration_temperature``: the noise temperature of the
        letters' durations (a character and the blank after it); the tokens between words (:func:`boundary_tokens`:
        the pauses) keep ``temperature``, like the pitch. ``duration_speaker`` ``[B]``: sample all durations as for
        this speaker (its rhythm, pauses included), the pitch for ``spk``; ``length_scale`` should then carry the
        factor that brings that rhythm to the voice's rate (:attr:`rhythm`, :attr:`duration_scales`).
        ``edge_scale`` (with ``duration_speaker``): the sentence's edges (:func:`edge_tokens`) keep the voice's own
        regressor durations, ``ceil(exp(logw) * edge_scale)``: a voice's generator learned its own sentence starts
        and ends (the female recordings, for one, start abruptly). ``min_letter_frames`` / ``rel_letter_floor``:
        :func:`floor_letters`, relative to the regressors' durations (off by default)."""
        h, _, logw, x_mask = tts.encoder(text, text_len, spk)
        cond, base = self.condition(tts, h, x_mask, spk, logw, self.stats)
        dcond = None
        if duration_speaker is not None:
            hd, _, logwd, _ = tts.encoder(text, text_len, duration_speaker)
            dcond = self.condition(tts, hd, x_mask, duration_speaker, logwd, self.stats)
        dmask = ~boundary_tokens(text) if duration_speaker is None else None  # the pauses keep the first row's
        y, vlogit = self.sample(cond, base, x_mask, temperature, generator=generator, spread=spread,
                                duration_temperature=duration_temperature, duration_cond=dcond, duration_mask=dmask)
        frames, pitch = self.frames_and_pitch(y, vlogit, x_mask, length_scale)
        if duration_speaker is not None and edge_scale is not None:
            own = torch.ceil(torch.exp(logw[:, 0]) * edge_scale)
            frames = torch.where(edge_tokens(text_len, text.shape[1]), own, frames) * x_mask[:, 0]
        if min_letter_frames or rel_letter_floor:
            ref = torch.exp(logw[:, 0]) * length_scale
            frames = floor_letters(frames, text, min_letter_frames, rel_letter_floor, ref) * x_mask[:, 0]
        return frames, pitch

    @classmethod
    def load(cls, path, device="cpu", tts=None) -> ProsodyPredictor:
        """``path``: a checkpoint, or a name of :data:`fast_tr_drifting.hub.PROSODY_MODELS` (e.g. ``drift``,
        downloaded). With ``tts``, the condition size is checked, and a checkpoint that records the fingerprint of the
        encoder it was trained on (:func:`tts_fingerprint`) warns when ``tts`` has another one."""
        from ..hub import resolve_prosody

        path = resolve_prosody(path)
        ck = torch.load(path, map_location="cpu", weights_only=False)
        cond_dim = ck["cond_dim"]
        if tts is not None:
            want = tts.encoder.d + tts.encoder.spk.embedding_dim + 2
            if want != cond_dim:
                raise ValueError(f"prosody model {path} expects {cond_dim} condition channels, the TTS gives {want}")
            if ck.get("tts_fingerprint") and ck["tts_fingerprint"] != tts_fingerprint(tts):
                warnings.warn(f"prosody model {path} was trained on the text encoder of another acoustic model "
                              f"(fingerprint {ck['tts_fingerprint']}, this one {tts_fingerprint(tts)}): check its "
                              "durations and pitch before using it", stacklevel=2)
        extra = sorted(k for k in ("word_dim", "sent_dim", "ctx_pitch_only") if ck["net_cfg"].get(k))
        if extra:
            raise ValueError(f"prosody model {path} uses {', '.join(extra)}: word / sentence features are not part "
                             "of this package (see kadirnar/drifting-tts)")
        net_cfg = {k: v for k, v in ck["net_cfg"].items() if k not in ("word_dim", "word_model", "sent_dim",
                                                                       "ctx_pitch_only", "pitch_layers",
                                                                       "ctx_boundaries")}
        p = cls(net_cfg, cond_dim)
        p.net.load_state_dict(ck["ema"])
        p.stats.load_state_dict(ck["stats"])
        p.duration_scales = {int(k): float(v) for k, v in ck.get("duration_scales", {}).items()}
        p.temperature = ck.get("temperature")
        p.duration_temperature = ck.get("duration_temperature")
        p.rhythm = {int(k): int(v) for k, v in (ck.get("rhythm") or {}).items()}
        p.pause_edges = {int(k): float(v) for k, v in ck.get("pause_edges", {}).items()}
        p.flow_steps = int(ck.get("flow_steps", 8))
        return p.to(device).eval()


def tts_fingerprint(tts) -> str:
    """Short hash of the weights a prosody predictor is conditioned on: the text encoder (with the duration
    predictor and speaker table) and the token pitch predictor of a :class:`DriftingTTS`."""
    import hashlib

    h = hashlib.sha256()
    state = {**{f"encoder.{k}": v for k, v in tts.encoder.state_dict().items()},
             **{f"pitch_predictor.{k}": v for k, v in tts.pitch_predictor.state_dict().items()}}
    for name in sorted(state):
        h.update(name.encode())
        h.update(state[name].detach().float().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()[:16]


def export_checkpoint(ck: dict, tts=None) -> dict:
    """A training checkpoint (``train-prosody``'s ``prosody_ema.pt``) -> what :meth:`ProsodyPredictor.load` needs
    (:data:`CHECKPOINT_KEYS`), without the training config, paths or step; with ``tts``, the fingerprint of its
    encoder. Tensors are copied to the CPU, contiguous."""
    out = {k: ck[k] for k in CHECKPOINT_KEYS if k in ck}
    for k in ("ema", "stats"):
        out[k] = {n: t.detach().cpu().contiguous().clone() for n, t in out[k].items()}
    out["net_cfg"] = dict(out["net_cfg"])
    out["duration_scales"] = {int(k): float(v) for k, v in out.get("duration_scales", {}).items()}
    if "pause_edges" in out:
        out["pause_edges"] = {int(k): float(v) for k, v in out["pause_edges"].items()}
    if "rhythm" in out:
        out["rhythm"] = {int(k): int(v) for k, v in out["rhythm"].items()}
    if tts is not None:
        out["tts_fingerprint"] = tts_fingerprint(tts)
    return out
