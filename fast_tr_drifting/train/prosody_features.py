"""Training features of the prosody predictor: multi-scale feature maps of token-level prosody sequences (the drift
loss of ``train-prosody`` compares them) and the word / utterance summaries."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor

from ..models.prosody_net import SPACE_ID, ProsodyStats


def word_index(ids: Tensor) -> Tensor:
    """Token ids ``[B, N]`` -> word index ``[B, N]`` (words are separated by space tokens)."""
    return torch.cumsum((ids == SPACE_ID).long(), 1)


def _masked_mean_std(x: Tensor, m: Tensor, dim: int) -> tuple[Tensor, Tensor]:
    n = m.sum(dim).clamp_min(1)
    mean = (x * m).sum(dim) / n
    var = (((x - mean.unsqueeze(dim)) ** 2) * m).sum(dim) / n
    return mean, (var + 1e-6).sqrt()


def summary_features(y: Tensor, mask: Tensor, word: Tensor, stats: ProsodyStats, standardize: bool = True,
                     n_words: int | None = None) -> tuple[Tensor, Tensor, Tensor]:
    """Word and utterance summaries of standardised sequences ``y`` ``[B, S, 2, N]``.

    Word (``[B, S, W, 4]``): mean ld, mean pitch, pitch range, log total frames. Utterance (``[B, S, 1, 6]``): mean
    and std of ld and of pitch, pitch slope over the utterance, log total frames. Returns both and the valid-word
    mask ``[B, W]``. ``n_words``: ``W`` if known on the host (else read from ``word``, a device sync)."""
    B, S, C, N = y.shape
    m = mask[:, None, :].float()  # [B, 1, N]
    ld_raw, _ = stats.denorm(y)
    frames = torch.exp(ld_raw) * m  # [B, S, N]
    W = n_words or (int(word.max()) + 1 if word.numel() else 1)
    onehot = F.one_hot(word.clamp_min(0), W).float() * mask[..., None].float()  # [B, N, W]
    cnt = onehot.sum(1)  # [B, W]
    wvalid = cnt > 0
    wmean = torch.einsum("bscn,bnw->bswc", y, onehot) / cnt.clamp_min(1)[:, None, :, None]  # [B, S, W, 2]
    big = 1e4
    p = y[:, :, 1]  # [B, S, N]
    sel = onehot.bool()[:, None]  # [B, 1, N, W]
    pmax = torch.where(sel, p[..., None], torch.full_like(p[..., None], -big)).amax(2)  # [B, S, W]
    pmin = torch.where(sel, p[..., None], torch.full_like(p[..., None], big)).amin(2)
    wrange = torch.where(wvalid[:, None], pmax - pmin, torch.zeros_like(pmax))
    wframes = torch.log(torch.einsum("bsn,bnw->bsw", frames, onehot).clamp_min(1.0))
    word_f = torch.cat([wmean, wrange[..., None], wframes[..., None]], -1)  # [B, S, W, 4]

    mean_ld, std_ld = _masked_mean_std(y[:, :, 0], m, -1)
    mean_p, std_p = _masked_mean_std(p, m, -1)
    n = mask.float().sum(-1, keepdim=True).clamp_min(2)  # [B, 1]
    pos = torch.arange(N, device=y.device)[None].float() / (n - 1) - 0.5  # [B, N]
    pos = (pos - (pos * mask).sum(-1, keepdim=True) / n) * mask
    slope = ((p - mean_p[..., None]) * pos[:, None] * m).sum(-1) / (pos**2).sum(-1)[:, None].clamp_min(1e-6)
    total = torch.log(frames.sum(-1).clamp_min(1.0))
    utt_f = torch.stack([mean_ld, std_ld, mean_p, std_p, slope, total], -1)[:, :, None]  # [B, S, 1, 6]
    if standardize:
        word_f = (word_f - stats.word[0]) / stats.word[1]
        utt_f = (utt_f - stats.utt[0]) / stats.utt[1]
    return word_f, utt_f, wvalid


def prosody_features(y: Tensor, mask: Tensor, word: Tensor, stats: ProsodyStats, pools=(3, 9, 27),
                     windows=(8, 32), n_words: int | None = None) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    """Multi-scale feature maps of standardised prosody sequences ``y`` ``[B, S, 2, N]`` (mask ``[B, N]``).

    Every map is ``[B, S, L, D]`` with a valid-location mask ``[B, L]``; each (utterance, location) is one drift
    problem. Maps: ``tok`` (per-token values), ``d1`` (first differences: smoothness), ``avg<k>`` (masked means over
    ``k`` tokens, stride ``k // 2``), ``win<k>`` (the whole window of ``k`` tokens flattened: local shape), ``word``
    and ``utt`` (standardised summaries, :func:`summary_features`)."""
    B, S, C, N = y.shape
    mf = mask[:, None, None, :].float()
    y = y * mf
    feats, valid = {"tok": y.transpose(2, 3)}, {"tok": mask}
    feats["d1"] = (y[..., 1:] - y[..., :-1]).transpose(2, 3) * mf[..., 1:].transpose(2, 3)
    valid["d1"] = mask[:, 1:] & mask[:, :-1]
    flat, mflat = y.reshape(B * S, C, N), mask[:, None].float()  # [B*S, C, N], [B, 1, N]
    for k in pools:
        stride = max(1, k // 2)
        pad = (0, max(0, k - N))
        s = F.avg_pool1d(F.pad(flat, pad), k, stride) * k
        c = F.avg_pool1d(F.pad(mflat, pad), k, stride) * k  # [B, 1, L]
        L = s.shape[-1]
        feats[f"avg{k}"] = (s.view(B, S, C, L) / c.clamp_min(1)[:, None]).transpose(2, 3)
        valid[f"avg{k}"] = c[:, 0] > 0
    for k in windows:
        stride = max(1, k // 2)
        n_win = max(1, math.ceil((N - k) / stride) + 1)
        total = (n_win - 1) * stride + k
        w = F.pad(y, (0, total - N)).unfold(-1, k, stride)  # [B, S, C, L, k]
        feats[f"win{k}"] = w.permute(0, 1, 3, 2, 4).reshape(B, S, n_win, C * k)
        starts = torch.arange(n_win, device=y.device) * stride
        valid[f"win{k}"] = starts[None] < mask.sum(-1, keepdim=True)
    word_f, utt_f, wvalid = summary_features(y, mask, word, stats, n_words=n_words)
    feats["word"], valid["word"] = word_f, wvalid
    feats["utt"], valid["utt"] = utt_f, torch.ones(B, 1, dtype=torch.bool, device=y.device)
    return feats, valid
