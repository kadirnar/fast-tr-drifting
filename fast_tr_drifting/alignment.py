"""Masks and hard alignments from integer durations (inference; the monotonic alignment search of training lives in
:mod:`fast_tr_drifting.train.mas`)."""

from __future__ import annotations

import torch
from torch import Tensor


def sequence_mask(lengths: Tensor, max_len: int | None = None) -> Tensor:
    max_len = int(lengths.max()) if max_len is None else max_len
    return torch.arange(max_len, device=lengths.device)[None] < lengths[:, None]


def generate_path(durations: Tensor, mask: Tensor) -> Tensor:
    """Hard alignment from integer durations ``[B, N]`` and a ``[B, N, T]`` mask."""
    B, N, T = mask.shape
    cum = torch.cumsum(durations, 1)
    path = sequence_mask(cum.flatten(), T).view(B, N, T).to(mask.dtype)
    path = path - torch.nn.functional.pad(path, (0, 0, 1, 0))[:, :-1]
    return path * mask
