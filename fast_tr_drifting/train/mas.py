"""Training-time alignment: monotonic alignment search (Glow-TTS / Grad-TTS) between the text encoder's prior
mean and the target mel, the prior and duration losses, and the token pitch targets (FastPitch)."""

from __future__ import annotations

import math

import numba
import numpy as np
import torch
from torch import Tensor

from ..alignment import sequence_mask  # noqa: F401 (re-exported for the trainers)

# numba threads for the batched MAS DP. A few threads already make it ~1 ms per batch; more threads mostly add
# contention, and in CPU-quota-limited containers (cgroup ``cpu.max``) they stall the whole training process.
MAS_THREADS = 4


@numba.njit(cache=True, boundscheck=False)
def _maximum_path_single(value: np.ndarray, t_x: int, t_y: int, out: np.ndarray) -> None:
    """Glow-TTS ``maximum_path_each`` on a ``[T_mel, N_text]`` layout: the inner loop over tokens is unit-stride."""
    neg_inf = -1e9
    for y in range(t_y):
        for x in range(max(0, t_x + y - t_y), min(t_x, y + 1)):
            v_cur = neg_inf if x == y else value[y - 1, x]
            if x == 0:
                v_prev = 0.0 if y == 0 else neg_inf
            else:
                v_prev = value[y - 1, x - 1]
            value[y, x] += max(v_prev, v_cur)
    index = t_x - 1
    for y in range(t_y - 1, -1, -1):
        out[y, index] = 1
        if index != 0 and (index == y or value[y - 1, index] < value[y - 1, index - 1]):
            index -= 1


@numba.njit(parallel=True, cache=True)
def _maximum_path_batch(values: np.ndarray, t_xs: np.ndarray, t_ys: np.ndarray, out: np.ndarray) -> None:
    for b in numba.prange(values.shape[0]):
        _maximum_path_single(values[b], t_xs[b], t_ys[b], out[b])


def _to_host(x: Tensor) -> Tensor:
    """Start a device-to-host copy into page-locked memory (the caller synchronises before reading)."""
    if x.device.type != "cuda":
        return x
    return torch.empty(x.shape, dtype=x.dtype, pin_memory=True).copy_(x, non_blocking=True)


@torch.no_grad()
def maximum_path(log_prior: Tensor, mask: Tensor) -> Tensor:
    """Most likely monotonic alignment.

    The DP runs on the CPU (numba) over a transposed ``[T_mel, N_text]`` copy; on CUDA both copies go
    through pinned buffers, so the only host sync is waiting for the scores.

    Args:
        log_prior: ``[B, N_text, T_mel]`` log-likelihood of every frame under every token.
        mask: ``[B, N_text, T_mel]`` valid (token, frame) pairs.
    Returns:
        hard alignment ``[B, N_text, T_mel]`` (each frame assigned to exactly one token).
    """
    device, dtype = log_prior.device, log_prior.dtype
    values = (log_prior * mask).transpose(1, 2).float().contiguous()  # [B, T, N]
    lens = torch.stack([mask[:, :, 0].sum(1), mask[:, 0].sum(1)]).int()  # text / mel lengths
    values, lens = _to_host(values), _to_host(lens)
    if device.type == "cuda":
        torch.cuda.current_stream(device).synchronize()
    out = torch.zeros(values.shape, dtype=torch.int8, pin_memory=device.type == "cuda")
    threads = numba.get_num_threads()
    numba.set_num_threads(max(1, min(MAS_THREADS, threads, values.shape[0])))
    try:
        _maximum_path_batch(values.numpy(), lens[0].numpy(), lens[1].numpy(), out.numpy())
    finally:
        numba.set_num_threads(threads)
    return out.to(device, non_blocking=True).transpose(1, 2).to(dtype, memory_format=torch.contiguous_format)


def gaussian_log_prior(mu: Tensor, y: Tensor) -> Tensor:
    """``log N(y_t; mu_n, I)`` for all token / frame pairs: ``[B, C, N] x [B, C, T] -> [B, N, T]``."""
    C = mu.shape[1]
    y_sq = -0.5 * (y**2).sum(1, keepdim=True)  # [B, 1, T]
    mu_sq = -0.5 * (mu**2).sum(1)[..., None]  # [B, N, 1]
    cross = torch.einsum("bcn,bct->bnt", mu, y)
    return y_sq + mu_sq + cross - 0.5 * C * math.log(2 * math.pi)


def align(mu: Tensor, x_mask: Tensor, y: Tensor, y_mask: Tensor) -> tuple[Tensor, Tensor]:
    """MAS between the prior ``mu`` [B, C, N] and the target mel ``y`` [B, C, T].

    Returns the hard alignment ``[B, N, T]`` and the target log-durations ``[B, 1, N]``.
    """
    attn_mask = x_mask.transpose(1, 2) * y_mask  # [B, N, T]
    with torch.no_grad():
        attn = maximum_path(gaussian_log_prior(mu.float(), y.float()), attn_mask)
    logw_target = torch.log(1e-8 + attn.sum(-1, keepdim=True)).transpose(1, 2) * x_mask
    return attn, logw_target


def prior_loss(mu_y: Tensor, y: Tensor, y_mask: Tensor) -> Tensor:
    """Gaussian NLL of the target under the aligned prior (per element)."""
    nll = 0.5 * ((y - mu_y) ** 2 + math.log(2 * math.pi)) * y_mask
    return nll.sum() / (y_mask.sum() * y.shape[1])


def duration_loss(logw: Tensor, logw_target: Tensor, x_mask: Tensor) -> Tensor:
    return ((logw - logw_target) ** 2 * x_mask).sum() / x_mask.sum()


def token_pitch(f0: Tensor, attn: Tensor, lf0_mean: float | Tensor, lf0_std: float | Tensor) -> Tensor:
    """Average normalised log-F0 of the voiced frames of every token (FastPitch); 0 for unvoiced tokens.

    Args:
        f0: ``[B, T]`` F0 in Hz (0 = unvoiced). attn: ``[B, N, T]`` hard alignment.
        lf0_mean / lf0_std: log-F0 statistics (floats or 0-dim tensors on the device of ``f0``).
    Returns:
        ``[B, 1, N]`` token pitch targets.
    """
    voiced = (f0 > 0).float()
    lf0 = torch.where(f0 > 0, (torch.log(f0.clamp_min(1.0)) - lf0_mean) / lf0_std, torch.zeros_like(f0))
    num = torch.bmm(attn, (lf0 * voiced)[:, :, None])[:, :, 0]
    den = torch.bmm(attn, voiced[:, :, None])[:, :, 0]
    return torch.where(den > 0, num / den.clamp_min(1.0), torch.zeros_like(num))[:, None]
