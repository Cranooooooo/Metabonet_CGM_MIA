"""ACF loss (no missingness).

Per-channel multi-lag autocorrelation: for each channel, compute the
sample-mean autocorrelation at lags 1..max_lag-1, then take the L2 norm of
the (real - fake) difference vector. Reports per-channel array and channel-
mean. Adapted from eval_ref_2/02_fidelity/metrics/feature_distance_eval.py.

Inputs : real, fake of shape (N, T, C).
Returns: dict with 'per_channel' (list of length C) and 'mean' (float).
"""
from __future__ import annotations

import numpy as np
import torch

from utils.metric_utils import reduce_per_channel


def _acf_torch(x: torch.Tensor, max_lag: int) -> torch.Tensor:
    """Per-(N, T, C) ACF at lags 0..max_lag-1, averaged over (N, T) → (max_lag, C)."""
    x = x - x.mean((0, 1))
    var = torch.var(x, unbiased=False, dim=(0, 1))
    out = []
    for i in range(max_lag):
        y = x[:, i:] * x[:, :-i] if i > 0 else x.pow(2)
        out.append(torch.mean(y, (0, 1)) / (var + 1e-12))
    return torch.stack(out)  # (max_lag, C)


def acf_loss(ori_data: np.ndarray, generated_data: np.ndarray,
             max_lag: int = 64) -> dict:
    x_real = torch.from_numpy(np.asarray(ori_data)).float()
    x_fake = torch.from_numpy(np.asarray(generated_data)).float()
    L = max(1, min(max_lag, x_real.shape[1]))
    a_real = _acf_torch(x_real, L)
    a_fake = _acf_torch(x_fake, L)
    diff = (a_real - a_fake).pow(2).sum(dim=0).sqrt().cpu().numpy()  # per-channel L2
    return reduce_per_channel(diff)
