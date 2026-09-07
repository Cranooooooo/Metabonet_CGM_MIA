"""Cross-correlation score (no missingness).

Multi-channel coupling: lag-0 Pearson correlation across channel pairs,
computed via the lower-triangular cross-correlation utility (cacf_torch).
The score is the mean absolute deviation between the C×C cross-correlation
matrices of real and synthetic data, divided by 10 for scaling parity with
the original eval_ref_1 protocol.

Inputs : real, fake of shape (N, T, C).
Returns: scalar score; lower is better.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn


def cacf_torch(x: torch.Tensor, max_lag: int, dim: tuple = (0, 1)) -> torch.Tensor:
    """Lower-triangular cross-correlation across channels at lags 0..max_lag-1."""
    def _tril(n):
        return [list(t) for t in torch.tril_indices(n, n)]

    ind = _tril(x.shape[2])
    x = (x - x.mean(dim, keepdims=True)) / x.std(dim, keepdims=True)
    x_l = x[..., ind[0]]
    x_r = x[..., ind[1]]
    cacf_list = []
    for i in range(max_lag):
        y = x_l[:, i:] * x_r[:, :-i] if i > 0 else x_l * x_r
        cacf_list.append(torch.mean(y, (1)))
    cacf = torch.cat(cacf_list, 1)
    return cacf.reshape(cacf.shape[0], -1, len(ind[0]))


class _CrossCorrelLoss(nn.Module):
    def __init__(self, x_real: torch.Tensor, max_lag: int = 1):
        super().__init__()
        self.max_lag = max_lag
        self.cross_correl_real = cacf_torch(x_real, max_lag).mean(0)[0]

    def compute(self, x_fake: torch.Tensor) -> torch.Tensor:
        cross_correl_fake = cacf_torch(x_fake, self.max_lag).mean(0)[0]
        loss = torch.abs(cross_correl_fake - self.cross_correl_real.to(x_fake.device)).sum(0)
        return loss / 10.0


def cross_correlation_score(ori_data: np.ndarray, generated_data: np.ndarray,
                            max_lag: int = 1) -> float:
    """Mean absolute deviation between real/fake C×C cross-correlation matrices."""
    x_real = torch.from_numpy(np.asarray(ori_data)).float()
    x_fake = torch.from_numpy(np.asarray(generated_data)).float()
    loss = _CrossCorrelLoss(x_real, max_lag=max_lag)
    return float(loss.compute(x_fake).mean().item())
