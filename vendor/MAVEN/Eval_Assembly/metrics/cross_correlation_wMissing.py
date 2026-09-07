"""Cross-correlation score (with missingness — real-only mask).

Generated samples are assumed to be fully observed. The C×C Pearson
correlation matrix of real is computed pairwise-complete (only timesteps
where both features in each pair are observed), while fake's matrix is
computed straight on the fully-observed signal. Score is the mean absolute
deviation across valid off-diagonal pairs (overlap >= min_overlap on real),
scaled by /10 to match the no-missing protocol.

Falls back to the no-missing path when ori_mask is fully observed.

Inputs : real, fake of shape (N, T, C); ori_mask of shape (N, T, C).
Returns: scalar score; lower is better.
"""
from __future__ import annotations

import numpy as np

from utils.mask_utils import is_fully_observed, pairwise_complete_corr
from metrics.cross_correlation_woMissing import cross_correlation_score as _cc_clean


def cross_correlation_score_masked(ori_data: np.ndarray, generated_data: np.ndarray,
                                   ori_mask: np.ndarray | None = None,
                                   min_overlap: int = 30,
                                   max_lag: int = 1) -> float:
    if is_fully_observed(ori_mask):
        return _cc_clean(ori_data, generated_data, max_lag=max_lag)

    M_real = np.asarray(ori_mask).astype(bool)
    M_fake = np.ones(np.asarray(generated_data).shape, dtype=bool)

    C_real, _ = pairwise_complete_corr(np.asarray(ori_data), M_real, min_overlap)
    C_fake, _ = pairwise_complete_corr(np.asarray(generated_data), M_fake, min_overlap)

    valid = (~np.isnan(C_real)) & (~np.isnan(C_fake))
    np.fill_diagonal(valid, False)
    if not valid.any():
        return float("nan")
    diff = np.abs(C_real - C_fake)[valid]
    return float(diff.sum() / 10.0)
