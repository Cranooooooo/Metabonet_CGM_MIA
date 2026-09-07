"""Shared mask helpers for the with-missing eval pipeline.

Convention: M is bool (or 0/1) array, shape matches X exactly.  M=1 = observed,
M=0 = missing. Missing positions in X are zero-filled (or whatever the data
producer wrote).

In this suite, only `real_mask` is a user-facing concept — generated samples
are assumed to be fully observed at every (n, t, c). When a metric needs to
present fake under the same missingness regime as real (context_fid,
discriminative, predictive), it draws a synthetic fake-mask by resampling
patterns from the real_mask pool.
"""
from __future__ import annotations

import numpy as np


def is_fully_observed(M: np.ndarray | None) -> bool:
    """True if M is None, all True, or all 1.  Used to short-circuit to the
    non-missing implementation so the two pipelines agree exactly when there
    is no missingness."""
    if M is None:
        return True
    return bool(np.asarray(M).astype(bool).all())


def sanitize(X: np.ndarray, M: np.ndarray | None) -> np.ndarray:
    """Replace any NaN/Inf in X with 0; force missing positions to 0."""
    X = np.asarray(X, dtype=np.float32)
    if M is None:
        return np.where(np.isfinite(X), X, 0.0).astype(np.float32)
    M = np.asarray(M).astype(bool)
    out = np.where(np.isfinite(X), X, 0.0).astype(np.float32)
    out = np.where(M, out, 0.0).astype(np.float32)
    return out


def resample_masks(target_n: int, mask_pool: np.ndarray,
                   rng: np.random.Generator | None = None) -> np.ndarray:
    """Sample target_n masks from a pool (with replacement).

    Used to project the real_mask distribution onto fake samples for metrics
    that need both sides to share the same missingness regime (context_fid,
    discriminative, predictive).

    mask_pool : (P, T, C) bool — pool of mask patterns to draw from.
    Returns   : (target_n, T, C) bool.
    """
    rng = rng or np.random.default_rng()
    pool = np.asarray(mask_pool).astype(bool)
    idx = rng.integers(0, pool.shape[0], size=target_n)
    return pool[idx]


def pairwise_complete_corr(X: np.ndarray, M: np.ndarray,
                           min_overlap: int = 30) -> tuple[np.ndarray, np.ndarray]:
    """Cross-correlation of features using pairwise-complete observations.

    Concatenates all samples along the time axis, then for each feature pair
    (i, j) computes Pearson correlation only over timesteps where M_t^(i) and
    M_t^(j) are both 1.

    Returns
    -------
    C_mat : (C, C) — Pearson correlation per pair (NaN where overlap < min_overlap)
    n_mat : (C, C) — count of common observations per pair
    """
    X = np.asarray(X, dtype=np.float64)
    M = np.asarray(M).astype(bool)
    Xf = X.reshape(-1, X.shape[-1])
    Mf = M.reshape(-1, M.shape[-1])
    Xf = np.where(Mf, Xf, 0.0)

    C = Xf.shape[1]
    C_mat = np.full((C, C), np.nan, dtype=np.float64)
    n_mat = np.zeros((C, C), dtype=np.int64)

    for i in range(C):
        for j in range(i, C):
            both = Mf[:, i] & Mf[:, j]
            n = int(both.sum())
            n_mat[i, j] = n
            n_mat[j, i] = n
            if n < min_overlap:
                continue
            xi = Xf[both, i]
            xj = Xf[both, j]
            mi = xi.mean(); mj = xj.mean()
            xi_c = xi - mi; xj_c = xj - mj
            denom = np.sqrt((xi_c ** 2).sum() * (xj_c ** 2).sum())
            if denom < 1e-12:
                continue
            r = float((xi_c * xj_c).sum() / denom)
            C_mat[i, j] = r
            C_mat[j, i] = r
    return C_mat, n_mat


def per_sample_observed_indices(M: np.ndarray) -> list[list[np.ndarray]]:
    """For peak / IPI metrics: per-sample, per-channel list of contiguous
    observed-segment index arrays. Each entry is a 1-D int array of timestep
    indices that form a maximal run of observed samples."""
    M = np.asarray(M).astype(bool)
    N, T, C = M.shape
    out: list[list[np.ndarray]] = []
    for n in range(N):
        per_c = []
        for c in range(C):
            m = M[n, :, c]
            runs: list[np.ndarray] = []
            i = 0
            while i < T:
                if not m[i]:
                    i += 1; continue
                j = i
                while j < T and m[j]:
                    j += 1
                runs.append(np.arange(i, j))
                i = j
            per_c.append(runs)
        out.append(per_c)
    return out
