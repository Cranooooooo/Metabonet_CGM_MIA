"""Statistical (non-neural) generation-quality metrics.

All operate on float32 arrays shaped ``(N, T, C)`` (windows, time, channels).
Each function returns a dict with ``value`` / ``lower_is_better`` /
``definition`` / ``config`` and, where it makes sense, a ``per_channel`` list so
the frontend can show multi-channel breakdowns.

References:
- Correlational, MMD : Diffusion-TS (Utils/cross_correlation.py) + arXiv:2511.18312
- ACD, SD, KD, VDS, FDDS : arXiv:2511.18312 (DiM-TS) appendix metric list

Score directions (all lower-is-better here) are stated per metric.
"""
from __future__ import annotations

import numpy as np
from scipy import stats as _stats
from scipy.stats import wasserstein_distance


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _flat_channels(x: np.ndarray) -> np.ndarray:
    """(N, T, C) -> (N*T, C): pool all timesteps/windows per channel."""
    n, t, c = x.shape
    return np.asarray(x, dtype=np.float64).reshape(n * t, c)


def _corr_matrix(x: np.ndarray) -> np.ndarray:
    """Cross-channel Pearson correlation matrix (C, C) over pooled samples."""
    flat = _flat_channels(x)  # (N*T, C)
    # np.corrcoef expects variables in rows.
    cm = np.corrcoef(flat, rowvar=False)
    return np.nan_to_num(cm, nan=0.0)


# --------------------------------------------------------------------------- #
# 1. Correlational score
# --------------------------------------------------------------------------- #
def correlational_score(real: np.ndarray, fake: np.ndarray) -> dict:
    """Cross-correlation score — faithful port of Diffusion-TS ``CrossCorrelLoss``.

    Replicates ``Utils/cross_correlation.py`` (the published reference): standardize
    each channel over pooled (window, time) with unbiased std, take the lag-0
    cross-products for the LOWER triangle (incl. diagonal) per window, average over
    windows, then report ``sum|cc_real - cc_fake| / 10`` (the reference's fixed
    normalization). Diagonal terms are identically 1 on both sides so they cancel,
    so this equals the reference value. Validated to match
    ``CrossCorrelLoss(real).compute(fake).mean()``. Lower is better.
    """
    def _cacf0(x):
        x = np.asarray(x, dtype=np.float64)
        mu = x.mean(axis=(0, 1), keepdims=True)
        sd = x.std(axis=(0, 1), ddof=1, keepdims=True) + 1e-12  # unbiased, matches torch.std
        xs = (x - mu) / sd
        c = xs.shape[2]
        rows, cols = np.tril_indices(c)               # lower triangle incl. diagonal
        prod = xs[:, :, rows] * xs[:, :, cols]        # (N, T, n_pairs)
        return prod.mean(axis=1).mean(axis=0), rows, cols   # (n_pairs,)

    cr, rows, cols = _cacf0(real)
    cf, _, _ = _cacf0(fake)
    diff = np.abs(cf - cr)
    value = float(diff.sum() / 10.0)
    c = int(real.shape[2])
    offmask = rows != cols
    per_channel = []
    for i in range(c):
        sel = offmask & ((rows == i) | (cols == i))
        per_channel.append(float(diff[sel].mean()) if sel.any() else 0.0)
    return {
        "value": value,
        "lower_is_better": True,
        "definition": "Cross-correlation score: faithful port of Diffusion-TS CrossCorrelLoss — sum|lag-0 cross-correlation(real) - cross-correlation(fake)| / 10 over lower-triangle channel pairs.",
        "config": {"channels": c, "aggregation": "crosscorrelloss_sum_l1_div10",
                   "reference": "Diffusion-TS Utils/cross_correlation.py"},
        "per_channel": per_channel,
    }


# --------------------------------------------------------------------------- #
# 5. MMD (RBF kernel)
# --------------------------------------------------------------------------- #
def _rbf_mmd2(x: np.ndarray, y: np.ndarray, gamma: float) -> float:
    """Unbiased-ish RBF MMD^2 between row-sets x (n,d) and y (m,d)."""
    def _sq_dists(a, b):
        a2 = (a * a).sum(1)[:, None]
        b2 = (b * b).sum(1)[None, :]
        return np.maximum(a2 + b2 - 2.0 * a @ b.T, 0.0)

    kxx = np.exp(-gamma * _sq_dists(x, x))
    kyy = np.exp(-gamma * _sq_dists(y, y))
    kxy = np.exp(-gamma * _sq_dists(x, y))
    return float(kxx.mean() + kyy.mean() - 2.0 * kxy.mean())


def mmd_rbf(real: np.ndarray, fake: np.ndarray, max_n: int = 1000,
            seed: int = 0) -> dict:
    """Windowed MMD (metric key ``mdd``): RBF-kernel Maximum Mean Discrepancy
    between real and generated, computed over whole flattened windows.

    Each window is flattened to a (T*C,) vector; we subsample up to ``max_n``
    windows per side, pick the RBF bandwidth via the median-heuristic on the
    pooled pairwise distances, and report MMD^2. Lower is better.
    """
    rng = np.random.default_rng(seed)
    r = np.asarray(real, dtype=np.float64)
    f = np.asarray(fake, dtype=np.float64)
    r = r.reshape(r.shape[0], -1)
    f = f.reshape(f.shape[0], -1)
    if r.shape[0] > max_n:
        r = r[rng.choice(r.shape[0], max_n, replace=False)]
    if f.shape[0] > max_n:
        f = f[rng.choice(f.shape[0], max_n, replace=False)]
    # median heuristic for bandwidth on a pooled subset
    pool = np.vstack([r, f])
    sub = pool[rng.choice(pool.shape[0], min(500, pool.shape[0]), replace=False)]
    d2 = ((sub[:, None, :] - sub[None, :, :]) ** 2).sum(-1)
    med = np.median(d2[d2 > 0])
    gamma = 1.0 / (med + 1e-12)
    value = _rbf_mmd2(r, f, gamma)
    return {
        "value": value,
        "lower_is_better": True,
        "definition": "RBF-kernel MMD^2 between flattened windows; bandwidth from the median heuristic.",
        "config": {"max_n": max_n, "kernel": "rbf", "gamma": float(gamma),
                   "feature": "flattened_window", "seed": seed},
    }


# --------------------------------------------------------------------------- #
# 6. ACD - Auto-Correlation Difference
# --------------------------------------------------------------------------- #
def _mean_autocorr(x: np.ndarray, max_lag: int) -> np.ndarray:
    """Mean (over windows & channels) autocorrelation curve, lags 0..max_lag-1.

    Returns shape (max_lag,) plus per-channel curves are computed by the caller.
    Per-window per-channel ACF, averaged over windows.
    """
    n, t, c = x.shape
    x = np.asarray(x, dtype=np.float64)
    xm = x - x.mean(axis=1, keepdims=True)
    var = (xm ** 2).mean(axis=1, keepdims=True) + 1e-12  # (n,1,c)
    lags = min(max_lag, t)
    # acf[k] = mean_t( xm[t]*xm[t+k] ) / var
    acf = np.zeros((n, lags, c))
    for k in range(lags):
        prod = xm[:, : t - k, :] * xm[:, k:, :]
        acf[:, k, :] = prod.mean(axis=1) / var[:, 0, :]
    return acf  # (n, lags, c)


def autocorrelation_difference(real: np.ndarray, fake: np.ndarray,
                               max_lag: int = 64) -> dict:
    """L2 (Frobenius) distance between mean autocorrelation curves (real vs generated).

    We compute per-window per-channel ACF (lags 0..max_lag-1), average over
    windows, and report the Frobenius norm of the (real-fake) difference over the
    full (lag, channel) curve as the headline value, with per-channel L2 in
    ``per_channel``. Channels are aggregated AFTER differencing (the headline then
    equals the L2 norm of the per-channel vector), so opposite-sign per-channel
    errors cannot cancel. Lower is better.
    """
    acf_r = _mean_autocorr(real, max_lag)  # (n, L, c)
    acf_f = _mean_autocorr(fake, max_lag)
    L = min(acf_r.shape[1], acf_f.shape[1])
    mr = acf_r[:, :L, :].mean(axis=0)  # (L, c)
    mf = acf_f[:, :L, :].mean(axis=0)
    per_channel = [float(np.sqrt(((mr[:, c] - mf[:, c]) ** 2).sum()))
                   for c in range(mr.shape[1])]
    # Frobenius over the full (lag, channel) difference == L2 of per_channel.
    value = float(np.sqrt(((mr - mf) ** 2).sum()))
    return {
        "value": value,
        "lower_is_better": True,
        "definition": "L2 (Frobenius) distance between per-channel mean autocorrelation curves (real vs generated), aggregated after differencing.",
        "config": {"max_lag": int(L), "aggregation": "frobenius_over_lag_channel"},
        "per_channel": per_channel,
    }


# --------------------------------------------------------------------------- #
# 7/8. Skewness / Kurtosis difference
# --------------------------------------------------------------------------- #
def skewness_difference(real: np.ndarray, fake: np.ndarray) -> dict:
    """Per-channel |skew(real) - skew(fake)|, aggregated by mean. Lower better.

    SD = | E[(x_r-mu_r)^3]/sigma_r^3 - E[(x_f-mu_f)^3]/sigma_f^3 | per channel
    (arXiv:2511.18312). Headline value is the mean across channels.
    """
    fr = _flat_channels(real)
    ff = _flat_channels(fake)
    c = fr.shape[1]
    per_channel = []
    for i in range(c):
        sr = float(_stats.skew(fr[:, i]))
        sf = float(_stats.skew(ff[:, i]))
        per_channel.append(abs(sr - sf))
    return {
        "value": float(np.mean(per_channel)),
        "lower_is_better": True,
        "definition": "Mean over channels of |skewness(real) - skewness(generated)| (pooled values).",
        "config": {"channels": c, "aggregation": "mean_over_channels"},
        "per_channel": per_channel,
    }


def kurtosis_difference(real: np.ndarray, fake: np.ndarray) -> dict:
    """Per-channel |kurtosis(real) - kurtosis(fake)|, aggregated by mean.

    KD = | E[(x_r-mu_r)^4]/sigma_r^4 - ... | per channel (arXiv:2511.18312).
    We use Fisher=False (so a normal -> 3) consistent with the raw 4th moment
    definition. Lower is better.
    """
    fr = _flat_channels(real)
    ff = _flat_channels(fake)
    c = fr.shape[1]
    per_channel = []
    for i in range(c):
        kr = float(_stats.kurtosis(fr[:, i], fisher=False))
        kf = float(_stats.kurtosis(ff[:, i], fisher=False))
        per_channel.append(abs(kr - kf))
    return {
        "value": float(np.mean(per_channel)),
        "lower_is_better": True,
        "definition": "Mean over channels of |kurtosis(real) - kurtosis(generated)| (4th moment, Fisher=False).",
        "config": {"channels": c, "aggregation": "mean_over_channels"},
        "per_channel": per_channel,
    }


# --------------------------------------------------------------------------- #
# Reference RBF-MMD (faithful port of PaD-TS eval_utils/MMD.py) — used by VDS/FDDS
# so OUR values reproduce the published reference (validated). Biased, multi-
# bandwidth RBF; same bandwidth set as PaD-TS MMD().
# --------------------------------------------------------------------------- #
_REF_MMD_BANDWIDTHS = [0.01, 0.05, 0.1, 0.2, 0.5, 0.7, 1.0, 1.5, 2.0]


def _ref_mmd_rbf(x, y):
    """PaD-TS MMD(x, y, 'rbf'): x, y are torch (n, dim). Biased multi-bandwidth."""
    import torch
    xx, yy, zz = x @ x.t(), y @ y.t(), x @ y.t()
    rx = xx.diag().unsqueeze(0).expand_as(xx)
    ry = yy.diag().unsqueeze(0).expand_as(yy)
    dxx = rx.t() + rx - 2.0 * xx
    dyy = ry.t() + ry - 2.0 * yy
    dxy = rx.t() + ry - 2.0 * zz
    XX = torch.zeros_like(xx)
    YY = torch.zeros_like(xx)
    XY = torch.zeros_like(xx)
    for a in _REF_MMD_BANDWIDTHS:
        XX += torch.exp(-0.5 * dxx / a)
        YY += torch.exp(-0.5 * dyy / a)
        XY += torch.exp(-0.5 * dxy / a)
    return float(torch.mean(XX + YY - 2.0 * XY))


def _ref_cross_corr_dist(x):
    """PaD-TS cross_correlation_distribution: per-window lower-triangle (no diag)
    inter-channel Pearson correlations. x torch (N,T,C) -> (N, n_pairs)."""
    import torch
    n, t, c = x.shape
    tri = torch.tril_indices(c, c)
    off = tri[0] != tri[1]
    rows, cols = tri[0][off], tri[1][off]
    out = []
    for i in range(n):
        cm = torch.corrcoef(x[i].t())          # (C, C)
        out.append(cm[rows, cols])
    out = torch.stack(out, dim=0)
    return torch.where(torch.isinf(out) | torch.isnan(out), torch.zeros_like(out), out)


# --------------------------------------------------------------------------- #
# 9. VDS - Value Distribution Shift  (faithful port of PaD-TS VDS_Naive)
# --------------------------------------------------------------------------- #
def value_distribution_shift(real: np.ndarray, fake: np.ndarray,
                             n_sample: int = 10000, seed: int = 0) -> dict:
    """Value Distribution Shift — faithful port of PaD-TS ``VDS_Naive`` (RBF-MMD).

    Per channel, draw ``n_sample`` random indices into the flattened (N*T) values
    (the SAME indices for real and fake, as in the reference) and report the
    multi-bandwidth RBF-MMD between the two value sets; average over channels.
    Validated to match ``VDS_Naive(real, fake, 'rbf').mean()`` (within the 10000-
    sample stochasticity). Lower is better.
    """
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    xr = torch.as_tensor(np.asarray(real, dtype=np.float32))
    xf = torch.as_tensor(np.asarray(fake, dtype=np.float32))
    n, t, c = xr.shape
    rng = np.random.default_rng(seed)
    per_channel = []
    for i in range(c):
        idx = rng.integers(0, n * t, n_sample)
        a = xr[:, :, i].reshape(-1)[idx].unsqueeze(-1).to(dev)
        b = xf[:, :, i].reshape(-1)[idx].unsqueeze(-1).to(dev)
        per_channel.append(_ref_mmd_rbf(a, b))
    return {
        "value": float(np.mean(per_channel)),
        "lower_is_better": True,
        "definition": "Value Distribution Shift: mean over channels of the multi-bandwidth RBF-MMD between real vs generated per-channel value distributions (faithful port of PaD-TS VDS_Naive).",
        "config": {"channels": int(c), "distance": "rbf_mmd_multibandwidth",
                   "n_sample": n_sample, "seed": seed,
                   "reference": "PaD-TS eval_utils/MMD.py VDS_Naive"},
        "per_channel": per_channel,
    }


# --------------------------------------------------------------------------- #
# 10. FDDS - Functional Dependency Distribution Shift
# --------------------------------------------------------------------------- #
def functional_dependency_distribution_shift(real: np.ndarray, fake: np.ndarray,
                                              seed: int = 0) -> dict:
    """Functional Dependency Distribution Shift — faithful port of PaD-TS FDDS.

    Reference path: ``cross_correlation_distribution`` gives, per window, the
    lower-triangle (no diagonal) inter-channel Pearson correlations; ``BMMD_Naive``
    then takes, per channel-pair, the multi-bandwidth RBF-MMD between the real and
    generated distributions of that pair's per-window correlation, averaged over
    pairs. Validated to match ``BMMD_Naive(cc_real, cc_fake, 'rbf').mean()``.
    Lower is better. Single-channel data has no pairs -> 0.0.
    """
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    c = int(real.shape[2])
    if c < 2:
        return {
            "value": 0.0, "lower_is_better": True,
            "definition": "Functional Dependency Distribution Shift (no channel pairs for single-channel data).",
            "config": {"pairs": 0, "distance": "rbf_mmd_multibandwidth"},
            "per_pair": [],
        }
    cr = _ref_cross_corr_dist(torch.as_tensor(np.asarray(real, dtype=np.float32)))  # (N, n_pairs)
    cf = _ref_cross_corr_dist(torch.as_tensor(np.asarray(fake, dtype=np.float32)))
    tri = torch.tril_indices(c, c)
    off = tri[0] != tri[1]
    rows, cols = tri[0][off].tolist(), tri[1][off].tolist()
    per_pair, vals = [], []
    for p in range(cr.shape[1]):
        a = cr[:, p].unsqueeze(-1).to(dev)
        b = cf[:, p].unsqueeze(-1).to(dev)
        d = _ref_mmd_rbf(a, b)
        per_pair.append({"pair": [int(rows[p]), int(cols[p])], "value": d})
        vals.append(d)
    return {
        "value": float(np.mean(vals)),
        "lower_is_better": True,
        "definition": "Functional Dependency Distribution Shift: mean over channel pairs of the multi-bandwidth RBF-MMD between real vs generated distributions of per-window inter-channel Pearson correlations (faithful port of PaD-TS BMMD_Naive over cross_correlation_distribution).",
        "config": {"pairs": len(vals), "distance": "rbf_mmd_multibandwidth", "seed": seed,
                   "reference": "PaD-TS eval_utils/MMD.py BMMD_Naive(cross_correlation_distribution)"},
        "per_pair": per_pair,
    }
