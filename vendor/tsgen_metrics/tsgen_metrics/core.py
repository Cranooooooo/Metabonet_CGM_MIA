"""tsgen_metrics — evaluate the quality of generated time series against real data.

Public API:

    from tsgen_metrics import evaluate
    result = evaluate(real, fake)            # real/fake: float32 (N, T, C)
    print(result["metrics"]["context_fid"]["value"])

Computes 10 metrics (all LOWER = better). Provenance per metric — see PROVENANCE:
  * Reference-validated (our code reproduces the published reference on identical
    input, verified by validate_against_reference.py):
        correlational  -> Diffusion-TS Utils/cross_correlation.py (CrossCorrelLoss)
        vds            -> PaD-TS eval_utils/MMD.py (VDS_Naive)
        fdds           -> PaD-TS eval_utils/MMD.py (BMMD_Naive o cross_correlation_distribution)
  * Own implementation, no single canonical published reference:
        mdd (Windowed MMD), acd, skewness_diff, kurtosis_diff
  * Proxy (trained nets; stochastic and NOT faithful to the official TS2Vec/TF-GRU
    implementations — use them for ranking within this code, not to match official
    numbers; for official values run the DiMTS/PaD-TS reference directly):
        context_fid, discriminative, predictive

Inputs are arrays shaped (N, T, C) = (windows, timesteps, channels). real and fake
must already be in the SAME value space (apply the same normalization to both
before calling — metrics compare them directly).
"""
from __future__ import annotations

import time
from typing import Any, Dict, Optional

import numpy as np

from . import statistical, neural

# Statistical metrics evaluated over multiple seeds (cheap); mean -> value, +std.
_STAT_KEYS = ["correlational", "mdd", "acd", "skewness_diff", "kurtosis_diff",
              "vds", "fdds"]

# Per-metric: short name (abbrev), full name, provenance status + reference.
#   reference-validated : reproduces a published reference exactly on identical input
#                         (verified by validate_against_reference.py).
#   reference-aligned   : uses the official implementation (vendored); faithful up to
#                         that implementation's own training stochasticity.
#   recipe-matched      : official hyper-parameter recipe, but a stochastic trained
#                         net with TF<->PyTorch differences -> tracks ranking, not bit-exact.
#   own                 : self-contained implementation; no single canonical reference.
PROVENANCE: Dict[str, Dict[str, str]] = {
    "correlational":  {"abbrev": "CC",   "name": "Cross-Correlation Difference",
                       "status": "reference-validated", "reference": "Diffusion-TS CrossCorrelLoss"},
    "mdd":            {"abbrev": "MDD",  "name": "Windowed MMD",
                       "status": "own", "reference": "RBF-MMD^2 over flattened windows (median-heuristic bandwidth)"},
    "acd":            {"abbrev": "ACD",  "name": "AutoCorrelation Difference",
                       "status": "own", "reference": "L2 between mean autocorrelation curves"},
    "skewness_diff":  {"abbrev": "SD",   "name": "Skewness Difference",
                       "status": "own", "reference": "per-channel skewness gap"},
    "kurtosis_diff":  {"abbrev": "KD",   "name": "Kurtosis Difference",
                       "status": "own", "reference": "per-channel kurtosis gap"},
    "vds":            {"abbrev": "VDS",  "name": "Value Distribution Shift",
                       "status": "reference-validated", "reference": "PaD-TS VDS_Naive (Li et al. 2025)"},
    "fdds":           {"abbrev": "FDDS", "name": "Functional Dependency Distribution Shift",
                       "status": "reference-validated", "reference": "PaD-TS BMMD_Naive(cross_correlation_distribution) (Li et al. 2025)"},
    "context_fid":    {"abbrev": "Context-FID", "name": "Context-FID",
                       "status": "reference-aligned", "reference": "DiM-TS Context_FID (vendored official TS2Vec)"},
    "discriminative": {"abbrev": "DS",   "name": "Discriminative Score",
                       "status": "recipe-matched", "reference": "DiM-TS/TimeGAN GRU (hidden=C//2, iters=2000, batch=128)"},
    "predictive":     {"abbrev": "PS",   "name": "Predictive Score (TSTR)",
                       "status": "recipe-matched", "reference": "DiM-TS/TimeGAN GRU (hidden=C//2, iters=5000, batch=128)"},
}


def _check(x: np.ndarray, name: str) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 3:
        raise ValueError(f"{name} must be 3-D (N, T, C); got shape {x.shape}")
    return x


def _subsample_pair(real, fake, n, seed):
    """Equal-count uniform subsample of real and fake. Draws each side at the
    final matched size m directly (no sort-then-truncate bias)."""
    rng = np.random.default_rng(seed)
    m = min(n, real.shape[0], fake.shape[0])
    r_idx = rng.choice(real.shape[0], size=m, replace=False)
    f_idx = rng.choice(fake.shape[0], size=m, replace=False)
    return (np.asarray(real[np.sort(r_idx)], dtype=np.float32),
            np.asarray(fake[np.sort(f_idx)], dtype=np.float32))


def _statistical_suite(real_sub, fake_sub, seed):
    metrics, per_channel = {}, {}
    m = statistical.correlational_score(real_sub, fake_sub)
    per_channel["correlational"] = m.pop("per_channel", None); metrics["correlational"] = m
    metrics["mdd"] = statistical.mmd_rbf(real_sub, fake_sub, seed=seed)  # MDD = Windowed MMD (whole-window RBF-MMD^2)
    m = statistical.autocorrelation_difference(real_sub, fake_sub)
    per_channel["acd"] = m.pop("per_channel", None); metrics["acd"] = m
    m = statistical.skewness_difference(real_sub, fake_sub)
    per_channel["skewness_diff"] = m.pop("per_channel", None); metrics["skewness_diff"] = m
    m = statistical.kurtosis_difference(real_sub, fake_sub)
    per_channel["kurtosis_diff"] = m.pop("per_channel", None); metrics["kurtosis_diff"] = m
    m = statistical.value_distribution_shift(real_sub, fake_sub, seed=seed)
    per_channel["vds"] = m.pop("per_channel", None); metrics["vds"] = m
    m = statistical.functional_dependency_distribution_shift(real_sub, fake_sub, seed=seed)
    per_channel["fdds"] = m.pop("per_pair", None); metrics["fdds"] = m
    return metrics, per_channel


def evaluate(real: np.ndarray,
             fake: np.ndarray,
             *,
             subsample: Optional[int] = 2000,
             seed: int = 0,
             n_seeds: int = 3,
             epochs: int = 40,
             device: Optional[str] = None,
             feature_names: Optional[list] = None,
             include_neural: bool = True,
             verbose: bool = False) -> Dict[str, Any]:
    """Evaluate generated ``fake`` against ``real`` (both (N, T, C), same space).

    Args:
        subsample: equal per-side window count (None = use all, capped at min N).
        seed: base RNG seed. Statistical metrics are averaged over ``n_seeds``
            independent subsamples (mean -> value, plus std). Neural metrics use
            ``seed`` only (single run; they are stochastic — see PROVENANCE).
        n_seeds: number of subsample seeds for the statistical metrics.
        epochs: training epochs for the proxy neural metrics.
        device: 'cuda' / 'cpu' (default: cuda if available).
        include_neural: set False to skip the (proxy) neural metrics entirely.
        feature_names: optional channel names recorded in the output.

    Returns a dict: {metrics: {<name>: {value, lower_is_better, status, reference,
    ...}}, per_channel, window, channels, ...}. All metrics are lower = better.
    """
    import torch
    real = _check(real, "real")
    fake = _check(fake, "fake")
    if real.shape[1:] != fake.shape[1:]:
        raise ValueError(f"real (T,C)={real.shape[1:]} != fake (T,C)={fake.shape[1:]}")
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    n = subsample if subsample else max(real.shape[0], fake.shape[0])

    t0 = time.time()
    real_sub, fake_sub = _subsample_pair(real, fake, n, seed)
    if verbose:
        print(f"[tsgen] real={real_sub.shape} fake={fake_sub.shape} device={dev}")

    metrics: Dict[str, Any] = {}

    # statistical (multi-seed): mean -> value, plus std / n_seeds
    base_metrics, per_channel = _statistical_suite(real_sub, fake_sub, seed)
    stat_values = {k: [base_metrics[k]["value"]] for k in _STAT_KEYS}
    for s in range(seed + 1, seed + max(1, n_seeds)):
        rs, fs = _subsample_pair(real, fake, n, s)
        extra, _ = _statistical_suite(rs, fs, s)
        for k in _STAT_KEYS:
            stat_values[k].append(extra[k]["value"])
    for k in _STAT_KEYS:
        m = base_metrics[k]
        vals = np.asarray(stat_values[k], dtype=np.float64)
        m["value"] = float(vals.mean()); m["std"] = float(vals.std()); m["n_seeds"] = int(vals.size)
        metrics[k] = m
    if verbose:
        print(f"[tsgen] statistical done ({time.time()-t0:.1f}s)")

    # neural (GPU) — optional. These use their own faithful/official defaults
    # (TS2Vec for Context-FID; official GRU iterations for DS/PS), so `epochs`
    # is intentionally not forwarded.
    if include_neural:
        metrics["context_fid"] = neural.context_fid(real_sub, fake_sub, str(dev), seed=seed)
        metrics["discriminative"] = neural.discriminative_score(real_sub, fake_sub, str(dev), seed=seed)
        metrics["predictive"] = neural.predictive_score(real_sub, fake_sub, str(dev), seed=seed)
        if verbose:
            print(f"[tsgen] neural done ({time.time()-t0:.1f}s)")

    # attach abbrev / full name / provenance to every metric
    for key, m in metrics.items():
        prov = PROVENANCE.get(key, {})
        m["abbrev"] = prov.get("abbrev", key)
        m["name"] = prov.get("name", key)
        m["status"] = prov.get("status", "own")
        m["reference"] = prov.get("reference", "none")

    return {
        "metrics": metrics,
        "per_channel": {k: v for k, v in per_channel.items() if v is not None},
        "window": int(real_sub.shape[1]),
        "channels": int(real_sub.shape[2]),
        "feature_names": list(feature_names) if feature_names else None,
        "subsample": {"real": int(real_sub.shape[0]), "fake": int(fake_sub.shape[0])},
        "seed": seed, "n_seeds_statistical": int(max(1, n_seeds)),
        "epochs": epochs, "include_neural": include_neural,
        "elapsed_sec": round(time.time() - t0, 1),
    }
