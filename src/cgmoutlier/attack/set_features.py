"""Set-to-set relationship features between a subject's windows and a generator's.

The framing
-----------
Fix one encoder. Every window -- real or synthetic -- becomes a point. A subject is
then a cloud A of n points (n ~ 183 here), a generator's released set is a cloud B of
K points (K = 2000), and the membership question is: does B look like it was fit on a
set containing A? So the design problem is not "which distance between two windows"
but "which relationship between two clouds".

That reframing subsumes a bug the previous code had. phi used to be computed in two
separate collapses -- first along B (min / mean / ball), then along A (min for
euc_min, mean for the rest, driven by a `min_dims` list in configs/mia.yaml). Adding
dtw_min and frechet_min without adding them to `min_dims` silently averaged two
nearest-neighbour statistics over a subject's 183 windows, which destroys exactly the
tail they exist to capture. Written as one set-level function, that class of mistake
has nowhere to live.

The hypothesis being tested
---------------------------
Membership leakage is LOCAL: one memorised window out of 183 is enough to identify a
subject, and it leaves the other 182 untouched. So statistics sensitive to the tail of
the cross-distance distribution (single linkage, Chamfer, low quantiles of the
per-point nearest-neighbour distance) should beat statistics of central tendency
(average linkage, centroid distance, spread ratio). The existing evidence is
consistent with this -- euc_min alone was the best feature set of eight, and adding
mean-type dims made it worse -- but it has never been tested against measures built
for the purpose.

Calibration
-----------
Every radius and bandwidth is passed in, never derived per call. A per-call quantile
would give member and non-member rows different thresholds, which is a condition-
dependent confound rather than a feature. See `calibrate` for how they are fixed once
per run over all folds.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

# grouped so an RQ level can select a family rather than naming columns
FAMILIES: Dict[str, List[str]] = {
    # tail-sensitive: what a single memorised window moves
    "linkage": ["min_min", "chamfer_ab", "chamfer_ba", "hausdorff_ab",
                "nn_q01", "nn_q05", "nn_q25"],
    "knn":     ["knn1", "knn5", "knn20", "knn1_ba"],
    # density / support overlap
    "coverage": ["ball_r", "precision", "recall", "density", "coverage"],
    # central tendency / whole-distribution
    "dist":    ["energy", "mmd2", "centroid", "spread_ratio", "cos_max", "cos_ball"],
}
ALL_NAMES: List[str] = [n for fam in FAMILIES.values() for n in fam]


def _cross(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """(n,d),(K,d) -> (n,K) Euclidean distance, normalised by sqrt(d)."""
    a2 = (A ** 2).sum(1, keepdims=True)
    b2 = (B ** 2).sum(1, keepdims=True).T
    d2 = np.clip(a2 + b2 - 2.0 * (A @ B.T), 0.0, None)
    return np.sqrt(d2, dtype=np.float64) / np.sqrt(A.shape[1])


def _self(A: np.ndarray) -> np.ndarray:
    D = _cross(A, A)
    np.fill_diagonal(D, np.inf)          # exclude self-pairs for kNN radii
    return D


def _knn_radius(D_self: np.ndarray, k: int) -> np.ndarray:
    """k-th nearest within-cloud distance per point (Kynkaanniemi support radius)."""
    k = min(k, D_self.shape[1] - 1)
    return np.partition(D_self, k - 1, axis=1)[:, k - 1]


def set_features(A: np.ndarray, B: np.ndarray, *, cal: dict) -> Dict[str, float]:
    """One vector describing how cloud A sits relative to cloud B.

    `cal` carries the fixed thresholds: r_ball, sigma (MMD bandwidth), cos_c0, and
    radii_B (B's own kNN radii, computed once per fold since B is shared).
    """
    A = np.ascontiguousarray(A, dtype=np.float64)
    B = np.ascontiguousarray(B, dtype=np.float64)
    D = _cross(A, B)                              # (n, K)
    nn_a = D.min(axis=1)                          # each subject window -> nearest synth
    nn_b = D.min(axis=0)                          # each synth -> nearest subject window
    out: Dict[str, float] = {}

    # --- tail-sensitive -------------------------------------------------------
    out["min_min"] = float(nn_a.min())                     # single linkage
    out["chamfer_ab"] = float(nn_a.mean())                 # mean per-point NN, A->B
    out["chamfer_ba"] = float(nn_b.mean())                 # ... and B->A
    out["hausdorff_ab"] = float(nn_a.max())
    out["nn_q01"] = float(np.quantile(nn_a, 0.01))
    out["nn_q05"] = float(np.quantile(nn_a, 0.05))
    out["nn_q25"] = float(np.quantile(nn_a, 0.25))

    for k in (1, 5, 20):
        kk = min(k, D.shape[1])
        out[f"knn{k}"] = float(np.partition(D, kk - 1, axis=1)[:, kk - 1].mean())
    out["knn1_ba"] = float(nn_b.mean())

    # --- density / support overlap -------------------------------------------
    out["ball_r"] = float((D < cal["r_ball"]).mean())
    rb = cal["radii_B"]                                    # (K,) B's own kNN radii
    ra = _knn_radius(_self(A), cal.get("k_support", 5)) if A.shape[0] > 2 \
        else np.full(A.shape[0], np.inf)
    # precision: synthetic points that fall inside A's support
    out["precision"] = float((D.min(axis=0) <= ra.max()).mean()) if np.isfinite(ra).any() else 0.0
    # recall: subject windows that fall inside B's support
    out["recall"] = float((D <= rb[None, :]).any(axis=1).mean())
    # density / coverage (Naeem et al.): how many B-balls a subject window sits in,
    # and what fraction of subject windows have any B point closer than their own radius
    out["density"] = float((D <= rb[None, :]).sum(axis=1).mean() / max(1, cal.get("k_support", 5)))
    out["coverage"] = float((nn_a <= ra).mean()) if np.isfinite(ra).any() else 0.0

    # --- central tendency / distributional ------------------------------------
    Daa = _cross(A, A); Dbb_mean = cal["Dbb_mean"]
    out["energy"] = float(2.0 * D.mean() - Daa.mean() - Dbb_mean)
    s = cal["sigma"]
    kab = np.exp(-(D ** 2) / (2 * s * s)).mean()
    kaa = np.exp(-(Daa ** 2) / (2 * s * s)).mean()
    out["mmd2"] = float(kaa + cal["kbb_mean"] - 2.0 * kab)
    out["centroid"] = float(np.linalg.norm(A.mean(0) - B.mean(0)) / np.sqrt(A.shape[1]))
    va = float(A.var(0).sum()); vb = cal["var_B"]
    out["spread_ratio"] = float(vb / va) if va > 0 else 0.0

    An = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    Bn = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
    C = An @ Bn.T
    out["cos_max"] = float(C.max())
    out["cos_ball"] = float((C > cal["cos_c0"]).mean())
    return out


def calibrate_fold(B: np.ndarray, *, sigma: float, k_support: int = 5) -> dict:
    """Per-fold constants that depend only on B, so every subject shares them.

    `sigma` comes from the global calibration and is required here because the MMD
    kernel term over B alone is a per-fold constant: computing it per subject would
    repeat a 2000x2000 kernel 402 times for the same answer.
    """
    B = np.ascontiguousarray(B, dtype=np.float64)
    Dbb = _self(B)
    radii = _knn_radius(Dbb, k_support)
    finite = np.isfinite(Dbb)
    kbb = float(np.exp(-(Dbb[finite] ** 2) / (2 * sigma * sigma)).mean())
    return dict(radii_B=radii, Dbb_mean=float(Dbb[finite].mean()),
                kbb_mean=kbb, var_B=float(B.var(0).sum()), k_support=k_support)


def calibrate_global(cross_samples: np.ndarray, r_q: float = 0.05,
                     cos_c0: float = 0.95) -> dict:
    """Thresholds shared by EVERY fold and every subject.

    r_ball is the r_q quantile of pooled real-vs-synthetic distances and sigma is the
    median of the same pool (the standard median heuristic). Both are computed once
    on the clean (pre-injection) sets so injecting memorisation cannot move the
    threshold that is supposed to detect it.
    """
    v = np.asarray(cross_samples, dtype=np.float64).ravel()
    return dict(r_ball=float(np.quantile(v, r_q)),
                sigma=float(np.median(v)) or 1.0,
                cos_c0=float(cos_c0))
