"""phi(x, S) similarity features for the MIA pipeline.

For each real chunk x (T, C) and a synthetic set S (K, T, C), compute a small
vector of similarity statistics that summarise how close x is to the synthetic
set produced by a generator. The default ("cheap") dims are fully vectorised and
return in seconds on the smoke shapes:

    euc_min   -- min Euclidean distance to S      (L2 normalised by sqrt(T*C))
    euc_avg   -- mean Euclidean distance to S      (same normalisation)
    euc_ball  -- fraction of S within Euclidean radius r_euc
    cos_ball  -- fraction of S with cosine similarity above cos_c0

The radius r_euc is calibrated dataset-wide as the ``r_euc_q``-th quantile of all
real-vs-synthetic Euclidean distances (not per-real), matching the legacy phi.

Optional elastic dims (dtw_min, dtw_ball, frechet_min) live behind
``cheap_only=False`` and are computed by :mod:`m7mia.mia.slow_features` -- banded
DTW / discrete Frechet over the Euclidean top-M candidates. They were NaN stubs
until 2026-08-04; see that module for what the band and the candidate cut do and
do not guarantee.

Everything flattens the (T, C) chunk to a (T*C,) vector, so the code is fully
dataset-agnostic (any T, C).
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

CHEAP_NAMES: List[str] = ["euc_min", "euc_avg", "euc_ball", "cos_ball"]
SLOW_NAMES: List[str] = ["dtw_min", "dtw_ball", "frechet_min"]


def _euc_dist_matrix(reals_flat: np.ndarray, syns_flat: np.ndarray) -> np.ndarray:
    """reals (N, F), syns (K, F) -> (N, K) Euclidean distance, float32."""
    # ||x - s||^2 = ||x||^2 + ||s||^2 - 2 x.s
    r2 = (reals_flat ** 2).sum(1, keepdims=True)        # (N, 1)
    s2 = (syns_flat ** 2).sum(1, keepdims=True).T       # (1, K)
    cross = reals_flat @ syns_flat.T                    # (N, K)
    d2 = np.clip(r2 + s2 - 2.0 * cross, 0.0, None)
    return np.sqrt(d2).astype(np.float32)


def _cos_sim_matrix(reals_flat: np.ndarray, syns_flat: np.ndarray) -> np.ndarray:
    """reals (N, F), syns (K, F) -> (N, K) cosine similarity, float32."""
    rn = reals_flat / (np.linalg.norm(reals_flat, axis=1, keepdims=True) + 1e-12)
    sn = syns_flat / (np.linalg.norm(syns_flat, axis=1, keepdims=True) + 1e-12)
    return (rn @ sn.T).astype(np.float32)


def calibrate_r_euc(
    reals: np.ndarray,
    syns: np.ndarray,
    *,
    r_euc_q: float = 0.05,
) -> float:
    """Euclidean ball radius = ``r_euc_q``-th quantile of all real-vs-syn dists.

    This is the SAME normalised Euclidean distance used inside
    :func:`compute_phi_matrix` (L2 divided by ``sqrt(T*C)``). Calibrate the radius
    ONCE on a fixed reference (e.g. all reals vs S_all) and pass the result as
    ``r_euc=`` to every :func:`compute_phi_matrix` call so the ``euc_ball`` column
    uses one membership-independent threshold for label-1 and label-0 rows alike.
    """
    reals = np.asarray(reals, dtype=np.float32)
    syns = np.asarray(syns, dtype=np.float32)
    N = reals.shape[0]
    K = syns.shape[0]
    scale = np.sqrt(np.float32(int(np.prod(reals.shape[1:]))))
    Deuc = _euc_dist_matrix(reals.reshape(N, -1), syns.reshape(K, -1)) / scale
    return float(np.quantile(Deuc, r_euc_q))


def compute_phi_matrix(
    reals: np.ndarray,
    syns: np.ndarray,
    *,
    cheap_only: bool = True,
    r_euc_q: float = 0.05,
    r_euc: float | None = None,
    cos_c0: float = 0.95,
    dtw_band: int = 12,
    dtw_cand: int = 32,
    r_dtw: float | None = None,
    learned: dict | None = None,
) -> Tuple[np.ndarray, List[str]]:
    """Compute phi for many real chunks against one synthetic set.

    Args:
        reals: (N, T, C) real chunks.
        syns:  (K, T, C) synthetic chunks (same T, C as reals).
        cheap_only: if True (default) return only the 4 vectorised dims; if
            False, append the slow dims (dtw_min, dtw_ball, frechet_min) as
            NaN-filled columns (heavy exact distances are stubbed here).
        r_euc_q: Euclidean ball radius = this quantile of all pairwise
            real-vs-synthetic Euclidean distances, used ONLY as a fallback when
            ``r_euc`` is not given.
        r_euc: explicit Euclidean ball radius. Pass a single value calibrated
            once on a fixed reference (see :func:`calibrate_r_euc`) so the
            ``euc_ball`` column uses the same membership-independent threshold for
            every call; otherwise each call re-derives its own quantile from its
            own (reals-vs-syns) distance matrix, which makes ``euc_ball`` a
            condition-dependent confound (different threshold for member vs
            non-member rows).
        cos_c0: cosine-ball similarity threshold.

    Returns:
        phi: (N, D) float32 feature matrix.
        names: list of D column names.
    """
    reals = np.asarray(reals, dtype=np.float32)
    syns = np.asarray(syns, dtype=np.float32)
    if reals.ndim != 3 or syns.ndim != 3:
        raise ValueError(f"reals/syns must be 3-D (N,T,C); got {reals.shape}, {syns.shape}")
    if reals.shape[1:] != syns.shape[1:]:
        raise ValueError(f"reals (T,C)={reals.shape[1:]} != syns (T,C)={syns.shape[1:]}")

    N, T, C = reals.shape
    K = syns.shape[0]
    rf = reals.reshape(N, T * C)
    sf = syns.reshape(K, T * C)

    # --- cheap dims: Euclidean + cosine via full (N, K) matrices ---
    scale = np.sqrt(np.float32(T * C))
    Deuc = _euc_dist_matrix(rf, sf) / scale     # (N, K)
    Dcos = _cos_sim_matrix(rf, sf)              # (N, K)

    euc_min = Deuc.min(axis=1)
    euc_avg = Deuc.mean(axis=1)
    # Use the explicitly-passed radius when given so the SAME threshold is shared
    # across member/non-member calls; only fall back to this call's own quantile
    # when no fixed radius is supplied.
    r_euc_val = float(r_euc) if r_euc is not None else float(np.quantile(Deuc, r_euc_q))
    euc_ball = (Deuc < r_euc_val).mean(axis=1).astype(np.float32)
    cos_ball = (Dcos > cos_c0).mean(axis=1).astype(np.float32)

    names = list(CHEAP_NAMES)
    cols = [euc_min, euc_avg, euc_ball, cos_ball]

    if not cheap_only:
        # Elastic dims. Deuc is handed over rather than recomputed: it is already the
        # cheap pass's output and it also picks the candidate set, so both feature
        # families rest on one distance basis.
        from .slow_features import compute_slow_phi
        slow, slow_names, _ = compute_slow_phi(
            reals, syns, Deuc=Deuc, band=dtw_band, n_cand=dtw_cand, r_dtw=r_dtw)
        names = names + list(slow_names)
        cols = cols + [slow[:, 0], slow[:, 1], slow[:, 2]]

    if learned is not None:
        # Frozen TS2Vec embeddings, passed in already computed: the encoder is fit
        # once per run, not once per phi call, or every fold would get its own space.
        from .learned_features import compute_learned_phi
        lp, lnames, _ = compute_learned_phi(learned["real_emb"], learned["syn_emb"],
                                            r_emb=learned.get("r_emb"), cos_c0=cos_c0)
        names = names + list(lnames)
        cols = cols + [lp[:, 0], lp[:, 1], lp[:, 2], lp[:, 3]]

    phi = np.stack(cols, axis=1).astype(np.float32)
    return phi, names


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    for (T, C) in [(288, 6), (1024, 12)]:
        reals = rng.standard_normal((20, T, C)).astype(np.float32)
        syns = rng.standard_normal((50, T, C)).astype(np.float32)
        phi, names = compute_phi_matrix(reals, syns)
        print(f"T={T} C={C}: phi={phi.shape} {phi.dtype} names={names}")
        phi2, names2 = compute_phi_matrix(reals, syns, cheap_only=False)
        print(f"  slow: phi={phi2.shape} names={names2}")
