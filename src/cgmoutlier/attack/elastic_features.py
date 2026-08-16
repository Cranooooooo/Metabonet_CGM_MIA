"""Elastic-distance phi dims: dtw_min, dtw_ball, frechet_min.

Why these exist
---------------
The cheap dims are all Euclidean or cosine, which compare timestep t of a real
window against timestep t of a synthetic one. CGM traces carry the same events at
shifted times -- the same post-meal excursion half an hour earlier reads as a large
Euclidean distance and a small elastic one. If the attack is losing signal because
of that, no choice of classifier over the cheap dims can recover it, so the feature
axis has to be tested before the classifier axis means anything.

Why it is affordable
--------------------
Exact DTW over every (real, synthetic) pair is O(N*K*T^2): 73k x 2k x 288^2 ~ 1.2e13
cell updates. Two standard reductions make it tractable without changing what the
min-type features measure:

  * Sakoe-Chiba band of half-width w. Warpings wider than w are excluded, which is
    the usual constraint for periodic daily signals anyway (a meal does not shift by
    six hours). Cost per pair drops to O(T*(2w+1)).
  * Euclidean pre-filter. Take only the M nearest synthetic windows by Euclidean
    distance and run DTW on those. Banded DTW >= a band-limited alignment cost and
    correlates strongly with Euclidean at these shapes, so the true DTW-nearest
    neighbour is essentially always inside a generous Euclidean top-M. This is a
    heuristic, not a bound -- `dtw_min` is therefore an upper bound on exact
    dtw_min. Recorded in the output metadata rather than glossed over.

`dtw_ball` uses the SAME candidate set, so it is the fraction of the top-M within
the radius, not of all K. It is comparable across cells only because M and the
radius are fixed for a run.

Frechet is the discrete Frechet distance under the same band and candidate set: a
max-over-couplings rather than a sum, so it reacts to a single worst-aligned point
where DTW averages it away.
"""
from __future__ import annotations

import os
from typing import List, Tuple

import numpy as np
from numba import njit, prange, set_num_threads as _numba_threads

# Numba's parallel backend sizes its pool from NUMBA_NUM_THREADS, which defaults to
# the machine's core count and ignores OMP_NUM_THREADS entirely. On this 128-core
# shared box that meant 138 threads per worker even with OMP_NUM_THREADS=3 set --
# twelve workers took 50 cores instead of the intended ~18 and pushed the load
# average past 270. Cap it here, in the module that owns the parallel kernel, so it
# applies however the process was launched.
_NUMBA_THREADS = int(os.environ.get("M7MIA_NUMBA_THREADS", "2"))
try:
    _numba_threads(_NUMBA_THREADS)
except Exception:                      # older numba, or already inside a parallel region
    pass

SLOW_NAMES: List[str] = ["dtw_min", "dtw_ball", "frechet_min"]


@njit(cache=True, fastmath=True)
def _banded_dtw(a, b, w):
    """Sakoe-Chiba banded DTW between (T, C) a and b; returns cost / T."""
    T, C = a.shape
    INF = np.inf
    prev = np.full(T + 1, INF, dtype=np.float64)
    cur = np.full(T + 1, INF, dtype=np.float64)
    prev[0] = 0.0
    for i in range(1, T + 1):
        lo = max(1, i - w)
        hi = min(T, i + w)
        for j in range(T + 1):
            cur[j] = INF
        if lo == 1:
            cur[0] = INF
        for j in range(lo, hi + 1):
            d = 0.0
            for c in range(C):
                diff = a[i - 1, c] - b[j - 1, c]
                d += diff * diff
            d = np.sqrt(d)
            best = prev[j]
            if prev[j - 1] < best:
                best = prev[j - 1]
            if cur[j - 1] < best:
                best = cur[j - 1]
            cur[j] = d + best
        for j in range(T + 1):
            prev[j] = cur[j]
    return prev[T] / T


@njit(cache=True, fastmath=True)
def _banded_frechet(a, b, w):
    """Discrete Frechet distance under the same band; max-over-couplings."""
    T, C = a.shape
    INF = np.inf
    prev = np.full(T, INF, dtype=np.float64)
    cur = np.full(T, INF, dtype=np.float64)
    for i in range(T):
        lo = max(0, i - w)
        hi = min(T - 1, i + w)
        for j in range(T):
            cur[j] = INF
        for j in range(lo, hi + 1):
            d = 0.0
            for c in range(C):
                diff = a[i, c] - b[j, c]
                d += diff * diff
            d = np.sqrt(d)
            if i == 0 and j == 0:
                cur[j] = d
                continue
            best = INF
            if i > 0 and prev[j] < best:
                best = prev[j]
            if j > 0 and cur[j - 1] < best:
                best = cur[j - 1]
            if i > 0 and j > 0 and prev[j - 1] < best:
                best = prev[j - 1]
            cur[j] = d if d > best else best
        for j in range(T):
            prev[j] = cur[j]
    return prev[T - 1]


@njit(parallel=True, cache=True, fastmath=True)
def _sweep(reals, syns, cand, w):
    """For each real i, min banded-DTW / min Frechet over its candidate set."""
    N = reals.shape[0]
    M = cand.shape[1]
    dtw_min = np.empty(N, dtype=np.float64)
    dtw_all = np.empty((N, M), dtype=np.float64)
    fre_min = np.empty(N, dtype=np.float64)
    for i in prange(N):
        best_d = np.inf
        best_f = np.inf
        for m in range(M):
            k = cand[i, m]
            d = _banded_dtw(reals[i], syns[k], w)
            dtw_all[i, m] = d
            if d < best_d:
                best_d = d
            f = _banded_frechet(reals[i], syns[k], w)
            if f < best_f:
                best_f = f
        dtw_min[i] = best_d
        fre_min[i] = best_f
    return dtw_min, dtw_all, fre_min


def compute_slow_phi(reals: np.ndarray, syns: np.ndarray, *, Deuc: np.ndarray,
                     band: int = 12, n_cand: int = 32,
                     r_dtw: float | None = None,
                     r_dtw_q: float = 0.05) -> Tuple[np.ndarray, List[str], dict]:
    """(N,T,C) reals, (K,T,C) syns, precomputed (N,K) Euclidean -> (N,3) phi.

    `Deuc` is reused from the cheap pass rather than recomputed; it is also what
    selects the candidate set, so the two feature families stay on one basis.
    `r_dtw` must be passed from a fixed calibration for the same reason `r_euc` is:
    a per-call quantile would make the ball radius condition-dependent.
    """
    reals = np.ascontiguousarray(reals, dtype=np.float64)
    syns = np.ascontiguousarray(syns, dtype=np.float64)
    N = reals.shape[0]
    M = min(int(n_cand), syns.shape[0])
    cand = np.argpartition(Deuc, M - 1, axis=1)[:, :M].astype(np.int64)
    cand = np.ascontiguousarray(cand)

    dtw_min, dtw_all, fre_min = _sweep(reals, syns, cand, int(band))
    r = float(r_dtw) if r_dtw is not None else float(np.quantile(dtw_all, r_dtw_q))
    dtw_ball = (dtw_all < r).mean(axis=1)

    phi = np.stack([dtw_min, dtw_ball, fre_min], axis=1).astype(np.float32)
    meta = dict(band=int(band), n_cand=M, r_dtw=r,
                note=("dtw_min/frechet_min are minima over the Euclidean top-M, so they "
                      "upper-bound the exact minima; dtw_ball is a fraction of M, not of K"))
    return phi, list(SLOW_NAMES), meta
