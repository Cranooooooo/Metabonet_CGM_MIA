"""Group B -- curve shape in raw space (B5, B6, B7a, B7b).

B5/B6 follow Hall et al., PLOS Biology 2018 ("glucotypes"): complexity-invariant DTW
between daily curves, then clustering, which is the shape-based route validated on
CGM specifically. B7 is the fit-free k-NN baseline the anomaly-detection benchmark
literature insists on, run twice -- mean and max over a subject's days -- because that
one switch is the difference between "an unusual person" and "a person with one
unusual day", and it is the stage nobody reports.

COST, AND WHAT IT COSTS THE RESULT
Exact CID-DTW over every pair of the 182k windows is 3.3e10 alignments. B5/B6 run on
a per-subject cap of `dtw_per_subject` days and cluster on a further subsample. That
makes them runnable and it makes them noisier, most so for subjects with few days.
"""
from __future__ import annotations

import numpy as np

from .common import SEED, kmedoids, method_rng, pdist_block, subject_slices


def _as3d(Z):
    """(n, T) or (n, T, C) -> (n, T, C), contiguous float64 for the numba kernel."""
    Z = np.ascontiguousarray(Z, np.float64)
    return Z[:, :, None] if Z.ndim == 2 else Z


def _ce(Z):
    """Complexity estimate: the arc length of the series (Batista et al.).

    Over all channels at once, so a day is one path through C-dimensional space rather
    than C separate curves — the same object `_banded_dtw` aligns.
    """
    d = np.diff(_as3d(Z), axis=1) ** 2
    return np.sqrt(d.sum(axis=(1, 2))) + 1e-9


def cid_dtw(A, B, band=12, threads=4):
    """Complexity-invariant banded DTW between every row of A and every row of B.

    CID = DTW x max(ce_a, ce_b)/min(ce_a, ce_b). Without the correction, a flat day
    and a volatile day at similar levels come out similar, which is the opposite of
    what a glycaemic-variability phenotype needs.

    Accepts (n, T) or (n, T, C). `_banded_dtw`'s signature was always (T, C) — the
    single-channel path reshaped to (T, 1) to meet it — so multichannel costs nothing
    but keeping the axis. Note that the channels must already be on a common scale,
    which the cohort's per-channel z-score provides: the kernel's per-step cost is a
    Euclidean norm across channels, so an unscaled channel would dominate the alignment.
    """
    from numba import njit, prange, set_num_threads
    set_num_threads(threads)
    from ..attack.elastic_features import _banded_dtw

    A, B = _as3d(A), _as3d(B)
    # _banded_dtw is @njit with no bounds checking: it takes T from `a` alone and reads
    # b[j-1, c] for j up to T, so a shorter or narrower B is read past its buffer and
    # returns a finite garbage cost with no exception. This function feeds B5/B6, which
    # vote in the consensus that defines the study's targets -- a silently wrong distance
    # here changes who is an outlier.
    if A.shape[1:] != B.shape[1:]:
        raise ValueError(
            f"cid_dtw needs matching (T, C): A is {A.shape[1:]}, B is {B.shape[1:]}. "
            f"numba would read past the end of B rather than raise.")

    @njit(parallel=True, cache=True, fastmath=True)
    def _run(A_, B_, band_, cA, cB):
        n, m = A_.shape[0], B_.shape[0]
        out = np.empty((n, m), np.float64)
        for i in prange(n):
            ai = A_[i].copy()
            for j in range(m):
                d = _banded_dtw(ai, B_[j].copy(), band_)
                hi = cA[i] if cA[i] > cB[j] else cB[j]
                lo = cB[j] if cA[i] > cB[j] else cA[i]
                out[i, j] = d * (hi / (lo + 1e-9))
        return out

    return _run(A, B, band, _ce(A), _ce(B))


def glucotype_medoids(W, n_sig=3, n_probe=2200, key="B5_B6.probe", seed=SEED,
                      verbose=True):
    """Three glycaemic signatures, as medoid days.

    Spectral clustering with a self-tuning (k-th neighbour) affinity, per
    Zelnik-Manor & Perona -- a single global bandwidth on CID-DTW distances put 2131
    of 2200 windows in one cluster. If it still degenerates, fall back to k-medoids,
    which cannot. Which route ran is returned, not swallowed.
    """
    from sklearn.cluster import SpectralClustering
    r = method_rng(key, seed)
    pix = np.sort(r.choice(W.shape[0], min(n_probe, W.shape[0]), replace=False))
    P = W[pix]
    D = cid_dtw(P, P)
    sig = np.sort(D, axis=1)[:, 7] + 1e-9
    aff = np.exp(-(D ** 2) / (sig[:, None] * sig[None, :])); np.fill_diagonal(aff, 1.0)
    lab = SpectralClustering(n_clusters=n_sig, affinity="precomputed", random_state=0,
                             assign_labels="kmeans").fit_predict(aff)
    route = "spectral"
    if min(np.bincount(lab, minlength=n_sig)) < 0.05 * lab.size:
        lab = kmedoids(D, n_sig); route = "kmedoids"
    med = [pix[m[np.argmin(D[np.ix_(m, m)].sum(1))]]
           for m in (np.flatnonzero(lab == c) for c in range(n_sig))]
    sizes = np.bincount(lab, minlength=n_sig).tolist()
    if verbose:
        print(f"  glucotypes via {route}, cluster sizes {sizes}", flush=True)
    return np.array(med), dict(route=route, sizes=sizes, n_probe=int(P.shape[0]))


def B5_B6(cur, sids, dtw_per_subject=30, key="B5_B6", seed=SEED, verbose=True):
    """-> (B5 scores, B6 scores, share vectors, meta). Computed together because they
    share the medoids and the DTW pass, which is 95% of the cost.

    `cur` is (n, T) or (n, T, C); the channel axis is kept rather than flattened,
    because DTW aligns a path in C dimensions and flattening would instead concatenate
    C separate days end to end and align that.
    """
    subs, sl = subject_slices(sids)
    r = method_rng(key, seed)
    take = np.sort(np.concatenate([
        sl[s] if sl[s].size <= dtw_per_subject
        else r.choice(sl[s], dtw_per_subject, replace=False) for s in subs]))
    cur = _as3d(cur)
    W, wsid = cur[take], sids[take]
    if verbose:
        print(f"  B5/B6: {W.shape[0]} windows x {W.shape[2]} channels after a "
              f"per-subject cap of {dtw_per_subject}", flush=True)
    med, meta = glucotype_medoids(W, key=key + ".probe", seed=seed, verbose=verbose)
    # `med` indexes W, not cur: glucotype_medoids draws its probe with
    # `r.choice(W.shape[0], ...)` and returns `pix[...]`, so every value is < len(W).
    # This read `cur[med]` until 2026-08-22, which on the published run meant three
    # windows taken from the first 26,250 rows of a 182,597-row array ordered by
    # (id, day) -- i.e. three days belonging to whichever low-id subjects happened to
    # sit at those offsets, not the glucotype medoids. No IndexError, no shape mismatch,
    # and B5/B6 came out plausible: a positive mean distance and a share vector summing
    # to 1. Two of the thirteen consensus votes were scored against arbitrary windows.
    Dwm = cid_dtw(W, W[med])
    dmin, nearest = Dwm.min(1), Dwm.argmin(1)
    b5 = np.array([dmin[wsid == s].mean() for s in subs])
    share = np.stack([np.bincount(nearest[wsid == s], minlength=3) / max(1, (wsid == s).sum())
                      for s in subs])
    b6 = np.linalg.norm(share - share.mean(0), axis=1)
    meta["dtw_per_subject"] = dtw_per_subject
    return b5, b6, share, meta


def B7(flat, sids, n_ref=6000, k=5, key="B7", seed=SEED):
    """-> (mean, max) of the per-day k-th nearest-neighbour distance to OTHER subjects.

    Excluding the subject's own windows from its own reference is the whole point: a
    subject with 700 days is otherwise its own nearest neighbour.
    """
    subs, sl = subject_slices(sids)
    r = method_rng(key, seed)
    ix = np.sort(r.choice(flat.shape[0], min(n_ref, flat.shape[0]), replace=False))
    REF, rsid = flat[ix], sids[ix]
    D = pdist_block(flat, REF) / np.sqrt(flat.shape[1])
    mean_s, max_s = np.empty(subs.size), np.empty(subs.size)
    for i, s in enumerate(subs):
        d = D[sl[s]].copy()
        d[:, rsid == s] = np.inf
        kk = min(k, d.shape[1] - 1)
        kth = np.partition(d, kk - 1, axis=1)[:, kk - 1]
        mean_s[i], max_s[i] = kth.mean(), kth.max()
    return mean_s, max_s
