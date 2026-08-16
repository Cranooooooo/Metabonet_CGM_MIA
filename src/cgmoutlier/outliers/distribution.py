"""Groups C and E -- the subject's whole distribution of days (C8, C9, C10, E13, E14).

These are the only methods that never collapse a subject to one point. A subject whose
average day is ordinary but whose spread of days is not is invisible to every
mean-aggregated method and visible here. In the full design space this route is about
2% of the options, and it is the only one that can see a single unusual day -- which
is what memorisation attaches to.

E13 and E14 are controls, not candidates:
  E13  day count. If the outlier lists track it, the study is measuring data volume.
  E14  C8 with the subject's own windows removed from the reference. C8 scores every
       subject against a pool that contains it, so a prolific subject drags the
       reference towards itself. The gap between them is the size of that bias.
"""
from __future__ import annotations

import numpy as np

from .common import SEED, method_rng, pdist_block, subject_slices


def _median_bandwidth(REF, n=1500):
    d = pdist_block(REF[:n], REF) / np.sqrt(REF.shape[1])
    return float(np.median(d[d > 0]))


def _mmd2(A, B, sigma):
    def km(P, Q):
        d = pdist_block(P, Q) / np.sqrt(P.shape[1])
        return float(np.exp(-(d ** 2) / (2 * sigma * sigma)).mean())
    return km(A, A) + km(B, B) - 2 * km(A, B)


def mmd_vs_cohort(Z, sids, n_ref=4000, key="C8", seed=SEED, leave_self_out=False):
    """RBF MMD^2 between each subject's windows and a fixed cohort reference.

    The bandwidth is the median heuristic computed once on the reference and shared:
    a per-subject bandwidth would rescale the statistic per subject.
    """
    subs, sl = subject_slices(sids)
    r = method_rng(key, seed)
    ix = np.sort(r.choice(Z.shape[0], min(n_ref, Z.shape[0]), replace=False))
    REF, rsid = Z[ix], sids[ix]
    sigma = _median_bandwidth(REF)
    out = np.empty(subs.size)
    for i, s in enumerate(subs):
        ref = REF[rsid != s] if leave_self_out else REF
        out[i] = _mmd2(Z[sl[s]], ref, sigma)
    return out, dict(sigma=sigma, n_ref=int(REF.shape[0]), leave_self_out=leave_self_out)


def sliced_wasserstein(flat, sids, n_proj=200, n_ref=4000, n_q=256, key="C10",
                       seed=SEED):
    """Mean W2 over random 1-D projections.

    MMD's sensitivity is capped by its kernel bandwidth and saturates once two
    distributions are far apart; Wasserstein keeps moving mass. Where the two
    disagree, the difference is in the tail.
    """
    subs, sl = subject_slices(sids)
    r = method_rng(key, seed)
    ix = np.sort(r.choice(flat.shape[0], min(n_ref, flat.shape[0]), replace=False))
    P = r.standard_normal((flat.shape[1], n_proj)).astype(np.float32)
    P /= np.linalg.norm(P, axis=0, keepdims=True)
    q = np.linspace(0, 1, n_q)
    refq = np.quantile(flat[ix] @ P, q, axis=0)          # (n_q, n_proj)
    out = np.empty(subs.size)
    for i, s in enumerate(subs):
        sq = np.quantile(flat[sl[s]] @ P, q, axis=0)
        out[i] = float(np.mean((sq - refq) ** 2))
    return out


def E13_day_count(sids):
    subs, sl = subject_slices(sids)
    return np.array([sl[s].size for s in subs], float)
