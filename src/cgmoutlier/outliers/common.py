"""Shared pieces for the outlier methods.

One rule the whole package follows: every reference pool, radius and kernel bandwidth
is drawn or fitted ONCE and shared by every subject. A per-subject reference would
give each subject its own yardstick, which is the difference between "this subject is
unusual" and "this subject was measured differently".
"""
from __future__ import annotations

import hashlib

import numpy as np

SEED = 2026


def rng(seed: int = SEED):
    return np.random.default_rng(seed)


def method_rng(key: str, seed: int = SEED):
    """An independent random stream per method, derived from the method's own name.

    The previous version of this code held ONE module-level generator that every
    method drew from in turn. A method's reference pool therefore depended on which
    other methods had run before it in the same process: `--only C8` and
    `--only C9,C8` produce different C8 scores, and neither reproduces a full run.
    Seeding from the key makes each draw a function of (key, seed) alone, so rerunning
    one method reproduces it exactly whatever else the run contains.

    Sampling noise does not disappear, it only becomes reproducible. B5/B6 still see
    30 days per subject rather than all of them, and a different `seed` is a different
    equally legitimate answer. Which subjects survive across seeds is measurable
    (scripts/seed_stability.py) and is a property worth reporting rather than hiding.
    """
    h = int.from_bytes(hashlib.blake2b(key.encode(), digest_size=8).digest(), "big")
    return np.random.default_rng([seed, h])


def subject_slices(sids: np.ndarray):
    """-> (unique ids, {id: row indices}). Ids are strings, not integers."""
    order = np.argsort(sids, kind="stable")
    s = sids[order]
    uniq, start = np.unique(s, return_index=True)
    end = np.r_[start[1:], s.size]
    return uniq, {u: order[a:b] for u, a, b in zip(uniq, start, end)}


def pdist_block(A: np.ndarray, B: np.ndarray, chunk: int = 4096) -> np.ndarray:
    """(nA,d),(nB,d) -> (nA,nB) Euclidean, blocked. The naive broadcast is 182k x 8k x
    288 floats and does not fit."""
    b2 = (B ** 2).sum(1)[None, :]
    out = np.empty((A.shape[0], B.shape[0]), np.float32)
    for i in range(0, A.shape[0], chunk):
        a = A[i:i + chunk]
        d2 = (a ** 2).sum(1)[:, None] + b2 - 2.0 * (a @ B.T)
        out[i:i + chunk] = np.sqrt(np.clip(d2, 0, None))
    return out


def robust_scale(df):
    """Median / IQR rather than mean / sd: the outliers we are looking for would
    otherwise set the scale they are measured against."""
    return (df - df.median()) / (df.quantile(.75) - df.quantile(.25) + 1e-12)


def robust_mahalanobis(df) -> np.ndarray:
    from sklearn.covariance import MinCovDet
    S = robust_scale(df).to_numpy()
    return MinCovDet(random_state=0, support_fraction=0.9).fit(S).mahalanobis(S)


def kmedoids(D: np.ndarray, k: int, iters: int = 60, seed: int = 0) -> np.ndarray:
    """PAM-style k-medoids on a precomputed dissimilarity.

    Here because sklearn has no k-medoids and sklearn_extra is a dependency this does
    not need. Used as the fallback when spectral clustering degenerates -- on this
    data a single global affinity bandwidth put 97% of windows in one cluster.
    """
    r = np.random.default_rng(seed)
    n = D.shape[0]
    med = [int(r.integers(n))]
    for _ in range(k - 1):
        d = D[med].min(0) ** 2
        med.append(int(r.choice(n, p=d / d.sum())))
    med = np.array(med)
    for _ in range(iters):
        lab = D[med].argmin(0)
        new = med.copy()
        for c in range(k):
            ix = np.flatnonzero(lab == c)
            if ix.size:
                new[c] = ix[np.argmin(D[np.ix_(ix, ix)].sum(1))]
        if np.array_equal(new, med):
            break
        med = new
    return D[med].argmin(0)
