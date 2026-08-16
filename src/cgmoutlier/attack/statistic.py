"""d_OUT - d_IN: does including a subject move the released samples closer to it?

For a target subject s with real windows R_s, and a released synthetic set S:

    d(s, S)  =  subject_reduce_i (  set_reduce_k  dist(R_s[i], S[k])  )

The gap is `d_OUT - d_IN`, positive when the model that saw s released samples closer
to s than the model that did not.

WHY THIS MODULE COMPUTES SEVERAL d AND NOT ONE
----------------------------------------------
docs/DESIGN.md leaves the distance open on purpose: "whether d is a mean over pairs, a
minimum, or something else is an open choice to be made once the models exist and the
alternatives can be compared on the same runs". Choosing it after seeing which one
separates the outlier and control arms on the *real* generator would be selecting the
test on the outcome. So the whole grid is computed on every run, and the choice is made
once on `copy_paste` -- a generator that memorises by construction, where the right
answer is known in advance -- and then frozen for everything after.

    set_reduce      min   nearest released sample: the memorisation reading
                    mean  distance to the released distribution as a whole
    subject_reduce  min   the single most exposed window
                    mean  the subject's average exposure
                    q10   a compromise: the tenth percentile over windows

THE K CONFOUND, AND WHY IT IS UNDONE HERE
-----------------------------------------
With K = training-set size, `include_s` releases about n_s more samples than `base`
(~0.1%). E[min distance] falls as K grows, so d_IN would be smaller than d_OUT for a
reason that has nothing to do with membership, and in exactly the direction that looks
like leakage. `match_k=True` (the default) subsamples both released sets to the same
size, with a seed fixed per target, before any distance is computed. Turning it off is
supported so the size of the artefact can be measured rather than asserted.

WHAT A POSITIVE GAP ON ONE TARGET IS NOT
----------------------------------------
Each target gets one training run, so a gap contains training noise as well as any
membership effect. The comparison that means something is between the two arms: the
control arm's gaps are the empirical null. A single outlier's positive gap is not
evidence of anything on its own.
"""
from __future__ import annotations

import hashlib

import numpy as np

SET_REDUCE = ("min", "mean")
SUBJECT_REDUCE = ("min", "q10", "mean")


def _flat(A: np.ndarray) -> np.ndarray:
    A = np.asarray(A, dtype=np.float32)
    return A.reshape(A.shape[0], -1) if A.ndim > 2 else A


def window_distances(R: np.ndarray, S: np.ndarray, *, set_reduce="min",
                     chunk: int = 8192) -> np.ndarray:
    """(n,) distance from each of R's windows to the synthetic set S.

    Chunked over S: at K=177k and T=288 the full (n, K) matrix is a few hundred MB per
    subject, and this runs once per target per model.

    Distances are Euclidean divided by sqrt(F), the same normalisation
    `attack.euclidean_features` uses, so numbers here and there are comparable.

    The expanded form ||x||^2 + ||s||^2 - 2 x.s is used because it is one BLAS call per
    chunk rather than a broadcast difference, which is what makes K=177k affordable. It
    costs precision exactly where the distance is near zero: for an identical pair the
    terms cancel to ~1e-9 in float32 and the sqrt returns ~5e-5 rather than 0. That is a
    floor on d, not a bias -- four orders of magnitude below the distances being
    compared -- but it is why copy_paste gives d_IN ~ 5e-5 and never a clean zero.
    """
    if set_reduce not in SET_REDUCE:
        raise ValueError(f"set_reduce must be one of {SET_REDUCE}, got {set_reduce!r}")
    Rf, Sf = _flat(R), _flat(S)
    if Rf.shape[1] != Sf.shape[1]:
        raise ValueError(f"real dim {Rf.shape[1]} != synthetic dim {Sf.shape[1]}")

    scale = np.sqrt(np.float32(Rf.shape[1]))
    r2 = (Rf ** 2).sum(1, keepdims=True)
    acc = np.full(Rf.shape[0], np.inf, np.float64) if set_reduce == "min" \
        else np.zeros(Rf.shape[0], np.float64)

    for i in range(0, Sf.shape[0], chunk):
        B = Sf[i:i + chunk]
        d2 = r2 + (B ** 2).sum(1)[None, :] - 2.0 * (Rf @ B.T)
        d = np.sqrt(np.clip(d2, 0.0, None)) / scale
        if set_reduce == "min":
            np.minimum(acc, d.min(1), out=acc)
        else:
            acc += d.sum(1)
    if set_reduce == "mean":
        acc /= Sf.shape[0]
    return acc.astype(np.float64)


def reduce_subject(d: np.ndarray, how: str) -> float:
    if how == "min":
        return float(np.min(d))
    if how == "mean":
        return float(np.mean(d))
    if how == "q10":
        return float(np.quantile(d, 0.10))
    raise ValueError(f"subject_reduce must be one of {SUBJECT_REDUCE}, got {how!r}")


def subject_distance(R, S, *, set_reduce="min", subject_reduce="mean", chunk=8192):
    return reduce_subject(window_distances(R, S, set_reduce=set_reduce, chunk=chunk),
                          subject_reduce)


def _match(S_in, S_out, target: str, seed: int, enabled: bool):
    """Cut both released sets to the same size, deterministically per target.

    Seeded with a sha1 of (seed, target), not Python's `hash`: str hashing is salted
    per process, so `hash` would give a different subsample on every run and two
    invocations on the same files would disagree.
    """
    if not enabled:
        return S_in, S_out, None
    k = int(min(len(S_in), len(S_out)))
    h = hashlib.sha1(f"{seed}|{target}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(h[:4], "big"))
    def cut(S):
        if len(S) == k:
            return S
        return S[np.sort(rng.choice(len(S), size=k, replace=False))]
    return cut(S_in), cut(S_out), k


def gap_for_pair(R_target, S_member, S_nonmember, *, target="?", seed=2026,
                 match_k=True, chunk=8192, variants=None):
    """-> list of rows, one per (set_reduce, subject_reduce) variant.

    `S_member` is from the model that saw the target, `S_nonmember` from the one that
    did not. Under the symmetric design every pair is (include_t, base) whatever arm t
    is in, so the base sits on the same side throughout; under the older asymmetric one
    a control's pair was (base, exclude_c) and the base's own offset entered the two
    arms with opposite sign. See cgmoutlier.loo.design. Nothing here depends on which
    form produced the pair -- the design names the two sides and this computes the gap.
    """
    S_in, S_out, k = _match(S_member, S_nonmember, target, seed, match_k)
    variants = variants or [(a, b) for a in SET_REDUCE for b in SUBJECT_REDUCE]

    rows = []
    for sr in sorted({a for a, _ in variants}):
        d_in = window_distances(R_target, S_in, set_reduce=sr, chunk=chunk)
        d_out = window_distances(R_target, S_out, set_reduce=sr, chunk=chunk)
        for a, b in variants:
            if a != sr:
                continue
            di, do = reduce_subject(d_in, b), reduce_subject(d_out, b)
            rows.append(dict(target=str(target), set_reduce=a, subject_reduce=b,
                             d_in=di, d_out=do, gap=do - di,
                             n_windows=int(len(d_in)), k_matched=k))
    return rows


def summarise(rows, alternative="greater"):
    """Two questions per variant, and they are not the same question.

    1. WITHIN each arm: is the gap positive at all? `p_within_outlier` /
       `p_within_control` are Wilcoxon signed-rank tests of gap > 0. This is what a
       known-positive generator has to pass. Note that BOTH arms are membership pairs
       -- symmetrically, (include_t, base) for either arm -- so a generator that
       memorises indiscriminately makes both arms strongly positive and separates
       neither. Reading "the arms did not separate" as a failed sanity check is wrong;
       it is the expected result for copy_paste.

    2. BETWEEN the arms: do outliers leak MORE than day-count-matched typical
       subjects? That is the study's question, and `auc` -- the probability a randomly
       drawn outlier gap exceeds a randomly drawn control gap, 0.5 being no separation
       -- is the answer. The control arm is the empirical null for the training noise
       a single run carries.
    """
    import pandas as pd
    from scipy.stats import mannwhitneyu, wilcoxon

    df = pd.DataFrame(rows)
    if "group" not in df.columns:
        raise ValueError("rows need a 'group' column of 'outlier' / 'control'")

    def within(x):
        """Wilcoxon signed-rank of gap > 0; nan when it cannot be computed."""
        x = np.asarray(x, float)
        if len(x) < 3 or np.allclose(x, 0):
            return np.nan
        try:
            return float(wilcoxon(x, alternative="greater").pvalue)
        except ValueError:
            return np.nan

    out = []
    for (a, b), g in df.groupby(["set_reduce", "subject_reduce"]):
        o = g.loc[g.group == "outlier"]
        c = g.loc[g.group == "control"]
        rec = dict(set_reduce=a, subject_reduce=b,
                   n_outlier=len(o), n_control=len(c),
                   median_gap_outlier=float(np.median(o.gap)) if len(o) else np.nan,
                   median_gap_control=float(np.median(c.gap)) if len(c) else np.nan,
                   mean_gap_outlier=float(np.mean(o.gap)) if len(o) else np.nan,
                   mean_gap_control=float(np.mean(c.gap)) if len(c) else np.nan,
                   median_d_in_outlier=float(np.median(o.d_in)) if len(o) else np.nan,
                   median_d_out_outlier=float(np.median(o.d_out)) if len(o) else np.nan,
                   median_d_in_control=float(np.median(c.d_in)) if len(c) else np.nan,
                   median_d_out_control=float(np.median(c.d_out)) if len(c) else np.nan,
                   p_within_outlier=within(o.gap), p_within_control=within(c.gap))
        if len(o) and len(c):
            u, p = mannwhitneyu(o.gap, c.gap, alternative=alternative)
            rec.update(auc=float(u) / (len(o) * len(c)), p_between=float(p))
            # The outliers share one base model, so their gaps are correlated and both
            # p-values are computed as if they were not: a screen, not a test.
            rec["p_note"] = "outliers share the base model; gaps are correlated"
        out.append(rec)
    return pd.DataFrame(out).sort_values(["set_reduce", "subject_reduce"])
