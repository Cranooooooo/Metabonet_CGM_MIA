"""Per-window features against ONE released set, for the supervised attacker panel.

THE THREAT MODEL THIS SERVES, AND HOW IT DIFFERS FROM `statistic`
----------------------------------------------------------------
`attack.statistic` computes `d_OUT - d_IN`: it holds both released sets at once and
takes their difference. That is the study's instrument, not an adversary -- nobody is
handed a model trained deliberately without their target.

Here the attacker holds ONE released set S and one subject's records, and a classifier
trained per subject decides membership from features of (window, S) alone. Nothing in
this module ever sees two sets together; the pairing exists only in how the resulting
scores are *evaluated*.

WHY THE NEGATIVE CLASS MUST SPAN SEVERAL RELEASED SETS
-----------------------------------------------------
The obvious construction gives subject t its positives from `include_t` and all its
negatives from `base`. Those are two different training runs, and this project has
already measured that a run's offset is large enough to dominate within-arm readings.
A classifier can then separate the classes by recognising *which release it is looking
at*, with no membership information at all -- and splitting train/test by window does
not break it, because the held-out windows face the same two sets.

So negatives are drawn from `base` AND from several `include_u` (u != t): t is a
non-member of every one of them, each carries its own independent offset, and no
property of a single release predicts the label any more.

`negative_control_sets` returns an assignment in which the *positive* set is also one
the subject does not belong to. Its true AUC is 0.5 by construction, so whatever it
reads is this panel's residual bias, measured rather than assumed.

K IS MATCHED ACROSS ALL SETS AT ONCE
------------------------------------
`include_*` releases ~0.1% more samples than `base`, and nearest-neighbour distance
falls with K, so an unmatched comparison biases every min-like feature in the direction
that looks like leakage. `attack.statistic` matches K pairwise; here a subject is
scored against six sets that must all be mutually comparable, so they are cut to one
common K -- pairwise matching would leave the negatives comparable to the positive but
not to each other.

THRESHOLDS ARE CALIBRATED OFF THE RELEASED SETS ENTIRELY
--------------------------------------------------------
`euc_ball` and `cos_ball` need a radius. Deriving it from the call's own distances
would give the member and non-member conditions different thresholds -- a confound
manufactured by the feature. `calibrate` therefore takes it from REAL-vs-REAL window
distances, which depend on no model at all and are shared by every set and subject.
"""
from __future__ import annotations

import hashlib
from typing import Dict, List, Tuple

import numpy as np

FEATURE_NAMES: List[str] = [
    "euc_min", "euc_avg", "euc_q01", "euc_q05", "euc_ball",
    "knn5", "knn20", "softmin", "cos_max", "cos_ball",
]

# The named subsets the panel's feature axis is built from.
FEATURE_SETS: Dict[str, List[str]] = {
    "F1_min": ["euc_min"],
    "F2_cheap4": ["euc_min", "euc_avg", "euc_ball", "cos_ball"],
    "F3_raw10": list(FEATURE_NAMES),
}


def _flat(A: np.ndarray) -> np.ndarray:
    A = np.asarray(A, dtype=np.float32)
    return A.reshape(A.shape[0], -1) if A.ndim > 2 else A


def calibrate(real_windows: np.ndarray, *, n: int = 2000, r_q: float = 0.05,
              cos_q: float = 0.95, seed: int = 2026) -> Dict[str, float]:
    """Membership-independent thresholds, from real windows only.

    Returns r_euc (the r_q quantile of real-vs-real distance), tau (the median, the
    softmin bandwidth) and cos_c0 (the cos_q quantile of real-vs-real cosine).

    Nothing here touches a released set, so the same constants apply to every model in
    the study and cannot move between the member and non-member conditions.
    """
    R = _flat(real_windows)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(R), size=int(min(n, len(R))), replace=False)
    A = R[np.sort(idx)]
    D = _pairwise(A, A)
    iu = np.triu_indices(len(A), k=1)
    v = D[iu]
    C = _cosine(A, A)[iu]
    return dict(r_euc=float(np.quantile(v, r_q)),
                tau=float(np.median(v)) or 1.0,
                cos_c0=float(np.quantile(C, cos_q)),
                n_calib=int(len(A)))


def _pairwise(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """(n,d),(K,d) -> (n,K) Euclidean distance normalised by sqrt(d).

    Same normalisation as `attack.statistic.window_distances`, so a number here and a
    number there mean the same thing. The expanded form is one BLAS call; it costs
    precision only where the distance is near zero (~5e-5 instead of 0), four orders
    below anything being compared.
    """
    a2 = (A ** 2).sum(1, keepdims=True)
    b2 = (B ** 2).sum(1, keepdims=True).T
    d2 = np.maximum(a2 + b2 - 2.0 * (A @ B.T), 0.0)
    return np.sqrt(d2, dtype=np.float32) / np.sqrt(np.float32(A.shape[1]))


def _cosine(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    An = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    Bn = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
    return (An @ Bn.T).astype(np.float32)


def per_window_features(R: np.ndarray, S: np.ndarray, *, cal: Dict[str, float],
                        block: int = 128) -> np.ndarray:
    """(n_windows, len(FEATURE_NAMES)) features for R's windows against one set S.

    Rows are processed `block` at a time and the Euclidean matrix is released before
    the cosine one is built, so peak memory is ~2 x block x K x 4 bytes (90 MB at the
    default) whatever n is. The first version held both full (n, K) matrices at once;
    at n=370 and K=176k that is over 500 MB per subject on top of the 200 MB release,
    and three such jobs were OOM-killed by the scheduler.

    Order statistics come from one `np.partition` per block rather than `np.quantile`,
    which sorts: the five columns that are order statistics need the same partial
    ordering, and a full sort of 176k values per row buys nothing.
    """
    Rf, Sf = _flat(R), _flat(S)
    if Rf.shape[1] != Sf.shape[1]:
        raise ValueError(f"real dim {Rf.shape[1]} != synthetic dim {Sf.shape[1]}")
    K = len(Sf)
    kths = [0, min(4, K - 1), min(19, K - 1),
            min(max(1, int(0.01 * K)) - 1, K - 1),
            min(max(1, int(0.05 * K)) - 1, K - 1)]
    i_min, i_k5, i_k20, i_q01, i_q05 = kths

    out = np.empty((len(Rf), len(FEATURE_NAMES)), dtype=np.float32)
    for lo in range(0, len(Rf), block):
        hi = min(lo + block, len(Rf))
        D = _pairwise(Rf[lo:hi], Sf)
        part = np.partition(D, kth=sorted(set(kths)), axis=1)
        mn = part[:, i_min]
        out[lo:hi, 0] = mn                                      # euc_min
        out[lo:hi, 1] = D.mean(1)                               # euc_avg
        out[lo:hi, 2] = part[:, i_q01]                          # euc_q01
        out[lo:hi, 3] = part[:, i_q05]                          # euc_q05
        out[lo:hi, 4] = (D < cal["r_euc"]).mean(1)              # euc_ball
        out[lo:hi, 5] = part[:, i_k5]                           # knn5
        out[lo:hi, 6] = part[:, i_k20]                          # knn20
        # softmin, offset by the row minimum so the exponential cannot underflow
        out[lo:hi, 7] = mn - cal["tau"] * np.log(
            np.exp(-(D - mn[:, None]) / cal["tau"]).mean(1))
        del D, part
        C = _cosine(Rf[lo:hi], Sf)
        out[lo:hi, 8] = C.max(1)                                # cos_max
        out[lo:hi, 9] = (C > cal["cos_c0"]).mean(1)             # cos_ball
        del C
    return out


def common_k(lengths: Dict[str, int]) -> int:
    """The K every one of a subject's released sets is cut to.

    Not pairwise: a subject is scored against one positive and several negative sets
    and all of them enter the same classifier, so a K that differs between two
    negatives is as much a confound as one that differs between positive and negative.
    Lengths are read from the files' headers, so this is known before any set is
    loaded.
    """
    return int(min(lengths.values()))


def cut_to_k(S: np.ndarray, k: int, *, target: str, set_name: str, seed: int = 2026
             ) -> np.ndarray:
    """Subsample one released set to k rows, reproducibly.

    The stream is derived from (seed, target, set_name) rather than advanced across a
    subject's sets in sequence: the extraction loops over RELEASED SETS on the outside
    so each 200 MB file is read once, which means a subject's sets are cut in an order
    that depends on the loop, not on the subject. A per-pair seed makes the result
    identical either way -- the same property `loo.train._job_seed` exists for.
    """
    if len(S) == k:
        return S
    h = hashlib.sha1(f"{seed}|{target}|{set_name}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(h[:4], "big"))
    return S[np.sort(rng.choice(len(S), size=k, replace=False))]


def normalise(F: np.ndarray, ref_mean: np.ndarray, ref_sd: np.ndarray) -> np.ndarray:
    """Express features in units of the release's own reference spread.

    THE PROBLEM THIS SOLVES. A subject's positives all come from ONE release, so
    whatever is idiosyncratic about that release -- the offset a single training run
    carries -- predicts the label perfectly in training, and a classifier will take it
    instead of membership. The negative control measured this directly on a three-
    subject pilot: with the raw features it read 0.667 where it must read 0.5.

    THE FIX. `ref_mean` and `ref_sd` come from a fixed set of BACKGROUND windows scored
    against that same release. Background subjects are members of every release, so
    the reference carries no membership information of its own; subtracting it removes
    any global property of the release while leaving intact the part that is specific
    to a subject who is not in the reference.

    This is the feature-level form of the cancellation the symmetric design already
    relies on: put the run's offset on both sides so it leaves the comparison.
    """
    return ((F - ref_mean[None, :]) / np.maximum(ref_sd[None, :], 1e-12)
            ).astype(np.float32)


def assign_negative_sets(target: str, all_targets: List[str], *, n_neg: int = 4,
                         seed: int = 2026) -> List[str]:
    """Which `include_u` releases supply subject t's negatives, plus `base`.

    Deterministic in the target's name so a rerun, or a run that processes the targets
    in another order, draws the same sets -- the property `loo.train._job_seed` exists
    for, applied to the attack side.
    """
    others = sorted(u for u in all_targets if u != target)
    h = hashlib.sha1(f"{seed}|neg|{target}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(h[:4], "big"))
    pick = rng.choice(len(others), size=int(min(n_neg, len(others))), replace=False)
    return ["base"] + [f"include_{others[i]}" for i in sorted(pick)]
