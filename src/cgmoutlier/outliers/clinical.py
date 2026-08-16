"""Group A -- clinical, literature-grounded (A1, A2, A3, A4).

Anchored on Battelino et al., Diabetes Care 2019, the international consensus on which
CGM metrics to report. A1/A4 use the eight consensus metrics; A2 is the consensus
threshold itself (CV > 36% = unstable glycaemia), a zero-fit baseline; A3 uses the
highest-discriminant-ratio member of each correlation cluster instead.

WHY THE BATTERY IS REDUCED BEFORE ANY DISTANCE IS TAKEN
The 32-metric battery collapses to five correlation clusters at |r_s| >= 0.85 and one
of them holds sixteen metrics -- everything hyperglycaemia-flavoured. Feeding all 32
into a distance weights "high blood glucose" 16:1 against everything else, so the
output ranks glycaemic control rather than atypicality.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..clinical import cgm_metrics as CM
from ..data.cohort import channel_raw
from .common import robust_mahalanobis, robust_scale

CONSENSUS = ["tir_70_180", "tbr_70", "t_lt_54", "tar_180", "t_gt_250", "mean", "gmi", "cv"]
CLUSTER_REPS = ["conga1", "grade_hypo", "grade_hyper", "cv", "gri"]
CV_UNSTABLE = 36.0


def per_subject_metrics(X, sids, man, keys=None, verbose=True) -> pd.DataFrame:
    keys = keys or sorted(set(CONSENSUS) | set(CLUSTER_REPS))
    # Always the CGM channel, whatever else the cohort carries. There is no time in
    # range for a basal rate, so group A is glucose-only in a multichannel run too --
    # which also means these four votes are IDENTICAL between a one-channel and a
    # three-channel pass, and any difference in the two outlier lists comes from the
    # other nine methods.
    g = channel_raw(X, man, "CGM")
    rows = np.empty((g.shape[0], len(keys)))
    for i, row in enumerate(g):
        m = CM.compute_window(row, man["dt_min"])
        rows[i] = [m[k] for k in keys]
        if verbose and i and i % 20000 == 0:
            print(f"  clinical {i}/{g.shape[0]}", flush=True)
    return pd.DataFrame(rows, columns=keys).assign(id=sids).groupby("id").mean()


def A1(C):  # consensus-8 robust Mahalanobis
    return robust_mahalanobis(C[CONSENSUS])


def A2(C):  # the consensus rule itself; positive = unstable
    return C["cv"].to_numpy() - CV_UNSTABLE


def A3(C):  # cluster-representative Mahalanobis
    return robust_mahalanobis(C[CLUSTER_REPS])


def A4(C):  # consensus-8 Isolation Forest
    from sklearn.ensemble import IsolationForest
    S = robust_scale(C[CONSENSUS])
    return -IsolationForest(n_estimators=400, random_state=0).fit(S).score_samples(S)
