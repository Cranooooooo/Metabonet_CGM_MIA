#!/usr/bin/env python
"""How much of the arm-level AUC is subject 1142?

Per-subject, removing 1142 drops the outlier/control mean difference from 0.0111 to
0.0043 -- 61% of it is one person. The headline 0.680 is a different statistic: a rank
comparison of 20 gaps against 20, not a mean of per-subject AUCs, so the two do not
imply each other. If 0.680 collapses to 0.5 without him, the finding IS him; if it
holds, he is the tail of a broad effect.

The frozen variant only (min x mean). Recomputing the AUC on a subset is not
variant-shopping -- the statistic is unchanged, one target is dropped.
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

from cgmoutlier._env import check as _envcheck                      # noqa: E402
_envcheck()

def arm_auc(out, ctl):
    if not len(out) or not len(ctl):
        return np.nan, np.nan
    u = stats.mannwhitneyu(out, ctl, alternative="greater")
    return u.statistic / (len(out) * len(ctl)), u.pvalue

print(f"{'rep':>5}{'n_out':>7}{'AUC all':>10}{'AUC -1142':>11}{'delta':>9}"
      f"{'p all':>9}{'p -1142':>10}")
rows = []
for r in (1, 2, 3):
    p = Path(f"results/attack/dimts_h128_rep{r}/gaps.parquet")
    if not p.exists():
        continue
    d = pd.read_parquet(p)
    d = d[(d.set_reduce == "min") & (d.subject_reduce == "mean")]
    d["target"] = d["target"].astype(str)
    o = d[d.group == "outlier"]["gap"].to_numpy()
    c = d[d.group == "control"]["gap"].to_numpy()
    o2 = d[(d.group == "outlier") & (d.target != "1142")]["gap"].to_numpy()
    a1, p1 = arm_auc(o, c)
    a2, p2 = arm_auc(o2, c)
    print(f"{r:>5}{len(o):>7}{a1:>10.4f}{a2:>11.4f}{a2-a1:>+9.4f}{p1:>9.4f}{p2:>10.4f}")
    rows.append((a1, a2))

if rows:
    A = np.array(rows)
    print(f"\n{'':>5}{'':>7}{A[:,0].mean():>10.4f}{A[:,1].mean():>11.4f}"
          f"{A[:,1].mean()-A[:,0].mean():>+9.4f}   <- mean over replicates")
    print(f"\n1142's own gap, and where it ranks among the 20 outliers:")
    for r in (1, 2, 3):
        p = Path(f"results/attack/dimts_h128_rep{r}/gaps.parquet")
        if not p.exists():
            continue
        d = pd.read_parquet(p)
        d = d[(d.set_reduce == "min") & (d.subject_reduce == "mean")]
        d["target"] = d["target"].astype(str)
        o = d[d.group == "outlier"].sort_values("gap", ascending=False)
        g = o[o.target == "1142"]
        if len(g):
            rank = int((o["gap"] > g["gap"].iloc[0]).sum()) + 1
            print(f"  rep{r}: gap {g['gap'].iloc[0]:+.6f}, rank {rank}/20, "
                  f"outlier median {o['gap'].median():+.6f}")
