#!/usr/bin/env python
"""How comparable are the candidate groups before anything is trained?

    python scripts/group_balance.py --by gender
    python scripts/group_balance.py --by age --by gender

A per-group generator campaign compares leakage between groups. Any covariate that
differs between them is a rival explanation for whatever difference is found, and the
strongest one is not clinical at all:

  * **subjects per group** -- membership leakage scales with a subject's share of the
    training data, so unequal group sizes alone can produce the effect being looked for
  * **windows per subject** -- a longer record is more of the training set
  * **total windows** -- what the generator actually sees
  * **within-group spread** -- a homogeneous group hides its members in the crowd

This prints all four per group, so the design can match on them BEFORE 600 GPU-hours are
committed rather than discovering afterwards that the comparison was about sample size.

It reports the matched size a balanced design would have to use: the smallest group's
count, since every group must be subsampled to it for the comparison to mean anything.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from cgmoutlier._env import check as _envcheck                      # noqa: E402
_envcheck()
from cgmoutlier.data.cohort import load as load_cohort              # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default="data/cohort/metabonet875")
    ap.add_argument("--by", action="append", default=None,
                    help="grouping column; repeat for a cross (age, gender, ...)")
    a = ap.parse_args()
    by = a.by or ["gender"]

    subs = pd.read_parquet(Path(a.cohort) / "subjects.parquet")
    subs["id"] = subs["id"].astype(str)
    X, sids, man = load_cohort(a.cohort)
    sids = pd.Series(sids).astype(str)

    # per-subject window count and a crude within-subject spread, from the windows
    # themselves rather than from a metric module, so this stays generator-agnostic
    Xf = np.asarray(X).reshape(len(X), -1)
    per = pd.DataFrame({"id": sids, "mean": Xf.mean(1), "sd": Xf.std(1)})
    agg = per.groupby("id").agg(n_windows=("mean", "size"),
                                subj_mean=("mean", "mean"),
                                subj_sd=("sd", "mean")).reset_index()
    d = subs.merge(agg, on="id", how="inner")
    print(f"[bal] {len(d)} subjects with windows\n")

    keys = []
    for col in by:
        s = d[col]
        if s.dtype.kind in "biufc" and s.nunique() > 8:
            med = s.median()
            d[f"_{col}"] = np.where(s > med, f"{col}>{med:g}", f"{col}<={med:g}")
            print(f"[bal] {col} split at median {med:g}")
        else:
            d[f"_{col}"] = s.astype("string")
        keys.append(f"_{col}")

    d = d.dropna(subset=keys)
    g = d.groupby(keys)
    tab = g.agg(subjects=("id", "size"),
                windows=("n_windows", "sum"),
                win_per_subj=("n_windows", "median"),
                level=("subj_mean", "median"),
                variability=("subj_sd", "median")).reset_index()
    tab = tab.sort_values("subjects", ascending=False)

    print(f"\n{'group':34}{'subjects':>10}{'windows':>10}{'win/subj':>10}"
          f"{'level':>9}{'variab':>9}")
    for _, r in tab.iterrows():
        name = " | ".join(str(r[k]) for k in keys)
        print(f"{name[:32]:34}{r.subjects:>10}{r.windows:>10}"
              f"{r.win_per_subj:>10.0f}{r.level:>9.3f}{r.variability:>9.3f}")

    usable = tab[tab.subjects >= 60]
    print(f"\n[bal] groups with >= 60 subjects (a background plus 20 targets "
          f"needs at least that): {len(usable)} of {len(tab)}")
    if len(usable) >= 2:
        n = int(usable.subjects.min())
        w = int(usable.win_per_subj.min())
        print(f"[bal] a matched design must subsample every group to "
              f"**{n} subjects**, and capping windows per subject at ~{w} would "
              f"equalise the second covariate too")
        print(f"[bal] largest/smallest subject ratio {usable.subjects.max()/n:.2f}x, "
              f"windows ratio {usable.windows.max()/usable.windows.min():.2f}x")
        print("\n[bal] imbalance to watch: the level and variability columns are the "
              "signal itself.\n      If they differ between groups, the generators are "
              "not learning the same\n      task and leakage is not comparable even at "
              "matched N.")
    else:
        print("[bal] not enough groups clear the size floor for a matched design")
    return 0


if __name__ == "__main__":
    sys.exit(main())
