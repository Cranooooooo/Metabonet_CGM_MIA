#!/usr/bin/env python
"""Can the 80 subjects that have models answer a question about clinically-defined
groups, or does that need a new campaign?

    python scripts/subgroup_coverage.py

WHY THIS COMES BEFORE THE ANALYSIS
----------------------------------
Membership leakage can only be measured for a subject we hold a counterfactual for --
a model trained WITHOUT them. That is the 40 targets per replicate and nobody else:
the 835 background subjects are members of every release, so there is no model to
compare against. The cohort's clinical variables cover all 875 people, but the
measurement covers 80.

So the question "do insulin-pump users leak more than injection users" is free if the
80 span both groups with enough people, and costs a fresh 41-model campaign if they do
not. This script decides which, before anything is claimed either way.

It also runs the analysis where coverage allows, because at that point it is two more
lines and the alternative is a second job for the same data.

WHAT IT WILL NOT DO
-------------------
Report a p-value per variable and let the smallest one be the finding. There are ten
demographic columns and eight consensus metrics here against 80 subjects; the smallest
of eighteen p-values is not evidence. Groups worth testing have to be named in advance
on clinical grounds -- `cv > 36%` is the international consensus definition of unstable
glycaemia and is the one pre-specified here. Everything else is printed as coverage
only, to decide what a designed experiment would target.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from cgmoutlier._env import check as _envcheck                      # noqa: E402
_envcheck()

CV_UNSTABLE = 36.0          # Battelino 2019 consensus threshold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default="data/cohort/metabonet875")
    ap.add_argument("--auc", default="results/attack/dimts_h128_subject_auc/"
                                     "per_subject.csv")
    ap.add_argument("--outliers", default="results/outliers")
    a = ap.parse_args()

    subs = pd.read_parquet(Path(a.cohort) / "subjects.parquet")
    subs["id"] = subs["id"].astype(str)
    print(f"[cov] cohort metadata: {len(subs)} subjects, "
          f"columns = {list(subs.columns)}\n", flush=True)

    auc = pd.read_csv(a.auc)
    auc["target"] = auc["target"].astype(str)
    print(f"[cov] measured subjects: {len(auc)} "
          f"({(auc.group == 'outlier').sum()} outlier, "
          f"{(auc.group == 'control').sum()} control)\n", flush=True)

    d = subs.merge(auc, left_on="id", right_on="target", how="left")
    d["measured"] = d["auc"].notna()

    print("=== coverage of the measured 80 across each clinical variable ===")
    print("(a group needs subjects on BOTH sides to be answerable at all)\n")
    for col in subs.columns:
        if col == "id":
            continue
        s = d[col]
        if s.dtype.kind in "biufc" and s.nunique() > 8:
            # continuous: split at the median of the FULL cohort, not of the measured
            # subset, so the groups mean the same thing in both
            med = s.median()
            g = pd.Series(np.where(s > med, f"{col}>med", f"{col}<=med"),
                          index=d.index).where(s.notna())
            label = f"{col} (median {med:.1f})"
        else:
            g = s.astype("string")
            label = col
        tab = pd.crosstab(g, d["measured"], dropna=False)
        if True not in tab.columns:
            tab[True] = 0
        rows = [(str(k), int(v.get(False, 0) + v.get(True, 0)), int(v.get(True, 0)))
                for k, v in tab.iterrows()]
        rows = [r for r in rows if r[1] > 0]
        both = sum(1 for r in rows if r[2] >= 10)
        flag = "OK" if both >= 2 else "insufficient"
        print(f"  {label:42} {flag}")
        for name, n_all, n_meas in sorted(rows, key=lambda r: -r[1])[:6]:
            print(f"      {name[:28]:30} cohort {n_all:4}   measured {n_meas:3}")
        print()

    # --- the one pre-specified clinical grouping -----------------------------
    cv_path = Path(a.outliers) / "A2.parquet"
    if cv_path.exists():
        cv = pd.read_parquet(cv_path)
        cv.columns = [c.lower() for c in cv.columns]
        idc = "id" if "id" in cv.columns else cv.columns[0]
        cvc = next((c for c in cv.columns if "cv" in c and c != idc), None)
        print(f"=== pre-specified: unstable glycaemia (CV > {CV_UNSTABLE}%) ===")
        print(f"    from {cv_path}, columns {list(cv.columns)}", flush=True)
        if cvc:
            cv[idc] = cv[idc].astype(str)
            # A2's score is CV - 36, so recover CV where that is what was stored
            v = cv[cvc].to_numpy(float)
            cvv = v + CV_UNSTABLE if np.nanmedian(v) < 10 else v
            cv = cv.assign(cv_pct=cvv)[[idc, "cv_pct"]].rename(columns={idc: "id"})
            m = auc.merge(cv, left_on="target", right_on="id", how="left")
            m["unstable"] = m["cv_pct"] > CV_UNSTABLE
            print(f"\n    cohort-wide: {(cvv > CV_UNSTABLE).sum()} of {len(cvv)} "
                  f"subjects unstable ({(cvv > CV_UNSTABLE).mean():.1%})")
            for grp, x in m.groupby("group"):
                for u, y in x.groupby("unstable"):
                    print(f"    {grp:8} {'unstable' if u else 'stable':10} "
                          f"n={len(y):3}  AUC mean {y.auc.mean():.4f}  "
                          f"max {y.auc.max():.4f}")
            ok = m.dropna(subset=["cv_pct"])
            if ok.unstable.sum() >= 10 and (~ok.unstable).sum() >= 10:
                from scipy.stats import mannwhitneyu
                p = mannwhitneyu(ok[ok.unstable].auc, ok[~ok.unstable].auc,
                                 alternative="greater")[1]
                print(f"\n    unstable > stable, one-sided Mann-Whitney p = {p:.4f}"
                      f"   (n={int(ok.unstable.sum())} vs "
                      f"{int((~ok.unstable).sum())}, subjects are the unit)")
            else:
                print("\n    insufficient coverage for a test on this grouping")
    return 0


if __name__ == "__main__":
    sys.exit(main())
