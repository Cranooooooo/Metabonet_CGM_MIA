#!/usr/bin/env python
"""Is CGM recorded in mixed units across the source studies?

    python scripts/check_units.py

The corrected three-channel cohort came out with CGM mean 96.05 mg/dL and sd 71.19,
against 145.94 / 57.65 for the single-channel cohort over the same corrected key and
144.78 / 56.94 for the shipped `metabonet875`. Requiring basal and bolus lowered the
mean by 50 and RAISED the spread by 14. A subset that is merely sicker or healthier
moves the mean; it does not move mean and spread in opposite directions. A mixture of
two unit systems does: mg/dL near 145 and mmol/L near 8 mix to about
0.65*145 + 0.35*8 = 97 with a wide, bimodal spread.

THIS IS LOAD-BEARING FOR THE IDENTIFIABILITY RESULT. If some subjects are recorded in
mmol/L, they are separable from everyone else by magnitude alone, and every space that
depends on level -- `level`, `quantile`, `spectrum`, `raw` -- would report them as
highly identifiable for a reason that has nothing to do with physiology. That is
exactly the pattern observed: all six spaces separate outliers from normals EXCEPT
`shape`, which is the one space that divides the level out.

So this reports the per-subject mean CGM, its modality, and its breakdown by source
study, on the raw values before any normalisation.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from cgmoutlier._env import check as _envcheck                      # noqa: E402
_envcheck()

from cgmoutlier.data.cohort import load as load_cohort


def raw_cgm(path: str):
    """Per-window mean CGM in the cohort's own units, plus its subject ids."""
    X, sids, man = load_cohort(path)
    X = np.asarray(X)[..., 0]
    cc = man.get("channel_constants")
    mu, sd = ((cc[0]["mean"], cc[0]["sd"]) if cc
              else (man["mean_mgdl"], man["sd_mgdl"]))
    return X.mean(axis=1) * man["zclip"] * sd + mu, np.asarray([str(s) for s in sids])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohorts", nargs="+",
                    default=["data/cohort/metabonet875",
                             "data/cohort/metabonet_sid_c1",
                             "data/cohort/metabonet_sid_c3"])
    ap.add_argument("--out", default="results/identifiability/units.csv")
    a = ap.parse_args()

    recs = []
    for path in a.cohorts:
        if not Path(path).exists():
            print(f"[skip] {path}")
            continue
        v, sids = raw_cgm(path)
        per = pd.Series(v).groupby(sids).mean()
        print(f"\n=== {path}: {len(per)} subjects, {len(v):,} windows ===")
        print(f"per-window mean CGM: "
              f"{np.percentile(v, [1, 25, 50, 75, 99]).round(1).tolist()} "
              f"(p1/p25/p50/p75/p99)")
        # a subject whose whole record sits below 30 cannot be mg/dL: 30 mg/dL is a
        # medical emergency, while 30 mmol/L would be 540 mg/dL
        low = per[per < 30]
        print(f"subjects with mean CGM < 30:  {len(low)} ({len(low) / len(per):.1%})")
        print(f"  their means: {low.round(2).head(10).to_dict()}")
        print(f"subjects with mean CGM in [30, 400]: {int(((per >= 30) & (per <= 400)).sum())}")
        print(f"subjects with mean CGM > 400: {int((per > 400).sum())}")
        # ratio of the two clusters' centres; 18.018 mg/dL per mmol/L
        if len(low):
            print(f"  mg/dL cluster centre {per[per >= 30].median():.1f}, "
                  f"low cluster centre {low.median():.2f}, "
                  f"ratio {per[per >= 30].median() / max(low.median(), 1e-9):.2f} "
                  f"(mg/dL per mmol/L is 18.02)")
        if "/" in per.index[0]:
            study = pd.Series(per.index).str.rsplit("/", n=1).str[0].to_numpy()
            tab = pd.DataFrame(dict(mean_cgm=per.to_numpy(), study=study))
            g = tab.groupby("study")["mean_cgm"].agg(["count", "median", "min", "max"])
            print(f"\nby study:\n{g.round(2).to_string()}")
            recs.append(g.assign(cohort=path))
    if recs:
        pd.concat(recs).to_csv(a.out)
        print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
