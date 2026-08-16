#!/usr/bin/env python
"""Does the shipped single-channel cohort contain 6-hour windows labelled as days?

    python scripts/check_duplicate_days.py

`scripts/verify_cohort_multi.py` found (subject, day) pairs carrying up to 1,152 cells
-- four rows per five-minute slot, not one. That is a property of the raw file, so it
reaches the SHIPPED cohort too, and `cgmoutlier.data.cohort.build` has no defence
against it:

    cnt = d.groupby(["id", "day"])["CGM"].count();  cnt[cnt >= t]      # 1152 >= 288
    v = g["CGM"].to_numpy(np.float32);  wins.append(v[:T])             # first 288 rows

With rows sorted by (id, date) and each timestamp repeated k times, the first 288 rows
span only 288/k distinct timestamps. At k=4 a "day" is the first six hours, sampled
four times over. Nothing raises: the array has the right shape and finite values.

TWO INDEPENDENT CHECKS, because one alone can be explained away:

  A. the raw file -- how many CGM-eligible days hold more than 288 cells, which is the
     population at risk, and whether the repeats are identical values (a duplicated
     export) or different ones (genuinely finer sampling)
  B. the shipped windows themselves -- a window built from k-fold repeated timestamps
     has runs of k identical consecutive samples. Real CGM is quantised to 1 mg/dL and
     does repeat a value now and then, so the test is the RUN-LENGTH DISTRIBUTION
     against the rest of the cohort, not the presence of any repeat at all.

Check B is the one that matters: it reads the artefact off the published data without
reference to how it was built.
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

T = 288


def run_lengths(x: np.ndarray) -> np.ndarray:
    """Length of each run of equal consecutive values, per row."""
    same = np.diff(x, axis=1) == 0
    return same.mean(axis=1)          # fraction of adjacent pairs that are equal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default="results/channel_coverage/cells_per_day.parquet")
    ap.add_argument("--cohort", default="data/cohort/metabonet875")
    ap.add_argument("--parquet", default="data/raw/metabonet_public.parquet")
    ap.add_argument("--out", default="results/identifiability/duplicate_days.csv")
    a = ap.parse_args()

    print("=== A. the raw file ===")
    cells = pd.read_parquet(a.cells)
    cgm = cells["CGM"]
    elig = cgm >= T
    over = cgm > T
    print(f"CGM-eligible (subject, day) pairs: {int(elig.sum()):,}")
    print(f"  with exactly {T} cells:  {int((cgm == T).sum()):,}")
    print(f"  with more than {T}:      {int((elig & over).sum()):,} "
          f"({(elig & over).sum() / max(elig.sum(), 1):.1%} of eligible)")
    vc = cgm[elig & over].value_counts().head(8)
    print(f"  the common oversized counts: {dict(vc)}")
    n_sub_over = (elig & over).groupby(level=0, observed=True).sum()
    print(f"  subjects with at least one oversized day: "
          f"{int((n_sub_over > 0).sum()):,} of "
          f"{int((elig.groupby(level=0, observed=True).sum() > 0).sum()):,}")

    print("\n=== B. the shipped windows ===")
    X, sids, man = load_cohort(a.cohort)
    X = np.asarray(X)[..., 0]
    frac = run_lengths(X)
    print(f"{X.shape[0]:,} windows | fraction of adjacent samples that are equal:")
    for q in (0.5, 0.9, 0.99, 0.999, 1.0):
        print(f"  q{q:<6} {np.quantile(frac, q):.4f}")
    # a k-fold repeated day has (k-1)/k of its adjacent pairs equal: 0.50 at k=2,
    # 0.67 at k=3, 0.75 at k=4
    for k, thr in ((2, 0.45), (3, 0.60), (4, 0.70)):
        n = int((frac >= thr).sum())
        print(f"  windows with >= {thr:.2f} equal-adjacent (consistent with k>={k}): "
              f"{n:,} ({n / X.shape[0]:.2%})")

    susp = frac >= 0.45
    print(f"\nsubjects holding a suspect window: "
          f"{len(set(sids[susp].tolist())):,} of {len(set(sids.tolist())):,}")
    pd.DataFrame(dict(subject=sids, equal_adjacent_frac=frac)).to_csv(a.out, index=False)

    # what a clean day looks like, for scale
    print(f"\nfor reference, median equal-adjacent fraction among the cleanest 50%: "
          f"{np.median(frac[frac <= np.median(frac)]):.4f}")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
