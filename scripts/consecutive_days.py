#!/usr/bin/env python
"""How many subjects have 7, 14 or 30 CONSECUTIVE complete days?

    python scripts/consecutive_days.py

Proposal 2 is to model a week, a fortnight or a month as one window instead of a day,
on the clinical reading that about fifteen days of CGM pins an individual down. That
proposal has a prerequisite nobody has checked: a longer window has to be CONTIGUOUS.
The cohort's 197,970 complete days are complete *individually* -- a subject with 200
usable days scattered over eighteen months may not hold a single unbroken fortnight.

A gap cannot be papered over here. Imputing the missing hours would inject a model's
idea of the subject into the very data an identifiability or membership measurement is
about to read, which is the one thing that must not happen.

So this reports, per window length L and per channel set:

  * subjects holding at least one run of L consecutive complete days
  * NON-OVERLAPPING windows, which is what an honest training set holds
  * sliding windows at stride 1, which is what is available if correlated windows are
    acceptable, and the ratio between the two -- the effective sample size is the
    non-overlapping count however many sliding windows are cut
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from cgmoutlier._env import check as _envcheck                      # noqa: E402
_envcheck()

T = 288


def runs(days: np.ndarray) -> np.ndarray:
    """Lengths of maximal runs of consecutive integers in a sorted unique array."""
    if days.size == 0:
        return np.zeros(0, dtype=int)
    brk = np.flatnonzero(np.diff(days) != 1)
    return np.diff(np.concatenate([[-1], brk, [days.size - 1]]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells",
                    default="results/channel_coverage_studyid/cells_per_day.parquet")
    ap.add_argument("--sets", nargs="+",
                    default=["CGM", "CGM+basal", "CGM+basal+bolus"])
    ap.add_argument("--lengths", nargs="+", type=int, default=[1, 7, 14, 30])
    ap.add_argument("--min-days", type=int, default=30)
    ap.add_argument("--out", default="results/channel_coverage_studyid/consecutive.csv")
    a = ap.parse_args()

    cells = pd.read_parquet(a.cells)
    ids = cells.index.get_level_values(0).astype(str).to_numpy()
    day = cells.index.get_level_values(1).values.astype("datetime64[D]").astype(np.int64)

    recs = []
    for cs in a.sets:
        chans = cs.split("+")
        full = (cells[chans] >= T).all(axis=1).to_numpy()
        sub_i, sub_d = ids[full], day[full]
        order = np.lexsort((sub_d, sub_i))
        sub_i, sub_d = sub_i[order], sub_d[order]
        # subjects that clear the existing cohort bar, so the comparison is like for like
        cnt = pd.Series(sub_i).value_counts()
        elig = set(cnt[cnt >= a.min_days].index)

        per_runs = {}
        start = 0
        for k in range(1, len(sub_i) + 1):
            if k == len(sub_i) or sub_i[k] != sub_i[start]:
                if sub_i[start] in elig:
                    per_runs[sub_i[start]] = runs(np.unique(sub_d[start:k]))
                start = k

        print(f"\n=== {cs}: {len(per_runs)} subjects with >= {a.min_days} complete days ===")
        print(f"{'L (days)':>10}{'subjects':>10}{'% of them':>11}"
              f"{'non-overlap':>13}{'sliding':>10}{'ratio':>8}{'longest run':>13}")
        longest = np.array([r.max() if r.size else 0 for r in per_runs.values()])
        for L in a.lengths:
            have = {s: r for s, r in per_runs.items() if (r >= L).any()}
            nonov = sum(int((r // L).sum()) for r in per_runs.values())
            slide = sum(int(np.maximum(r - L + 1, 0).sum()) for r in per_runs.values())
            print(f"{L:>10}{len(have):>10}{len(have) / max(len(per_runs), 1):>11.1%}"
                  f"{nonov:>13,}{slide:>10,}{slide / max(nonov, 1):>8.1f}"
                  f"{'':>13}")
            recs.append(dict(channels=cs, length=L, subjects=len(have),
                             pct_subjects=len(have) / max(len(per_runs), 1),
                             non_overlapping=nonov, sliding=slide))
        print(f"  longest run per subject: median {int(np.median(longest))}, "
              f"p90 {int(np.percentile(longest, 90))}, max {int(longest.max())}")
        print(f"  subjects whose longest run is 1 day: "
              f"{int((longest == 1).sum())} ({(longest == 1).mean():.1%})")

    pd.DataFrame(recs).to_csv(a.out, index=False)
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
