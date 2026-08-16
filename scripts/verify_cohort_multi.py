#!/usr/bin/env python
"""Is the multi-channel cohort actually well formed?

    python scripts/verify_cohort_multi.py --cohort data/cohort/metabonet834_c3

`build_cohort_multi.py` reported 173,148 windows built, 173,420 days eligible, and
31,292 days that "failed to reassemble". Those three numbers are inconsistent: at most
272 eligible days can be missing, so 31,292 failures means some (subject, day) was
visited more than once. Either the file is not in (id, date) order, or some days carry
more than 288 rows -- duplicate timestamps, which the eligibility rule `cells >= 288`
admits and the window builder then rejects.

A duplicated window is the failure that matters. It would inflate a subject's day count,
and day count drives nearest-neighbour distance independently of membership, which is
the exact confound the stratified draw exists to prevent.

This checks the built arrays for duplicates, and the cell counts for days above 288.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from cgmoutlier._env import check as _envcheck                      # noqa: E402
_envcheck()

T = 288


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default="data/cohort/metabonet834_c3")
    ap.add_argument("--cells", default="results/channel_coverage/cells_per_day.parquet")
    ap.add_argument("--channels", nargs="+", default=["CGM", "basal", "bolus"])
    a = ap.parse_args()
    p = Path(a.cohort)

    X = np.load(p / "windows.npy", mmap_mode="r")
    sids = np.load(p / "subject_ids.npy", allow_pickle=True)
    days = np.load(p / "days.npy", allow_pickle=True)
    print(f"windows {X.shape} | subject_ids {sids.shape} | days {days.shape}")

    keys = list(zip(sids.tolist(), days.tolist()))
    dup = [k for k, c in Counter(keys).items() if c > 1]
    print(f"\ndistinct (subject, day): {len(set(keys)):,} of {len(keys):,}")
    print(f"DUPLICATED (subject, day): {len(dup):,}")
    if dup:
        print("  examples:", dup[:5])
        print("  a duplicate means the same real day is in the cohort twice; the "
              "cohort must be rebuilt before it is used")

    # how many eligible days have MORE than 288 cells -- duplicate timestamps
    cells = pd.read_parquet(a.cells)
    full = (cells[a.channels] >= T).all(axis=1)
    over = (cells[a.channels] > T).any(axis=1)
    print(f"\neligible days (all channels >= {T} cells): {int(full.sum()):,}")
    print(f"  of which some channel has > {T} cells:    {int((full & over).sum()):,}")
    print(f"  max cells seen in one (subject, day):     "
          f"{int(cells[a.channels].to_numpy().max()):,}")

    # the same question restricted to the subjects that were actually built
    built = set(sids.tolist())
    m = full & full.index.get_level_values(0).isin(built)
    print(f"\nfor the {len(built)} built subjects: {int(m.sum()):,} eligible days, "
          f"{len(set(keys)):,} distinct windows built, "
          f"{int(m.sum()) - len(set(keys)):,} missing")

    # per-subject day counts, which is what the draw is stratified on
    n = pd.Series(Counter(sids.tolist())).sort_values()
    print(f"\ndays per subject: min={n.min()} median={int(n.median())} max={n.max()}")

    bad = int((~np.isfinite(np.asarray(X[:1000]))).sum())
    print(f"non-finite values in the first 1000 windows: {bad}")
    print(f"\nverdict: {'REBUILD' if dup or bad else 'usable'}")
    return 1 if (dup or bad) else 0


if __name__ == "__main__":
    sys.exit(main())
