#!/usr/bin/env python
"""What is in the raw parquet, and which channels are dense enough to model?

    python scripts/inspect_raw.py --parquet data/raw/metabonet_public.parquet

The cohort this study ran on is single-channel by a decision recorded in
`cgmoutlier.data.cohort`: requiring the insulin channels to be present gated out 63% of
the subjects that have usable CGM. That trade was made to double the cohort, and it is
now the leading suspect for why nothing is attackable -- a day of CGM is one smooth,
homeostatically bounded channel whose effective dimensionality is far below its 288
samples, and nothing in it is idiosyncratic the way a programmed basal profile is.

So this reports, per column:

  * null fraction, taken from the PARQUET STATISTICS rather than by reading 154M rows --
    every row group stores its own null count, so the whole table is metadata
  * per-subject coverage: how many subjects have the channel present on enough
    complete days to be usable, which is the number that decides the cohort size
  * co-availability with CGM, which is what the 63% figure refers to

Stage A is metadata only and prints immediately. Stage B reads id plus one channel at a
time in batches, so peak memory is a batch rather than the table.
"""
import argparse
import sys
from collections import defaultdict

import numpy as np

from cgmoutlier._env import check as _envcheck                      # noqa: E402
_envcheck()

T_PER_DAY = 288


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default="data/raw/metabonet_public.parquet")
    ap.add_argument("--min-days", type=int, default=30)
    ap.add_argument("--batch", type=int, default=4_000_000)
    ap.add_argument("--max-channels", type=int, default=12,
                    help="stage B is one pass per channel; cap it")
    ap.add_argument("--stage", choices=("a", "b", "ab"), default="ab",
                    help="'a' is metadata only and returns in seconds")
    a = ap.parse_args()

    import pyarrow.parquet as pq

    pf = pq.ParquetFile(a.parquet)
    names = list(pf.schema_arrow.names)
    md = pf.metadata
    print(f"=== stage A: metadata ===")
    print(f"{a.parquet}")
    print(f"{md.num_rows:,} rows | {md.num_columns} columns | "
          f"{md.num_row_groups} row groups")
    print(f"\ncolumns and dtypes:")
    for n in names:
        print(f"  {n:34} {pf.schema_arrow.field(n).type}")

    # null counts straight from the row-group statistics -- no data is read
    print(f"\n{'column':34}{'null frac':>12}{'non-null rows':>16}")
    nulls = {}
    for j, n in enumerate(names):
        tot = nn = 0
        ok = True
        for i in range(md.num_row_groups):
            st = md.row_group(i).column(j).statistics
            if st is None:
                ok = False
                break
            tot += md.row_group(i).num_rows
            nn += md.row_group(i).num_rows - st.null_count
        if ok:
            nulls[n] = 1 - nn / max(tot, 1)
            print(f"  {n:32}{nulls[n]:>12.4f}{nn:>16,}")
        else:
            print(f"  {n:32}{'(no stats)':>12}")

    # candidate signal channels: numeric, not the obvious keys
    keys = {"id", "date", "day", "time", "timestamp", "study", "subject"}
    chans = [n for n in names
             if n.lower() not in keys
             and str(pf.schema_arrow.field(n).type).startswith(("double", "float",
                                                                "int"))]
    chans = [c for c in chans if nulls.get(c, 1.0) < 0.999][:a.max_channels]
    print(f"\ncandidate channels: {chans}")

    idc = "id" if "id" in names else names[0]
    datec = next((c for c in ("date", "timestamp", "time") if c in names), None)
    print(f"id column = {idc!r}, time column = {datec!r}")

    # a couple of rows, so the units of each channel are on the record
    head = next(pf.iter_batches(batch_size=5)).to_pydict()
    print("\nfirst 5 rows:")
    for n in names:
        print(f"  {n:34} {head[n]}")

    if a.stage == "a":
        return 0

    print(f"\n=== stage B: per-subject coverage (>= {a.min_days} complete days) ===")
    print(f"{'channel':22}{'subjects':>10}{'w/ CGM too':>12}{'rows/subj':>12}")
    have = {}
    for c in chans:
        cols = [idc, c] + ([datec] if datec else [])
        cnt = defaultdict(int)
        for b in pf.iter_batches(batch_size=a.batch, columns=cols):
            d = b.to_pydict()
            v = np.asarray(d[c], dtype="float64")
            ids = np.asarray(d[idc])
            m = ~np.isnan(v)
            if m.any():
                u, k = np.unique(ids[m], return_counts=True)
                for uu, kk in zip(u, k):
                    cnt[uu] += int(kk)
        keep = {k for k, v in cnt.items() if v >= a.min_days * T_PER_DAY}
        have[c] = keep
        med = int(np.median(list(cnt.values()))) if cnt else 0
        print(f"  {c:20}{len(keep):>10}{'':>12}{med:>12,}")

    cgm = next((c for c in chans if c.upper() == "CGM"), None)
    if cgm:
        print(f"\n=== co-availability with {cgm} "
              f"({len(have[cgm])} subjects have {cgm}) ===")
        for c in chans:
            if c == cgm:
                continue
            both = have[cgm] & have[c]
            print(f"  {cgm} + {c:18} {len(both):>6} subjects   "
                  f"({len(both)/max(len(have[cgm]),1):.1%} of {cgm}-usable)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
