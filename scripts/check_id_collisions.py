#!/usr/bin/env python
"""Are the duplicated (subject, day) rows one person exported twice, or two people?

    python scripts/check_id_collisions.py

Up to 1,152 rows exist for a single (id, day) -- four per five-minute slot. Two
explanations, with very different consequences:

  DUPLICATE EXPORT   the same person's day appears more than once. Harmless once
                     deduplicated; the cohort keeps its subject count.
  ID COLLISION       `id` is only unique WITHIN a source study, so "102" in Loop and
                     "102" in DCLP3 are different people merged into one subject. Then
                     the 875-subject cohort contains composite subjects, every
                     per-subject distance is computed across two people's data, and
                     the membership design -- which holds a subject in or out as a
                     unit -- is not measuring what it says.

The discriminator is direct: for a duplicated (id, date), do the rows come from
different `source_file`s, and do their CGM values agree? Identical values from one
source is a duplicated export. Different values from different sources is a collision.

Row order matters for how this was missed: had the file been sorted by (id, date), the
copies would be adjacent and any groupby would have merged them into one oversized
group. They are not adjacent, which is why `build_cohort_multi.py` emitted the same day
twice from two different batches -- so the file is ordered by source first.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from cgmoutlier._env import check as _envcheck                      # noqa: E402
_envcheck()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default="data/raw/metabonet_public.parquet")
    ap.add_argument("--batch", type=int, default=4_000_000)
    ap.add_argument("--example", default="102")
    ap.add_argument("--out", default="results/channel_coverage/id_studies.csv")
    a = ap.parse_args()

    import pyarrow.parquet as pq
    import pyarrow.compute as pc
    pf = pq.ParquetFile(a.parquet)

    # 1. which ids appear under more than one source_file
    pairs = []
    for b in pf.iter_batches(batch_size=a.batch, columns=["id", "source_file"]):
        pairs.append(b.to_pandas().drop_duplicates())
    pairs = pd.concat(pairs).drop_duplicates()
    n_files = pairs.groupby("id")["source_file"].nunique().sort_values(ascending=False)
    print(f"{len(n_files):,} distinct ids over {pairs.source_file.nunique()} source files")
    print(f"ids appearing in more than one source file: "
          f"{int((n_files > 1).sum()):,} ({(n_files > 1).mean():.1%})")
    print(f"  distribution of files per id: {dict(n_files.value_counts().head())}")
    if (n_files > 1).any():
        print(f"  worst: {dict(n_files.head(8))}")
        for sid in n_files.head(3).index:
            print(f"    {sid}: {sorted(pairs[pairs.id == sid].source_file)}")
    pairs.to_csv(a.out, index=False)

    # 2. one duplicated day, read in full
    sid = a.example if a.example in set(pairs.id) else n_files.index[0]
    tab = pq.read_table(a.parquet, columns=["id", "date", "CGM", "basal", "bolus",
                                            "source_file", "cgm_device"],
                        filters=[("id", "=", sid)]).to_pandas()
    tab["day"] = tab["date"].values.astype("datetime64[D]")
    cnt = tab.groupby("day").size().sort_values(ascending=False)
    print(f"\n=== subject {sid}: {len(tab):,} rows, "
          f"{tab.source_file.nunique()} source file(s) {sorted(tab.source_file.unique())}")
    print(f"rows per day, largest: {dict(cnt.head(5))}")

    day = cnt.index[0]
    d = tab[tab.day == day].sort_values("date")
    print(f"\nday {day}: {len(d)} rows, {d.date.nunique()} distinct timestamps, "
          f"source files {sorted(d.source_file.unique())}")
    g = d.groupby("date")["CGM"]
    disagree = g.nunique(dropna=True)
    print(f"timestamps with more than one row:  {int((g.size() > 1).sum())}")
    print(f"  ... of which CGM values DISAGREE: {int((disagree > 1).sum())}")
    print(f"\nfirst 12 rows of that day:")
    print(d.head(12)[["date", "CGM", "basal", "bolus", "source_file",
                      "cgm_device"]].to_string(index=False))

    # 3. the same question across the whole file, cheaply: count rows and distinct
    #    timestamps per (id, day) for one channel, on a sample of row groups
    print(f"\n=== duplication rate over {min(12, pf.metadata.num_row_groups)} "
          f"sampled row groups ===")
    rg = np.linspace(0, pf.metadata.num_row_groups - 1,
                     min(12, pf.metadata.num_row_groups)).astype(int)
    s = pf.read_row_groups(list(rg), columns=["id", "date", "source_file"]).to_pandas()
    s["day"] = s["date"].values.astype("datetime64[D]")
    per = s.groupby(["id", "day"]).agg(rows=("date", "size"),
                                       stamps=("date", "nunique"),
                                       files=("source_file", "nunique"))
    dup = per[per.rows > per.stamps]
    print(f"{len(per):,} (id, day) pairs sampled; {len(dup):,} have repeated timestamps "
          f"({len(dup) / max(len(per), 1):.1%})")
    if len(dup):
        print(f"  repeat factor rows/stamps: "
              f"{dict((dup.rows / dup.stamps).round(2).value_counts().head())}")
        print(f"  of those, how many span >1 source file: "
              f"{int((dup.files > 1).sum()):,} ({(dup.files > 1).mean():.1%})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
