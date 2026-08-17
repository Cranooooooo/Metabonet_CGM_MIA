#!/usr/bin/env python
"""Is a multi-study `id` one person, or several?

The MetaboNet data dictionary says `id` is the "Unique identifier for the subject" and
notes "Consolidation may occur when duplicates found across studies". Read one way that
means the maintainers merged the same person's records from several studies under one
id, and `id` is already the person. Read the other way it means ids collide.

The distinction decides whether re-keying the cohort on (source_file, id) -- as this
repository now does -- is a correction or a mistake. If a multi-study id is one person,
re-keying SPLITS that person into several "subjects", and a membership design can then
put half of someone in the outlier arm and half in the background.

The evidence that suggested a collision was subject 102: two CGM devices reporting
different values at the same minute. But the dictionary also says timestamps are
localized per study and `docs/DATA.md` records that they are de-identified and SHIFTED.
One person in two studies, each shifted independently, produces exactly that overlap.

Demographics settle it. Age, gender, ethnicity and age_of_diagnosis are static per
subject. If the five source files under one id report the same person, they agree.
"""
import sys
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from cgmoutlier._env import check as _envcheck                      # noqa: E402
_envcheck()

COLS = ["gender", "ethnicity", "age_of_diagnosis", "age", "height"]
pf = pq.ParquetFile("data/raw/metabonet_public.parquet")
parts = []
for b in pf.iter_batches(batch_size=4_000_000, columns=["id", "source_file"] + COLS):
    d = b.to_pandas()
    parts.append(d.drop_duplicates(["id", "source_file"]))
g = (pd.concat(parts).drop_duplicates(["id", "source_file"])
       .sort_values(["id", "source_file"]))
nstudy = g.groupby("id")["source_file"].nunique()
multi = nstudy[nstudy > 1].index
print(f"{len(nstudy):,} ids, {len(multi):,} appear in more than one source_file\n")

def agrees(series):
    v = series.dropna().unique()
    return len(v) <= 1

rows = []
for sid in multi:
    d = g[g["id"] == sid]
    rec = {"id": sid, "n_studies": d["source_file"].nunique()}
    for c in COLS:
        rec[c] = agrees(d[c])
    rows.append(rec)
r = pd.DataFrame(rows)

print("For ids spanning several studies, do the static demographics AGREE?")
print(f"{'field':>18}{'agree':>8}{'disagree':>10}{'% agree':>9}")
for c in COLS:
    a = int(r[c].sum()); n = len(r)
    print(f"{c:>18}{a:>8}{n-a:>10}{a/n:>8.0%}")

allagree = r[COLS].all(axis=1)
print(f"\nall five fields agree: {int(allagree.sum())} of {len(r)} "
      f"({allagree.mean():.0%})")
print("\nIf these are the same person consolidated across studies, agreement should be")
print("near total. Widespread disagreement means the ids collide and (source_file, id)")
print("is the right key.")

print("\n=== the example that started this: id 102 ===")
d = g[g["id"] == "102"][["source_file"] + COLS]
print(d.to_string(index=False))

print("\n=== five more multi-study ids ===")
for sid in list(multi)[:5]:
    d = g[g["id"] == sid][["source_file"] + COLS]
    if len(d) > 1:
        print(f"\nid {sid}:")
        print(d.to_string(index=False))
