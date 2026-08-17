#!/usr/bin/env python
"""Is `weight` usable as a per-subject divisor for the insulin channels?

Insulin is dosed per kilogram, so basal and bolus belong in U/hr/kg rather than U/hr.
Without that division the channel carries body size, which is a stable personal
attribute and would inflate any identifiability measurement for a reason that has
nothing to do with a subject's insulin regimen.

Three things have to hold before the division is safe:
  * every subject in the cohort has a weight
  * it is CONSTANT within a subject, or the choice of which value to use has to be made
  * it is in kilograms, not pounds -- a mixed-unit column would rescale a subset of
    subjects by 2.2 and manufacture exactly the separation this is meant to remove
"""
import sys
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from cgmoutlier._env import check as _envcheck                      # noqa: E402
_envcheck()

pf = pq.ParquetFile("data/raw/metabonet_public.parquet")
parts = []
for b in pf.iter_batches(batch_size=4_000_000, columns=["id", "source_file", "weight"]):
    d = b.to_pandas().dropna(subset=["weight"])
    if len(d):
        d["sid"] = d["source_file"].str.cat(d["id"], sep="/")
        parts.append(d.groupby("sid")["weight"].agg(["min", "max", "count"]))
g = pd.concat(parts).groupby(level=0).agg(min=("min", "min"), max=("max", "max"),
                                          count=("count", "sum"))
print(f"subjects with any weight: {len(g):,}")
varying = g[g["max"] - g["min"] > 1e-6]
print(f"  weight varies within subject: {len(varying):,} ({len(varying)/len(g):.1%})")
if len(varying):
    rel = ((varying["max"] - varying["min"]) / varying["min"])
    print(f"    relative spread: median {rel.median():.3%}, p90 {rel.quantile(.9):.3%}, "
          f"max {rel.max():.3%}")

print(f"\nweight distribution (kg if these look like adults):")
for q in (0.01, 0.25, 0.5, 0.75, 0.99):
    print(f"  q{q:<5} {g['min'].quantile(q):8.1f}")
print(f"  min {g['min'].min():.1f}   max {g['max'].max():.1f}")
lb = g[g["min"] > 150]
print(f"subjects over 150 (would be pounds, not kg): {len(lb)}")

for coh in ("data/cohort/metabonet_sid_c3", "data/cohort/metabonet_sid_c1"):
    try:
        s = np.load(f"{coh}/subject_ids.npy", allow_pickle=True).astype(str)
    except FileNotFoundError:
        continue
    subs = set(s.tolist())
    have = subs & set(g.index)
    print(f"\n{coh}: {len(subs)} subjects, {len(have)} have a weight, "
          f"{len(subs - set(g.index))} do NOT")
    miss = sorted(subs - set(g.index))[:8]
    if miss:
        print(f"  missing: {miss}")
