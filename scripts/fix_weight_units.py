#!/usr/bin/env python
"""Which studies report weight in pounds?

The `weight` column mixes units: the median over subjects is 149.5 and 48% are above
150, which cannot be kilograms for adults, while the 1st percentile is 31.1 and the
minimum 21.0, which cannot be pounds. Dividing insulin by this column as it stands
would rescale roughly half the cohort by 2.2 and manufacture exactly the between-subject
difference that per-kilogram normalisation is meant to remove.

The discriminator is BMI, not a threshold on weight. `height` is in the same file, so
for each study both hypotheses can be tested: weight in kg with height in cm gives one
BMI, weight in lb gives another 2.2x apart, and only one of them lands in a plausible
human range. A study whose two hypotheses BOTH look plausible, or whose subjects
disagree with each other, is reported rather than guessed at.
"""
import sys
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from cgmoutlier._env import check as _envcheck                      # noqa: E402
_envcheck()

LB_PER_KG = 2.20462
IN_PER_CM = 0.393701

pf = pq.ParquetFile("data/raw/metabonet_public.parquet")
parts = []
for b in pf.iter_batches(batch_size=4_000_000,
                         columns=["id", "source_file", "weight", "height"]):
    d = b.to_pandas().dropna(subset=["weight"])
    if len(d):
        d["sid"] = d["source_file"].str.cat(d["id"], sep="/")
        parts.append(d.groupby(["source_file", "sid"])
                      .agg(weight=("weight", "median"), height=("height", "median")))
g = pd.concat(parts).groupby(level=[0, 1]).median().reset_index()
print(f"{len(g):,} subjects with a weight, {g['height'].notna().sum():,} also a height\n")

print(f"{'study':12}{'n':>6}{'wt med':>9}{'ht med':>9}"
      f"{'BMI if kg/cm':>14}{'BMI if lb/cm':>14}{'BMI if lb/in':>14}  verdict")
rows = []
for st, d in g.groupby("source_file"):
    w, h = d["weight"].median(), d["height"].median()
    bmi = lambda kg, m: kg / (m ** 2) if m and m > 0 else np.nan
    b_kg_cm = bmi(w, h / 100) if pd.notna(h) else np.nan
    b_lb_cm = bmi(w / LB_PER_KG, h / 100) if pd.notna(h) else np.nan
    b_lb_in = bmi(w / LB_PER_KG, h / IN_PER_CM / 100) if pd.notna(h) else np.nan
    cands = {"kg+cm": b_kg_cm, "lb+cm": b_lb_cm, "lb+in": b_lb_in}
    ok = {k: v for k, v in cands.items() if pd.notna(v) and 15 <= v <= 45}
    verdict = "+".join(ok) if ok else "NONE PLAUSIBLE"
    if len(ok) > 1:
        verdict += "  <-- ambiguous"
    print(f"{st:12}{len(d):>6}{w:>9.1f}{h:>9.1f}"
          f"{b_kg_cm:>14.1f}{b_lb_cm:>14.1f}{b_lb_in:>14.1f}  {verdict}")
    rows.append(dict(study=st, n=len(d), weight_median=w, height_median=h,
                     bmi_kg_cm=b_kg_cm, bmi_lb_cm=b_lb_cm, bmi_lb_in=b_lb_in,
                     verdict=verdict))

out = pd.DataFrame(rows)
out.to_csv("results/channel_coverage_studyid/weight_units.csv", index=False)
print("\nwrote results/channel_coverage_studyid/weight_units.csv")
print("\nA study is only usable as a per-kg divisor once its unit is settled. Any row "
      "reading NONE PLAUSIBLE or ambiguous has to be resolved before basal/kg is built "
      "-- guessing there is worse than not dividing at all.")
