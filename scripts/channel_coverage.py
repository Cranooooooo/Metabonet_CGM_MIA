#!/usr/bin/env python
"""How many subjects survive if the cohort requires channel X as well as CGM?

    python scripts/channel_coverage.py --parquet data/raw/metabonet_public.parquet

This replaces stage B of `inspect_raw.py`. Three things were wrong with that version
and each of them changes the answer:

  * it made ONE PASS PER CHANNEL over 154M rows. Twelve passes at 0.8 of a core do not
    fit in a two-hour walltime. Here every channel is counted in a single pass --
    `groupby(id, day).sum()` over a boolean frame counts all channels at once, so extra
    channels are nearly free and the cost is one read of the file.
  * it counted a subject as usable once its total non-null samples reached
    min_days * 288. That is not the cohort rule. `cgmoutlier.data.cohort.complete_days`
    requires all 288 cells of a GIVEN DAY, then counts such days; a subject with 8640
    samples scattered over 200 half-empty days passes the old test and contributes zero
    windows. The 63% figure refers to the real rule, so this counts cells per (id, day)
    and thresholds per day.
  * its channel list came from a dtype test, which admits `age`, `height`, `weight` and
    `age_of_diagnosis` -- per-subject constants broadcast down every row, so they score
    as the densest "channels" in the file and mean nothing.

NON-NULL IS NOT THE SAME AS INFORMATIVE. `bolus` and `carbs` are 89% and 67% non-null,
which would make them look denser than CGM, but a bolus pump writes 0.0 at every
five-minute cell where nothing was delivered. A channel that is 99% zeros carries its
information in a handful of cells per day and behaves nothing like a continuous trace,
so the non-zero fraction is counted in the same pass and reported beside the non-null
one. Read the two together; either alone is misleading.
"""
import argparse
import sys

import pandas as pd

from cgmoutlier._env import check as _envcheck                      # noqa: E402
_envcheck()

T_PER_DAY = 288

# Time-varying numeric columns, named rather than inferred -- see the docstring. The
# demographics (age, height, weight, age_of_diagnosis) and the flags (is_test,
# subject_split_across_traintest) are numeric and constant within a subject.
CHANS = ["CGM", "basal", "bolus", "insulin", "carbs", "calories_burned",
         "heartrate", "steps", "skin_temp", "galvanic_skin_response", "air_temp",
         "workout_duration", "workout_intensity"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default="data/raw/metabonet_public.parquet")
    ap.add_argument("--min-days", type=int, default=30)
    ap.add_argument("--batch", type=int, default=2_000_000)
    ap.add_argument("--out", default="results/channel_coverage")
    ap.add_argument("--subject-key", choices=("id", "study_id"), default="id",
                    help="'id' is unique only WITHIN a source study -- 241 of 1,291 "
                         "ids appear under two or more, and those are different "
                         "people. 'study_id' is source_file + id, the correct key.")
    a = ap.parse_args()

    import pyarrow.parquet as pq
    from pathlib import Path
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    pf = pq.ParquetFile(a.parquet)
    names = set(pf.schema_arrow.names)
    chans = [c for c in CHANS if c in names]
    missing = [c for c in CHANS if c not in names]
    print(f"channels: {chans}")
    if missing:
        print(f"absent from this file: {missing}")
    nz = [c + "__nz" for c in chans]

    parts, studies, rows = [], [], 0
    for i, b in enumerate(pf.iter_batches(batch_size=a.batch,
                                          columns=["id", "date", "source_file"] + chans)):
        d = b.to_pandas()
        if a.subject_key == "study_id":
            d["id"] = d["source_file"].str.cat(d["id"], sep="/")
        day = d["date"].values.astype("datetime64[D]")
        f = d[chans].notna()
        f[nz] = d[chans].fillna(0).ne(0).values          # non-null AND not exactly zero
        f["id"], f["day"] = d["id"].astype("category"), day
        parts.append(f.groupby(["id", "day"], observed=True)[chans + nz].sum())
        studies.append(d[["id", "source_file"]].drop_duplicates())
        rows += len(d)
        if (i + 1) % 10 == 0:
            print(f"  {rows:,} rows | {i + 1} batches", flush=True)

    cells = pd.concat(parts).groupby(level=[0, 1], observed=True).sum()
    study = (pd.concat(studies).drop_duplicates("id").set_index("id")["source_file"])
    print(f"\n{rows:,} rows -> {len(cells):,} (subject, day) pairs, "
          f"{cells.index.get_level_values(0).nunique():,} subjects, "
          f"{study.nunique()} studies")

    # a day is usable for a channel when every one of its 288 cells carries a reading
    full = cells[chans] >= T_PER_DAY
    cgm = full["CGM"]
    cgm_subj = set(cgm.groupby(level=0, observed=True).sum().pipe(
        lambda s: s[s >= a.min_days].index))
    print(f"\nCGM alone: {len(cgm_subj)} subjects with >= {a.min_days} complete days "
          f"(the cohort rule; the shipped cohort has 875)")

    print(f"\n{'channel':22}{'non-null':>10}{'non-zero':>10}{'full days':>12}"
          f"{'subjects':>10}{'+CGM subj':>11}{'% of CGM':>10}")
    recs = []
    for c in chans:
        both = full[c] & cgm
        sub = set(full[c].groupby(level=0, observed=True).sum().pipe(
            lambda s: s[s >= a.min_days].index))
        bsub = set(both.groupby(level=0, observed=True).sum().pipe(
            lambda s: s[s >= a.min_days].index))
        nn = cells[c].sum() / (len(cells) * T_PER_DAY)
        nzf = cells[c + "__nz"].sum() / max(cells[c].sum(), 1)
        print(f"  {c:20}{nn:>10.3f}{nzf:>10.3f}{int(full[c].sum()):>12,}"
              f"{len(sub):>10}{len(bsub):>11}{len(bsub) / max(len(cgm_subj), 1):>10.1%}")
        recs.append(dict(channel=c, nonnull_frac=nn, nonzero_frac_of_nonnull=nzf,
                         full_days=int(full[c].sum()), subjects=len(sub),
                         subjects_with_cgm=len(bsub),
                         pct_of_cgm=len(bsub) / max(len(cgm_subj), 1)))
    pd.DataFrame(recs).to_csv(out / "by_channel.csv", index=False)

    # the combination the study would actually use, and the one the 63% refers to
    print("\ncombinations (subjects with >= "
          f"{a.min_days} days complete in ALL of the listed channels):")
    combos = [["CGM"], ["CGM", "basal"], ["CGM", "insulin"], ["CGM", "basal", "bolus"],
              ["CGM", "basal", "bolus", "carbs"],
              ["CGM", "basal", "bolus", "carbs", "insulin"]]
    crecs = []
    for cb in combos:
        cb = [c for c in cb if c in chans]
        m = full[cb].all(axis=1)
        n = int((m.groupby(level=0, observed=True).sum() >= a.min_days).sum())
        d = int(m.sum())
        print(f"  {'+'.join(cb):44} {n:>5} subjects  {d:>9,} days  "
              f"({n / max(len(cgm_subj), 1):.1%} of CGM-only)")
        crecs.append(dict(combo="+".join(cb), subjects=n, days=d,
                          pct_of_cgm=n / max(len(cgm_subj), 1)))
    pd.DataFrame(crecs).to_csv(out / "by_combo.csv", index=False)

    # which studies supply the multi-channel subjects -- if it is one study, a
    # multi-channel cohort is that study's cohort and the comparison is confounded
    key = [c for c in ("CGM", "basal", "bolus") if c in chans]
    m = full[key].all(axis=1)
    multi = set(m.groupby(level=0, observed=True).sum().pipe(
        lambda s: s[s >= a.min_days].index))
    tab = pd.DataFrame({"cgm_only": pd.Series({s: sum(study.get(i) == s for i in cgm_subj)
                                               for s in study.unique()}),
                        "multi": pd.Series({s: sum(study.get(i) == s for i in multi)
                                            for s in study.unique()})})
    tab = tab[tab.sum(axis=1) > 0].sort_values("cgm_only", ascending=False)
    print(f"\nsubjects by study ({'+'.join(key)} vs CGM alone):")
    print(tab.to_string())
    tab.to_csv(out / "by_study.csv")

    # per-day bar sensitivity: 288/288 is strict, and it may be the bar rather than the
    # channel that removes the subjects
    print(f"\nsensitivity to the per-day completeness bar "
          f"({'+'.join(key)}, >= {a.min_days} days):")
    for bar in (1.0, 0.9, 0.8, 0.7):
        f2 = cells[chans] >= int(bar * T_PER_DAY)
        n = int((f2[key].all(axis=1).groupby(level=0, observed=True).sum()
                 >= a.min_days).sum())
        nc = int((f2["CGM"].groupby(level=0, observed=True).sum() >= a.min_days).sum())
        print(f"  {bar:>5.0%} of a day: {n:>5} multi-channel   {nc:>5} CGM-only")

    cells.to_parquet(out / "cells_per_day.parquet")
    print(f"\nwrote {out}/ (by_channel, by_combo, by_study, cells_per_day)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
