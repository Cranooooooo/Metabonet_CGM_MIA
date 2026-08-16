#!/usr/bin/env python
"""Parquet -> multi-channel cohort, on the same rules as the single-channel one.

    python scripts/build_cohort_multi.py --channels CGM basal bolus \
        --out data/cohort/metabonet834_c3

WHY THIS EXISTS. `cgmoutlier.data.cohort` is single-channel by a decision recorded in
its docstring: requiring the insulin channels "gate[s] out 63% of the subjects that
have usable CGM". `scripts/channel_coverage.py` measured that on the released file and
it does not reproduce -- CGM alone gives 875 subjects (exactly the shipped cohort) and
CGM+basal+bolus gives 834, a loss of 4.7%, not 63%. The premise that made this study
single-channel is therefore not true of this data, and a day of CGM alone is the
leading suspect for why no attack separates outliers from normals.

WHAT IS DELIBERATELY THE SAME as the single-channel builder, so the two cohorts differ
in channel count and nothing else:

  * a (subject, day) is kept only when all 288 five-minute cells are present -- now in
    EVERY requested channel, which is the only rule that changes
  * subjects need >= min_days such days
  * the subject draw is stratified on day count, since day count drives a subject's
    nearest-neighbour distance independently of membership
  * population z-score, clip to +-5, divide by 5

WHAT IS NECESSARILY DIFFERENT: the normalisation constants are PER CHANNEL and are
written into the manifest as a list. Applying CGM's mean and sd to basal, or reusing
the single-channel constants here, leaves every clinical metric wrong but plausible --
the failure `docs/DATA.md` records as having happened once already. `to_mgdl` in the
single-channel module reads scalar `mean_mgdl`/`sd_mgdl` fields and would silently
mis-scale this manifest, so those fields are absent here rather than set to the CGM
values: a KeyError is the intended behaviour.

BOLUS AND CARBS ARE EVENT CHANNELS, NOT TRACES. Measured non-zero fractions among
non-null cells: basal 0.751, insulin 0.767, bolus 0.066, carbs 0.016. A z-scored bolus
channel is 93% one constant value and 7% spikes. That is a legitimate channel to attack
-- the timing of a bolus is far more idiosyncratic than a homeostatically bounded
glucose curve -- but it is not something a Gaussian-noise diffusion model reconstructs
the way it does a smooth trace, and any generator trained on it must be checked against
that rather than against an aggregate reconstruction error.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from cgmoutlier._env import check as _envcheck                      # noqa: E402
_envcheck()

from cgmoutlier.data.cohort import DEMO_COLS, T, DT_MIN, ZCLIP, stratified_draw


def eligible(cells: pd.DataFrame, channels: list[str], min_days: int):
    """(subject -> n_days, set of eligible (id, day)) under the all-288-cells rule.

    Days come back as int32 days-since-epoch, matching how the second pass keys them:
    150M pandas Timestamps is 1.2 GB of nothing.
    """
    full = (cells[channels] >= T).all(axis=1)
    full = full[full]
    per = full.groupby(level=0, observed=True).size()
    per = per[per >= min_days]
    keep = full.index[full.index.get_level_values(0).isin(per.index)]
    days = keep.get_level_values(1).values.astype("datetime64[D]").astype(np.int32)
    return per, set(zip(keep.get_level_values(0).astype(str), days.tolist()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default="data/raw/metabonet_public.parquet")
    ap.add_argument("--subject-key", choices=("id", "study_id"), default="study_id",
                    help="'id' is unique only WITHIN a source study -- 241 of 1,291 "
                         "ids appear under two or more and those are different people "
                         "(docs/PITFALLS.md 13). 'study_id' is source_file + id.")
    ap.add_argument("--cells", default=None,
                    help="per-(subject, day) cell counts from scripts/channel_coverage.py; "
                         "defaults to the file matching --subject-key")
    ap.add_argument("--channels", nargs="+", default=["CGM", "basal", "bolus"])
    ap.add_argument("--out", default="data/cohort/metabonet834_c3")
    ap.add_argument("--min-days", type=int, default=30)
    ap.add_argument("--n-subjects", type=int, default=None)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--batch", type=int, default=4_000_000)
    a = ap.parse_args()

    import pyarrow.parquet as pq
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    chans = list(a.channels)

    # the expensive pass over 154M rows was already made; reuse its cell counts
    if a.cells is None:
        a.cells = ("results/channel_coverage_studyid/cells_per_day.parquet"
                   if a.subject_key == "study_id"
                   else "results/channel_coverage/cells_per_day.parquet")
    print(f"[cohort] subject key = {a.subject_key}, cell counts from {a.cells}")
    cells = pd.read_parquet(a.cells)
    per, ok = eligible(cells, chans, a.min_days)
    print(f"[cohort] {'+'.join(chans)}: {len(ok):,} complete days over {per.size} "
          f"subjects with >= {a.min_days} days")
    picked = (stratified_draw(per, a.n_subjects, a.seed) if a.n_subjects
              else np.sort(per.index.to_numpy()))
    picked_set = set(picked)
    print(f"[cohort] taking {len(picked)} subjects")

    # Second pass. NO ASSUMPTION IS MADE ABOUT ROW ORDER. An earlier version streamed
    # and carried the last (subject, day) across the batch boundary, on the assumption
    # that the file is in (id, date) order. It is not -- a subject's rows for one day
    # turn up in several distant blocks, and that version emitted 5,919 days TWICE. A
    # duplicated window inflates a subject's day count, which drives its
    # nearest-neighbour distance independently of membership: the exact confound the
    # stratified draw exists to remove. So the picked subjects' rows are gathered in
    # full and grouped once, globally.
    #
    # A day is 288 five-minute slots. Some (subject, day) groups hold up to 1,152 rows,
    # because two studies' copies of a person's id both report. Projecting onto the
    # canonical grid handles that without having to know why: take each timestamp's
    # slot index, keep the first row per slot, require all 288 slots filled. Off-grid
    # timestamps are dropped and the day then fails completeness unless the grid is
    # genuinely covered. Slot and day are computed per batch so that `date` is never
    # carried for 150M rows.
    demo_have = [c for c in DEMO_COLS if c in set(pq.read_schema(a.parquet).names)]
    src = ["source_file"] if a.subject_key == "study_id" else []
    idtype = pd.CategoricalDtype(categories=sorted(picked_set))
    keep, demos, seen, off = [], [], 0, 0
    for b in pq.ParquetFile(a.parquet).iter_batches(
            batch_size=a.batch, columns=["id", "date"] + src + chans + demo_have):
        d = b.to_pandas()
        if src:
            d["id"] = d["source_file"].str.cat(d["id"], sep="/")
        d = d[d["id"].isin(picked_set)]
        seen += len(d)
        if d.empty:
            continue
        demos.append(d[["id"] + demo_have].drop_duplicates("id"))
        day = d["date"].values.astype("datetime64[D]")
        tod = ((d["date"].values - day).astype("timedelta64[s]").astype(np.int64))
        ongrid = (tod % (60 * DT_MIN)) == 0
        off += int((~ongrid).sum())
        # one shared category dtype: per-batch categories differ, and concat of
        # mismatched categoricals silently falls back to object -- 150M python strings
        k = pd.DataFrame({"id": pd.Categorical(d["id"], dtype=idtype),
                          "day": day.astype("datetime64[D]").astype(np.int32),
                          "slot": (tod // (60 * DT_MIN)).astype(np.int16)})
        for c in chans:
            k[c] = d[c].to_numpy(np.float32)
        keep.append(k[ongrid])
    rows = pd.concat(keep, ignore_index=True)
    del keep
    print(f"[cohort] {seen:,} rows for picked subjects gathered; {off:,} "
          f"({off / max(seen, 1):.2%}) off the {DT_MIN}-minute grid and dropped")
    rows = rows.drop_duplicates(["id", "day", "slot"], keep="first")

    wins, sids, days = [], [], []
    incomplete = 0
    for (sid, day), g in rows.groupby(["id", "day"], sort=True, observed=True):
        if (sid, day) not in ok:
            continue
        if len(g) != T:
            incomplete += 1          # eligible by raw cell count, not by grid coverage
            continue
        v = g.sort_values("slot")[chans].to_numpy(np.float32)
        if not np.isfinite(v).all():
            incomplete += 1
            continue
        wins.append(v); sids.append(sid); days.append(day)

    X = np.stack(wins)
    sids = np.asarray(sids)
    n_elig = sum(1 for s, _ in ok if s in picked_set)
    print(f"[cohort] {n_elig:,} days eligible by raw cell count -> {X.shape} windows; "
          f"{incomplete:,} did not cover the 288-slot grid")
    assert len(set(zip(sids.tolist(), [str(d) for d in days]))) == len(sids), \
        "a (subject, day) was emitted twice"

    demo = {r["id"]: {c: r[c] for c in demo_have}
            for _, r in pd.concat(demos).drop_duplicates("id").iterrows()}

    # Per-channel constants -- never shared across channels, never reused across cohorts.
    #
    # dtype=np.float64 IS LOAD-BEARING. Reducing a 2-D float32 array along axis 0 does
    # not use pairwise summation: numpy accumulates row by row into a float32 buffer. At
    # 49.6M cells of about 145 the running sum reaches 7.2e9, where the float32 spacing
    # is 512, so adding another 145 changes nothing and the sum stalls. That reported
    # CGM mean 96.05 / sd 71.19 for this cohort against a true 145.94 / 57.65, while
    # basal and bolus -- whose sums stay near 3.5e6 -- came out correct. The
    # single-channel path escaped it because an (N, 1) reduction is dispatched as a
    # contiguous 1-D pairwise sum.
    flat = X.reshape(-1, X.shape[-1])
    mu = flat.mean(axis=0, dtype=np.float64)
    sd = flat.std(axis=0, dtype=np.float64)
    chk = np.median(X.reshape(-1, X.shape[-1]), axis=0)
    for i, c in enumerate(chans):
        if abs(mu[i] - chk[i]) > 5 * sd[i]:
            raise SystemExit(f"channel {c}: mean {mu[i]:.4f} is {abs(mu[i] - chk[i]) / max(sd[i], 1e-9):.1f} "
                             f"sd from the median {chk[i]:.4f}; the constants are wrong")
    Xn = (np.clip((X - mu) / sd, -ZCLIP, ZCLIP) / ZCLIP).astype(np.float32)
    for i, c in enumerate(chans):
        nzf = float((X[..., i] != 0).mean())
        print(f"[cohort] {c:10} mean={mu[i]:10.4f} sd={sd[i]:10.4f} nonzero={nzf:.3f}")

    np.save(out / "windows.npy", Xn)
    np.save(out / "subject_ids.npy", sids)
    np.save(out / "days.npy",
            np.asarray(days, dtype=np.int32).astype("datetime64[D]"))
    dm = pd.DataFrame.from_dict(demo, orient="index").rename_axis("id").reset_index()
    dm["n_days"] = dm["id"].map(per)
    dm.to_parquet(out / "subjects.parquet")

    man = dict(source=str(a.parquet), channels=chans, C=len(chans), T=T, dt_min=DT_MIN,
               zclip=ZCLIP, n_subjects=int(len(picked)), n_windows=int(X.shape[0]),
               min_days=a.min_days, seed=a.seed, stratified_on="n_days",
               channel_constants=[dict(name=c, mean=float(mu[i]), sd=float(sd[i]),
                                       nonzero_frac=float((X[..., i] != 0).mean()))
                                  for i, c in enumerate(chans)],
               denorm_formula="raw[:, :, i] = x * zclip * sd[i] + mean[i]",
               note=("Per-channel constants. There are deliberately no scalar "
                     "mean_mgdl/sd_mgdl fields: cohort.to_mgdl would apply CGM's "
                     "constants to every channel, so it must fail loudly here."))
    (out / "manifest.json").write_text(json.dumps(man, indent=2))
    print(f"[cohort] wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
