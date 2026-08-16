#!/usr/bin/env python
"""One window = L days instead of one day.

    python scripts/build_cohort_multiday.py --days 7 --mode contiguous
    python scripts/build_cohort_multiday.py --days 14 --mode concat

Reassembles an existing single-day cohort rather than re-reading the 154M-row parquet:
`metabonet_sid_c1` already holds 197,970 complete days with their subject and date, and
an L-day window is L of those laid end to end.

TWO CONSTRUCTIONS, AND THE DIFFERENCE MATTERS
---------------------------------------------
`contiguous`  L days adjacent in the CALENDAR. This is a real week of a person's life,
              and it is what a clinical claim about "15 days of data" means. It is also
              scarce: measured on this cohort, 809 subjects have an unbroken week, 40
              have a fortnight, 11 have three weeks. Above a week it cannot support a
              design.

`concat`      the subject's complete days in date order, chopped into blocks of L,
              skipping gaps. Every subject with >= 30 complete days yields windows at
              any L <= 30, so all three lengths are available -- but a block spanning a
              gap has a fabricated transition at that seam, and it is not a week.

For a copy_paste ceiling measurement the seams are arguably irrelevant: copy_paste
reproduces training windows verbatim, so whether a window is memorised does not depend
on its internal transitions being physiological. That is an argument, not a
measurement, which is why L=7 is built BOTH ways -- it is the only length where both
constructions exist, so it is the control that says whether `concat` can be trusted at
14 and 21.

Windows are NON-OVERLAPPING. Sliding at stride 1 would multiply the count by L but the
neighbours share L-1 days; the effective sample size is the non-overlapping count
however many windows are cut, and a bootstrap-based generator would then see near
duplicates of held-out data.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from cgmoutlier._env import check as _envcheck                      # noqa: E402
_envcheck()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/cohort/metabonet_sid_c1")
    ap.add_argument("--days", type=int, required=True)
    ap.add_argument("--mode", choices=("contiguous", "concat"), required=True)
    ap.add_argument("--min-windows", type=int, default=4,
                    help="drop subjects with fewer than this many L-day windows; a "
                         "per-subject AUC over two windows is not a measurement")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    src = Path(a.src)
    out = Path(a.out or f"data/cohort/{src.name}_d{a.days}_{a.mode}")
    out.mkdir(parents=True, exist_ok=True)

    X = np.load(src / "windows.npy", mmap_mode="r")
    sids = np.load(src / "subject_ids.npy", allow_pickle=True).astype(str)
    days = np.load(src / "days.npy", allow_pickle=True).astype("datetime64[D]").astype(np.int64)
    man = json.loads((src / "manifest.json").read_text())
    T, C = X.shape[1], X.shape[2]
    print(f"[multiday] source {X.shape}, {len(set(sids))} subjects")

    wins, wsid, wstart, seams = [], [], [], []
    for s in sorted(set(sids)):
        idx = np.flatnonzero(sids == s)
        d = days[idx]
        o = np.argsort(d, kind="stable")
        idx, d = idx[o], d[o]

        if a.mode == "contiguous":
            # split into maximal runs of adjacent calendar days, then chop each run
            brk = np.flatnonzero(np.diff(d) != 1) + 1
            blocks = np.split(np.arange(len(idx)), brk)
        else:
            blocks = [np.arange(len(idx))]

        for b in blocks:
            for k in range(0, len(b) - a.days + 1, a.days):
                sel = idx[b[k:k + a.days]]
                dd = d[b[k:k + a.days]]
                wins.append(np.asarray(X[sel]).reshape(a.days * T, C))
                wsid.append(s)
                wstart.append(dd[0])
                seams.append(int((np.diff(dd) != 1).sum()))

    wsid = np.asarray(wsid)
    keep_s = {s for s in set(wsid.tolist()) if (wsid == s).sum() >= a.min_windows}
    m = np.array([s in keep_s for s in wsid])
    Xw = np.stack([w for w, k in zip(wins, m) if k]).astype(np.float32)
    wsid, wstart = wsid[m], np.asarray(wstart)[m]
    seams = np.asarray(seams)[m]

    print(f"[multiday] {a.mode}, L={a.days}: {Xw.shape[0]:,} windows over "
          f"{len(keep_s)} subjects (>= {a.min_windows} windows each)")
    print(f"[multiday] window shape {Xw.shape[1:]}  = {a.days} x {T} x {C}")
    print(f"[multiday] windows per subject: min {min((wsid==s).sum() for s in keep_s)}, "
          f"median {int(np.median([(wsid==s).sum() for s in keep_s]))}, "
          f"max {max((wsid==s).sum() for s in keep_s)}")
    print(f"[multiday] seams (gaps inside a window): {seams.sum():,} total, "
          f"{(seams > 0).mean():.1%} of windows have at least one")
    if a.mode == "contiguous" and seams.sum():
        sys.exit("contiguous mode produced a seam; that is a bug")

    np.save(out / "windows.npy", Xw)
    np.save(out / "subject_ids.npy", wsid)
    np.save(out / "day_start.npy", wstart)
    np.save(out / "seams.npy", seams)
    m2 = dict(man)
    m2.update(source=str(src), days_per_window=a.days, mode=a.mode, T=int(Xw.shape[1]),
              C=int(C), dt_min=man.get("dt_min", 5), n_subjects=len(keep_s),
              n_windows=int(Xw.shape[0]), min_windows_per_subject=a.min_windows,
              seams_total=int(seams.sum()),
              note=("An L-day window. Constants are inherited from the source cohort "
                    "unchanged -- this is a regrouping of the same normalised values, "
                    "not a renormalisation, so a metric computed here is on the same "
                    "scale as one computed there."))
    (out / "manifest.json").write_text(json.dumps(m2, indent=2))
    print(f"[multiday] wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
