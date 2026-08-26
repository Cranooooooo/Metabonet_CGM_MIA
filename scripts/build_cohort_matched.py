#!/usr/bin/env python
"""Build a ONE-DAY cohort that matches the seven-day pilot's training regime exactly.

WHY THIS EXISTS
---------------
The seven-day h256 pilot returned arm AUC 0.950 against the published one-day 0.680.
Two things differ between them and only one is the hypothesis:

    window length     288 timesteps      ->  2016            <- the hypothesis
    training regime   36.3 epochs        ->  201.1           <- a confound

The seven-day cohort holds 7,001 windows because a week needs seven consecutive
complete days. 22,000 steps at batch 64 over 7,001 windows is 201 epochs; the published
one-day run was 100,000 steps over 176,445 windows, which is 36. A generator that has
been round its training set 5.5x more often memorises more for reasons that have nothing
to do with how long a window is.

WHAT THIS BUILDS
----------------
The same 573 subjects the seven-day pilot uses, their ONE-DAY windows, subsampled to the
same 7,001 total. Train it at the same 22,000 steps and the same width, and the only
thing left that differs from the seven-day run is T.

    one-day matched   7,001 windows, T=288,  22,000 steps -> 201 epochs
    seven-day h256    7,001 windows, T=2016, 22,000 steps -> 201 epochs

The subsample is proportional to each subject's day count and takes a contiguous block
from each, so a subject keeps roughly the share of the cohort it had before. Every
subject the design names is guaranteed at least one window -- a target with no windows
would silently drop out of the arm and unbalance the pairing.

    python scripts/build_cohort_matched.py \
        --src data/cohort/metabonet_sid_c1 \
        --design results/design_d7_pilot/rep1 \
        --target-windows 7001 \
        --out data/cohort/metabonet_sid_d1_matched
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="the full one-day cohort to draw from")
    ap.add_argument("--design", default=None,
                    help="the design whose subjects must all survive")
    ap.add_argument("--target-windows", type=int, default=None,
                    help="subsample to this many windows; omit to keep every window of "
                         "every kept subject, which is what a matrix cell wants")
    ap.add_argument("--subjects", default=None,
                    help="newline-delimited subject ids to keep, instead of --design")
    ap.add_argument("--match-blocks", default=None, metavar="CSV",
                    help="keep exactly the (subject,day) pairs listed in CSV. A one-day "
                         "cohort matches on days.npy, a seven-day one on day_start.npy, "
                         "so both land on the same count AND the same calendar days. "
                         "This is what a 2x2 matrix needs: equal windows per subject "
                         "gives equal epochs, and equal per-subject counts make the "
                         "day-count control matching identical in every cell -- so the "
                         "cells differ on T and C and nothing else. Note what it cannot "
                         "fix: a seven-day window holds seven days of data and a one-day "
                         "window holds one, so the seven-day cells necessarily see 7x the "
                         "raw signal. That is not a confound to remove, it is what a "
                         "longer window IS.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing cohort directory")
    a = ap.parse_args()

    src, out = Path(a.src), Path(a.out)
    X = np.load(src / "windows.npy", mmap_mode="r")
    sids = np.load(src / "subject_ids.npy", allow_pickle=True)
    man = json.loads((src / "manifest.json").read_text())

    # A one-day cohort carries days.npy; a multi-day one carries day_start.npy and
    # seams.npy instead, and no subjects.parquet. Rather than name them, take every
    # sidecar array whose first axis is the window axis and subset it alongside the
    # windows. Getting this wrong silently mis-aligns a window with its subject.
    side = {}
    for f in sorted(src.glob("*.npy")):
        if f.name in ("windows.npy", "subject_ids.npy"):
            continue
        arr = np.load(f, allow_pickle=True)
        if arr.shape and arr.shape[0] == X.shape[0]:
            side[f.name] = arr
        else:
            print(f"[matched] {f.name} is not per-window ({arr.shape}); not carried")
    # date order for the contiguous block. A one-day cohort keys it on days.npy; a
    # multi-day one has no days.npy and keys on day_start.npy. Reading only days.npy
    # left multi-day cohorts silently falling back to array order.
    days = side.get("days.npy")
    if days is None:
        days = side.get("day_start.npy")
    print(f"[matched] carrying {len(side)} per-window sidecar(s): "
          f"{', '.join(side) if side else 'none'}")

    if a.subjects:
        keep = {l.strip() for l in Path(a.subjects).read_text().splitlines() if l.strip()}
        print(f"[matched] keeping {len(keep)} subjects from {a.subjects}")
    else:
        design = json.loads((Path(a.design) / "design.json").read_text())
        keep = set()
        for k in ("outliers", "controls"):
            keep.update(map(str, design.get(k, [])))
        base = json.loads((Path(a.design) / "jobs" / "base.json").read_text())
        keep.update(map(str, base["subjects"]))
        print(f"[matched] design names {len(keep)} subjects")

    have = set(map(str, np.unique(sids)))
    missing = sorted(keep - have)
    if missing:
        sys.exit(f"[matched] {len(missing)} design subjects are absent from {src}: "
                 f"{missing[:5]} — the comparison would not be like for like")

    if a.match_blocks:
        import csv as _csv
        with open(a.match_blocks) as fh:
            want = {(r["subject"], int(r["day"])) for r in _csv.DictReader(fh)}
        if days is None:
            sys.exit(f"[matched] {src} carries neither days.npy nor day_start.npy, so "
                     f"its windows cannot be matched to a (subject,day) reference")
        key = np.array([(str(a_), int(b_)) for a_, b_ in
                        zip(sids.astype(str), days.astype("int64"))], dtype=object)
        sel = np.sort(np.flatnonzero([tuple(k) in want for k in key]))
        got = {tuple(k) for k in key[sel]}
        lost = want - got
        if lost:
            sys.exit(f"[matched] {len(lost)} of {len(want)} reference blocks are absent "
                     f"from {src} (e.g. {sorted(lost)[:3]}). The cells would not hold the "
                     f"same days, which is the whole point of matching them.")
        print(f"[matched] block-matched to {a.match_blocks}: {len(sel):,} windows over "
              f"{len(np.unique(sids[sel]))} subjects")
        write(out, src, X, sids, side, man, sel, a,
              how=f"block-matched to {a.match_blocks}")
        return 0

    if a.target_windows is None:
        sel = np.sort(np.flatnonzero(np.isin(sids, list(keep))))
        print(f"[matched] keeping every window: {len(sel):,} over "
              f"{len(np.unique(sids[sel]))} subjects")
        write(out, src, X, sids, side, man, sel, a, how="all windows of the kept subjects")
        return 0

    rng = np.random.default_rng(a.seed)
    idx_by_subject = {}
    for s in sorted(keep):
        w = np.flatnonzero(sids == s)
        if days is not None:
            w = w[np.argsort(days[w])]       # date order, so a block is contiguous in time
        idx_by_subject[s] = w

    total = sum(len(v) for v in idx_by_subject.values())
    frac = a.target_windows / total
    print(f"[matched] those subjects hold {total:,} one-day windows; "
          f"keeping {frac:.3%} to reach {a.target_windows:,}")

    chosen = []
    for s, w in idx_by_subject.items():
        n = max(1, int(round(len(w) * frac)))      # never zero: a target must survive
        start = 0 if n >= len(w) else int(rng.integers(0, len(w) - n + 1))
        chosen.append(w[start:start + n])
    sel = np.sort(np.concatenate(chosen))

    # trim or top up to land exactly on the target, without emptying anybody
    if len(sel) > a.target_windows:
        counts = {s: 0 for s in idx_by_subject}
        for i in sel:
            counts[str(sids[i])] += 1
        drop, pool = len(sel) - a.target_windows, list(sel)
        rng.shuffle(pool)
        removed = set()
        for i in pool:
            if drop == 0:
                break
            s = str(sids[i])
            if counts[s] > 1:
                counts[s] -= 1
                removed.add(int(i))
                drop -= 1
        sel = np.array(sorted(set(map(int, sel)) - removed))
    # the docstring promised "trim or top up"; only trim was ever implemented, so a
    # per-subject rounding shortfall landed silently under target (6,996 for 7,001 once)
    if len(sel) < a.target_windows:
        print(f"[matched] WARNING: {len(sel):,} windows, {a.target_windows - len(sel)} "
              f"short of the {a.target_windows:,} target -- per-subject rounding cannot "
              f"be topped up without breaking the contiguous block; reporting, not hiding")
    print(f"[matched] selected {len(sel):,} windows over "
          f"{len(np.unique(sids[sel]))} subjects")

    write(out, src, X, sids, side, man, sel, a, how=f"subsampled to {a.target_windows}")
    return 0


def write(out, src, X, sids, side, man, sel, a, how="unspecified"):
    # build_cohort.py refuses to write into a non-empty cohort dir, because nothing in a
    # score file records which cohort produced it. This builder is the one that gets run
    # repeatedly, and it had no such guard -- matrix_d1_c1 was silently rewritten under a
    # design that had already been built against it.
    if (out / "windows.npy").exists() and not a.force:
        sys.exit(f"[matched] {out} already holds a cohort. Every result computed against "
                 f"it records nothing about which build it was. Delete it or pass --force.")
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "windows.npy", np.ascontiguousarray(X[sel]))
    np.save(out / "subject_ids.npy", sids[sel])
    for name, arr in side.items():
        np.save(out / name, arr[sel])
    # copying this verbatim left a 506-subject cohort carrying a 1,329-subject table,
    # and subgroup_coverage.py merges FROM the parquet, so every denominator it computed
    # would silently include subjects the cohort does not contain
    kept_ids = set(map(str, np.unique(sids[sel])))
    if (src / "subjects.parquet").exists():
        try:
            import pandas as pd
            t = pd.read_parquet(src / "subjects.parquet")
            key = "id" if "id" in t.columns else t.columns[0]
            t = t[t[key].astype(str).isin(kept_ids)]
            t.to_parquet(out / "subjects.parquet")
            print(f"[matched] subjects.parquet filtered to {len(t)} rows")
        except Exception as e:
            print(f"[matched] could not filter subjects.parquet ({e}); NOT copying a "
                  f"stale one -- a missing table fails loudly, a wrong one does not")

    man = dict(man)
    man.update(n_subjects=int(len(np.unique(sids[sel]))), n_windows=int(len(sel)),
               subsampled_from=str(src), subsample_seed=a.seed, selection=how)
    # `matched_to` used to be stamped unconditionally, including on the path that keeps
    # every window and matches nothing. A manifest that claims a property it does not
    # have is worse than one that is silent.
    if how.startswith("exposure-matched") or how.startswith("subsampled"):
        man["matched_to"] = how
        man["note"] = (man.get("note", "").replace(
            " SUBSAMPLED — not a cohort for any other purpose.", "")
            + " SUBSAMPLED — not a cohort for any other purpose.")
    else:
        man.pop("matched_to", None)
    (out / "manifest.json").write_text(json.dumps(man, indent=1))

    per = np.bincount(np.unique(sids[sel], return_inverse=True)[1])
    print(f"[matched] windows per subject: min {per.min()} median "
          f"{int(np.median(per))} max {per.max()}")
    print(f"[matched] wrote {out}")


if __name__ == "__main__":
    sys.exit(main())
