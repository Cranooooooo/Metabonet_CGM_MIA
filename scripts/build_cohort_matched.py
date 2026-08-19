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
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args()

    src, out = Path(a.src), Path(a.out)
    X = np.load(src / "windows.npy", mmap_mode="r")
    sids = np.load(src / "subject_ids.npy", allow_pickle=True)
    days = np.load(src / "days.npy", allow_pickle=True)
    man = json.loads((src / "manifest.json").read_text())

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

    if a.target_windows is None:
        sel = np.sort(np.flatnonzero(np.isin(sids, list(keep))))
        print(f"[matched] keeping every window: {len(sel):,} over "
              f"{len(np.unique(sids[sel]))} subjects")
        write(out, src, X, sids, days, man, sel, a)
        return 0

    rng = np.random.default_rng(a.seed)
    idx_by_subject = {}
    for s in sorted(keep):
        w = np.flatnonzero(sids == s)
        order = np.argsort(days[w])          # date order, so a block is contiguous in time
        idx_by_subject[s] = w[order]

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
    print(f"[matched] selected {len(sel):,} windows over "
          f"{len(np.unique(sids[sel]))} subjects")

    write(out, src, X, sids, days, man, sel, a)
    return 0


def write(out, src, X, sids, days, man, sel, a):
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "windows.npy", np.ascontiguousarray(X[sel]))
    np.save(out / "subject_ids.npy", sids[sel])
    np.save(out / "days.npy", days[sel])
    if (src / "subjects.parquet").exists():
        import shutil
        shutil.copy(src / "subjects.parquet", out / "subjects.parquet")

    man = dict(man)
    man.update(n_subjects=int(len(np.unique(sids[sel]))), n_windows=int(len(sel)),
               subsampled_from=str(src), subsample_seed=a.seed,
               matched_to="the seven-day pilot's training regime: same subjects, same "
                          "window count, so the same step budget gives the same epochs",
               note=man.get("note", "") + " SUBSAMPLED — not a cohort for any other purpose.")
    (out / "manifest.json").write_text(json.dumps(man, indent=1))

    per = np.bincount(np.unique(sids[sel], return_inverse=True)[1])
    print(f"[matched] windows per subject: min {per.min()} median "
          f"{int(np.median(per))} max {per.max()}")
    print(f"[matched] wrote {out}")


if __name__ == "__main__":
    sys.exit(main())
