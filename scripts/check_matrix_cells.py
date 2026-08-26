#!/usr/bin/env python
"""Refuse to launch a matrix whose cells are not like for like.

WHAT "LIKE FOR LIKE" HAS TO MEAN, and why the first version of this check was not enough.

It used to verify only that every cell CONTAINED every subject the design names. That
passed on 2026-08-22, and the matrix it cleared was confounded two ways:

  epochs      d1_c1 held 107,927 windows and d7_c1 held 6,674. At a shared 100,000
              steps x batch 64 that is 62.7 epochs against 1016.2 -- 16.2x, in the same
              direction as the hypothesis. Equal STEPS is not equal training.
  exposure    controls were matched on one-day day counts and reused in all four cells.
              In d7_c1 outlier Loop/968 had 5 windows against its control's 31. The two
              arms then differ on how much data each subject contributed, not only on
              membership -- and design.json reported the one-day matching numbers for
              every cell, so reading it told you the opposite.

Both are invisible to a membership check. So this now asserts the three things that make
a cell comparable: the same subjects, the same windows per subject, and therefore the
same epoch budget. What it deliberately does NOT check is data volume -- a seven-day
window holds seven days and a one-day window holds one, and that is the variable under
study, not a defect.
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

CELLS = ("matrix_d1_c1", "matrix_d1_c2", "matrix_d7_c1", "matrix_d7_c2")
TOL = 0.15          # same tolerance loo/design.py matches controls to
STEPS, BATCH = 100_000, 64


def main():
    d = json.loads(Path("results/matrix/design/rep1/design.json").read_text())
    base = json.loads(Path("results/matrix/design/rep1/jobs/base.json").read_text())
    outs, ctrls = list(map(str, d["outliers"])), list(map(str, d["controls"]))
    need = set(outs + ctrls + list(map(str, base["subjects"])))
    print(f"  design names {len(need)} subjects "
          f"({len(outs)} outliers + {len(ctrls)} controls + background)")

    bad, counts = False, {}
    for cell in CELLS:
        p = Path("data/cohort") / cell
        if not p.exists():
            print(f"  {cell:16} MISSING COHORT")
            bad = True
            continue
        sids = np.load(p / "subject_ids.npy", allow_pickle=True).astype(str)
        counts[cell] = Counter(sids.tolist())
        m = json.loads((p / "manifest.json").read_text())
        miss = need - set(counts[cell])
        flag = "OK" if not miss else f"MISSING {len(miss)}: {sorted(miss)[:6]}"
        ep = STEPS * BATCH / len(sids)
        print(f"  {cell:16} {m['n_subjects']:>4} subj {m['n_windows']:>8,} win "
              f"T={m['T']:<5} C={m.get('C','?')}  {ep:>7.1f} epochs   {flag}")
        bad |= bool(miss)
    if bad:
        raise SystemExit("  a cell is missing design subjects -- nothing is launched")

    # 1. epochs. Equal steps is not equal training unless the cells hold equal windows.
    tot = {c: sum(v.values()) for c, v in counts.items()}
    lo, hi = min(tot.values()), max(tot.values())
    print(f"\n  windows per cell: {tot}")
    if hi / lo > 1.01:
        bad = True
        print(f"  EPOCH MISMATCH: {hi/lo:.2f}x between the largest and smallest cell "
              f"({STEPS*BATCH/hi:.1f} to {STEPS*BATCH/lo:.1f} epochs at {STEPS:,} steps). "
              f"Any cross-cell difference would be partly a training-length effect.")
    else:
        print(f"  epochs match within 1%: {STEPS*BATCH/hi:.0f} per cell at {STEPS:,} steps")

    # 2. exposure. The same subject must contribute the same number of windows to
    #    every cell, or "which cell" and "how much data" move together.
    off = [s for s in sorted(need)
           if len({counts[c].get(s, 0) for c in counts}) > 1]
    if off:
        bad = True
        print(f"  EXPOSURE MISMATCH: {len(off)} subjects have different window counts "
              f"across cells, e.g. " +
              "; ".join(f"{s} {[counts[c].get(s,0) for c in CELLS]}" for s in off[:3]))
    else:
        print(f"  exposure identical across cells for all {len(need)} design subjects")

    # 3. control matching, re-derived IN EACH CELL rather than trusted from design.json
    print(f"\n  {'cell':16}{'median gap':>12}{'max gap':>10}{'over tol':>10}")
    for cell in CELLS:
        g = sorted(abs(counts[cell][o] - counts[cell][c]) /
                   max(counts[cell][o], counts[cell][c], 1) for o, c in zip(outs, ctrls))
        over = sum(x > TOL for x in g)
        print(f"  {cell:16}{g[len(g)//2]:>12.3f}{max(g):>10.3f}{over:>10}")
        if over:
            bad = True
    # 4. the control arm must be normals, screened as strictly as the targets were.
    #    Targets are the intersection across seeds; controls drawn from "not a consensus
    #    outlier" clear a far looser bar, and Loop/1041 (19 method-flags of 52, against
    #    28-47 for the targets) was drawn as a control on 08-22 under exactly that gap.
    uf = Path("results/matrix/outliers/union_normals.json")
    if not uf.exists():
        bad = True
        print(f"\n  NO UNION AUDIT at {uf} -- run scripts/union_normals.py. Without it "
              f"there is no evidence the control arm is normal, and 'not checked' must "
              f"not read as 'clean'.")
    else:
        u = json.loads(uf.read_text())
        flags = u.get("flags", {})
        dirty = [(c, flags.get(c, 0)) for c in ctrls if flags.get(c, 0)]
        if dirty:
            bad = True
            print(f"\n  CONTROL ARM NOT CLEAN: " +
                  ", ".join(f"{c} flagged {n}x" for c, n in dirty))
        else:
            print(f"\n  all {len(ctrls)} controls are outside the union of every "
                  f"method's top-{u['top_pct']}% over {u['n_seeds']} seeds "
                  f"({u['n_clean']} of {u['n_subjects']} subjects were never flagged)")

    if bad:
        raise SystemExit("\n  cells are not like for like -- nothing is launched. "
                         "Fix the cohorts or the design; do not launch and explain later.")
    print("\n  all four cells: same subjects, same exposure, same epochs, controls matched")
    return 0


if __name__ == "__main__":
    sys.exit(main())
