#!/usr/bin/env python
"""Refuse to launch a matrix whose cells are not like for like.

Every cell must contain every subject the shared design names -- targets, controls and
background. A cell missing even one target would silently drop it from the arm and
unbalance the pairing, and the difference between cells would then be partly "who was
in it" rather than the condition.
"""
import json
import sys
from pathlib import Path

import numpy as np

CELLS = ("matrix_d1_c1", "matrix_d1_c2", "matrix_d7_c1", "matrix_d7_c2")


def main():
    d = json.loads(Path("results/matrix/design/rep1/design.json").read_text())
    base = json.loads(Path("results/matrix/design/rep1/jobs/base.json").read_text())
    need = set(map(str, d["outliers"] + d["controls"] + base["subjects"]))
    print(f"  design names {len(need)} subjects "
          f"({len(d['outliers'])} outliers + {len(d['controls'])} controls + background)")
    bad = False
    for cell in CELLS:
        p = Path("data/cohort") / cell
        if not p.exists():
            print(f"  {cell:16} MISSING COHORT")
            bad = True
            continue
        have = set(np.load(p / "subject_ids.npy", allow_pickle=True).tolist())
        m = json.loads((p / "manifest.json").read_text())
        miss = need - have
        flag = "OK" if not miss else f"MISSING {len(miss)}: {sorted(miss)[:6]}"
        print(f"  {cell:16} {m['n_subjects']:>4} subjects "
              f"{m['n_windows']:>8,} windows   {flag}")
        bad |= bool(miss)
    if bad:
        raise SystemExit("  a cell is missing design subjects -- nothing is launched")
    print("  all four cells carry every design subject")
    return 0


if __name__ == "__main__":
    sys.exit(main())
