#!/usr/bin/env python
"""Does adding basal and bolus change who the outliers are?

    python scripts/compare_outlier_sets.py \
        --a results/outliers_sid_c3/stability_1ch --b results/outliers_sid_c3/stability_3ch

Both runs are over the SAME cohort, the same subjects and the same days; the only
difference is how many channels the methods see. That is what makes a difference in the
two lists attributable to the channels.

READ THE COMPARISON KNOWING WHAT CANNOT MOVE. Group A -- A1 to A4, four of the thirteen
candidates -- is glucose-only in both runs, because its metrics are the Battelino
consensus and a basal rate has no time in range. Nearly a third of the vote is therefore
identical by construction, so:

  * an overlap of zero is impossible
  * the measured difference is a FLOOR on the effect of adding channels, not a ceiling
  * a subject flagged by group A alone will appear in both lists whatever the channels do

The per-method breakdown separates that out: it reports, for each method, how much its
own top-5% set moved. A1-A4 must show zero drift, and if they do not, the two runs are
not on the same cohort and nothing else here means anything.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


def load_side(path: Path):
    """-> (stable list, per-seed lists, label). Accepts a seed_stability directory or a
    single run directory."""
    st = path / "stability.json"
    if st.exists():
        d = json.loads(st.read_text())
        return set(d["always"]), {k: set(v) for k, v in d["per_seed"].items()}, "stable"
    c = json.loads((path / "consensus.json").read_text())
    return set(c["outliers"]), {"single": set(c["outliers"])}, "single-seed"


def top_sets(path: Path, top_pct: float):
    """Per-method top-`top_pct` subject sets, for the drift table."""
    d = path / "seed2026" if (path / "seed2026").exists() else path
    out = {}
    for p in sorted(d.glob("[ABCDE]*.parquet")):
        s = pd.read_parquet(p).set_index("id")["score"]
        out[p.stem] = set(s.nlargest(max(1, int(len(s) * top_pct / 100))).index.astype(str))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="the CGM-only run")
    ap.add_argument("--b", required=True, help="the multichannel run")
    ap.add_argument("--label-a", default="CGM")
    ap.add_argument("--label-b", default="CGM+basal+bolus")
    ap.add_argument("--top-pct", type=float, default=5.0)
    ap.add_argument("--out", default="results/outliers_sid_c3/comparison.json")
    a = ap.parse_args()

    A, perA, kindA = load_side(Path(a.a))
    B, perB, kindB = load_side(Path(a.b))
    both, only_a, only_b = A & B, A - B, B - A
    jac = len(both) / max(len(A | B), 1)

    print(f"=== consensus outliers ({kindA} / {kindB}) ===")
    print(f"  {a.label_a:22} {len(A):>4}")
    print(f"  {a.label_b:22} {len(B):>4}")
    print(f"  in both                {len(both):>4}   Jaccard {jac:.3f}")
    print(f"  only {a.label_a:17} {len(only_a):>4}   {sorted(only_a)}")
    print(f"  only {a.label_b:17} {len(only_b):>4}   {sorted(only_b)}")

    print(f"\n=== per-method drift in the top {a.top_pct}% ===")
    ta, tb = top_sets(Path(a.a), a.top_pct), top_sets(Path(a.b), a.top_pct)
    print(f"{'method':>8}{'|A|':>7}{'|B|':>7}{'shared':>8}{'Jaccard':>9}  note")
    rows = []
    for k in sorted(set(ta) | set(tb)):
        sa, sb = ta.get(k, set()), tb.get(k, set())
        j = len(sa & sb) / max(len(sa | sb), 1)
        note = ""
        if k.startswith("A") and j < 0.999:
            note = "<-- group A must not move; the two runs differ in more than channels"
        if k == "E13" and j < 0.999:
            note = "<-- E13 is day count; it cannot depend on channels"
        print(f"{k:>8}{len(sa):>7}{len(sb):>7}{len(sa & sb):>8}{j:>9.3f}  {note}")
        rows.append(dict(method=k, n_a=len(sa), n_b=len(sb),
                         shared=len(sa & sb), jaccard=j))

    invariant = [r for r in rows if r["method"].startswith("A") or r["method"] == "E13"]
    broken = [r["method"] for r in invariant if r["jaccard"] < 0.999]
    print(f"\ninvariant methods that moved: {broken if broken else 'none'}")

    rec = dict(a=str(a.a), b=str(a.b), label_a=a.label_a, label_b=a.label_b,
               n_a=len(A), n_b=len(B), n_both=len(both), jaccard=jac,
               both=sorted(both), only_a=sorted(only_a), only_b=sorted(only_b),
               per_method=rows, invariant_methods_that_moved=broken,
               per_seed_a={k: sorted(v) for k, v in perA.items()},
               per_seed_b={k: sorted(v) for k, v in perB.items()})
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rec, indent=2))
    print(f"\nwrote {a.out}")
    return 1 if broken else 0


if __name__ == "__main__":
    sys.exit(main())
