#!/usr/bin/env python
"""Who is a CLEAN normal -- and it is not "whoever is not in the outlier list".

THE ASYMMETRY THIS FIXES
------------------------
Targets are the INTERSECTION: `stability.json['always']`, the 13 subjects the consensus
flags in every seed. That is a deliberately strict bar, because a target has to be an
outlier for the claim to mean anything.

Controls were then drawn from "everyone the consensus did not flag" -- which is the
complement of a *loose* set. A subject flagged by six of the thirteen methods, or flagged
by the consensus in three seeds out of four, clears that bar and can be drawn as a
"normal". The two arms are then not opposites: one is screened at the strictest setting
and the other is barely screened at all, and any leakage difference between them is
measured against a control arm that is partly outliers.

So the control pool has to be the complement of the UNION: every subject that no method
flagged, in any seed. Strict on both sides.

    python scripts/union_normals.py --outliers results/matrix/outliers --design results/matrix/design/rep1
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outliers", required=True, help="dir holding seed*/ score parquets")
    ap.add_argument("--design", default=None, help="design to audit against")
    ap.add_argument("--top-pct", type=float, default=5.0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    import pandas as pd
    root = Path(a.outliers)
    seeds = sorted(p for p in root.glob("seed*") if p.is_dir())
    if not seeds:
        sys.exit(f"no seed*/ directories under {root}")

    # `candidates` is the voting set. E13 (day count) sits in the directory as a control
    # and is deliberately not a voter, so it must not widen the union either.
    cand = json.loads((seeds[0] / "consensus.json").read_text())["candidates"]
    print(f"[union] {len(seeds)} seeds x {len(cand)} methods, top {a.top_pct}%")

    flags = Counter()          # subject -> how many (method, seed) cells flagged it
    per_seed_methods = 0
    everyone = set()
    for sd in seeds:
        for m in cand:
            f = sd / f"{m}.parquet"
            if not f.exists():
                print(f"[union] MISSING {f} -- refusing to call a union complete when a "
                      f"method is absent")
                sys.exit(1)
            t = pd.read_parquet(f)
            # the index is a RangeIndex and the subject key is an `id` COLUMN; and the
            # value is `score`, not the last column -- A2 carries a trailing `cv`. The
            # first version of this took s.index and s.columns[-1] and returned 0 flags
            # for all 13 consensus outliers, which is how it was caught.
            if "id" not in t.columns or "score" not in t.columns:
                sys.exit(f"[union] {f} has columns {list(t.columns)}; expected id+score")
            t = t.set_index(t["id"].astype(str))["score"]
            everyone |= set(t.index)
            cut = max(1, int(len(t) * a.top_pct / 100))
            for i in t.nlargest(cut).index:
                flags[str(i)] += 1
            per_seed_methods += 1

    n = len(everyone)
    union = set(flags)
    clean = everyone - union
    print(f"[union] {n} subjects; {len(union)} flagged at least once by at least one "
          f"method in at least one seed; {len(clean)} never flagged")
    print(f"[union] flag-count distribution: " +
          ", ".join(f"{k}x:{v}" for k, v in sorted(Counter(flags.values()).items())[:10]))

    out = dict(n_subjects=n, n_seeds=len(seeds), n_methods=len(cand), top_pct=a.top_pct,
               n_union=len(union), n_clean=len(clean),
               clean=sorted(clean), flags={k: v for k, v in flags.most_common()})

    if a.design:
        d = json.loads((Path(a.design) / "design.json").read_text())
        ctrls = list(map(str, d["controls"]))
        outs = list(map(str, d["outliers"]))
        print(f"\n[union] auditing {a.design}")
        print(f"  {'control':16}{'flags':>7}   verdict")
        dirty = []
        for c in ctrls:
            f = flags.get(c, 0)
            if f:
                dirty.append((c, f))
            print(f"  {c:16}{f:>7}   {'CLEAN' if not f else 'FLAGGED -- not a normal'}")
        print(f"\n  outliers, for contrast (these MUST be high -- they are the "
              f"consensus picks; all-zero means the join is broken, not that they are clean):")
        for o in outs:
            print(f"  {o:16}{flags.get(o,0):>7}")
        if not any(flags.get(o, 0) for o in outs):
            sys.exit("[union] every consensus outlier has zero flags -- the subject-key "
                     "join is wrong. Refusing to report a union that cannot be right.")
        out["dirty_controls"] = dirty
        if dirty:
            print(f"\n  {len(dirty)} of {len(ctrls)} controls were flagged by at least one "
                  f"method. The control arm is not a clean normal arm.")
        else:
            print(f"\n  all {len(ctrls)} controls are outside the union -- clean normals")

    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(out, indent=1))
        print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
