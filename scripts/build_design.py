#!/usr/bin/env python
"""Stage 3: turn the consensus outlier list into the exact set of training runs.

    python scripts/build_design.py --consensus results/outliers/consensus.json

Writes `results/design/design.json` plus one file per run under `results/design/jobs/`,
each holding the literal subject list that run trains on. Nothing is trained here and
no data is read beyond the cohort's subject index -- the point is that the design can
be inspected, checked into the repository, and argued with before any GPU time is spent.

Read `matching` in the output before proceeding. If the controls could not be matched
on day count, the two arms differ on something other than membership and the experiment
answers a different question than the one asked.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cgmoutlier._env import check as _envcheck   # noqa: E402
_envcheck()
from cgmoutlier.data.cohort import load          # noqa: E402
from cgmoutlier.loo import build, save           # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default="data/cohort/metabonet875")
    ap.add_argument("--consensus", default="results/outliers/consensus.json")
    ap.add_argument("--outliers", default=None,
                    help="comma-separated ids, instead of reading --consensus")
    ap.add_argument("--stability", default=None,
                    help="results/seed_stability/stability.json -- use its `always` "
                         "list, the subjects flagged under EVERY base seed. This is "
                         "the list to carry into the generator stage: a subject "
                         "flagged under one draw is a property of that draw, and each "
                         "one costs two models")
    ap.add_argument("--out", default="results/design")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--tol", type=float, default=0.15,
                    help="relative day-count gap above which a control is flagged")
    ap.add_argument("--asymmetric", action="store_true",
                    help="the pre-2026-08-07 form: base trains on ALL normals, controls "
                         "are exclude_c runs. It puts the base model's own offset into "
                         "the two arms with opposite sign -- see loo.design. Kept only "
                         "to rebuild the copy_paste validation design")
    ap.add_argument("--replicates", type=int, default=1,
                    help="build N designs with DISJOINT control draws, under "
                         "<out>/rep1..repN. Each is a complete design directory. "
                         "Replicates are the only thing that makes the p-value rest on "
                         "independent units: within one design every target shares a "
                         "base model")
    ap.add_argument("--n-candidates", type=int, default=None,
                    help="randomise the control draw among the k nearest available "
                         "subjects by day count. Default 1 for a single design (the "
                         "deterministic greedy draw) and 8 with --replicates, because "
                         "k=1 returns the SAME controls every time and the replicates "
                         "would be fake. Read median_rel_gap to see what k cost")
    ap.add_argument("--control-pool", default=None, metavar="JSON_OR_TXT",
                    help="restrict controls to these subjects: either a newline-"
                         "delimited list, or the union_normals.json written by "
                         "scripts/union_normals.py (its `clean` key). Targets are the "
                         "intersection across seeds, so without this the controls are "
                         "screened at a much looser bar than the targets and a subject "
                         "flagged by several methods can be drawn as a normal.")
    ap.add_argument("--exclude-from-background", default=None, metavar="TXT",
                    help="newline-delimited subjects to keep out of the base's training "
                         "set as well as out of the control arm -- typically the "
                         "borderline outliers the consensus flagged in some seeds but "
                         "not all")
    a = ap.parse_args()

    _, sids, man = load(a.cohort)
    subs, counts = np.unique(sids, return_counts=True)
    days = {str(s): int(c) for s, c in zip(subs, counts)}

    borderline = None
    if a.outliers:
        outliers = a.outliers.split(",")
    elif a.stability:
        s = json.loads(Path(a.stability).read_text())
        outliers = list(s["always"])
        borderline = {k: v for k, v in s["counts"].items()
                      if 0 < v < s["n_seeds"] and str(k) not in set(map(str, outliers))}
        print(f"{len(outliers)} outliers flagged under all {s['n_seeds']} seeds "
              f"{tuple(s['seeds'])}, from {a.stability}")
        print(f"  not carried forward ({len(borderline)} subjects flagged under some "
              f"but not all): "
              f"{' '.join(sorted(borderline, key=lambda k: -borderline[k]))}")
        print("  they stay in the control pool and are tagged, not removed -- removing "
              "them would leave a hand-picked 'most normal' control set and tighten "
              "the null")
    else:
        c = json.loads(Path(a.consensus).read_text())
        outliers = c["outliers"]
        print(f"{len(outliers)} outliers from {a.consensus} "
              f"(>={c['min_methods']} of {c['n_candidates']} at top-{c['cut_pct']}%)")

    symmetric = not a.asymmetric
    k = a.n_candidates if a.n_candidates is not None else (8 if a.replicates > 1 else 1)
    if a.replicates > 1 and k == 1:
        sys.exit("--n-candidates 1 returns the same controls for every replicate; the "
                 "replicates would differ only by training noise. Use k > 1.")

    control_pool = None
    if a.control_pool:
        raw = Path(a.control_pool).read_text()
        if a.control_pool.endswith(".json"):
            control_pool = json.loads(raw)["clean"]
        else:
            control_pool = [l.strip() for l in raw.splitlines() if l.strip()]
        print(f"control pool restricted to {len(control_pool)} subjects "
              f"from {a.control_pool}")

    xbg = []
    if a.exclude_from_background:
        xbg = [l.strip() for l in
               Path(a.exclude_from_background).read_text().splitlines() if l.strip()]
        print(f"excluding {len(xbg)} borderline subjects from the background too")

    subs = [str(s) for s in subs]
    used, made = [], []
    for r in range(1, a.replicates + 1):
        d = build(subs, outliers, days, seed=a.seed + r - 1, tol=a.tol,
                  symmetric=symmetric, n_candidates=k, exclude=used,
                  borderline=borderline, control_pool=control_pool,
                  exclude_from_background=xbg)
        out = save(d, a.out if a.replicates == 1 else Path(a.out) / f"rep{r}")
        used += list(d.controls)
        made.append((r, d, out))

        m = d.matching
        tag = f"replicate {r}  " if a.replicates > 1 else ""
        kind = ("1 base on %d background, %d outlier + %d control include runs"
                % (len(d.background), len(d.outliers), len(d.controls))) if symmetric \
            else ("1 base on %d normals, %d include, %d exclude"
                  % (len(d.normals), len(d.outliers), len(d.controls)))
        print(f"\n{tag}{len(d.jobs)} training runs: {kind}")
        print(f"  day count -- outliers median {m['outlier_days']['median']:.0f} "
              f"(range {m['outlier_days']['min']}-{m['outlier_days']['max']}), "
              f"pool median {m['pool_days']['median']:.0f}")
        print(f"  control matching -- median gap {m['median_rel_gap']:.1%}, "
              f"worst {m['max_rel_gap']:.1%}, {m['n_over_tol']} over the {a.tol:.0%} bar")
        print(f"  controls: {' '.join(sorted(d.controls))}")
        for key in ("warning", "borderline_controls"):
            if key in m:
                print(f"  {key.upper()}: {m[key]}")
        print(f"  wrote {out}/design.json and {len(d.jobs)} job files")

    if a.replicates > 1:
        allc = [c for _, d, _ in made for c in d.controls]
        assert len(set(allc)) == len(allc), "control draws overlap across replicates"
        print(f"\n{a.replicates} replicates, {len(set(allc))} distinct controls, no "
              f"subject used twice. {sum(len(d.jobs) for _, d, _ in made)} models total.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
