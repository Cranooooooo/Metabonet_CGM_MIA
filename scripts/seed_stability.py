#!/usr/bin/env python
"""How much of the outlier list is the data, and how much is the subsample?

Several methods do not see all the data. B5/B6 compare 30 days per subject rather than
every day; C8/C9/C10/E14 score every subject against one reference pool of a few
thousand windows. Those choices are what make the methods affordable, and they mean a
different draw is a different, equally legitimate answer.

This script runs the whole stack under several base seeds and reports, per subject,
in how many of them it reached the consensus list. A subject flagged under every seed
is a property of the cohort. A subject flagged under one is a property of that draw,
and carrying it into a 57-model experiment would spend GPU time on noise.

    python scripts/seed_stability.py --seeds 2026,7,101,999 --out results/seed_stability

Cost: one full method run per seed. The clinical metrics (A1-A4) and the encoder are
cached across seeds; the DTW and MMD passes are not, and they dominate the runtime.

What this does NOT cover: the methods' own tuning constants -- 30 days, 4000 reference
windows, three glucotypes, the 5% cut, the 7-of-13 bar. Those are held fixed here.
Re-running under a different cut is a separate sweep and a separate claim.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter

import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cgmoutlier._env import check as _envcheck   # noqa: E402
_envcheck()
from cgmoutlier.outliers.run import consensus, run          # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default="data/cohort/metabonet875")
    ap.add_argument("--out", default="results/seed_stability")
    ap.add_argument("--seeds", default="2026,7,101,999")
    ap.add_argument("--only", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--top-pct", type=float, default=5.0)
    ap.add_argument("--min-methods", type=int, default=7)
    ap.add_argument("--analyse-only", action="store_true",
                    help="read seedN/ directories that already exist; run nothing")
    ap.add_argument("--channels", default="all")
    ap.add_argument("--clinical-cache", default=None,
                    help="a _clinical.parquet for this cohort; group A is invariant to "
                         "both the seed and the channel set, so one file serves every "
                         "seed of every channel set on the same cohort")
    a = ap.parse_args()

    seeds = [int(s) for s in a.seeds.split(",")]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    only = a.only.split(",") if a.only else None

    per_seed = {}
    for sd in seeds:
        d = out / f"seed{sd}"
        d.mkdir(parents=True, exist_ok=True)
        # A1-A4 read from cohort-level metrics that no seed touches; hard-link the
        # cache in so each seed does not recompute 875 subjects' clinical battery.
        # ⚠ The default used to be results/outliers/_clinical.parquet -- a cache built
        # on 2026-08-06 over metabonet875, keyed on the OLD bare `id`. Copying it into a
        # run on any other cohort is wrong, and on 2026-08-19 it took down the matrix
        # prep with a raw KeyError. Where the key SCHEME happens to match it would have
        # been worse: silently scoring one cohort's subjects with another's metrics.
        # The default is now local to this run; sharing must be asked for explicitly.
        src = (Path(a.clinical_cache) if a.clinical_cache
               else out / "_clinical.parquet")
        if src.exists() and not (d / "_clinical.parquet").exists():
            shutil.copy(src, d / "_clinical.parquet")
        if not a.analyse_only:
            run(a.cohort, d, only=only, device=a.device, seed=sd,
                channels=a.channels)
        c = consensus(d, a.top_pct, a.min_methods)
        (d / "consensus.json").write_text(json.dumps(c, indent=2))
        per_seed[sd] = c["outliers"]
        print(f"seed {sd}: {len(c['outliers'])} outliers", flush=True)

    cnt = Counter(s for lst in per_seed.values() for s in lst)
    n = len(seeds)
    always = sorted(s for s, k in cnt.items() if k == n)
    sometimes = sorted((s for s, k in cnt.items() if k < n), key=lambda s: -cnt[s])

    per_method = method_drift(out, seeds, a.top_pct)

    rec = dict(seeds=seeds, per_seed={str(k): v for k, v in per_seed.items()},
               per_method=per_method,
               n_seeds=n, counts={s: cnt[s] for s in cnt},
               always=always, sometimes=sometimes,
               jaccard_to_first=_jaccard(per_seed, seeds),
               note=("`always` is the list to carry forward: a subject that survives "
                     "every subsample is flagged by the cohort, not by the draw."))
    (out / "stability.json").write_text(json.dumps(rec, indent=2))

    print(f"\nflagged under all {n} seeds: {len(always)}")
    print(" ", " ".join(always))
    print(f"flagged under some but not all: {len(sometimes)}")
    for s in sometimes:
        print(f"   {s}: {cnt[s]}/{n}")
    print("\nper-method agreement across seeds (median pairwise Spearman, and the\n"
          "share of each seed's top-5% list that every other seed also flags):")
    print(f"  {'method':<6} {'rho':>7} {'top5% overlap':>15}")
    for k, v in sorted(per_method.items(), key=lambda kv: kv[1]["median_rho"]):
        mark = "   <-- unstable" if v["median_rho"] < 0.9 else ""
        print(f"  {k:<6} {v['median_rho']:>7.3f} {v['median_top_overlap']:>14.0%}{mark}")

    print(f"\nwrote {out / 'stability.json'}")


def method_drift(out, seeds, top_pct):
    """How much each METHOD moves when only the base seed changes.

    The consensus surviving a reseed is the headline, but it can survive for two very
    different reasons: every method is stable, or one method is noise and the other
    twelve outvote it. Those call for different write-ups, so both are measured.
    """
    import itertools

    import pandas as pd

    keys = sorted({p.stem for sd in seeds
                   for p in (out / f"seed{sd}").glob("[ABCDE]*.parquet")})
    rep = {}
    for k in keys:
        cols = {}
        for sd in seeds:
            f = out / f"seed{sd}" / f"{k}.parquet"
            if f.exists():
                s = pd.read_parquet(f).set_index("id")["score"]
                s.index = s.index.astype(str)
                cols[sd] = s
        if len(cols) < 2:
            continue
        df = pd.DataFrame(cols).dropna()
        n = max(1, int(round(len(df) * top_pct / 100)))
        rhos, ovl = [], []
        for x, y in itertools.combinations(cols, 2):
            rhos.append(df[x].corr(df[y], method="spearman"))
            ovl.append(len(set(df[x].nlargest(n).index) &
                           set(df[y].nlargest(n).index)) / n)
        rep[k] = dict(n_seeds=len(cols), cut_n=n,
                      median_rho=round(float(np.median(rhos)), 4),
                      min_rho=round(float(np.min(rhos)), 4),
                      median_top_overlap=round(float(np.median(ovl)), 4),
                      min_top_overlap=round(float(np.min(ovl)), 4))
    return rep


def _jaccard(per_seed, seeds):
    a = set(per_seed[seeds[0]])
    return {str(s): round(len(a & set(per_seed[s])) / max(1, len(a | set(per_seed[s]))), 4)
            for s in seeds}


if __name__ == "__main__":
    sys.exit(main())
