#!/usr/bin/env python
"""Measure generation quality for one or more released sample sets.

    python scripts/eval_quality.py --runs results/runs/dimts_h128_rep1/base ...

Writes results/quality/<label>.json per run plus a combined table on stdout. The real
reference is the run's OWN training set -- meta.json records the job, and the design
records that job's subjects -- so a model is scored against the distribution it was
asked to reproduce rather than against the whole cohort.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from cgmoutlier._env import check as _envcheck                      # noqa: E402
_envcheck()
from cgmoutlier.data.cohort import load as load_cohort              # noqa: E402
from cgmoutlier.loo import training_set                             # noqa: E402
from cgmoutlier.quality import evaluate                             # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True,
                    help="run directories, each holding samples.npy and meta.json")
    ap.add_argument("--design", default=None,
                    help="design dir holding jobs/<name>.json; inferred from the run "
                         "path when it follows the *_rep<r> convention")
    ap.add_argument("--cohort", default="data/cohort/metabonet875")
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--n-fid", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--skip", nargs="*", default=[])
    ap.add_argument("--out", default="results/quality")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    X, sids, man = load_cohort(a.cohort)
    rows = []
    for rp in a.runs:
        rp = Path(rp)
        meta = json.loads((rp / "meta.json").read_text())
        label = f"{rp.parent.name}__{rp.name}"

        design = a.design
        if design is None:
            rep = rp.parent.name.rsplit("_rep", 1)
            design = (f"results/design_sym/rep{rep[1]}" if len(rep) == 2
                      else "results/design_sym/rep1")
        job = json.loads((Path(design) / "jobs" / f"{meta['job']}.json").read_text())
        real, _ = training_set(X, sids, job["subjects"])
        synth = np.load(rp / "samples.npy", mmap_mode="r")

        print(f"\n=== {label}  ({meta['job']}, {job['n_subjects']} subjects, "
              f"{len(real):,} real / {len(synth):,} synthetic windows) ===", flush=True)
        q = evaluate(np.asarray(real), np.asarray(synth[:max(a.n, a.n_fid) * 4]),
                     n=a.n, n_fid=a.n_fid, seed=a.seed, device=a.device, skip=a.skip)
        q.update(label=label, run=str(rp), job=meta["job"],
                 generator=meta.get("generator"), params=meta.get("params"),
                 fit_seconds=meta.get("fit_seconds"))
        (out / f"{label}.json").write_text(json.dumps(q, indent=2))
        for k in ("context_fid", "discriminative_score", "discriminative_accuracy",
                  "predictive_score_tstr"):
            if k in q:
                print(f"  {k:<26} {q[k]:.5f}")
        rows.append(q)

    print(f"\n{'run':<34}{'ContextFID':>12}{'discrim':>10}{'pred MAE':>10}")
    for q in rows:
        print(f"{q['label']:<34}{q.get('context_fid', float('nan')):>12.4f}"
              f"{q.get('discriminative_score', float('nan')):>10.4f}"
              f"{q.get('predictive_score_tstr', float('nan')):>10.4f}")
    print("\ndiscriminative: 0 = a classifier cannot tell real from synthetic (best), "
          "0.5 = trivial")
    print("predictive:     MAE of one-step-ahead, trained on synthetic, scored on "
          "real. Lower is better")
    print("Context-FID:    Frechet distance in a per-call TS2Vec space. Lower is "
          "better; scale is arbitrary and only comparable WITHIN this table")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
