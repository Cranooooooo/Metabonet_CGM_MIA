#!/usr/bin/env python
"""Discriminative accuracy is one draw of a lower bound. Take the distribution instead.

WHY THIS EXISTS
---------------
The seven-day h96 base scored 0.9475 on 2026-08-17 05:36 and 0.5908 at 21:34, on a
byte-identical samples.npy with the same `seed`. The GRU discriminator's weights were
initialised from torch's unseeded global rng, so the reading depended on how many models
had been scored earlier in the same process (`src/cgmoutlier/quality.py`, now fixed).

That defect is what makes this script necessary, but it is not the whole story. Sixteen
scorings of the one-day single-channel base span 0.482-0.543 -- where a generator really
is indistinguishable, the classifier lands at chance every time. The spread appears only
where a separable feature exists, because the classifier either finds it or does not.

So the statistic to report is the MAXIMUM over restarts, read as "the best a post-hoc
classifier managed", with the spread as the honest error bar. A single low reading is
weak evidence of quality; a single high reading is strong evidence of separability.

WHAT IT DECIDES
---------------
Whether the seven-day quality collapse is capacity or data volume. h96 reached 0.9475
at least once, so its samples ARE separable. If h256 stays near chance across every
restart -- the way the one-day models do -- then capacity was the answer and the
seven-day run needs a wider model, not more data.

    qsub -l select=1:ncpus=16:ngpus=1 -l walltime=25:00:00 \
         -o logs/87_disc_stability.log scripts/pbs/87_disc_stability.pbs      # -> glong
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

from cgmoutlier._env import check as _envcheck                      # noqa: E402
_envcheck()

from cgmoutlier.data.cohort import load as load_cohort              # noqa: E402
from cgmoutlier.loo.train import training_set                       # noqa: E402
from cgmoutlier.quality import discriminative_score                 # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True,
                    help="run directories holding samples.npy")
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--job-file", default=None,
                    help="scores against this job's training set; default the whole cohort")
    ap.add_argument("--restarts", type=int, default=8)
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=2026,
                    help="fixed across restarts -- the SUBSAMPLE is held still so that "
                         "only the classifier's initialisation varies")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    X, sids, man = load_cohort(a.cohort)

    def reference(run):
        """The real half must be THIS run's training set, not one shared across --runs.

        --runs is nargs="+" but --job-file was a single value resolved once, before the
        loop. scripts/pbs/87_disc_stability.pbs passed design_d7_pilot/rep1 (553
        subjects) for three runs, one of which -- cp_d7_contiguous_rep1 -- records
        job_file design_multiday/d7_contiguous/rep1 (533 subjects). 38 reference subjects
        were never seen by that model, and the 0.5100 it produced is the anchor the
        report uses to license reading a high discriminator as "failed to fit". The
        mismatch is silent: training_set only raises if a subject is missing from the
        COHORT, and the pilot's subjects are a subset of the contiguous cohort's.
        """
        if a.job_file:
            jf = Path(a.job_file)
        else:
            mp = Path(run) / "meta.json"
            if not mp.exists():
                return X, str(a.cohort)
            jf = Path(json.loads(mp.read_text()).get("job_file", ""))
            if not jf or not jf.exists():
                return X, str(a.cohort)
        job = json.loads(jf.read_text())
        r, _ = training_set(X, sids, job["subjects"])
        return r, str(jf)

    out = {}
    for run in a.runs:
        p = Path(run) / "samples.npy"
        if not p.exists():
            print(f"[disc] {run}: no samples.npy, skipping", flush=True)
            continue
        real, ref = reference(run)
        print(f"[disc] {run}: real {real.shape} against {ref}", flush=True)
        synth = np.load(p)
        accs = []
        for i in range(a.restarts):
            _, acc = discriminative_score(real, synth, n=a.n, seed=a.seed,
                                          init_seed=1000 + i)
            accs.append(acc)
            print(f"[disc] {Path(run).parent.name}/{Path(run).name} "
                  f"init_seed={1000 + i}  acc={acc:.4f}", flush=True)
        v = np.array(accs)
        # record what it was scored against: the old artefacts kept subsample_seed and
        # n_synth but neither the cohort nor the job file, so a mismatch could not be
        # audited after the fact
        out[run] = {"n_restarts": len(v), "accs": [float(x) for x in v],
                    "cohort": str(a.cohort), "reference": ref, "n_real": int(len(real)),
                    "min": float(v.min()), "median": float(np.median(v)),
                    "max": float(v.max()), "spread": float(v.max() - v.min()),
                    "n_synth": int(len(synth)), "subsample_seed": a.seed}
        print(f"[disc] === {run}\n"
              f"       min {v.min():.4f}  median {np.median(v):.4f}  max {v.max():.4f}"
              f"  spread {v.max() - v.min():.4f}", flush=True)

    print(f"\n{'run':46}{'min':>8}{'median':>9}{'max':>8}{'spread':>9}")
    for run, d in out.items():
        print(f"{run[-46:]:46}{d['min']:>8.4f}{d['median']:>9.4f}"
              f"{d['max']:>8.4f}{d['spread']:>9.4f}")
    print("\nRead the MAX column. A generator whose max stays near 0.50 across restarts "
          "is indistinguishable; one that spikes has a feature the classifier can find.")

    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(out, indent=1))
        print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
