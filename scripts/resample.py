#!/usr/bin/env python
"""Release a new sample set from an already-trained run. No training.

Two uses, one path:

    # rescue: training finished, the sampling pass was killed
    python scripts/resample.py \
        --from-run results/runs/dimts_h96_rep1/base \
        --job-file results/design_sym/rep1/jobs/base.json

    # the denoising-step question: same model, fewer steps, a DIFFERENT released set
    python scripts/resample.py \
        --from-run results/runs/dimts_h128_rep1/base \
        --out     results/runs/dimts_h128_rep1_st50/base \
        --job-file results/design_sym/rep1/jobs/base.json \
        --sampling-timesteps 50

`--out` defaults to `--from-run`, which is the rescue case: the run gets the sample
file it was always meant to have. Give it a new directory for anything that is a
different released set rather than a repair -- a step count writes samples that are
not the ones the run's own meta.json describes, and overwriting in place would leave
nothing to compare against.

WHY THIS EXISTS AT ALL. Sampling is ~40% of a run's cost at h=128 and the checkpoints
are the other 60%; a killed sampling loop used to mean retraining. It is also the only
honest way to answer `sampling_timesteps` 500 -> 50, since retraining for it would vary
the weights as well as the steps.
"""
import argparse
import sys

from cgmoutlier._env import check as _envcheck                     # noqa: E402
_envcheck()
from cgmoutlier.loo.train import resample                             # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-run", required=True,
                    help="a run directory holding ckpt_<T>/ and train.npy")
    ap.add_argument("--job-file", required=True,
                    help="the design job these checkpoints were trained on; its "
                         "subject list is checked against the run's own train.npy")
    ap.add_argument("--out", default=None,
                    help="where samples.npy goes; default --from-run (the rescue case)")
    ap.add_argument("--cohort", default="data/cohort/metabonet875")
    ap.add_argument("--generator", default=None,
                    help="default: the run's meta.json, else dimts")
    ap.add_argument("--K", type=int, default=None,
                    help="released samples; default = the job's training-set size, "
                         "which is what the original run released")
    ap.add_argument("--sampling-timesteps", type=int, default=None,
                    help="denoising steps at sample time; default 500 (the full loop, "
                         "what every existing run used). Below it, DDIM")
    ap.add_argument("--milestone", type=int, default=None,
                    help="which checkpoint-<n>.pt; default the highest, which is the "
                         "end of training for a run that finished it")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()

    resample(a.from_run, a.job_file, a.cohort, out=a.out, generator=a.generator,
             K=a.K, sampling_timesteps=a.sampling_timesteps, milestone=a.milestone,
             seed=a.seed, device=a.device, overwrite=a.overwrite)
    return 0


if __name__ == "__main__":
    sys.exit(main())
