#!/usr/bin/env python
"""The OTHER endpoint of the privacy-quality curve: a generator that ignores the data.

The curve has two anchors and neither is a real generator:

    copy_paste    replays training windows verbatim -> quality perfect, privacy worst
    random        ignores the training set entirely -> privacy perfect, quality worst

Everything a real generator can do lies between them, so both have to be on the plot or
the axis has no scale. `copy_paste` already exists as a registered generator; this makes
the other end.

WHAT "RANDOM" MEANS HERE, and why not something even simpler. Pure white noise on an
arbitrary scale would score badly for a trivial reason -- wrong units -- and would make
the endpoint look further away than it is. This draws iid Gaussian noise matched to the
cohort's PER-CHANNEL mean and standard deviation, so the released set has the right
marginal scale and nothing else: no temporal structure, no individual, no distribution
shape beyond the first two moments. That is the honest "knows nothing about the data"
endpoint, and it is the harder (more conservative) version of the anchor.

The constants come from the cohort's own manifest, which every model in the comparison
was also normalised by, so no information about any individual crosses over.

It writes 27 run directories with the same layout the real generators produce, so
run_attack.py / subject_auc.py / eval_quality_tsgem.py all work on it unchanged. Every
"model" is independent noise, so include_t and base differ by nothing that depends on t
and the attack should read AUC ~ 0.5 by construction. If it does not, the attack has a
bug -- which makes this a negative control as well as an endpoint.

    python scripts/make_random_release.py --cohort data/cohort/matrix_d1_c1 \
        --design results/matrix/design/rep1 --out results/runs/random_d1_c1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cgmoutlier.data.cohort import load as load_cohort       # noqa: E402
from cgmoutlier.loo import training_set                      # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--design", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args()

    X, sids, man = load_cohort(a.cohort)
    X = np.asarray(X)
    T, C = X.shape[1], X.shape[2]
    # per-channel first two moments of the WHOLE cohort -- the same constants the
    # manifest already publishes, not anything derived from a target
    mu = X.reshape(-1, C).mean(0)
    sd = X.reshape(-1, C).std(0)
    print(f"cohort {a.cohort}  T={T} C={C}  per-channel mean {mu.round(4)} sd {sd.round(4)}")

    jobs = sorted(p.stem for p in (Path(a.design) / "jobs").glob("*.json"))
    out = Path(a.out)
    for i, name in enumerate(jobs):
        j = json.loads((Path(a.design) / "jobs" / f"{name}.json").read_text())
        real, _ = training_set(X, sids, j["subjects"])
        K = len(real)
        rng = np.random.default_rng(a.seed + 1000 * i)
        S = (rng.standard_normal((K, T, C)).astype(np.float32) * sd + mu).astype(np.float32)
        d = out / name
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "samples.npy", S)
        (d / "meta.json").write_text(json.dumps({
            "job": name, "role": j["role"], "target": j["target"], "group": j["group"],
            "job_file": str(Path(a.design) / "jobs" / f"{name}.json"),
            "n_subjects": j["n_subjects"], "n_train_windows": K, "K": K,
            "T": T, "C": C, "generator": "random_gaussian",
            "params": {"seed": int(a.seed + 1000 * i),
                       "note": "iid N(mu, sd) per channel, cohort constants only"},
            "cohort": a.cohort, "fit_seconds": 0.0, "sample_seconds": 0.0,
            "sample_range": [float(S.min()), float(S.max())]}, indent=2))
        print(f"  {name:28s} K={K:,}  range [{S.min():.3f}, {S.max():.3f}]")
    print(f"wrote {len(jobs)} random releases to {out}")


if __name__ == "__main__":
    main()
