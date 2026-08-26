#!/usr/bin/env python
"""Generation quality for a cell's BASE generator, via vendor/tsgen_metrics.

WHY NOT quality.py. That module's `discriminative_score` is one number out of ten, and
it is the only one this campaign has ever reported. "The samples are good but the model
leaks" is a claim about generation quality, and a single GRU classifier cannot support
it -- CC, VDS and FDDS measure distortions a discriminator does not see. tsgen_metrics
carries reference-validated implementations of those three (Diffusion-TS, PaD-TS) plus a
reference-aligned Context-FID, so the quality table has provenance.

quality.py stays, with a different job: `disc_stability.py` runs the 8-restart
discriminator on ALL 27 models per cell to find the ones that did not fit, whose gaps
are then unreadable. That is a per-model gate, not a quality report. The two are not
comparable and are not meant to be -- tsgen_metrics uses the official TimeGAN recipe
(hidden=C//2) while quality.py uses hidden=32.

Quality is a property of the CONFIGURATION, so it is measured on `base` -- the generator
that saw no target at all.

    python scripts/eval_quality_tsgem.py --runs results/runs/matrix_d1_c1/base \
        --cohort data/cohort/matrix_d1_c1 --design results/matrix/design/rep1 \
        --out results/matrix/quality_tsgem/d1_c1.json
"""
import argparse, json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vendor" / "tsgen_metrics"))

from cgmoutlier._env import check as _envcheck          # noqa: E402
_envcheck()
from cgmoutlier.data.cohort import load as load_cohort  # noqa: E402
from cgmoutlier.loo.train import training_set           # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="run dirs holding samples.npy")
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--design", required=True, help="dir holding jobs/<name>.json")
    ap.add_argument("--subsample", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--label", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from tsgen_metrics import evaluate

    X, sids, _ = load_cohort(a.cohort)
    out = {}
    for run in a.runs:
        rp = Path(run)
        sp = rp / "samples.npy"
        if not sp.exists():
            print(f"[tsgem] {run}: no samples.npy, skipping", flush=True)
            continue
        m = json.loads((rp / "meta.json").read_text())
        # the real half must be THIS run's training set, not the whole cohort: a base
        # trained on 475 of 506 subjects is not being asked to reproduce the other 31
        jf = Path(a.design) / "jobs" / f"{m['job']}.json"
        job = json.loads(jf.read_text())
        real, _ = training_set(X, sids, job["subjects"])
        fake = np.load(str(sp))
        print(f"[tsgem] {rp.parent.name}/{rp.name}: real {real.shape} fake {fake.shape} "
              f"against {jf}", flush=True)
        res = evaluate(np.ascontiguousarray(real, np.float32),
                       np.ascontiguousarray(fake, np.float32),
                       subsample=a.subsample, seed=a.seed, verbose=True)
        rec = {k: v.get("value") for k, v in res["metrics"].items()}
        rec.update(run=str(rp), job=m["job"], n_real=int(len(real)), n_fake=int(len(fake)),
                   subsample=a.subsample, seed=a.seed, label=a.label)
        out[str(rp)] = rec
        print("  " + "  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                               for k, v in rec.items() if isinstance(v, (int, float))),
              flush=True)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(f"\nwrote {a.out}   (all 10 metrics: lower = better)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
