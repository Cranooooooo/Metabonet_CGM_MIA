"""Multi-channel physiological time-series eval (no missingness).

Runs eight fidelity metrics tailored for multi-channel physiological signals:

    Classics (kept verbatim from the eval_ref_1 protocol)
        1. context_fid          — TS2Vec embedding-space Frechet
        2. cross_correlation    — lag-0 channel-pair correlation matrix MAE
        3. discriminative       — |0.5 - acc| of a post-hoc GRU classifier
        4. predictive           — TSTR one-step-ahead MAE on real

    Physiological supplements (per-channel + channel-mean)
        5. w1_marginal          — per-channel W1 on flattened values
        6. acf_loss             — per-channel multi-lag autocorrelation L2
        7. w1_peak_amp          — per-channel W1 on detected peak amplitudes
        8. w1_ipi               — per-channel W1 on inter-peak intervals

Each metric is run `--iterations` times (those that are stochastic — the
deterministic ones run only once); per-channel results are kept in addition
to the channel-mean. Format mirrors eval_ref_1/woMissing_scripts/eval.py.

Usage:
    python eval_woMissing.py --real real.npy --fake fake.npy [--fs 1.0] \\
        [--iterations 3] [--metrics context_fid cross_correlation ...]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from utils.metric_utils import display_scores
from metrics.context_fid_woMissing       import context_fid
from metrics.cross_correlation_woMissing import cross_correlation_score
from metrics.discriminative_woMissing    import discriminative_score
from metrics.predictive_woMissing        import predictive_score
from metrics.w1_marginal_woMissing       import w1_marginal
from metrics.acf_loss_woMissing          import acf_loss
from metrics.w1_peak_amp_woMissing       import w1_peak_amp
from metrics.w1_ipi_woMissing            import w1_ipi


_METRICS = ["context_fid", "cross_correlation", "discriminative", "predictive",
            "w1_marginal", "acf_loss", "w1_peak_amp", "w1_ipi"]


# ───── runners ─────

def _run_scalar(name: str, fn, iterations: int):
    print(f"\n=== {name} ===")
    scores = []
    for i in range(iterations):
        s = fn()
        if isinstance(s, tuple):  # discriminative returns (score, fake_acc, real_acc)
            s = s[0]
        scores.append(float(s))
        print(f"  iter {i}: {name} = {s:.6f}")
    mean, ci = display_scores(scores)
    return {"per_iter": scores, "mean": mean, "ci95": ci}


def _run_dict(name: str, fn, iterations: int):
    print(f"\n=== {name} ===")
    iters = []
    for i in range(iterations):
        d = fn()
        iters.append(d)
        print(f"  iter {i}: {name}.mean = {d['mean']:.6f}")
    means = [d["mean"] for d in iters]
    mean, ci = display_scores(means)
    pc_stack = np.stack([np.asarray(d["per_channel"], dtype=float) for d in iters], axis=0)
    return {
        "per_iter": [d["per_channel"] for d in iters],
        "per_channel_mean": np.nanmean(pc_stack, axis=0).tolist(),
        "mean": mean,
        "ci95": ci,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", required=True)
    ap.add_argument("--fake", required=True)
    ap.add_argument("--iterations", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fs", type=float, default=1.0,
                    help="Sampling rate (Hz). Used by IPI metric.")
    ap.add_argument("--max_lag", type=int, default=64, help="ACF max lag.")
    ap.add_argument("--peak_kwargs_json", type=str, default=None,
                    help="JSON dict of kwargs forwarded to scipy.signal.find_peaks.")
    ap.add_argument("--out_json", type=str, default=None,
                    help="If set, dump the full result dictionary as JSON here.")
    ap.add_argument("--metrics", nargs="+", default=_METRICS, choices=_METRICS)
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    real = np.load(args.real).astype(np.float32)
    fake = np.load(args.fake).astype(np.float32)
    print(f"real: {real.shape}  fake: {fake.shape}")

    peak_kwargs = json.loads(args.peak_kwargs_json) if args.peak_kwargs_json else None

    runners = {
        "context_fid":       lambda: _run_scalar(
            "context_fid",
            lambda: context_fid(real[:], fake[: real.shape[0]]),
            args.iterations),
        "cross_correlation": lambda: _run_scalar(
            "cross_correlation",
            lambda: cross_correlation_score(real[:], fake[: real.shape[0]]),
            args.iterations),
        "discriminative":    lambda: _run_scalar(
            "discriminative",
            lambda: discriminative_score(real[:], fake[: real.shape[0]]),
            args.iterations),
        "predictive":        lambda: _run_scalar(
            "predictive",
            lambda: predictive_score(real, fake[: real.shape[0]]),
            args.iterations),
        "w1_marginal":       lambda: _run_dict(
            "w1_marginal",
            lambda: w1_marginal(real, fake[: real.shape[0]]),
            iterations=1),  # deterministic
        "acf_loss":          lambda: _run_dict(
            "acf_loss",
            lambda: acf_loss(real, fake[: real.shape[0]], max_lag=args.max_lag),
            iterations=1),
        "w1_peak_amp":       lambda: _run_dict(
            "w1_peak_amp",
            lambda: w1_peak_amp(real, fake[: real.shape[0]], peak_kwargs=peak_kwargs),
            iterations=1),
        "w1_ipi":            lambda: _run_dict(
            "w1_ipi",
            lambda: w1_ipi(real, fake[: real.shape[0]], fs=args.fs, peak_kwargs=peak_kwargs),
            iterations=1),
    }

    results = {}
    for name in args.metrics:
        results[name] = runners[name]()

    if args.out_json:
        with open(args.out_json, "w") as fh:
            json.dump(results, fh, indent=2)
        print(f"\nWrote {args.out_json}")


if __name__ == "__main__":
    main()
