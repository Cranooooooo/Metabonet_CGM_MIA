"""Multi-channel physiological time-series eval (with missingness).

Same eight metrics as eval_woMissing.py, each routed through its mask-aware
variant. Generated samples are assumed to be fully observed at every (n, t, c)
— so only `--real_mask` is exposed on the CLI. When a metric needs to put
fake under the same missingness regime as real (context_fid, discriminative,
predictive), it draws a synthetic fake-mask internally by resampling from
the real-mask pool. If `--real_mask` is omitted (or fully True), every metric
short-circuits to the no-missing path (parity guarantee).

Usage:
    python eval_wMissing.py \\
        --real real.npy --real_mask real_mask.npy \\
        --fake fake.npy \\
        [--fs 1.0] [--iterations 3] [--metrics ...]
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
from utils.mask_utils   import is_fully_observed
from metrics.context_fid_wMissing       import context_fid_masked
from metrics.cross_correlation_wMissing import cross_correlation_score_masked
from metrics.discriminative_wMissing    import discriminative_score_masked
from metrics.predictive_wMissing        import predictive_score_masked
from metrics.w1_marginal_wMissing       import w1_marginal_masked
from metrics.acf_loss_wMissing          import acf_loss_masked
from metrics.w1_peak_amp_wMissing       import w1_peak_amp_masked
from metrics.w1_ipi_wMissing            import w1_ipi_masked


_METRICS = ["context_fid", "cross_correlation", "discriminative", "predictive",
            "w1_marginal", "acf_loss", "w1_peak_amp", "w1_ipi"]


def _load_mask(path, like_shape):
    if path is None:
        return None
    M = np.load(path)
    M = np.asarray(M).astype(bool)
    if M.shape != like_shape:
        raise ValueError(f"mask shape {M.shape} != data shape {like_shape}")
    return M


def _run_scalar(name: str, fn, iterations: int):
    print(f"\n=== {name} ===")
    scores = []
    for i in range(iterations):
        s = fn()
        if isinstance(s, tuple):
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
    ap.add_argument("--real_mask", default=None,
                    help="Optional. .npy bool array, same shape as --real. "
                         "Omit (or pass an all-True mask) to short-circuit to the "
                         "no-missing pipeline.")
    ap.add_argument("--iterations", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fs", type=float, default=1.0,
                    help="Sampling rate (Hz). Used by IPI metric.")
    ap.add_argument("--max_lag", type=int, default=64, help="ACF max lag.")
    ap.add_argument("--min_overlap", type=int, default=30,
                    help="Pairwise-complete correlation min overlap.")
    ap.add_argument("--peak_kwargs_json", type=str, default=None,
                    help="JSON dict forwarded to scipy.signal.find_peaks.")
    ap.add_argument("--out_json", type=str, default=None,
                    help="If set, dump the full result dictionary as JSON here.")
    ap.add_argument("--metrics", nargs="+", default=_METRICS, choices=_METRICS)
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    real = np.load(args.real).astype(np.float32)
    fake = np.load(args.fake).astype(np.float32)
    real_m = _load_mask(args.real_mask, real.shape)

    print(f"real: {real.shape}  fake: {fake.shape}"
          f"  real_mask: {None if real_m is None else real_m.shape}")
    if real_m is not None:
        print(f"  real observed-rate per sample (mean): {real_m.mean():.4f}")
    if real_m is None or is_fully_observed(real_m):
        print("  (real_mask is fully observed — short-circuiting to no-missing path)")

    peak_kwargs = json.loads(args.peak_kwargs_json) if args.peak_kwargs_json else None

    runners = {
        "context_fid":       lambda: _run_scalar(
            "context_fid",
            lambda: context_fid_masked(real[:], fake[: real.shape[0]],
                                       ori_mask=real_m, seed=args.seed),
            args.iterations),
        "cross_correlation": lambda: _run_scalar(
            "cross_correlation",
            lambda: cross_correlation_score_masked(real[:], fake[: real.shape[0]],
                                                   ori_mask=real_m,
                                                   min_overlap=args.min_overlap),
            args.iterations),
        "discriminative":    lambda: _run_scalar(
            "discriminative",
            lambda: discriminative_score_masked(real[:], fake[: real.shape[0]],
                                                ori_mask=real_m, seed=args.seed),
            args.iterations),
        "predictive":        lambda: _run_scalar(
            "predictive",
            lambda: predictive_score_masked(real, fake[: real.shape[0]],
                                            ori_mask=real_m, seed=args.seed),
            args.iterations),
        "w1_marginal":       lambda: _run_dict(
            "w1_marginal",
            lambda: w1_marginal_masked(real, fake[: real.shape[0]], ori_mask=real_m),
            iterations=1),
        "acf_loss":          lambda: _run_dict(
            "acf_loss",
            lambda: acf_loss_masked(real, fake[: real.shape[0]],
                                    ori_mask=real_m, max_lag=args.max_lag),
            iterations=1),
        "w1_peak_amp":       lambda: _run_dict(
            "w1_peak_amp",
            lambda: w1_peak_amp_masked(real, fake[: real.shape[0]],
                                       ori_mask=real_m, peak_kwargs=peak_kwargs),
            iterations=1),
        "w1_ipi":            lambda: _run_dict(
            "w1_ipi",
            lambda: w1_ipi_masked(real, fake[: real.shape[0]],
                                  ori_mask=real_m,
                                  fs=args.fs, peak_kwargs=peak_kwargs),
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
