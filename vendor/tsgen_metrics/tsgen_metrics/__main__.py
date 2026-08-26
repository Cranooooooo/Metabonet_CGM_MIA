"""CLI: evaluate generated time series against real.

    python -m tsgen_metrics --real real.npy --fake fake.npy --out scores.json

real.npy / fake.npy are float32 arrays shaped (N, T, C) in the SAME value space.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from .core import evaluate


def main(argv=None):
    ap = argparse.ArgumentParser(description="Time-series generation-quality metrics (10 metrics, lower=better).")
    ap.add_argument("--real", required=True, help="real .npy, shape (N,T,C)")
    ap.add_argument("--fake", required=True, help="generated .npy, shape (N,T,C)")
    ap.add_argument("--out", default=None, help="write full result JSON here")
    ap.add_argument("--subsample", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-seeds", dest="n_seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=40, help="proxy-neural training epochs")
    ap.add_argument("--device", default=None, help="cuda | cpu (default: auto)")
    ap.add_argument("--no-neural", action="store_true", help="skip proxy neural metrics")
    args = ap.parse_args(argv)

    real = np.load(args.real)
    fake = np.load(args.fake)
    res = evaluate(real, fake, subsample=args.subsample, seed=args.seed,
                   n_seeds=args.n_seeds, epochs=args.epochs, device=args.device,
                   include_neural=not args.no_neural, verbose=True)

    print(f"\n{'abbrev':<13}{'value':>12}{'±std':>10}  status")
    print("-" * 56)
    for key, m in res["metrics"].items():
        std = m.get("std")
        std_s = f"{std:.4f}" if std is not None else ""
        print(f"{m.get('abbrev', key):<13}{m['value']:>12.5f}{std_s:>10}  {m['status']}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
