#!/usr/bin/env python3
"""Validate tsgen_metrics' deterministic metrics against the published references.

On an identical (real, fake) pair, compute each reference-backed metric BOTH with
tsgen_metrics AND with the original published implementation, and report the
relative difference. The deterministic metrics should match (correlational / FDDS
exactly; VDS within the reference's 10000-sample stochasticity).

References (point at your local checkouts):
  --diffts-dir  Diffusion-TS repo (uses Utils/cross_correlation.py: CrossCorrelLoss)
  --padts-dir   PaD-TS repo        (uses eval_utils/MMD.py: VDS_Naive, BMMD_Naive,
                                    cross_correlation_distribution)

    python validate_against_reference.py --real real.npy --fake fake.npy \
        --diffts-dir /path/to/Diffusion-TS --padts-dir /path/to/PaD-TS

Neural metrics (context_fid/discriminative/predictive) are intentionally NOT
validated here: they are stochastic trained nets and tsgen's versions are proxies,
not faithful to the official TS2Vec/TF-GRU. For official neural values, run the
DiMTS/PaD-TS reference directly.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tsgen_metrics import statistical              # noqa: E402
from tsgen_metrics.core import _subsample_pair      # noqa: E402


def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", required=True)
    ap.add_argument("--fake", required=True)
    ap.add_argument("--diffts-dir", required=True, help="Diffusion-TS checkout (for CrossCorrelLoss)")
    ap.add_argument("--padts-dir", required=True, help="PaD-TS checkout (for MMD.py)")
    ap.add_argument("--subsample", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    real = np.load(args.real)
    fake = np.load(args.fake)
    real_sub, fake_sub = _subsample_pair(real, fake, args.subsample, args.seed)

    # reference imports
    mmd = _load_module(Path(args.padts_dir) / "eval_utils" / "MMD.py", "padts_mmd")
    sys.path.insert(0, str(args.diffts_dir))
    from Utils.cross_correlation import CrossCorrelLoss   # noqa: E402

    def ref_corr(r, f):
        loss = CrossCorrelLoss(torch.tensor(r).float(), name="cc")
        return float(loss.compute(torch.tensor(f).float()).mean().item())

    def ref_vds(r, f):
        np.random.seed(args.seed)
        return float(mmd.VDS_Naive(torch.tensor(r).float(), torch.tensor(f).float(), "rbf").mean())

    def ref_fdds(r, f):
        cr = mmd.cross_correlation_distribution(torch.tensor(r).float()).unsqueeze(-1)
        cf = mmd.cross_correlation_distribution(torch.tensor(f).float()).unsqueeze(-1)
        return float(mmd.BMMD_Naive(cr, cf, "rbf").mean())

    rows = [
        ("correlational", statistical.correlational_score(real_sub, fake_sub)["value"], ref_corr(real_sub, fake_sub)),
        ("vds",  statistical.value_distribution_shift(real_sub, fake_sub, seed=args.seed)["value"], ref_vds(real_sub, fake_sub)),
        ("fdds", statistical.functional_dependency_distribution_shift(real_sub, fake_sub, seed=args.seed)["value"], ref_fdds(real_sub, fake_sub)),
    ]
    print(f"{'metric':<16}{'tsgen':>12}{'reference':>12}{'rel.diff':>10}")
    print("-" * 50)
    ok = True
    for name, ours, ref in rows:
        rel = abs(ours - ref) / max(abs(ref), 1e-9)
        flag = "" if rel < 0.10 else "  <-- CHECK"
        if rel >= 0.10:
            ok = False
        print(f"{name:<16}{ours:>12.5f}{ref:>12.5f}{rel*100:>9.1f}%{flag}")
    print("\nOK: all reference-backed metrics within 10%." if ok else
          "\nWARNING: a metric diverged >10% from reference.")


if __name__ == "__main__":
    main()
