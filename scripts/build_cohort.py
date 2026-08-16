#!/usr/bin/env python
"""Stage 1: raw parquet -> a cohort of complete 24 h CGM windows.

    python scripts/build_cohort.py --config configs/data.yaml

The committed cohort under `data/cohort/metabonet875` was produced by exactly this,
so you only need to run it to build a DIFFERENT one -- a different day floor, a
different subject count, a different draw.

Three things this stage fixes that everything downstream depends on:

* **A day is kept only if all 288 cells are present.** The consensus guidance accepts
  70% capture; interpolating the other 30% would put invented curve shape in front of
  methods whose entire job is to judge curve shape.
* **Normalisation constants are computed from THIS cohort** and written to the
  manifest. Inheriting another cohort's mean and sd shifts every mg/dL number --
  time-in-range, GMI, the CV threshold -- with nothing looking broken.
* **The subject draw is stratified on day count.** Drawing uniformly would
  over-represent short records, and day count already correlates with several outlier
  scores; the stratification keeps that confound from being built in at stage 1.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cgmoutlier._env import check as _envcheck   # noqa: E402
_envcheck()
from cgmoutlier.data.cohort import build          # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--parquet", default=None, help="override config source.parquet")
    ap.add_argument("--out", default=None, help="override output directory")
    ap.add_argument("--n-subjects", type=int, default=None)
    ap.add_argument("--min-days", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    a = ap.parse_args()

    import yaml
    cfg = yaml.safe_load(Path(a.config).read_text())
    parquet = a.parquet or cfg["source"]["parquet"]
    out = a.out or f"data/cohort/{Path(parquet).stem}"
    n_subjects = a.n_subjects if a.n_subjects is not None else cfg["cohort"]["n_subjects"]
    min_days = a.min_days if a.min_days is not None else cfg["cohort"]["min_days"]
    seed = a.seed if a.seed is not None else cfg["cohort"]["seed"]

    if not Path(parquet).exists():
        print(f"no raw parquet at {parquet}\n"
              f"  python scripts/fetch_data.py --check", file=sys.stderr)
        return 1
    if Path(out).exists() and any(Path(out).iterdir()):
        print(f"{out} is not empty; refusing to overwrite a cohort.\n"
              f"  Every score in results/ was computed against some cohort, and there\n"
              f"  is nothing in a score file that records which one.", file=sys.stderr)
        return 1

    build(parquet, out, n_subjects=n_subjects, min_days=min_days, seed=seed)
    print(f"\nnext:  python scripts/run_outliers.py --cohort {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
