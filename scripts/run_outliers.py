#!/usr/bin/env python
"""Stage 2: score every subject for atypicality, fourteen ways."""
import argparse, json, os, sys
from pathlib import Path

from cgmoutlier._env import check as _envcheck                # noqa: E402
_envcheck()
from cgmoutlier.outliers.common import SEED                      # noqa: E402
from cgmoutlier.outliers.run import (ALL, CONTROLS, N_CANDIDATES,   # noqa: E402
                                     consensus, run)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default="data/cohort/metabonet875")
    ap.add_argument("--out", default="results/outliers")
    ap.add_argument("--only", default=None)
    ap.add_argument("--dtw-per-subject", type=int, default=30)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--top-pct", type=float, default=5.0)
    ap.add_argument("--min-methods", type=int, default=7)
    ap.add_argument("--expect-candidates", type=int, default=None,
                    help="fail unless this many methods voted; defaults to "
                         "the size of the default set minus controls")
    ap.add_argument("--seed", type=int, default=SEED,
                    help="base seed; each method derives its own stream from it and "
                         "its own key, so a rerun of one method is exact")
    ap.add_argument("--channels", default="all",
                    help="'all', or a comma-separated list of channel names. Group A "
                         "is CGM-only whatever this says -- its metrics are defined on "
                         "glucose -- so the denominator stays 13 either way")
    ap.add_argument("--clinical-cache", default=None,
                    help="reuse a _clinical.parquet from another run on the SAME "
                         "cohort; group A is bit-identical across channel sets")
    a = ap.parse_args()
    run(a.cohort, a.out, only=(a.only.split(",") if a.only else None),
        dtw_per_subject=a.dtw_per_subject, device=a.device, seed=a.seed,
        channels=a.channels, clinical_cache=a.clinical_cache)
    expect = a.expect_candidates
    if expect is None:
        expect = (len([k for k in a.only.split(',') if k not in CONTROLS])
                  if a.only else N_CANDIDATES)
    c = consensus(a.out, a.top_pct, a.min_methods, expect=expect)
    Path(a.out, "consensus.json").write_text(json.dumps(c, indent=2))
    print(f"\n{len(c['outliers'])} consensus outliers "
          f"(>= {c['min_methods']}/{c['n_candidates']} at top-{c['cut_pct']}%)")
    print(" ", " ".join(c["outliers"]))


if __name__ == "__main__":
    sys.exit(main())
