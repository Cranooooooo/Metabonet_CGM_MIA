#!/usr/bin/env python
"""Pick the distance variant, on evidence, from an existing gaps.parquet.

    python scripts/choose_variant.py --generator copy_paste

Reads what `run_attack.py` already wrote -- no distances are recomputed -- and reports
the three things the choice should rest on:

  1. DETECTION. Is the gap positive within each arm (Wilcoxon)? A variant that cannot
     see membership on copy_paste, which memorises by construction, is out. On the
     first run this eliminated every `set_reduce=mean` variant: adding one subject to
     178k released samples does not move the mean distance to the released set.

  2. DAY-COUNT DEPENDENCE. `subject_reduce=min` takes the single most exposed window,
     so it falls as a subject contributes more windows -- more draws, a lower minimum
     -- whether or not the subject was a member. Controls are matched on day count
     precisely to keep that out of the arm comparison (docs/DESIGN.md), but a variant
     that is strongly day-driven spends the matching's budget on itself. The
     correlation is measured WITHIN the control arm, where every gap is the same kind
     of quantity and the only thing varying is the subject.

  3. SPREAD. The between-arm test has 24 against 24. A variant whose control-arm gaps
     are widely dispersed needs a larger real effect to clear its own noise; the
     reported `cohen_d`-style ratio is the outlier-minus-control difference over the
     pooled spread, which is what actually determines whether 24 targets can resolve
     anything.

None of these is decided by "which gap is biggest". A larger gap on a generator that
memorises everything says nothing about resolving a smaller effect on one that does not.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from cgmoutlier._env import check as _envcheck                     # noqa: E402
_envcheck()
from cgmoutlier.attack.statistic import summarise                    # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generator", default="copy_paste")
    ap.add_argument("--attack", default=None, help="default results/attack/<generator>")
    a = ap.parse_args()

    d = Path(a.attack) if a.attack else Path("results/attack") / a.generator
    p = d / "gaps.parquet"
    if not p.exists():
        sys.exit(f"no {p}; run scripts/run_attack.py --generator {a.generator} first")
    df = pd.read_parquet(p)

    S = summarise(df.to_dict("records")).set_index(["set_reduce", "subject_reduce"])

    rows = []
    for (sr, br), g in df.groupby(["set_reduce", "subject_reduce"]):
        o = g[g.group == "outlier"]
        c = g[g.group == "control"]
        # day-count dependence, measured inside the control arm only
        rho, prho = spearmanr(c.n_windows, c.gap)
        pooled = np.sqrt((o.gap.var(ddof=1) + c.gap.var(ddof=1)) / 2)
        rows.append(dict(
            set_reduce=sr, subject_reduce=br,
            detects=bool(S.loc[(sr, br), "p_within_outlier"] < 0.01
                         and S.loc[(sr, br), "p_within_control"] < 0.01),
            median_gap=float(np.median(g.gap)),
            rho_gap_vs_ndays=round(float(rho), 3), p_rho=round(float(prho), 4),
            control_sd=float(c.gap.std(ddof=1)),
            effect_over_spread=float((o.gap.mean() - c.gap.mean()) / pooled)
            if pooled else np.nan,
            auc=float(S.loc[(sr, br), "auc"]),
        ))

    R = pd.DataFrame(rows).sort_values(["detects", "median_gap"], ascending=False)
    pd.set_option("display.width", 200)
    print(R.to_string(index=False))
    (d / "variant_choice.json").write_text(json.dumps(rows, indent=2))

    ok = R[R.detects]
    print("\ndetects        gap positive in BOTH arms at p < 0.01 -- the known-positive bar")
    print("rho_...        Spearman(gap, n_windows) inside the control arm; nearer 0 is")
    print("               less of the subject's day count leaking into the statistic")
    print("effect_...     (mean outlier gap - mean control gap) / pooled sd")
    if len(ok):
        best = ok.iloc[(ok.rho_gap_vs_ndays.abs()).argsort()].iloc[0]
        print(f"\nleast day-driven variant that detects membership: "
              f"{best.set_reduce} x {best.subject_reduce} "
              f"(rho={best.rho_gap_vs_ndays})")
    else:
        print("\nNO variant detects membership. The statistic, not the generator, is "
              "what to look at.")
    print(f"\nwrote {d/'variant_choice.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
