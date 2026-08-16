#!/usr/bin/env python
"""Does real-data identifiability predict who the membership attack found?

    python scripts/identifiability_vs_outliers.py

`scripts/identifiability.py` measures, per subject and without any generator, how well
that subject's real days retrieve each other against all other subjects. The membership
study produced two findings to test it against:

  1. subject 1142 is the ONLY attackable subject (per-subject AUC 0.717 under the panel,
     +0.152 net of its own control, 17.8 SD above the control mean)
  2. consensus outliers leak more than day-matched controls in aggregate (AUC 0.680),
     while 79 of 80 individual subjects sit at 0.50-0.52

If real-data identifiability ranks 1142 at the extreme, then the attack found something
already present in the raw data and needed no generator to reveal it -- which makes the
ceiling measurement, not the attack, the cheap instrument for this question. If 1142 is
unremarkable here, the leak is a property of the generator's fit to that subject and
the two measurements are independent.

The outlier test is the aggregate version of the same question. Note the direction: an
outlier is a subject far from the others, which should make it EASIER to retrieve, so
the expected sign is negative mean rank. A null result here says the consensus outlier
definition does not pick out subjects that are distinguishable day-to-day.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from cgmoutlier._env import check as _envcheck                      # noqa: E402
_envcheck()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-subject", nargs="+",
                    default=["results/identifiability/metabonet875_per_subject.csv"])
    ap.add_argument("--outliers", default="results/outliers/consensus.json")
    ap.add_argument("--flagged", default="1142",
                    help="the subject the attack found; checked individually")
    ap.add_argument("--out", default="results/identifiability/vs_outliers.csv")
    a = ap.parse_args()

    outl = set(json.loads(Path(a.outliers).read_text())["outliers"])
    rows = []
    for path in a.per_subject:
        ps = pd.read_csv(path, index_col="id")
        ps.index = ps.index.astype(str)
        # The outlier consensus and the flagged subject were recorded under the OLD key
        # (bare `id`). Cohorts rebuilt on (source_file, id) index as "Loop/1142", so the
        # label is matched on the id part. 22 of the 24 consensus outliers are
        # single-study and map one-to-one; the two composites map to several subjects
        # each and are counted as outliers in all of them, which is the conservative
        # direction for a test that asks whether outliers are MORE identifiable.
        bare = (ps.index.str.rsplit("/", n=1).str[-1] if ps.index.str.contains("/").any()
                else ps.index)
        if bare is not ps.index:
            amb = sum(1 for o in outl if (bare == o).sum() > 1)
            print(f"[map] {path}: keyed by (study, id); {amb} of {len(outl)} consensus "
                  f"outliers match more than one subject")
        name = Path(path).name.replace("_per_subject.csv", "")
        print(f"\n=== {name}: {len(ps)} subjects, {len(ps.columns)} spaces ===")
        print(f"{'space':22}{'1142 rank':>11}{'pctile':>9}{'z':>8}"
              f"{'outlier med':>13}{'normal med':>12}{'MWU p':>10}")
        obare = pd.Series(np.asarray(bare), index=ps.index)
        for col in ps.columns:
            v = ps[col].dropna()
            if v.empty:
                continue
            # lower normalised rank = more identifiable
            is_o = obare.reindex(v.index).isin(outl).to_numpy()
            o, n = v[is_o], v[~is_o]
            p = (stats.mannwhitneyu(o, n, alternative="less").pvalue
                 if len(o) and len(n) else np.nan)
            hit = v[obare.reindex(v.index).to_numpy() == a.flagged]
            if len(hit):
                x = float(hit.iloc[0])       # single-study, so at most one in practice
                pct = float((v < x).mean())
                z = float((x - v.mean()) / max(v.std(), 1e-12))
            else:
                x = pct = z = np.nan
            print(f"  {col:20}{x:>11.4f}{pct:>9.1%}{z:>8.2f}"
                  f"{o.median():>13.4f}{n.median():>12.4f}{p:>10.4f}")
            rows.append(dict(cohort=name, space=col, flagged=a.flagged,
                             flagged_rank=x, flagged_pctile=pct, flagged_z=z,
                             n_outliers=int(is_o.sum()),
                             outlier_median=float(o.median()) if len(o) else np.nan,
                             normal_median=float(n.median()) if len(n) else np.nan,
                             mwu_p=p))

        # the most identifiable subjects overall, whether or not they are outliers
        best = ps.mean(axis=1).sort_values()
        print(f"\n  most identifiable (mean normalised rank over spaces; "
              f"* = consensus outlier):")
        for sid, val in best.head(12).items():
            print(f"    {str(sid):>14} {val:.4f} "
                  f"{'*' if obare.get(sid) in outl else ''}")
        pos = [i for i, s in enumerate(best.index) if obare.get(s) == a.flagged]
        print(f"    ... {a.flagged} sits at position "
              f"{pos[0] + 1} of {len(best)}" if pos
              else f"    ... {a.flagged} is not in this cohort")

    pd.DataFrame(rows).to_csv(a.out, index=False)
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
