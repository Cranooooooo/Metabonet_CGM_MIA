#!/usr/bin/env python
"""How much of the published study rests on composite subjects?

    python scripts/check_collision_impact.py

`scripts/check_id_collisions.py` established that `id` is unique only WITHIN a
`source_file`: 241 of 1,291 ids (18.7%) appear under more than one study, and subject
"102" carries five studies, two CGM devices reporting at the same minute, and two
different basal profiles. The correct subject key is (source_file, id).

Everything in this repository keys on `id` alone, so a composite subject is treated as
one person by the cohort builder, by the outlier consensus, and by the membership
design, which holds a subject in or out as a unit.

FOUR QUESTIONS, in the order they change what has to be redone:

  1. how many of the 875 cohort subjects are composite, and how many windows do they
     hold
  2. are the interleaved windows actually IN the cohort? `complete_days` counts 288
     CGM cells and `build` then takes the first 288 rows in date order. Where the
     second study contributes NaN CGM the interleave makes `isfinite` fail and the day
     is dropped -- harmless. Where BOTH studies report CGM, the window alternates
     between two people and passes every check.
  3. are the consensus outliers enriched for composite ids? If they are, "outliers"
     may partly mean "two people averaged", and the headline AUC 0.680 has a competing
     explanation that has nothing to do with membership.
  4. is subject 1142 -- the only attackable subject, and rank 1 of 875 on real-data
     identifiability -- composite?

THE INTERLEAVE DETECTOR. Two people's traces alternating sample by sample produce a
strong period-2 component: consecutive samples come from different people and jump,
while samples two apart come from the same person and do not. So acf(2) > acf(1),
which never happens in a real five-minute CGM trace (acf falls monotonically over the
first few lags). The statistic is acf(2) - acf(1), and its null is the rest of the
cohort.
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

from cgmoutlier.data.cohort import load as load_cohort

T = 288


def acf_gap(X: np.ndarray) -> np.ndarray:
    """acf(2) - acf(1) per window. Positive is the interleaving signature."""
    Z = (X - X.mean(1, keepdims=True)) / np.maximum(X.std(1, keepdims=True), 1e-6)
    a1 = (Z[:, 1:] * Z[:, :-1]).mean(1)
    a2 = (Z[:, 2:] * Z[:, :-2]).mean(1)
    return a2 - a1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default="data/cohort/metabonet875")
    ap.add_argument("--id-studies", default="results/channel_coverage/id_studies.csv")
    ap.add_argument("--cells", default="results/channel_coverage/cells_per_day.parquet")
    ap.add_argument("--outliers", default="results/outliers/consensus.json")
    ap.add_argument("--identifiability",
                    default="results/identifiability/metabonet875_per_subject.csv")
    ap.add_argument("--flagged", default="1142")
    ap.add_argument("--out", default="results/identifiability/collision_impact.csv")
    a = ap.parse_args()

    pairs = pd.read_csv(a.id_studies, dtype={"id": str})
    nfiles = pairs.groupby("id")["source_file"].nunique()
    studies = pairs.groupby("id")["source_file"].apply(lambda s: "+".join(sorted(s)))

    X, sids, man = load_cohort(a.cohort)
    X = np.asarray(X)[..., 0]
    sids = np.asarray([str(s) for s in sids])
    subs = pd.Index(sorted(set(sids.tolist())))
    nf = nfiles.reindex(subs).fillna(1).astype(int)
    comp = nf > 1

    print("=== 1. composite subjects in the cohort ===")
    print(f"{len(subs)} subjects; {int(comp.sum())} composite ({comp.mean():.1%})")
    print(f"  studies per composite subject: {dict(nf[comp].value_counts())}")
    w = pd.Series(sids).value_counts().reindex(subs).fillna(0).astype(int)
    print(f"windows held by composite subjects: {int(w[comp].sum()):,} of "
          f"{len(sids):,} ({w[comp].sum() / len(sids):.1%})")

    print("\n=== 2. are interleaved windows in the cohort ===")
    g = acf_gap(X)
    owner_comp = comp.reindex(sids).to_numpy()
    print(f"acf(2) - acf(1):  single-study subjects  median {np.median(g[~owner_comp]):+.4f}"
          f"  q99 {np.quantile(g[~owner_comp], 0.99):+.4f}")
    print(f"                  composite subjects     median {np.median(g[owner_comp]):+.4f}"
          f"  q99 {np.quantile(g[owner_comp], 0.99):+.4f}")
    thr = float(np.quantile(g[~owner_comp], 0.999))
    print(f"threshold = single-study q99.9 = {thr:+.4f}")
    for lab, m in (("single-study", ~owner_comp), ("composite", owner_comp)):
        n = int((g[m] > thr).sum())
        print(f"  windows above it, {lab:13}: {n:>6,} of {int(m.sum()):>7,} "
              f"({n / max(int(m.sum()), 1):.2%})")
    n_pos = int((g > 0).sum())
    print(f"windows with acf(2) > acf(1) at all: {n_pos:,} ({n_pos / len(g):.2%}) "
          f"-- impossible for a smooth trace")

    # days where two studies BOTH report CGM are the ones that can interleave silently
    cells = pd.read_parquet(a.cells)
    over = cells["CGM"] > T
    per_sub_over = over.groupby(level=0, observed=True).sum()
    per_sub_over.index = per_sub_over.index.astype(str)
    po = per_sub_over.reindex(subs).fillna(0).astype(int)
    print(f"\nsubjects with >=1 day carrying more than {T} CGM cells: {int((po > 0).sum())}")
    print(f"  such days in total: {int(po.sum()):,}")
    print(f"  all of them composite? {bool((po[po > 0].index.isin(subs[comp])).all())}")

    print("\n=== 3. are the consensus outliers composite ===")
    outl = [str(s) for s in json.loads(Path(a.outliers).read_text())["outliers"]]
    o = pd.Index([s for s in outl if s in subs])
    k = int(comp.reindex(o).sum())
    p = stats.fisher_exact([[k, len(o) - k],
                            [int(comp.sum()) - k,
                             len(subs) - len(o) - (int(comp.sum()) - k)]],
                           alternative="greater")[1]
    print(f"{k} of {len(o)} consensus outliers are composite ({k / max(len(o), 1):.1%}) "
          f"against a base rate of {comp.mean():.1%}; Fisher one-sided p = {p:.4g}")
    print("  " + ", ".join(f"{s}{'*' if comp.get(s, False) else ''}"
                           f"({studies.get(s, '?')})" for s in o[:12]))

    print("\n=== 4. subject 1142 ===")
    f = a.flagged
    print(f"  in cohort: {f in subs} | studies: {studies.get(f, '?')} "
          f"| composite: {bool(comp.get(f, False))}")
    print(f"  days with >{T} CGM cells: {int(po.get(f, 0))}")
    m = sids == f
    if m.any():
        print(f"  windows: {int(m.sum())} | median acf(2)-acf(1) {np.median(g[m]):+.4f} "
              f"| above threshold: {int((g[m] > thr).sum())}")

    if Path(a.identifiability).exists():
        ps = pd.read_csv(a.identifiability, index_col="id")
        ps.index = ps.index.astype(str)
        mr = ps.mean(axis=1).reindex(subs)
        print("\n=== 5. identifiability vs composite status ===")
        print(f"  mean normalised rank, composite   {mr[comp].median():.4f}")
        print(f"                        single      {mr[~comp].median():.4f}")
        print(f"  Mann-Whitney (composite more identifiable) p = "
              f"{stats.mannwhitneyu(mr[comp].dropna(), mr[~comp].dropna(), alternative='less').pvalue:.4g}")

    pd.DataFrame(dict(id=subs, n_studies=nf.to_numpy(), studies=studies.reindex(subs),
                      windows=w.to_numpy(), oversized_days=po.to_numpy(),
                      is_outlier=subs.isin(outl))).to_csv(a.out, index=False)
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
