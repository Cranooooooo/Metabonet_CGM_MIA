#!/usr/bin/env python
"""Upper bound on any membership attack: can a day be traced to its subject at all?

    python scripts/identifiability.py --cohort data/cohort/metabonet834_c3

No generator is involved. Every window here is REAL, and the question is the one that
sits underneath the whole study: given one real day, how well can it be matched to the
subject it came from, using the other real days? If a day of CGM does not identify its
own subject, then no generator trained on that subject can leak the identity either,
and the flat attack results are a property of the data rather than of the attack. This
is a ceiling, not an attack: the attacker here is handed the real windows.

WHAT THE SPACES SEPARATE. Identifiability that lives in the daily mean is not the same
finding as identifiability that lives in the shape of the curve. A generator that only
matches marginal distributions already reproduces the first and leaks nothing further,
so `level` (two numbers per channel) is the null that `shape` has to beat.

  level      per-channel mean and sd -- the trivially personal part
  shape      per-channel z-scored trace; level removed, curve kept
  raw        the trace as stored; level and shape together
  quantile   11 quantiles per channel -- the day's distribution, time discarded
  acf        autocorrelation to lag 36 (3 h) -- temporal texture, level discarded
  spectrum   log power in 16 bands -- the same texture in frequency

WHAT THE CHANNEL SUBSETS SEPARATE. Run on a multi-channel cohort, each space is
evaluated on CGM alone and on all channels, over THE SAME SUBJECTS AND THE SAME DAYS.
The difference is then attributable to channel content and not to cohort size, which
is the comparison that decides whether retraining the generators on more channels is
worth the GPU time.

METRIC. Leave-one-day-out retrieval: hold out a day, rank every subject by the distance
from that day to their nearest other day, and record where the true subject lands.
Chance top-1 is 1/n_subjects (0.12% at 834). AUC is 1 - mean normalised rank, so 0.5 is
chance and 1.0 is perfect, and it is the quantity comparable to the attack AUCs already
in `results/attack_panel/`.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from cgmoutlier._env import check as _envcheck                      # noqa: E402
_envcheck()

from cgmoutlier.data.cohort import load as load_cohort

SPACES = ["level", "shape", "raw", "quantile", "acf", "spectrum"]


def featurise(X: np.ndarray, space: str) -> np.ndarray:
    """(n, T, C) -> (n, d). Every space is per-channel and then concatenated."""
    n, T, C = X.shape
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True)
    if space == "level":
        return np.concatenate([mu.reshape(n, C), sd.reshape(n, C)], axis=1)
    if space == "raw":
        return X.reshape(n, T * C)
    if space == "shape":
        return ((X - mu) / np.maximum(sd, 1e-6)).reshape(n, T * C)
    if space == "quantile":
        q = np.quantile(X, np.linspace(0, 1, 11), axis=1)        # (11, n, C)
        return q.transpose(1, 0, 2).reshape(n, -1)
    if space == "acf":
        Z = (X - mu) / np.maximum(sd, 1e-6)
        return np.stack([(Z[:, k:] * Z[:, :-k]).mean(axis=1)
                         for k in range(1, 37)], axis=1).reshape(n, -1)
    if space == "spectrum":
        P = np.abs(np.fft.rfft((X - mu), axis=1)) ** 2            # level -> DC, dropped
        edges = np.linspace(1, P.shape[1], 17).astype(int)
        return np.log1p(np.stack([P[:, a:b].sum(axis=1)
                                  for a, b in zip(edges[:-1], edges[1:])],
                                 axis=1)).reshape(n, -1)
    raise ValueError(space)


def retrieve(F: np.ndarray, starts: np.ndarray, owner: np.ndarray, chunk: int = 1024):
    """Leave-one-out nearest-subject retrieval over windows sorted by subject.

    Returns (mid-rank of the true subject, n_subjects). Rank 0 means strictly nearest.

    TIES ARE COUNTED AS HALF. `bolus` is 95% exact zeros, so its quantile vector is
    identical for many subjects and their distances are exactly equal. Scoring a tie as
    rank 0 -- which `(Ds < true).sum()` alone does -- turns "indistinguishable from 40
    other people" into a correct top-1 identification, and reported 62% top-1 on
    bolus quantile. The mid-rank convention is also what makes 1 - mean(rank) the
    Mann-Whitney AUC rather than an optimistic bound on it.
    """
    F = np.ascontiguousarray(F, dtype=np.float32)
    F = (F - F.mean(0)) / np.maximum(F.std(0), 1e-8)   # spaces have unequal dimensions
    sq = (F ** 2).sum(1)
    n, S = F.shape[0], starts.size
    ranks = np.empty(n, dtype=np.float32)
    for i in range(0, n, chunk):
        j = min(i + chunk, n)
        D = sq[i:j, None] + sq[None, :] - 2.0 * (F[i:j] @ F.T)
        D[np.arange(j - i), np.arange(i, j)] = np.inf   # a day never retrieves itself
        Ds = np.minimum.reduceat(D, starts, axis=1)     # (chunk, n_subjects)
        true = Ds[np.arange(j - i), owner[i:j]][:, None]
        tol = 1e-6 * np.maximum(np.abs(true), 1.0)
        below = (Ds < true - tol).sum(axis=1)
        tied = (np.abs(Ds - true) <= tol).sum(axis=1) - 1      # excluding itself
        ranks[i:j] = below + 0.5 * tied
    return ranks, S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default="data/cohort/metabonet834_c3")
    ap.add_argument("--max-days", type=int, default=30,
                    help="cap days per subject so prolific subjects do not dominate")
    ap.add_argument("--min-days", type=int, default=10)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--spaces", nargs="+", default=SPACES)
    ap.add_argument("--within-study", action="store_true",
                    help="rank a day only against subjects from ITS OWN study")
    ap.add_argument("--out", default="results/identifiability")
    a = ap.parse_args()

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    X, sids, man = load_cohort(a.cohort)
    chans = man.get("channels", [man.get("channel", "CGM")])
    X = np.asarray(X)
    print(f"[ident] {a.cohort}: {X.shape} windows, channels={chans}")

    # equalise the days per subject: a subject's day count drives its own retrievability
    rng = np.random.default_rng(a.seed)
    keep = []
    for s in np.unique(sids):
        idx = np.flatnonzero(sids == s)
        if idx.size < a.min_days:
            continue
        keep.append(rng.choice(idx, min(a.max_days, idx.size), replace=False))
    keep = np.sort(np.concatenate(keep))
    X, sids = X[keep], sids[keep]
    order = np.argsort(sids, kind="stable")
    X, sids = X[order], sids[order]
    uniq, first, counts = np.unique(sids, return_index=True, return_counts=True)
    owner = np.repeat(np.arange(uniq.size), counts)
    print(f"[ident] {X.shape[0]:,} windows, {uniq.size} subjects, "
          f"{counts.min()}-{counts.max()} days each (chance top-1 = "
          f"{1 / uniq.size:.3%})")

    subsets = {"all": list(range(len(chans)))}
    if len(chans) > 1:
        for i, c in enumerate(chans):
            subsets[c] = [i]

    # Restricting retrieval to the query's own study is the control for everything that
    # is constant within a study and differs between them: recording units, CGM device,
    # pump model, protocol, era. Across studies, a subject can be "identified" by any of
    # those, and none of them is the person. Subject ids are "<study>/<id>" and the
    # windows are sorted by subject, so each study is a contiguous block.
    if a.within_study:
        study = np.asarray([s.rsplit("/", 1)[0] for s in uniq])
        blocks = [np.flatnonzero(study == s) for s in pd.unique(study)]
        blocks = [b for b in blocks if b.size >= 2]
        print(f"[ident] within-study: {len(blocks)} studies, "
              f"{[int(b.size) for b in blocks]} subjects each")
    else:
        blocks = [np.arange(uniq.size)]

    def evaluate(F):
        """-> per-window normalised rank, top-1 hit, top-5 hit, and chance top-1.

        With one block this is the plain global retrieval. With one block per study a
        query is ranked only among its own study's subjects, so the chance rate differs
        per block and is accumulated as a weighted mean rather than 1/n_subjects.
        """
        nrank = np.full(F.shape[0], np.nan)
        hit1 = np.zeros(F.shape[0], dtype=bool)
        hit5 = np.zeros(F.shape[0], dtype=bool)
        chance = 0.0
        for b in blocks:
            wsel = np.flatnonzero(np.isin(owner, b))       # contiguous: sorted by id
            r, S = retrieve(F[wsel], (first[b] - wsel[0]).astype(np.intp),
                            owner[wsel] - b[0])
            nrank[wsel] = r / max(S - 1, 1)
            hit1[wsel], hit5[wsel] = r == 0, r < 5      # r == 0 is strictly nearest
            chance += wsel.size / F.shape[0] / S
        return nrank, hit1, hit5, chance

    rows, per_subject = [], {}
    for name, cols in subsets.items():
        Xs = X[..., cols]
        for space in a.spaces:
            F = featurise(Xs, space)
            nrank, hit1, hit5, chance = evaluate(F)
            top1, top5 = float(hit1.mean()), float(hit5.mean())
            auc = float(1 - np.nanmean(nrank))
            print(f"  {name:8} {space:9} dim={F.shape[1]:>5}  "
                  f"top1={top1:7.3%}  top5={top5:7.3%}  AUC={auc:.4f}  "
                  f"lift={top1 / max(chance, 1e-12):8.1f}x", flush=True)
            rows.append(dict(channels=name, space=space, dim=int(F.shape[1]),
                             top1=top1, top5=top5, auc=auc,
                             lift=top1 / max(chance, 1e-12), chance_top1=chance,
                             within_study=bool(a.within_study),
                             n_windows=int(X.shape[0]), n_subjects=int(uniq.size)))
            per_subject[f"{name}__{space}"] = pd.Series(
                nrank).groupby(owner).mean().to_numpy()

    df = pd.DataFrame(rows)
    df.to_csv(out / f"{Path(a.cohort).name}.csv", index=False)
    ps = pd.DataFrame(per_subject, index=uniq).rename_axis("id")
    ps.to_csv(out / f"{Path(a.cohort).name}_per_subject.csv")
    (out / f"{Path(a.cohort).name}_meta.json").write_text(json.dumps(
        dict(cohort=a.cohort, channels=chans, max_days=a.max_days,
             min_days=a.min_days, seed=a.seed,
             n_windows=int(X.shape[0]), n_subjects=int(uniq.size)), indent=2))
    print(f"[ident] wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
