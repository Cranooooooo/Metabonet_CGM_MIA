#!/usr/bin/env python
"""Per-cell risk maps on a released set, under every definition that is defensible.

The mask-head localiser needs something to be compared against, and "which cells carry
the leak" turns out to have more than one honest answer. This script computes the
model-free ones and measures how much they agree, so that when MAVEN's mask head produces
a map we can say which -- if any -- it reproduces.

THE MAPS
--------
For each released sample `s`, let `r` be its nearest window in the generator's own
training set.

  RAW        raw[c] = (s[c] - r[c])**2
             The per-cell contribution to the frozen attack's squared distance.
             SELECTING HIGH RAW is the efficient lever against the L2 statistic: moving
             along the residual restricted to a cell set preserves a fraction of the rate
             equal to that set's share of the residual NORM, and results/matrix/ceiling
             measured the top 20% of cells at 66-72% of the squared norm, i.e. ~82% of
             the norm.
  QUANTILE   the same residual computed after sorting each channel, i.e. between the two
             windows' empirical quantile functions, attributed back to time indices
             through `s`'s own rank permutation (level u belongs to time argsort(s)[u]).
             MODEL_DESIGN 2.3: this is the coordinate in which the arms separate (arm AUC
             0.828-0.905 against 0.562-0.627 raw), because the leaking quantity is a W2
             distance between per-window value distributions.

THE SIGN IS AN OPEN CHOICE AND THE TWO READINGS SELECT OPPOSITE CELLS
---------------------------------------------------------------------
High residual = where `s` already differs from `r` = the efficient direction to push
further, if the objective is to defeat the L2 statistic.
Low residual = where `s` agrees with `r` = the memorised content, if the objective is to
remove what was copied.
These are not the same cells and no measurement in this repo settles which is right, so
both are reported. Choosing one after seeing which lowers the arm AUC would be exactly
the failure PAPER_PLAN Tip 4 rules out; the choice has to be argued or pre-registered.

WHY AGREEMENT IS THE OUTPUT
---------------------------
results/matrix/ceiling measured RAW against QUANTILE on real training windows and found
Jaccard 0.15-0.20 at top-20%, a lift of only 1.4-1.8x over chance -- while copy_paste,
which memorises verbatim, read 0.601 (5.4x). So on a real generator the two coordinates
genuinely disagree about which cells matter, and a localiser must be attributed to one of
them rather than to "the leak" in general.

    python scripts/localise_cells.py --run results/runs/pilot_maven_d1_c1/base \
        --design results/matrix/design/rep1 --cohort data/cohort/matrix_d1_c1 \
        --out results/matrix/cellmaps/pilot_maven_d1_c1.npz
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cgmoutlier.data.cohort import load as load_cohort          # noqa: E402
from cgmoutlier.loo import training_set                         # noqa: E402


def _flat(A):
    A = np.asarray(A, dtype=np.float32)
    return A.reshape(A.shape[0], -1) if A.ndim > 2 else A


def nearest(Q, TR, chunk=512):
    """Index of each row of Q's nearest row in TR, and that distance / sqrt(F)."""
    Q, TR = _flat(Q), _flat(TR)
    F = Q.shape[1]
    idx = np.empty(len(Q), np.int64)
    dst = np.empty(len(Q), np.float32)
    t2 = (TR ** 2).sum(1)
    for a in range(0, len(Q), chunk):
        b = min(a + chunk, len(Q))
        d2 = (Q[a:b] ** 2).sum(1)[:, None] - 2.0 * (Q[a:b] @ TR.T) + t2[None, :]
        j = np.argmin(d2, axis=1)
        idx[a:b] = j
        dst[a:b] = np.sqrt(np.clip(np.take_along_axis(d2, j[:, None], 1)[:, 0], 0, None)) / np.sqrt(F)
    return idx, dst


def topk_jaccard(A, B, frac, invert_a=False, invert_b=False):
    k = max(int(round(frac * A.shape[1])), 1)
    sa = -A if not invert_a else A
    sb = -B if not invert_b else B
    ia = np.argpartition(sa, k - 1, axis=1)[:, :k]
    ib = np.argpartition(sb, k - 1, axis=1)[:, :k]
    out = np.empty(len(A))
    for i in range(len(A)):
        inter = len(np.intersect1d(ia[i], ib[i]))
        out[i] = inter / (2 * k - inter)
    return float(out.mean())


def cumulative_share(contrib, fracs):
    srt = -np.sort(-contrib, axis=1)
    tot = srt.sum(1, keepdims=True); tot[tot == 0] = 1.0
    cum = np.cumsum(srt, axis=1) / tot
    F = contrib.shape[1]
    return {f: float(cum[:, max(int(round(f * F)) - 1, 0)].mean()) for f in fracs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--design", required=True)
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--job", default=None)
    ap.add_argument("--n", type=int, default=1024, help="released samples to map")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    run = Path(a.run)
    job = a.job or json.loads((run / "meta.json").read_text())["job"]
    subjects = json.loads((Path(a.design) / "jobs" / f"{job}.json").read_text())["subjects"]

    X, sids, _ = load_cohort(a.cohort)
    TR, _ = training_set(X, sids, subjects)
    S_all = np.asarray(np.load(run / "samples.npy", mmap_mode="r"), np.float32)
    rng = np.random.default_rng(a.seed)
    pick = np.sort(rng.choice(len(S_all), size=min(a.n, len(S_all)), replace=False))
    S = S_all[pick]                                   # (n, T, C)
    n, T, C = S.shape
    print(f"run={run}  job={job}  released {len(S_all):,} (mapping {n})  T={T} C={C}")

    # ---------------------------------------------------------------- RAW
    j_raw, d_raw = nearest(S, TR)
    raw = (_flat(S) - _flat(TR)[j_raw]) ** 2                     # (n, F)

    # ----------------------------------------------------------- QUANTILE
    Ss, TRs = np.sort(S, axis=1), np.sort(np.asarray(TR), axis=1)
    j_q, d_q = nearest(Ss, TRs)
    lvl = (_flat(Ss) - _flat(TRs)[j_q]) ** 2                     # (n, F) by level
    order = np.argsort(S, axis=1)                                # level -> time index
    q3 = np.empty((n, T, C), np.float32)
    np.put_along_axis(q3, order, lvl.reshape(n, T, C), axis=1)
    quant = q3.reshape(n, -1)

    print(f"  nearest-training distance   raw {np.median(d_raw):.5f}   "
          f"quantile {np.median(d_q):.5f}   ratio {np.median(d_q)/np.median(d_raw):.3f}")
    print(f"  same training window chosen in both spaces: "
          f"{100.0*(j_raw == j_q).mean():.1f}% of released samples")

    fr = [0.05, 0.10, 0.20, 0.30]
    print("\n=== leverage: share of the squared residual in the top cells")
    for nm, M in (("RAW", raw), ("QUANTILE", quant)):
        sh = cumulative_share(M, fr)
        print(f"  {nm:9s} " + "  ".join(f"top{int(f*100)}%={sh[f]*100:.1f}%" for f in fr))

    print("\n=== agreement between the two coordinates (top-k Jaccard)")
    print("  frac   HIGH-residual   LOW-residual   chance")
    for f in (0.10, 0.20, 0.30):
        hi = topk_jaccard(raw, quant, f)
        lo = topk_jaccard(raw, quant, f, invert_a=True, invert_b=True)
        print(f"  {int(f*100):>3d}%   {hi:.3f} ({hi/(f/(2-f)):.2f}x)   "
              f"{lo:.3f} ({lo/(f/(2-f)):.2f}x)   {f/(2-f):.3f}")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, idx=pick, raw=raw.astype(np.float32),
                        quantile=quant.astype(np.float32),
                        nn_raw=j_raw, nn_quantile=j_q,
                        d_raw=d_raw, d_quantile=d_q, T=T, C=C, job=job)
    print(f"\nwrote {a.out}  ({raw.nbytes*2/1e6:.0f} MB uncompressed)")
    print("The mask-head map goes into the same comparison: scripts/maven_surprisal.py")


if __name__ == "__main__":
    main()
