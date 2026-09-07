#!/usr/bin/env python
"""Three measurements that bound ANY release-time privacy edit, before one is built.

All three read released sample sets that already exist on disk. No model, no GPU, no
training. They exist because each can kill a design cheaply, and two of them can kill
it before a line of the mechanism is written.

1  THE CEILING (docs/MODEL_DESIGN.md 7, failure mode 2)
   The attack reads `min` over the released set. An edit that pushes the released set
   away from a training window raises that minimum only until the SECOND-nearest
   released sample takes over. So the achievable gain per window is capped at
   `d2 - d1`, and the whole design is capped at the distribution of that quantity.
   Compare it against the membership gap `g` the edit has to remove (--gap): if the
   ceiling is not comfortably above `g`, no amount of mechanism helps.

2  SPARSITY (does a cell-selective edit have leverage?)
   Once the nearest released sample is fixed, the squared distance is a sum over cells.
   If the top 20% of cells carry 60% of it, masking or editing those cells moves most
   of the distance for a fifth of the distortion. If 20% of cells carry 20%, a
   "targeted" edit is just a random edit and the mask-head localiser has nothing to
   find. This is the leverage assumption docs/MODEL_DESIGN.md 2.6d rests on and it has
   never been measured.

3  RAW vs SORTED AGREEMENT (property, or statistic?)
   MODEL_DESIGN 2.3 established that the leak lives in the per-window value
   distribution -- the sorted coordinate -- not in timing. A localiser that selects
   cells by their contribution to the RAW L2 distance is selecting on the frozen
   attack's own formula, which is the adaptive-attack objection. If the cells selected
   in raw space and in sorted space largely coincide, the localiser is tracking a
   property of the data and that objection weakens. If they do not, it is tracking the
   statistic and the defence would be tuned to one attack.

   Sorted-space contributions are attributed back to time indices through the window's
   own rank permutation (level u belongs to time argsort(x)[u]), so the two maps are
   over the same index set and can be compared directly.

    python scripts/nn_ceiling.py --cohort data/cohort/matrix_d1_c1 \
        --design results/matrix/design/rep1 \
        --run results/runs/sweep_d1_c1_ms3/base --gap 0.00240

Distances use the attack's convention: Euclidean divided by sqrt(T*C), so every number
here is on the same scale as `d_in`/`d_out` in results/matrix/sweep/attack/*/summary.json.
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


def two_nearest(R, S, chunk=256):
    """-> (d1, d2, idx1), distances already divided by sqrt(F)."""
    F = R.shape[1]
    d1 = np.empty(len(R), np.float32)
    d2 = np.empty(len(R), np.float32)
    i1 = np.empty(len(R), np.int64)
    Sn = (S ** 2).sum(1)
    for a in range(0, len(R), chunk):
        b = min(a + chunk, len(R))
        # (r - s)^2 = |r|^2 - 2 r.s + |s|^2; the |r|^2 term is constant per row and
        # does not change the ordering, but it is kept so the values are distances.
        d = (R[a:b] ** 2).sum(1)[:, None] - 2.0 * (R[a:b] @ S.T) + Sn[None, :]
        np.maximum(d, 0.0, out=d)
        part = np.argpartition(d, 1, axis=1)[:, :2]
        v = np.take_along_axis(d, part, axis=1)
        order = np.argsort(v, axis=1)
        part = np.take_along_axis(part, order, axis=1)
        v = np.take_along_axis(v, order, axis=1)
        d1[a:b] = np.sqrt(v[:, 0] / F)
        d2[a:b] = np.sqrt(v[:, 1] / F)
        i1[a:b] = part[:, 0]
    return d1, d2, i1


def cumulative_share(contrib, fracs):
    """contrib (n, F) per-cell squared contributions -> mean cumulative share at each
    fraction of cells, taking cells in decreasing order of contribution per row."""
    srt = -np.sort(-contrib, axis=1)
    tot = srt.sum(1, keepdims=True)
    tot[tot == 0] = 1.0
    cum = np.cumsum(srt, axis=1) / tot
    F = contrib.shape[1]
    return {f: float(cum[:, max(int(round(f * F)) - 1, 0)].mean()) for f in fracs}


def topk_jaccard(A, B, frac):
    """Mean Jaccard of the top-`frac` cell sets of two per-row contribution maps."""
    k = max(int(round(frac * A.shape[1])), 1)
    ia = np.argpartition(-A, k - 1, axis=1)[:, :k]
    ib = np.argpartition(-B, k - 1, axis=1)[:, :k]
    out = np.empty(len(A))
    for i in range(len(A)):
        inter = len(np.intersect1d(ia[i], ib[i], assume_unique=False))
        out[i] = inter / (2 * k - inter)
    return float(out.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--design", required=True)
    ap.add_argument("--run", required=True, help="a run dir holding samples.npy")
    ap.add_argument("--job", default=None, help="default: read the run's meta.json")
    ap.add_argument("--n-real", type=int, default=1500)
    ap.add_argument("--gap", type=float, default=None,
                    help="the membership gap g this cell has to remove, for scale")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    run = Path(a.run)
    job = a.job or json.loads((run / "meta.json").read_text())["job"]
    subjects = json.loads((Path(a.design) / "jobs" / f"{job}.json").read_text())["subjects"]

    X, sids, man = load_cohort(a.cohort)
    real, _ = training_set(X, sids, subjects)
    S_raw = np.asarray(np.load(run / "samples.npy", mmap_mode="r"), dtype=np.float32)
    T, C = real.shape[1], real.shape[2]

    rng = np.random.default_rng(a.seed)
    pick = rng.choice(len(real), size=min(a.n_real, len(real)), replace=False)
    R3 = np.asarray(real)[np.sort(pick)].astype(np.float32)      # (n, T, C)

    print(f"cell={a.cohort}  job={job}  T={T} C={C}")
    print(f"real windows {len(real):,} (using {len(R3):,})   released {len(S_raw):,}")

    # ---------------------------------------------------------------- 1 the ceiling
    R, S = _flat(R3), _flat(S_raw)
    d1, d2, i1 = two_nearest(R, S)
    ceil = d2 - d1
    qs = [5, 25, 50, 75, 95]
    print("\n=== 1  CEILING: how far can any edit push before the 2nd neighbour takes over")
    print(f"  d1        median {np.median(d1):.5f}   [{np.percentile(d1,5):.5f}, {np.percentile(d1,95):.5f}]")
    print(f"  d2 - d1   median {np.median(ceil):.5f}   " +
          "  ".join(f"p{q}={np.percentile(ceil,q):.5f}" for q in qs))
    if a.gap:
        frac = float((ceil < a.gap).mean())
        print(f"  the gap to remove g = {a.gap:.5f}")
        print(f"  ceiling / g = {np.median(ceil)/a.gap:.2f}x at the median")
        print(f"  {frac*100:.1f}% of windows have a ceiling BELOW g -- on those the edit "
              f"cannot reach the target however it is built")

    # ------------------------------------------------------------------- 2 sparsity
    nn = S[i1]                                     # (n, F) the nearest released sample
    contrib_raw = (R - nn) ** 2                    # (n, F)
    fr = [0.05, 0.10, 0.20, 0.30, 0.50]
    sh_raw = cumulative_share(contrib_raw, fr)
    print("\n=== 2  SPARSITY: share of the squared distance carried by the top cells")
    print("  RAW space   " + "   ".join(f"top{int(f*100)}%={sh_raw[f]*100:.1f}%" for f in fr))

    # sorted space: NN recomputed there, contributions attributed back to time index
    Rs = np.sort(R3, axis=1)                       # (n, T, C) per-channel quantiles
    Ss = np.sort(S_raw, axis=1)
    d1s, d2s, i1s = two_nearest(_flat(Rs), _flat(Ss))
    contrib_srt_lvl = (_flat(Rs) - _flat(Ss)[i1s]) ** 2          # (n, F) by LEVEL
    order = np.argsort(R3, axis=1)                               # level -> time index
    contrib_srt = np.empty_like(contrib_srt_lvl)
    c3 = contrib_srt_lvl.reshape(len(R3), T, C)
    o3 = np.empty_like(c3)
    np.put_along_axis(o3, order, c3, axis=1)       # level u belongs to time order[u]
    contrib_srt = o3.reshape(len(R3), -1)
    sh_srt = cumulative_share(contrib_srt, fr)
    print("  SORTED      " + "   ".join(f"top{int(f*100)}%={sh_srt[f]*100:.1f}%" for f in fr))
    print(f"  sorted d1 median {np.median(d1s):.5f}  (raw {np.median(d1):.5f}; "
          f"ratio {np.median(d1s)/np.median(d1):.3f} -- MODEL_DESIGN 2.3 measured 0.185-0.214)")

    # ---------------------------------------------------- 3 do they select the same cells
    print("\n=== 3  AGREEMENT: do raw and sorted space select the same cells?")
    print("  (1.0 = identical sets, ~frac = chance)")
    for f in (0.10, 0.20, 0.30):
        j = topk_jaccard(contrib_raw, contrib_srt, f)
        chance = f / (2 - f)
        print(f"  top{int(f*100)}%  Jaccard {j:.3f}   chance {chance:.3f}   "
              f"lift {j/chance:.2f}x")

    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump({
            "cohort": a.cohort, "job": job, "run": str(run), "T": T, "C": C,
            "n_real_used": len(R3), "n_released": len(S_raw), "gap": a.gap,
            "d1_median": float(np.median(d1)), "d2m1_median": float(np.median(ceil)),
            "d2m1_pct": {str(q): float(np.percentile(ceil, q)) for q in qs},
            "frac_ceiling_below_gap": (float((ceil < a.gap).mean()) if a.gap else None),
            "share_raw": {str(k): v for k, v in sh_raw.items()},
            "share_sorted": {str(k): v for k, v in sh_srt.items()},
            "d1_sorted_median": float(np.median(d1s)),
            "jaccard": {str(f): topk_jaccard(contrib_raw, contrib_srt, f)
                        for f in (0.10, 0.20, 0.30)},
        }, open(a.out, "w"), indent=2)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
