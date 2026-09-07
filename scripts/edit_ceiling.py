#!/usr/bin/env python
"""Can ANY release-time directed edit close the membership gap? Measured, not bounded.

WHY THIS REPLACES scripts/nn_ceiling.py's FIRST SECTION
-------------------------------------------------------
`nn_ceiling.py` reports `d2 - d1`, the spacing between a real window's first and second
nearest released samples. docs/MODEL_DESIGN.md 7 (failure mode 2) names that as the cap
on a directed edit's gain. It is a **per-pair** cap and it is pessimistic in one specific
way: it assumes only the nearest released sample moves. A real edit moves the WHOLE
released set, so the second-nearest is moving too, and the true cap is whatever the local
density does under a global displacement. It is also aggregated wrongly -- per window,
where the frozen attack's `g` is per subject (`subject_reduce="mean"`).

This script measures the thing itself. It applies the most favourable directed edit that
exists and reads the attack's own output.

THE EDIT
--------
For each released sample `s`, find its nearest window `r` in the model's OWN training set
and push it directly away:

    s' = s + delta * sqrt(F) * (s - r) / ||s - r||

This is the upper bound of what any per-sample directed mechanism can achieve at that
displacement: it moves every sample exactly along the direction that raises its distance
to the training data fastest (MODEL_DESIGN 2.5). No real mechanism does better; RQE's
barycentric term and a mask-head re-imputation both do strictly worse, because they also
have to stay on the data manifold. **So a delta at which this fails to close the gap is a
delta at which nothing closes it.**

`delta` is in per-cell RMS units -- the displacement's L2 norm is `delta*sqrt(F)` over `F`
cells -- so it is directly comparable to `g`, to `d_in`, and to `d2-d1`.

BOTH ARMS ARE EDITED. In deployment the holder edits whatever they release, and in this
study's structure that means each model's own release. Editing only the member side would
measure a defence nobody could deploy. Each set is pushed away from its own training set.

WHAT IT DECIDES
---------------
  gap(delta) -> 0 at a delta well below the fidelity budget   the mechanism has room
  gap(delta) saturates above 0                                the neighbours take over;
                                                              no release-time edit works
                                                              on this cell, ours included

Arm AUC is reported at every delta, so this is the risk axis of PAPER_PLAN Figure 2
traced without building any mechanism at all.

    python scripts/edit_ceiling.py --cell d1_c1 --runs results/runs/sweep_d1_c1_ms3 \
        --design results/matrix/design/rep1 --out results/matrix/edit_ceiling/d1_c1.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cgmoutlier.data.cohort import load as load_cohort               # noqa: E402
from cgmoutlier.loo import training_set                              # noqa: E402
from cgmoutlier.attack.statistic import (window_distances,          # noqa: E402
                                         subject_distance, _match)

GRID = [0.0, 0.0005, 0.001, 0.002, 0.004, 0.008, 0.016, 0.032]


def _flat(A):
    A = np.asarray(A, dtype=np.float32)
    return A.reshape(A.shape[0], -1) if A.ndim > 2 else A


def push_direction(S, TR, chunk=512, mode="dual"):
    """Unit vector along which each released sample is pushed, and a diagnostic.

    MODEL_DESIGN 2.5 specifies the direction as `(S_nn - R)`: S_nn is the released
    sample NEAREST TO A REAL WINDOW R. That is the pairing the attack reads -- it takes
    `min over released` for each real window -- and it is not the same relation as "the
    training window nearest to this released sample". A released sample can be closest
    to training window r1 while being the one r2's minimum selects; pushing it away from
    r1 then does nothing to r2's distance, which is the number the attack reports.

      dual   (the design's, and the default) for every training window r, find the
             released sample s(r) that r's `min` selects, and push s(r) away from r.
             A released sample selected by several windows gets the normalised sum of
             those directions. A released sample no window selects is NOT MOVED -- it
             carries no risk, and leaving it alone is free fidelity.
      primal for every released sample, push it away from its own nearest training
             window. Kept only because it was measured first and the two disagree; it
             is the wrong pairing and its numbers are not the design's.

    Returns (U, dnn, n_moved) with U (n, F) unit rows, zero where nothing pushes."""
    S, TR = _flat(S), _flat(TR)
    n, F = S.shape
    U = np.zeros_like(S)

    if mode == "primal":
        d = np.empty(n, np.float32)
        t2 = (TR ** 2).sum(1)
        for a in range(0, n, chunk):
            b = min(a + chunk, n)
            q = S[a:b]
            d2 = (q ** 2).sum(1)[:, None] - 2.0 * (q @ TR.T) + t2[None, :]
            j = np.argmin(d2, axis=1)
            v = q - TR[j]
            nrm = np.linalg.norm(v, axis=1)
            d[a:b] = nrm / np.sqrt(F)
            ok = nrm > 1e-8
            U[a:b][ok] = v[ok] / nrm[ok, None]
        return U, d, int((np.linalg.norm(U, axis=1) > 0).sum())

    # dual: iterate over TRAINING windows, accumulate onto the released sample each selects
    dnn = np.empty(len(TR), np.float32)
    s2 = (S ** 2).sum(1)
    for a in range(0, len(TR), chunk):
        b = min(a + chunk, len(TR))
        r = TR[a:b]
        d2 = (r ** 2).sum(1)[:, None] - 2.0 * (r @ S.T) + s2[None, :]
        j = np.argmin(d2, axis=1)
        dnn[a:b] = np.sqrt(np.clip(np.take_along_axis(d2, j[:, None], 1)[:, 0], 0, None)) / np.sqrt(F)
        v = S[j] - r
        nrm = np.linalg.norm(v, axis=1)
        ok = nrm > 1e-8
        np.add.at(U, j[ok], v[ok] / nrm[ok, None])
    nrm = np.linalg.norm(U, axis=1)
    moved = nrm > 1e-8
    U[moved] /= nrm[moved, None]
    return U, dnn, int(moved.sum())


def mannwhitney_auc(a, b):
    """P(a > b) with ties at 0.5 -- the arm AUC the pipeline reports."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    gt = (a[:, None] > b[None, :]).sum()
    eq = (a[:, None] == b[None, :]).sum()
    return float((gt + 0.5 * eq) / (len(a) * len(b)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True)
    ap.add_argument("--cohort", default=None, help="default data/cohort/matrix_<cell>")
    ap.add_argument("--runs", required=True, help="dir holding base/ and include_*/")
    ap.add_argument("--design", required=True)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--grid", type=float, nargs="*", default=None)
    ap.add_argument("--direction", default="dual", choices=["dual", "primal"],
                    help="dual is MODEL_DESIGN 2.5's (S_nn - R); primal is the "
                         "reversed pairing, kept only for the comparison")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    cohort = a.cohort or f"data/cohort/matrix_{a.cell}"
    grid = a.grid if a.grid else GRID
    runs, design = Path(a.runs), Path(a.design)

    X, sids, man = load_cohort(cohort)
    F = int(np.prod(np.asarray(X).shape[1:]))
    sqF = np.sqrt(np.float32(F))

    jobs = sorted(p.stem for p in (design / "jobs").glob("include_*.json"))
    print(f"cell={a.cell}  cohort={cohort}  F={F}  targets={len(jobs)}  "
          f"direction={a.direction}  grid={grid}")

    # ---- base: its release, its training set, and its push direction (done once)
    bj = json.loads((design / "jobs" / "base.json").read_text())
    S_base0 = np.asarray(np.load(runs / "base" / "samples.npy", mmap_mode="r"), np.float32)
    TR_base, _ = training_set(X, sids, bj["subjects"])
    t0 = time.time()
    U_base, dnn_base, zb = push_direction(S_base0, TR_base, mode=a.direction)
    print(f"base: {len(S_base0):,} released, nearest-training distance median "
          f"{np.median(dnn_base):.5f}, {zb} coincident ({time.time()-t0:.0f}s)")

    rows, per_target = [], {}
    for name in jobs:
        j = json.loads((design / "jobs" / f"{name}.json").read_text())
        tgt, grp = j["target"], j["group"]
        S_in0 = np.asarray(np.load(runs / name / "samples.npy", mmap_mode="r"), np.float32)
        R_t, _ = training_set(X, sids, [tgt])
        TR_in, _ = training_set(X, sids, j["subjects"])
        U_in, dnn_in, zi = push_direction(S_in0, TR_in, mode=a.direction)

        gaps = {}
        for delta in grid:
            S_in = S_in0 + (delta * sqF) * U_in.reshape(S_in0.shape)
            S_out = S_base0 + (delta * sqF) * U_base.reshape(S_base0.shape)
            Si, So, _k = _match(S_in, S_out, tgt, a.seed, True)
            wi = window_distances(R_t, Si, set_reduce="min")
            wo = window_distances(R_t, So, set_reduce="min")
            # RISK, not contrast: the per-subject AUC exactly as the pipeline defines
            # it in results/matrix/sweep/subject_auc/*/per_subject.csv -- an UNPAIRED
            # rank AUC of the subject's d_out window distances against its d_in ones.
            # Not the paired fraction: that CSV carries both, and for Loop/1142 they
            # read 0.747 and 0.958. The paired one is the stronger statistic and it is
            # not what PAPER_PLAN's "patients with AUC > 0.55" counts, so using it here
            # would silently report a different quantity than every table in the paper.
            auc = mannwhitney_auc(wo, wi)
            d_in, d_out = float(wi.mean()), float(wo.mean())
            gaps[delta] = {"d_in": d_in, "d_out": d_out, "gap": d_out - d_in,
                           "subject_auc": auc}
        per_target[tgt] = {"group": grp, "n_windows": int(len(R_t)),
                           "n_moved": zi, "by_delta": gaps}
        g0 = gaps[grid[0]]["gap"]
        print(f"  {tgt:14s} {grp:8s} n={len(R_t):3d}  gap(0)={g0:+.5f}  " +
              "  ".join(f"{d:g}:{gaps[d]['gap']:+.5f}" for d in grid[1:]))

    print("\n=== what the edit does, on BOTH axes ===")
    print("  RISK    = per-subject AUC: can this individual be identified at all")
    print("  CONTRAST= arm AUC: are outliers identified MORE than controls")
    print("  PAPER_PLAN 3b measured these moving in opposite directions. A defence is")
    print("  judged on RISK; CONTRAST is the study's original hypothesis, not the goal.")
    print()
    print("  delta     RISK>0.55  medAUC  maxAUC | CONTRAST  medgapOUT  medgapCTRL  gap<=0")
    summary = []
    for delta in grid:
        og = [v["by_delta"][delta]["gap"] for v in per_target.values() if v["group"] == "outlier"]
        cg = [v["by_delta"][delta]["gap"] for v in per_target.values() if v["group"] == "control"]
        auc = mannwhitney_auc(og, cg)
        nneg = int(sum(1 for g in og + cg if g <= 0))
        sa = np.array([v["by_delta"][delta]["subject_auc"] for v in per_target.values()])
        n55 = int((sa > 0.55).sum())
        summary.append({"delta": delta, "arm_auc": auc,
                        "risk_n_above_055": n55, "risk_median_auc": float(np.median(sa)),
                        "risk_max_auc": float(sa.max()),
                        "median_gap_outlier": float(np.median(og)) if og else None,
                        "median_gap_control": float(np.median(cg)) if cg else None,
                        "n_gap_nonpositive": nneg, "n_total": len(og) + len(cg)})
        print(f"  {delta:<9g} {n55:>4d}/{len(sa)}    {np.median(sa):.3f}   {sa.max():.3f} |"
              f"  {auc:.3f}    {np.median(og):+.5f}    {np.median(cg):+.5f}   "
              f"{nneg}/{len(og)+len(cg)}")
    print("\nRISK is the number to read: PAPER_PLAN reports 'patients with AUC > 0.55',")
    print("and 3b's headline (11 of 26 at 30k, 22 of 26 at 100k) is that column. A gap")
    print("driven NEGATIVE is overshoot -- MODEL_DESIGN 2.6c: a one-sided AUC collapsing")
    print("toward 0 is a PERFECT distinguisher for an attacker who knows a defence is")
    print("deployed, not a success. Which is why gap<=0 is printed beside both axes, and")
    print("why the per-subject AUC above is two-sided in effect: it counts windows, so a")
    print("subject whose gap inverted lands BELOW 0.5 and is not counted as safe by luck.")

    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump({"cell": a.cell, "direction": a.direction, "cohort": cohort, "runs": str(runs), "F": F,
                   "grid": grid, "seed": a.seed, "summary": summary,
                   "per_target": per_target}, open(a.out, "w"), indent=2)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
