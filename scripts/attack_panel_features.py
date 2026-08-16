#!/usr/bin/env python
"""Stage A of the supervised attacker panel: per-window features, computed once.

    python scripts/attack_panel_features.py --generator dimts_h128 --rep 1

Writes results/attack_panel/<runs>/rep<N>.npz holding, for every target subject, a
(n_windows, 10) feature block against each of its released sets: one the subject is a
member of and several it is not.

WHY THIS IS A SEPARATE SCRIPT FROM THE TABLE
--------------------------------------------
The features are the expensive part -- 240 distance matrices of (n_windows x ~176k)
per replicate -- and every cell of the panel reads the same ones. Training the
classifiers is seconds. Splitting them means the feature pass runs once for the whole
table, and adding a classifier or a feature subset later costs nothing.

WHY THE LOOP IS OVER RELEASED SETS, NOT OVER SUBJECTS
-----------------------------------------------------
Each released set is 200 MB on disk and is needed by several subjects (`base` by all
40 of them). Looping over subjects would read it 40 times. Looping over sets reads
each once, at the cost of having to know every subject's common K in advance -- which
is why `panel.common_k` works off the file headers and `panel.cut_to_k` seeds per
(subject, set) rather than per subject.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from cgmoutlier._env import check as _envcheck                      # noqa: E402
_envcheck()
from cgmoutlier.data.cohort import load as load_cohort              # noqa: E402
from cgmoutlier.loo.train import training_set                       # noqa: E402
from cgmoutlier.attack import panel as P                            # noqa: E402


def npy_len(path: Path) -> int:
    """Rows of a .npy without reading its 200 MB of data.

    mmap rather than np.lib.format's header reader: the latter is private API whose
    signature has moved between numpy versions, and this has to keep working in both
    environments.
    """
    return int(np.load(path, mmap_mode="r").shape[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="results/runs/dimts_h128_rep{rep}")
    ap.add_argument("--design", default="results/design_sym/rep{rep}")
    ap.add_argument("--cohort", default="data/cohort/metabonet875")
    ap.add_argument("--rep", type=int, required=True)
    ap.add_argument("--out", default="results/attack_panel/dimts_h128")
    ap.add_argument("--n-neg", type=int, default=4,
                    help="include_u releases supplying negatives, on top of base")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--n-ref", type=int, default=1024,
                    help="background windows forming each release's reference; they "
                         "are members of every release, so the reference itself "
                         "carries no membership signal")
    ap.add_argument("--limit", type=int, default=None,
                    help="first N targets only; for checking the plumbing before "
                         "committing three hours to it")
    a = ap.parse_args()

    runs = Path(a.runs.format(rep=a.rep))
    design = Path(a.design.format(rep=a.rep))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    jobs = {p.stem: json.loads(p.read_text()) for p in (design / "jobs").glob("*.json")}
    targets = sorted(n[len("include_"):] for n in jobs if n.startswith("include_"))
    if a.limit:
        targets = targets[:a.limit]
    groups = {j["target"]: j.get("group") for j in jobs.values() if j.get("target")}
    print(f"[panel] rep{a.rep}: {len(targets)} targets, runs={runs}", flush=True)

    lengths = {n: npy_len(runs / n / "samples.npy") for n in jobs}

    # Which sets each subject needs, and the K they are all cut to.
    need, kof = {}, {}
    for t in targets:
        sets = [f"include_{t}"] + P.assign_negative_sets(t, targets, n_neg=a.n_neg,
                                                         seed=a.seed)
        need[t] = sets
        kof[t] = P.common_k({s: lengths[s] for s in sets})
    inverse = {}
    for t, sets in need.items():
        for s in sets:
            inverse.setdefault(s, []).append(t)
    print(f"[panel] {len(inverse)} distinct released sets to read; "
          f"K per subject {min(kof.values()):,}-{max(kof.values()):,}", flush=True)

    X, sids, _ = load_cohort(a.cohort)
    cal = P.calibrate(X, seed=a.seed)
    print(f"[panel] thresholds from REAL windows only: "
          f"r_euc={cal['r_euc']:.4f} tau={cal['tau']:.4f} cos_c0={cal['cos_c0']:.4f}",
          flush=True)

    real = {t: training_set(X, sids, [t])[0] for t in targets}

    # The reference every release is normalised against: a FIXED sample of windows
    # from the 835 background subjects. They are members of every release in this
    # replicate, so the reference carries no membership signal of its own, and it is
    # the same windows for every release so the normalisations are comparable.
    bg = jobs["base"]["subjects"]
    Xbg, _ = training_set(X, sids, bg)
    rng = np.random.default_rng(a.seed)
    ref = Xbg[np.sort(rng.choice(len(Xbg), size=int(min(a.n_ref, len(Xbg))),
                                 replace=False))]
    print(f"[panel] reference: {len(ref):,} background windows from "
          f"{len(bg)} subjects", flush=True)

    feats = {}                                    # (target, set) -> (n, D) float32
    refstat = {}                                  # set -> (2, D) mean and sd
    t0 = time.time()
    for i, (s, ts) in enumerate(sorted(inverse.items()), 1):
        S = np.load(runs / s / "samples.npy")
        # The reference is cut to each subject's K, so normalise per (subject, K)
        # rather than once per release: a feature computed at one K is not on the
        # same scale as the same feature at another.
        for k in sorted({kof[t] for t in ts}):
            Sk = P.cut_to_k(S, k, target="__ref__", set_name=s, seed=a.seed)
            Fr = P.per_window_features(ref, Sk, cal=cal)
            refstat[f"{s}|{k}"] = np.stack([Fr.mean(0), Fr.std(0)]).astype(np.float32)
            del Sk
        for t in ts:
            Sc = P.cut_to_k(S, kof[t], target=t, set_name=s, seed=a.seed)
            feats[f"{t}|{s}"] = P.per_window_features(real[t], Sc, cal=cal)
        del S
        print(f"[panel] {i}/{len(inverse)} {s}: {len(ts)} subject(s), "
              f"{time.time()-t0:.0f}s elapsed", flush=True)

    meta = dict(rep=a.rep, runs=str(runs), design=str(design), seed=a.seed,
                n_neg=a.n_neg, feature_names=P.FEATURE_NAMES, cal=cal,
                targets=targets, groups={t: groups.get(t) for t in targets},
                positive={t: f"include_{t}" for t in targets},
                negatives={t: need[t][1:] for t in targets},
                k={t: kof[t] for t in targets}, n_ref=int(len(ref)),
                n_windows={t: int(len(real[t])) for t in targets})
    np.savez_compressed(out / f"rep{a.rep}.npz", meta=json.dumps(meta), **feats,
                        **{f"REF|{k}": v for k, v in refstat.items()})
    print(f"[panel] wrote {out/f'rep{a.rep}.npz'} "
          f"({len(feats)} blocks, {time.time()-t0:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
