#!/usr/bin/env python
"""Synthesise a cohort with the same shape as the real one, so the pipeline can be
run end to end on a machine that has no access to the data.

    python scripts/make_fake_data.py --out data/cohort/fake60 --n-subjects 60

WHAT THIS IS FOR: checking that the code runs, that the shapes line up, and that a
change did not break an import. Fifteen methods over sixty subjects takes a couple of
minutes on a laptop CPU.

WHAT IT IS NOT FOR: any statement about outliers. The subjects here are drawn from
three hand-written archetypes plus noise, so "which subjects are atypical" has a known
answer by construction and tells you nothing about CGM. Numbers produced from fake
data are labelled as such in the manifest (`"fake": true`) and the loader passes that
flag through, so nothing downstream can mistake one for the other.

The archetypes are loosely glucose-shaped -- a dawn rise, three meal excursions, an
overnight plateau -- purely so that plots look like CGM and so the clinical metrics do
not divide by zero. They are not a model of glycaemia.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cgmoutlier._env import check as _envcheck   # noqa: E402
_envcheck()

T = 288          # 24 h at 5 min
MEALS = [(84, 40), (156, 55), (228, 45)]   # (index, amplitude in mg/dL)


def _day(rs, base, cv, meal_scale, spike=False):
    t = np.arange(T)
    x = np.full(T, base, float)
    x += 12.0 * np.sin(2 * np.pi * (t - 60) / T)                    # circadian
    for pos, amp in MEALS:
        jitter = int(rs.normal(0, 8))
        d = t - (pos + jitter)
        x += amp * meal_scale * np.exp(-(d ** 2) / (2 * 22.0 ** 2))  # excursion
        x -= 0.35 * amp * meal_scale * np.exp(-((d - 45) ** 2) / (2 * 30.0 ** 2))
    if spike:                                                        # a rare bad day
        p = int(rs.integers(0, T))
        x += 130 * np.exp(-((t - p) ** 2) / (2 * 15.0 ** 2))
    x += rs.normal(0, base * cv * 0.35, T)
    return np.clip(x, 40, 400)


ARCHETYPES = {                 # base mg/dL, CV, meal scale, P(spike day)
    "tight":   (108.0, 0.16, 0.7, 0.01),
    "typical": (145.0, 0.28, 1.0, 0.05),
    "brittle": (185.0, 0.45, 1.4, 0.30),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/cohort/fake60")
    ap.add_argument("--n-subjects", type=int, default=60)
    ap.add_argument("--min-days", type=int, default=30)
    ap.add_argument("--max-days", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    rs = np.random.default_rng(a.seed)
    names = list(ARCHETYPES)
    # 70/25/5 -- the brittle minority is what makes an outlier list non-empty
    probs = np.array([0.25, 0.70, 0.05])

    X, ids, rows = [], [], []
    for i in range(a.n_subjects):
        arch = names[rs.choice(3, p=probs)]
        base, cv, ms, p_spike = ARCHETYPES[arch]
        base = base * rs.normal(1.0, 0.06)
        cv = cv * rs.normal(1.0, 0.10)
        n_days = int(rs.integers(a.min_days, a.max_days + 1))
        days = np.stack([_day(rs, base, cv, ms, spike=rs.random() < p_spike)
                         for _ in range(n_days)])
        X.append(days)
        sid = f"F{i:04d}"
        ids += [sid] * n_days
        rows.append(dict(id=sid, archetype=arch, n_days=n_days,
                         mean_mgdl=float(days.mean())))

    X = np.concatenate(X)[:, :, None].astype(np.float32)   # (N, 288, 1) mg/dL
    sids = np.asarray(ids)

    # Same transform as data/cohort.build: z-score, clip at +-5 sd, then divide by the
    # clip so the result lands in [-1, 1]. Constants come from THIS cohort, as with a
    # real one -- reusing the real cohort's would silently rescale the fake data.
    ZCLIP = 5.0
    mean, sd = float(X.mean()), float(X.std())
    Xn = (np.clip((X - mean) / sd, -ZCLIP, ZCLIP) / ZCLIP).astype(np.float32)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "windows.npy", Xn)
    np.save(out / "subject_ids.npy", sids)

    import pandas as pd
    pd.DataFrame(rows).to_parquet(out / "subjects.parquet")

    (out / "manifest.json").write_text(json.dumps(dict(
        name=out.name, fake=True, source="scripts/make_fake_data.py", seed=a.seed,
        n_subjects=a.n_subjects, n_windows=int(X.shape[0]), T=T, C=1, dt_min=5,
        channel="CGM", units="mg/dL", mean_mgdl=mean, sd_mgdl=sd, zclip=ZCLIP,
        denorm_formula="mgdl = x * zclip * sd_mgdl + mean_mgdl",
        note=("SYNTHETIC. Three hand-written archetypes plus noise. Use for checking "
              "that the pipeline runs; every outlier finding on it is circular."),
    ), indent=2))

    from collections import Counter
    print(f"{a.n_subjects} fake subjects, {X.shape[0]} windows -> {out}")
    print("  archetypes:", dict(Counter(r["archetype"] for r in rows)))
    print(f"  mg/dL mean {mean:.1f} sd {sd:.1f}")
    print("\nnext:  python scripts/run_outliers.py --cohort", out,
          "--out results/fake --device cpu")


if __name__ == "__main__":
    sys.exit(main())
