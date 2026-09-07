#!/usr/bin/env python
"""Quality as a function of training budget, for an IG-FM run that kept its checkpoints.

WHY THIS EXISTS. DiM-TS's quality-vs-budget curve on this project peaks at 20% of the
budget and then DEGRADES -- on d7_c1 from Context-FID 0.131 at 20k steps to 0.327 at
100k. If IG-FM does the same, the seven-day campaign can be trained at a fraction of the
budget and becomes affordable (120 h/model -> ~23 h/model). If IG-FM instead improves
monotonically -- which its endpoint hints at, since 25000 iterations already beats
DiM-TS's best-ever score -- the budget cannot be cut and we need a different plan.
Nobody has measured it: every IG-FM run so far set `keep_checkpoints: 2`, and pruning is
oldest-first, so the deleted weights are exactly the early points this curve is about.

WHY NOT `scripts/resample.py`. That path needs `generator.resample`, which DiM-TS has and
the IG-FM adapter does not. Adding it would mean editing `generators/igfm.py` while a
campaign job runs against it, which this project has deliberately frozen. So the
checkpoint is restored here instead: `_save` writes {"model","ema","opt","iter"} and
`sample()` draws from `self._ema.module`, so restoring the EMA and setting `_fitted` is
the whole of what a `resample` method would do.

THE DRAW MUST BE SEEDED HERE, AND NOTHING ELSE WILL DO IT
---------------------------------------------------------
The generator's `seed` argument is only stored; the sole `torch.manual_seed` on this path
lives inside `fit()`, which this script never calls. `sample_unconditional` draws its
noise from the global default generator, which is seeded from OS entropy at process
start. Left alone, each of the 8 milestones would carry its own uncontrolled draw, the
curve would be irreproducible, and a difference of a few thousandths -- the size of the
signal we are looking for near the endpoint -- could be manufactured or erased by it.
Seeding immediately before each `sample()` makes all 8 milestones integrate the SAME
initial noise, so the only thing differing between points is the model. That is a paired
comparison and is strictly stronger than what the campaign itself does.

It is NOT the campaign's own draw, and cannot be: training advances the RNG before
`run()` samples, which is why `loo/train.py:resample()` stamps `bit_reproducible: False`.
The same stamp is written here.

WHAT IS AND IS NOT COMPARABLE
-----------------------------
Every milestone releases the same K with the same `sampling_steps` and is scored in one
`eval_quality_tsgem` call at one seed, so the points differ only in training length. The
last milestone should land near the campaign's own recorded score; treat a difference of
a few thousandths as the instrument (Context-FID's same-seed spread is up to 0.0009 and
its seed-to-seed sd is 0.008) and a difference of tenths as evidence this is not the same
model.

⚠️ THE LEFTMOST POINT IS DEPRESSED BY EMA WARM-UP, NOT ONLY BY UNDERTRAINING.
`sample()` draws from the EMA, decay 0.999. At 2000 iterations the EMA still carries
0.999^2000 ~= 13.5% of the random initialisation; 1.8% at 4000, 0.25% at 6000. m2000 is a
faithful measurement of "a run configured for 2000 steps", but it is NOT clean evidence
about undertraining, and the bias tilts the shape toward monotone-improving -- i.e.
toward "the budget cannot be cut", which costs GPU-hours rather than producing a false
claim. DiM-TS's comparison curve was built through a different mechanism (`resample` from
milestone dirs), so the two curves' early points are not measured the same way.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from cgmoutlier._env import check as _envcheck                      # noqa: E402
_envcheck()
from cgmoutlier.data.cohort import load as load_cohort              # noqa: E402
from cgmoutlier.loo.train import training_set, _job_seed            # noqa: E402
from cgmoutlier.generators.registry import get as get_generator     # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="a run dir holding iter-*.pt and meta.json")
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--design", default="results/matrix/design/rep1")
    ap.add_argument("--job", default="base")
    ap.add_argument("--milestones", required=True, help="comma-separated iteration counts")
    ap.add_argument("--out-root", required=True, help="one subdir per milestone is written here")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    import torch
    assert torch.cuda.is_available(), "CUDA 不可用 —— 采样会退回 CPU 并慢到没有意义"

    run = Path(a.run)
    meta = json.loads((run / "meta.json").read_text())
    params = dict(meta.get("params") or {})
    X, sids, _ = load_cohort(a.cohort)
    job = json.loads((Path(a.design) / "jobs" / f"{a.job}.json").read_text())

    # Provenance. IG-FM writes no train.npy, so `resample()`'s re-derivation check is not
    # available -- but meta.json already carries what is needed. Without this, pointing
    # --run at the protected campaign model results/runs/igfm_priv_d1_c1/base (same T,
    # same C, same K, and it has iter-24000/25000 on disk right now) sails past the only
    # other guard, the shape check, and silently yields a two-point "curve" from it.
    fp = hashlib.sha1("\n".join(map(str, job["subjects"])).encode()).hexdigest()[:16]
    for key, want in (("job", job["name"]), ("cohort", a.cohort), ("subjects_sha1", fp)):
        got = meta.get(key)
        if got is not None and got != want:
            raise SystemExit(f"{run}/meta.json 的 {key} 是 {got!r},不是 {want!r} —— "
                             f"这个 run 不是本 cohort 的本 job,拒绝拿它画曲线")

    Xtr, _ = training_set(X, sids, job["subjects"])
    N, T, C = Xtr.shape
    K = int(meta.get("K") or N)
    job_seed = _job_seed(a.seed, job["name"])
    print(f"[curve] {run}  N={N:,} T={T} C={C} K={K:,} 采样种子={job_seed}", flush=True)

    want_ms = [int(x) for x in a.milestones.split(",")]
    have = {int(p.stem.split("-")[1]): p for p in run.glob("iter-*.pt")}
    print(f"[curve] 盘上的存档: {sorted(have)}", flush=True)
    missing = [m for m in want_ms if m not in have]
    if missing:
        raise SystemExit(f"缺少这些里程碑的存档: {missing}\n"
                         f"(keep_checkpoints 是从旧到新剪的,早期里程碑正是这条曲线要看的 —— "
                         f"重训时把 keep_checkpoints 开够)")

    out_root = Path(a.out_root); out_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for m in want_ms:
        od = out_root / f"m{m}"
        # Both files, not just samples.npy: meta.json is written first below, but a kill
        # between the two still must not leave a dir that gets skipped AND then crashes
        # eval_quality_tsgem's single 8-milestone call on a missing meta.
        if (od / "samples.npy").exists() and (od / "meta.json").exists():
            print(f"[curve] m{m}: 已有样本和 meta,跳过", flush=True)
            rows.append({"milestone": m, "skipped": True})
            continue

        p = dict(params); p.update(workdir="", resume_dir="")   # never write/read checkpoints
        gen = get_generator("igfm")(T=T, C=C, params=p, device=a.device, seed=job_seed)
        gen._model = gen._build().to(torch.device(a.device))
        gen._ema = torch.optim.swa_utils.AveragedModel(gen._model)
        ck = torch.load(have[m], map_location=a.device, weights_only=False)
        # Without this, all 8 milestones could load the same weights and the curve would
        # be flat for a reason that has nothing to do with training.
        assert int(ck["iter"]) == m, f"{have[m]} 里记的是 {ck['iter']},不是 {m}"
        gen._model.load_state_dict(ck["model"])     # strict: a mismatch raises, not silently wrong
        gen._ema.load_state_dict(ck["ema"])
        gen._fitted = True

        torch.manual_seed(job_seed)                 # 见文件头:不在这里种,8 个点各自是一次盲抽
        t0 = time.time()
        S = np.asarray(gen.sample(K), dtype=np.float32)
        dt = time.time() - t0
        if S.shape != (K, T, C):
            raise ValueError(f"m{m}: 采样返回 {S.shape},期望 {(K, T, C)}")
        nan = float(np.isnan(S).mean())
        if nan:
            raise ValueError(f"m{m}: {nan:.2%} 的样本是 NaN,拒绝写出去当数据读")

        od.mkdir(parents=True, exist_ok=True)
        # meta BEFORE the samples rename: eval_quality_tsgem reads meta.json with no
        # existence guard, so a dir with samples and no meta takes down the one call that
        # scores all 8 milestones, not just that dir. Same failure loo/train.py:run()
        # carries an explicit guard for.
        (od / "meta.json").write_text(json.dumps(
            {"job": job["name"], "role": job["role"], "milestone": m,
             "from_checkpoint": str(have[m]), "K": K, "T": int(T), "C": int(C),
             "generator": "igfm", "params": params, "job_seed": job_seed,
             "cohort": a.cohort, "subjects_sha1": fp, "sample_seconds": round(dt, 2),
             "sample_range": [float(S.min()), float(S.max())],
             "bit_reproducible": False,
             "note": "budget-curve point from a restored checkpoint "
                     "(scripts/igfm_budget_curve.py), NOT a campaign model. The draw is "
                     "seeded per milestone so the 8 points are paired, but it is not the "
                     "draw the campaign made -- training advances the RNG before it "
                     "samples."}, indent=1))
        tmp = od / "samples.tmp.npy"
        np.save(tmp, S); tmp.rename(od / "samples.npy")
        print(f"[curve] m{m}: {dt/60:.1f} 分钟 -> {od/'samples.npy'}  "
              f"范围 [{S.min():.3f}, {S.max():.3f}]", flush=True)
        rows.append({"milestone": m, "sample_seconds": round(dt, 2)})
        del gen, ck
        torch.cuda.empty_cache()

    (out_root / "curve_index.json").write_text(json.dumps(
        {"run": str(run), "cohort": a.cohort, "K": K, "job_seed": job_seed,
         "milestones": want_ms, "rows": rows}, indent=1))
    print(f"\n[curve] 写好 {len(want_ms)} 个里程碑 -> {out_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
