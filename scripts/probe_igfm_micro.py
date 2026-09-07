#!/usr/bin/env python
"""How big a micro-batch fits, and how slow it is, at a window length IG-FM never ran.

WHY THIS EXISTS. `micro_batch` is hard-coded to 32 in the adapter and that number was
chosen for T=288. Self-attention is O(T^2), so at T=2016 the same 32 is 49x the
attention work per sequence. Launching a 27-model campaign on a guess risks an OOM hours
in, or a job silently 20x slower than budgeted that dies at the walltime with nothing.

HOW THE FIXED COST IS REMOVED, AND WHY THE OBVIOUS WAY DOES NOT WORK
--------------------------------------------------------------------
`fit()` does a lot exactly once: build 8 transformer layers, deepcopy the net for the
EMA, move the training set to the device, select SDPA/cuBLAS kernels on the first pass,
grow the caching allocator. Call that F. A run of S iterations costs F + S*t, and the
projection wants t, not (F + S*t)/S.

The tempting fix -- time S iterations, then subtract the MEAN per-iteration time W times
-- is algebraically a no-op:

    (dt - (dt/S)*W) / (S - W)  ==  dt*(S-W)/S / (S-W)  ==  dt/S

It removes nothing, and the bias it fails to remove is WORST where t is smallest, i.e.
at the calibration point, so calibrating against it produces a correction of the wrong
sign. This script therefore times TWO runs, short and long, and differences them:

    t = (t_long - t_short) / (S_long - S_short)          # F cancels exactly

That costs one extra F per micro and is worth it.

CHECKPOINTS MUST BE OFF, AND `save_every` DOES NOT DO IT
--------------------------------------------------------
`_save(total_iters)` runs unconditionally at the end of `fit()`, outside the `save_every`
guard, and `resume_dir` defaults to `workdir`. So a probe that sets only `save_every`
still writes `iter-<steps>.pt` -- and the SECOND timing run, or any requeue, then finds
it, takes the `start_it >= total_iters` early return, and reports a fit that never
happened as a success at ~0.05 s/iter. Both `workdir` and `resume_dir` are set empty
here: `_save` returns immediately and the resume block is skipped.

WHAT IS HELD FIXED. The EFFECTIVE batch stays 256, exactly as at T=288 -- the adapter
computes accum = 256 // micro -- so a smaller micro is a pure cost question, not a
different experiment. Model configuration is the campaign's own, passed verbatim.

    python scripts/probe_igfm_micro.py --cohort data/cohort/matrix_d7_c1 \
        --params "$(cat scripts/pbs/dev/igfm_priv.params)" --out results/probe/d7_c1.json
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np

from cgmoutlier._env import check as _envcheck                      # noqa: E402
_envcheck()
from cgmoutlier.data.cohort import load as load_cohort              # noqa: E402
from cgmoutlier.loo.train import training_set                       # noqa: E402
from cgmoutlier.generators.registry import get as get_generator     # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--design", default="results/matrix/design/rep1")
    ap.add_argument("--job", default="base")
    ap.add_argument("--params", required=True, help="the campaign's --params, verbatim")
    ap.add_argument("--micro", default="32,16,8,4,2",
                    help="tried in this order; probing stops after --stop-after successes")
    ap.add_argument("--stop-after", type=int, default=2,
                    help="successes to collect before stopping. 2 so the cheapest of the "
                         "two largest that fit can be chosen -- more micro means fewer "
                         "accumulation steps but the same FLOPs, and training_step syncs "
                         "on .item() once per micro-batch, so bigger is not always faster")
    ap.add_argument("--short", type=int, default=8)
    ap.add_argument("--long", type=int, default=24)
    ap.add_argument("--target-iters", type=int, default=25000)
    ap.add_argument("--sample-n", type=int, default=64)
    ap.add_argument("--sample-batch", default="64,32,16,8",
                    help="tried in order until one fits. Sampling memory is governed by "
                         "sample_batch, NOT by micro_batch, so a sampling OOM must not be "
                         "charged to micro_batch -- that would fail every micro at once "
                         "and hide a knob the campaign can simply turn down.")
    ap.add_argument("--target-n", type=int, default=0, help="0 = the job's own N")
    ap.add_argument("--budget-seconds", type=float, default=0,
                    help="0 = no limit. Otherwise stop cleanly and keep what was measured "
                         "rather than being killed at the walltime with nothing written.")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import torch
    # constraints.txt exists because a pip upgrade once silently made CUDA unavailable and
    # everything fell back to CPU without an error. A CPU timing reported as a GPU timing
    # is the worst thing this script could produce, so it refuses to run at all.
    assert torch.cuda.is_available(), "CUDA 不可用 —— 拒绝把 CPU 计时当 GPU 计时报出去"

    X, sids, _ = load_cohort(a.cohort)
    job = json.loads((Path(a.design) / "jobs" / f"{a.job}.json").read_text())
    Xtr, _ = training_set(X, sids, job["subjects"])
    N, T, C = Xtr.shape
    target_n = a.target_n or int(N)
    base_params = json.loads(a.params)
    t_start = time.time()
    print(f"[probe] {a.cohort}: N={N:,} T={T} C={C}  device={torch.cuda.get_device_name(0)}",
          flush=True)
    print(f"[probe] 每个 micro 跑两次({a.short} 步和 {a.long} 步)做差,消掉一次性开销",
          flush=True)

    rec = {"cohort": a.cohort, "N": int(N), "T": int(T), "C": int(C),
           "params": base_params, "short": a.short, "long": a.long,
           "target_iters": a.target_iters, "target_n": target_n, "attempts": []}
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)

    def flush():
        # Written after EVERY attempt. If the walltime kills this process, whatever was
        # already measured is on disk -- and the case that overruns the walltime is
        # exactly the "it is very slow" answer that most changes the decision.
        ok = [x for x in rec["attempts"] if x.get("ok")]
        rec["recommended"] = min(ok, key=lambda x: x["projected_total_hours"]) if ok else None
        out.write_text(json.dumps(rec, indent=1))

    def timed_fit(micro, steps):
        p = dict(base_params)
        # workdir AND resume_dir empty: no checkpoint is written and none is read.
        p.update(steps=steps, micro_batch=micro, workdir="", resume_dir="")
        gen = get_generator("igfm")(T=T, C=C, params=p, device="cuda", seed=2026)
        torch.cuda.synchronize(); t0 = time.time()
        gen.fit(Xtr)
        torch.cuda.synchronize()
        return gen, time.time() - t0

    def release(*objs):
        for o in objs:
            del o
        gc.collect()                    # break the traceback -> frame -> tensors cycle
        torch.cuda.empty_cache()

    n_ok = 0
    for micro in [int(x) for x in a.micro.split(",")]:
        if n_ok >= a.stop_after:
            rec["attempts"].append({"micro": micro, "ok": None, "note": "已有足够成功样本,跳过"})
            continue
        spent = time.time() - t_start
        if a.budget_seconds and spent > a.budget_seconds:
            rec["attempts"].append({"micro": micro, "ok": None,
                                    "note": f"时间预算用完({spent:.0f}s),跳过"})
            flush(); continue

        att = {"micro": micro, "accum": max(1, int(base_params.get("batch_size", 256)) // micro)}
        g = None
        try:
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            g0, t_short = timed_fit(micro, a.short)
            release(g0)
            torch.cuda.reset_peak_memory_stats()
            g, t_long = timed_fit(micro, a.long)
            per_iter = (t_long - t_short) / (a.long - a.short)
            att.update(ok=True,
                       t_short=round(t_short, 2), t_long=round(t_long, 2),
                       sec_per_iter=round(per_iter, 4),
                       fixed_overhead_s=round(t_short - per_iter * a.short, 2),
                       peak_alloc_gib=round(torch.cuda.max_memory_allocated() / 2**30, 2),
                       # reserved, not allocated: allocated hides allocator overhead and
                       # fragmentation, and "does it fit with headroom" is the question.
                       peak_reserved_gib=round(torch.cuda.max_memory_reserved() / 2**30, 2),
                       projected_fit_hours=round(per_iter * a.target_iters / 3600, 2))
        except torch.cuda.OutOfMemoryError:
            att.update(ok=False, error="OOM")
            print(f"[probe] micro={micro:>3}  训练 OOM", flush=True)
        except Exception as e:                                   # noqa: BLE001
            att.update(ok=False, error=type(e).__name__, detail=str(e)[:300])
            print(f"[probe] micro={micro:>3}  {type(e).__name__}: {str(e)[:200]}", flush=True)

        if att.get("ok"):
            # Sampling gets its own try and its own knob. It is timed on the model that
            # is already in memory; the weights are junk after 24 iterations but the cost
            # is weight-independent -- a fixed number of denoiser passes per sample.
            att["sample"] = {}
            for sb in [int(x) for x in a.sample_batch.split(",")]:
                try:
                    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
                    t0 = time.time()
                    g.sample(a.sample_n, {"sample_batch": sb})
                    torch.cuda.synchronize()
                    ds = time.time() - t0
                    att["sample"] = {
                        "sample_batch": sb, "n": a.sample_n, "seconds": round(ds, 2),
                        "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 2**30, 2),
                        "projected_hours": round(ds / a.sample_n * target_n / 3600, 2)}
                    break
                except torch.cuda.OutOfMemoryError:
                    att["sample"] = {"sample_batch": sb, "error": "OOM"}
                    torch.cuda.empty_cache()
            sh = att["sample"].get("projected_hours")
            if sh is None:
                att.update(ok=False, error="sampling OOM at every sample_batch")
                print(f"[probe] micro={micro:>3}  训练跑得动,但采样在所有 sample_batch 上都 OOM",
                      flush=True)
            else:
                att["projected_total_hours"] = round(att["projected_fit_hours"] + sh, 2)
                n_ok += 1
                print(f"[probe] micro={micro:>3} accum={att['accum']:>3}  OK  "
                      f"{att['sec_per_iter']:.3f} s/iter  一次性开销 "
                      f"{att['fixed_overhead_s']:.1f}s  显存 {att['peak_reserved_gib']:.1f} GiB"
                      f"  ->  训练 {att['projected_fit_hours']:.1f} h + 采样 {sh:.1f} h "
                      f"(sample_batch={att['sample']['sample_batch']}) = "
                      f"{att['projected_total_hours']:.1f} h/模型", flush=True)
        release(g)
        rec["attempts"].append(att)
        flush()

    flush()
    best = rec["recommended"]
    if best is None:
        print("\n[probe] 没有一个配置跑得起来 —— 不要提交正式作业。")
        return 1
    print(f"\n[probe] 建议 micro_batch={best['micro']} (accum={best['accum']}, "
          f"sample_batch={best['sample']['sample_batch']}), "
          f"单模型约 {best['projected_total_hours']:.1f} 小时,"
          f"27 个约 {best['projected_total_hours'] * 27:.0f} 卡时")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
