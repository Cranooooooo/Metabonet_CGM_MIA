#!/usr/bin/env python
"""What does one model of the paired design actually cost?

    python scripts/probe_generator.py --generator fourier_diff timevae

49 models is not a number to commit to on an estimate, and the published defaults are
not comparable across baselines: DiM-TS denoises in 500 steps and FourierDiffusion in
1000, DiM-TS counts a step budget and FourierDiffusion counts epochs. Reading those
off the configs answers "which default is bigger", not "which is cheaper".

So this measures one thing per generator, on identical data, at whatever budget the
caller sets:

    fit_seconds        a deliberately small training budget -- a rate, not a run
    seconds_per_sample sampling throughput, extrapolated to the real K

Sampling is measured because it is the term most likely to dominate. With K = the
training-set size, every model in the design releases ~177k sequences; at 500 or 1000
denoising steps each, that can cost more than the training it follows. If it does, the
choice is between generators, between a smaller K, and between fewer denoising steps --
and those are different trades that should be made on a number.

WHAT THIS DOES NOT MEASURE
--------------------------
Quality. Nothing here says whether a generator's samples are usable CGM, and a
generator that is fast because it has not learned anything is not a saving. The
clinical battery in `cgmoutlier.clinical` is what would answer that, on real runs.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from cgmoutlier._env import check as _envcheck                     # noqa: E402
_envcheck()
from cgmoutlier.data.cohort import load as load_cohort                # noqa: E402
from cgmoutlier.generators.registry import get as get_generator       # noqa: E402

# A small, explicitly-stated budget per generator: enough steps to time a step, few
# enough to fit in a probe. These are NOT the training settings for a real run.
# The step counts must be >= 4 for --train-rate: the second point is budget // 4, and
# max(1, 1 // 4) == 1 gives two identical points and no slope.
PROBE_BUDGET = {
    # fourier_diff: collapse_ratio_max 关掉。fit() 末尾有一道无条件的质量闸,
    # ratio>0.5 直接 RuntimeError —— 冒烟预算下模型必然还在塌缩状态,那道闸会把
    # 已经量到的成本一起抛掉(probe() 的异常直接冒到 main 的 except)。
    # keep_best 在探针里【必须关掉】。它的回调每个 epoch 拿 dm.X_train[:512] 做一次
    # 整批前向(fourier_diff.py:158),512 这个数是写死的 —— T=288 时无所谓,T=2016 时
    # 每条约 93 MB 激活,一次要 46.5 GiB,直接 OOM。而且 batch_size 改成 6 或 4 都
    # 不影响它,所以看起来像"显存装不下这个模型",实际只是这一个评估批。
    # ⚠️ 正式跑必须把 keep_best 开回 True(它挡的是真实的塌缩,35 个 fold 中过招),
    #    那时要改的是把这个 512 按 T 缩小或分块,不是关掉它。
    "fourier_diff": dict(train=dict(max_epochs=4, batch_size=64,
                                    collapse_ratio_max=1.1, keep_best=False),
                         sample=dict(num_diffusion_steps=1000, sample_batch=128)),
    # diffusion_ts: gradient_accumulate_every 默认是 2,也就是一个「步」其实喂
    # 2*batch_size 条。设成 1,一个步就正好是 batch_size 条,等预算换算才不会差 2 倍。
    "diffusion_ts": dict(train=dict(max_epochs=200, batch_size=64,
                                    gradient_accumulate_every=1), sample=dict()),
    # diffwave: total_iters 优先于 max_epochs(适配器 :121),直接给迭代数更干净。
    "diffwave":     dict(train=dict(total_iters=200, batch_size=64),
                         sample=dict(sample_batch=128)),
    "timevae":      dict(train=dict(max_epochs=8, batch_size=64), sample=dict()),
    "igfm":         dict(train=dict(steps=200, batch_size=64), sample=dict()),
    "copy_paste":   dict(train=dict(), sample=dict()),
}


def count_params(gen):
    """Best-effort parameter count. Adapters hold their model under different names."""
    import torch
    for attr in ("_model", "model", "net", "_net"):
        m = getattr(gen, attr, None)
        if isinstance(m, torch.nn.Module):
            return int(sum(p.numel() for p in m.parameters()))
    return None


# ⚠️ 这张表必须和适配器【真正读的】键一致。原先 diffusion_ts 写 train_num_steps、
#    padts/diffwave 写 max_steps —— 三个适配器其实都读 max_epochs,所以那些预算被
#    静默忽略,适配器跑的是自己的默认值。证据留在 results/probe/generators.json:
#    diffusion_ts 的 50 步和 200 步只差 1.9 秒("setup" 434.7 秒 ~= 默认 12000 步),
#    diffwave 的每步成本是负的。改键之前,这个探针量的是「默认预算跑一遍要多久」。
STEP_KEY = {"fourier_diff": "max_epochs", "timevae": "max_epochs",
            "diffusion_ts": "max_epochs", "diffwave": "total_iters",
            "igfm": "steps"}


def train_rate(name, X, device, seed, budget, gen=None, params=None):
    """Seconds per training step, from two budgets rather than one.

    A single small budget cannot separate the per-step cost from the fixed setup:
    the first run of this probe reported 431 s for 200 steps of a 224k-parameter
    model, which is setup, not 2.15 s/step. Two points give the slope, and the
    intercept is the overhead that a real 100k-step run pays once.

    `gen`, if given, is trained at the HIGH budget and left fitted, so the caller
    can sample from it instead of paying for a third full-budget fit. The low
    point always uses a throwaway model -- reusing one would train it twice and
    the second fit would start from the first fit's weights.
    """
    key = STEP_KEY.get(name)
    if key is None or key not in budget:
        return None
    lo = dict(budget); lo[key] = max(1, budget[key] // 4)
    if lo[key] == budget[key]:
        return None
    pts = []
    for b, g in ((lo, None), (budget, gen)):
        if g is None:
            # ⚠️ 低点这个一次性模型必须用【和高点相同的 params】。原先这里不传
            # params,而调用方的 gen 是带 params 造的 —— 一旦用 --gen-params 探
            # 容量档位,两个计时点就来自【两个不同大小的模型】,斜率(t1-t0)/(n1-n0)
            # 把「大模型 vs 小模型」的差算成了「每步成本」。默认档(params=None)下
            # 两边一致,所以这个 bug 在 8/9 月所有默认容量的探针里都不会显形。
            g = get_generator(name)(T=X.shape[1], C=X.shape[2],
                                    params=dict(params or {}), device=device, seed=seed)
        t0 = time.time()
        g.fit(X, b)
        pts.append((b[key], time.time() - t0))
    (n0, t0_), (n1, t1_) = pts
    per = (t1_ - t0_) / (n1 - n0)
    setup = t0_ - per * n0
    rec = dict(unit=key, per_unit_seconds=round(per, 4),
               setup_seconds=round(setup, 1), points=pts, _hi_seconds=t1_)
    # A non-positive slope, or stepping that is a rounding error next to setup,
    # means nothing about the per-step rate was measured. Two causes produce
    # identical symptoms and this test cannot tell them apart:
    #   (a) the adapter never reads this key, so both runs used its own default
    #       -- the bug that produced diffwave -0.0065 s/step against 561 s of
    #       'setup' in results/probe/generators.json;
    #   (b) setup genuinely swamps a deliberately tiny probe budget.
    # The discriminator is whether per*n grows in proportion when the budget is
    # raised ~10x. It does under (b) and does not under (a). Check the adapter's
    # cfg.get() for this key first -- that is one grep and settles it.
    if per <= 0 or (setup > 0 and per * n1 < 0.1 * setup):
        rec["unmeasured"] = (
            f"setup ({setup:.0f}s) dominates both budgets ({n0}, {n1} {key}); "
            f"the per-{key} cost is NOT resolved. Either the adapter ignores "
            f"'{key}' (grep its cfg.get first), or the budget is too small. "
            f"Do not read per_unit_seconds.")
    return rec


def probe(name, X, n_sample, K, device, seed, params=None, with_rate=False):
    budget = PROBE_BUDGET.get(name, dict(train=dict(), sample=dict()))
    N, T, C = X.shape
    gen = get_generator(name)(T=T, C=C, params=dict(params or {}),
                              device=device, seed=seed)

    # with_rate 时,全预算那一次 fit 由 train_rate 负责(它的高点就是全预算),
    # 这里不再重复跑一遍 —— 原先是 3 次 fit/生成器/格子,对 2 小时的 gdev 上限太奢侈。
    if with_rate:
        rate = train_rate(name, X, device, seed, budget["train"], gen=gen, params=params)
        # train_rate 可能返回 None(键不在预算里,或高低两点相等),那样 gen
        # 从来没被 fit 过,下面的 sample 会直接抛 RuntimeError。兜回普通路径。
        if rate is None:
            t0 = time.time(); gen.fit(X, budget["train"]); t_fit = time.time() - t0
        else:
            t_fit = rate.pop("_hi_seconds", None)
    else:
        rate = None
        t0 = time.time()
        gen.fit(X, budget["train"])
        t_fit = time.time() - t0

    t0 = time.time()
    S = gen.sample(n_sample, budget["sample"])
    t_sample = time.time() - t0

    S = np.asarray(S)
    per = t_sample / max(1, n_sample)
    return dict(
        generator=name, n_train=int(N), T=int(T), C=int(C),
        gen_params=dict(params or {}),
        params=count_params(gen), train_budget=budget["train"],
        sample_budget=budget["sample"],
        fit_seconds=(round(t_fit, 1) if t_fit is not None else None),
        train_rate=rate, n_sampled=int(n_sample),
        sample_seconds=round(t_sample, 1),
        seconds_per_sample=round(per, 5),
        hours_for_K=round(per * K / 3600, 2),
        hours_for_49_models_sampling_only=round(per * K * 49 / 3600, 1),
        sample_shape=list(S.shape), sample_nan_frac=float(np.isnan(S).mean()),
        sample_range=[float(np.nanmin(S)), float(np.nanmax(S))],
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generator", nargs="+", required=True)
    ap.add_argument("--cohort", default="data/cohort/metabonet875")
    ap.add_argument("--n-train", type=int, default=20_000,
                    help="windows used for the probe's training budget")
    ap.add_argument("--n-sample", type=int, default=512,
                    help="samples drawn to time sampling")
    ap.add_argument("--K", type=int, default=177_000,
                    help="the real released-set size, for the extrapolation")
    ap.add_argument("--out", default="results/probe/generators.json")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--gen-params", default=None,
                    help="JSON,并进生成器的构造参数(容量等)。main() 原本从不给"
                         " probe() 传 params,所以容量档位没法从命令行试。")
    ap.add_argument("--batch-size", type=int, default=None,
                    help="override every generator's train batch. T=2016 needs a "
                         "much smaller one: IG-FM OOM'd at 32 and 16 on a 40 GB A100 "
                         "and only fit at 8, and these three have no OOM fallback.")
    ap.add_argument("--train-rate", action="store_true",
                    help="also fit at a second, smaller budget so the per-step cost "
                         "can be separated from the fixed setup. Roughly doubles the "
                         "probe's runtime.")
    a = ap.parse_args()

    X, sids, man = load_cohort(a.cohort)
    X = np.ascontiguousarray(np.asarray(X)[:a.n_train], dtype=np.float32)
    if a.batch_size is not None:      # 覆盖每个生成器的训练批大小
        for spec in PROBE_BUDGET.values():
            if "batch_size" in spec["train"]:
                spec["train"]["batch_size"] = a.batch_size
    print(f"[probe] {X.shape} windows, device={a.device}, "
          f"extrapolating to K={a.K:,}\n", flush=True)

    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in a.generator:
        print(f"[probe] {name} ...", flush=True)
        try:
            r = probe(name, X, a.n_sample, a.K, a.device, a.seed,
                      params=json.loads(a.gen_params) if a.gen_params else None,
                      with_rate=a.train_rate)
        except Exception as e:                      # one bad adapter must not cost
            r = dict(generator=name, error=f"{type(e).__name__}: {e}")   # the others
            print(f"[probe] {name} FAILED: {r['error']}", file=sys.stderr, flush=True)
        rows.append(r)
        # 每量完一个就写盘。原先只在全部跑完才写 —— gdev 的 2 小时是硬上限,
        # 而「非常慢」恰恰是最耗时间也最该被记下来的那个答案。
        out.write_text(json.dumps(rows, indent=2))
        try:                       # 一个生成器的显存不该拖累下一个
            import torch, gc
            gc.collect(); torch.cuda.empty_cache()
        except Exception:
            pass
        print(json.dumps(r, indent=2), flush=True)


    ok = [r for r in rows if "error" not in r]
    if ok:
        print(f"\n{'generator':<14}{'params':>10}{'s/sample':>11}"
              f"{'h for K':>10}{'h, 49 models':>14}")
        for r in sorted(ok, key=lambda r: r.get("seconds_per_sample") or 0.0):
            p = f"{r['params']:,}" if r["params"] else "?"
            print(f"{r['generator']:<14}{p:>10}{r['seconds_per_sample']:>11.5f}"
                  f"{r['hours_for_K']:>10.2f}{r['hours_for_49_models_sampling_only']:>14.1f}")
        print("\nsampling only; training is on top. Quality is not measured here.")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
