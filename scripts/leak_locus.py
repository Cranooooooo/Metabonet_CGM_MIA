#!/usr/bin/env python
"""泄漏本身藏在哪个坐标里 —— 不是「两组的差别藏在哪」。

为什么要有这个脚本
------------------
`localise_leak.py` 在五个变换下算的是每个受试者的 **gap**,然后我们从中算出两臂之间的
可分开程度(arm AUC)。**那衡量的是「离群者和普通人有多不一样」,不是「一个人能不能被
认出来」。** 两者是不同的问题,而设计防御需要的是后者:

  · 如果一个人的可识别性在 `sorted` 下依然很高、在 `zscore` 下塌到 0.5,
    那么泄漏确实藏在数值分布里,防御就该动数值分布。
  · 如果反过来,`sorted` 一做可识别性就塌了,那泄漏是绑在时序上的,
    防御必须动时间结构 —— 那会是完全不同的机制。

之前我们只测了前一个问题,却拿它的答案去回答后一个。这个脚本补上后者。

怎么测
------
对每个受试者、每个变换 T:把他的真实记录和两个发布集都先做 T 变换,再逐窗口算
「到最近一条假数据的距离」,得到 d_out 和 d_in 两组数,然后取它们的不配对秩 AUC ——
和 `subject_auc.py` 里 `auc_ci` 的点估计完全一样的统计量,这样数字可以直接和
results/matrix/subject_auc/ 里已有的 raw 结果对上。

0.5 = 完全认不出。读三个数:超过 0.55 的人数、中位数、最高值。

    python scripts/leak_locus.py --design results/matrix/design/rep1 \
        --cohort data/cohort/matrix_d1_c1 --runs results/runs/matrix_d1_c1 \
        --out results/matrix/leak_locus/d1_c1.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cgmoutlier.data.cohort import load as load_cohort            # noqa: E402
from cgmoutlier.attack.statistic import window_distances, _match  # noqa: E402

_diff   = lambda X: np.diff(X, axis=1)
_sorted = lambda X: np.sort(X, axis=1)
_hourly = lambda X: X[:, :X.shape[1] // 12 * 12].reshape(
              len(X), X.shape[1] // 12, 12, X.shape[2]).mean(2)
_zscore = lambda X: (X - X.mean(1, keepdims=True)) / (X.std(1, keepdims=True) + 1e-6)


def _floor(X):
    """地板对照:把每个窗口内部的数值彻底打乱,同一份数据、同一套距离计算,
    但任何「这一段属于谁」的结构都不复存在。它应当读出 0.5。
    读不出 0.5 就说明流程本身有偏,那么上面所有变换的数字都要重看。
    用固定种子,所以可复现。"""
    rng = np.random.default_rng(20260831)
    Y = X.copy()
    for i in range(len(Y)):
        for c in range(Y.shape[2]):
            rng.shuffle(Y[i, :, c])
    return Y


TRANSFORMS = {
    # ---- 单项破坏 ----
    "raw":            ("什么都不破坏",            lambda X: X),
    "diff":           ("水平",                    _diff),
    "sorted":         ("时序",                    _sorted),
    "hourly":         ("一小时内的细节",          _hourly),
    "zscore":         ("水平 + 幅度",             _zscore),
    # ---- 两项同时破坏 ----
    "diff+sorted":    ("水平 + 时序",             lambda X: _sorted(_diff(X))),
    "zscore+sorted":  ("水平 + 幅度 + 时序",      lambda X: _sorted(_zscore(X))),
    "hourly+diff":    ("细节 + 水平",             lambda X: _diff(_hourly(X))),
    "hourly+sorted":  ("细节 + 时序",             lambda X: _sorted(_hourly(X))),
    "hourly+zscore":  ("细节 + 水平 + 幅度",      lambda X: _zscore(_hourly(X))),
    # ---- 地板 ----
    "shuffle":        ("全部(地板对照)",          _floor),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", required=True)
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    d = json.loads((Path(a.design) / "design.json").read_text())
    X, sids, _ = load_cohort(a.cohort)
    X = np.asarray(X)
    sids = np.asarray([str(s) for s in sids])
    S_base = np.asarray(np.load(Path(a.runs) / "base" / "samples.npy", mmap_mode="r"),
                        np.float32)

    per = {}
    for p in d["pairs"]:
        t, grp = str(p["target"]), p["group"]
        f = Path(a.runs) / p["member"] / "samples.npy"
        if not f.exists():
            print(f"[locus] {t}: 缺 {f},跳过", flush=True)
            continue
        R = np.ascontiguousarray(X[sids == t], np.float32)
        S_in, S_out, _ = _match(np.asarray(np.load(str(f)), np.float32), S_base,
                                t, a.seed, True)
        rec = {}
        for name, (_d, fn) in TRANSFORMS.items():
            Rt, Si, So = fn(R), fn(S_in), fn(S_out)
            wi = window_distances(Rt, Si, set_reduce="min")
            wo = window_distances(Rt, So, set_reduce="min")
            n = len(wi)
            # 和 subject_auc.py 的 auc_ci 点估计同一个统计量
            rec[name] = float(stats.mannwhitneyu(wo, wi, alternative="two-sided").statistic
                              / (n * n))
        per[t] = dict(group=grp, n_windows=int(len(R)), **rec)
        print(f"[locus] {t:14} {grp:8} " +
              "  ".join(f"{k}={v:.3f}" for k, v in rec.items()), flush=True)

    print("\n=== 每个人自己能不能被认出来,在五个变换下 ===")
    print("  0.5 = 完全认不出。看「>0.55 的人数」这一列。")
    print(f"  {'变换':16s}{'破坏了什么':22s}{'>0.55':>8s}{'中位':>8s}{'最高':>8s}"
          f"{'离群中位':>10s}{'普通中位':>10s}")
    summary = {}
    for name, (desc, _) in TRANSFORMS.items():
        v = [r[name] for r in per.values()]
        o = [r[name] for r in per.values() if r["group"] == "outlier"]
        c = [r[name] for r in per.values() if r["group"] != "outlier"]
        summary[name] = dict(n_above_055=int(sum(x > 0.55 for x in v)), n_total=len(v),
                             median=float(np.median(v)), max=float(max(v)),
                             median_outlier=float(np.median(o)),
                             median_control=float(np.median(c)))
        print(f"  {name:16s}{desc:22s}{summary[name]['n_above_055']:>4d}/{len(v):<3d}"
              f"{np.median(v):>8.3f}{max(v):>8.3f}{np.median(o):>10.3f}{np.median(c):>10.3f}")
    print("\n  怎么读:")
    print("   · 地板对照(shuffle)应当接近 0.5。它不接近 0.5,上面所有数字都要重看。")
    print("   · 单项里人数掉得最多的,就是承载泄漏最多的那种信息。")
    print("   · 两项同时破坏如果掉到接近地板,说明泄漏就由这两样承载,防御动它们即可;")
    print("     如果还剩不少,说明还有第三样东西没被这几个变换覆盖到。")

    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(dict(cohort=a.cohort, runs=a.runs, seed=a.seed,
                       summary=summary, per_subject=per), open(a.out, "w"), indent=2)
        print(f"\n写入 {a.out}")


if __name__ == "__main__":
    main()
