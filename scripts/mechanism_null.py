#!/usr/bin/env python
"""信号住在哪 —— 在【真零假设】下重算,而不是对着那个坏掉的 floor。

`leak_locus.py` 的消融本身(毁掉水平/时序/细节后重跑攻击)是有效的,坏的是它拿来
比较的那个参照:`_floor` 沿时间轴打乱,逐值保留了窗口的数值分布,而 CGM 的身份
几乎全住在那里。所以「毁掉 X 之后降到地板」这句话没有依据。

这里换成 membership_null.py 那个零假设:对目标 t,每个 include_j(j≠t)都是合法
非成员,于是每个变换下都有 25 个真零假设抽样,可以问「这个变换之后还剩几个
受试者显著」。只跑泄漏最重的那个 arm —— 机制是关于【它的信号住在哪】。

    qsub scripts/pbs/dev/mechanism_null.pbs
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

from cgmoutlier._env import check as _envcheck                       # noqa: E402
_envcheck()
from cgmoutlier.data.cohort import load as load_cohort               # noqa: E402
from cgmoutlier.attack.statistic import window_distances, _match     # noqa: E402

_diff   = lambda X: np.diff(X, axis=1)
_sorted = lambda X: np.sort(X, axis=1)
_hourly = lambda X: X[:, :X.shape[1] // 12 * 12].reshape(
              len(X), X.shape[1] // 12, 12, X.shape[2]).mean(2)
_level  = lambda X: X - X.mean(1, keepdims=True)
_zscore = lambda X: (X - X.mean(1, keepdims=True)) / (X.std(1, keepdims=True) + 1e-6)

TRANSFORMS = {
    "untouched":      lambda X: X,
    "hourly":         _hourly,                       # 一小时内的细节
    "level":          _level,                        # 只去绝对水平
    "zscore":         _zscore,                       # 水平 + 幅度
    "diff":           _diff,                         # 水平 + 低频
    "sorted":         _sorted,                       # 时序
    "diff+sorted":    lambda X: _sorted(_diff(X)),   # 两者
}
ARMS = {"d1_c1": "results/runs/matrix_d1_c1",
        "d1_c2": "results/runs/matrix_d1_c2",
        "d7_c1": "results/runs/matrix_d7_c1"}
DESIGN = Path("results/matrix/design/rep1"); SEED = 2026


def main() -> int:
    design = json.loads((DESIGN / "design.json").read_text())
    pairs = design["pairs"]
    out = {}
    for cell, tree in ARMS.items():
        tree = Path(tree)
        X, sids, _ = load_cohort(f"data/cohort/matrix_{cell}")
        sids = np.asarray([str(s) for s in sids])
        names = ["base"] + [p["member"] for p in pairs]
        if any(not (tree / n / "samples.npy").exists() for n in names):
            print(f"[mech] {cell}: 模型不全,跳过", flush=True); continue
        S = {n: np.load(tree / n / "samples.npy", mmap_mode="r") for n in names}
        kmin = min(len(v) for v in S.values())
        cell_out = {}
        for tname, fn in TRANSFORMS.items():
            nsig = 0; sig_o = 0; sig_c = 0; gaps = []
            for p in pairs:
                t, grp, mem = str(p["target"]), p["group"], p["member"]
                R = fn(np.ascontiguousarray(np.asarray(X)[sids == t], np.float32))
                dm = {}
                for n in names:
                    Sn, _, _ = _match(np.asarray(S[n], np.float32),
                                      np.empty((kmin,) + S[n].shape[1:], np.float32),
                                      t, SEED, True)
                    dm[n] = float(window_distances(R, fn(Sn), set_reduce="min").mean())
                others = [q["member"] for q in pairs if q["member"] != mem]
                ref = float(np.mean([dm[o] for o in others]))
                obs = ref - dm[mem]                       # 不依赖 base
                null = np.array([ref - dm[o] for o in others])
                p_emp = float((null >= obs).sum() + 1) / (len(null) + 1)
                gaps.append(obs)
                if p_emp < 0.05:
                    nsig += 1; sig_o += grp == "outlier"; sig_c += grp == "control"
            cell_out[tname] = dict(n_sig=nsig, sig_outlier=int(sig_o),
                                   sig_control=int(sig_c),
                                   median_gap=float(np.median(gaps)))
            print(f"[mech] {cell:6s} {tname:12s} 显著 {nsig:2d}/26 "
                  f"(离群 {sig_o}/13 对照 {sig_c}/13)  gap 中位 {np.median(gaps):+.5f}",
                  flush=True)
        out[cell] = cell_out
        print(flush=True)
    dest = Path("results/matrix/mechanism_null.json")
    dest.write_text(json.dumps(out, indent=1))
    print(f"[mech] 写出 {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
