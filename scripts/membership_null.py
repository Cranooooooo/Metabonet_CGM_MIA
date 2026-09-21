#!/usr/bin/env python
"""一个真正的成员推断零假设,以及一个不依赖单一 base 的统计量。

为什么需要这个
--------------
`scripts/leak_locus.py` 里的 `_floor` 被当作「地板对照」用了一整篇论文,但它不是。
它做的是 `rng.shuffle(Y[i, :, c])` —— 沿【时间轴】打乱窗口内部的值,于是每个窗口的
数值多重集【逐值保留】。而本项目自己测过,CGM 的可识别性几乎全部住在
level / distribution 空间(`shape` 是唯一对离群点无效的那个)。所以这个「地板」
恰好保住了承载身份的那一半信息。

实证三条,都在盘上:
  * igfm_priv_d1_c1: raw 0.5203305785 vs shuffle 0.5202040816,小数点后四位相同
  * Loop/1142 @ d1_c1: raw 0.941 / sorted 0.948 / shuffle 0.950 —— 地板比原样还高
  * Loop/1142 @ d7_c1: raw / sorted / shuffle 全是 1.000 —— 在「摧毁一切身份」的
    对照下被完美识别

所以「N 个病人超过地板 F」这句话,比的是两个从未被排序过的数。

这个脚本换两样东西进去
----------------------
1. 【真正的零假设】。对称设计里 base 训练时排除了全部 26 个目标,所以对目标 t 而言,
   每一个 include_j(j≠t)也都是合法的【非成员】模型 —— 它见过 j,没见过 t。
   于是 `mean d(R_t, S_base) - mean d(R_t, S_include_j)` 就是同一个统计量在零假设
   下的一次抽样,盘上现成有 25 个。观测值落在这 25 个里的位置,就是这个受试者的
   经验 p 值。不需要任何新训练。

2. 【不依赖单一 base 的统计量】。26 对共用一个 base,所以 base 自己那次抽样的偏移
   以同样的符号进入每一项 —— 论文里那个「gap 中位数」在很大程度上是在估计 base 的
   个体偏移(docs/PITFALLS.md §21 讲的就是这个机制)。把参照换成
   `mean_{j≠t} d(R_t, S_include_j)`,这个偏移【从结构上】消失,因为参照不再是一个模型。

统计量与已发表的口径一致:set_reduce="min"、subject_reduce="mean",即冻结的那个变体。

    qsub scripts/pbs/dev/membership_null.pbs
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

from cgmoutlier._env import check as _envcheck                       # noqa: E402
_envcheck()
from cgmoutlier.data.cohort import load as load_cohort               # noqa: E402
from cgmoutlier.attack.statistic import window_distances, _match     # noqa: E402

CELLS = {
    # 三个格子 x 每个格子有的全部 arm。键名 = "<格子>/<arm>"。
    "d1_c1/ours":         "results/runs/loo_igfm_priv_d1_c1",
    "d1_c1/backbone":     "results/runs/loo_igfm_d1_c1",
    "d1_c1/dimts":        "results/runs/matrix_d1_c1",
    "d1_c1/copy_paste":   "results/runs/cp_matrix_d1_c1",
    "d1_c1/diffwave":     "results/runs/bl_diffwave_d1_c1",
    "d1_c1/fourierdiff":  "results/runs/bl_fourier_diff_d1_c1",
    "d1_c1/diffusionts":  "results/runs/bl_diffusion_ts_d1_c1",
    "d1_c2/ours":         "results/runs/loo_igfm_priv_d1_c2",
    "d1_c2/backbone":     "results/runs/loo_igfm_d1_c2",
    "d1_c2/dimts":        "results/runs/matrix_d1_c2",
    "d1_c2/copy_paste":   "results/runs/cp_matrix_d1_c2",
    "d1_c2/diffwave":     "results/runs/bl_diffwave_d1_c2",
    "d1_c2/fourierdiff":  "results/runs/bl_fourier_diff_d1_c2",
    "d1_c2/diffusionts":  "results/runs/bl_diffusion_ts_d1_c2",
    "d7_c1/ours":         "results/runs/loo_igfm_priv_d7_c1",
    "d7_c1/dimts":        "results/runs/matrix_d7_c1",
    "d7_c1/copy_paste":   "results/runs/cp_matrix_d7_c1",
}
DESIGN = Path("results/matrix/design/rep1")
SEED = 2026


def cohort_of(cell):
    return "data/cohort/matrix_" + cell.split("/")[0]


def main() -> int:
    out_all = {}
    for cell, tree in CELLS.items():
        tree = Path(tree)
        if not (tree / "base" / "samples.npy").exists():
            print(f"[null] {cell}: 没有 {tree}/base,跳过", flush=True); continue
        design = json.loads((DESIGN / "design.json").read_text())
        pairs = design["pairs"]
        X, sids, _ = load_cohort(cohort_of(cell))
        sids = np.asarray([str(s) for s in sids])

        names = ["base"] + [p["member"] for p in pairs]
        paths = {n: tree / n / "samples.npy" for n in names}
        missing = [n for n in names if not paths[n].exists()]
        if missing:
            print(f"[null] {cell}: 缺 {len(missing)} 个模型,跳过", flush=True); continue

        S = {n: np.load(paths[n], mmap_mode="r") for n in names}
        kmin = min(len(v) for v in S.values())
        print(f"[null] {cell}: {len(names)} 个释放,统一裁到 K={kmin:,}", flush=True)

        rows = []
        for i, p in enumerate(pairs, 1):
            t, grp, mem = str(p["target"]), p["group"], p["member"]
            R = np.ascontiguousarray(np.asarray(X)[sids == t], np.float32)
            # 所有释放用【同一个按目标播种的 RNG】裁到同样大小,彼此可比
            d = {}
            for n in names:
                Sn, _, _ = _match(np.asarray(S[n], np.float32),
                                  np.empty((kmin,) + S[n].shape[1:], np.float32),
                                  t, SEED, True)
                d[n] = window_distances(R, Sn, set_reduce="min")
            dm = {n: float(v.mean()) for n, v in d.items()}

            others = [p2["member"] for p2 in pairs if p2["member"] != mem]
            obs = dm["base"] - dm[mem]                       # 已发表口径
            null = np.array([dm["base"] - dm[o] for o in others])   # 25 个真零假设抽样
            loo_ref = float(np.mean([dm[o] for o in others]))
            loo_gap = loo_ref - dm[mem]                      # 不依赖 base
            base_off = dm["base"] - loo_ref                  # base 自己的偏移
            # 经验 p:观测值比多少个零假设抽样更极端(单侧,gap 越大越像成员)
            p_emp = float((null >= obs).sum() + 1) / (len(null) + 1)
            z = float((obs - null.mean()) / (null.std(ddof=1) + 1e-12))
            rows.append(dict(target=t, group=grp, n_windows=int(len(R)),
                             obs_gap=obs, null_mean=float(null.mean()),
                             null_sd=float(null.std(ddof=1)), z=z, p_emp=p_emp,
                             loo_gap=loo_gap, base_offset=base_off))
            print(f"[null] {cell} {i:2d}/26 {t:14s} {grp:8s} "
                  f"obs={obs:+.5f} null={null.mean():+.5f}±{null.std(ddof=1):.5f} "
                  f"z={z:+.2f} p={p_emp:.3f} loo={loo_gap:+.5f} base_off={base_off:+.5f}",
                  flush=True)

        out_all[cell] = rows
        A = np.array([r["obs_gap"] for r in rows])
        B = np.array([r["base_offset"] for r in rows])
        L = np.array([r["loo_gap"] for r in rows])
        og = [r for r in rows if r["group"] == "outlier"]
        cg = [r for r in rows if r["group"] == "control"]
        print(f"\n[null] === {cell} 小结 ===")
        print(f"  base 偏移: 中位 {np.median(B):+.5f}  26 个里 {int((B<0).sum())} 个为负")
        print(f"     (系统性为负 = base 释放整体更贴近真实数据,即共享 base 的偏移)")
        print(f"  已发表口径 gap 中位: {np.median(A):+.5f}")
        print(f"  去掉 base 之后   : {np.median(L):+.5f}")
        print(f"  经验 p<0.05 的受试者: {sum(1 for r in rows if r['p_emp']<0.05)}/26 "
              f"(离群 {sum(1 for r in og if r['p_emp']<0.05)}/13, "
              f"对照 {sum(1 for r in cg if r['p_emp']<0.05)}/13)")
        if og and cg:
            u = stats.mannwhitneyu([r["loo_gap"] for r in og],
                                   [r["loo_gap"] for r in cg], alternative="two-sided")
            print(f"  两臂对比(用不依赖 base 的 loo_gap): AUC "
                  f"{u.statistic/(len(og)*len(cg)):.3f}  p={u.pvalue:.3f}")
        print(flush=True)

    dest = Path("results/matrix/membership_null.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out_all, indent=1))
    print(f"[null] 写出 {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
