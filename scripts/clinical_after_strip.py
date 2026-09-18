#!/usr/bin/env python
"""临床血糖指标:真实 vs 释放 vs 剥离水平后的释放。

为什么要有这个
--------------
论文 R6 提了一个释放时的缓解手段:把窗口的整体水平拿掉,再随机重配一个。它的
【隐私侧】已经由 leak_locus 的 level / reoffset 两个变体测了,但【临床代价】没有。
不测就写"shape-based 指标得以保留"是没有依据的 —— 而且其中一部分指标按定义就
不可能保留(平均血糖、TIR、%T<54 全部定义在数值刻度上)。这个脚本把两边都给出来。

同时修掉会议(2026-09-01)上提的那条:临床指标表不能只报一个总体均值 delta。
> "if you express the percentage per person, per data set, and then have a
>  distribution ... a slight delta with a very tight distribution is probably better"
所以这里每个指标报【逐窗口分布】的中位数与四分位距,并对真实分布给出两个分布
距离(KS 统计量、Wasserstein),而不是 mean±sd。

变换在 mg/dL 空间做。减去窗口均值和反归一化的仿射映射可交换,所以这和 leak_locus
在归一化空间里做的是同一个变换,只是读数单位不同。

    qsub scripts/pbs/dev/clinical_strip.pbs
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

from cgmoutlier._env import check as _envcheck                      # noqa: E402
_envcheck()
from cgmoutlier.data.cohort import load as load_cohort, channel_raw  # noqa: E402
import cgm_clinical_metrics as ccm                                   # noqa: E402

CELLS = ("d1_c1", "d1_c2")
ARMS = {                       # 展示名 -> 释放目录
    "ours":        "results/runs/loo_igfm_priv_{cell}/base",
    "dimts":       "results/runs/matrix_{cell}/base",
    "copy_paste":  "results/runs/cp_matrix_{cell}/base",
}


def _strip(G):
    """只剥离水平。"""
    return G - G.mean(1, keepdims=True)


def _reoffset(G, seed=20260901):
    """剥离水平,再从同一批的水平分布里置换重配。与 leak_locus._reoffset 同义。"""
    rng = np.random.default_rng(seed)
    mu = G.mean(1, keepdims=True)
    return G - mu + mu[rng.permutation(len(G))]


def _describe(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return None
    q1, med, q3 = np.percentile(v, [25, 50, 75])
    return {"n": int(v.size), "median": float(med), "q1": float(q1), "q3": float(q3),
            "iqr": float(q3 - q1), "mean": float(v.mean()), "sd": float(v.std(ddof=1))}


def _distance(v, ref):
    """对真实分布的两个距离。KS 对形状敏感,Wasserstein 有量纲、可读成 mg/dL。"""
    v = np.asarray(v, float); ref = np.asarray(ref, float)
    v = v[np.isfinite(v)]; ref = ref[np.isfinite(ref)]
    if v.size == 0 or ref.size == 0:
        return None
    ks = stats.ks_2samp(v, ref)
    return {"ks": float(ks.statistic), "ks_p": float(ks.pvalue),
            "wasserstein": float(stats.wasserstein_distance(v, ref))}


def main() -> int:
    out = {}
    for cell in CELLS:
        coh = f"data/cohort/matrix_{cell}"
        X, _sids, man = load_cohort(coh)
        real = np.asarray(channel_raw(X, man, "CGM"), np.float64)
        if real.ndim == 3:
            real = real[..., 0]
        print(f"[clin] {cell}: 真实 {real.shape}", flush=True)

        ref = {n: np.asarray(v, float) for n, _g, _d, v in ccm.full_battery(real)}
        cellout = {"real": {n: _describe(v) for n, v in ref.items()}}

        for arm, tmpl in ARMS.items():
            p = Path(tmpl.format(cell=cell)) / "samples.npy"
            if not p.exists():
                print(f"[clin] {cell}/{arm}: 缺 {p},跳过", flush=True)
                continue
            S = np.asarray(np.load(p), np.float32)
            G = np.asarray(channel_raw(S, man, "CGM"), np.float64)
            if G.ndim == 3:
                G = G[..., 0]
            for variant, fn in (("released", lambda A: A),
                                ("level_stripped", _strip),
                                ("reoffset", _reoffset)):
                Gv = fn(G)
                rec = {}
                for n, _g, _d, v in ccm.full_battery(Gv):
                    rec[n] = {"dist": _describe(v), "vs_real": _distance(v, ref[n])}
                cellout[f"{arm}/{variant}"] = rec
                ok = sum(1 for n in rec if rec[n]["vs_real"]
                         and rec[n]["vs_real"]["ks"] < 0.1)
                print(f"[clin] {cell}/{arm}/{variant}: {len(rec)} 个指标,"
                      f"其中 {ok} 个 KS<0.1", flush=True)
        out[cell] = cellout

    dest = Path("results/clinical/after_strip.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(f"\n[clin] 写出 {dest}")

    print("\n=== 剥离水平的临床代价(d1_c1, ours):KS 对真实分布 ===")
    c = out.get("d1_c1", {})
    if "ours/released" in c:
        print(f"  {'指标':14s}{'released':>10s}{'stripped':>10s}{'reoffset':>10s}")
        for n in c["ours/released"]:
            row = []
            for v in ("released", "level_stripped", "reoffset"):
                d = c.get(f"ours/{v}", {}).get(n, {}).get("vs_real")
                row.append(f"{d['ks']:.3f}" if d else "   —")
            print(f"  {n:14s}" + "".join(f"{x:>10s}" for x in row))
        print("\n  KS 越小越像真实。定义在数值刻度上的指标(平均血糖/TIR/%T<54)")
        print("  在剥离之后必然崩掉 —— 那不是 bug,是这个手段的代价,要如实报。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
