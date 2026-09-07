#!/usr/bin/env python
"""分配器的证伪:「孤立度」预不预测「谁真的泄漏」?

要证伪什么
----------
设计里的分配器按每个训练窗口的**孤立度**分配保护力度,前提是「孤立 == 更容易被认出」。
这个脚本判它的死活,不用训练、不用显卡。

第一版有三个会让结论直接是错的缺陷,审查抓出来了,这里逐条修掉 ——
每一条都写在下面,因为它们都是容易再犯的:

1. **参照池不能包含这个人自己的窗口。** 第一版拿 include_t 的整个训练集当池子,
   而它含这个人自己的记录。k 固定为 5,但 26 人里有 12 人的记录数 ≤ 5,
   于是「第 5 近邻」对短记录的人来说必然落到别人身上、对长记录的人落到自己身上 ——
   **孤立度变成了记录条数的代理**。审查的模拟:这样算出来的孤立度和记录条数的相关是
   -0.909,而记录条数和可识别性本身有 -0.16 的相关,两者一乘,**能从纯噪声里造出
   rho ≈ +0.39,而 n=26 的显著门槛正好是 0.387**。也就是第一版可能报出一个
   「显著支持设计」的假结果。项目里 outliers/shape.py 早就解决过同一个问题
   (它把自己的列置为 inf),这里照它做。

2. **分配器的单位是窗口,不是人。** 第一版把逐窗口的孤立度按人平均掉再做 26 个点的
   相关 —— 人层面接近零和窗口层面很强可以同时成立。所以**主分析改成逐窗口**:
   窗口的孤立度 对 窗口的泄漏量(d_out - d_in),按人分层。人层面的相关降为次要结果。

3. **n=26 的负结果需要功效说明。** |rho| 要超过 0.387 才到 p<0.05,而 80% 功效需要
   真值 |rho| ≈ 0.526。所以「点估计接近零」根本不能证伪 —— 一个 0.15 完全兼容真值 0.5。
   本脚本给出 Fisher-z 置信区间,并**预先声明判据**:只有当置信区间**上界**低于
   分配器实际需要的最小相关(默认 0.3)时,才算证伪。这条是项目自己的教训:
   「只有一个阴性对照什么也证明不了」。

另外三条修正
------------
· 用 scipy 的 spearmanr(带并列校正)。第一版用双 argsort 的序数秩,在有并列时
  结果依赖输入顺序 —— 实测同一份数据换个行序,相关系数动了 0.047。
· 加**偏相关**:扣掉记录条数之后孤立度还剩多少解释力。这才是该看的主数字。
· `raw` 这个空间和离群者的**挑选标准部分重合** —— 当初选离群者用的 13 个方法里,
  B7a/B7b 就是「原始波形的 k 近邻距离(均值/最大)」,同样的 k=5、同样的 /sqrt(F)。
  所以 raw 上的正相关有一部分是在重新测量挑选标准本身。脚本按组分别报一次。

空间的命名也改了:第一版把「按小时平均」叫成「水平坐标」,那是错的 ——
它保留了粗时序,不是纯水平。

    python scripts/isolation_vs_leak.py --design results/matrix/design/rep1 \
        --cohort data/cohort/matrix_d1_c1 --runs results/runs/matrix_d1_c1 \
        --auc results/matrix/subject_auc/d1_c1 --out results/matrix/isolation/d1_c1.json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cgmoutlier.data.cohort import load as load_cohort            # noqa: E402
from cgmoutlier.loo import training_set                           # noqa: E402
from cgmoutlier.attack.statistic import window_distances, _match  # noqa: E402


def _flat(A):
    A = np.asarray(A, dtype=np.float32)
    return A.reshape(A.shape[0], -1) if A.ndim > 2 else A


def _hourly(X):
    T = X.shape[1]
    k = T // 12
    return X[:, :k * 12].reshape(len(X), k, 12, X.shape[2]).mean(2)


SPACES = {
    "raw":      ("原始波形(和离群者挑选标准部分重合)", lambda X: X),
    "coarse":   ("按小时平均的粗波形",                 _hourly),
    "shape":    ("去掉水平和幅度后的形状",             lambda X: (X - X.mean(1, keepdims=True))
                                                       / (X.std(1, keepdims=True) + 1e-6)),
    "detail":   ("高频残差(阴性对照)",                 lambda X: X - np.repeat(
                     _hourly(X), 12, axis=1)[:, :X.shape[1]]),
}


def isolation(Q, POOL, pool_sids, target, k, chunk=256):
    """Q 每行到 POOL 中第 k 近邻的距离,**排除这个人自己的所有窗口**。

    自排除是关键(见模块说明第 1 条):不排除的话,记录条数少的人会因为
    「第 k 近邻只能落到别人身上」而被判成孤立,孤立度就退化成记录条数的代理。
    做法和 src/cgmoutlier/outliers/shape.py 一致:把自己的列置为 inf。
    """
    Qf, Pf = _flat(Q), _flat(POOL)
    F = Qf.shape[1]
    own = (np.asarray(pool_sids) == target)
    n_other = int((~own).sum())
    if n_other < k:
        raise ValueError(f"{target}: 池子里只有 {n_other} 个他人窗口,不足 k={k}")
    out = np.empty(len(Qf), np.float32)
    p2 = (Pf ** 2).sum(1)
    for a in range(0, len(Qf), chunk):
        b = min(a + chunk, len(Qf))
        d2 = (Qf[a:b] ** 2).sum(1)[:, None] - 2.0 * (Qf[a:b] @ Pf.T) + p2[None, :]
        np.maximum(d2, 0.0, out=d2)
        d2[:, own] = np.inf
        out[a:b] = np.sqrt(np.partition(d2, k - 1, axis=1)[:, k - 1]) / np.sqrt(F)
    return out


def rho_ci(x, y):
    """带并列校正的 Spearman,外加 Fisher-z 95% 置信区间。"""
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x)
    r, p = stats.spearmanr(x, y)
    if n < 4 or not np.isfinite(r):
        return float(r), float(p), float("nan"), float("nan"), n
    z = np.arctanh(np.clip(r, -0.999999, 0.999999))
    se = 1.0 / np.sqrt(n - 3)
    return float(r), float(p), float(np.tanh(z - 1.96 * se)), float(np.tanh(z + 1.96 * se)), n


def partial_rho(x, y, z):
    """扣掉 z 之后 x 与 y 的偏相关:先把三者都换成秩,再对 z 做线性残差化。"""
    rx, ry, rz = (stats.rankdata(v).astype(float) for v in (x, y, z))
    def resid(a):
        A = np.c_[np.ones(len(rz)), rz]
        return a - A @ np.linalg.lstsq(A, a, rcond=None)[0]
    r, p = stats.pearsonr(resid(rx), resid(ry))
    return float(r), float(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", required=True)
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--runs", required=True, help="results/runs/matrix_<cell>")
    ap.add_argument("--auc", required=True, help="results/matrix/subject_auc/<cell>")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--need", type=float, default=0.30,
                    help="分配器实际需要的最小相关;证伪判据是置信区间上界低于它")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    auc = {r["target"]: (float(r["auc"]), r["group"], int(r["n_windows"]))
           for r in csv.DictReader(open(Path(a.auc) / "per_subject.csv"))}
    X, sids, _ = load_cohort(a.cohort)
    X = np.asarray(X); sids = np.asarray([str(s) for s in sids])
    S_base = np.asarray(np.load(Path(a.runs) / "base" / "samples.npy", mmap_mode="r"),
                        np.float32)

    per_subj, win = {}, {sp: dict(iso=[], leak=[], subj=[]) for sp in SPACES}
    for name in sorted(p.stem for p in (Path(a.design) / "jobs").glob("include_*.json")):
        j = json.loads((Path(a.design) / "jobs" / f"{name}.json").read_text())
        t = str(j["target"])
        f = Path(a.runs) / name / "samples.npy"
        if t not in auc or not f.exists():
            print(f"[iso] 跳过 {t}(缺 AUC 或样本)", flush=True)
            continue
        POOL, pool_sids = training_set(X, sids, j["subjects"])
        R = np.ascontiguousarray(X[sids == t], np.float32)
        # 逐窗口的泄漏量:和攻击同一套距离,d_out - d_in,正值 = 这个窗口被记住了
        S_in, S_out, _ = _match(np.asarray(np.load(str(f)), np.float32), S_base,
                                t, a.seed, True)
        leak = (window_distances(R, S_out, set_reduce="min")
                - window_distances(R, S_in, set_reduce="min"))
        rec = {}
        for sp, (_d, fn) in SPACES.items():
            iso = isolation(fn(R), fn(POOL), pool_sids, t, a.k)
            rec[sp] = float(iso.mean())
            win[sp]["iso"].extend(iso.tolist())
            win[sp]["leak"].extend(leak.tolist())
            win[sp]["subj"].extend([t] * len(iso))
        per_subj[t] = dict(group=auc[t][1], auc=auc[t][0], n_windows=auc[t][2],
                           leak_mean=float(leak.mean()), **rec)
        print(f"[iso] {t:14s} {auc[t][1]:8s} auc={auc[t][0]:.3f} "
              f"leak={leak.mean():+.5f}  " +
              "  ".join(f"{sp}={rec[sp]:.4f}" for sp in SPACES), flush=True)

    n_sub = len(per_subj)
    n_win = len(win["raw"]["iso"])
    print(f"\n受试者 {n_sub} 人,窗口 {n_win} 段。k={a.k},自排除。")

    print(f"\n=== 主分析:逐窗口。孤立度 vs 该窗口的泄漏量 ===")
    print(f"  {'空间':10s}{'含义':34s}{'rho':>8s}{'95% 区间':>18s}{'p':>10s}")
    summary = {}
    for sp, (desc, _) in SPACES.items():
        r, p, lo, hi, n = rho_ci(win[sp]["iso"], win[sp]["leak"])
        summary[f"window_{sp}"] = dict(rho=r, p=p, ci=[lo, hi], n=n)
        print(f"  {sp:10s}{desc:34s}{r:>+8.3f}   [{lo:+.3f}, {hi:+.3f}]{p:>10.2e}")

    print(f"\n=== 次要:人层面(n={n_sub})。孤立度均值 vs 可识别性 ===")
    print("  同时报扣掉记录条数之后的偏相关 —— 那才是该看的数。")
    print(f"  {'空间':10s}{'rho':>8s}{'95% 区间':>18s}{'p':>9s}{'偏相关':>9s}{'偏 p':>9s}")
    A = [per_subj[t]["auc"] for t in per_subj]
    NW = [per_subj[t]["n_windows"] for t in per_subj]
    for sp in SPACES:
        v = [per_subj[t][sp] for t in per_subj]
        r, p, lo, hi, n = rho_ci(v, A)
        pr, pp = partial_rho(v, A, NW)
        summary[f"subject_{sp}"] = dict(rho=r, p=p, ci=[lo, hi], partial=pr, partial_p=pp)
        print(f"  {sp:10s}{r:>+8.3f}   [{lo:+.3f}, {hi:+.3f}]{p:>9.3f}{pr:>+9.3f}{pp:>9.3f}")

    r_nw, p_nw, *_ = rho_ci(NW, A)
    print(f"\n  混淆:记录条数 vs 可识别性 rho={r_nw:+.3f} (p={p_nw:.3f})")
    for g in ("outlier", "control"):
        idx = [t for t in per_subj if per_subj[t]["group"] == g]
        if len(idx) >= 4:
            rr, pp2, *_ = rho_ci([per_subj[t]["raw"] for t in idx],
                                 [per_subj[t]["auc"] for t in idx])
            print(f"  组内({g}, n={len(idx)}): raw 孤立度 vs 可识别性 rho={rr:+.3f} (p={pp2:.3f})")

    print(f"\n=== 判据(事先声明) ===")
    print(f"  分配器需要至少 |rho| = {a.need:.2f} 才有实用价值。")
    prim = summary["window_raw"]
    print(f"  主分析(逐窗口 raw): rho={prim['rho']:+.3f},区间 [{prim['ci'][0]:+.3f}, {prim['ci'][1]:+.3f}]")
    if prim["ci"][1] < a.need:
        print(f"  → **证伪**:区间上界 {prim['ci'][1]:+.3f} 低于 {a.need:.2f},分配器不成立。")
    elif prim["ci"][0] > a.need:
        print(f"  → **支持**:区间下界 {prim['ci'][0]:+.3f} 高于 {a.need:.2f}。")
    else:
        print(f"  → **不确定**:区间跨过 {a.need:.2f},这个样本量判不了。不要当成任何一方的证据。")
    print("  提醒:raw 这个空间和离群者的挑选标准部分重合,以组内那两行为准。")

    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(dict(cohort=a.cohort, design=a.design, runs=a.runs, auc=a.auc,
                       k=a.k, need=a.need, n_subjects=n_sub, n_windows=n_win,
                       summary=summary, per_subject=per_subj), open(a.out, "w"), indent=2)
        print(f"\n写入 {a.out}")


if __name__ == "__main__":
    main()
