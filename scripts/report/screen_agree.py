"""Do the two gates disagree, or does one just sit below its threshold?"""
import json, numpy as np
KEYS = ["context_fid","correlational","vds","fdds","discriminative","predictive",
        "mdd","acd","skewness_diff","kurtosis_diff"]
ids = json.load(open('results/matrix/report/subject_ids.json'))
for cell in ("d1_c1","d1_c2","d7_c1"):
    d = json.load(open(f'results/matrix/quality_tsgem/{cell}_allmodels.json'))
    q = json.load(open(f'results/matrix/quality/{cell}/disc_stability.json'))
    dmax = {k.split('/')[-2]: v['max'] for k,v in q.items()}
    names = sorted(d, key=lambda k: d[k]["job"]); jobs = [d[k]["job"] for k in names]
    Zs = {}
    for key in KEYS:
        v = np.array([d[k].get(key, np.nan) for k in names], float)
        if not np.isfinite(v).all(): continue
        mad = np.median(np.abs(v - np.median(v)))
        if mad == 0: continue
        z = (v - np.median(v)) / (1.4826 * mad)
        for j, zz in zip(jobs, z): Zs.setdefault(j, {})[key] = zz
    flagged = sorted(k for k,v in dmax.items() if v > 0.75)
    print(f"\n===== {cell}")
    print(f"  判别器点名的模型,它们的十指标 max|z|:")
    for j in flagged:
        z = Zs.get(j, {})
        top = sorted(z.items(), key=lambda x: -abs(x[1]))[:3]
        print(f"    {j:22} 判别器max {dmax[j]:.3f}   "
              + "  ".join(f"{k}={v:+.1f}" for k,v in top))
    if not flagged: print("    (无)")
    allz = np.array([max(abs(x) for x in Zs[j].values()) for j in Zs])
    print(f"  全格 max|z| 分布: 中位 {np.median(allz):.1f}  90分位 {np.percentile(allz,90):.1f}  最大 {allz.max():.1f}")
    # 判别器 max 和十指标 max|z| 相关吗
    j2 = [j for j in Zs if j in dmax]
    a = np.array([dmax[j] for j in j2]); b = np.array([max(abs(x) for x in Zs[j].values()) for j in j2])
    from scipy import stats
    r, p = stats.spearmanr(a, b)
    print(f"  判别器max vs 十指标max|z| 的 Spearman: rho={r:+.3f}, p={p:.3f}  (n={len(j2)})")
