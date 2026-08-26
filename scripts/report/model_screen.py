"""Flag models that deviate from the other 26 of their own cell, on all ten metrics.

WHY WITHIN-CELL. There is no absolute "good enough" for context-FID or FDDS, and the
three cells sit at different levels anyway (context_fid 0.095 / 0.160 / 0.331 on base).
What is meaningful is whether ONE model is unlike its 26 siblings, which were trained on
the same data, the same architecture and the same budget, differing only by which single
subject was added. Under that construction the 27 metric values should be one cluster;
anything outside it is a training failure, not a property of the subject.

Robust z = (x - median) / (1.4826 * MAD), so a handful of bad models cannot inflate the
scale and hide themselves. Two structurally-zero metrics are dropped for single-channel
cells: `correlational` and `fdds` measure cross-channel dependence, and with C=1 they are
0.0000 for every model -- a MAD of zero, which would make every z infinite.
"""
import json, sys
import numpy as np

KEYS = ["context_fid","correlational","vds","fdds","discriminative","predictive",
        "mdd","acd","skewness_diff","kurtosis_diff"]
ids = json.load(open('results/matrix/report/subject_ids.json'))
Z = 3.5

for cell in ("d1_c1","d1_c2","d7_c1"):
    try:
        d = json.load(open(f'results/matrix/quality_tsgem/{cell}_allmodels.json'))
    except FileNotFoundError:
        print(f"{cell}: not scored yet"); continue
    names = sorted(d, key=lambda k: d[k]["job"])
    jobs = [d[k]["job"] for k in names]
    print(f"\n===== {cell}   {len(names)} models")
    flags = {j: [] for j in jobs}
    for key in KEYS:
        v = np.array([d[k].get(key, np.nan) for k in names], float)
        if not np.isfinite(v).all():
            continue
        mad = np.median(np.abs(v - np.median(v)))
        if mad == 0:
            print(f"  {key:16} MAD=0 (structurally constant here) -- skipped")
            continue
        z = (v - np.median(v)) / (1.4826 * mad)
        for j, zz in zip(jobs, z):
            if zz > Z:
                flags[j].append(f"{key} z={zz:+.1f}")
    dq = json.load(open(f'results/matrix/quality/{cell}/disc_stability.json'))
    dmax = {k.split('/')[-2]: v['max'] for k, v in dq.items()}
    bad = {j: f for j, f in flags.items() if f}
    print(f"  {'model':22}{'编号':>7}{'判别器max':>10}   十指标偏离 (robust z > {Z})")
    for j in sorted(bad, key=lambda x: -len(bad[x])):
        t = j.replace('include_','').replace('__','/')
        num = ids.get(t, '-')
        print(f"  {j:22}{str(num):>7}{dmax.get(j, float('nan')):>10.3f}   {'; '.join(bad[j])}")
    if not bad:
        print("  none deviate")
    dbad = sorted(k for k, v in dmax.items() if v > 0.75)
    print(f"  判别器 max>0.75 认定的坏模型: {dbad or '无'}")
    print(f"  十指标认定的坏模型          : {sorted(bad) or '无'}")
