#!/usr/bin/env python
"""排序变换到底动了哪一臂:两臂各自的 gap,不只是它们的可分开程度。"""
import json
import numpy as np
def mw(a,b):
    a,b=np.asarray(a,float),np.asarray(b,float)
    return float(((a[:,None]>b[None,:]).sum()+0.5*(a[:,None]==b[None,:]).sum())/(len(a)*len(b)))
T=("raw","diff","sorted","hourly","zscore")
print("=== 每个变换下,两臂各自的 gap 中位数(不是它们的可分开程度) ===")
print(f"{'cell':8s}{'变换':10s}{'离群臂 gap':>14s}{'普通臂 gap':>14s}{'两臂可分开度':>14s}")
for c in ("d1_c1","d1_c2","d7_c1"):
    j=json.load(open(f"results/matrix/localise/{c}/per_transform.json"))
    for t in T:
        o=[r[t] for r in j.values() if r["group"]=="outlier"]
        n=[r[t] for r in j.values() if r["group"]=="control"]
        print(f"{c:8s}{t:10s}{np.median(o):>14.5f}{np.median(n):>14.5f}{mw(o,n):>14.3f}")
    print()
print("=== 关键问题:排序之后离群臂的 gap 是变大了还是变小了 ===")
for c in ("d1_c1","d1_c2","d7_c1"):
    j=json.load(open(f"results/matrix/localise/{c}/per_transform.json"))
    o_raw=np.median([r["raw"] for r in j.values() if r["group"]=="outlier"])
    o_srt=np.median([r["sorted"] for r in j.values() if r["group"]=="outlier"])
    n_raw=np.median([r["raw"] for r in j.values() if r["group"]=="control"])
    n_srt=np.median([r["sorted"] for r in j.values() if r["group"]=="control"])
    print(f"  {c}: 离群臂 {o_raw:.5f} -> {o_srt:.5f} ({(o_srt/o_raw-1)*100:+.0f}%)   "
          f"普通臂 {n_raw:.5f} -> {n_srt:.5f} ({(n_srt/n_raw-1)*100:+.0f}%)")
print("\n  两臂都变小 = 排序主要是把「共同的噪声」削掉了,不是发现了更多泄漏。")
print("  只有离群臂变大 = 排序确实揭示了更多泄漏。")
