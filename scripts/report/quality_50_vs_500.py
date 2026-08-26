"""Quality is the gate. Do 50-step samples clear it?

The question is NOT whether the MIA answer moves -- that is downstream of quality and
meaningless if the samples are bad. It is whether a 50-step release is a good enough
generator to be worth attacking at all, and whether the RANKING between cells survives,
since an epoch sweep compares points against each other.
"""
import json, numpy as np
KEYS = ["context_fid","correlational","vds","fdds","discriminative","predictive",
        "mdd","acd","skewness_diff","kurtosis_diff"]
CELLS = ["d1_c1","d1_c2","d7_c1"]
def one(p):
    return list(json.load(open(p)).values())[0]
A = {c: one(f'results/matrix/quality_tsgem/{c}_st500.json') for c in CELLS}
B = {c: one(f'results/matrix/quality_tsgem/{c}_st50.json') for c in CELLS}
print("base 生成质量,10 指标(lower = better)")
print(f"  {'metric':16}" + "".join(f"{c+' 500':>13}{c+' 50':>12}" for c in CELLS))
for k in KEYS:
    row = f"  {k:16}"
    for c in CELLS:
        a, b = A[c].get(k), B[c].get(k)
        row += f"{a:>13.4f}{b:>12.4f}"
    print(row)
print("\n每个指标:50 步相对 500 步变差多少倍(>1 = 变差)")
print(f"  {'metric':16}" + "".join(f"{c:>10}" for c in CELLS))
for k in KEYS:
    row = f"  {k:16}"
    for c in CELLS:
        a, b = A[c].get(k), B[c].get(k)
        row += f"{(b/a if a and a > 1e-9 else float('nan')):>10.2f}"
    print(row)
print("\n跨格排序是否保住(扫描要靠它)")
for k in KEYS:
    a = [A[c].get(k, np.nan) for c in CELLS]; b = [B[c].get(k, np.nan) for c in CELLS]
    if not (np.isfinite(a).all() and np.isfinite(b).all()) or len(set(a)) < 3:
        print(f"  {k:16} 不可比(有并列或缺值)"); continue
    ra = [sorted(a).index(x) for x in a]; rb = [sorted(b).index(x) for x in b]
    print(f"  {k:16} 500步排序 {ra}   50步排序 {rb}   {'一致' if ra == rb else '翻转'}")
