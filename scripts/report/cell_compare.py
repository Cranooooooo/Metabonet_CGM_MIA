import json, csv, numpy as np, pandas as pd
from scipy import stats
m = json.load(open('results/matrix/report/subject_ids.json'))
G = {}
for cell in ('d1_c1','d1_c2'):
    g = pd.read_parquet(f'results/matrix/attack/{cell}/gaps.parquet')
    f = g[(g.set_reduce=='min') & (g.subject_reduce=='mean')]
    G[cell] = dict(zip(f.target, f.gap)); GRP = dict(zip(f.target, f.group))
t = sorted(G['d1_c1'], key=lambda x: -G['d1_c1'][x])
a = np.array([G['d1_c1'][x] for x in t]); b = np.array([G['d1_c2'][x] for x in t])
rho, p = stats.spearmanr(a, b)
print(f"两个 cell 的 gap 排名相关: Spearman rho = {rho:+.3f}, p = {p:.2e}  (n=26)")
print(f"  两 cell 各自 gap 最高的 6 人:")
print(f"    d1_c1: {[m[x] for x in sorted(G['d1_c1'], key=lambda x:-G['d1_c1'][x])[:6]]}")
print(f"    d1_c2: {[m[x] for x in sorted(G['d1_c2'], key=lambda x:-G['d1_c2'][x])[:6]]}")
print(f"\n逐人 AUC(subject_auc.py)")
print(f"{'编号':>7}{'组':>9}{'d1_c1':>8}{'d1_c2':>8}{'窗口':>6}")
S = {}
for cell in ('d1_c1','d1_c2'):
    S[cell] = {r['target']: r for r in
               csv.DictReader(open(f'results/matrix/subject_auc/{cell}/per_subject.csv'))}
for x in sorted(S['d1_c1'], key=lambda k: -float(S['d1_c2'][k]['auc'])):
    r1, r2 = S['d1_c1'][x], S['d1_c2'][x]
    print(f"{m[x]:>7}{r1['group']:>9}{float(r1['auc']):>8.3f}{float(r2['auc']):>8.3f}"
          f"{float(r1['n_windows']):>6.0f}")
for cell in ('d1_c1','d1_c2'):
    o=[float(v['auc']) for v in S[cell].values() if v['group']=='outlier']
    c=[float(v['auc']) for v in S[cell].values() if v['group']=='control']
    print(f"  {cell}: outlier 中位 {np.median(o):.3f}  normal 中位 {np.median(c):.3f}"
          f"   >0.55: {sum(x>.55 for x in o)}/13 vs {sum(x>.55 for x in c)}/13")
