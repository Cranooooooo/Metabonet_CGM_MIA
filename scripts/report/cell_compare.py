import json, numpy as np, pandas as pd
from scipy import stats
m = json.load(open('results/matrix/report/subject_ids.json'))
for cell in ('d1_c1','d1_c2','d7_c1'):
    q = json.load(open(f'results/matrix/quality/{cell}/disc_stability.json'))
    susp = {k.split('/')[-2].replace('include_','').replace('__','/')
            for k,v in q.items() if v['max'] > 0.75}
    g = pd.read_parquet(f'results/matrix/attack/{cell}/gaps.parquet')
    f = g[(g.set_reduce=='min') & (g.subject_reduce=='mean')]
    line = f"{cell}: "
    for label, keep in (("全部", f), ("剔除可疑", f[~f.target.isin(susp)])):
        o = keep[keep.group=='outlier'].gap.values; c = keep[keep.group=='control'].gap.values
        u = stats.mannwhitneyu(o, c, alternative='greater')
        line += f"  {label} n={len(o)}v{len(c)} AUC {u.statistic/(len(o)*len(c)):.4f} p {u.pvalue:.4f}"
    print(line + f"   可疑: {sorted(m[s] for s in susp) if susp else '无'}")
