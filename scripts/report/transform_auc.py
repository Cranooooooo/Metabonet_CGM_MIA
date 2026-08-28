"""Arm AUC inside each transform space.

The raw gap difference is NOT comparable across transforms: each transform changes the
space the distance lives in, so 0.053 in z-scored space and 0.024 in raw space are
different units. The arm AUC is unitless -- the probability a random outlier's gap
exceeds a random control's -- and is the quantity that can be read down the column.
"""
import json, glob, os, numpy as np
from scipy import stats
KEYS = ["raw","diff","sorted","hourly","zscore"]
for cell in ('d1_c1','d1_c2','d7_c1'):
    p = f'results/matrix/localise/{cell}/per_transform.json'
    if not os.path.exists(p): print(f"\n  {cell}: not yet"); continue
    d = json.load(open(p))
    print(f"\n  ===== {cell}")
    print(f"  {'transform':10}{'arm AUC':>10}{'p':>9}{'vs raw':>9}   {'destroys':<28}")
    what = {"raw":"nothing","diff":"absolute level","sorted":"all timing",
            "hourly":"detail below one hour","zscore":"level and scale"}
    base = None
    for k in KEYS:
        o = np.array([v[k] for v in d.values() if v['group']=='outlier'])
        c = np.array([v[k] for v in d.values() if v['group']!='outlier'])
        u = stats.mannwhitneyu(o, c, alternative='greater')
        auc = u.statistic/(len(o)*len(c))
        if k == 'raw': base = auc
        rel = (auc-0.5)/(base-0.5) if base and base != 0.5 else float('nan')
        print(f"  {k:10}{auc:>10.3f}{u.pvalue:>9.3f}{rel:>9.2f}   {what[k]:<28}")
