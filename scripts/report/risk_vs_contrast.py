"""Risk and contrast are different quantities and they move differently."""
import json, csv, numpy as np
print(f"  {'steps':>8}{'epochs':>8} | {'RISK: 全体中位':>15}{'全体>0.55':>11}{'最高':>7}"
      f" | {'CONTRAST: arm AUC':>19}")
for m in (2,3,4,6,8,10):
    r=[x for x in json.load(open(f'results/matrix/sweep/attack/d1_c1_ms{m}/summary.json')) if x['frozen']][0]
    rows=list(csv.DictReader(open(f'results/matrix/sweep/subject_auc/d1_c1_ms{m}/per_subject.csv')))
    au=np.array([float(x['auc']) for x in rows])
    print(f"  {m*10000:>8}{round(m*10000*64/5741):>8} | {np.median(au):>15.3f}"
          f"{f'{(au>0.55).sum()} of 26':>11}{au.max():>7.3f} | {r['auc']:>19.3f}")
