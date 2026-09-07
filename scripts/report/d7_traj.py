import json, csv, os, numpy as np
def row(base, cell, m):
    a=[x for x in json.load(open(f'{base}/attack/{cell}_ms{m}/summary.json')) if x['frozen']][0]
    s=list(csv.DictReader(open(f'{base}/subject_auc/{cell}_ms{m}/per_subject.csv')))
    au=np.array([float(x['auc']) for x in s]); g=np.array([x['group'] for x in s])
    q=f'{base}/quality/{cell}_ms{m}.json'; t=f'{base}/quality_tsgem/{cell}_ms{m}.json'
    dm=np.median([v['max'] for v in json.load(open(q)).values()]) if os.path.exists(q) else float('nan')
    fid=list(json.load(open(t)).values())[0]['context_fid'] if os.path.exists(t) else float('nan')
    o,n=au[g=='outlier'],au[g!='outlier']
    return m*10000, round(m*10000*64/5741), (au>.55).sum(), au.max(), a['auc'], (o>.55).sum(), (n>.55).sum(), fid, dm
print(f"  {'cell':7}{'steps':>8}{'epochs':>8}{'risk>0.55':>11}{'maxAUC':>8}{'armAUC':>8}"
      f"{'out':>6}{'norm':>6}{'ctxFID':>9}{'discMax':>9}")
for m in (3,4):
    v=row('results/matrix/sweep','d1_c1',m)
    print(f"  {'d1_c1':7}{v[0]:>8,}{v[1]:>8}{f'{v[2]}/26':>11}{v[3]:>8.3f}{v[4]:>8.3f}"
          f"{f'{v[5]}/13':>6}{f'{v[6]}/13':>6}{v[7]:>9.4f}{v[8]:>9.4f}")
for m in (3,4):
    v=row('results/matrix/sweep','d7_c1',m)
    print(f"  {'d7_c1':7}{v[0]:>8,}{v[1]:>8}{f'{v[2]}/26':>11}{v[3]:>8.3f}{v[4]:>8.3f}"
          f"{f'{v[5]}/13':>6}{f'{v[6]}/13':>6}{v[7]:>9.4f}{v[8]:>9.4f}")
print()
a=[x for x in json.load(open('results/matrix/attack/d7_c1/summary.json')) if x['frozen']][0]
s=list(csv.DictReader(open('results/matrix/subject_auc/d7_c1/per_subject.csv')))
au=np.array([float(x['auc']) for x in s]); g=np.array([x['group'] for x in s])
o,n=au[g=='outlier'],au[g!='outlier']
print(f"  {'d7_c1':7}{100000:>8,}{1115:>8}{f'{(au>.55).sum()}/26':>11}{au.max():>8.3f}"
      f"{a['auc']:>8.3f}{f'{(o>.55).sum()}/13':>6}{f'{(n>.55).sum()}/13':>6}"
      f"{0.3313:>9.4f}{0.6675:>9.4f}   <- 100k(500步口径)")
