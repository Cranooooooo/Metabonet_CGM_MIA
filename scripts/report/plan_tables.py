import json, csv, os, numpy as np
def row(cell, atk, sauc, qt, qd):
    r = [x for x in json.load(open(atk)) if x['frozen']][0]
    rows = list(csv.DictReader(open(sauc)))
    au = np.array([float(x['auc']) for x in rows]); g = np.array([x['group'] for x in rows])
    o, n = au[g == 'outlier'], au[g != 'outlier']
    fid = list(json.load(open(qt)).values())[0]['context_fid'] if os.path.exists(qt) else None
    dm = None
    if os.path.exists(qd):
        d = json.load(open(qd))
        b = [v for k, v in d.items() if k.rstrip('/').endswith('/base')]
        dm = (b[0] if b else list(d.values())[0])['max']
    return np.median(o), np.median(n), (o>0.55).sum(), (n>0.55).sum(), r['auc'], fid, dm

print("STEP 1")
print(f"  {'cond':8}{'out中位':>9}{'norm中位':>10}{'out>.55':>9}{'norm>.55':>10}{'组间AUC':>10}{'ctxFID':>9}{'判别器':>8}")
for c in ('d1_c1','d1_c2','d7_c1'):
    v = row(c, f'results/matrix/attack/{c}/summary.json',
            f'results/matrix/subject_auc/{c}/per_subject.csv',
            f'results/matrix/quality_tsgem/{c}_st500.json',
            f'results/matrix/quality/{c}/disc_stability.json')
    print(f"  {c:8}{v[0]:>9.3f}{v[1]:>10.3f}{f'{v[2]}/13':>9}{f'{v[3]}/13':>10}"
          f"{v[4]:>10.3f}{v[5]:>9.4f}{v[6]:>8.3f}")

print("\nSTEP 3b  d1_c1 训练轨迹")
print(f"  {'steps':>8}{'out中位':>9}{'norm中位':>10}{'out>.55':>9}{'norm>.55':>10}{'组间AUC':>10}{'ctxFID':>9}{'判别器':>8}")
for m in (2,3,4,6,8,10):
    v = row('x', f'results/matrix/sweep/attack/d1_c1_ms{m}/summary.json',
            f'results/matrix/sweep/subject_auc/d1_c1_ms{m}/per_subject.csv',
            f'results/matrix/sweep/quality_tsgem/d1_c1_ms{m}.json',
            f'results/matrix/sweep/quality/d1_c1_ms{m}.json')
    print(f"  {m*10000:>8}{v[0]:>9.3f}{v[1]:>10.3f}{f'{v[2]}/13':>9}{f'{v[3]}/13':>10}"
          f"{v[4]:>10.3f}{v[5]:>9.4f}{v[6]:>8.3f}")
