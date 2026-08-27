import json, csv, os, numpy as np
MS = [2,3,4,6,8,10]
EP = {m: round(m*10000*64/5741) for m in MS}
for cell in ('d1_c1','d1_c2'):
    print(f"\n===== {cell}")
    print(f"  {'ms':>3}{'epoch':>7}{'Q2 组间AUC':>12}{'p':>8}"
          f"{'Q1 中位':>9}{'out中位':>9}{'norm中位':>10}{'>0.55':>9}"
          f"{'判别器max':>10}{'ctxFID':>9}")
    for m in MS:
        a = f'results/matrix/sweep/attack/{cell}_ms{m}/summary.json'
        s = f'results/matrix/sweep/subject_auc/{cell}_ms{m}/per_subject.csv'
        q = f'results/matrix/sweep/quality/{cell}_ms{m}.json'
        t = f'results/matrix/sweep/quality_tsgem/{cell}_ms{m}.json'
        if not os.path.exists(a): print(f"  {m:>3}{EP[m]:>7}   缺"); continue
        r = [x for x in json.load(open(a)) if x['frozen']][0]
        rows = list(csv.DictReader(open(s)))
        au = np.array([float(x['auc']) for x in rows])
        g = np.array([x['group'] for x in rows])
        qm = np.median([v['max'] for v in json.load(open(q)).values()]) if os.path.exists(q) else float('nan')
        fid = list(json.load(open(t)).values())[0]['context_fid'] if os.path.exists(t) else float('nan')
        print(f"  {m:>3}{EP[m]:>7}{r['auc']:>12.4f}{r['p_between']:>8.3f}"
              f"{np.median(au):>9.3f}{np.median(au[g=='outlier']):>9.3f}"
              f"{np.median(au[g!='outlier']):>10.3f}"
              f"{f'{(au[g==chr(111)+chr(117)+chr(116)+chr(108)+chr(105)+chr(101)+chr(114)]>0.55).sum()}/{(au[g!=chr(111)+chr(117)+chr(116)+chr(108)+chr(105)+chr(101)+chr(114)]>0.55).sum()}':>9}"
              f"{qm:>10.4f}{fid:>9.4f}")
