"""The d1 epoch sweep, indexed by optimizer STEPS."""
import json, csv, os, numpy as np
MS = [2,3,4,6,8,10]
QK = ["context_fid","correlational","vds","fdds","discriminative","predictive",
      "mdd","acd","skewness_diff","kurtosis_diff"]
for cell in ('d1_c1','d1_c2'):
    print(f"\n{'='*104}\n{cell}   (T=288, C={1 if cell.endswith('c1') else 2}, "
          f"h=256, 6,072 windows, 200-step sampling)\n{'='*104}")
    print(f"  {'steps':>8}{'组间AUC':>10}{'p':>8}  |{'outlier中位':>12}{'normal中位':>12}"
          f"{'全体中位':>10}  |{'out>0.55':>10}{'norm>0.55':>11}")
    for m in MS:
        a = f'results/matrix/sweep/attack/{cell}_ms{m}/summary.json'
        s = f'results/matrix/sweep/subject_auc/{cell}_ms{m}/per_subject.csv'
        if not os.path.exists(a): print(f"  {m*10000:>8}   缺"); continue
        r = [x for x in json.load(open(a)) if x['frozen']][0]
        rows = list(csv.DictReader(open(s)))
        au = np.array([float(x['auc']) for x in rows]); g = np.array([x['group'] for x in rows])
        o, n = au[g == 'outlier'], au[g != 'outlier']
        print(f"  {m*10000:>8}{r['auc']:>10.4f}{r['p_between']:>8.3f}  |"
              f"{np.median(o):>12.3f}{np.median(n):>12.3f}{np.median(au):>10.3f}  |"
              f"{f'{(o>0.55).sum()}/13':>10}{f'{(n>0.55).sum()}/13':>11}")
    print(f"\n  base 生成质量(tsgen_metrics,全部 lower = better)")
    print(f"  {'steps':>8}" + "".join(f"{k[:11]:>12}" for k in QK))
    for m in MS:
        t = f'results/matrix/sweep/quality_tsgem/{cell}_ms{m}.json'
        q = f'results/matrix/sweep/quality/{cell}_ms{m}.json'
        if not os.path.exists(t): print(f"  {m*10000:>8}   缺"); continue
        v = list(json.load(open(t)).values())[0]
        print(f"  {m*10000:>8}" + "".join(f"{v.get(k, float('nan')):>12.4f}" for k in QK))
    print(f"\n  {'steps':>8}{'判别器max中位':>16}{'spread中位':>13}   (8 次重启,全部 27 个模型)")
    for m in MS:
        q = f'results/matrix/sweep/quality/{cell}_ms{m}.json'
        if not os.path.exists(q): continue
        d = json.load(open(q))
        print(f"  {m*10000:>8}{np.median([x['max'] for x in d.values()]):>16.4f}"
              f"{np.median([x['spread'] for x in d.values()]):>13.4f}")
