import json, csv, numpy as np
m = json.load(open('results/matrix/report/subject_ids.json'))
CELLS = ['d1_c1','d1_c2','d7_c1']
print(f"{'cell':8}{'outlier gap':>13}{'normal gap':>12}{'组间AUC':>10}{'p':>9}")
for c in CELLS:
    r = [x for x in json.load(open(f'results/matrix/attack/{c}/summary.json')) if x['frozen']][0]
    print(f"{c:8}{r['median_gap_outlier']:>13.5f}{r['median_gap_control']:>12.5f}"
          f"{r['auc']:>10.4f}{r['p_between']:>9.4f}")
S = {c: {r['target']: r for r in csv.DictReader(
        open(f'results/matrix/subject_auc/{c}/per_subject.csv'))} for c in CELLS}
print(f"\n逐人 AUC")
print(f"{'编号':>7}{'组':>9}" + "".join(f"{c:>9}" for c in CELLS))
for t in sorted(S['d1_c1'], key=lambda k: -float(S['d7_c1'][k]['auc'])):
    g = S['d1_c1'][t]['group']
    print(f"{m[t]:>7}{g:>9}" + "".join(f"{float(S[c][t]['auc']):>9.3f}" for c in CELLS))
print()
for c in CELLS:
    o=[float(v['auc']) for v in S[c].values() if v['group']=='outlier']
    n=[float(v['auc']) for v in S[c].values() if v['group']=='control']
    print(f"  {c}: outlier 中位 {np.median(o):.3f}  normal 中位 {np.median(n):.3f}"
          f"   >0.55  {sum(x>.55 for x in o)}/13 vs {sum(x>.55 for x in n)}/13")
