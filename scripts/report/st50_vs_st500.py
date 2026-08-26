"""Does dropping 500 denoising steps to 50 change the MIA answer?"""
import json, csv, numpy as np
from scipy import stats
ids = json.load(open('results/matrix/report/subject_ids.json'))
CELLS = ['d1_c1','d1_c2','d7_c1']
def frozen(p):
    return [x for x in json.load(open(p)) if x['frozen']][0]
print("Q2 组间 AUC(冻结 min x mean)")
print(f"  {'cell':8}{'500 步':>10}{'50 步':>10}{'差':>9}   {'p(500)':>9}{'p(50)':>9}")
for c in CELLS:
    a = frozen(f'results/matrix/attack/{c}/summary.json')
    b = frozen(f'results/matrix/attack/{c}_st50/summary.json')
    print(f"  {c:8}{a['auc']:>10.4f}{b['auc']:>10.4f}{b['auc']-a['auc']:>+9.4f}"
          f"   {a['p_between']:>9.4f}{b['p_between']:>9.4f}")
print("\nQ1 逐人 AUC")
print(f"  {'cell':8}{'500中位':>10}{'50中位':>10}{'Spearman':>11}{'p':>9}{'|差|中位':>10}")
for c in CELLS:
    A = {r['target']: float(r['auc']) for r in csv.DictReader(open(f'results/matrix/subject_auc/{c}/per_subject.csv'))}
    B = {r['target']: float(r['auc']) for r in csv.DictReader(open(f'results/matrix/subject_auc/{c}_st50/per_subject.csv'))}
    k = sorted(set(A) & set(B))
    a = np.array([A[x] for x in k]); b = np.array([B[x] for x in k])
    rho, p = stats.spearmanr(a, b)
    print(f"  {c:8}{np.median(a):>10.3f}{np.median(b):>10.3f}{rho:>+11.3f}{p:>9.1e}"
          f"{np.median(np.abs(a-b)):>10.3f}")
print("\n逐人 AUC 明细(d1_c2:唯一 p<0.05 的格子)")
A = {r['target']: (float(r['auc']), r['group']) for r in csv.DictReader(open('results/matrix/subject_auc/d1_c2/per_subject.csv'))}
B = {r['target']: float(r['auc']) for r in csv.DictReader(open('results/matrix/subject_auc/d1_c2_st50/per_subject.csv'))}
print(f"  {'编号':>7}{'组':>9}{'500步':>8}{'50步':>8}{'差':>8}")
for t in sorted(A, key=lambda x: -A[x][0]):
    print(f"  {ids[t]:>7}{A[t][1]:>9}{A[t][0]:>8.3f}{B[t]:>8.3f}{B[t]-A[t][0]:>+8.3f}")
