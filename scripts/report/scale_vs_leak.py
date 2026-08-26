"""Does shrinking the cohort make everyone leak, and is it the data volume or the epochs?"""
import csv, json, glob, os
import numpy as np

def stats(path):
    if not os.path.exists(path): return None
    rows = list(csv.DictReader(open(path)))
    a = np.array([float(r['auc']) for r in rows])
    g = np.array([r['group'] for r in rows])
    return dict(n=len(a), med=float(np.median(a)), lo=float(a.min()), hi=float(a.max()),
                over=int((a > 0.55).sum()),
                med_o=float(np.median(a[g=='outlier'])) if (g=='outlier').any() else None,
                med_c=float(np.median(a[g!='outlier'])) if (g!='outlier').any() else None)

ROWS = [
 ("published d1 (h128)", "results/subject_auc/per_subject.csv",
  "results/runs/dimts_h128_rep1/base/meta.json"),
 ("d1 matched control",  "results/subject_auc_d1_matched/per_subject.csv",
  "results/runs/dimts_d1_matched_rep1/base/meta.json"),
 ("d7 h256 pilot",       "results/subject_auc_d7_h256/per_subject.csv",
  "results/runs/dimts_d7_h256_rep1/base/meta.json"),
 ("matrix d1_c1",        "results/matrix/subject_auc/d1_c1/per_subject.csv",
  "results/runs/matrix_d1_c1/base/meta.json"),
 ("matrix d1_c2",        "results/matrix/subject_auc/d1_c2/per_subject.csv",
  "results/runs/matrix_d1_c2/base/meta.json"),
 ("matrix d7_c1",        "results/matrix/subject_auc/d7_c1/per_subject.csv",
  "results/runs/matrix_d7_c1/base/meta.json"),
]
print(f"{'campaign':22}{'窗口':>9}{'人':>6}{'epochs':>8}{'n':>4}"
      f"{'AUC中位':>9}{'范围':>16}{'>0.55':>8}{'out':>7}{'norm':>7}")
for name, csvp, metap in ROWS:
    s = stats(csvp)
    if s is None:
        print(f"{name:22}  (no per_subject.csv on disk)"); continue
    if os.path.exists(metap):
        m = json.load(open(metap)); w = m['n_train_windows']
        ep = m['params'].get('steps', 0)*64/w
        sub = m['n_subjects']
    else:
        w = ep = sub = float('nan')
    o = f"{s['med_o']:.3f}" if s['med_o'] is not None else "  -  "
    c = f"{s['med_c']:.3f}" if s['med_c'] is not None else "  -  "
    print(f"{name:22}{w:>9,}{sub:>6}{ep:>8.0f}{s['n']:>4}{s['med']:>9.3f}"
          f"   [{s['lo']:.3f},{s['hi']:.3f}]{s['over']:>6}/{s['n']:<3}{o:>7}{c:>7}")
