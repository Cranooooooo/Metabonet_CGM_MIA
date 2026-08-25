"""Measure the copy_paste bootstrap ceiling instead of asserting it.

THE CLAIM. sample() draws n indices WITH replacement from N training windows
(copy_paste.py:26). With n = N a given training window is missed with probability
(1-1/N)^N -> e^-1 = 0.3679, so 63.2% appear at least once. A window that appears
verbatim gives d_in = 0 while d_out > 0 and counts 1 in the per-subject AUC; one that
does not appear is at chance, 0.5. Ceiling 0.632*1 + 0.368*0.5 = 0.816.

THE CHECK. If that is right, the released set of n = N windows contains exactly
N * 0.632 DISTINCT windows -- which needs only samples.npy to count. No training set,
no job file.
"""
import json, numpy as np
from pathlib import Path

RUN = Path("results/runs/copy_paste")
runs = [p for p in sorted(RUN.iterdir()) if (p / "samples.npy").exists()][:5]
print(f"{'run':22}{'N':>9}{'n_rel':>9}{'n/N':>7}{'distinct':>10}{'measured':>10}{'1-1/e':>9}{'ceiling':>9}")
for r in runs:
    m = json.load(open(str(r / "meta.json")))
    N, n = m["n_train_windows"], m["params"]["n_samples"]
    S = np.load(str(r / "samples.npy"))
    d = len(np.unique(S.reshape(len(S), -1), axis=0))
    frac = d / N
    pred = 1 - (1 - 1/N)**n
    print(f"{r.name:22}{N:>9,}{n:>9,}{n/N:>7.3f}{d:>10,}{frac:>10.4f}{pred:>9.4f}"
          f"{frac + (1-frac)*0.5:>9.4f}")
