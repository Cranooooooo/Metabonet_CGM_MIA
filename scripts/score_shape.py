#!/usr/bin/env python
"""Are the outlier scores long-tailed or roughly normal?

The consensus takes the top 5% of each method. That cut is a choice, and how sensitive
the resulting list is to it depends on the shape of each score's distribution: on a
long-tailed score the subjects near the cut are sparse and a small move in the
percentage swaps people in and out, while on a near-normal score the cut sits in a dense
region and is stable.

Reported per method: skewness, excess kurtosis, the Shapiro-Wilk p-value against
normality, and -- the operationally useful one -- how many of the top-5% set survive at
top-4% and top-6%.
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

from cgmoutlier._env import check as _envcheck                      # noqa: E402
_envcheck()

d = Path(sys.argv[1] if len(sys.argv) > 1 else "results/outliers_sid_c3/stability_1ch/seed2026")
print(f"scores from {d}\n")
print(f"{'method':>6}{'skew':>9}{'kurt':>9}{'SW p':>10}{'log-skew':>10}"
      f"{'top4∩top5':>11}{'top6⊇top5':>11}  shape")
for p in sorted(d.glob("[ABCDE]*.parquet")):
    s = pd.read_parquet(p).set_index("id")["score"].astype(float)
    n = len(s)
    sk, ku = float(stats.skew(s)), float(stats.kurtosis(s))
    swp = float(stats.shapiro(s.sample(min(4000, n), random_state=0)).pvalue)
    pos = s - s.min() + 1e-9
    lsk = float(stats.skew(np.log(pos)))
    t5 = set(s.nlargest(max(1, int(n * 0.05))).index)
    t4 = set(s.nlargest(max(1, int(n * 0.04))).index)
    t6 = set(s.nlargest(max(1, int(n * 0.06))).index)
    shape = ("heavy right tail" if sk > 2 else
             "right-skewed" if sk > 0.5 else
             "near-symmetric" if abs(sk) <= 0.5 else "left-skewed")
    print(f"{p.stem:>6}{sk:>9.2f}{ku:>9.2f}{swp:>10.1e}{lsk:>10.2f}"
          f"{len(t4 & t5)/len(t5):>10.0%}{len(t5 & t6)/len(t5):>10.0%}  {shape}")

print("\nskew 0 and kurt 0 is normal. SW p < 0.05 rejects normality -- at n>1000 it "
      "rejects almost anything, so read the skew.")
print("top4/top5 is what fraction of the 5% set is still there at 4%; a long tail makes "
      "the boundary sparse and that number is then close to 4/5 = 80% by construction. "
      "Well below 80% means the cut is unstable for a reason beyond arithmetic.")
