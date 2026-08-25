"""One real subject, window by window, to show where 0.816 comes from."""
import json, numpy as np
from pathlib import Path
from cgmoutlier.data.cohort import load as load_cohort
from cgmoutlier.attack.statistic import window_distances

tgt = "973"
inc = Path("results/runs/copy_paste/include_973")
base = Path("results/runs/copy_paste/base")
m = json.load(open(str(inc / "meta.json")))
X, sids, _ = load_cohort(m["cohort"])
R = X[np.asarray(sids).astype(str) == tgt]          # 这个人自己的窗口
S_in = np.load(str(inc / "samples.npy"), mmap_mode="r")
S_out = np.load(str(base / "samples.npy"), mmap_mode="r")
k = min(len(S_in), len(S_out))
d_in  = window_distances(R, np.asarray(S_in[:k]),  set_reduce="min")
d_out = window_distances(R, np.asarray(S_out[:k]), set_reduce="min")

# `d_in < 1e-9` was too strict. window_distances computes
# sqrt(|r|^2 + |s|^2 - 2 r.s) in float32; for an EXACT copy the three terms cancel only
# to float32 precision, so a verbatim match reads ~1e-4, not 0. The first run counted
# 25/67 zeros and the arithmetic did not close. The distribution is clearly bimodal --
# there is a two-order-of-magnitude gap between the copies and the nearest non-copy --
# so threshold in the gap rather than at machine zero.
srt = np.sort(d_in)
print(f"  d_in 最小的 12 个: {np.array2string(srt[:12], precision=5)}")
print(f"  d_in 排序后的跳变位置: "
      f"{np.argmax(np.diff(srt) / (srt[:-1] + 1e-6)) + 1} / {len(srt)}")
zero = d_in < 0.01
print(f"受试者 {tgt}: {len(R)} 个窗口")
print(f"  d_in 恰好为 0 的窗口: {zero.sum()}/{len(R)} = {zero.mean():.4f}   (预测 0.632)")
print(f"  d_in  非零部分 中位 {np.median(d_in[~zero]):.4f}" if (~zero).any() else "")
print(f"  d_out 全部       中位 {np.median(d_out):.4f}   最小 {d_out.min():.4f}")
print(f"\n  前 12 个窗口:")
print(f"  {'窗口':>5}{'d_in':>10}{'d_out':>10}   判定")
for i in range(min(12, len(R))):
    tag = "抓现行 (d_in=0)" if zero[i] else "只能猜"
    print(f"  {i:>5}{d_in[i]:>10.4f}{d_out[i]:>10.4f}   {tag}")
u = (d_out[None, :] > d_in[:, None]).mean() + 0.5*(d_out[None, :] == d_in[:, None]).mean()
print(f"\n  逐人 AUC 实测 = {u:.4f}")
print(f"  按实测零比例的预测 = {zero.mean() + (1-zero.mean())*0.5:.4f}")
