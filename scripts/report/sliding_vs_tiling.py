"""How many seven-day windows does tiling discard that sliding would keep?"""
import numpy as np, collections, json
s = np.load('data/cohort/matrix_detect/subject_ids.npy', allow_pickle=True).astype(str)
d = np.load('data/cohort/matrix_detect/days.npy').astype('int64')
by = collections.defaultdict(list)
for a, b in zip(s, d): by[a].append(b)
tile = slide = 0; runs = []
for sub, days in by.items():
    dd = np.sort(np.unique(days))
    brk = np.flatnonzero(np.diff(dd) != 1) + 1
    for run in np.split(dd, brk):
        L = len(run); runs.append(L)
        if L >= 7:
            tile += L // 7
            slide += L - 7 + 1
runs = np.array(runs)
print(f"  506 人,连续日段 {len(runs)} 条")
print(f"    段长 中位 {int(np.median(runs))}  最长 {runs.max()}  >=7 天的段 {(runs>=7).sum()}")
print(f"\n  七天窗口数")
print(f"    平铺(现在,步长7)   {tile:,}")
print(f"    滑动(步长1)        {slide:,}     = {slide/tile:.1f} 倍")
print(f"\n  对 epoch 的影响(100,000 步 x batch 64)")
for n, lab in ((6072, '现在 d7(块匹配后)'), (tile, '平铺全量'), (slide, '滑动全量')):
    print(f"    {lab:22} {n:>7,} 窗口 -> {100000*64/n:>7.0f} epochs")
print(f"\n  单天窗口做参照: {len(s):,} 个 -> {100000*64/len(s):.0f} epochs")
