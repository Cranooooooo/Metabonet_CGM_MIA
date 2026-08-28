"""Hour-of-day leakage curves, outliers against controls.

The attack distance is additive over the time axis, so fixing the nearest neighbour the
attack already chose and splitting the squared distance per timestep gives a curve whose
total is the reported statistic. Plotted against the control arm, because any distance
rises where glucose is most variable whether or not anyone is being identified.
"""
import json, numpy as np, os
def curves(cell):
    d = json.load(open(f'results/matrix/localise/{cell}/per_timestep.json'))
    T = d['T']; tg = d['targets']
    o = np.array([v['curve'] for v in tg.values() if v['group'] == 'outlier'])
    c = np.array([v['curve'] for v in tg.values() if v['group'] != 'outlier'])
    return T, o, c, tg
def fold(a, T):
    """T slots of 5 min -> hour of day (24), averaging over whatever days are in the window."""
    per_day = 288
    days = T // per_day
    a = a[:, :days*per_day].reshape(len(a), days, 24, 12).mean(3).mean(1)
    return a
for cell in ('d1_c1','d1_c2','d7_c1'):
    p = f'results/matrix/localise/{cell}/per_timestep.json'
    if not os.path.exists(p): print(f"{cell}: missing"); continue
    T, o, c, tg = curves(cell)
    fo, fc = fold(o, T), fold(c, T)
    diff = fo.mean(0) - fc.mean(0)
    print(f"\n===== {cell}  T={T}  ({T//288} day(s))  outliers {len(o)}  controls {len(c)}")
    print("  hour-of-day, outlier minus control (positive = leakage concentrated there)")
    for h in range(24):
        bar = '#' * max(0, int(round(diff[h] / max(abs(diff).max(), 1e-12) * 40)))
        print(f"    {h:02d}:00 {diff[h]:+9.5f} {bar}")
    pk = int(np.argmax(diff))
    print(f"  peak hour {pk:02d}:00   diff {diff[pk]:+.5f}   "
          f"outlier {fo.mean(0)[pk]:+.5f}  control {fc.mean(0)[pk]:+.5f}")
    np.save(f'results/matrix/localise/{cell}/folded_outlier.npy', fo)
    np.save(f'results/matrix/localise/{cell}/folded_control.npy', fc)
    if T > 288:
        ho = o.reshape(len(o), T//12, 12).mean(2); hc = c.reshape(len(c), T//12, 12).mean(2)
        np.save(f'results/matrix/localise/{cell}/hourly168_outlier.npy', ho)
        np.save(f'results/matrix/localise/{cell}/hourly168_control.npy', hc)
        print(f"  also wrote the unfolded {T//12}-hour curves")
