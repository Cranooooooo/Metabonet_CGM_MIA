"""Re-run the design document's load-bearing measurement with the pipeline's own RNG.

docs/MODEL_DESIGN.md is built on: in raw space both arms show a positive gap, but in
per-channel-sorted space (all timing destroyed) the CONTROL arm's gap falls to zero while
the outlier arm's does not. That was computed with np.random.RandomState because the
environment there lacked default_rng, so the K-matching subsample differed from the one
the attack actually uses. The between-space comparison is unaffected by that -- both
spaces saw the same subsample -- but the absolute figures are not, and they are quoted.

This uses attack.statistic._match, the same function run_attack uses.
"""
import json, numpy as np
from pathlib import Path
from cgmoutlier._env import check as _envcheck
_envcheck()
from cgmoutlier.data.cohort import load as load_cohort
from cgmoutlier.attack.statistic import _match

CELL, SEED = "d1_c1", 2026
d = json.loads(Path("results/matrix/design/rep1/design.json").read_text())
X, sids, man = load_cohort(f"data/cohort/matrix_{CELL}")
sids = np.asarray(sids).astype(str)
S_base = np.load(f"results/runs/matrix_{CELL}/base/samples.npy")

def dstat(R, S):
    Rf = R.reshape(len(R), -1).astype(np.float32)
    Sf = S.reshape(len(S), -1).astype(np.float32)
    r2 = (Rf**2).sum(1, keepdims=True); best = np.full(len(Rf), np.inf)
    for i in range(0, len(Sf), 4096):
        B = Sf[i:i+4096]
        v = (r2 + (B**2).sum(1)[None,:] - 2.0*(Rf@B.T)).min(1)
        np.minimum(best, v, out=best)
    return float(np.mean(np.sqrt(np.clip(best,0,None))/np.sqrt(Rf.shape[1])))

rows = []
for p in d["pairs"]:
    t, grp = str(p["target"]), p["group"]
    f = Path(f"results/runs/matrix_{CELL}") / p["member"] / "samples.npy"
    if not f.exists(): continue
    R = np.ascontiguousarray(X[sids == t], np.float32)
    S_in, S_out, _ = _match(np.load(str(f)), S_base, t, SEED, True)
    rec = {"g": grp}
    for name, fn in (("raw", lambda A: A), ("sorted", lambda A: np.sort(A, axis=1))):
        din, dout = dstat(fn(R), fn(S_in)), dstat(fn(R), fn(S_out))
        rec[name] = (dout - din, din)
    rows.append(rec); print(f"  {t:14} {grp:8} "
        + "  ".join(f"{k}: gap {v[0]:+.5f} d_in {v[1]:.4f}" for k,v in rec.items() if k!='g'), flush=True)

print(f"\n  {'space':8}{'d_in':>9}{'outlier gap/d_in':>19}{'control gap/d_in':>19}")
for sp in ("raw","sorted"):
    o = [r[sp][0]/r[sp][1] for r in rows if r["g"]=="outlier"]
    c = [r[sp][0]/r[sp][1] for r in rows if r["g"]!="outlier"]
    din = np.mean([r[sp][1] for r in rows])
    print(f"  {sp:8}{din:>9.4f}{100*np.mean(o):>18.1f}%{100*np.mean(c):>18.1f}%")
