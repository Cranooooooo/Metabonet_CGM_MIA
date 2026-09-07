#!/usr/bin/env python
"""Where in a patient's record, and in what form, is the leak?

The attack reports one number per target: gap = d(t, base) - d(t, include_t), with
d = mean over t's windows of the distance to the NEAREST released sample. This script
takes that number apart two ways, and does so using the attack's own machinery -- the
same K-matching, the same nearest neighbours -- so that what it localises corresponds to
what Step 1 reports rather than to a second, differently-behaved attack.

WHEN: the squared Euclidean distance is a sum over the time axis, so once the nearest
neighbour is fixed (chosen exactly as the attack chooses it, on the whole window) the
distance splits per timestep and the parts sum back to the whole. This is a
decomposition, not an approximation.

WHAT: the same statistic recomputed after transforms that each destroy one kind of
information. If the gap survives sorting -- which destroys all timing -- the leak is
distributional. If it needs the raw series, it is tied to events at particular times.

BOTH ARE READ AGAINST THE CONTROL ARM. Any distance rises where glucose is most
variable, membership or not, so an outlier curve alone shows post-prandial excursions
rather than leakage. Only the outlier-minus-control difference is evidence.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np

from cgmoutlier._env import check as _envcheck                     # noqa: E402
_envcheck()
from cgmoutlier.data.cohort import load as load_cohort             # noqa: E402
from cgmoutlier.attack.statistic import _match                     # noqa: E402


def nearest(R, S, chunk=4096):
    """Index of the nearest released sample for each of R's windows, and that distance.

    Same expansion the attack uses, for the same reason: one BLAS call per chunk.
    """
    Rf = R.reshape(len(R), -1).astype(np.float32)
    Sf = S.reshape(len(S), -1).astype(np.float32)
    r2 = (Rf ** 2).sum(1, keepdims=True)
    best = np.full(len(Rf), np.inf, np.float64)
    idx = np.zeros(len(Rf), np.int64)
    for i in range(0, len(Sf), chunk):
        B = Sf[i:i + chunk]
        d2 = r2 + (B ** 2).sum(1)[None, :] - 2.0 * (Rf @ B.T)
        j = d2.argmin(1)
        v = d2[np.arange(len(Rf)), j]
        m = v < best
        best[m], idx[m] = v[m], j[m] + i
    return idx, np.sqrt(np.clip(best, 0, None))


TRANSFORMS = {
    "raw":     lambda X: X,
    "diff":    lambda X: np.diff(X, axis=1),
    "sorted":  lambda X: np.sort(X, axis=1),
    "hourly":  lambda X: X[:, :X.shape[1] // 12 * 12].reshape(
                             len(X), X.shape[1] // 12, 12, X.shape[2]).mean(2),
    "zscore":  lambda X: (X - X.mean(1, keepdims=True)) /
                         (X.std(1, keepdims=True) + 1e-6),
}


def d_stat(R, S):
    """The frozen statistic: mean over windows of the min distance, scaled by sqrt(F)."""
    _, dist = nearest(R, S)
    return float(np.mean(dist / np.sqrt(R.reshape(len(R), -1).shape[1])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", required=True)
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args()

    d = json.loads((Path(a.design) / "design.json").read_text())
    X, sids, man = load_cohort(a.cohort)
    sids = np.asarray(sids).astype(str)
    T, C = man["T"], man.get("C", X.shape[-1])
    S_base = np.load(f"{a.runs}/base/samples.npy")

    per_time, per_form = {}, {}
    for p in d["pairs"]:
        t, grp = str(p["target"]), p["group"]
        f = Path(a.runs) / p["member"] / "samples.npy"
        if not f.exists():
            print(f"[loc] {t}: no {f}, skipping", flush=True)
            continue
        R = np.ascontiguousarray(X[sids == t], np.float32)
        S_in, S_out, _ = _match(np.load(str(f)), S_base, t, a.seed, True)

        # --- WHEN: fix the attack's nearest neighbours, then split by timestep ---
        i_in, _ = nearest(R, S_in)
        i_out, _ = nearest(R, S_out)
        c_in = ((R - S_in[i_in]) ** 2).sum(2)          # (n_windows, T)
        c_out = ((R - S_out[i_out]) ** 2).sum(2)
        per_time[t] = dict(group=grp, n_windows=int(len(R)),
                           curve=(c_out - c_in).mean(0).tolist())

        # --- WHAT: the same statistic in each transformed space ---
        rec = {}
        for name, fn in TRANSFORMS.items():
            rec[name] = d_stat(fn(R), fn(S_out)) - d_stat(fn(R), fn(S_in))
        per_form[t] = dict(group=grp, **rec)
        print(f"[loc] {t:14} {grp:8} " +
              "  ".join(f"{k}={v:+.5f}" for k, v in rec.items()), flush=True)

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / "per_timestep.json").write_text(json.dumps(
        dict(T=T, C=C, cohort=a.cohort, runs=a.runs, targets=per_time)))
    (out / "per_transform.json").write_text(json.dumps(per_form, indent=1))

    print(f"\n  {'transform':10}{'outlier mean':>14}{'control mean':>14}{'difference':>13}")
    for name in TRANSFORMS:
        o = [v[name] for v in per_form.values() if v["group"] == "outlier"]
        c = [v[name] for v in per_form.values() if v["group"] != "outlier"]
        print(f"  {name:10}{np.mean(o):>14.5f}{np.mean(c):>14.5f}"
              f"{np.mean(o)-np.mean(c):>13.5f}")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
