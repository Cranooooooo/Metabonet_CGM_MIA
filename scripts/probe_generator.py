#!/usr/bin/env python
"""What does one model of the paired design actually cost?

    python scripts/probe_generator.py --generator fourier_diff timevae

49 models is not a number to commit to on an estimate, and the published defaults are
not comparable across baselines: DiM-TS denoises in 500 steps and FourierDiffusion in
1000, DiM-TS counts a step budget and FourierDiffusion counts epochs. Reading those
off the configs answers "which default is bigger", not "which is cheaper".

So this measures one thing per generator, on identical data, at whatever budget the
caller sets:

    fit_seconds        a deliberately small training budget -- a rate, not a run
    seconds_per_sample sampling throughput, extrapolated to the real K

Sampling is measured because it is the term most likely to dominate. With K = the
training-set size, every model in the design releases ~177k sequences; at 500 or 1000
denoising steps each, that can cost more than the training it follows. If it does, the
choice is between generators, between a smaller K, and between fewer denoising steps --
and those are different trades that should be made on a number.

WHAT THIS DOES NOT MEASURE
--------------------------
Quality. Nothing here says whether a generator's samples are usable CGM, and a
generator that is fast because it has not learned anything is not a saving. The
clinical battery in `cgmoutlier.clinical` is what would answer that, on real runs.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from cgmoutlier._env import check as _envcheck                     # noqa: E402
_envcheck()
from cgmoutlier.data.cohort import load as load_cohort                # noqa: E402
from cgmoutlier.generators.registry import get as get_generator       # noqa: E402

# A small, explicitly-stated budget per generator: enough steps to time a step, few
# enough to fit in a probe. These are NOT the training settings for a real run.
# The step counts must be >= 4 for --train-rate: the second point is budget // 4, and
# max(1, 1 // 4) == 1 gives two identical points and no slope.
PROBE_BUDGET = {
    "fourier_diff": dict(train=dict(max_epochs=4, batch_size=64),
                         sample=dict(num_diffusion_steps=1000, sample_batch=128)),
    "timevae":      dict(train=dict(max_epochs=8, batch_size=64), sample=dict()),
    "diffusion_ts": dict(train=dict(train_num_steps=200, batch_size=64), sample=dict()),
    "diffwave":     dict(train=dict(max_steps=200, batch_size=64), sample=dict()),
    "padts":        dict(train=dict(max_steps=200, batch_size=64), sample=dict()),
    "igfm":         dict(train=dict(max_steps=200, batch_size=64), sample=dict()),
    "copy_paste":   dict(train=dict(), sample=dict()),
}


def count_params(gen):
    """Best-effort parameter count. Adapters hold their model under different names."""
    import torch
    for attr in ("_model", "model", "net", "_net"):
        m = getattr(gen, attr, None)
        if isinstance(m, torch.nn.Module):
            return int(sum(p.numel() for p in m.parameters()))
    return None


STEP_KEY = {"fourier_diff": "max_epochs", "timevae": "max_epochs",
            "diffusion_ts": "train_num_steps", "diffwave": "max_steps",
            "padts": "max_steps", "igfm": "max_steps"}


def train_rate(name, X, device, seed, budget):
    """Seconds per training step, from two budgets rather than one.

    A single small budget cannot separate the per-step cost from the fixed setup:
    the first run of this probe reported 431 s for 200 steps of a 224k-parameter
    model, which is setup, not 2.15 s/step. Two points give the slope, and the
    intercept is the overhead that a real 100k-step run pays once.
    """
    key = STEP_KEY.get(name)
    if key is None or key not in budget:
        return None
    lo = dict(budget); lo[key] = max(1, budget[key] // 4)
    pts = []
    for b in (lo, budget):
        gen = get_generator(name)(T=X.shape[1], C=X.shape[2], device=device, seed=seed)
        t0 = time.time()
        gen.fit(X, b)
        pts.append((b[key], time.time() - t0))
    (n0, t0_), (n1, t1_) = pts
    if n1 == n0:
        return None
    per = (t1_ - t0_) / (n1 - n0)
    setup = t0_ - per * n0
    rec = dict(unit=key, per_unit_seconds=round(per, 4),
               setup_seconds=round(setup, 1), points=pts)
    # A non-positive slope means the two budgets cost the same: setup swamped the
    # stepping and nothing about the per-step rate was measured. Say so, rather than
    # let a negative number be read as a rate. diffwave came out at -0.0065 s/step
    # against 561 s of setup -- a single budget point would have reported 2.8 s/step,
    # two orders of magnitude high.
    if per <= 0 or (setup > 0 and per * n1 < 0.1 * setup):
        rec["unmeasured"] = (
            f"setup ({setup:.0f}s) dominates both budgets ({n0}, {n1} {key}); "
            f"the per-{key} cost is not resolved. Re-probe with budgets large enough "
            f"that stepping is a visible share of the total.")
    return rec


def probe(name, X, n_sample, K, device, seed, params=None, with_rate=False):
    budget = PROBE_BUDGET.get(name, dict(train=dict(), sample=dict()))
    N, T, C = X.shape
    gen = get_generator(name)(T=T, C=C, params=dict(params or {}),
                              device=device, seed=seed)

    t0 = time.time()
    gen.fit(X, budget["train"])
    t_fit = time.time() - t0
    rate = train_rate(name, X, device, seed, budget["train"]) if with_rate else None

    t0 = time.time()
    S = gen.sample(n_sample, budget["sample"])
    t_sample = time.time() - t0

    S = np.asarray(S)
    per = t_sample / max(1, n_sample)
    return dict(
        generator=name, n_train=int(N), T=int(T), C=int(C),
        params=count_params(gen), train_budget=budget["train"],
        sample_budget=budget["sample"],
        fit_seconds=round(t_fit, 1), train_rate=rate, n_sampled=int(n_sample),
        sample_seconds=round(t_sample, 1),
        seconds_per_sample=round(per, 5),
        hours_for_K=round(per * K / 3600, 2),
        hours_for_49_models_sampling_only=round(per * K * 49 / 3600, 1),
        sample_shape=list(S.shape), sample_nan_frac=float(np.isnan(S).mean()),
        sample_range=[float(np.nanmin(S)), float(np.nanmax(S))],
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generator", nargs="+", required=True)
    ap.add_argument("--cohort", default="data/cohort/metabonet875")
    ap.add_argument("--n-train", type=int, default=20_000,
                    help="windows used for the probe's training budget")
    ap.add_argument("--n-sample", type=int, default=512,
                    help="samples drawn to time sampling")
    ap.add_argument("--K", type=int, default=177_000,
                    help="the real released-set size, for the extrapolation")
    ap.add_argument("--out", default="results/probe/generators.json")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--train-rate", action="store_true",
                    help="also fit at a second, smaller budget so the per-step cost "
                         "can be separated from the fixed setup. Roughly doubles the "
                         "probe's runtime.")
    a = ap.parse_args()

    X, sids, man = load_cohort(a.cohort)
    X = np.ascontiguousarray(np.asarray(X)[:a.n_train], dtype=np.float32)
    print(f"[probe] {X.shape} windows, device={a.device}, "
          f"extrapolating to K={a.K:,}\n", flush=True)

    rows = []
    for name in a.generator:
        print(f"[probe] {name} ...", flush=True)
        try:
            r = probe(name, X, a.n_sample, a.K, a.device, a.seed,
                      with_rate=a.train_rate)
        except Exception as e:                      # one bad adapter must not cost
            r = dict(generator=name, error=f"{type(e).__name__}: {e}")   # the others
            print(f"[probe] {name} FAILED: {r['error']}", file=sys.stderr, flush=True)
        rows.append(r)
        print(json.dumps(r, indent=2), flush=True)

    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2))

    ok = [r for r in rows if "error" not in r]
    if ok:
        print(f"\n{'generator':<14}{'params':>10}{'s/sample':>11}"
              f"{'h for K':>10}{'h, 49 models':>14}")
        for r in sorted(ok, key=lambda r: r["seconds_per_sample"]):
            p = f"{r['params']:,}" if r["params"] else "?"
            print(f"{r['generator']:<14}{p:>10}{r['seconds_per_sample']:>11.5f}"
                  f"{r['hours_for_K']:>10.2f}{r['hours_for_49_models_sampling_only']:>14.1f}")
        print("\nsampling only; training is on top. Quality is not measured here.")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
