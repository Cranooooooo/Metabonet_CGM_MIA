#!/usr/bin/env python
"""The mask head as a localiser: which cells does MAVEN fail to predict from context?

THE IDEA BEING TESTED
---------------------
MAVEN's velocity net is mask-conditioned -- `AEMaskVelocityNet.forward(x, t, M_cond,
X_cond, X_prior)` -- and its auxiliary task is masked imputation. Sampling has never used
that interface: `cgm_train_sample.draw` passes `M_cond = 0`. This script uses it.

Mask a cell, re-impute it from the rest of the window, and measure the error:

    r[t,c] = mean over rounds of ( x_gen[t,c] - impute(x_gen | context)[t,c] )**2

A cell with LOW r follows from the window's own context -- reproducing it is structure,
not memory. A cell with HIGH r does not follow from context, yet the model generated it
anyway, which is where memorised specifics would have to live.

The attraction is that this references no attack, no training window and no distance
formula: it is computable from the model alone. If it agrees with the model-free maps in
scripts/localise_cells.py, we have an intrinsic localiser that DiM-TS structurally cannot
have, and the adaptive-attack objection to selecting cells by the frozen statistic's own
per-cell contribution goes away.

THE PREDICTION THAT WOULD FALSIFY IT
------------------------------------
MODEL_DESIGN 2.3 measured that the leak lives in level and scale -- destroying them
(`zscore`) drops the arm AUC to 0.527/0.657/0.467, the weakest of five transforms. Level
is exactly what a masked autoencoder predicts best from context. So the honest prior is
that surprisal points at NOISY cells rather than LEAKING ones, and this measurement is
built to detect that rather than to confirm the idea. Agreement with the QUANTILE map is
the test; agreement with RAW alone is not enough, since results/matrix/ceiling measured
RAW and QUANTILE at a Jaccard lift of only 1.4-1.8x over chance.

WHAT IS MEASURED, PRECISELY
---------------------------
- The mask is drawn in the TIME domain and pushed into image space with `phi_A.encode`.
  That is exact only because view A is cell-aligned (MODEL_DESIGN 1.2: every image cell
  has exactly one time pre-image). The STFT branch has no such property, which is the
  structural reason a value-domain edit has to live on the delay/day-fold view.
- `X_prior` is passed as None. In training the prior comes from the same batch's
  unconditional pass and therefore carries information about the masked cells; feeding
  the sample's own values back in would make the imputation reproduce them and the
  surprisal would read ~0 by construction. None is what makes this a context-only test.
- Observed cells are conditioned on, not clamped, matching training.
- View A only. The released sample is the gate-fused output of both branches, so this
  measures predictability under the delay branch's model, not under the released fusion.
  That is the right object for a localiser and the WRONG one for the edit itself, which
  would have to handle the fusion. Localiser now; the edit is a separate question.

    python scripts/maven_surprisal.py --run results/runs/pilot_maven_d1_c1/base \
        --maps results/matrix/cellmaps/pilot_maven_d1_c1.npz \
        --out results/matrix/cellmaps/pilot_maven_d1_c1_surprisal.npz
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
VENDOR = REPO / "vendor" / "MAVEN"
sys.path.insert(0, str(VENDOR))
from cgm_train_sample import Affine, GatedTwoView                        # noqa: E402
from visual_transforms import DelayEmbed, PeriodFold                     # noqa: E402


def topk_jaccard(A, B, frac, invert_a=False, invert_b=False):
    k = max(int(round(frac * A.shape[1])), 1)
    sa = A if invert_a else -A
    sb = B if invert_b else -B
    ia = np.argpartition(sa, k - 1, axis=1)[:, :k]
    ib = np.argpartition(sb, k - 1, axis=1)[:, :k]
    out = np.empty(len(A))
    for i in range(len(A)):
        inter = len(np.intersect1d(ia[i], ib[i]))
        out[i] = inter / (2 * k - inter)
    return float(out.mean())


def spearman(a, b):
    ra = np.argsort(np.argsort(a, axis=1), axis=1).astype(np.float64)
    rb = np.argsort(np.argsort(b, axis=1), axis=1).astype(np.float64)
    ra -= ra.mean(1, keepdims=True); rb -= rb.mean(1, keepdims=True)
    num = (ra * rb).sum(1)
    den = np.sqrt((ra ** 2).sum(1) * (rb ** 2).sum(1))
    return float(np.mean(num / np.maximum(den, 1e-12)))


@torch.no_grad()
def impute(net, phi, I_obs, M_img, steps, dev, amp, gen, use_prior=True, clamp=True):
    """Integrate the FM ODE conditioned on the observed cells. Returns image-space x0.

    Three things here are corrections applied 2026-08-30 after a code review found the
    first version departed from training in ways that all attenuated the result:

    ENDPOINT NOISE is `phi.encode(randn(b,T,D))`, not i.i.d. image noise. Both `draw`
    (cgm_train_sample.py:163) and the training loop (:330, :347) build it that way. On the
    delay view with tau < m the two image copies of a time cell then carry IDENTICAL noise;
    i.i.d. image noise does not, and that redundancy is exactly what attention exploits.

    THE PRIOR is rebuilt in distribution rather than passed as None. Training visits only
    `(M=0, Xc=0, Xp=None)` and `(M!=0, Xc=M*I, Xp=(1-M)*prior)`, so `(M!=0, Xp=None)` is
    neither, and a 30k-step-trained `prior_proj` reads the resulting 0 as "mid-range", not
    as "no information". The training prior is `xtg - tg*vf_g`, a one-step denoise of a
    NOISED image at random t -- not the sample's own values, so rebuilding it leaks
    nothing. It costs one extra unconditional pass per step.

    OBSERVED CELLS ARE CLAMPED to the true interpolant. `recon(vf_i, vti, M_tgt)` masks the
    impute loss with the MASKED cells only, so the net's velocity at observed positions is
    supervised by nothing; integrating 200 steps over all tokens drives them somewhere
    arbitrary, and full self-attention mixes that into the masked tokens through
    `in_proj(x)`. Training never integrates -- it presents the true interpolant at observed
    cells -- so clamping is what reproduces training, and the first version's docstring had
    this exactly backwards.
    """
    b, D = I_obs.shape[0], I_obs.shape[-1]
    Xc = M_img * I_obs
    zero = torch.zeros_like(I_obs)
    eps = phi.encode(torch.randn(b, phi.T, D, device=dev, generator=gen))
    x = eps
    ts = torch.linspace(1.0, 0.0, steps + 1, device=dev)
    for i in range(steps):
        t = ts[i].expand(b)
        with torch.autocast("cuda", dtype=torch.bfloat16,
                            enabled=(amp == "bf16" and dev.type == "cuda")):
            if use_prior:
                _, vf_g = net(x, t, zero, zero, None)
                Xp = (1.0 - M_img) * (x - ts[i] * vf_g.float()).detach()
            else:
                Xp = None
            _, vf = net(x, t, M_img, Xc, Xp)
        x = x - vf.float() * (ts[i] - ts[i + 1])
        if clamp:
            tn = ts[i + 1]
            x = M_img * ((1.0 - tn) * I_obs + tn * eps) + (1.0 - M_img) * x
    return x


def jaccard_chance(F, frac, trials=4000, seed=0):
    """E[Jaccard] of two independent top-k sets. NOT k/(2F-k): that is E[num]/E[den],
    which runs ~4% low at frac=0.10 and would overstate every lift quoted against it."""
    k = max(int(round(frac * F)), 1)
    rng = np.random.default_rng(seed)
    acc = 0.0
    for _ in range(trials):
        a = rng.choice(F, k, replace=False)
        b = rng.choice(F, k, replace=False)
        inter = len(np.intersect1d(a, b))
        acc += inter / (2 * k - inter)
    return acc / trials


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run dir: samples.npy and ck/")
    ap.add_argument("--maps", default=None, help="npz from scripts/localise_cells.py")
    ap.add_argument("--milestone", type=int, default=None)
    ap.add_argument("--rounds", type=int, default=16)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--mask-lo", type=float, default=0.2)
    ap.add_argument("--mask-hi", type=float, default=0.6)
    ap.add_argument("--n", type=int, default=None, help="default: the maps' subsample")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--amp", default="bf16")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    run = Path(a.run); ck = run / "ck"
    cfg = json.loads((ck / "maven_config.json").read_text())
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    aff = Affine.from_dict(cfg["affine"])
    T, C = int(cfg["T"]), int(cfg["C"])

    S_all = np.asarray(np.load(run / "samples.npy", mmap_mode="r"), np.float32)
    if a.maps:
        z = np.load(a.maps)
        pick = z["idx"]
        raw_map, q_map = z["raw"], z["quantile"]
        print(f"maps={a.maps}  reusing its {len(pick)} sampled indices")
    else:
        rng = np.random.default_rng(a.seed)
        pick = np.sort(rng.choice(len(S_all), size=min(a.n or 1024, len(S_all)), replace=False))
        raw_map = q_map = None
    S = S_all[pick]
    n = len(S)

    # view A only; L_B comes from the config so the STFT never has to be refitted
    if cfg["view_a"] == "dayfold":
        phi_A = PeriodFold(T, int(cfg["fold_period"]))
    else:
        phi_A = DelayEmbed(T, tau=int(cfg["delay_tau"]), m=int(cfg["delay_m"]))
    phi_A = phi_A.to(dev)
    dims = [int(x) for x in str(cfg["dims"]).split(",")]
    model = GatedTwoView(C, int(cfg["L_A"]), int(cfg["L_B"]), dims, int(cfg["n_heads"])).to(dev)
    p = ck / (f"milestone-{a.milestone}.pt" if a.milestone else "final.pt")
    blob = torch.load(p, map_location=dev, weights_only=False)
    model.load_state_dict(blob["model"])
    ema = torch.optim.swa_utils.AveragedModel(model)
    ema.module.load_state_dict(blob["ema_module"])
    net = ema.module.net["A"].eval()
    print(f"loaded {p.name} (iter {blob.get('iter')})  view_a={cfg['view_a']} "
          f"L_A={cfg['L_A']}  n={n}  rounds={a.rounds} steps={a.steps}")

    X01 = torch.from_numpy(aff.forward(S)).to(dev)                 # (n,T,C) in [-1,1]
    gen = torch.Generator(device=dev).manual_seed(a.seed)
    err = np.zeros((n, T, C), np.float64)
    cnt = np.zeros((n, T, C), np.float64)

    t0 = time.time()
    for r in range(a.rounds):
        ratio = float(torch.empty(1).uniform_(a.mask_lo, a.mask_hi).item())
        for s in range(0, n, a.batch):
            e = min(s + a.batch, n)
            x0 = X01[s:e]
            keep = (torch.rand(x0.shape, device=dev, generator=gen) >= ratio).float()
            I_obs = phi_A.encode(x0)
            M_img = phi_A.encode(keep)          # exact: view A is cell-aligned
            xi = impute(net, phi_A, I_obs, M_img, a.steps, dev, a.amp, gen)
            rec = phi_A.decode(xi)                                  # (b,T,C) in [-1,1]
            d = ((rec - x0) ** 2 * (1.0 - keep)).double().cpu().numpy()
            err[s:e] += d
            cnt[s:e] += (1.0 - keep).double().cpu().numpy()
        print(f"  round {r+1}/{a.rounds}  ratio={ratio:.2f}  ({time.time()-t0:.0f}s)",
              flush=True)

    sur = (err / np.maximum(cnt, 1.0)).astype(np.float32)
    print(f"\nsurprisal: median {np.median(sur):.6f}  "
          f"p95 {np.percentile(sur,95):.6f}  cells never masked {int((cnt==0).sum())}")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, idx=pick, surprisal=sur, counts=cnt.astype(np.int32),
                        rounds=a.rounds, steps=a.steps, T=T, C=C)
    print(f"wrote {a.out}")

    if raw_map is not None:
        sf = sur.reshape(n, -1)
        print("\n=== does the mask head reproduce either model-free map?")
        print("  (Jaccard, with the lift over chance in brackets)")
        print("  frac   surprisal vs RAW      surprisal vs QUANTILE   RAW vs QUANTILE")
        for f in (0.10, 0.20, 0.30):
            ch = f / (2 - f)
            j1 = topk_jaccard(sf, raw_map, f)
            j2 = topk_jaccard(sf, q_map, f)
            j3 = topk_jaccard(raw_map, q_map, f)
            print(f"  {int(f*100):>3d}%   {j1:.3f} ({j1/ch:.2f}x)          "
                  f"{j2:.3f} ({j2/ch:.2f}x)           {j3:.3f} ({j3/ch:.2f}x)")
        print(f"\n  Spearman over cells   surprisal-RAW {spearman(sf, raw_map):+.3f}   "
              f"surprisal-QUANTILE {spearman(sf, q_map):+.3f}   "
              f"RAW-QUANTILE {spearman(raw_map, q_map):+.3f}")
        print("\n  A lift near 1.0 means the mask head is localising something ELSE than")
        print("  the leak -- most likely predictability, exactly as MODEL_DESIGN 2.3's")
        print("  zscore result predicts. That is a real answer, not a failed run.")


if __name__ == "__main__":
    main()
