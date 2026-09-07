#!/usr/bin/env python
"""MAVEN trained and sampled on one CGM cohort array, for the MIA pipeline.

WHY THIS FILE EXISTS
--------------------
Upstream ships two training scripts and our design needs one piece from each:

  train_maven_maskae.py  the masked-imputation objective (what we keep), but NO view
                         fusion -- it writes one .npy per branch and expects
                         fusion_nview.py to fuse them afterwards with a RETRIEVAL
                         editor that queries the training set per sample.
  train_maven.py         the learned per-channel fusion gate (what we keep), but not
                         the mask-conditioned AE velocity net.

We want maskae's objective with the base's gate, because the retrieval editor is the
most direct route from a training window into a released sample and we have demoted it
to a measured ablation. That combination is not an upstream configuration, so it is
built here. Everything else -- the transforms, the velocity net, the two-view container
-- is imported from the vendored tree unchanged.

THE THREE THINGS THAT ARE NOT UPSTREAM DEFAULTS, AND WHY
--------------------------------------------------------
1. `--mask_mode cell` is the default here, against upstream's `channel`. At C = 1
   channel masking is one coin flip per window: the loss is identically zero on ~60 %
   of draws and self-distillation on the rest. Cell masking hides 20-60 % of a window's
   tokens on every draw and is the missingness CGM actually has. See docs/MODEL_DESIGN
   1.9.1.

2. `--view_a dayfold` at T = 2016. Not elegance -- feasibility. The velocity net is
   full self-attention, quadratic in the token count; DelayEmbed gives L = 4024 at
   T = 2016 (~33 GB of attention matrix per layer at batch 128 in bf16) and the day
   fold gives L = 2016 (~8 GB). 2016 = 7*288 folds with no padding.

3. Cohort space in, cohort space out. Upstream min-max scales a CSV to [-1,1] and
   writes samples as (x+1)/2. Our attack measures ||real - synthetic|| in COHORT units,
   so a released set on a different scale makes every distance wrong. The affine here
   is fitted on the training array only, recorded in config.json, and inverted exactly
   on output. `vendor/DiM-TS/cgm_train_sample.py` does not rescale at all; this one has
   to, because MAVEN's noise endpoint is standard normal and our cohort has std ~0.19.

Contract, matching vendor/DiM-TS/cgm_train_sample.py so the two are interchangeable:
    --data_npy   (N,T,C) float32, cohort space
    --out_npy    (K,T,C) float32, cohort space
    --skip_train + --load_milestone M   resample from a stored checkpoint, no training
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from train_maven_maskae import (AEMaskVelocityNet, TwoViewMaskAE,   # noqa: E402
                                sample_image_mask, recon, lambda_at)
from visual_transforms import DelayEmbed, PeriodFold, STFTEmbedder  # noqa: E402


# ---------------------------------------------------------------- cohort scaling
class Affine:
    """Per-channel map cohort space <-> [-1,1], exactly invertible.

    Fitted on the TRAINING array only. Its whole dependence on the training set is 2C
    scalars, which is the same argument the fusion gate rests on: a global constant
    cannot carry an individual window's statistics into an individual released sample.
    Recorded in config.json so a resample reproduces it without refitting -- refitting
    on a different array would silently change the output scale.
    """

    def __init__(self, lo, hi):
        self.lo = np.asarray(lo, dtype=np.float64)
        self.hi = np.asarray(hi, dtype=np.float64)
        rng = self.hi - self.lo
        if not np.all(rng > 0):
            raise ValueError(f"channel with zero range: lo={self.lo}, hi={self.hi}")

    @classmethod
    def fit(cls, X):
        return cls(X.min(axis=(0, 1)), X.max(axis=(0, 1)))

    @classmethod
    def from_dict(cls, d):
        return cls(d["lo"], d["hi"])

    def to_dict(self):
        return {"lo": self.lo.tolist(), "hi": self.hi.tolist()}

    def forward(self, x):
        return ((x - self.lo) / (self.hi - self.lo) * 2.0 - 1.0).astype(np.float32)

    def inverse(self, z):
        # Deliberately NOT clipped to [lo, hi]. A sample landing outside the training
        # range is a fact about the model and the attack should see it; clamping would
        # pull outliers toward the training support, which is exactly the direction
        # that would flatter a privacy measurement.
        return ((z.astype(np.float64) + 1.0) / 2.0 * (self.hi - self.lo)
                + self.lo).astype(np.float32)


# ------------------------------------------------------------------- transforms
def build_views(X01, T, D, args):
    """View A is a gather map, view B the complex STFT. Both exactly invertible."""
    mode = args.view_a
    if mode == "auto":
        mode = "dayfold" if T % args.fold_period == 0 and T > args.fold_period else "delay"
    if mode == "dayfold":
        if T % args.fold_period:
            raise ValueError(
                f"--view_a dayfold needs T divisible by --fold_period; "
                f"T={T} % {args.fold_period} = {T % args.fold_period}. A padded fold "
                f"drops cells from every loss.")
        phi_A = PeriodFold(T, args.fold_period)
    elif mode == "delay":
        phi_A = DelayEmbed(T, tau=args.delay_tau, m=args.delay_m)
    else:
        raise ValueError(f"unknown --view_a {mode!r}")
    phi_B = STFTEmbedder(T, D, n_fft=args.stft_n_fft, hop_length=args.stft_hop)
    phi_B.fit(X01)
    return phi_A, phi_B, mode


class GatedTwoView(nn.Module):
    """TwoViewMaskAE plus the per-channel fusion gate ported from train_maven.py.

    The gate is C scalars through a sigmoid. It is fitted against real batches -- it is
    NOT independent of the training data and this file does not pretend otherwise --
    but C global scalars cannot carry one window's statistics into one released sample,
    which a per-sample retrieval index can. That is the whole reason it is here instead
    of fusion_nview.py's editor.
    """

    def __init__(self, D, L_A, L_B, dims, n_heads):
        super().__init__()
        self.two = TwoViewMaskAE(D, L_A, L_B, dims, n_heads)
        self.gate_logit = nn.Parameter(torch.zeros(D))

    @property
    def net(self):
        return self.two.net

    def gate(self):
        return torch.sigmoid(self.gate_logit)


# ---------------------------------------------------------------------- sampling
@torch.no_grad()
def draw(model, phi, T, D, n, dev, steps, batch, amp, generator):
    """Euler-integrate both views from shared time-domain noise, then gate-fuse."""
    model.eval()
    g = model.gate().view(1, 1, D)
    out, done = [], 0
    while done < n:
        b = min(batch, n - done)
        # ONE noise draw shared by both views: they are two views of one sample, not
        # two independent samples that happen to be averaged.
        eps = torch.randn(b, T, D, device=dev, generator=generator)
        dec = {}
        for k in ("A", "B"):
            ph, net = phi[k], model.net[k]
            x = ph.encode(eps)
            zero = torch.zeros_like(x)
            ts = torch.linspace(1.0, 0.0, steps + 1, device=dev)
            for i in range(steps):
                with torch.autocast("cuda", dtype=torch.bfloat16,
                                    enabled=(amp == "bf16" and dev.type == "cuda")):
                    _, vf = net(x, ts[i].expand(b), zero, zero, None)
                x = x - vf.float() * (ts[i] - ts[i + 1])
            dec[k] = ph.decode(x)
        out.append((g * dec["A"] + (1.0 - g) * dec["B"]).cpu().numpy())
        done += b
    model.train()
    return np.concatenate(out, 0).astype(np.float32)


# -------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_npy", required=True)
    ap.add_argument("--out_npy", required=True)
    ap.add_argument("--results_folder", required=True)
    ap.add_argument("--K", type=int, default=0, help="0 = training-set size")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--gpu", default="0")
    # architecture
    ap.add_argument("--dims", default="128,128,64,64,64,64,128,128")
    ap.add_argument("--n_heads", type=int, default=8)
    # views
    ap.add_argument("--view_a", default="auto", choices=["auto", "delay", "dayfold"])
    ap.add_argument("--fold_period", type=int, default=288, help="one day at 5-min sampling")
    ap.add_argument("--delay_tau", type=int, default=4)
    ap.add_argument("--delay_m", type=int, default=8)
    # 96 samples = 8 h at 5-min sampling, against upstream's 16 = 80 min, which puts the
    # postprandial limb, the overnight drift and the circadian component in the DC bin.
    # hop must stay n_fft/2: Hann at 50% overlap is what makes the inverse exact.
    ap.add_argument("--stft_n_fft", type=int, default=96)
    ap.add_argument("--stft_hop", type=int, default=48)
    # objective
    ap.add_argument("--mask_mode", default="cell", choices=["channel", "cell"])
    ap.add_argument("--mask_lo", type=float, default=0.2)
    ap.add_argument("--mask_hi", type=float, default=0.6)
    ap.add_argument("--coarse_weight", type=float, default=1.0)
    ap.add_argument("--fuse_alpha", type=float, default=0.1)
    ap.add_argument("--lambda_schedule", default="100000:2")
    # optimisation
    ap.add_argument("--total_iters", type=int, default=100000)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--micro_batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--ema_decay", type=float, default=0.999)
    ap.add_argument("--amp", default="bf16", choices=["off", "bf16"])
    # sampling / checkpoints
    ap.add_argument("--sampling_steps", type=int, default=200)
    ap.add_argument("--sample_batch", type=int, default=64)
    ap.add_argument("--save_cycle", type=int, default=10000)
    ap.add_argument("--load_milestone", type=int, default=None)
    ap.add_argument("--skip_train", action="store_true")
    ap.add_argument("--log_every", type=int, default=100)
    a = ap.parse_args()

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", a.gpu)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(a.seed)
    np.random.seed(a.seed % 2 ** 32)
    if a.amp == "bf16" and dev.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    ck = Path(a.results_folder)
    ck.mkdir(parents=True, exist_ok=True)
    cfg_path = ck / "maven_config.json"

    X = np.load(a.data_npy).astype(np.float32)          # (N,T,C), cohort space
    if X.ndim != 3:
        raise ValueError(f"--data_npy must be (N,T,C); got {X.shape}")
    N, T, D = X.shape

    # The affine is fitted ONCE, on the first (training) invocation, and reloaded on
    # every resample. Refitting on a resample would work silently and put the samples
    # on a different scale than the run that trained them.
    if a.skip_train:
        if not cfg_path.exists():
            raise FileNotFoundError(
                f"{cfg_path} is missing: a resample cannot reconstruct the scaling or "
                f"the architecture of the run it is resampling, and guessing them "
                f"would fail as an opaque shape error.")
        saved = json.loads(cfg_path.read_text())
        aff = Affine.from_dict(saved["affine"])
        for k in ("dims", "view_a", "fold_period", "delay_tau", "delay_m",
                  "stft_n_fft", "stft_hop", "n_heads"):
            setattr(a, k, saved[k])
    else:
        aff = Affine.fit(X)

    X01 = aff.forward(X)
    dims = [int(x) for x in str(a.dims).split(",")]
    phi_A, phi_B, view_a_used = build_views(X01, T, D, a)
    phi = {"A": phi_A.to(dev), "B": phi_B.to(dev)}
    model = GatedTwoView(D, phi_A.L, phi_B.L, dims, a.n_heads).to(dev)
    ema = torch.optim.swa_utils.AveragedModel(
        model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(decay=a.ema_decay))

    n_par = sum(p.numel() for p in model.parameters())
    print(f"[maven] {X.shape} cohort[{X.min():.4f},{X.max():.4f}] -> [-1,1]  "
          f"view_a={view_a_used} L_A={phi_A.L} L_B={phi_B.L}  "
          f"mask={a.mask_mode}  params={n_par/1e6:.2f}M", flush=True)

    if not a.skip_train:
        cfg_path.write_text(json.dumps(
            {"affine": aff.to_dict(), "dims": a.dims, "view_a": view_a_used,
             "fold_period": a.fold_period, "delay_tau": a.delay_tau,
             "delay_m": a.delay_m, "stft_n_fft": a.stft_n_fft, "stft_hop": a.stft_hop,
             "n_heads": a.n_heads, "mask_mode": a.mask_mode, "seed": a.seed,
             "total_iters": a.total_iters, "T": T, "C": D, "N": N,
             "L_A": phi_A.L, "L_B": phi_B.L, "params": n_par}, indent=1))

    # -------------------------------------------------------------- resume/restore
    if a.load_milestone is not None:
        p = ck / f"milestone-{int(a.load_milestone)}.pt"
        if not p.exists():
            avail = sorted(int(q.stem.split("-")[1]) for q in ck.glob("milestone-*.pt"))
            raise FileNotFoundError(f"{p} not found; milestones present: {avail}")
        blob = torch.load(p, map_location=dev, weights_only=False)
        ema.module.load_state_dict(blob["ema_module"])
        model.load_state_dict(blob["model"])
        print(f"[maven] loaded {p.name} (iter {blob.get('iter')})", flush=True)
    elif a.skip_train:
        p = ck / "final.pt"
        if not p.exists():
            raise FileNotFoundError(f"--skip_train with no --load_milestone needs {p}")
        blob = torch.load(p, map_location=dev, weights_only=False)
        ema.module.load_state_dict(blob["ema_module"])
        model.load_state_dict(blob["model"])
        print(f"[maven] loaded {p.name} (iter {blob.get('iter')})", flush=True)

    # -------------------------------------------------------------------- training
    if not a.skip_train:
        data_t = torch.from_numpy(X01).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
        sched = [tuple(map(float, p.split(":"))) for p in a.lambda_schedule.split(",")]
        sched = [(int(b), v) for b, v in sched]
        micro = min(a.micro_batch_size, a.batch_size)
        t0 = time.time()
        losses = []
        for it in range(a.total_iters):
            lam = lambda_at(it, sched)
            opt.zero_grad(set_to_none=True)
            tot, nseen = 0.0, 0
            while nseen < a.batch_size:
                b = min(micro, a.batch_size - nseen)
                idx = torch.randint(0, N, (b,), device=dev)
                x0 = data_t[idx]
                loss_acc = x0.new_zeros(())
                x0hat = {}
                for k in ("A", "B"):
                    ph, net = phi[k], model.net[k]
                    I = ph.encode(x0)
                    valid = ph.image_valid(b, D, dev)
                    zero = torch.zeros_like(I)
                    # GEN pass, unconditional
                    x1g = ph.encode(torch.randn_like(x0))
                    tg = torch.rand(b, device=dev)
                    xtg = (1 - tg)[:, None, None] * I + tg[:, None, None] * x1g
                    vtg = x1g - I
                    with torch.autocast("cuda", dtype=torch.bfloat16,
                                        enabled=(a.amp == "bf16" and dev.type == "cuda")):
                        vc_g, vf_g = net(xtg, tg, zero, zero, None)
                        L_gen = (recon(vf_g, vtg, valid)
                                 + a.coarse_weight * recon(vc_g, vtg, valid))
                    prior = (xtg - tg[:, None, None] * vf_g.float()).detach()
                    x0hat[k] = ph.decode(prior)
                    # IMPUTE pass. mask_mode='cell' is what makes this non-degenerate
                    # at C = 1; see the module docstring.
                    M_cond, M_tgt = sample_image_mask(b, I.shape[1], D, dev,
                                                      a.mask_mode, a.mask_lo, a.mask_hi)
                    M_cond, M_tgt = M_cond * valid, M_tgt * valid
                    x1i = ph.encode(torch.randn_like(x0))
                    ti = torch.rand(b, device=dev)
                    xti = (1 - ti)[:, None, None] * I + ti[:, None, None] * x1i
                    vti = x1i - I
                    Xc, Xp = M_cond * I, (1.0 - M_cond) * prior
                    with torch.autocast("cuda", dtype=torch.bfloat16,
                                        enabled=(a.amp == "bf16" and dev.type == "cuda")):
                        vc_i, vf_i = net(xti, ti, M_cond, Xc, Xp)
                        L_imp = (recon(vf_i, vti, M_tgt)
                                 + a.coarse_weight * recon(vc_i, vti, M_tgt))
                    loss_acc = loss_acc + (L_imp + lam * L_gen)
                # Fusion gate, both branches detached: this trains the C gate scalars
                # and nothing else. Detaching is what keeps it from back-propagating a
                # reconstruction signal into the velocity nets.
                g = model.gate().view(1, 1, D)
                fused = g * x0hat["A"].detach() + (1.0 - g) * x0hat["B"].detach()
                loss_acc = loss_acc + a.fuse_alpha * F.mse_loss(fused, x0)
                w = b / a.batch_size
                (loss_acc * w).backward()
                tot += float(loss_acc.item()) * w
                nseen += b
            if a.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), a.grad_clip)
            opt.step()
            ema.update_parameters(model)
            losses.append(tot)
            if it == 0 or (it + 1) % a.log_every == 0:
                gv = model.gate().detach().cpu().numpy()
                print(f"[maven] iter {it+1}/{a.total_iters} ({time.time()-t0:.0f}s) "
                      f"loss={np.mean(losses[-a.log_every:]):.5f} lam={lam} "
                      f"gate={np.array2string(gv, precision=3)}", flush=True)
            # WEIGHT milestones, not sample milestones. Upstream's --milestone_every
            # writes samples only, which cannot be resampled from later; the whole
            # risk-versus-training-length curve needs weights it can rewind to.
            if a.save_cycle and (it + 1) % a.save_cycle == 0:
                m = (it + 1) // a.save_cycle
                torch.save({"model": model.state_dict(),
                            "ema_module": ema.module.state_dict(),
                            "iter": it + 1}, ck / f"milestone-{m}.pt")
                print(f"[maven] milestone-{m} @ iter {it+1}", flush=True)
        torch.save({"model": model.state_dict(),
                    "ema_module": ema.module.state_dict(),
                    "iter": a.total_iters}, ck / "final.pt")

    # -------------------------------------------------------------------- sampling
    K = int(a.K) or N
    gen = torch.Generator(device=dev)
    gen.manual_seed(a.seed + 1)
    t0 = time.time()
    Z = draw(ema.module, phi, T, D, K, dev, a.sampling_steps, a.sample_batch, a.amp, gen)
    S = aff.inverse(Z)
    Path(a.out_npy).parent.mkdir(parents=True, exist_ok=True)
    np.save(a.out_npy, S)
    print(f"[maven] sampled {S.shape} in {time.time()-t0:.0f}s -> {a.out_npy}\n"
          f"[maven] cohort real [{X.min():.4f},{X.max():.4f}]  "
          f"synthetic [{S.min():.4f},{S.max():.4f}]", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
