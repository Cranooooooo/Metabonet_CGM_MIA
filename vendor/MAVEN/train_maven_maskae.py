#!/usr/bin/env python3
"""MAVEN-MaskAE — multi-resolution autoencoder mask-FM generator (latest design).

MAVEN lifts each multivariate time-series window into TWO image views via
exactly-invertible transforms (delay embedding + complex STFT) and runs per-view
flow matching (FM) in image space; the two views are fused post-hoc by the
retrieval-guided editor (`fusion_nview.py`). This file is the **latest model
design**: each view's velocity network is a *mask-conditioned autoencoder
transformer* with an explicit masked-imputation auxiliary task.

Per view, the velocity net (`AEMaskVelocityNet`) has an autoencoder hidden-dim
schedule, default `[128,128,64,64,64,64,128,128]`: the width shrinks to a 64-d
bottleneck (the COARSE representation) then expands back to 128 (the FINE
representation). It exposes TWO output heads — one at the last bottleneck (64)
block (coarse) and one at the final (128) block (fine) — and the velocity
reconstruction loss is computed at BOTH heads (deep supervision at coarse + fine).

Per training step, per view, two forward passes (IG-FM style):
  GEN pass    : unconditional FM on the image tokens -> soft prior x0_pred_g.
  IMPUTE pass : mask whole CHANNELS of the image, then regenerate the masked
                channels conditioned on the observed channels (X_cond) and the
                gen soft prior for the masked ones (X_prior). L_imp is computed
                ONLY over the masked cells -> forces inter-channel structure.
  L_view = [L_imp@fine + cw*L_imp@coarse] + lambda * [L_gen@fine + cw*L_gen@coarse]

Sampling: unconditional (mask=0), integrate the FM ODE using the FINE head, decode
each view to TS, then fuse the two views with `fusion_nview.py` (editor).

Masking is applied at the input (before the first block); tokens (= image
positions) are preserved through the dim changes, so the masked cells M_target
are known at every block, including the 64-d bottleneck.

Usage (one view-pair on one dataset):
  python -u train_maven_maskae.py --dataset GENERAL_Energy --name maskae_energy \
      --gpu 0 --window 64 --dims 128,128,64,64,64,64,128,128 --amp bf16
Then fuse the two saved per-view fakes with fusion_nview.py and evaluate with
Eval_Assembly/.  See DESIGN.md for the design rationale and the ablation ladder.
"""
import argparse, os, sys, time, math
def _peek_gpu():
    for i, a in enumerate(sys.argv[1:], 1):
        if a == '--gpu' and i + 1 < len(sys.argv): return sys.argv[i + 1]
        if a.startswith('--gpu='): return a.split('=', 1)[1]
    return None
_g = _peek_gpu()
if _g is not None: os.environ['CUDA_VISIBLE_DEVICES'] = _g
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import numpy as np, torch, torch.nn as nn
from train_maven import load_dataset_csv, SinusoidalTime
from visual_transforms import build_transforms
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


# ============================================================
# Mask-conditioned autoencoder velocity net with two heads (coarse + fine)
# ============================================================
class AEMaskVelocityNet(nn.Module):
    """FM velocity predictor on image tokens [B, L, D] with IG-FM mask conditioning
    and an autoencoder hidden-dim schedule. Two output heads: `head_coarse` at the
    last bottleneck (min-width) block, `head_fine` at the final block. Both project
    back to D and predict the linear-path velocity (target = x1 - x0)."""

    def __init__(self, feature_size, dims, n_heads=8, seq_len=120):
        super().__init__()
        d0 = dims[0]
        self.in_proj = nn.Linear(feature_size, d0)
        self.mask_proj = nn.Linear(feature_size, d0)          # mask indicator -> hidden
        self.cond_proj = nn.Linear(feature_size, d0, bias=False)   # observed values
        self.prior_proj = nn.Linear(feature_size, d0, bias=False)  # gen soft prior
        # zero-init conditioning so the net == unconditional FM at init
        nn.init.zeros_(self.cond_proj.weight); nn.init.zeros_(self.prior_proj.weight)
        self.pos_emb = nn.Parameter(torch.zeros(1, seq_len, d0)); nn.init.trunc_normal_(self.pos_emb, std=0.02)
        self.time_emb = SinusoidalTime(128)

        self.projs = nn.ModuleList(); self.blocks = nn.ModuleList(); self.tmlps = nn.ModuleList()
        prev = d0
        for d in dims:
            self.projs.append(nn.Linear(prev, d) if d != prev else nn.Identity())
            nh = n_heads if d % n_heads == 0 else max(h for h in (8, 4, 2, 1) if d % h == 0)
            self.blocks.append(nn.TransformerEncoderLayer(
                d_model=d, nhead=nh, dim_feedforward=d * 4, activation='gelu',
                batch_first=True, norm_first=True, dropout=0.0))
            self.tmlps.append(nn.Sequential(nn.Linear(128, d), nn.SiLU(), nn.Linear(d, d)))
            prev = d
        bottleneck = min(dims)
        self.idx_coarse = max(i for i, d in enumerate(dims) if d == bottleneck)  # last bottleneck block
        self.idx_fine = len(dims) - 1                                            # final block
        self.head_coarse = nn.Sequential(nn.LayerNorm(dims[self.idx_coarse]), nn.Linear(dims[self.idx_coarse], feature_size))
        self.head_fine = nn.Sequential(nn.LayerNorm(dims[self.idx_fine]), nn.Linear(dims[self.idx_fine], feature_size))

    def forward(self, x, t, M_cond, X_cond, X_prior=None):
        h = self.in_proj(x) + self.mask_proj(M_cond) + self.cond_proj(X_cond)
        if X_prior is not None:
            h = h + self.prior_proj(X_prior)
        h = h + self.pos_emb[:, :x.shape[1]]
        tfeat = self.time_emb(t)
        out_coarse = None
        for i, (proj, blk, tmlp) in enumerate(zip(self.projs, self.blocks, self.tmlps)):
            h = proj(h)
            h = h + tmlp(tfeat)[:, None, :]
            h = blk(h)
            if i == self.idx_coarse:
                out_coarse = self.head_coarse(h)
        return out_coarse, self.head_fine(h)


class TwoViewMaskAE(nn.Module):
    """One AEMaskVelocityNet per visual view (A = delay image, B = STFT image)."""
    def __init__(self, D, L_A, L_B, dims, n_heads):
        super().__init__()
        self.net = nn.ModuleDict({
            'A': AEMaskVelocityNet(D, dims, n_heads=n_heads, seq_len=L_A),
            'B': AEMaskVelocityNet(D, dims, n_heads=n_heads, seq_len=L_B)})


# ============================================================
# Helpers: masking, loss, lambda schedule
# ============================================================
def sample_image_mask(B, L, D, device, mode='channel', lo=0.2, hi=0.6):
    """Return (M_cond, M_target), both [B,L,D], 1=present. `channel` masks whole
    channels (the D axis) across all tokens -> forces inter-channel reconstruction."""
    ratio = float(torch.empty(1).uniform_(lo, hi).item())
    if mode == 'channel':
        pick = torch.bernoulli(torch.full((B, 1, D), ratio, device=device)).expand(B, L, D)
    else:
        pick = torch.bernoulli(torch.full((B, L, D), ratio, device=device))
    return 1.0 - pick, pick      # M_cond (observed), M_target (masked)


def recon(v_pred, v_tgt, mask):
    return ((v_pred - v_tgt) ** 2 * mask).sum() / mask.sum().clamp(min=1.0)


def lambda_at(it, sched):
    for b, v in sched:
        if it < b: return float(v)
    return float(sched[-1][1])


# ============================================================
# Main
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True); ap.add_argument('--name', required=True)
    ap.add_argument('--gpu', default='0'); ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--window', type=int, default=64)
    ap.add_argument('--dims', default='128,128,64,64,64,64,128,128',
                    help='autoencoder hidden-dim schedule (bottleneck=min, fine=last)')
    ap.add_argument('--total_iters', type=int, default=120000)
    ap.add_argument('--batch_size', type=int, default=256); ap.add_argument('--micro_batch_size', type=int, default=128)
    ap.add_argument('--lr', type=float, default=2e-4); ap.add_argument('--weight_decay', type=float, default=1e-4)
    ap.add_argument('--grad_clip', type=float, default=1.0); ap.add_argument('--ema_decay', type=float, default=0.999)
    ap.add_argument('--n_heads', type=int, default=8)
    # one transform setting per view (coarse/fine come from the AE, not the transform)
    ap.add_argument('--delay_tau', type=int, default=4); ap.add_argument('--delay_m', type=int, default=8)
    ap.add_argument('--stft_n_fft', type=int, default=16); ap.add_argument('--stft_hop', type=int, default=8)
    ap.add_argument('--mask_mode', default='channel', choices=['channel', 'cell'])
    ap.add_argument('--mask_lo', type=float, default=0.2); ap.add_argument('--mask_hi', type=float, default=0.6)
    ap.add_argument('--lambda_schedule', default='120000:2',
                    help='comma iter:lambda boundaries (single = constant); e.g. 40000:1,80000:2,120000:4')
    ap.add_argument('--coarse_weight', type=float, default=1.0, help='weight on the coarse (bottleneck) deep-supervision head')
    ap.add_argument('--amp', default='bf16', choices=['off', 'bf16'])
    ap.add_argument('--sampling_steps', type=int, default=200); ap.add_argument('--sample_batch', type=int, default=128)
    ap.add_argument('--num_samples', type=int, default=None)
    ap.add_argument('--milestone_every', type=int, default=30000); ap.add_argument('--milestone_steps', type=int, default=100)
    ap.add_argument('--out_dir', default=os.path.join(PROJECT_ROOT, 'outputs'))
    ap.add_argument('--checkpoint_dir', default=os.path.join(PROJECT_ROOT, 'checkpoints'))
    ap.add_argument('--log_every', type=int, default=500)
    args = ap.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    if args.amp == 'bf16':
        torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision('high')
    sched = []
    for p in args.lambda_schedule.split(','):
        b, v = p.split(':'); sched.append((int(b), float(v)))
    dims = [int(x) for x in args.dims.split(',')]

    data, _ = load_dataset_csv(args.dataset, window=args.window, seed=args.seed)
    N, T, D = data.shape; data_t = torch.from_numpy(data).to(dev)
    phi_A, phi_B = build_transforms(data, T, tau=args.delay_tau, m=args.delay_m, period=None,
                                    branch_b='stft', n_fft=args.stft_n_fft, hop_length=args.stft_hop)
    phi = {'A': phi_A.to(dev), 'B': phi_B.to(dev)}
    model = TwoViewMaskAE(D, phi_A.L, phi_B.L, dims, args.n_heads).to(dev)
    print(f'[maskae] {args.dataset} {data.shape} dims={dims} params={sum(p.numel() for p in model.parameters())/1e6:.2f}M '
          f'L_A={phi_A.L} L_B={phi_B.L} coarse@blk{model.net["A"].idx_coarse} fine@blk{model.net["A"].idx_fine}', flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    ema = torch.optim.swa_utils.AveragedModel(model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(decay=args.ema_decay))

    cell_dir = os.path.join(args.out_dir, args.name); os.makedirs(cell_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    ds_lc = args.dataset.lower().replace('general_', '')
    truth_path = os.path.join(cell_dir, f'{ds_lc}_norm_truth_{args.window}_train.npy')
    if not os.path.exists(truth_path):
        np.save(truth_path, ((data + 1.0) / 2.0).astype(np.float32))
    micro = min(args.micro_batch_size, args.batch_size)

    @torch.no_grad()
    def sample_save(emod, tag, n_steps):
        emod.eval()
        for b in ('A', 'B'):
            ph = phi[b]; net = emod.net[b]; outs = []; done = 0; nS = args.num_samples or N
            while done < nS:
                bb = min(args.sample_batch, nS - done); x = ph.encode(torch.randn(bb, T, D, device=dev)); zero = torch.zeros_like(x)
                ts = torch.linspace(1.0, 0.0, n_steps + 1, device=dev)
                for i in range(n_steps):
                    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=(args.amp == 'bf16')):
                        _, vf = net(x, ts[i].expand(bb), zero, zero, None)   # FINE head
                    x = x - vf.float() * (ts[i] - ts[i + 1])
                outs.append(ph.decode(x).cpu().numpy()); done += bb
            out = np.concatenate(outs, 0).astype(np.float32)
            suff = '' if tag == 'FINAL' else f'_{tag}'
            np.save(os.path.join(cell_dir, f'{args.name}{suff}_branch{b}_fake.npy'), ((out + 1.0) / 2.0).astype(np.float32))
        model.train()

    losses = []; t0 = time.time(); model.train()
    for it in range(args.total_iters):
        lam = lambda_at(it, sched); opt.zero_grad(set_to_none=True); tot = 0.0; nseen = 0
        while nseen < args.batch_size:
            bsz = min(micro, args.batch_size - nseen); idx = torch.randint(0, N, (bsz,), device=dev); x0 = data_t[idx]
            loss_acc = x0.new_zeros(())
            for b in ('A', 'B'):
                ph = phi[b]; net = model.net[b]; I_b = ph.encode(x0); valid = ph.image_valid(bsz, D, dev); zero = torch.zeros_like(I_b)
                # GEN pass (unconditional)
                x1g = ph.encode(torch.randn_like(x0)); tg = torch.rand(bsz, device=dev)
                xtg = (1 - tg)[:, None, None] * I_b + tg[:, None, None] * x1g; vtg = x1g - I_b
                with torch.autocast('cuda', dtype=torch.bfloat16, enabled=(args.amp == 'bf16')):
                    vc_g, vf_g = net(xtg, tg, zero, zero, None)
                    L_gen = recon(vf_g, vtg, valid) + args.coarse_weight * recon(vc_g, vtg, valid)
                x0g = (xtg - tg[:, None, None] * vf_g.float()).detach()      # soft prior
                # IMPUTE pass (mask whole channels)
                M_cond, M_tgt = sample_image_mask(bsz, I_b.shape[1], D, dev, args.mask_mode, args.mask_lo, args.mask_hi)
                M_cond = M_cond * valid; M_tgt = M_tgt * valid
                x1i = ph.encode(torch.randn_like(x0)); ti = torch.rand(bsz, device=dev)
                xti = (1 - ti)[:, None, None] * I_b + ti[:, None, None] * x1i; vti = x1i - I_b
                Xc = M_cond * I_b; Xp = (1.0 - M_cond) * x0g
                with torch.autocast('cuda', dtype=torch.bfloat16, enabled=(args.amp == 'bf16')):
                    vc_i, vf_i = net(xti, ti, M_cond, Xc, Xp)
                    L_imp = recon(vf_i, vti, M_tgt) + args.coarse_weight * recon(vc_i, vti, M_tgt)
                loss_acc = loss_acc + (L_imp + lam * L_gen)
            w = bsz / args.batch_size; (loss_acc * w).backward(); tot += float(loss_acc.item()) * w; nseen += bsz
        if args.grad_clip > 0: torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step(); ema.update_parameters(model); losses.append(tot)
        if it == 0 or (it + 1) % args.log_every == 0:
            print(f"[maskae] iter {it+1}/{args.total_iters} ({time.time()-t0:.0f}s) loss={np.mean(losses[-args.log_every:]):.5f} lam={lam}", flush=True)
        if args.milestone_every > 0 and (it + 1) % args.milestone_every == 0 and (it + 1) < args.total_iters:
            sample_save(ema.module, f'M{(it+1)//1000}K', args.milestone_steps); print(f'[maskae] milestone @ {it+1}', flush=True)

    torch.save({'model': model.state_dict(), 'ema_module': ema.module.state_dict(), 'iter': args.total_iters, 'config': vars(args)},
               os.path.join(args.checkpoint_dir, f'{args.name}.pt'))
    sample_save(ema.module, 'FINAL', args.sampling_steps); print('[maskae] done', flush=True)


if __name__ == '__main__':
    main()
