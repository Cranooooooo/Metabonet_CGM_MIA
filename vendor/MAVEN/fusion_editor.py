"""JAVELIN-style retrieval-guided fusion editor for MAVEN **V2**.

Post-hoc fusion of the two FROZEN branches' *paired* (shared-noise) outputs.
Direct descendant of V1's iter-4 `fusion_editor.py` (which beat DiMTS 39-59%
on KDDCup C-FID/CC/Disc). See `JAVELIN_FUSION_PLAN.md` and the V1
`fusion_editor_log.md` for the full design rationale.

Core editor (unchanged from V1 iter-4, the converged form):

    fused = clamp(x_base + α · smooth(V(h)) Uᵀ · x_base(1−x_base), 0, 1)

  - rank-r channel-coupled residual on the base branch (U shared basis →
    preserves cross-correlation; never per-channel),
  - bounded-skew gate `x_base(1−x_base)` (vanishes at [0,1] walls),
  - cap-and-decay cosine α schedule (stays in the small-α C-FID basin),
  - depthwise Gaussian time-smoothing of V (kills high-freq jitter),
  - loss = MMD + 0.5·Corr + λ_anc·MSE(·, x_base) + pred_w·TSTR-MAML,
  - TS2Vec-space K-NN retrieval, query = base branch, cached once.

**V2 generalization vs V1:** the *base/anchor branch is selectable* via
`--base_branch {A,B}`. V1 hard-coded B because branchB was strong on KDDCup;
on Stocks branchA (delay-embed) is the strong branch, so the editor must
anchor on whichever branch wins for that dataset. The base is the anchor +
the retrieval query; the other branch enters only as the divergence
direction `x_other − x_base`.

Usage:
    python fusion_editor.py --name maven_stocks_seed1 --base_branch A [--epochs 50]
"""
import argparse
import glob
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
from torch.func import functional_call

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'Eval_Assembly'))


# ============================================================
# Editor: rank-r residual + bounded-skew + warm-up/decay α
# ============================================================
class FusionEditor(nn.Module):
    """fused = clamp(x_base + α · smooth(V(h)) Uᵀ · x_base(1−x_base), 0, 1).

    Inputs concatenated per cell as `[x_base, x_other − x_base,
    n_mean − x_base, n_std, n_med − x_base]` (5·D channels). The 5th block
    is the neighbour *median trajectory* — preserves temporal coherence the
    marginal mean/std collapse over.
    """

    def __init__(self, T, D, hidden=128, n_layers=2, n_heads=4, rank=8,
                 alpha_max=0.003, warmup_frac=0.15, alpha_end_frac=0.25,
                 smooth_sigma=3.0, dropout=0.0, bounded_skew=True):
        super().__init__()
        self.bounded_skew = bool(bounded_skew)          # ablation: domain gate
        self.T, self.D, self.r = int(T), int(D), int(rank)
        self.in_proj = nn.Linear(5 * D, hidden)
        self.pos = nn.Parameter(torch.zeros(1, T, hidden))
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=n_heads, dim_feedforward=hidden * 4,
            activation='gelu', batch_first=True, norm_first=True, dropout=dropout,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm_out = nn.LayerNorm(hidden)
        self.V_head = nn.Linear(hidden, self.r)
        self.U = nn.Parameter(torch.empty(D, self.r))
        nn.init.trunc_normal_(self.U, std=1.0 / (D ** 0.5))

        # α schedule + magnitude
        self.alpha_param = nn.Parameter(torch.tensor(0.0))
        self.alpha_max = float(alpha_max)
        self.warmup_frac = float(warmup_frac)
        self.alpha_end_frac = float(alpha_end_frac)
        self.register_buffer('_step', torch.zeros((), dtype=torch.long))
        self.register_buffer('_total_steps', torch.tensor(1, dtype=torch.long))

        # Fixed depthwise Gaussian smoother on V (low-pass over time).
        ks = int(2 * round(3 * smooth_sigma) + 1)
        t = torch.arange(ks) - ks // 2
        k = torch.exp(-(t.float() ** 2) / (2 * smooth_sigma ** 2))
        k = (k / k.sum()).view(1, 1, ks).repeat(self.r, 1, 1)         # [r,1,ks]
        self.register_buffer('smooth_k', k)
        self.smooth_pad = ks // 2

    def set_schedule(self, total_steps):
        self._total_steps.fill_(int(total_steps))

    def _alpha(self):
        frac = (self._step.float() / self._total_steps.float()).clamp(0.0, 1.0)
        wf, ef = self.warmup_frac, self.alpha_end_frac
        pi = torch.tensor(math.pi, device=frac.device)
        if frac.item() <= wf:
            w = (frac / wf).clamp(0.0, 1.0)
            sched = 0.5 * (1.0 - torch.cos(pi * w))                   # 0 -> 1
        else:
            w = ((frac - wf) / max(1e-6, 1.0 - wf)).clamp(0.0, 1.0)
            sched = ef + (1.0 - ef) * 0.5 * (1.0 + torch.cos(pi * w))  # 1 -> ef
        zero = torch.zeros_like(self.alpha_param)
        gain = F.softplus(self.alpha_param) / F.softplus(zero)
        return self.alpha_max * sched * gain

    def forward(self, x_base, x_other, n_mean, n_std, n_med):
        h_in = torch.cat([x_base, x_other - x_base, n_mean - x_base, n_std,
                          n_med - x_base], dim=-1)
        h = self.in_proj(h_in) + self.pos
        h = self.norm_out(self.encoder(h))
        V = self.V_head(h)                                            # [B,T,r]
        # depthwise low-pass over time, per rank
        V_s = F.conv1d(V.transpose(1, 2), self.smooth_k,
                       padding=self.smooth_pad, groups=self.r).transpose(1, 2)
        corr = V_s @ self.U.t()                                       # [B,T,D]
        if self.training:
            self._step += 1
        alpha = self._alpha()
        bound = x_base * (1.0 - x_base) if self.bounded_skew else 1.0
        fused = (x_base + alpha * corr * bound).clamp(0.0, 1.0)
        return fused, alpha, corr


# ============================================================
# Losses
# ============================================================
def mmd2(x, y, sigmas=(1.0, 2.0, 4.0, 8.0, 16.0)):
    x = x.reshape(x.size(0), -1)
    y = y.reshape(y.size(0), -1)
    xx = torch.cdist(x, x).pow(2)
    yy = torch.cdist(y, y).pow(2)
    xy = torch.cdist(x, y).pow(2)
    out = x.new_zeros(())
    for s in sigmas:
        out = out + (
            torch.exp(-xx / (2 * s))
            + torch.exp(-yy / (2 * s))
            - 2 * torch.exp(-xy / (2 * s))
        ).mean()
    return out


def corr_loss(f, r):
    def corr(z):
        z = z.reshape(-1, z.size(-1))
        z = z - z.mean(0, keepdim=True)
        s = z.std(0, keepdim=True).clamp(min=1e-6)
        zn = z / s
        return (zn.t() @ zn) / (zn.size(0) - 1)
    return (corr(f) - corr(r)).pow(2).mean()


# ============================================================
# In-loop TSTR predictor (Agent 1) — mirrors eval metric exactly
# ============================================================
class _PFake(nn.Module):
    def __init__(self, D, hidden):
        super().__init__()
        in_dim = D - 1 if D > 1 else 1
        self.gru = nn.GRU(in_dim, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        return torch.sigmoid(self.fc(out))


def _pred_slice(z, D):
    if D > 1:
        return z[:, :-1, :D - 1], z[:, 1:, D - 1:D]
    return z[:, :-1, :1], z[:, 1:, 0:1]


# ============================================================
# In-loop adversarial discriminator (mirrors the eval Disc metric)
# ============================================================
class _DiscNet(nn.Module):
    """GRU real/fake classifier — same family as the post-hoc Discriminative
    metric's GRU. The editor is trained to fool it (adversarial), directly
    targeting the Disc score the way TSTR-MAML targets Pred."""

    def __init__(self, D, hidden):
        super().__init__()
        self.gru = nn.GRU(D, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])           # [B,1] logit per sequence


# ============================================================
# Retrieval (cached TS2Vec NN on base-branch queries)
# ============================================================
def build_ts2vec_retrieval(real, fake_base_for_query, k, gpu='0', cache_path=None):
    if cache_path and os.path.exists(cache_path):
        idx = np.load(cache_path)
        print(f'[fusion-javelin] TS2Vec retrieval cache hit -> {cache_path} '
              f'shape={idx.shape}')
        return idx
    from utils.ts2vec_loader import get_TS2Vec_class
    print(f'[fusion-javelin] fitting TS2Vec on real {real.shape} ...')
    TS2Vec = get_TS2Vec_class()
    enc = TS2Vec(input_dims=real.shape[-1], device=int(gpu),
                 batch_size=8, lr=1e-3, output_dims=320, max_train_length=3000)
    enc.fit(real, verbose=False)
    real_emb = enc.encode(real, encoding_window='full_series')
    print(f'  real embed: {real_emb.shape}')
    qB_emb = enc.encode(fake_base_for_query, encoding_window='full_series')
    print(f'  fake_base embed: {qB_emb.shape}')
    nn_idx = NearestNeighbors(n_neighbors=k, metric='cosine').fit(real_emb)
    _, idx = nn_idx.kneighbors(qB_emb)
    idx = idx.astype(np.int64)
    if cache_path:
        np.save(cache_path, idx)
        print(f'  cached -> {cache_path}')
    return idx


# ============================================================
# Training
# ============================================================
def train_editor(args):
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', args.gpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    outdir = os.path.join(PROJECT_ROOT, 'outputs', args.name)
    truth_path = glob.glob(os.path.join(outdir, '*_norm_truth_64_train.npy'))
    if not truth_path:
        raise FileNotFoundError(f'no truth.npy in {outdir}')
    real = np.load(truth_path[0]).astype('float32')
    fake_A = np.load(os.path.join(outdir, f'{args.name}_branchA_fake.npy')).astype('float32')
    fake_B = np.load(os.path.join(outdir, f'{args.name}_branchB_fake.npy')).astype('float32')
    N, T, D = real.shape
    n_pairs = min(fake_A.shape[0], fake_B.shape[0])
    fake_A = fake_A[:n_pairs]; fake_B = fake_B[:n_pairs]

    # --- select base (anchor + query) vs other (divergence direction) ---
    if args.base_branch.upper() == 'A':
        base_np, other_np = fake_A, fake_B
    else:
        base_np, other_np = fake_B, fake_A
    print(f'[fusion-javelin] {args.name}: real {real.shape}, '
          f'pairs={n_pairs}, T={T} D={D}, base=branch{args.base_branch.upper()}')

    real_t = torch.from_numpy(real).to(device)
    base_t = torch.from_numpy(base_np).to(device)
    other_t = torch.from_numpy(other_np).to(device)

    # ----- TS2Vec retrieval (query = base branch) -----
    if args.no_retrieval:
        # ablation: replace TS2Vec NN with RANDOM real "neighbours"
        rng = np.random.RandomState(args.seed)
        nbr_idx = rng.randint(0, N, size=(n_pairs, args.k_neighbors)).astype(np.int64)
        print('[fusion-javelin] retrieval OFF (random neighbours, ablation)')
    else:
        cache_path = os.path.join(
            outdir, f'_ts2vec_knn_K{args.k_neighbors}_base{args.base_branch.upper()}.npy')
        nbr_idx = build_ts2vec_retrieval(
            real, base_np, args.k_neighbors, gpu=args.gpu, cache_path=cache_path)
    nbr_idx_t = torch.from_numpy(nbr_idx).to(device)
    print(f'[fusion-javelin] retrieval ready: indices {nbr_idx_t.shape}')

    # ----- editor -----
    editor = FusionEditor(T, D, hidden=args.hidden, n_layers=args.n_layers,
                          n_heads=args.n_heads, rank=args.rank,
                          alpha_max=args.alpha_max,
                          warmup_frac=args.warmup_frac,
                          alpha_end_frac=args.alpha_end_frac,
                          smooth_sigma=args.smooth_sigma,
                          bounded_skew=not args.no_bounded_skew).to(device)
    n_params = sum(p.numel() for p in editor.parameters())
    n_steps = args.epochs * max(1, n_pairs // args.batch_size)
    editor.set_schedule(n_steps)
    print(f'[fusion-javelin] FusionEditor: {n_params/1e6:.2f}M params '
          f'(H={args.hidden}, L={args.n_layers}, r={args.rank}, '
          f'α_max={args.alpha_max}, σ_smooth={args.smooth_sigma})')

    opt = torch.optim.AdamW(editor.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)

    # ----- in-loop TSTR predictor (Agent 1) -----
    P_fake = _PFake(D, max(1, D // 2)).to(device)
    p_opt = torch.optim.Adam(P_fake.parameters(), lr=args.pred_lr)

    # ----- in-loop adversarial discriminator (targets the Disc metric) -----
    use_disc = args.disc_weight > 0.0
    if use_disc:
        D_disc = _DiscNet(D, max(8, D)).to(device)
        d_opt = torch.optim.Adam(D_disc.parameters(), lr=args.disc_lr)
        bce = nn.BCEWithLogitsLoss()
        print(f'[fusion-javelin] adversarial Disc ON: disc_weight={args.disc_weight} '
              f'disc_lr={args.disc_lr}')

    anneal_steps = max(1, int(0.5 * n_steps))
    log_every = max(50, n_steps // 50)
    print(f'[fusion-javelin] train: epochs={args.epochs} steps={n_steps} '
          f'bs={args.batch_size} lr={args.lr} pred_weight={args.pred_weight} '
          f'anchor anneal {args.anchor_start}->{args.anchor_end}')

    editor.train()
    P_fake.train()
    t0 = time.time()
    losses = []
    for step in range(n_steps):
        idx = torch.randint(0, n_pairs, (args.batch_size,), device=device)
        rdx = torch.randint(0, N, (args.batch_size,), device=device)
        xBase, xOther, xR = base_t[idx], other_t[idx], real_t[rdx]
        nbrs = real_t[nbr_idx_t[idx]]                              # [B,K,T,D]
        n_mean = nbrs.mean(dim=1)
        n_std = nbrs.std(dim=1) + 1e-6
        n_med = nbrs.median(dim=1).values
        fused, alpha, corr = editor(xBase, xOther, n_mean, n_std, n_med)

        # --- 1) train P_fake on detached fused (inner TSTR step) ---
        Xf_d, Yf_d = _pred_slice(fused.detach(), D)
        p_opt.zero_grad()
        F.l1_loss(P_fake(Xf_d), Yf_d).backward()
        p_opt.step()

        # --- 2) MAML 1-step lookahead (gradient flows to fused) ---
        with torch.backends.cudnn.flags(enabled=False):
            Xf, Yf = _pred_slice(fused, D)
            params = dict(P_fake.named_parameters())
            inner_loss = F.l1_loss(P_fake(Xf), Yf)
            grads = torch.autograd.grad(inner_loss, list(params.values()),
                                        create_graph=True, retain_graph=True)
            fast_params = {n: p - args.pred_lr * g
                           for (n, p), g in zip(params.items(), grads)}
            Xr, Yr = _pred_slice(xR, D)
            pred_r = functional_call(P_fake, fast_params, (Xr,))
            L_pred = F.l1_loss(pred_r, Yr)

        if step < anneal_steps:
            t_frac = step / anneal_steps
            lam_anc = args.anchor_start + t_frac * (args.anchor_end - args.anchor_start)
        else:
            lam_anc = args.anchor_end
        # --- 3) adversarial discriminator: train D on real-vs-fused.detach,
        #         then editor minimizes BCE(D(fused), real) (fool D) ---
        L_disc = fused.new_zeros(())
        if use_disc:
            d_opt.zero_grad()
            d_real = D_disc(xR)
            d_fake = D_disc(fused.detach())
            d_loss = bce(d_real, torch.ones_like(d_real)) + \
                     bce(d_fake, torch.zeros_like(d_fake))
            d_loss.backward(); d_opt.step()
            # editor wants D to call fused "real"
            L_disc = bce(D_disc(fused), torch.ones_like(d_fake))

        L_mmd = mmd2(fused, xR)
        L_corr = corr_loss(fused, xR)
        L_anc = F.mse_loss(fused, xBase.detach())

        loss = (args.mmd_weight * L_mmd
                + args.corr_weight * L_corr
                + lam_anc * L_anc
                + args.pred_weight * L_pred
                + args.disc_weight * L_disc)
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
        if (step + 1) % log_every == 0:
            recent = float(np.mean(losses[-log_every:]))
            print(f'[fusion-javelin] step {step+1}/{n_steps} '
                  f'({time.time()-t0:.0f}s) loss={recent:.5f} '
                  f'mmd={L_mmd.item():.5f} corr={L_corr.item():.5f} '
                  f'anc={L_anc.item():.5f} pred={L_pred.item():.5f} '
                  f'disc={float(L_disc):.5f} λ_anc={lam_anc:.3f} '
                  f'α={alpha.item():.6f}', flush=True)

    # ----- produce edited fused -----
    editor.eval()
    edited = np.empty_like(base_np)
    inf_bs = 256
    with torch.no_grad():
        for i in range(0, n_pairs, inf_bs):
            xBase = base_t[i:i + inf_bs]; xOther = other_t[i:i + inf_bs]
            nbrs = real_t[nbr_idx_t[i:i + inf_bs]]
            n_mean = nbrs.mean(dim=1); n_std = nbrs.std(dim=1) + 1e-6
            n_med = nbrs.median(dim=1).values
            fused, _, _ = editor(xBase, xOther, n_mean, n_std, n_med)
            edited[i:i + inf_bs] = fused.cpu().numpy()
    out_path = os.path.join(outdir, f'{args.name}{args.out_tag}_fake.npy')
    np.save(out_path, edited.astype('float32'))
    print(f'[fusion-javelin] saved -> {out_path}  shape={edited.shape}  '
          f'range=[{edited.min():.3f},{edited.max():.3f}]')
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True)
    ap.add_argument('--base_branch', default='B', choices=['A', 'B', 'a', 'b'],
                    help='which branch is the anchor + retrieval query '
                         '(the strong branch: A for Stocks, B for KDDCup)')
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--k_neighbors', type=int, default=16)
    ap.add_argument('--hidden', type=int, default=128)
    ap.add_argument('--n_layers', type=int, default=2)
    ap.add_argument('--n_heads', type=int, default=4)
    ap.add_argument('--rank', type=int, default=8)
    ap.add_argument('--alpha_max', type=float, default=0.003,
                    help='peak α (iter-4 caps at iter-2 territory)')
    ap.add_argument('--warmup_frac', type=float, default=0.15)
    ap.add_argument('--alpha_end_frac', type=float, default=0.25,
                    help='fraction of alpha_max retained at end of training')
    ap.add_argument('--no_retrieval', action='store_true',
                    help='ablation: random neighbours instead of TS2Vec NN')
    ap.add_argument('--no_bounded_skew', action='store_true',
                    help='ablation: drop the x(1-x) domain gate')
    ap.add_argument('--smooth_sigma', type=float, default=3.0,
                    help='σ for the depthwise Gaussian time-smoother on V')
    ap.add_argument('--batch_size', type=int, default=128)
    ap.add_argument('--lr', type=float, default=5e-4)
    ap.add_argument('--weight_decay', type=float, default=1e-4)
    ap.add_argument('--mmd_weight', type=float, default=1.0)
    ap.add_argument('--corr_weight', type=float, default=0.5)
    ap.add_argument('--anchor_start', type=float, default=1.0)
    ap.add_argument('--anchor_end', type=float, default=0.05)
    ap.add_argument('--pred_weight', type=float, default=1.0,
                    help='weight on TSTR-MAML lookahead loss')
    ap.add_argument('--pred_lr', type=float, default=1e-3,
                    help='inner-loop lr for P_fake')
    ap.add_argument('--disc_weight', type=float, default=0.0,
                    help='weight on in-loop adversarial Disc loss (0=off)')
    ap.add_argument('--disc_lr', type=float, default=1e-3,
                    help='lr for the in-loop discriminator')
    ap.add_argument('--seed', type=int, default=2023)
    ap.add_argument('--gpu', default='0')
    ap.add_argument('--out_tag', default='_javelinfused')
    args = ap.parse_args()
    train_editor(args)


if __name__ == '__main__':
    main()
