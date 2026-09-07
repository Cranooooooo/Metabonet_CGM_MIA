#!/usr/bin/env python3
"""train_maven.py — MAVEN training pipeline (2026-05-20).

Dual-Visual GENERATION-only Flow Matching (V2 — cascade removed).

  * Two visual branches A/B (delay embedding / STFT) lift the 1-D multivariate
    window into 2-D images; Flow Matching runs in image space (linear gather
    maps keep the linear-path FM well-defined, design §3.1).
  * Each branch is a single unconditional gen VelocityNet — the impH/impL
    soft-prior cascade of V1 is removed (gen-only: 2 forwards/step, ~3x faster).
  * The two branches are coupled by a TS-domain cross-consistency loss L_cross.
  * Generation outputs are fused in TS domain by a learnable per-channel gate
    (baseline); the gate is trained by a detached fuse loss. The JAVELIN-style
    retrieval/closed-form fusion is a separate post-hoc module (see
    JAVELIN_FUSION_PLAN.md) operating on the saved per-branch fakes.

Per training step: 2 branches x 1 gen stage = 2 forwards + 1 backward.

    L = L_gen                                    (both branches, FM velocity)
      + gamma * L_cross                          (cross-branch consistency)
      + fuse_alpha * L_fuse                      (gate training, branches detached)

Sampling uses SHARED TS-domain noise for both branches so out_A[i]/out_B[i] are
paired views of the same draw — required for meaningful per-sample fusion.
`<name>_fake.npy` is the gate-fused unconditional generation, (N,T,D) in [0,1];
per-branch fakes `<name>_branchA/B_fake.npy` are saved paired for fusion.

Usage:
    python -u train_maven.py --dataset GENERAL_Stocks \
        --name maven_stocks_seed2023 --seed 2023 --gpu 0
"""

import argparse
import json
import math
import os
import sys
import time

# CUDA_VISIBLE_DEVICES must be set BEFORE torch is imported.
def _peek_gpu_arg():
    for i, a in enumerate(sys.argv[1:], start=1):
        if a == '--gpu' and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
        if a.startswith('--gpu='):
            return a.split('=', 1)[1]
    return None

_gpu = _peek_gpu_arg()
if _gpu is not None:
    os.environ['CUDA_VISIBLE_DEVICES'] = _gpu

# 6 forwards/step retain a large activation graph; expandable segments avoids
# fragmentation OOMs on smaller GPUs (e.g. 11 GB RTX 2080 Ti). Set before torch.
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler

from visual_transforms import build_transforms

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


# ============================================================
# Data: stride=1 sliding window from raw CSV (matches r10)
# ============================================================
DATASET_CSV_MAP = {
    # name -> (csv_path, drop_first_col)
    'GENERAL_Stocks': (os.path.join(PROJECT_ROOT, 'datasets', 'stock_data.csv'), False),
    'GENERAL_Etth':   (os.path.join(PROJECT_ROOT, 'datasets', 'ETTh.csv'),       True),
    'GENERAL_Energy': (os.path.join(PROJECT_ROOT, 'datasets', 'energy_data.csv'), False),
    'GENERAL_KDDCup': (os.path.join(PROJECT_ROOT, 'datasets', 'kddcup.csv'),      False),
}


def load_dataset_csv(dataset_name, window=64, seed=2023, csv_override=None):
    """Read CSV -> MinMaxScaler -> [-1,1] -> stride-1 windows of length
    `window`, random-permuted by seed. Identical to r10."""
    if csv_override:
        csv_path = csv_override
        drop_first = DATASET_CSV_MAP.get(dataset_name, (None, False))[1]
    else:
        csv_path, drop_first = DATASET_CSV_MAP[dataset_name]

    df = pd.read_csv(csv_path, header=0)
    if drop_first:
        df.drop(df.columns[0], axis=1, inplace=True)
    raw = df.values.astype(np.float32)
    scaler = MinMaxScaler()
    raw01 = scaler.fit_transform(raw).astype(np.float32)
    data_neg = raw01 * 2.0 - 1.0  # [-1, 1]
    N_total, D = data_neg.shape
    N = N_total - window + 1
    if N <= 0:
        raise ValueError(f'Window {window} > raw length {N_total} for {dataset_name}')
    windows = np.zeros((N, window, D), dtype=np.float32)
    for i in range(N):
        windows[i] = data_neg[i: i + window]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(N)
    return windows[perm], scaler


# ============================================================
# Model: VelocityNet — 2D-ified, with stage embeddings
# ============================================================
class SinusoidalTime(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):  # t: [B] in [0,1]
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device) / max(1, half - 1)
        )
        args = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class VelocityNet(nn.Module):
    """FM linear-path velocity predictor, operating on a flattened image-token
    sequence [B, L, D]. Encoder-only Transformer backbone (n_layers);
    `seq_len = L` is the image token count. Gen-only: no conditioning inputs,
    no stage embedding.

    Inputs:
      x        : [B, L, D]   noisy image at flow-time t
      t        : [B]         flow time in [0,1]
    Output:
      v_pred   : [B, L, D]   predicted linear-path velocity (target = x1 - x0)
    """

    def __init__(self, feature_size, hidden=256, n_layers=4,
                 n_heads=8, seq_len=120, dropout=0.0):
        super().__init__()
        self.in_proj = nn.Linear(feature_size, hidden)

        self.time_emb = SinusoidalTime(hidden)
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden),
        )
        # learned positional embedding sized to the image token grid
        self.pos_emb = nn.Parameter(torch.zeros(1, seq_len, hidden))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

        # Encoder-only backbone (the prior bottleneck only hosted the now-removed
        # L_decor/L_corrmap losses; a single Transformer encoder stack suffices).
        layer_kwargs = dict(
            d_model=hidden, nhead=n_heads, dim_feedforward=hidden * 4,
            activation='gelu', batch_first=True, norm_first=True, dropout=dropout,
        )
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(**layer_kwargs), num_layers=n_layers)
        self.out_norm = nn.LayerNorm(hidden)
        self.out_proj = nn.Linear(hidden, feature_size)

    def forward(self, x, t):
        h = self.in_proj(x)
        h = h + self.pos_emb[:, : x.shape[1]]
        h = h + self.time_mlp(self.time_emb(t))[:, None, :]
        h = self.encoder(h)
        return self.out_proj(self.out_norm(h))


class JointFusionEditor(nn.Module):
    """Rank-r channel-coupled residual editor on the strong branch, operating in
    [0,1] TS space. Ported from the post-hoc `fusion_editor.py` iter-4 design,
    retrieval stripped (joint training supplies the manifold signal via the
    branches + MMD-to-real). Used when --fuse_mode editor so fusion gradients
    flow into both branches (the lever post-hoc fusion lacked).

        fused = clamp( base + α · smooth(V[base,other−base]) Uᵀ · base(1−base), 0,1)
    """

    def __init__(self, T, D, hidden=128, n_layers=2, n_heads=4, rank=8,
                 alpha_max=0.05, smooth_sigma=3.0):
        super().__init__()
        self.T, self.D, self.r = int(T), int(D), int(rank)
        self.in_proj = nn.Linear(2 * D, hidden)
        self.pos = nn.Parameter(torch.zeros(1, T, hidden))
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=n_heads, dim_feedforward=hidden * 4,
            activation='gelu', batch_first=True, norm_first=True, dropout=0.0)
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm_out = nn.LayerNorm(hidden)
        self.V_head = nn.Linear(hidden, self.r)
        self.U = nn.Parameter(torch.empty(D, self.r))
        nn.init.trunc_normal_(self.U, std=1.0 / (D ** 0.5))
        self.alpha_param = nn.Parameter(torch.tensor(-2.0))   # softplus(-2)≈0.13
        self.alpha_max = float(alpha_max)
        ks = int(2 * round(3 * smooth_sigma) + 1)
        t = torch.arange(ks) - ks // 2
        k = torch.exp(-(t.float() ** 2) / (2 * smooth_sigma ** 2))
        k = (k / k.sum()).view(1, 1, ks).repeat(self.r, 1, 1)
        self.register_buffer('smooth_k', k)
        self.smooth_pad = ks // 2

    def forward(self, base01, other01):
        h = self.in_proj(torch.cat([base01, other01 - base01], dim=-1)) + self.pos
        h = self.norm_out(self.encoder(h))
        V = self.V_head(h)                                            # [B,T,r]
        V_s = F.conv1d(V.transpose(1, 2), self.smooth_k,
                       padding=self.smooth_pad, groups=self.r).transpose(1, 2)
        corr = V_s @ self.U.t()                                       # [B,T,D]
        alpha = self.alpha_max * F.softplus(self.alpha_param) / F.softplus(
            torch.zeros((), device=base01.device))
        bound = base01 * (1.0 - base01)
        return (base01 + alpha * corr * bound).clamp(0.0, 1.0)


class MAVEN(nn.Module):
    """Container for both branch VelocityNets + the fusion gate, so EMA wraps a
    single module. Visual transforms are kept outside (constant buffers)."""

    def __init__(self, feature_size, seq_len_A, seq_len_B, hidden=256,
                 n_layers=4, n_heads=8,
                 fuse_mode='gate', seq_len_T=64, editor_rank=8,
                 editor_hidden=128, alpha_max=0.05):
        super().__init__()
        self.net = nn.ModuleDict({
            'A': VelocityNet(feature_size, hidden, n_layers,
                             n_heads, seq_len=seq_len_A),
            'B': VelocityNet(feature_size, hidden, n_layers,
                             n_heads, seq_len=seq_len_B),
        })
        # per-channel fusion logit; sigmoid(0)=0.5 -> starts as naive average,
        # free to move toward pick-best (design §6.1 / history §1.3).
        self.gate_logit = nn.Parameter(torch.zeros(feature_size))
        self.fuse_mode = fuse_mode
        self.editor = None
        if fuse_mode == 'editor':
            self.editor = JointFusionEditor(
                seq_len_T, feature_size, hidden=editor_hidden,
                rank=editor_rank, alpha_max=alpha_max)

    def gate(self):
        return torch.sigmoid(self.gate_logit)


# ============================================================
# Loss helpers
# ============================================================
def masked_mse(pred, target, mask):
    return ((pred - target) ** 2 * mask).sum() / mask.sum().clamp(min=1.0)


def mmd2(x, y, sigmas=(1.0, 2.0, 4.0, 8.0, 16.0)):
    """Multi-scale RBF MMD^2 over flattened [B, T*D] (matches post-hoc editor)."""
    x = x.reshape(x.size(0), -1); y = y.reshape(y.size(0), -1)
    xx = torch.cdist(x, x).pow(2); yy = torch.cdist(y, y).pow(2)
    xy = torch.cdist(x, y).pow(2)
    out = x.new_zeros(())
    for s in sigmas:
        out = out + (torch.exp(-xx / (2 * s)) + torch.exp(-yy / (2 * s))
                     - 2 * torch.exp(-xy / (2 * s))).mean()
    return out


def corr_loss(f, r):
    """Channel-correlation-matrix MSE (direct Cross-corr supervision)."""
    def corr(z):
        z = z.reshape(-1, z.size(-1)); z = z - z.mean(0, keepdim=True)
        zn = z / z.std(0, keepdim=True).clamp(min=1e-6)
        return (zn.t() @ zn) / (zn.size(0) - 1)
    return (corr(f) - corr(r)).pow(2).mean()


# ============================================================
# Single training step: 2 branches x 1 gen stage = 2 forwards
# ============================================================
def maven_step(maven, transforms, x0, M_obs,
                gamma, fuse_alpha,
                fuse_mode='gate', base_branch='A', beta=1.0,
                mmd_w=1.0, corr_w=10.0, anchor_w=0.05, t_weight=True,
                share_noise=True, only_branch=None, amp='off'):
    """One gen-only MAVEN step. x0: [B,T,D]; M_obs: [B,T,D] (all 1s for full
    GENERAL data, kept for signature symmetry / future masking).

    Cross-view co-training: when `share_noise`, both branches use ONE shared
    time-domain noise z and ONE shared t, so they are two visual views of the
    SAME FM trajectory point; their denoised estimates must agree, making
    L_cross a strong multi-view mutual-distillation signal (each view regularizes
    the other → each branch beats independent training, even w/o fusion).
    `share_noise=False` (+ gamma=0) = independent single-view branches (baseline).

    fuse_mode='gate'   : legacy detached per-channel gate (baseline).
    fuse_mode='editor' : rank-r editor on (non-detached) x̂0 -> fusion loss
                         (MMD+Corr+anchor) backprops into BOTH branches; the
                         per-sample loss is weighted by (1−t) (design §M.6)."""
    B, T, D = x0.shape
    device = x0.device

    L_gen = x0.new_zeros(())         # gen FM loss (both branches)
    xhat0_ts = {}                    # TS-domain x_hat0 per branch, for L_cross
    t_br = {}                        # per-branch FM time (for (1-t) weighting)
    metrics = {}

    # shared noise + t across branches (co-training): same FM trajectory point
    z_shared = torch.randn_like(x0) if share_noise else None
    t_shared = torch.rand(B, device=device) if share_noise else None

    _branches = (only_branch,) if only_branch is not None else ('A', 'B')
    for b in _branches:
        phi = transforms[b]
        net = maven.net[b]
        I_b = phi.encode(x0)                              # [B,L,D] clean image
        valid = phi.image_valid(B, D, device)             # real (non-pad) cells

        # --- gen stage: unconditional FM ---
        eps_ts = z_shared if share_noise else torch.randn_like(x0)
        x1 = phi.encode(eps_ts)                           # TS noise -> image
        t = t_shared if share_noise else torch.rand(B, device=device)
        t_br[b] = t
        x_t = (1 - t)[:, None, None] * I_b + t[:, None, None] * x1
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=(amp == 'bf16')):
            v_pred = net(x_t, t)
            L_gen = L_gen + masked_mse(v_pred, x1 - I_b, valid)
        xhat0_gen = x_t - t[:, None, None] * v_pred.float()
        xhat0_ts[b] = phi.decode(xhat0_gen)

    # single-branch training (branch-parallel mode): only the gen FM loss, no
    # cross-view term and no fusion (those need both branches).
    if only_branch is not None:
        metrics.update(L_gen=L_gen.item(), L_cross=0.0, L_fuse=0.0)
        return L_gen, metrics

    # --- cross-view consistency / mutual distillation (design §M.6) ---
    L_cross = F.mse_loss(xhat0_ts['A'], xhat0_ts['B'])

    if fuse_mode == 'editor':
        # --- JOINT editor fusion: gradients flow into BOTH branches ---
        other = 'B' if base_branch == 'A' else 'A'
        base01 = (xhat0_ts[base_branch] + 1.0) * 0.5     # [-1,1] -> [0,1]
        other01 = (xhat0_ts[other] + 1.0) * 0.5
        real01 = (x0 + 1.0) * 0.5
        fused01 = maven.editor(base01, other01)            # NOT detached
        # reliability weight: x̂0 is clean at low t (mean of both branches' t)
        if t_weight:
            w = (1.0 - 0.5 * (t_br['A'] + t_br['B'])).clamp(min=0.0)   # [B]
        else:
            w = torch.ones_like(t_br['A'])                            # ablation
        wbar = w.mean()
        L_mmd = mmd2(fused01, real01)
        L_corr = corr_loss(fused01, real01)
        L_anc = (((fused01 - base01.detach()) ** 2).mean(dim=(1, 2)) * w).mean()
        L_fuse = mmd_w * L_mmd + corr_w * L_corr + anchor_w * L_anc
        L_fuse = wbar * L_fuse
        fuse_term = beta * L_fuse
        metrics.update(L_mmd=float(L_mmd.item()), L_corr=float(L_corr.item()),
                       L_anc=float(L_anc.item()))
    else:
        # --- legacy fusion-gate training (branches detached) ---
        g = maven.gate()
        fused_gen = (g * xhat0_ts['A'].detach()
                     + (1.0 - g) * xhat0_ts['B'].detach())
        L_fuse = F.mse_loss(fused_gen, x0)
        fuse_term = fuse_alpha * L_fuse

    L_total = (
        L_gen
        + gamma * L_cross
        + fuse_term
    )
    metrics.update(
        L_gen=L_gen.item(), L_cross=L_cross.item(),
        L_fuse=float(L_fuse.item()),
    )
    return L_total, metrics


# ============================================================
# Sampling: SHARED-noise paired generation + gate fusion
# ============================================================
@torch.no_grad()
def sample_maven(maven, transforms, n_samples, T, D, n_steps, batch, device, amp='off'):
    """Both branches integrate the gen FM ODE from the SAME TS-domain noise z,
    so out_A[i] and out_B[i] are paired views of one draw (required for
    meaningful per-sample fusion; independent noise collapses the convex
    combination's variance). Returns (fused, branchA, branchB), each [n,T,D]."""
    outs = {'A': [], 'B': []}
    n_done = 0
    while n_done < n_samples:
        b = min(batch, n_samples - n_done)
        z_ts = torch.randn(b, T, D, device=device)        # shared TS noise
        for br in ('A', 'B'):
            phi = transforms[br]
            net = maven.net[br]
            x = phi.encode(z_ts)                          # same draw -> image
            ts = torch.linspace(1.0, 0.0, n_steps + 1, device=device)
            for i in range(n_steps):
                t_cur = ts[i].expand(b)
                with torch.autocast('cuda', dtype=torch.bfloat16, enabled=(amp == 'bf16')):
                    v = net(x, t_cur)
                x = x - v.float() * (ts[i] - ts[i + 1])
            outs[br].append(phi.decode(x).cpu().numpy())
        n_done += b
    out_A = np.concatenate(outs['A'], axis=0).astype(np.float32)
    out_B = np.concatenate(outs['B'], axis=0).astype(np.float32)
    if getattr(maven, 'fuse_mode', 'gate') == 'editor' and maven.editor is not None:
        # apply the trained editor to the final paired samples ([-1,1]->[0,1])
        base_br = getattr(maven, '_base_branch', 'A')
        base_np = out_A if base_br == 'A' else out_B
        other_np = out_B if base_br == 'A' else out_A
        fused01 = np.empty_like(base_np)
        ib = 256
        for i in range(0, base_np.shape[0], ib):
            b01 = torch.from_numpy((base_np[i:i+ib] + 1.0) * 0.5).to(device)
            o01 = torch.from_numpy((other_np[i:i+ib] + 1.0) * 0.5).to(device)
            fused01[i:i+ib] = maven.editor(b01, o01).cpu().numpy()
        fused = (fused01 * 2.0 - 1.0).astype(np.float32)     # back to [-1,1]
    else:
        g = maven.gate().detach().cpu().numpy()[None, None, :]      # [1,1,D]
        fused = (g * out_A + (1.0 - g) * out_B).astype(np.float32)
    return fused, out_A, out_B


# ============================================================
# Main
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, choices=list(DATASET_CSV_MAP.keys()))
    ap.add_argument('--name', required=True)
    ap.add_argument('--gpu', default='0')
    ap.add_argument('--seed', type=int, default=2023)
    # Schedule
    ap.add_argument('--total_iters', type=int, default=150_000)
    ap.add_argument('--ckpt_every', type=int, default=25_000)
    # Optimization
    ap.add_argument('--batch_size', type=int, default=96,
                    help='design §10 specifies 256; lowered to 96 because the '
                         '6-forward step OOMs a 11 GB GPU above ~96 on D=59 data')
    ap.add_argument('--micro_batch_size', type=int, default=None,
                    help='optional per-forward micro-batch size for gradient '
                         'accumulation; effective batch remains --batch_size')
    ap.add_argument('--lr', type=float, default=5e-4)
    ap.add_argument('--weight_decay', type=float, default=1e-4)
    ap.add_argument('--grad_clip', type=float, default=1.0)
    ap.add_argument('--ema_decay', type=float, default=0.999)
    # Architecture
    ap.add_argument('--hidden_dim', type=int, default=256)
    ap.add_argument('--n_layers', type=int, default=4,
                    help='encoder-only Transformer depth per branch')
    ap.add_argument('--n_heads', type=int, default=8)
    # Visual transforms
    ap.add_argument('--delay_tau', type=int, default=4, help='branch-A delay tau')
    ap.add_argument('--delay_m', type=int, default=8, help='branch-A embedding m')
    ap.add_argument('--branch_b', default='stft', choices=['stft', 'periodfold'],
                    help="branch-B transform: 'stft' (complex STFT, ImagenTime "
                         "style — default) or 'periodfold' (gather-based fold)")
    ap.add_argument('--stft_n_fft', type=int, default=16, help='branch-B STFT n_fft')
    ap.add_argument('--stft_hop', type=int, default=8,
                    help='branch-B STFT hop (= n_fft/2 for exact reconstruction)')
    ap.add_argument('--fold_period', type=int, default=None,
                    help='branch-B fold period P when --branch_b periodfold '
                         '(default: FFT-picked, capped at floor(1.5*sqrt(T)))')
    # Loss weights
    ap.add_argument('--gamma', type=float, default=0.5,
                    help='cross-view consistency (L_cross) weight; 0 = independent branches')
    ap.add_argument('--no_share_noise', action='store_true',
                    help='ablation: independent noise/t per branch (no co-training)')
    ap.add_argument('--fuse_alpha', type=float, default=0.1, help='L_fuse weight')
    # --- joint fusion (editor) ---
    ap.add_argument('--fuse_mode', default='gate', choices=['gate', 'editor'],
                    help="'editor' = joint rank-r editor on x̂0, grads flow to branches")
    ap.add_argument('--only_branch', default=None, choices=['A', 'B'],
                    help='branch-parallel: train+sample ONLY this branch, save '
                         'its <name>_branch{A,B}_fake.npy (independent runs to fuse later)')
    ap.add_argument('--base_branch', default='A', choices=['A', 'B'],
                    help='anchor/strong branch for the editor')
    ap.add_argument('--beta', type=float, default=1.0, help='joint fusion-loss weight')
    ap.add_argument('--mmd_w', type=float, default=1.0)
    ap.add_argument('--corr_w', type=float, default=10.0)
    ap.add_argument('--anchor_w', type=float, default=0.05)
    ap.add_argument('--editor_rank', type=int, default=8)
    ap.add_argument('--editor_hidden', type=int, default=128)
    ap.add_argument('--editor_alpha_max', type=float, default=0.05)
    ap.add_argument('--no_t_weight', action='store_true',
                    help='ablation: drop the (1-t) reliability weighting')
    # Sampling
    ap.add_argument('--sampling_steps', type=int, default=200,
                    help='Euler steps for the final (canonical) fake')
    ap.add_argument('--milestone_sample_steps', type=int, default=100)
    ap.add_argument('--sample_batch', type=int, default=128)
    ap.add_argument('--num_samples', type=int, default=None,
                    help='number of samples to draw (default = N)')
    ap.add_argument('--no_final_sample', action='store_true')
    ap.add_argument('--no_milestone_sample', action='store_true')
    # I/O
    ap.add_argument('--checkpoint_dir', default=os.path.join(PROJECT_ROOT, 'checkpoints'))
    ap.add_argument('--results_dir', default=os.path.join(PROJECT_ROOT, 'results'))
    ap.add_argument('--out_dir', default=os.path.join(PROJECT_ROOT, 'outputs'),
                    help='per-cell dir for truth.npy + fake.npy (DiMTS eval inputs)')
    ap.add_argument('--log_every', type=int, default=500)
    ap.add_argument('--resume', default=None)
    ap.add_argument('--data_csv', default=None)
    ap.add_argument('--window', type=int, default=64)
    ap.add_argument('--amp', default='off', choices=['off', 'bf16'],
                    help='bf16 autocast for the VelocityNet fwd + sampling '
                         '(~2.4-2.8x on Ampere); TF32 auto-enabled. '
                         'params/EMA stay fp32.')
    args = ap.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.amp == 'bf16':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision('high')
        print('[maven] amp=bf16 enabled (autocast + TF32); params/EMA stay fp32')

    # --- Data ---
    print(f'[maven] loading {args.dataset} (T={args.window}, stride=1, seed={args.seed})')
    data, scaler = load_dataset_csv(
        args.dataset, window=args.window, seed=args.seed, csv_override=args.data_csv)
    N, T, D = data.shape
    print(f'[maven] data shape (N,T,D) = {data.shape}, '
          f'range [{data.min():.3f}, {data.max():.3f}]')
    data_t = torch.from_numpy(data).to(device)

    # --- Visual transforms (branch A delay embed / branch B STFT or fold) ---
    phi_A, phi_B = build_transforms(
        data, T, tau=args.delay_tau, m=args.delay_m, period=args.fold_period,
        branch_b=args.branch_b, n_fft=args.stft_n_fft, hop_length=args.stft_hop)
    phi_A, phi_B = phi_A.to(device), phi_B.to(device)
    transforms = {'A': phi_A, 'B': phi_B}
    print(f'[maven] branch A {phi_A.extra_repr()}')
    print(f'[maven] branch B {phi_B.extra_repr()}')

    # --- Model + EMA ---
    maven = MAVEN(
        feature_size=D, seq_len_A=phi_A.L, seq_len_B=phi_B.L,
        hidden=args.hidden_dim, n_layers=args.n_layers, n_heads=args.n_heads,
        fuse_mode=args.fuse_mode, seq_len_T=T, editor_rank=args.editor_rank,
        editor_hidden=args.editor_hidden, alpha_max=args.editor_alpha_max,
    ).to(device)
    maven._base_branch = args.base_branch
    n_params = sum(p.numel() for p in maven.parameters())
    print(f'[maven] MAVEN: {n_params/1e6:.2f}M params '
          f'(D={D}, H={args.hidden_dim}, n_layers={args.n_layers}, '
          f'heads={args.n_heads}, L_A={phi_A.L}, L_B={phi_B.L})')

    optim = torch.optim.AdamW(
        maven.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    ema = torch.optim.swa_utils.AveragedModel(
        maven,
        multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(decay=args.ema_decay),
    )
    ema.module._base_branch = args.base_branch
    ema.module.fuse_mode = args.fuse_mode

    # --- Resume ---
    start_iter = 0
    if args.resume:
        print(f'[maven] resuming from {args.resume}')
        sd = torch.load(args.resume, map_location=device, weights_only=False)
        maven.load_state_dict(sd['model'], strict=False)
        ema.module.load_state_dict(sd['ema_module'], strict=False)
        if 'ema_n_averaged' in sd:
            ema.n_averaged.fill_(sd['ema_n_averaged'])
        if 'optim' in sd:
            try:
                optim.load_state_dict(sd['optim'])
            except (ValueError, KeyError) as e:
                print(f'[maven] optim state not restored (param set changed, '
                      f'e.g. added editor): {e}. Using fresh optimizer.')
        # warm-start fine-tune (e.g. add editor): restart the iter counter so
        # --total_iters means "this many fine-tune steps", not absolute.
        start_iter = 0 if args.fuse_mode == 'editor' else int(sd.get('iter', 0))

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    cell_out_dir = os.path.join(args.out_dir, args.name)
    os.makedirs(cell_out_dir, exist_ok=True)

    # Save real reference (DiMTS-eval format: [0,1], shape (N,T,D))
    ds_lc = args.dataset.lower().replace('general_', '')
    truth_path = os.path.join(cell_out_dir, f'{ds_lc}_norm_truth_{args.window}_train.npy')
    if not os.path.exists(truth_path):
        real01 = ((data + 1.0) / 2.0).astype(np.float32)
        np.save(truth_path, real01)
        print(f'[maven] saved truth -> {truth_path}  shape={real01.shape}')

    micro_bs = args.micro_batch_size or args.batch_size
    micro_bs = min(micro_bs, args.batch_size)
    if micro_bs <= 0:
        raise ValueError('--micro_batch_size must be positive')
    losses = []
    t_start = time.time()
    maven.train()
    print(f'[maven] start training (gen-only): total_iters={args.total_iters} '
          f'bs={args.batch_size} micro_bs={micro_bs} lr={args.lr} '
          f'ema_decay={args.ema_decay}')

    def save_ckpt(path, it_done):
        torch.save({
            'model': maven.state_dict(),
            'ema_module': ema.module.state_dict(),
            'ema_n_averaged': int(ema.n_averaged.item()),
            'optim': optim.state_dict(),
            'iter': it_done,
            'config': vars(args),
            'fold_period': getattr(phi_B, 'P', None),
        }, path)

    for it in range(start_iter, args.total_iters):
        optim.zero_grad(set_to_none=True)
        total_loss = 0.0
        met_accum = {}
        n_seen = 0
        while n_seen < args.batch_size:
            bsz = min(micro_bs, args.batch_size - n_seen)
            idx = torch.randint(0, N, (bsz,), device=device)
            x0 = data_t[idx]                              # [B,T,D]
            M_obs = torch.ones(bsz, T, D, device=device)  # full data
            loss, met = maven_step(
                maven, transforms, x0, M_obs,
                gamma=args.gamma, fuse_alpha=args.fuse_alpha,
                fuse_mode=args.fuse_mode, base_branch=args.base_branch,
                beta=args.beta, mmd_w=args.mmd_w, corr_w=args.corr_w,
                anchor_w=args.anchor_w, t_weight=not args.no_t_weight,
                share_noise=not args.no_share_noise,
                only_branch=args.only_branch,
                amp=args.amp,
            )
            weight = bsz / args.batch_size
            (loss * weight).backward()
            total_loss += float(loss.item()) * weight
            for k, v in met.items():
                met_accum[k] = met_accum.get(k, 0.0) + float(v) * weight
            n_seen += bsz
        met = met_accum
        if args.grad_clip and args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(maven.parameters(), args.grad_clip)
        optim.step()
        ema.update_parameters(maven)
        losses.append(total_loss)

        if it == start_iter or (it + 1) % args.log_every == 0:
            elapsed = time.time() - t_start
            recent = float(np.mean(losses[-args.log_every:]))
            print(
                f'[maven] iter {it+1}/{args.total_iters} ({elapsed:.0f}s) '
                f'loss={recent:.5f} L_gen={met["L_gen"]:.4f} '
                f'L_cross={met["L_cross"]:.4f} L_fuse={met["L_fuse"]:.4f}',
                flush=True,
            )

        if (it + 1) % args.ckpt_every == 0 and (it + 1) < args.total_iters:
            mid_path = os.path.join(args.checkpoint_dir, f'{args.name}_{it+1}.pt')
            save_ckpt(mid_path, it + 1)
            print(f'[maven] mid-ckpt @ iter {it+1} -> {mid_path}', flush=True)

            if not args.no_milestone_sample:
                ms_n = args.num_samples if args.num_samples is not None else N
                t_ms = time.time()
                ema.eval()
                ms_fused, _, _ = sample_maven(
                    ema.module, transforms, ms_n, T, D,
                    n_steps=args.milestone_sample_steps,
                    batch=args.sample_batch, device=device, amp=args.amp)
                ms_fake01 = ((ms_fused + 1.0) / 2.0).astype(np.float32)
                ms_path = os.path.join(
                    cell_out_dir, f'{args.name}_M{(it+1)//1000}K_fake.npy')
                np.save(ms_path, ms_fake01)
                maven.train()
                print(f'[maven] milestone fake @ iter {it+1} -> {ms_path} '
                      f'({(time.time()-t_ms)/60:.1f} min, '
                      f'{args.milestone_sample_steps} steps)', flush=True)

    train_time = time.time() - t_start
    final_path = os.path.join(args.checkpoint_dir, f'{args.name}.pt')
    save_ckpt(final_path, args.total_iters)
    print(f'[maven] done. train_time={train_time/60:.1f} min, final -> {final_path}')

    # ===== Final unconditional sampling from EMA for DiMTS eval =====
    sample_time = 0.0
    fake_path = None
    if not args.no_final_sample:
        n_sample = args.num_samples if args.num_samples is not None else N
        print(f'[maven] sampling {n_sample} from EMA: {args.sampling_steps} '
              f'Euler steps, batch={args.sample_batch}', flush=True)
        t_sample0 = time.time()
        ema.eval()
        if args.only_branch is not None:
            # branch-parallel: sample ONLY this branch's fake (single-branch ODE)
            br = args.only_branch
            phi = transforms[br]; net = ema.module.net[br]
            outs = []; n_done = 0
            with torch.no_grad():
                while n_done < n_sample:
                    bb = min(args.sample_batch, n_sample - n_done)
                    x = phi.encode(torch.randn(bb, T, D, device=device))
                    ts = torch.linspace(1.0, 0.0, args.sampling_steps + 1, device=device)
                    for i in range(args.sampling_steps):
                        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=(args.amp == 'bf16')):
                            v = net(x, ts[i].expand(bb))
                        x = x - v.float() * (ts[i] - ts[i + 1])
                    outs.append(phi.decode(x).cpu().numpy()); n_done += bb
            out = np.concatenate(outs, axis=0).astype(np.float32)
            bpath = os.path.join(cell_out_dir, f'{args.name}_branch{br}_fake.npy')
            np.save(bpath, ((out + 1.0) / 2.0).astype(np.float32))
            sample_time = time.time() - t_sample0
            print(f'[maven] single-branch {br} sampled shape={out.shape} '
                  f'in {sample_time/60:.1f} min -> {bpath}', flush=True)
            fake_path = bpath
        else:
            fused, out_A, out_B = sample_maven(
                ema.module, transforms, n_sample, T, D,
                n_steps=args.sampling_steps, batch=args.sample_batch, device=device, amp=args.amp)
            gate = ema.module.gate().detach().cpu().numpy()
            # DiMTS eval expects [0,1]; fused is the canonical fake.
            fake01 = ((fused + 1.0) / 2.0).astype(np.float32)
            fake_path = os.path.join(cell_out_dir, f'{args.name}_fake.npy')
            np.save(fake_path, fake01)
            # per-branch fakes for the design §11 step-3 fusion ablation
            np.save(os.path.join(cell_out_dir, f'{args.name}_branchA_fake.npy'),
                    ((out_A + 1.0) / 2.0).astype(np.float32))
            np.save(os.path.join(cell_out_dir, f'{args.name}_branchB_fake.npy'),
                    ((out_B + 1.0) / 2.0).astype(np.float32))
            sample_time = time.time() - t_sample0
            print(f'[maven] sampled fake.npy shape={fake01.shape} '
                  f'range=[{fake01.min():.3f},{fake01.max():.3f}] '
                  f'gate(mean/min/max)=({gate.mean():.3f}/{gate.min():.3f}/'
                  f'{gate.max():.3f}) in {sample_time/60:.1f} min -> {fake_path}',
              flush=True)

    # JSON log (sub-sampled loss curve)
    sub = max(1, len(losses) // 200)
    log_path = os.path.join(args.results_dir, f'{args.name}.json')
    with open(log_path, 'w') as f:
        json.dump({
            'name': args.name,
            'dataset': args.dataset,
            'config': vars(args),
            'n_params': n_params,
            'data_shape': list(data.shape),
            'branch_A_grid': list(phi_A.grid),
            'branch_B_grid': list(phi_B.grid),
            'branch_b': args.branch_b,
            'fold_period': getattr(phi_B, 'P', None),
            'train_time_sec': train_time,
            'sample_time_sec': sample_time,
            'final_fake_path': fake_path,
            'truth_path': truth_path,
            'final_loss': float(np.mean(losses[-200:])) if losses else None,
            'loss_curve': [
                {'iter': i * sub + 1, 'loss': float(losses[i * sub])}
                for i in range(len(losses) // sub)
            ],
        }, f, indent=2)
    print(f'[maven] log -> {log_path}')


if __name__ == '__main__':
    main()
