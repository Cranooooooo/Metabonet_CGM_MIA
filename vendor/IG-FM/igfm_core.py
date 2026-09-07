#!/usr/bin/env python3
"""train_r10.py — FM-EncDec rewrite (2026-05-14).

Replaces the previous SSDTS+Mamba2 r10 with the FM-EncDec architecture from
simple_fm.py (Model_6_VLM_RL_Based EXPERIMENT_LOG.md "FM-big D_all" line),
wired into r10's gen+imputation IDG framework — single phase, lambda ramp.

Per-step: dual forward pass (gen branch + impute branch with X_prior soft prior)
    L = L_imp + lambda(it) * L_gen + decor_alpha * L_decor + corrmap_alpha * L_corrmap
    lambda(it) = 1 (0..100K) | 2 (100K..200K) | 4 (200K..300K)

Architecture: VelocityNet (encoder-bottleneck-decoder Transformer)
    hidden=128, n_enc=4, n_dec=4, n_heads=8
    Inputs:  x_t [B,T,D], t [B], M_cond [B,T,D], X_cond [B,T,D], X_prior [B,T,D] or None
    Output:  v_pred [B,T,D]   (linear-path FM velocity)

Loss (per branch):
    v_target = x1 - x0
    L_gen = mean over M_real_obs of (v_pred_GEN - v_target)^2   (gen branch, M_cond=0)
    L_imp = mean over M_target of   (v_pred_IMP - v_target)^2   (impute branch)
    L_decor   = off-diag mean of channel-corr on impute-branch bottleneck z
    L_corrmap = MSE of cross-corr (x0 vs x0_pred against z) on impute branch
    No Fourier loss (per user directive).

Training:
    AdamW(lr=5e-4, weight_decay=1e-4), grad_clip 1.0, EMA decay=0.999
    bs=256 with-replacement sampling (no DataLoader); 300K iters
    Window T=64 stride=1 from raw CSV (Stocks 3622, ETTh 17357,
    Energy 19672, KDDCup 10857 windows).

Usage:
    python -u train_r10.py --dataset GENERAL_Stocks \
        --name r10_general_stocks_seed2023 --seed 2023 --gpu 0
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

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
# Dataset CSVs live in the repo's top-level data/ directory (one level up from
# IG_FM/). Override any individual path with --data_csv.
DATA_DIR = os.path.join(os.path.dirname(PROJECT_ROOT), 'data')


# ============================================================
# Data: stride=1 sliding window from raw CSV (matches simple_fm)
# ============================================================
DATASET_CSV_MAP = {
    # name → (csv_path, drop_first_col)
    'GENERAL_Stocks': (
        os.path.join(DATA_DIR, 'stock_data.csv'),
        False,
    ),
    'GENERAL_Etth': (
        os.path.join(DATA_DIR, 'ETTh.csv'),
        True,
    ),
    'GENERAL_Energy': (
        os.path.join(DATA_DIR, 'energy_data.csv'),
        False,
    ),
    'GENERAL_KDDCup': (
        os.path.join(DATA_DIR, 'kddcup.csv'),
        False,
    ),
}


def load_dataset_csv(dataset_name, window=64, seed=2023, csv_override=None):
    """Mirror simple_fm.load_etth: read CSV → MinMaxScaler → [-1, 1] → stride=1
    windows of length `window`, then random permute by seed."""
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
    data_neg = (raw01 * 2.0 - 1.0)  # [-1, 1]
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
# Model: VelocityNet (encoder-bottleneck-decoder Transformer + mask cond)
# ============================================================
class SinusoidalTime(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):  # t: [B] in [0, 1]
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
    """FM linear-path velocity predictor with mask conditioning.

    Inputs:
      x        : [B, T, D]   noisy sample at flow-time t
      t        : [B]         flow time in [0, 1]
      M_cond   : [B, T, D]   conditioning mask (1 = observed, given to model)
      X_cond   : [B, T, D]   M_cond * x0 (observed values; 0 elsewhere)
      X_prior  : [B, T, D]   IDG soft prior from gen branch (or None)
    Output:
      v_pred   : [B, T, D]   predicted linear-path velocity (target = x1 - x0)
    Side effect: stores impute-branch bottleneck z in self._last_bottleneck.
    """

    def __init__(self, feature_size, hidden=128, n_enc_layers=4, n_dec_layers=4,
                 n_heads=8, window=64, dropout=0.0, level_dims=0):
        super().__init__()
        self.in_proj    = nn.Linear(feature_size, hidden)
        self.mask_proj  = nn.Linear(feature_size, hidden)
        self.cond_proj  = nn.Linear(feature_size, hidden, bias=False)
        self.prior_proj = nn.Linear(feature_size, hidden, bias=False)
        # Zero-init the conditioning projections so the baseline (X_cond=0,
        # X_prior=0) reproduces the unconditional FM model exactly at init.
        nn.init.zeros_(self.cond_proj.weight)
        nn.init.zeros_(self.prior_proj.weight)

        self.time_emb = SinusoidalTime(hidden)
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden),
        )
        self.pos_emb = nn.Parameter(torch.zeros(1, window, hidden))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

        layer_kwargs = dict(
            d_model=hidden, nhead=n_heads, dim_feedforward=hidden * 4,
            activation='gelu', batch_first=True, norm_first=True, dropout=dropout,
        )
        enc_layer = nn.TransformerEncoderLayer(**layer_kwargs)
        dec_layer = nn.TransformerEncoderLayer(**layer_kwargs)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_enc_layers)
        self.bottleneck_norm = nn.LayerNorm(hidden)
        self.decoder = nn.TransformerEncoder(dec_layer, num_layers=n_dec_layers)
        self.out_norm = nn.LayerNorm(hidden)
        self.out_proj = nn.Linear(hidden, feature_size)

        # 改动二:0 = 关(原版行为逐位不变)
        self.level_dims = int(level_dims)
        self._last_bottleneck = None
        self._last_bottleneck_raw = None

    def forward(self, x, t, M_cond, X_cond, X_prior=None):
        h = self.in_proj(x) + self.mask_proj(M_cond) + self.cond_proj(X_cond)
        if X_prior is not None:
            h = h + self.prior_proj(X_prior)
        h = h + self.pos_emb[:, : x.shape[1]]
        t_emb = self.time_mlp(self.time_emb(t))            # [B, hidden]
        h = h + t_emb[:, None, :]
        z = self.encoder(h)
        z = self.bottleneck_norm(z)
        # ⚠️ 池化之后那段通道沿时间是常数,方差为 0,而两个辅助损失都要沿时间
        # 做 z-score(除以 std+1e-6),会把相关矩阵炸到 1e6 量级。
        # 所以两份都留:raw 给损失(它塑造编码器的表示),池化后的给解码器
        # (它才是那道容量限制)。两者各司其职,缺一不可。
        self._last_bottleneck_raw = z
        # 改动二:水平那一段沿时间池化,**必须在解码器之前**。
        # 放在 training_step 里池化是无效的 —— 那时解码器已经用过未池化的 z,
        # 只有损失看到了池化结果,信息通路一点没变。这里池化之后,
        # 这段通道对每个时刻是同一个数,结构上装不下逐点痕迹。
        if self.level_dims > 0:
            z = pool_level_subspace(z, self.level_dims)
        self._last_bottleneck = z
        h = self.decoder(z)
        h = self.out_norm(h)
        return self.out_proj(h)


# ============================================================
# IDG mask sampling (per-batch task type) — excludes pure_gen
# ============================================================
def sample_cond_masks_idg(M_real_obs, p_pure_gen=0.10, p_cond_low=0.40,
                          cond_low_lo=0.10, cond_low_hi=0.30,
                          cond_high_lo=0.30, cond_high_hi=0.80):
    """Renormalized IDG sampling: only cond_low or cond_high task. Pure-gen
    is handled by the separate gen branch, so it's excluded here."""
    p_high = 1.0 - p_pure_gen - p_cond_low
    total = p_cond_low + p_high
    r = float(torch.rand(1).item()) * total
    if r < p_cond_low:
        lo, hi = cond_low_lo, cond_low_hi
    else:
        lo, hi = cond_high_lo, cond_high_hi
    ratio = float(torch.empty(1).uniform_(lo, hi).item())
    pick = torch.bernoulli(torch.full_like(M_real_obs, ratio))
    M_target = M_real_obs * pick
    M_cond = M_real_obs * (1.0 - pick)
    return M_cond, M_target


# ============================================================
# Lambda schedule: piecewise constant on iter ranges
# ============================================================
def lambda_schedule(it, schedule):
    """schedule = [(boundary_iter, lambda_val), ...] sorted ascending."""
    for boundary, val in schedule:
        if it < boundary:
            return float(val)
    return float(schedule[-1][1])


# ============================================================
# Loss helpers
# ============================================================
def decor_loss_z(z_bot):
    """K-factor decorrelation on bottleneck z [B, T, hidden]: off-diag of the
    per-batch channel-correlation matrix, mean of squares."""
    zn = z_bot - z_bot.mean(dim=1, keepdim=True)
    zn = zn / (zn.std(dim=1, keepdim=True) + 1e-6)
    T_dim = zn.shape[1]
    corr = torch.einsum('btd,bte->bde', zn, zn) / T_dim
    off_diag = corr - torch.diag_embed(torch.diagonal(corr, dim1=-2, dim2=-1))
    return (off_diag ** 2).mean()


def corrmap_loss_xz(x0, x_pred, z_bot):
    """Cross-corr map MSE: cov(input channels, bottleneck channels) for x0
    vs for x_pred. Mirrors simple_fm.py."""
    T_dim = x0.shape[1]
    def zsc(x):
        return (x - x.mean(dim=1, keepdim=True)) / (x.std(dim=1, keepdim=True) + 1e-6)
    z_x0 = zsc(x0)
    z_xp = zsc(x_pred)
    z_zb = zsc(z_bot)
    M_real = torch.einsum('btd,bth->bdh', z_x0, z_zb) / T_dim
    M_fake = torch.einsum('btd,bth->bdh', z_xp, z_zb) / T_dim
    return F.mse_loss(M_fake, M_real)




# ============================================================
# 隐私模块(三个,各自独立开关,默认全关 -> 原版行为逐位不变)
# ============================================================
# 为什么要动这三处,依据全部来自本项目的实测,不是设计直觉:
#
#   第 3 步测出:泄漏由「绝对水平」和「时间先后顺序」**共同**承载,毁掉其中
#   一样风险只降一半;而一小时以内的细节完全不承载泄漏。
#
#   原版的填充任务是【逐格独立随机遮点】。这意味着哪怕遮掉 80% 的格子,
#   剩下的点仍然能把整体水平平均出来、时序被左右邻居钉死 —— 模型从来不需要
#   自己生成这两样东西,而它们正是承载全部泄漏的两样。模型于是把它们当作
#   「抄过来就行」的旁路信息,这正是记忆最容易发生的地方。
#
# ⚠️ 所有开关默认 False / 0。全关时这几个函数一个都不会被调用,
#    训练路径和 igfm_core.py.stock 完全一致,之前那个质量 0.0394 仍可复现。

def sample_cond_masks_coord(M_real_obs, *, p_block=0.35, p_level=0.25,
                            block_lo=0.15, block_hi=0.55,
                            p_pure_gen=0.10, p_cond_low=0.40,
                            cond_low_lo=0.10, cond_low_hi=0.30,
                            cond_high_lo=0.30, cond_high_hi=0.80):
    """改动一:按坐标遮,而不是逐格随机遮。返回 (M_cond, M_target, task)。

    三种任务按概率抽:
      'block' — 遮掉**连续的时间块**。左右邻居没了,模型必须自己生成先后顺序,
                不能靠插值糊过去。
      'level' — 遮的比例照旧,但条件那一路会在 training_step 里被**去掉整体水平**
                (只有这里返回 task='level' 才会触发),模型必须自己生成绝对水平。
      'cell'  — 原版的逐格随机遮,保留一部分,免得把通用填充能力练没了。

    可证伪的预测:改完之后 'level' 这一档的填充误差应当明显高于 'cell' 档。
    如果两者一样低,说明模型仍在从别处偷水平,这个改动没生效 —— 训练日志里
    按任务分开记 L_imp 就能直接读出来,不用等泄漏基线。
    """
    r = float(torch.rand(1).item())
    B, T, D = M_real_obs.shape
    if r < p_block:
        ratio = float(torch.empty(1).uniform_(block_lo, block_hi).item())
        span = max(1, int(round(T * ratio)))
        # 每个样本自己抽一个起点:整批用同一个起点会让位置本身变成可学的捷径
        start = torch.randint(0, max(1, T - span + 1), (B,), device=M_real_obs.device)
        idx = torch.arange(T, device=M_real_obs.device)[None, :]
        hit = ((idx >= start[:, None]) & (idx < (start + span)[:, None])).float()
        pick = hit[:, :, None].expand(-1, -1, D)
        task = "block"
    elif r < p_block + p_level:
        ratio = float(torch.empty(1).uniform_(cond_high_lo, cond_high_hi).item())
        pick = torch.bernoulli(torch.full_like(M_real_obs, ratio))
        task = "level"
    else:
        p_high = 1.0 - p_pure_gen - p_cond_low
        total = p_cond_low + p_high
        rr = float(torch.rand(1).item()) * total
        lo, hi = ((cond_low_lo, cond_low_hi) if rr < p_cond_low
                  else (cond_high_lo, cond_high_hi))
        ratio = float(torch.empty(1).uniform_(lo, hi).item())
        pick = torch.bernoulli(torch.full_like(M_real_obs, ratio))
        task = "cell"
    return M_real_obs * (1.0 - pick), M_real_obs * pick, task


def strip_level(X_cond, M_cond, eps=1e-6):
    """把条件那一路的**整体水平**拿掉,只留形状。改动一的 'level' 任务用。

    只在被观测到的格子上算均值 —— 拿全体(含被遮的 0)算会把遮挡比例本身
    泄漏进均值里,模型就能反推出水平,任务又白设了。

    返回 (去掉水平的 X_cond, mu)。mu 必须交出去:条件不止 X_cond 一路,
    还有一路 X_prior 也会喂进模型(forward 里 prior_proj 那一项)。
    只处理 X_cond 的话,模型直接从 X_prior 读水平就行,这个任务形同虚设,
    而训练照跑、损失照降 —— per-task 损失还会显示成「模型仍在别处偷水平」,
    结论正好反了。审查抓到的就是这一处。

    两路必须减【同一个】mu:各减各的会把两段之间的相对高低也抹掉,
    而那属于形状,是要留给模型的。
    """
    n = M_cond.sum(dim=1, keepdim=True).clamp(min=1.0)
    mu = (X_cond * M_cond).sum(dim=1, keepdim=True) / n
    return (X_cond - mu) * M_cond, mu



def pool_level_subspace(z, n_level):
    """改动二:把瓶颈的前 n_level 个通道**沿时间池化**再广播回去。

    池化之后这段通道对每个时刻都是同一个数,**结构上装不下逐点的痕迹** ——
    它只能表达「这个窗口整体在什么水平」。这不是靠损失函数劝出来的,
    是把容量直接拿掉,模型想记也记不下。
    """
    if n_level <= 0:
        return z
    lvl = z[..., :n_level].mean(dim=1, keepdim=True).expand(-1, z.shape[1], -1)
    return torch.cat([lvl, z[..., n_level:]], dim=-1)


def decor_loss_groups(z_bot, sizes):
    """改动二配套:只罚**组与组之间**的相关,组内不罚。

    ⚠️ 它罚不到池化真正保留的那个量。这个损失沿时间做 z-score,先把每个通道的
    时间均值减掉了 —— 而时间均值恰恰就是 pool_level_subspace 广播给解码器的东西。
    所以它约束的是【均值之外的波动】,那部分本来就被池化丢掉了。
    真正的保护来自池化那道硬容量限制,这个损失只是在其余两组上起作用。
    不要把它写成「水平子空间是被这个损失隔离出来的」。

    原版的 decor_loss_z 罚所有非对角项,等于要求 128 个通道两两无关 ——
    那既过强(组内本来就该协同),又没说清楚要分开的是什么。
    这里只要求「水平 / 时序 / 细节」三组彼此不串味,组内随便相关。
    """
    zn = z_bot - z_bot.mean(dim=1, keepdim=True)
    zn = zn / (zn.std(dim=1, keepdim=True) + 1e-6)
    corr = torch.einsum('btd,bte->bde', zn, zn) / z_bot.shape[1]
    mask = torch.zeros(corr.shape[-2:], device=z_bot.device)
    o = 0
    for s in sizes:                      # 组内置 0,组间置 1
        mask[o:o + s, o:o + s] = 1.0
        o += s
    cross = corr * (1.0 - mask)
    denom = (1.0 - mask).sum().clamp(min=1.0)
    return (cross ** 2).sum() / (corr.shape[0] * denom)


def isolation_weights(x0, *, bucket=12, knn=5, power=1.0, eps=1e-6):
    """改动三:按窗口给保护力度。越「孤立」的窗口权重越大。

    ⛔ 这一版【不在本次运行里启用】。审查指出部署的统计量和验证过的那个对不上,
       三处:桶宽(验证是 12 个采样点 = 1 小时,原来的参数语义用反了成了 2 小时)、
       近邻阶数(验证用第 5 近邻,原来写的第 1 近邻)、以及最要命的参照池 ——
       验证脚本 scripts/isolation_vs_leak.py 专门排除了【同一个人的其它窗口】,
       并在注释里记着:不排除的话「孤立度」会变成「这个人有多少段记录」的代理,
       能从纯噪声里造出 rho≈0.39。而训练时的微批只有 32 条、只排了对角线,
       撞上同人窗口的概率约 8%,一撞上就主导最近邻。

       前两处已在下面改对。第三处修不了 —— 训练步拿不到受试者编号,
       要正确实现得把编号一路传进来。在那之前这个模块不启用。

    孤立度 = 到同批其它窗口的第 knn 近距离,在按小时平均的粗波形上算。
    权重归一到均值 1,只改力度的【分配】,不改总量。
    """
    B, T, D = x0.shape
    n_buck = max(1, T // max(1, bucket))
    coarse = x0[:, : n_buck * bucket].reshape(B, n_buck, bucket, D).mean(2).reshape(B, -1)
    d = torch.cdist(coarse, coarse)
    d = d + torch.eye(B, device=x0.device) * 1e9        # 排除自己(但排不掉同一个人的其它窗口)
    k = min(max(1, knn), max(1, B - 1))
    nn = d.topk(k, dim=1, largest=False).values[:, -1]
    w = (nn / (nn.mean() + eps)) ** power
    return (w / w.mean().clamp(min=eps)).detach()
# ============================================================
# Single training step: gen branch + impute branch + helper losses
# ============================================================
def training_step(model, x0, M_real_obs, lambda_gen,
                  p_pure_gen, p_cond_low,
                  cond_low_lo, cond_low_hi, cond_high_lo, cond_high_hi,
                  decor_alpha, corrmap_alpha,
                  coord_mask=False, p_block=0.35, p_level=0.25,
                  block_lo=0.15, block_hi=0.55,
                  level_dims=0, order_dims=0,
                  iso_weight=False, iso_bucket=12, iso_knn=5, iso_power=1.0):
    """One IDG step. x0: [B, T, D]; M_real_obs: same shape (all 1s for full data).

    三个隐私模块的开关(默认全关,全关时这个函数和 igfm_core.py.stock 完全一致):
      coord_mask  改动一 按坐标遮(连续时间块 / 去掉水平),取代逐格随机遮
      level_dims  改动二 瓶颈前 N 个通道沿时间池化;和 order_dims 一起决定分组去相关
      iso_weight  改动三 按窗口孤立度分配保护力度
    """
    B, T, D = x0.shape
    device = x0.device

    if coord_mask:
        M_cond, M_target, task = sample_cond_masks_coord(
            M_real_obs, p_block=p_block, p_level=p_level,
            block_lo=block_lo, block_hi=block_hi,
            p_pure_gen=p_pure_gen, p_cond_low=p_cond_low,
            cond_low_lo=cond_low_lo, cond_low_hi=cond_low_hi,
            cond_high_lo=cond_high_lo, cond_high_hi=cond_high_hi,
        )
    else:
        M_cond, M_target = sample_cond_masks_idg(
            M_real_obs, p_pure_gen=p_pure_gen, p_cond_low=p_cond_low,
            cond_low_lo=cond_low_lo, cond_low_hi=cond_low_hi,
            cond_high_lo=cond_high_lo, cond_high_hi=cond_high_hi,
        )
        task = "cell"

    # --- Gen branch (unconditional FM) ---
    x1_g = torch.randn_like(x0)
    t_g = torch.rand(B, device=device)
    x_t_g = (1 - t_g)[:, None, None] * x0 + t_g[:, None, None] * x1_g
    v_target_g = x1_g - x0
    zero = torch.zeros_like(x0)
    v_pred_g = model(x_t_g, t_g, zero, zero, X_prior=None)
    L_gen = (
        ((v_pred_g - v_target_g) ** 2 * M_real_obs).sum()
        / M_real_obs.sum().clamp(min=1.0)
    )
    # x0_pred from velocity (linear path inversion) for IDG soft prior
    x0_pred_g = x_t_g - t_g[:, None, None] * v_pred_g

    # --- Impute branch (conditional FM with X_prior) ---
    x1_i = torch.randn_like(x0)
    t_i = torch.rand(B, device=device)
    x_t_i = (1 - t_i)[:, None, None] * x0 + t_i[:, None, None] * x1_i
    v_target_i = x1_i - x0
    X_cond = M_cond * x0
    X_prior = (1.0 - M_cond) * x0_pred_g.detach()
    # 改动一的 'level' 任务:【两路条件都】只给形状,不给整体水平。
    # X_prior 来自生成分支对真实 x0 的估计,在被遮的位置上原样喂进模型 ——
    # 只处理 X_cond 等于把后门留着,任务形同虚设。审查抓到的就是这一处。
    if coord_mask and task == "level":
        X_cond, mu = strip_level(X_cond, M_cond)
        X_prior = (X_prior - mu) * (1.0 - M_cond)
    v_pred_i = model(x_t_i, t_i, M_cond, X_cond, X_prior=X_prior)
    # 改动三:按窗口分配力度。只改分配不改总量(权重均值已归一到 1)
    w = (isolation_weights(x0, bucket=iso_bucket, knn=iso_knn, power=iso_power)[:, None, None]
         if iso_weight else 1.0)
    L_imp = (
        (((v_pred_i - v_target_i) ** 2) * M_target * w).sum()
        / (M_target * (w if iso_weight else 1.0)).sum().clamp(min=1.0)
    )

    # --- Helper losses on impute-branch bottleneck z ---
    # 用池化【前】的 z:池化后的那段沿时间是常数,z-score 会除以 0。
    # 解码器看到的是池化后的版本(那是容量限制),损失塑造的是编码器的表示。
    z_bot = model._last_bottleneck_raw
    if decor_alpha > 0:
        if level_dims > 0 and order_dims > 0:
            sizes = [level_dims, order_dims, z_bot.shape[-1] - level_dims - order_dims]
            L_decor = decor_loss_groups(z_bot, sizes)
        else:
            L_decor = decor_loss_z(z_bot)
    else:
        L_decor = torch.zeros((), device=device)
    if corrmap_alpha > 0:
        x0_pred_i = x_t_i - t_i[:, None, None] * v_pred_i
        L_corrmap = corrmap_loss_xz(x0, x0_pred_i, z_bot)
    else:
        L_corrmap = torch.zeros((), device=device)

    L_total = (
        L_imp
        + lambda_gen * L_gen
        + decor_alpha * L_decor
        + corrmap_alpha * L_corrmap
    )
    metrics = dict(
        task=task,          # 改动一的可证伪预测靠它:按任务分开看 L_imp
        L_imp=L_imp.item(),
        L_gen=L_gen.item(),
        L_decor=float(L_decor.item()),
        L_corrmap=float(L_corrmap.item()),
        lambda_=float(lambda_gen),
    )
    return L_total, metrics


# ============================================================
# Sampling: Euler ODE, t=1 → 0
# ============================================================
@torch.no_grad()
def sample_unconditional(model, n_samples, T, D, n_steps=200, batch=256, device=None):
    """Euler integration of the FM ODE from x_1 ~ N(0, I) backward to x_0."""
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    out = []
    n_done = 0
    while n_done < n_samples:
        b = min(batch, n_samples - n_done)
        x = torch.randn(b, T, D, device=device)
        zero = torch.zeros_like(x)
        ts = torch.linspace(1.0, 0.0, n_steps + 1, device=device)
        for i in range(n_steps):
            t_cur = ts[i].expand(b)
            v = model(x, t_cur, zero, zero, X_prior=None)
            dt = (ts[i] - ts[i + 1])
            x = x - v * dt
        out.append(x.cpu().numpy())
        n_done += b
    return np.concatenate(out, axis=0).astype(np.float32)


# ============================================================
# Main
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, choices=list(DATASET_CSV_MAP.keys()))
    ap.add_argument('--name', required=True)
    ap.add_argument('--gpu', default='0')
    ap.add_argument('--seed', type=int, default=2023)
    # Config (per-dataset YAML; see dataset_configs.yaml)
    ap.add_argument('--config', default=os.path.join(PROJECT_ROOT, 'dataset_configs.yaml'),
                    help='per-dataset YAML config. Keys window/total_iters/hidden_dim/'
                         'lambda_schedule are taken from the entry matching --dataset unless '
                         'the corresponding flag is passed explicitly on the command line. '
                         'Pass --config "" to disable and use built-in defaults.')
    # Schedule (default None -> resolved from --config by dataset)
    ap.add_argument('--total_iters', type=int, default=None)
    ap.add_argument('--lambda_schedule', default=None,
                    help='comma-sep iter:lambda boundaries; values used while it < boundary')
    ap.add_argument('--ckpt_every', type=int, default=25_000)
    # Optimization
    ap.add_argument('--batch_size', type=int, default=256)
    ap.add_argument('--lr', type=float, default=5e-4)
    ap.add_argument('--weight_decay', type=float, default=1e-4)
    ap.add_argument('--grad_clip', type=float, default=1.0)
    ap.add_argument('--ema_decay', type=float, default=0.999)
    # Architecture
    ap.add_argument('--hidden_dim', type=int, default=None)
    ap.add_argument('--n_enc_layers', type=int, default=4)
    ap.add_argument('--n_dec_layers', type=int, default=4)
    ap.add_argument('--n_heads', type=int, default=8)
    # Loss weights
    ap.add_argument('--decor_alpha', type=float, default=0.1)
    ap.add_argument('--corrmap_alpha', type=float, default=0.1)
    # IDG mask sampling (per batch)
    ap.add_argument('--p_pure_gen', type=float, default=0.10)
    ap.add_argument('--p_cond_low', type=float, default=0.40)
    ap.add_argument('--cond_low_lo', type=float, default=0.10)
    ap.add_argument('--cond_low_hi', type=float, default=0.30)
    ap.add_argument('--cond_high_lo', type=float, default=0.30)
    ap.add_argument('--cond_high_hi', type=float, default=0.80)
    ap.add_argument('--fixed_mask_ratio', type=float, default=None,
                    help='if set, pin all cond_*_lo/hi to this ratio so per-batch '
                         'random drop always picks exactly this mask ratio. '
                         'Used for the r10 step-6 grid (r in {0.2, 0.4, 0.8}).')
    # Sampling (for downstream eval)
    ap.add_argument('--sampling_steps', type=int, default=200,
                    help='Euler steps for the FINAL sample (used by DiMTS eval as canonical fake)')
    ap.add_argument('--milestone_sample_steps', type=int, default=100,
                    help='Euler steps for milestone fakes (cheaper than final, default 100)')
    ap.add_argument('--sample_batch', type=int, default=128)
    ap.add_argument('--num_samples', type=int, default=None,
                    help='number of samples to draw (default = N, dataset size)')
    ap.add_argument('--no_final_sample', action='store_true',
                    help='skip sampling at end of training (for smoke tests)')
    ap.add_argument('--no_milestone_sample', action='store_true',
                    help='skip per-ckpt milestone sampling (for smoke tests)')
    # I/O
    ap.add_argument('--checkpoint_dir', default=os.path.join(PROJECT_ROOT, 'checkpoints'))
    ap.add_argument('--results_dir', default=os.path.join(PROJECT_ROOT, 'results'))
    ap.add_argument('--out_dir', default=os.path.join(PROJECT_ROOT, 'outputs'),
                    help='per-cell dir for truth.npy + fake.npy (DiMTS eval inputs)')
    ap.add_argument('--log_every', type=int, default=500)
    ap.add_argument('--resume', default=None)
    ap.add_argument('--data_csv', default=None,
                    help='override raw CSV path (default: hardcoded per dataset)')
    ap.add_argument('--window', type=int, default=None)
    args = ap.parse_args()

    # ---- Resolve per-dataset config from YAML --------------------------------
    # Precedence per key: explicit CLI flag > config[datasets][<dataset>] >
    # config[defaults] > built-in fallback below. argparse leaves these None when
    # the flag is omitted, which is how we detect "not set on CLI".
    _CFG_DEFAULTS = {
        'window': 64,
        'total_iters': 300_000,
        'hidden_dim': 128,
        'lambda_schedule': '100000:1,200000:2,300000:4',
    }
    cfg = {}
    if args.config and os.path.exists(args.config):
        import yaml
        with open(args.config) as f:
            cfg = yaml.safe_load(f) or {}
        print(f'[r10] loaded config: {args.config}')
    elif args.config:
        print(f'[r10] config file not found ({args.config}); using built-in defaults')
    cfg_defaults = cfg.get('defaults') or {}
    cfg_ds = (cfg.get('datasets') or {}).get(args.dataset) or {}
    for k, builtin in _CFG_DEFAULTS.items():
        if getattr(args, k) is not None:
            print(f'[r10] config: {k:16s}= {getattr(args, k)}  (from CLI)')
            continue
        if k in cfg_ds:
            val, src = cfg_ds[k], f'config:datasets.{args.dataset}'
        elif k in cfg_defaults:
            val, src = cfg_defaults[k], 'config:defaults'
        else:
            val, src = builtin, 'builtin'
        setattr(args, k, val)
        print(f'[r10] config: {k:16s}= {val}  (from {src})')

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # --- Fixed mask ratio override: pin all cond_*_lo/hi to one value ---
    if args.fixed_mask_ratio is not None:
        r = float(args.fixed_mask_ratio)
        assert 0.0 < r < 1.0, f'fixed_mask_ratio must be in (0,1), got {r}'
        args.cond_low_lo = args.cond_low_hi = r
        args.cond_high_lo = args.cond_high_hi = r
        print(f'[r10] fixed_mask_ratio={r} -> all cond_*_lo/hi pinned to {r}')

    # Parse lambda schedule
    schedule = []
    for piece in args.lambda_schedule.split(','):
        b, v = piece.split(':')
        schedule.append((int(b), float(v)))

    # Data
    print(f'[r10] loading {args.dataset} (T={args.window}, stride=1, seed={args.seed})')
    data, scaler = load_dataset_csv(
        args.dataset, window=args.window, seed=args.seed, csv_override=args.data_csv,
    )
    N, T, D = data.shape
    print(f'[r10] data shape (N,T,D) = {data.shape}, '
          f'range [{data.min():.3f}, {data.max():.3f}]')
    data_t = torch.from_numpy(data).to(device)

    # Model
    model = VelocityNet(
        feature_size=D, hidden=args.hidden_dim,
        n_enc_layers=args.n_enc_layers, n_dec_layers=args.n_dec_layers,
        n_heads=args.n_heads, window=T,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'[r10] VelocityNet: {n_params/1e6:.2f}M params (D={D}, T={T}, H={args.hidden_dim}, '
          f'enc={args.n_enc_layers}, dec={args.n_dec_layers}, heads={args.n_heads})')

    # Optim + EMA
    optim = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    ema = torch.optim.swa_utils.AveragedModel(
        model,
        multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(decay=args.ema_decay),
    )

    # Resume
    start_iter = 0
    if args.resume:
        print(f'[r10] resuming from {args.resume}')
        sd = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(sd['model'], strict=False)
        ema.module.load_state_dict(sd['ema_module'])
        if 'ema_n_averaged' in sd:
            ema.n_averaged.fill_(sd['ema_n_averaged'])
        if 'optim' in sd:
            optim.load_state_dict(sd['optim'])
        start_iter = int(sd.get('iter', 0))

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    cell_out_dir = os.path.join(args.out_dir, args.name)
    os.makedirs(cell_out_dir, exist_ok=True)

    # Save real reference (DiMTS-eval format: [0, 1], shape (N, T, D))
    ds_lc = args.dataset.lower().replace('general_', '')
    truth_path = os.path.join(cell_out_dir, f'{ds_lc}_norm_truth_64_train.npy')
    if not os.path.exists(truth_path):
        real01 = ((data + 1.0) / 2.0).astype(np.float32)
        np.save(truth_path, real01)
        print(f'[r10] saved truth -> {truth_path}  shape={real01.shape}')

    M_real_obs = torch.ones(args.batch_size, T, D, device=device)  # full data
    losses = []
    t_start = time.time()
    model.train()
    print(f'[r10] start training: total_iters={args.total_iters} bs={args.batch_size} '
          f'lr={args.lr} ema_decay={args.ema_decay} schedule={args.lambda_schedule}')

    for it in range(start_iter, args.total_iters):
        idx = torch.randint(0, N, (args.batch_size,), device=device)
        x0 = data_t[idx]                         # [B, T, D]
        lam = lambda_schedule(it, schedule)
        loss, met = training_step(
            model, x0, M_real_obs, lam,
            p_pure_gen=args.p_pure_gen, p_cond_low=args.p_cond_low,
            cond_low_lo=args.cond_low_lo, cond_low_hi=args.cond_low_hi,
            cond_high_lo=args.cond_high_lo, cond_high_hi=args.cond_high_hi,
            decor_alpha=args.decor_alpha, corrmap_alpha=args.corrmap_alpha,
        )

        optim.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip and args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optim.step()
        ema.update_parameters(model)
        losses.append(loss.item())

        if it == start_iter or (it + 1) % args.log_every == 0:
            elapsed = time.time() - t_start
            recent = float(np.mean(losses[-args.log_every:]))
            print(
                f'[r10] iter {it+1}/{args.total_iters} ({elapsed:.0f}s) '
                f'loss(recent)={recent:.5f} L_imp={met["L_imp"]:.5f} '
                f'L_gen={met["L_gen"]:.5f} L_decor={met["L_decor"]:.5f} '
                f'L_corrmap={met["L_corrmap"]:.5f} lam={lam:g}',
                flush=True,
            )

        if (it + 1) % args.ckpt_every == 0 and (it + 1) < args.total_iters:
            mid_path = os.path.join(args.checkpoint_dir, f'{args.name}_{it+1}.pt')
            torch.save({
                'model': model.state_dict(),
                'ema_module': ema.module.state_dict(),
                'ema_n_averaged': int(ema.n_averaged.item()),
                'optim': optim.state_dict(),
                'iter': it + 1,
                'config': vars(args),
            }, mid_path)
            print(f'[r10] mid-ckpt @ iter {it+1} -> {mid_path}', flush=True)

            # Milestone sampling: cheap fake from EMA for click-through eval.
            if not args.no_milestone_sample:
                ms_n = args.num_samples if args.num_samples is not None else N
                t_ms = time.time()
                ema.eval()
                ms_fake_neg = sample_unconditional(
                    ema, n_samples=ms_n, T=T, D=D,
                    n_steps=args.milestone_sample_steps,
                    batch=args.sample_batch, device=device,
                )
                ms_fake01 = ((ms_fake_neg + 1.0) / 2.0).astype(np.float32)
                ms_fake_path = os.path.join(
                    cell_out_dir, f'{args.name}_M{(it+1)//1000}K_fake.npy',
                )
                np.save(ms_fake_path, ms_fake01)
                model.train()
                print(f'[r10] milestone fake @ iter {it+1} -> {ms_fake_path} '
                      f'({(time.time()-t_ms)/60:.1f} min, {args.milestone_sample_steps} steps)',
                      flush=True)

    train_time = time.time() - t_start
    final_path = os.path.join(args.checkpoint_dir, f'{args.name}.pt')
    torch.save({
        'model': model.state_dict(),
        'ema_module': ema.module.state_dict(),
        'ema_n_averaged': int(ema.n_averaged.item()),
        'optim': optim.state_dict(),
        'iter': args.total_iters,
        'config': vars(args),
    }, final_path)
    print(f'[r10] done. train_time={train_time/60:.1f} min, final -> {final_path}')

    # ===== Final unconditional sampling from EMA for DiMTS eval =====
    sample_time = 0.0
    fake_path = None
    if not args.no_final_sample:
        n_sample = args.num_samples if args.num_samples is not None else N
        print(f'[r10] sampling {n_sample} from EMA: {args.sampling_steps} Euler steps, '
              f'batch={args.sample_batch}', flush=True)
        t_sample0 = time.time()
        ema.eval()
        fake_neg = sample_unconditional(
            ema, n_samples=n_sample, T=T, D=D,
            n_steps=args.sampling_steps, batch=args.sample_batch, device=device,
        )
        # DiMTS eval expects [0, 1]
        fake01 = ((fake_neg + 1.0) / 2.0).astype(np.float32)
        fake_path = os.path.join(cell_out_dir, f'{args.name}_fake.npy')
        np.save(fake_path, fake01)
        sample_time = time.time() - t_sample0
        print(f'[r10] sampled fake.npy shape={fake01.shape} '
              f'range=[{fake01.min():.3f},{fake01.max():.3f}] '
              f'in {sample_time/60:.1f} min -> {fake_path}', flush=True)

    # JSON log (sub-sampled loss curve)
    sub = max(1, len(losses) // 200)
    log_path = os.path.join(args.results_dir, f'{args.name}.json')
    with open(log_path, 'w') as f:
        json.dump({
            'name': args.name,
            'dataset': args.dataset,
            'config': vars(args),
            'lambda_schedule': schedule,
            'n_params': n_params,
            'data_shape': list(data.shape),
            'train_time_sec': train_time,
            'sample_time_sec': sample_time,
            'final_fake_path': fake_path,
            'truth_path': truth_path,
            'final_loss': float(np.mean(losses[-200:])),
            'loss_curve': [
                {'iter': i * sub + 1, 'loss': float(losses[i * sub])}
                for i in range(len(losses) // sub)
            ],
        }, f, indent=2)
    print(f'[r10] log -> {log_path}')


if __name__ == '__main__':
    main()
