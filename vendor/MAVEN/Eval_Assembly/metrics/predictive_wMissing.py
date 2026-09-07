"""Predictive score (with missingness — real-only mask).

Generated samples are assumed to be fully observed. To make the predictor see
the same gap regime at train time as it will encounter at test time on real,
we draw a synthetic fake-mask from the real_mask pool and zero-fill fake
accordingly before training.

Input is the first dim-1 channels concatenated with their mask
(2*(dim-1) channels — mask channel kept here for the predictor's benefit;
unlike discriminative, this metric is regression, not classification, so the
mask is auxiliary input, not a discriminative leak).

Loss / metric are masked: only positions where the target value is actually
observed contribute. Falls back exactly to the no-missing path when ori_mask
is fully observed.

Inputs : real, fake of shape (N, T, C); ori_mask of shape (N, T, C) bool.
Returns: scalar masked MAE on real (lower is better).
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from tqdm.auto import tqdm

from utils.mask_utils import is_fully_observed, sanitize, resample_masks
from metrics.predictive_woMissing import predictive_score as _pred_clean


class _Predictor(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.gru = nn.GRU(in_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        return torch.sigmoid(self.fc(out))


def predictive_score_masked(ori_data, generated_data,
                            ori_mask=None,
                            iterations: int = 5000, batch_size: int = 128,
                            seed: int = 0) -> float:
    if is_fully_observed(ori_mask):
        return _pred_clean(ori_data, generated_data,
                           iterations=iterations, batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Mo = np.asarray(ori_mask).astype(bool)
    real = sanitize(ori_data, Mo)

    # Synthesize a fake-mask from the real-mask pool and apply it to fake so
    # train-time gap regime matches test-time gap regime.
    rng = np.random.default_rng(seed)
    Mg = resample_masks(generated_data.shape[0], Mo, rng=rng)
    fake = sanitize(generated_data, Mg)

    no, seq_len, dim = real.shape
    hidden_dim = max(1, int(dim / 2))

    if dim == 1:
        in_dim = 2  # autoregressive single-channel: history value + history mask
        def build_batch(X, M, idx):
            Xb = X[idx]; Mb = M[idx].astype(np.float32)
            feat   = Xb[:, :-1, :1] * Mb[:, :-1, :1]
            feat_m = Mb[:, :-1, :1]
            Xin = np.concatenate([feat, feat_m], axis=-1).astype(np.float32)
            Yt  = Xb[:, 1:, :1].astype(np.float32)
            Yt_m = Mb[:, 1:, :1].astype(np.float32)
            return Xin, Yt, Yt_m
    else:
        in_dim = 2 * (dim - 1)
        def build_batch(X, M, idx):
            Xb = X[idx]; Mb = M[idx].astype(np.float32)
            feat   = Xb[:, :-1, :(dim - 1)] * Mb[:, :-1, :(dim - 1)]
            feat_m = Mb[:, :-1, :(dim - 1)]
            Xin = np.concatenate([feat, feat_m], axis=-1).astype(np.float32)
            Yt   = Xb[:, 1:, (dim - 1):dim].astype(np.float32)
            Yt_m = Mb[:, 1:, (dim - 1):dim].astype(np.float32)
            return Xin, Yt, Yt_m

    model = _Predictor(in_dim=in_dim, hidden_dim=hidden_dim).to(device)
    opt = torch.optim.Adam(model.parameters())

    model.train()
    n_gen = fake.shape[0]
    for _ in tqdm(range(iterations), desc="pred-train", total=iterations):
        idx = np.random.permutation(n_gen)[:batch_size]
        Xin, Yt, Ytm = build_batch(fake, Mg, idx)
        Xin = torch.from_numpy(Xin).to(device)
        Yt  = torch.from_numpy(Yt).to(device)
        Ytm = torch.from_numpy(Ytm).to(device)
        pred = model(Xin)
        denom = Ytm.sum().clamp_min(1.0)
        loss = (torch.abs(pred - Yt) * Ytm).sum() / denom
        opt.zero_grad(); loss.backward(); opt.step()

    model.eval()
    idx = np.random.permutation(no)[:no]
    Xin, Yt, Ytm = build_batch(real, Mo, idx)
    with torch.no_grad():
        Xin_t = torch.from_numpy(Xin).to(device)
        pred = model(Xin_t).cpu().numpy()
    mask = Ytm.astype(bool).reshape(-1)
    err = np.abs(pred.reshape(-1) - Yt.reshape(-1))
    if mask.sum() == 0:
        return float("nan")
    return float(err[mask].mean())
