"""Neural generation-quality metrics: Context-FID, Discriminative, Predictive.

These follow the DiM-TS / TimeGAN reference recipes as closely as a PyTorch
reimplementation allows:

  * Context-FID  -> uses the vendored official **TS2Vec** encoder
    (tsgen_metrics/_vendor/ts2vec) exactly as DiM-TS's Utils/context_fid.py does
    (input_dims=C, batch_size=8, lr=1e-3, output_dims=320, max_train_length=3000),
    trained on REAL only, then the standard FID between real/fake representations.
    Faithful (TS2Vec is itself PyTorch); residual differences are only TS2Vec's
    own training stochasticity (the official also averages several runs).
  * Discriminative / Predictive -> GRU recipes with the official hyper-parameters
    (hidden = C//2, batch = 128, iterations = 2000 / 5000, Adam lr=1e-3, tanh GRU).
    These are TRAINED nets: stochastic and TF-vs-PyTorch internals differ, so they
    reproduce the official RECIPE and track its ranking, but are NOT bit-identical
    to the official TF numbers. For the exact official values run DiM-TS/PaD-TS
    run_eval.py directly.

All inputs are float32 arrays shaped (N, T, C).
"""
from __future__ import annotations

import numpy as np
import scipy
import torch
import torch.nn as nn


def _seed_all(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _to_tensor(x, device):
    return torch.as_tensor(np.asarray(x, dtype=np.float32), device=device)


def _ts2vec_device(device):
    """TS2Vec wants a torch device / cuda index. Map our 'cuda'/'cpu' string."""
    d = str(device)
    if d.startswith("cuda"):
        return 0 if ":" not in d else int(d.split(":")[1])
    return "cpu"


# --------------------------------------------------------------------------- #
# 1. Context-FID  (official TS2Vec + standard FID; faithful to DiM-TS)
# --------------------------------------------------------------------------- #
def _calculate_fid(act1, act2):
    """Standard FID between two activation sets — identical to DiM-TS calculate_fid."""
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    covmean = scipy.linalg.sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean))


def context_fid(real, fake, device, seed=0, **_ignored) -> dict:
    """Context-FID via the official TS2Vec encoder (DiM-TS Utils/context_fid.py).

    Trains TS2Vec on REAL, encodes real & fake to full-series representations, and
    returns the FID between them. Lower is better.
    """
    _seed_all(seed)
    from ._vendor.ts2vec.ts2vec import TS2Vec

    real = np.asarray(real, dtype=np.float32)
    fake = np.asarray(fake, dtype=np.float32)
    model = TS2Vec(input_dims=real.shape[-1], device=_ts2vec_device(device),
                   batch_size=8, lr=0.001, output_dims=320, max_train_length=3000)
    model.fit(real, verbose=False)              # default n_iters (faithful to official)
    rep_real = model.encode(real, encoding_window="full_series")
    rep_fake = model.encode(fake, encoding_window="full_series")
    idx = np.random.permutation(real.shape[0])  # FID is permutation-invariant (cosmetic, as official)
    value = _calculate_fid(rep_real[idx], rep_fake[idx])
    return {
        "value": value,
        "lower_is_better": True,
        "definition": "Context-FID: Frechet distance between real and generated TS2Vec full-series representations (encoder trained on real). Faithful to DiM-TS Utils/context_fid.py.",
        "config": {"encoder": "TS2Vec (official, vendored)", "output_dims": 320,
                   "batch_size": 8, "lr": 0.001, "max_train_length": 3000, "seed": seed},
    }


# --------------------------------------------------------------------------- #
# 2. Discriminative score  (official GRU recipe: hidden=C//2, iters=2000, batch=128)
# --------------------------------------------------------------------------- #
class _GRUClassifier(nn.Module):
    def __init__(self, in_ch, hidden):
        super().__init__()
        self.gru = nn.GRU(in_ch, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        _, h = self.gru(x)
        return self.fc(h[-1]).squeeze(-1)


def discriminative_score(real, fake, device, iterations=2000, batch=128,
                         seed=0, test_frac=0.2, **_ignored) -> dict:
    """|0.5 - test accuracy| of a GRU real/fake classifier (lower = better).

    Faithful to the DiM-TS/TimeGAN recipe: hidden = C//2, batch = 128, 2000
    iterations of Adam(lr=1e-3); each iteration draws a real batch and a fake
    batch and minimises BCE(real->1) + BCE(fake->0). Stochastic (see module doc).
    """
    _seed_all(seed)
    C = real.shape[2]
    hidden = max(C // 2, 1)
    rng = np.random.default_rng(seed)

    def split(a):
        a = np.asarray(a, dtype=np.float32)
        idx = rng.permutation(a.shape[0])
        n_test = int(a.shape[0] * test_frac)
        return a[idx[n_test:]], a[idx[:n_test]]

    r_tr, r_te = split(real)
    f_tr, f_te = split(fake)
    r_tr_t, f_tr_t = _to_tensor(r_tr, device), _to_tensor(f_tr, device)

    model = _GRUClassifier(C, hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    bce = nn.BCEWithLogitsLoss()
    ones = torch.ones(batch, device=device)
    zeros = torch.zeros(batch, device=device)
    model.train()
    for _ in range(iterations):
        ri = torch.as_tensor(rng.integers(0, r_tr_t.shape[0], batch), device=device)
        fi = torch.as_tensor(rng.integers(0, f_tr_t.shape[0], batch), device=device)
        opt.zero_grad()
        loss = bce(model(r_tr_t[ri]), ones) + bce(model(f_tr_t[fi]), zeros)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        Xte = _to_tensor(np.concatenate([r_te, f_te]), device)
        yte = _to_tensor(np.concatenate([np.ones(len(r_te)), np.zeros(len(f_te))]), device)
        preds = []
        for s in range(0, Xte.shape[0], 512):
            preds.append((torch.sigmoid(model(Xte[s:s + 512])) > 0.5).float())
        acc = float((torch.cat(preds) == yte).float().mean().item())
    return {
        "value": abs(0.5 - acc),
        "lower_is_better": True,
        "definition": "|0.5 - test accuracy| of a GRU classifier trained to tell real from generated (0 = indistinguishable). Official recipe (hidden=C//2, iters=2000, batch=128).",
        "config": {"hidden": hidden, "iterations": iterations, "batch": batch,
                   "test_frac": test_frac, "seed": seed, "test_accuracy": acc},
    }


# --------------------------------------------------------------------------- #
# 3. Predictive score (TSTR)  (official GRU recipe: hidden=C//2, iters=5000, batch=128)
# --------------------------------------------------------------------------- #
class _GRUPredictor(nn.Module):
    def __init__(self, in_ch, hidden, out_ch):
        super().__init__()
        self.gru = nn.GRU(in_ch, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, out_ch)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out)


def predictive_score(real, fake, device, iterations=5000, batch=128, seed=0,
                     **_ignored) -> dict:
    """TSTR one-step-ahead MAE: GRU trained on FAKE, evaluated on REAL (lower=better).

    Faithful to DiM-TS/TimeGAN: features [0..C-2] predict the LAST feature at the
    next step; hidden = C//2, batch = 128, 5000 iterations of Adam(lr=1e-3).
    Single-channel data falls back to next-step self-prediction. Stochastic.
    """
    _seed_all(seed)
    C = real.shape[2]
    hidden = max(C // 2, 1)
    multichannel = C > 1
    in_ch = C - 1 if multichannel else 1
    rng = np.random.default_rng(seed)

    def make_xy(a):
        a = np.asarray(a, dtype=np.float32)
        if multichannel:
            return a[:, :-1, :-1], a[:, 1:, -1:]
        return a[:, :-1, :], a[:, 1:, :]

    Xf, Yf = make_xy(fake)
    Xr, Yr = make_xy(real)
    Xf_t, Yf_t = _to_tensor(Xf, device), _to_tensor(Yf, device)

    model = _GRUPredictor(in_ch, hidden, 1).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    l1 = nn.L1Loss()
    model.train()
    for _ in range(iterations):
        bi = torch.as_tensor(rng.integers(0, Xf_t.shape[0], batch), device=device)
        opt.zero_grad()
        loss = l1(model(Xf_t[bi]), Yf_t[bi])
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        Xr_t, Yr_t = _to_tensor(Xr, device), _to_tensor(Yr, device)
        errs = []
        for s in range(0, Xr_t.shape[0], 512):
            p = model(Xr_t[s:s + 512])
            errs.append(torch.abs(p - Yr_t[s:s + 512]).mean().item() * min(512, Xr_t.shape[0] - s))
        mae = float(sum(errs) / Xr_t.shape[0])
    return {
        "value": mae,
        "lower_is_better": True,
        "definition": "TSTR one-step-ahead MAE: GRU trained on generated to predict the next-step last feature, evaluated on real. Official recipe (hidden=C//2, iters=5000, batch=128).",
        "config": {"hidden": hidden, "iterations": iterations, "batch": batch, "seed": seed},
    }
