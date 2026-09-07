"""Discriminative score (no missingness).

Post-hoc GRU classifier trained to distinguish real from synthetic windows.
Score = |0.5 - test_accuracy|: 0 is perfectly indistinguishable, 0.5 is
perfectly separable. Identical protocol to
eval_ref_1/woMissing_scripts/Utils/discriminative_metric.py.

Inputs : real, fake of shape (N, T, C).
Returns: (discriminative_score, fake_acc, real_acc).
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from sklearn.metrics import accuracy_score
from tqdm.auto import tqdm

from utils.metric_utils import train_test_divide, extract_time


class _Discriminator(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.gru = nn.GRU(dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, h = self.gru(x)
        return self.fc(h.squeeze(0))


def _batch(data, time, batch_size):
    no = len(data)
    idx = np.random.permutation(no)[:batch_size]
    return [data[i] for i in idx], [time[i] for i in idx]


def discriminative_score(ori_data, generated_data,
                         iterations: int = 2000, batch_size: int = 128):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    no, seq_len, dim = np.asarray(ori_data).shape
    ori_time, _ = extract_time(ori_data)
    gen_time, _ = extract_time(ori_data)

    hidden_dim = max(1, int(dim / 2))

    train_x, train_x_hat, test_x, test_x_hat, train_t, train_t_hat, test_t, test_t_hat = (
        train_test_divide(ori_data, generated_data, ori_time, gen_time)
    )

    model = _Discriminator(dim, hidden_dim).to(device)
    opt = torch.optim.Adam(model.parameters())
    bce = nn.BCEWithLogitsLoss()

    model.train()
    for _ in tqdm(range(iterations), desc="disc-train", total=iterations):
        X_mb, _ = _batch(train_x, train_t, batch_size)
        Xh_mb, _ = _batch(train_x_hat, train_t_hat, batch_size)
        x_real = torch.from_numpy(np.asarray(X_mb, dtype=np.float32)).to(device)
        x_fake = torch.from_numpy(np.asarray(Xh_mb, dtype=np.float32)).to(device)
        logit_real = model(x_real)
        logit_fake = model(x_fake)
        loss = bce(logit_real, torch.ones_like(logit_real)) + bce(
            logit_fake, torch.zeros_like(logit_fake)
        )
        opt.zero_grad(); loss.backward(); opt.step()

    model.eval()
    with torch.no_grad():
        x_real_te = torch.from_numpy(np.asarray(test_x, dtype=np.float32)).to(device)
        x_fake_te = torch.from_numpy(np.asarray(test_x_hat, dtype=np.float32)).to(device)
        y_pred_real = torch.sigmoid(model(x_real_te)).squeeze(-1).cpu().numpy()
        y_pred_fake = torch.sigmoid(model(x_fake_te)).squeeze(-1).cpu().numpy()

    y_pred = np.concatenate([y_pred_real, y_pred_fake])
    y_label = np.concatenate([np.ones(len(y_pred_real)), np.zeros(len(y_pred_fake))])
    acc = accuracy_score(y_label, (y_pred > 0.5))
    fake_acc = accuracy_score(np.zeros(len(y_pred_fake)), (y_pred_fake > 0.5))
    real_acc = accuracy_score(np.ones(len(y_pred_real)), (y_pred_real > 0.5))
    return float(np.abs(0.5 - acc)), float(fake_acc), float(real_acc)
