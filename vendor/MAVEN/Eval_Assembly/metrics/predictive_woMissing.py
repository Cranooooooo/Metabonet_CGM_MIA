"""Predictive score (no missingness).

Post-hoc one-step-ahead RNN trained on synthetic data and evaluated on real:
input is the first dim-1 channels (history), target is the next-step value
of the LAST channel. MAE on the original (real) data is reported. Identical
protocol to eval_ref_1/woMissing_scripts/Utils/predictive_metric.py.

For C=1, the target is the only channel and the input collapses to width 0;
in that case we shift to a single-channel autoregressive setup (input = x_t,
target = x_{t+1}) so the metric is still well-defined.

Inputs : real, fake of shape (N, T, C).
Returns: scalar MAE on real (lower is better).
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from sklearn.metrics import mean_absolute_error
from tqdm.auto import tqdm


class _Predictor(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.gru = nn.GRU(in_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        return torch.sigmoid(self.fc(out))


def predictive_score(ori_data, generated_data,
                     iterations: int = 5000, batch_size: int = 128) -> float:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    no, seq_len, dim = np.asarray(ori_data).shape
    hidden_dim = max(1, int(dim / 2))

    if dim == 1:
        in_slice = lambda x: x[:-1, :1]
        tgt_slice = lambda x: np.reshape(x[1:, 0], [-1, 1])
        in_dim = 1
    else:
        in_slice = lambda x: x[:-1, :(dim - 1)]
        tgt_slice = lambda x: np.reshape(x[1:, dim - 1], [-1, 1])
        in_dim = dim - 1

    model = _Predictor(in_dim=in_dim, hidden_dim=hidden_dim).to(device)
    opt = torch.optim.Adam(model.parameters())
    l1 = nn.L1Loss()

    model.train()
    for _ in tqdm(range(iterations), desc="pred-train", total=iterations):
        idx = np.random.permutation(len(generated_data))[:batch_size]
        X_mb = [in_slice(generated_data[i]) for i in idx]
        Y_mb = [tgt_slice(generated_data[i]) for i in idx]
        X = torch.from_numpy(np.asarray(X_mb, dtype=np.float32)).to(device)
        Y = torch.from_numpy(np.asarray(Y_mb, dtype=np.float32)).to(device)
        loss = l1(model(X), Y)
        opt.zero_grad(); loss.backward(); opt.step()

    model.eval()
    idx = np.random.permutation(len(ori_data))[:no]
    X_mb = [in_slice(ori_data[i]) for i in idx]
    Y_mb = [tgt_slice(ori_data[i]) for i in idx]
    # Batch inference: a single forward over N=10857 high-D sequences OOMs
    # alongside a training process (h=512 KDDCup leaves <1 GB free), so chunk it.
    X_all = np.asarray(X_mb, dtype=np.float32)
    chunks = []
    eval_bs = 256
    with torch.no_grad():
        for s in range(0, no, eval_bs):
            X = torch.from_numpy(X_all[s:s + eval_bs]).to(device)
            chunks.append(model(X).cpu().numpy())
    pred = np.concatenate(chunks, axis=0)
    return float(sum(mean_absolute_error(Y_mb[i], pred[i]) for i in range(no)) / no)
