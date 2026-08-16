"""Group D -- learned representation (D11, D12).

Deliberately thin. Wu & Keogh, and the TimeSeAD benchmark, both find that deep anomaly
detectors' advantage over classical methods is largely an artefact of flawed
evaluation; loading the set with more of them buys apparent coverage, not evidence.

The encoder is FROZEN and fit on windows only -- no labels, no synthetic data, one fit
shared by every subject. It is also NOT the encoder any quality metric uses: sharing
one would make "good quality" and "resists attack" two projections of one space, so a
privacy-utility plot would show a relationship created by construction.
"""
from __future__ import annotations

import numpy as np

from .common import SEED, subject_slices


def D11_lof(E, sids, k=20):
    """Density relative to a subject's own neighbours rather than to the cohort centre.
    A small, sparse cluster of similar subjects scores high here and near zero on any
    centre-distance method."""
    from sklearn.neighbors import LocalOutlierFactor
    subs, sl = subject_slices(sids)
    Es = np.stack([np.asarray(E)[sl[s]].mean(0) for s in subs])
    return -LocalOutlierFactor(n_neighbors=k).fit(Es).negative_outlier_factor_


def D12_autoencoder_max(flat, sids, epochs=20, latent=32, device="cuda", seed=SEED,
                        verbose=True):
    """Per-window reconstruction error, MAX over a subject's days.

    The only method in the set that targets one unusual day directly. Everything else
    averages it away, and memorisation attaches to a specific window. The 95th
    percentile is returned alongside the max because a max over 700 days is one bad
    sensor read away from being noise.
    """
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    dev = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
    T = flat.shape[1]

    class AE(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = nn.Sequential(
                nn.Conv1d(1, 16, 7, 2, 3), nn.ReLU(),
                nn.Conv1d(16, 32, 7, 2, 3), nn.ReLU(),
                nn.Conv1d(32, latent, 7, 2, 3))
            self.dec = nn.Sequential(
                nn.ConvTranspose1d(latent, 32, 8, 2, 3), nn.ReLU(),
                nn.ConvTranspose1d(32, 16, 8, 2, 3), nn.ReLU(),
                nn.ConvTranspose1d(16, 1, 8, 2, 3))

        def forward(self, x):
            return self.dec(self.enc(x))[..., :x.shape[-1]]

    m = AE().to(dev)
    opt = torch.optim.Adam(m.parameters(), 1e-3)
    Xt = torch.from_numpy(np.ascontiguousarray(flat, np.float32)).unsqueeze(1)
    dl = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(Xt),
                                     batch_size=256, shuffle=True, drop_last=True)
    for ep in range(epochs):
        tot = 0.0
        for (b,) in dl:
            b = b.to(dev)
            loss = ((m(b) - b) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss)
        if verbose and (ep + 1) % 5 == 0:
            print(f"  AE epoch {ep+1}/{epochs} loss {tot/len(dl):.6f}", flush=True)

    m.eval(); errs = []
    with torch.no_grad():
        for i in range(0, Xt.shape[0], 2048):
            b = Xt[i:i + 2048].to(dev)
            errs.append(((m(b) - b) ** 2).mean(dim=(1, 2)).cpu().numpy())
    err = np.concatenate(errs)
    subs, sl = subject_slices(sids)
    return (np.array([err[sl[s]].max() for s in subs]),
            np.array([np.quantile(err[sl[s]], 0.95) for s in subs]))
