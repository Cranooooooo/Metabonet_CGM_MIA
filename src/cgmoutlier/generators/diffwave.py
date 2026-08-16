"""DiffWave generator adapter (wraps vendor/DiffWave).

Wraps philsyn/DiffWave-unconditional's WaveNet (a channel-agnostic dilated-conv
denoiser) for unconditional multivariate time-series generation. Replicates the
train+sample recipe in the IGFM reference run_diffwave_ts.py:

  * fit:    DDPM eps-prediction training via `training_loss` (Adam, MSE) over
            (N, T, C) arrays fed as (B, C, T) tensors.
  * sample: ancestral sampling via `sampling`, returning (n, T, C).

The WaveNet is range-agnostic (it predicts the injected noise), so the ~[-1, 1]
inputs are used as-is. save/load persists the net state_dict plus a small JSON
config; the diffusion schedule is recomputed from (T, beta_0, beta_T) on load.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np

from .base import GeneratorBase

# vendor/DiffWave must be on sys.path so "from WaveNet import ..." / "from util
# import ..." resolve (the vendored modules use top-level absolute imports).
_VENDOR_SRC = Path(__file__).resolve().parents[3] / "vendor" / "DiffWave"


def _ensure_vendor_on_path() -> None:
    p = str(_VENDOR_SRC)
    if p not in sys.path:
        sys.path.insert(0, p)


def _set_seeds(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class DiffWaveGenerator(GeneratorBase):
    """DiffWave adapter. params: res_channels, skip_channels, num_res_layers,
    dilation_cycle, diffusion_steps (T), beta_0, beta_T. train_cfg: batch_size,
    max_epochs (or total_iters), lr."""

    _CKPT = "diffwave.pt"
    _CFG = "diffwave_config.json"

    # -- internals --------------------------------------------------------
    def _resolve_device(self) -> str:
        import torch

        want = str(self.device)
        if want.startswith("cuda") and not torch.cuda.is_available():
            return "cpu"
        return want

    def _build_net(self):
        _ensure_vendor_on_path()
        from WaveNet import WaveNet_Speech_Commands  # noqa: E402

        p = self.params
        net = WaveNet_Speech_Commands(
            in_channels=self.C,
            out_channels=self.C,
            res_channels=int(p.get("res_channels", 128)),
            skip_channels=int(p.get("skip_channels", 128)),
            num_res_layers=int(p.get("num_res_layers", 12)),
            dilation_cycle=int(p.get("dilation_cycle", 6)),
        )
        return net

    def _build_diffusion_hp(self, device):
        _ensure_vendor_on_path()
        from util import calc_diffusion_hyperparams  # noqa: E402

        p = self.params
        hp = calc_diffusion_hyperparams(
            int(p.get("diffusion_steps", 200)),
            float(p.get("beta_0", 1e-4)),
            float(p.get("beta_T", 0.02)),
        )
        for k in hp:
            if k != "T":
                hp[k] = hp[k].to(device)
        return hp

    # -- core API ---------------------------------------------------------
    def fit(self, X: np.ndarray, train_cfg: Dict[str, Any] | None = None) -> "DiffWaveGenerator":
        import torch
        import torch.nn as nn

        X = self._check_X(X)  # (N, T, C) float32
        cfg = dict(train_cfg or {})
        self._device = self._resolve_device()
        _set_seeds(self.seed)

        net = self._build_net().to(self._device)
        diffusion_hp = self._build_diffusion_hp(self._device)

        bs = int(cfg.get("batch_size", 64))
        bs = max(1, min(bs, X.shape[0]))
        lr = float(cfg.get("lr", 2e-4))

        # iteration budget: total_iters takes precedence, else max_epochs * iters/epoch.
        # (N, T, C) -> (N, C, T) tensors for the conv1d WaveNet.
        data = torch.from_numpy(X).float().permute(0, 2, 1).contiguous()
        ds = torch.utils.data.TensorDataset(data)
        drop_last = data.shape[0] > bs
        loader = torch.utils.data.DataLoader(
            ds, batch_size=bs, shuffle=True, num_workers=0,
            pin_memory=False, drop_last=drop_last,
        )
        iters_per_epoch = max(1, len(loader))
        if cfg.get("total_iters") is not None:
            total_iters = int(cfg["total_iters"])
        else:
            total_iters = int(cfg.get("max_epochs", 100)) * iters_per_epoch
        total_iters = max(1, total_iters)

        from util import training_loss  # noqa: E402

        opt = torch.optim.Adam(net.parameters(), lr=lr)
        mse = nn.MSELoss()

        net.train()
        n_iter = 0
        while n_iter < total_iters:
            for (x,) in loader:
                if n_iter >= total_iters:
                    break
                x = x.to(self._device)  # (B, C, T)
                opt.zero_grad()
                loss = training_loss(net, mse, x, diffusion_hp)
                loss.backward()
                opt.step()
                n_iter += 1

        self._net = net
        self._diffusion_hp = diffusion_hp
        self._fitted = True
        return self

    def sample(self, n: int, sample_cfg: Dict[str, Any] | None = None) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("DiffWaveGenerator.sample before fit")
        import torch

        from util import sampling  # noqa: E402

        cfg = dict(sample_cfg or {})
        n = int(n)
        batch = int(cfg.get("sample_batch", 128))
        batch = max(1, min(batch, n))

        self._net.eval()
        out = []
        remaining = n
        with torch.no_grad():
            while remaining > 0:
                b = min(batch, remaining)
                x = sampling(self._net, (b, self.C, self.T), self._diffusion_hp)  # (b, C, T)
                out.append(x.detach().cpu())
                remaining -= b
        fakes = torch.cat(out, dim=0)  # (n, C, T)
        fakes = fakes.permute(0, 2, 1).contiguous().numpy()  # (n, T, C)
        return fakes[:n].astype(np.float32)

    def save(self, path: str) -> None:
        if not self._fitted:
            raise RuntimeError("nothing to save")
        import torch

        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        torch.save(self._net.state_dict(), str(p / self._CKPT))
        cfg = {
            "T": self.T,
            "C": self.C,
            "params": self.params,
        }
        with open(p / self._CFG, "w") as fh:
            json.dump(cfg, fh, indent=2)

    def load(self, path: str) -> "DiffWaveGenerator":
        import torch

        p = Path(path)
        with open(p / self._CFG) as fh:
            cfg = json.load(fh)
        self.T = int(cfg.get("T", self.T))
        self.C = int(cfg.get("C", self.C))
        if "params" in cfg:                       # restore even an empty/all-default dict
            self.params = dict(cfg["params"])

        self._device = self._resolve_device()
        net = self._build_net()
        state = torch.load(str(p / self._CKPT), map_location=self._device)
        net.load_state_dict(state)
        self._net = net.to(self._device)
        self._diffusion_hp = self._build_diffusion_hp(self._device)
        self._fitted = True
        return self
