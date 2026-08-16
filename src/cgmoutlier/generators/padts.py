"""PaD-TS generator adapter (wraps vendor/PaD-TS).

Trains the dual-tower DiT `PaD_TS` (a time tower + a feature tower) with a
Gaussian diffusion MSE+MMD objective, then samples via the reverse diffusion
``p_sample_loop``. Mirrors the recipe in the reference ``run_padts_ts.py``
(model build via DATASET_HP, ``create_gaussian_diffusion(predict_xstart=True,
noise_schedule='cosine', loss='MSE_MMD')``, ``Batch_Same_Sampler``, AdamW with
linear LR anneal, ``data_preprocessing.sampling.sampling``) but feeds our
(N, T, C) arrays through a tiny in-memory Dataset, runs an inline training loop
so we control device / epochs, and persists the model (+ EMA) under save(dir).

The reference training loop counts in *steps* (``lr_anneal_steps``); here we
express the budget as ``max_epochs`` over the dataloader (steps = epochs *
len(loader)), which keeps the same per-step AdamW + linear-anneal recipe.
"""
from __future__ import annotations

import copy
import random
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np

from .base import GeneratorBase

# vendored source lives here; insert on sys.path so its absolute imports
# ("from Model import ...", "from eval_utils.MMD import ...") resolve.
_VENDOR_SRC = Path(__file__).resolve().parents[3] / "vendor" / "PaD-TS"


def _ensure_vendor_on_path() -> None:
    p = str(_VENDOR_SRC)
    if p not in sys.path:
        sys.path.insert(0, p)


def _import_vendor():
    _ensure_vendor_on_path()
    from Model import PaD_TS  # noqa: E402
    from diffmodel_init import create_gaussian_diffusion  # noqa: E402
    from resample import Batch_Same_Sampler  # noqa: E402
    from data_preprocessing.sampling import sampling  # noqa: E402
    return PaD_TS, create_gaussian_diffusion, Batch_Same_Sampler, sampling


def _set_seeds(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class _NpyDataset:
    """Yields torch (T, C) tensors. Mirrors PaD-TS CustomDataset.__getitem__."""

    def __init__(self, arr: np.ndarray):
        import torch

        self._torch = torch
        self.samples = arr  # (N, T, C) float32

    def __len__(self) -> int:
        return self.samples.shape[0]

    def __getitem__(self, idx):
        return self._torch.from_numpy(self.samples[idx]).float()


class PaDTSGenerator(GeneratorBase):
    """PaD-TS adapter. params: hidden_size (dominant width knob; doubled across
    the two towers and ~hidden_size^2 in cost, must divide by num_heads),
    num_heads, n_encoder, n_decoder, mlp_ratio, diffusion_steps, mmd_alpha.
    train_cfg: batch_size, max_epochs, lr."""

    _CKPT = "padts.pt"

    # -- internals --------------------------------------------------------
    def _resolve_device(self):
        import torch

        want = str(self.device)
        if want.startswith("cuda") and not torch.cuda.is_available():
            return "cpu"
        return want

    def _build_model(self):
        PaD_TS, _, _, _ = _import_vendor()
        p = self.params
        hidden_size = int(p.get("hidden_size", 128))
        num_heads = int(p.get("num_heads", 4))
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by "
                f"num_heads ({num_heads})")
        model = PaD_TS(
            hidden_size=hidden_size,
            num_heads=num_heads,
            n_encoder=int(p.get("n_encoder", 1)),
            n_decoder=int(p.get("n_decoder", 3)),
            feature_last=True,
            mlp_ratio=float(p.get("mlp_ratio", 4.0)),
            dropout=float(p.get("dropout", 0.0)),
            input_shape=(self.T, self.C),
        )
        return model

    def _build_diffusion(self):
        _, create_gaussian_diffusion, _, _ = _import_vendor()
        p = self.params
        return create_gaussian_diffusion(
            predict_xstart=True,
            diffusion_steps=int(p.get("diffusion_steps", 250)),
            noise_schedule="cosine",
            loss="MSE_MMD",
            rescale_timesteps=False,
        )

    # -- core API ---------------------------------------------------------
    def fit(self, X: np.ndarray, train_cfg: Dict[str, Any] | None = None) -> "PaDTSGenerator":
        import torch
        from torch.optim import AdamW

        _, _, Batch_Same_Sampler, _ = _import_vendor()

        X = self._check_X(X)  # (N, T, C) float32
        cfg = dict(train_cfg or {})
        self._device = self._resolve_device()
        _set_seeds(self.seed)

        device = torch.device(self._device)
        self._model = self._build_model().to(device)
        self._diffusion = self._build_diffusion()
        schedule_sampler = Batch_Same_Sampler(self._diffusion)

        bs = int(cfg.get("batch_size", 64))
        bs = max(1, min(bs, X.shape[0]))
        max_epochs = int(cfg.get("max_epochs", 100))
        lr = float(cfg.get("lr", 1e-4))
        weight_decay = float(cfg.get("weight_decay", 0.0))
        mmd_alpha = float(self.params.get("mmd_alpha", cfg.get("mmd_alpha", 0.0008)))
        log_interval = int(cfg.get("log_interval", 0))

        ds = _NpyDataset(X)
        drop_last = X.shape[0] > bs
        loader = torch.utils.data.DataLoader(
            ds, batch_size=bs, shuffle=True, num_workers=0,
            pin_memory=(self._device != "cpu"), drop_last=drop_last,
        )
        steps_per_epoch = max(1, len(loader))
        total_steps = max(1, max_epochs * steps_per_epoch)

        opt = AdamW(self._model.parameters(), lr=lr, weight_decay=weight_decay)

        # EMA shadow of the model weights (decay configurable; reference does
        # not sample from EMA, so it is persisted but not the sampling source).
        ema_decay = float(cfg.get("ema_decay", 0.0))
        self._ema_model = copy.deepcopy(self._model) if ema_decay > 0 else None
        if self._ema_model is not None:
            for p in self._ema_model.parameters():
                p.requires_grad_(False)

        self._model.train()
        step = 0
        for _epoch in range(max_epochs):
            for data in loader:
                data = data.to(device)
                opt.zero_grad()
                t, weights = schedule_sampler.sample(data.shape[0], device)
                losses = self._diffusion.training_losses(self._model, data, t)
                loss = (losses["mse"] * weights).mean()
                if "mmd" in losses:
                    loss = loss + mmd_alpha * losses["mmd"]
                loss.backward()
                # linear LR anneal over the full step budget (matches reference)
                frac_done = step / total_steps
                for g in opt.param_groups:
                    g["lr"] = lr * (1 - frac_done)
                opt.step()
                step += 1
                if self._ema_model is not None:
                    with torch.no_grad():
                        for tp, sp in zip(self._ema_model.parameters(),
                                          self._model.parameters()):
                            tp.detach().mul_(ema_decay).add_(sp, alpha=1 - ema_decay)
                if log_interval and step % log_interval == 0:
                    print(f"[padts] step {step}/{total_steps} loss={float(loss):.6f}",
                          flush=True)

        self._fitted = True
        return self

    def sample(self, n: int, sample_cfg: Dict[str, Any] | None = None) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("PaDTSGenerator.sample before fit")
        import torch

        _, _, _, sampling = _import_vendor()
        cfg = dict(sample_cfg or {})
        n = int(n)
        device = torch.device(self._device)
        # default the sample batch to the training batch (reference parity); avoids
        # the vendor sampler's (n//bs)+1 loop generating up to ~2n and discarding n.
        batch_size = int(cfg.get("batch_size", self.params.get("sample_batch", 256)))
        batch_size = max(1, min(batch_size, n))

        model = self._model.to(device)
        model.eval()
        with torch.no_grad():
            fakes_t = sampling(model, self._diffusion, n, self.T, self.C, batch_size)
        fakes = fakes_t.cpu().numpy()[:n].astype(np.float32)
        return fakes

    def save(self, path: str) -> None:
        if not self._fitted:
            raise RuntimeError("nothing to save")
        import torch

        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": self._model.state_dict(),
                "ema": (self._ema_model.state_dict()
                        if getattr(self, "_ema_model", None) is not None else None),
                "T": self.T,
                "C": self.C,
                "params": self.params,
            },
            str(p / self._CKPT),
        )

    def load(self, path: str) -> "PaDTSGenerator":
        import torch

        self._device = self._resolve_device()
        data = torch.load(str(Path(path) / self._CKPT), map_location=self._device)
        self.T = int(data.get("T", self.T))
        self.C = int(data.get("C", self.C))
        if "params" in data:                      # restore even an empty/all-default dict
            self.params = dict(data["params"])

        device = torch.device(self._device)
        self._model = self._build_model().to(device)
        self._model.load_state_dict(data["model"])
        self._diffusion = self._build_diffusion()

        self._ema_model = None
        if data.get("ema") is not None:
            self._ema_model = self._build_model().to(device)
            self._ema_model.load_state_dict(data["ema"])
            for p in self._ema_model.parameters():
                p.requires_grad_(False)

        self._fitted = True
        return self
