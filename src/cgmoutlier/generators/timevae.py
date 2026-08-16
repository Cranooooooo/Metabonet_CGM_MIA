"""TimeVAE generator adapter wrapping vendor/TimeVAE.

Wraps the vendored PyTorch TimeVAE (an interpretable variational autoencoder for
time series). Training and prior sampling go through vendor/TimeVAE/src/vae's
`instantiate_vae_model` / `train_vae` / `get_prior_samples` utilities; checkpoints
use the model's own `save` / `TimeVAE.load`.

See vendor/TimeVAE/run_timevae_ts.py for the reference train+sample recipe.
"""
from __future__ import annotations

import os
import random
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np

from .base import GeneratorBase

# vendor/TimeVAE/src must be on sys.path so "from vae...." imports resolve.
_VENDOR_SRC = (
    Path(__file__).resolve().parents[3] / "vendor" / "TimeVAE" / "src"
)


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


class TimeVAEGenerator(GeneratorBase):
    """TimeVAE adapter. params: latent_dim, hidden_layer_sizes, reconstruction_wt,
    use_residual_conn, trend_poly. train_cfg: batch_size, max_epochs."""

    def fit(self, X: np.ndarray, train_cfg: Dict[str, Any] | None = None) -> "TimeVAEGenerator":
        _ensure_vendor_on_path()
        from vae.vae_utils import instantiate_vae_model, train_vae  # noqa: E402

        X = self._check_X(X)  # (N, T, C) float32
        # params first, then train_cfg on top. `batch_size` and `max_epochs` used to be
        # read from train_cfg alone, which `loo.train.run` never passes -- so they were
        # unreachable from --params and silently fixed at the vendored 16 and 1000
        # whatever a caller asked for. Reading self.params too is what makes the budget
        # a declared choice rather than an inherited constant.
        p = self.params
        cfg = {**p, **(train_cfg or {})}

        batch_size = int(cfg.get("batch_size", 16))
        max_epochs = int(cfg.get("max_epochs", 1000))

        # instantiate_vae_model hardcodes set_seeds(123) internally.
        vae = instantiate_vae_model(
            vae_type="timeVAE",
            sequence_length=self.T,
            feature_dim=self.C,
            batch_size=batch_size,
            latent_dim=int(p.get("latent_dim", 8)),
            hidden_layer_sizes=list(p.get("hidden_layer_sizes", [50, 100, 200])),
            reconstruction_wt=float(p.get("reconstruction_wt", 3.0)),
            use_residual_conn=bool(p.get("use_residual_conn", True)),
            trend_poly=int(p.get("trend_poly", 0)),
            custom_seas=None,
        )
        # Re-seed AFTER instantiation to override the vendor's hardcoded seed=123.
        _set_seeds(self.seed)

        train_vae(vae=vae, train_data=X, max_epochs=max_epochs, verbose=int(cfg.get("verbose", 0)))

        self._vae = vae
        self._fitted = True
        return self

    def sample(self, n: int, sample_cfg: Dict[str, Any] | None = None) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("TimeVAEGenerator.sample before fit")
        import torch

        from vae.vae_utils import get_prior_samples  # noqa: E402

        _set_seeds(self.seed)
        self._vae.eval()
        with torch.no_grad():
            fakes = get_prior_samples(self._vae, num_samples=int(n))
        if not isinstance(fakes, np.ndarray):
            fakes = np.asarray(fakes)
        return fakes.astype(np.float32)

    def save(self, path: str) -> None:
        if not self._fitted:
            raise RuntimeError("nothing to save")
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        # TimeVAE.save writes <model_name>_weights.pth and <model_name>_parameters.pkl.
        self._vae.save(str(p))

    def load(self, path: str) -> "TimeVAEGenerator":
        _ensure_vendor_on_path()
        from vae.vae_utils import load_vae_model  # noqa: E402

        self._vae = load_vae_model(vae_type="timeVAE", dir_path=str(Path(path)))
        self._fitted = True
        return self
