"""Common interface for all generators in the MIA framework.

Every generator (Diffusion-TS, FourierDiffusion, TimeVAE, copy-paste) implements
this so the LOO orchestration and MIA code are generator-agnostic.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict
import numpy as np


class GeneratorBase(ABC):
    """Train on (N, T, C) chunks; sample (n, T, C) synthetic chunks.

    Args:
        T: sequence length (e.g. 288 for METABONET, 1024 for ECG)
        C: number of channels (6 for METABONET, 12 for ECG)
        params: model hyperparameters (from configs/generators.yaml `params`)
        device: 'cuda' or 'cpu'
    """

    def __init__(self, T: int, C: int, params: Dict[str, Any] | None = None,
                 device: str = "cuda", seed: int = 2026):
        self.T = int(T)
        self.C = int(C)
        self.params = dict(params or {})
        self.device = device
        self.seed = int(seed)
        self._fitted = False

    # -- core API ---------------------------------------------------------
    @abstractmethod
    def fit(self, X: np.ndarray, train_cfg: Dict[str, Any] | None = None) -> "GeneratorBase":
        """Train on X of shape (N, T, C). Returns self."""

    @abstractmethod
    def sample(self, n: int, sample_cfg: Dict[str, Any] | None = None) -> np.ndarray:
        """Return (n, T, C) float32 synthetic chunks."""

    @abstractmethod
    def save(self, path: str) -> None:
        """Persist the trained model under `path` (a directory)."""

    @abstractmethod
    def load(self, path: str) -> "GeneratorBase":
        """Restore from `path`; sets fitted=True. Returns self."""

    # -- shared helpers ---------------------------------------------------
    def _check_X(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 3 or X.shape[1] != self.T or X.shape[2] != self.C:
            raise ValueError(
                f"expected X of shape (N, {self.T}, {self.C}), got {X.shape}")
        return X

    @property
    def fitted(self) -> bool:
        return self._fitted
