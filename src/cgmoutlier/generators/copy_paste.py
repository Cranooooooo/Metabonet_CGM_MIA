"""Copy-paste sanity generator: 'samples' are real training chunks replayed.

This is the MIA sanity baseline. Because its synthetic set literally contains the
training chunks, the membership attack should score AUC ~ 1.0 — it verifies the
labels / aggregation / classifier plumbing before real generators are plugged in.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import numpy as np

from .base import GeneratorBase


class CopyPasteGenerator(GeneratorBase):
    def fit(self, X: np.ndarray, train_cfg: Dict[str, Any] | None = None) -> "CopyPasteGenerator":
        self._X = self._check_X(X)
        self._fitted = True
        return self

    def sample(self, n: int, sample_cfg: Dict[str, Any] | None = None) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("CopyPasteGenerator.sample before fit")
        rng = np.random.default_rng(self.seed)
        idx = rng.integers(0, self._X.shape[0], size=int(n))
        return self._X[idx].copy().astype(np.float32)

    def save(self, path: str) -> None:
        if not self._fitted:
            raise RuntimeError("nothing to save")
        p = Path(path); p.mkdir(parents=True, exist_ok=True)
        np.save(p / "train_chunks.npy", self._X)

    def load(self, path: str) -> "CopyPasteGenerator":
        self._X = np.load(Path(path) / "train_chunks.npy").astype(np.float32)
        self._fitted = True
        return self
