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

    # -- training-trajectory support --------------------------------------
    #
    # Measuring privacy risk against training length needs samples from PARTLY trained
    # models, not just the final one. Only DiM-TS could do that, because only DiM-TS had
    # a bespoke resample path; every other generator could be trained and sampled but not
    # rewound, so a risk-vs-training-length curve could not be drawn for it at all.
    #
    # These two methods add it once, for everyone, on top of the save/load contract that
    # already exists. A generator opts in by calling `_checkpoint` from inside its own
    # training loop; nothing else changes, and a generator that does not call it simply
    # has no milestones to offer.

    def _checkpoint(self, root: str | None, milestone: int) -> None:
        """Persist the current weights as milestone `milestone` under `root`.

        Deliberately routed through the generator's own `save`, rather than reaching into
        a vendored trainer's checkpoint format: `save`/`load` is the one thing every
        generator here already implements correctly, and a milestone written by `save` is
        readable by `load` without a second format to keep in step.
        """
        if root is None:
            return
        from pathlib import Path
        d = Path(root) / f"milestone-{int(milestone)}"
        d.mkdir(parents=True, exist_ok=True)
        was = self._fitted
        self._fitted = True          # save() guards on it; mid-training we are fit enough
        try:
            self.save(str(d))
        finally:
            self._fitted = was

    def resample(self, n: int, *, from_dir: str, milestone: int | None = None,
                 sample_cfg: Dict[str, Any] | None = None) -> np.ndarray:
        """Sample from a stored milestone without training.

        `milestone=None` takes the highest available, which is the end of training for a
        run that finished. Raises rather than silently falling back to the final weights:
        a curve point that quietly came from a different milestone than it claims is worse
        than a missing point.
        """
        from pathlib import Path
        root = Path(from_dir)
        avail = sorted(int(q.name.split("-")[1]) for q in root.glob("milestone-*")
                       if q.is_dir() and q.name.split("-")[1].isdigit())
        if not avail:
            raise FileNotFoundError(
                f"no milestone-* directories under {root}. The generator must call "
                f"_checkpoint() during fit() for a training trajectory to exist.")
        m = avail[-1] if milestone is None else int(milestone)
        if m not in avail:
            raise FileNotFoundError(f"milestone {m} not in {avail} under {root}")
        self.load(str(root / f"milestone-{m}"))
        return self.sample(n, sample_cfg)

    @staticmethod
    def milestones(from_dir: str) -> list[int]:
        """Which milestones a run directory holds, ascending."""
        from pathlib import Path
        return sorted(int(q.name.split("-")[1]) for q in Path(from_dir).glob("milestone-*")
                      if q.is_dir() and q.name.split("-")[1].isdigit())

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
