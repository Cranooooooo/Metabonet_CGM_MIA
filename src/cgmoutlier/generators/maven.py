"""MAVEN, our generator, behind the same interface as every baseline.

Unlike DiM-TS this needs no second environment -- MAVEN is plain PyTorch with no
custom CUDA kernel -- but it is still driven as a subprocess, for two reasons that are
not style preferences. The vendored tree puts itself on `sys.path` and imports modules
under generic names (`visual_transforms`, `train_maven`); importing that in-process
would pollute the interpreter the attack and the metrics also run in. And a subprocess
returns its GPU memory on exit, which matters when the LOO orchestration trains 27
models back to back in one job.

    gen = MAVENGenerator(T=288, C=1)
    gen.fit(X).sample(6072)

The heavy lifting is in `vendor/MAVEN/cgm_train_sample.py`; that file's docstring
explains what was changed from upstream and why. This class only marshals arguments,
so a defect in the model shows up there rather than being spread across both.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np

from .base import GeneratorBase

REPO = Path(__file__).resolve().parents[3]
VENDOR = REPO / "vendor" / "MAVEN"
SCRIPT = VENDOR / "cgm_train_sample.py"


class MAVENGenerator(GeneratorBase):
    """params, all optional except where noted:

        workdir      durable directory for checkpoints and samples; temp if unset
        dims         AE width schedule, e.g. "128,128,64,64,64,64,128,128"
        view_a       auto | delay | dayfold   (auto -> dayfold when T % fold_period == 0)
        fold_period  288, one day at five-minute sampling
        stft_n_fft   96 = 8 h, against upstream's 16 = 80 min
        stft_hop     48; must stay n_fft/2 or the inverse STFT is not exact
        mask_mode    cell (default) | channel  -- `channel` is degenerate at C = 1
        steps        optimiser iterations (alias: total_iters)
        save_cycle   iterations per weight milestone
        n_samples    released set size; defaults to the training-set size
        batch_size, micro_batch_size, lr, sampling_steps, sample_batch, gpu, seed
    """

    # names this class accepts on the left, the script's flag on the right; anything
    # not listed is rejected rather than silently dropped, because a mistyped
    # hyperparameter that trains happily at the default is the expensive kind of bug
    _FLAGS = {
        "dims": "--dims", "n_heads": "--n_heads",
        "view_a": "--view_a", "fold_period": "--fold_period",
        "delay_tau": "--delay_tau", "delay_m": "--delay_m",
        "stft_n_fft": "--stft_n_fft", "stft_hop": "--stft_hop",
        "mask_mode": "--mask_mode", "mask_lo": "--mask_lo", "mask_hi": "--mask_hi",
        "coarse_weight": "--coarse_weight", "fuse_alpha": "--fuse_alpha",
        "lambda_schedule": "--lambda_schedule",
        "batch_size": "--batch_size", "micro_batch_size": "--micro_batch_size",
        "lr": "--lr", "weight_decay": "--weight_decay", "grad_clip": "--grad_clip",
        "ema_decay": "--ema_decay", "amp": "--amp",
        "sampling_steps": "--sampling_steps", "sample_batch": "--sample_batch",
        "save_cycle": "--save_cycle", "log_every": "--log_every",
    }
    _CONSUMED = {"workdir", "steps", "total_iters", "n_samples", "K", "seed", "gpu",
                 "python", "repo", "hidden_size"}

    def __init__(self, T: int, C: int, params: Dict[str, Any] | None = None, **kw):
        super().__init__(T, C, params, **kw)
        p = self.params
        self.workdir = Path(p["workdir"]) if p.get("workdir") else None
        self._samples = None

    # ---------------------------------------------------------------- plumbing
    def _preflight(self):
        if not SCRIPT.exists():
            raise FileNotFoundError(
                f"missing {SCRIPT}. The MAVEN source is vendored at {VENDOR}; if that "
                f"directory is empty this is a fresh clone that never ran the vendor "
                f"step.")

    def _flags(self, cfg, wd, out_npy, n, train: bool):
        cmd = [sys.executable, "-u", str(SCRIPT),
               "--data_npy", str(wd / "train.npy"),
               "--out_npy", str(out_npy),
               "--results_folder", str(wd / "ck"),
               "--K", str(int(n)),
               "--seed", str(int(cfg.get("seed", self.seed))),
               "--gpu", str(cfg.get("gpu", 0))]
        if train:
            cmd += ["--total_iters",
                    str(int(cfg.get("steps", cfg.get("total_iters", 100_000))))]
        unknown = set(cfg) - set(self._FLAGS) - self._CONSUMED
        if unknown:
            raise ValueError(
                f"MAVENGenerator got parameter(s) it does not recognise: "
                f"{sorted(unknown)}. They would have been ignored silently and the run "
                f"would have used the defaults. Known: {sorted(self._FLAGS)}")
        for k, flag in self._FLAGS.items():
            if k in cfg and cfg[k] is not None:
                cmd += [flag, str(cfg[k])]
        return cmd

    def _run(self, cmd):
        r = subprocess.run(cmd, cwd=REPO)
        if r.returncode:
            raise RuntimeError(f"MAVEN exited {r.returncode}; see output above")

    # -------------------------------------------------------------------- api
    def fit(self, X: np.ndarray, train_cfg: Dict[str, Any] | None = None):
        X = self._check_X(X)
        self._preflight()
        cfg = dict(self.params)
        cfg.update(train_cfg or {})

        import tempfile
        wd = (self.workdir or Path(tempfile.mkdtemp(prefix="maven_"))).resolve()
        (wd / "ck").mkdir(parents=True, exist_ok=True)
        np.save(wd / "train.npy", X.astype(np.float32))

        n = int(cfg.get("n_samples", cfg.get("K", len(X))))
        self._run(self._flags(cfg, wd, wd / "all_samples.npy", n, train=True))
        out = wd / "all_samples.npy"
        if not out.exists():
            raise RuntimeError(f"MAVEN produced no {out}")
        self._samples = np.load(out)
        self._fitted = True
        self.workdir = wd
        return self

    def resample(self, n: int, *, from_dir=None, out_dir=None, milestone=None,
                 sampling_timesteps=None) -> np.ndarray:
        """Sample from a stored milestone with no training.

        The architecture and the cohort-space affine come from the run's own
        `ck/maven_config.json`, never from this object's params. A width that
        disagreed with the checkpoint would surface as a shape error inside
        `load_state_dict` without naming the run; an affine refitted on a different
        array would not error at all, and would put the samples on a different scale
        than the model that produced them.
        """
        self._preflight()
        wd = Path(from_dir or self.workdir or "").resolve()
        if not (wd / "train.npy").exists():
            raise FileNotFoundError(
                f"{wd}/train.npy is missing; a resample needs the run's own training "
                f"array to rebuild the transforms")
        if not (wd / "ck" / "maven_config.json").exists():
            raise FileNotFoundError(
                f"{wd}/ck/maven_config.json is missing; the scaling and architecture "
                f"of this run are unknown and guessing them would be silent")

        cfg = {k: v for k, v in self.params.items()
               if k in ("gpu", "seed", "sample_batch", "sampling_steps")}
        if sampling_timesteps is not None:
            cfg["sampling_steps"] = int(sampling_timesteps)
        out_dir = Path(out_dir or wd).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        out_npy = out_dir / "all_samples.npy"

        cmd = self._flags(cfg, wd, out_npy, int(n), train=False) + ["--skip_train"]
        if milestone is not None:
            cmd += ["--load_milestone", str(int(milestone))]
        self._run(cmd)
        if not out_npy.exists():
            raise RuntimeError(f"MAVEN produced no {out_npy}")
        self._samples = np.load(out_npy)
        self._fitted = True
        self.workdir = wd
        return self._samples[:int(n)]

    @staticmethod
    def milestones(from_dir: str) -> list[int]:
        ck = Path(from_dir) / "ck"
        return sorted(int(p.stem.split("-")[1]) for p in ck.glob("milestone-*.pt"))

    def sample(self, n: int, sample_cfg: Dict[str, Any] | None = None) -> np.ndarray:
        """Training and sampling happen in one subprocess call, so this returns what
        that call produced. Asking for more than was generated is an error, not a
        silently short array."""
        if self._samples is None:
            raise RuntimeError(
                "no samples in hand: call fit(), or resample() if this run has "
                "milestones but its sampling pass never finished")
        if n > len(self._samples):
            raise ValueError(
                f"asked for {n} samples, the run produced {len(self._samples)}. "
                f"Set params['n_samples'] before fit().")
        return self._samples[:n]

    def save(self, path: str) -> None:
        if self.workdir is None:
            raise RuntimeError("nothing to save")
        shutil.copytree(self.workdir, path, dirs_exist_ok=True)

    def load(self, path: str):
        """Restore a finished run, or one that trained but never sampled -- which is
        what a walltime kill during the sampling loop leaves behind."""
        self.workdir = Path(path)
        samples = self.workdir / "all_samples.npy"
        self._samples = np.load(samples) if samples.exists() else None
        ck = self.workdir / "ck"
        if self._samples is None and not (
                list(ck.glob("milestone-*.pt")) or (ck / "final.pt").exists()):
            raise FileNotFoundError(
                f"{path} holds neither all_samples.npy nor any checkpoint; there is "
                f"nothing here to restore")
        self._fitted = True
        return self
