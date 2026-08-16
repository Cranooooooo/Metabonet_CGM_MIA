"""DiM-TS, driven as a subprocess in its own environment.

Every other generator here is a class you import and call. DiM-TS cannot be, because
its Mamba selective-scan kernel needs a torch build that will not coexist with the
others' — see `docs/DIMTS.md`. So this class holds the same interface as the rest and
implements it by writing the training array to disk, invoking
`vendor/DiM-TS/cgm_train_sample.py` under the DiM-TS interpreter, and reading the
samples back.

That keeps the awkwardness in one place: callers pick a generator by name and never
learn which one needs a second environment.

    gen = DiMTSGenerator(T=288, C=1, params={"python": "/path/to/dimts/bin/python"})
    gen.fit(X).sample(50_000)

Nothing about the Mamba kernel is imported here, so this module loads in the ordinary
environment and fails only if you actually call `fit`.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict

import numpy as np

from .base import GeneratorBase

REPO = Path(__file__).resolve().parents[3]
VENDOR = REPO / "vendor" / "DiM-TS"
SCRIPT = VENDOR / "cgm_train_sample.py"


class DiMTSGenerator(GeneratorBase):
    """params:
        python    interpreter of the DiM-TS environment (required)
        repo      DiM-TS source tree; defaults to vendor/DiM-TS
        workdir   where checkpoints and samples go; a temp dir if unset
        hidden_size  model width (default 64); this is the capacity knob
        steps        optimiser steps (default 100_000, matching diffusion_ts)
        n_samples    released samples; defaults to the training-set size
        seed, gpu, batch_size, save_cycle, sample_batch  passed straight through
    """

    def __init__(self, T: int, C: int, params: Dict[str, Any] | None = None, **kw):
        super().__init__(T, C, params, **kw)
        p = self.params
        self.python = p.get("python") or os.environ.get("DIMTS_PYTHON")
        self.repo = Path(p.get("repo", VENDOR))
        self.workdir = Path(p["workdir"]) if p.get("workdir") else None
        self._samples = None

    # ------------------------------------------------------------------ checks
    def _preflight(self):
        if not self.python:
            raise RuntimeError(
                "DiM-TS needs its own environment; no interpreter given.\n"
                "  params={'python': '/path/to/envs/dimts/bin/python'}\n"
                "  or export DIMTS_PYTHON=...\n"
                "  Setup: docs/DIMTS.md")
        if not Path(self.python).exists():
            raise FileNotFoundError(f"DIMTS_PYTHON does not exist: {self.python}")
        if not SCRIPT.exists():
            raise FileNotFoundError(f"missing {SCRIPT}")
        # `import torch` FIRST, exactly as cgm_train_sample.py does. The kernel links
        # against torch's libc10.so, which torch puts on the loader's path when it is
        # imported; probing the bare import made this check stricter than the code path
        # it is guarding and failed a working environment with
        #   ImportError: libc10.so: cannot open shared object file
        probe = subprocess.run(
            [self.python, "-c", "import torch; import selective_scan_cuda_oflex_rh"],
            capture_output=True, text=True)
        if probe.returncode:
            last = probe.stderr.strip().splitlines()[-1] if probe.stderr else ""
            hint = ("  A CXXABI error means LD_LIBRARY_PATH is not picking up the\n"
                    "  environment's libstdc++.")
            if "libc10" in last or "libtorch" in last:
                hint = ("  A missing libc10/libtorch means the kernel was built against\n"
                        "  a different torch than this interpreter has. Rebuild the\n"
                        "  environment; do not reinstall torch over the top.")
            elif "No module named" in last:
                hint = ("  The kernel is not built. It is NOT on PyPI -- build it from\n"
                        "  vendor/DiM-TS/kernels/selective_scan.")
            raise RuntimeError(
                f"the DiM-TS environment cannot load its CUDA kernel:\n"
                f"  {last}\n{hint}\n  See docs/DIMTS.md.")

    # ---------------------------------------------------------------- plumbing
    def _env(self):
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join([
            str(self.repo / "Models" / "interpretable_diffusion"), str(self.repo),
            env.get("PYTHONPATH", "")])
        return env

    def _cmd(self, wd, cfg, out_npy, n):
        """The invocation shared by fit and resample.

        K defaults to the training-set size: an attack can only see a memorised window
        if it is actually released, so releasing fewer samples than the model was
        trained on lowers measured leakage for a reason that has nothing to do with
        the model.
        """
        return [self.python, str(SCRIPT),
                "--data_npy", str(wd / "train.npy"),
                "--out_npy", str(out_npy),
                "--results_folder", str(wd / "ckpt"),
                "--hidden_size", str(cfg.get("hidden_size", 64)),
                "--gpu", str(cfg.get("gpu", 0)),
                "--seed", str(cfg.get("seed", 2026)),
                "--max_steps", str(cfg.get("steps", cfg.get("max_steps", 100_000))),
                "--save_cycle", str(cfg.get("save_cycle", 10_000)),
                "--batch_size", str(cfg.get("batch_size", 64)),
                "--K", str(n),
                "--sample_batch", str(cfg.get("sample_batch", 1000)),
                # 0 for this single-channel cohort, and not as a tuning choice: the
                # term matches cross-CHANNEL correlation distributions and one channel
                # has no channel pairs. Left overridable for multi-channel callers.
                "--mmd_alpha", str(cfg.get("mmd_alpha",
                                           0.0 if self.C < 2 else 0.0008))]

    # -------------------------------------------------------------------- api
    def fit(self, X: np.ndarray, train_cfg: Dict[str, Any] | None = None):
        X = self._check_X(X)
        self._preflight()
        cfg = dict(self.params)
        cfg.update(train_cfg or {})

        # Absolute. The subprocess runs with cwd=vendor/DiM-TS, so a repo-relative
        # workdir resolves against the wrong directory there and the child dies on a
        # missing train.npy. This went unnoticed while the default was mkdtemp(), which
        # is absolute by construction.
        wd = (self.workdir or Path(tempfile.mkdtemp(prefix="dimts_"))).resolve()
        wd.mkdir(parents=True, exist_ok=True)
        np.save(wd / "train.npy", X.astype(np.float32))
        (wd / "config.json").write_text(json.dumps(
            {k: v for k, v in cfg.items() if k not in ("python", "repo", "workdir")},
            indent=2))

        n = int(cfg.get("n_samples", cfg.get("K", len(X))))
        cmd = self._cmd(wd, cfg, wd / "all_samples.npy", n)
        r = subprocess.run(cmd, cwd=self.repo, env=self._env())
        if r.returncode:
            raise RuntimeError(f"DiM-TS exited {r.returncode}; see output above")

        out = wd / "all_samples.npy"
        if not out.exists():
            raise RuntimeError(f"DiM-TS produced no {out}")
        self._samples = np.load(out)
        self._fitted = True
        self.workdir = wd
        return self

    def resample(self, n: int, *, from_dir=None, out_dir=None, milestone=None,
                 sampling_timesteps=None) -> np.ndarray:
        """Draw a fresh sample set from a checkpoint, with no training.

        `fit` is 60% of a run's cost and sampling is the rest, so a run whose training
        finished and whose sampling died is recoverable for the smaller half -- and the
        denoising-step question (`sampling_timesteps` 500 -> 50) is a sample-time
        change the trained weights are indifferent to. Both are this one path.

        The architecture comes from the trained run's own `config.json`, never from
        this object's params: a width that disagrees with the checkpoint would fail
        inside `load_state_dict` with a shape error rather than saying which run it
        was pointed at. Only sample-time knobs are the caller's to set.

        Not bit-identical to an uninterrupted run: training advances the RNG before
        sampling, so a restored model draws from a different position in the stream.
        Same weights and same distribution, different draw -- declared in meta.json by
        `loo.train.resample`, which is the only caller that writes one.
        """
        self._preflight()
        wd = Path(from_dir or self.workdir or "").resolve()
        if not (wd / "train.npy").exists():
            raise FileNotFoundError(
                f"{wd}/train.npy is missing; resampling needs the run's own training "
                f"array to rebuild the model's dataloader")
        ckpts = sorted((wd / f"ckpt_{self.T}").glob("checkpoint-*.pt"))
        if not ckpts:
            raise FileNotFoundError(
                f"no checkpoint-*.pt under {wd}/ckpt_{self.T}; this run has no trained "
                f"weights to resample from")

        saved = wd / "config.json"
        cfg = json.loads(saved.read_text()) if saved.exists() else {}
        if not cfg:
            raise FileNotFoundError(
                f"{saved} is missing; the checkpoint's architecture is unknown and "
                f"guessing it would fail as an opaque shape error")
        # Sample-time only. `hidden_size`/`steps`/`batch_size` stay as trained.
        for k in ("python", "repo", "gpu", "sample_batch"):
            if k in self.params:
                cfg[k] = self.params[k]

        out_dir = Path(out_dir or wd).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        out_npy = out_dir / "all_samples.npy"

        cmd = self._cmd(wd, cfg, out_npy, int(n)) + ["--skip_train"]
        if milestone is not None:
            cmd += ["--load_milestone", str(int(milestone))]
        if sampling_timesteps is not None:
            cmd += ["--sampling_timesteps", str(int(sampling_timesteps))]
        r = subprocess.run(cmd, cwd=self.repo, env=self._env())
        if r.returncode:
            raise RuntimeError(f"DiM-TS exited {r.returncode}; see output above")
        if not out_npy.exists():
            raise RuntimeError(f"DiM-TS produced no {out_npy}")

        self._samples = np.load(out_npy)
        self._fitted = True
        self.workdir = wd
        return self._samples[:int(n)]

    def sample(self, n: int, sample_cfg: Dict[str, Any] | None = None) -> np.ndarray:
        """DiM-TS trains and samples in one subprocess call, so this returns from what
        that call produced rather than sampling again. Asking for more than was
        generated is an error, not a silent short array."""
        if self._samples is None:
            raise RuntimeError(
                "no samples in hand: call fit(), or resample() if this run has "
                "checkpoints but its sampling pass never finished")
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
        """Restore a finished run, or a run that trained but never sampled.

        The second case is not hypothetical -- it is what a walltime kill during the
        sampling loop leaves behind, and an earlier version of this method raised
        FileNotFoundError on it, which made ten checkpoints on disk look like nothing.
        A run with weights but no samples loads as fitted; `sample` still refuses,
        because there is genuinely nothing to return until `resample` is called.
        """
        self.workdir = Path(path)
        samples = self.workdir / "all_samples.npy"
        self._samples = np.load(samples) if samples.exists() else None
        if self._samples is None and not any(
                (self.workdir / f"ckpt_{self.T}").glob("checkpoint-*.pt")):
            raise FileNotFoundError(
                f"{path} holds neither all_samples.npy nor a checkpoint; "
                f"there is nothing here to restore")
        self._fitted = True
        return self
