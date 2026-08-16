"""Diffusion-TS generator adapter (wraps vendor/Diffusion-TS).

Trains the interpretable diffusion model `Diffusion_TS` with the vendored
`engine.solver.Trainer`, then samples from the EMA model. Mirrors the recipe in
vendor/Diffusion-TS/run_diffts_ts.py but feeds our (N, T, C) arrays through a tiny
in-memory Dataset and persists the model + EMA state_dict under save(dir).
"""
from __future__ import annotations

import glob
import os
import re
import sys
import random
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

import numpy as np

from .base import GeneratorBase

# vendored source lives here; insert on sys.path so its absolute imports
# ("from Models...", "from engine...") resolve.
_VENDOR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "vendor", "Diffusion-TS")
)


def _milestone_of(path: str) -> int:
    """从 checkpoint-<N>.pt 取出 N；用于按数值而非字典序排序（10 < 9 的坑）。"""
    m = re.search(r"checkpoint-(\d+)\.pt$", os.path.basename(path))
    return int(m.group(1)) if m else -1


def _import_vendor():
    if _VENDOR not in sys.path:
        sys.path.insert(0, _VENDOR)
    from Models.interpretable_diffusion.gaussian_diffusion import Diffusion_TS  # noqa: E402
    from engine.solver import Trainer  # noqa: E402
    return Diffusion_TS, Trainer


class _NpyDataset:
    """Returns torch (T, C) tensors — matches what the vendor Trainer expects."""

    def __init__(self, arr: np.ndarray):
        import torch
        self._torch = torch
        self.samples = arr  # (N, T, C) float32

    def __len__(self) -> int:
        return self.samples.shape[0]

    def __getitem__(self, idx):
        return self._torch.from_numpy(self.samples[idx]).float()


class DiffusionTSGenerator(GeneratorBase):
    """Diffusion-TS adapter. params: d_model, n_layer_enc, n_layer_dec, n_heads,
    timesteps, mlp_hidden_times (from configs/generators.yaml diffusion_ts.params)."""

    _CKPT = "diffusion_ts.pt"

    # -- internals --------------------------------------------------------
    def _seed_all(self):
        import torch
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

    def _resolve_device(self):
        import torch
        want = str(self.device)
        if want.startswith("cuda") and not torch.cuda.is_available():
            return "cpu"
        return want

    def _build_model(self):
        Diffusion_TS, _ = _import_vendor()
        p = self.params
        timesteps = int(p.get("timesteps", 500))
        model = Diffusion_TS(
            seq_length=self.T,
            feature_size=self.C,
            n_layer_enc=int(p.get("n_layer_enc", 1)),
            n_layer_dec=int(p.get("n_layer_dec", 1)),
            d_model=int(p.get("d_model", 32)),
            timesteps=timesteps,
            sampling_timesteps=timesteps,
            loss_type="l1",
            beta_schedule="cosine",
            n_heads=int(p.get("n_heads", 4)),
            mlp_hidden_times=int(p.get("mlp_hidden_times", 4)),
            attn_pd=0.0,
            resid_pd=0.0,
            kernel_size=1,
            padding_size=0,
        )
        return model.to(self._device)

    def _build_trainer(self, model, dataloader, train_cfg: Dict[str, Any]):
        _, Trainer = _import_vendor()
        cfg = {
            "solver": {
                "max_epochs": int(train_cfg.get("max_epochs", 12000)),
                "gradient_accumulate_every": int(train_cfg.get("gradient_accumulate_every", 2)),
                "save_cycle": int(train_cfg.get("save_cycle", 2000)),
                "results_folder": str(self._results_folder),
                "base_lr": float(train_cfg.get("base_lr", 1.0e-4)),
                "ema": {"decay": 0.995, "update_interval": 10},
                "scheduler": {
                    "target": "engine.lr_sch.ReduceLROnPlateauWithWarmup",
                    "params": {
                        "factor": 0.5,
                        "patience": int(train_cfg.get("patience", 3000)),
                        "min_lr": 1.0e-5,
                        "threshold": 1.0e-1,
                        "threshold_mode": "rel",
                        "warmup_lr": 8.0e-4,
                        "warmup": int(train_cfg.get("warmup", 500)),
                        "verbose": False,
                    },
                },
            },
        }
        args = SimpleNamespace(name="diffusion_ts", save_dir=str(self._results_folder))
        return Trainer(config=cfg, args=args, model=model,
                       dataloader={"dataloader": dataloader, "dataset": dataloader.dataset},
                       logger=None)

    # -- core API ---------------------------------------------------------
    def fit(self, X: np.ndarray, train_cfg: Dict[str, Any] | None = None) -> "DiffusionTSGenerator":
        import torch
        X = self._check_X(X)
        train_cfg = dict(train_cfg or {})
        self._device = self._resolve_device()
        self._seed_all()

        # 中间 checkpoint 的落盘位置。
        # 默认仍是 /tmp（保持旧行为，供 smoke / 一次性调用使用），但 train_folds 会传
        # resume_dir 指到 fold checkpoint 目录下的持久化子目录 —— /tmp 重启即清空、
        # 且路径带 PID，重启后找不到，等于没有断点（PROGRESS.md §6 教训 6）。
        resume_dir = train_cfg.get("resume_dir")
        self._results_folder = (str(resume_dir) if resume_dir else
                                os.path.join("/tmp", f"m7mia_diffts_{os.getpid()}_{id(self)}"))
        os.makedirs(self._results_folder, exist_ok=True)

        self._model = self._build_model()

        # batch must not exceed N (Trainer uses drop_last=True via our loader)
        bs = int(train_cfg.get("batch_size", 64))
        bs = max(1, min(bs, X.shape[0]))
        ds = _NpyDataset(X)
        dataloader = torch.utils.data.DataLoader(
            ds, batch_size=bs, shuffle=True, num_workers=0,
            pin_memory=False, drop_last=(X.shape[0] >= bs and X.shape[0] > 1),
        )

        self._trainer = self._build_trainer(self._model, dataloader, train_cfg)

        # --- 断点轮转：只保留最新 KEEP 个 milestone -------------------------
        # vendor 的 Trainer 每 save_cycle 步存一次且从不删除。10 万步 / save_cycle=2000
        # = 50 个 milestone，每个约 160 MB（model + EMA + optimizer）→ 单个 fold 8 GB，
        # 84 个 fold 就是 670 GB。续训只需要最新的一个，留 2 个是防写坏。
        keep = int(train_cfg.get("keep_checkpoints", 2))
        _orig_save = self._trainer.save
        # ⚠️ 用 trainer 自己的 results_folder，不要用我们传进去的路径 ——
        # vendor 在 solver.py:39 追加了 `_{seq_length}` 后缀
        # (`Path(results_folder + f'_{model.seq_length}')`)，实际目录是 <path>_288。
        _folder = str(self._trainer.results_folder)
        self._results_folder = _folder

        def _save_and_rotate(milestone, verbose=False):
            _orig_save(milestone, verbose)
            cks = sorted(glob.glob(os.path.join(_folder, "checkpoint-*.pt")),
                         key=_milestone_of)
            for old_ck in cks[:-keep]:
                try:
                    os.remove(old_ck)
                except OSError:
                    pass
        self._trainer.save = _save_and_rotate

        # --- 从最新 milestone 续训 ------------------------------------------
        # ⚠️ vendor 的 train() 用**局部** step=0 作为循环上界（solver.py:99），不是
        # self.step。所以 load() 恢复 self.step 之后，循环仍会再跑满 train_num_steps ——
        # 直接续训会变成训 20 万步。必须把已完成的步数从上界里扣掉。
        done = 0
        cks = sorted(glob.glob(os.path.join(_folder, "checkpoint-*.pt")),
                     key=_milestone_of)
        if cks:
            ms = _milestone_of(cks[-1])
            try:
                self._trainer.load(ms)
                done = int(self._trainer.step)
                total = int(self._trainer.train_num_steps)
                self._trainer.train_num_steps = max(0, total - done)
                print(f"[diffusion_ts] 从 milestone {ms} 续训: 已完成 {done}/{total} 步，"
                      f"还需 {self._trainer.train_num_steps} 步")
            except Exception as e:      # 断点损坏 -> 从头训，不要让整个 fold 挂掉
                print(f"[diffusion_ts] 断点 {cks[-1]} 加载失败 ({e})，从头训练")
                self._trainer.train_num_steps = int(self._trainer.train_num_steps)

        if self._trainer.train_num_steps > 0:
            self._trainer.train()
        else:
            print("[diffusion_ts] 断点显示已训满，跳过训练")
        self._fitted = True
        return self

    def sample(self, n: int, sample_cfg: Dict[str, Any] | None = None) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("DiffusionTSGenerator.sample before fit")
        sample_cfg = dict(sample_cfg or {})
        n = int(n)
        size_every = int(sample_cfg.get("size_every", min(256, max(1, n))))
        size_every = max(1, min(size_every, n))
        fakes = self._trainer.sample(num=n, size_every=size_every, shape=(self.T, self.C))
        fakes = np.asarray(fakes)[:n].astype(np.float32)
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
                "ema": self._trainer.ema.state_dict(),
                "T": self.T,
                "C": self.C,
                "params": self.params,
            },
            str(p / self._CKPT),
        )

    def load(self, path: str) -> "DiffusionTSGenerator":
        import torch
        self._device = self._resolve_device()
        data = torch.load(str(Path(path) / self._CKPT), map_location=self._device)
        self.T = int(data.get("T", self.T))
        self.C = int(data.get("C", self.C))
        if data.get("params"):
            self.params = dict(data["params"])

        self._results_folder = os.path.join(
            "/tmp", f"m7mia_diffts_{os.getpid()}_{id(self)}")
        self._model = self._build_model()
        self._model.load_state_dict(data["model"])

        # rebuild a Trainer so sample() can use its EMA path; a 1-element loader
        # is enough — train() is never called after load.
        dummy = np.zeros((1, self.T, self.C), dtype=np.float32)
        ds = _NpyDataset(dummy)
        dataloader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False)
        self._trainer = self._build_trainer(
            self._model, dataloader,
            {"max_epochs": 0, "save_cycle": 1, "gradient_accumulate_every": 1})
        self._trainer.ema.load_state_dict(data["ema"])
        self._fitted = True
        return self
