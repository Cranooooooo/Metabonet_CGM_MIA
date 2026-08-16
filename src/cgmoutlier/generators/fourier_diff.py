"""FourierDiffusion adapter (wraps vendor/FourierDiffusion).

Trains a score-based diffusion model in the *time* domain (fourier_transform=
False) for comparability with the other baselines, using a PyTorch-Lightning
Trainer + ScoreModule + VPScheduler, fed by an in-memory (numpy-backed)
Datamodule subclass. Data is standardized per feature before training; sampling
draws from the VP prior, runs reverse diffusion in batches via DiffusionSampler,
then un-standardizes back to the original scale.

Mirrors the recipe in vendor/FourierDiffusion/run_fdiff_ts.py. The vendor source
is imported by inserting its `src` dir on sys.path; vendor/ is never edited.
"""
from __future__ import annotations

import os
import random
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np

from .base import GeneratorBase

# vendor/FourierDiffusion/src on sys.path so that `from fdiff...` resolves.
_VENDOR_SRC = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..",
                 "vendor", "FourierDiffusion", "src")
)


def _ensure_vendor_on_path() -> None:
    if _VENDOR_SRC not in sys.path:
        sys.path.insert(0, _VENDOR_SRC)


class FourierDiffGenerator(GeneratorBase):
    """Score-based (Fourier)Diffusion generator over (N, T, C) chunks."""

    def fit(self, X: np.ndarray, train_cfg: Dict[str, Any] | None = None
            ) -> "FourierDiffGenerator":
        _ensure_vendor_on_path()
        import torch
        import pytorch_lightning as pl
        from fdiff.models.score_models import ScoreModule
        from fdiff.schedulers.sde import VPScheduler
        from fdiff.dataloaders.datamodules import Datamodule

        X = self._check_X(X)
        cfg = dict(train_cfg or {})

        # --- seeds (reproducibility) ------------------------------------
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        pl.seed_everything(self.seed, workers=True)

        # --- hyperparameters --------------------------------------------
        d_model = int(self.params.get("d_model", 36))
        num_layers = int(self.params.get("num_layers", 5))
        n_head = int(self.params.get("n_head", 6))

        batch_size = int(cfg.get("batch_size", 64))
        batch_size = max(1, min(batch_size, X.shape[0]))
        lr_max = float(cfg.get("lr_max", 1.0e-3))
        max_epochs = int(cfg.get("max_epochs", 200))  # smoke overrides this

        use_cuda = (self.device == "cuda") and torch.cuda.is_available()

        # Floor for per-feature std so zero-variance (t, c) positions never cause
        # a divide-by-zero in standardization (which would push NaNs into the
        # training data and corrupt the model's weights -> all-NaN samples).
        std_floor = float(cfg.get("std_floor", 1e-6))

        # --- in-memory numpy-backed Datamodule --------------------------
        # We standardize the data OURSELVES (with a floored std) and hand the
        # already-standardized tensor to the vendor with standardize=False, so
        # the vendor's divide-by-std (which has no zero-variance guard) never
        # runs. Stats are stashed to un-standardize at sample time.
        class _NpyDatamodule(Datamodule):
            def __init__(_self, arr: np.ndarray):
                _self._arr = arr
                super().__init__(
                    data_dir=".",
                    random_seed=self.seed,
                    batch_size=batch_size,
                    fourier_transform=False,
                    standardize=False,  # we pre-standardize in setup()
                )

            @property
            def dataset_name(_self) -> str:
                return "m7mia"

            def prepare_data(_self) -> None:  # no download / mkdir
                pass

            def download_data(_self) -> None:
                pass

            def setup(_self, stage: str = "fit") -> None:
                arr = torch.from_numpy(_self._arr).float()
                mean = arr.mean(dim=0)
                std = torch.clamp(arr.std(dim=0), min=std_floor)
                _self.feat_mean = mean
                _self.feat_std = std
                _self.X_train = (arr - mean) / std
                _self.X_test = _self.X_train[:1]  # dummy; val loop disabled

            def val_dataloader(_self):
                return None

        dm = _NpyDatamodule(X)
        dm.prepare_data()
        dm.setup("fit")
        N, T, C = dm.X_train.shape
        num_training_steps = max(1, (N // batch_size)) * max_epochs

        # Stash standardization stats so sample()/save() can un-standardize.
        self._feat_mean = dm.feat_mean.cpu().numpy().astype(np.float32)
        self._feat_std = dm.feat_std.cpu().numpy().astype(np.float32)

        # --- model ------------------------------------------------------
        noise_scheduler = VPScheduler(
            beta_min=0.1, beta_max=20.0, fourier_noise_scaling=False, eps=1e-5)
        # Initialise the diagonal scaling (G / G_matrix). This is otherwise set
        # lazily inside marginal_prob during training; setting it explicitly
        # guarantees prior_sampling works regardless of training length.
        noise_scheduler.set_noise_scaling(T)
        score_model = ScoreModule(
            n_channels=C,
            max_len=T,
            noise_scheduler=noise_scheduler,
            fourier_noise_scaling=False,
            d_model=d_model,
            num_layers=num_layers,
            n_head=n_head,
            num_training_steps=num_training_steps,
            lr_max=lr_max,
            likelihood_weighting=False,
        )

        # --- 塌缩防护：每个 epoch 末评估，保留最优权重 ----------------------
        # 2026-07-27。原实现保存的是**最后一个** epoch 的权重，而 fourier_diff 会在训练
        # 后期塌进 score≡0 的退化解（PROGRESS.md §8.3）—— 35 个 fold 中招。
        # 实测轨迹显示模型在中途是健康的（step 6000 时 loss/平台=0.042，比某些 fold 的
        # 最终值还好），塌缩发生在之后。所以只要保留 best 而不是 last 就能避开，
        # 不需要换种子重训（那样 10M 的期望重试次数是 3 次）。
        # 评估成本：每 epoch 一次 512 样本前向，相对 1130 步训练可忽略。
        keep_best = bool(cfg.get("keep_best", True))
        eval_batch = dm.X_train[:512].clone()

        class _KeepBest(pl.Callback):
            def __init__(_s):
                _s.best = float("inf")
                _s.best_state = None
                _s.best_epoch = -1
                _s.history = []

            def on_train_epoch_end(_s, tr, pl_module):
                from fdiff.utils.losses import get_sde_loss_fn
                from fdiff.utils.dataclasses import DiffusableBatch
                lf = get_sde_loss_fn(scheduler=pl_module.noise_scheduler,
                                     train=False, likelihood_weighting=False)
                xb = eval_batch.to(pl_module.device)
                was_training = pl_module.training
                pl_module.eval()
                with torch.no_grad():
                    torch.manual_seed(0)      # 固定噪声/时间步 -> epoch 间可比
                    L = float(lf(pl_module, DiffusableBatch(
                        X=xb, y=None, timesteps=None)))
                if was_training:
                    pl_module.train()
                _s.history.append(L)
                if L < _s.best:
                    _s.best = L
                    _s.best_epoch = tr.current_epoch
                    _s.best_state = {k: v.detach().cpu().clone()
                                     for k, v in pl_module.state_dict().items()}

        cb = _KeepBest()

        # --- trainer ----------------------------------------------------
        trainer = pl.Trainer(
            max_epochs=max_epochs,
            accelerator="gpu" if use_cuda else "cpu",
            devices=1,
            gradient_clip_val=1.0,
            logger=False,
            enable_progress_bar=False,
            enable_checkpointing=False,
            num_sanity_val_steps=0,
            limit_val_batches=0,
            callbacks=[cb] if keep_best else [],
        )
        trainer.fit(model=score_model, datamodule=dm)

        if keep_best and cb.best_state is not None:
            last = cb.history[-1]
            score_model.load_state_dict(cb.best_state)
            print(f"[fourier_diff] keep_best: 取 epoch {cb.best_epoch} "
                  f"(loss={cb.best:.6g})，丢弃最后一个 epoch (loss={last:.6g})"
                  + ("  ← 最后一个 epoch 明显更差，很可能已塌缩"
                     if last > 4 * cb.best else ""))
            self._best_epoch = cb.best_epoch
            self._loss_history = list(cb.history)

            # 逐 epoch 损失轨迹。塌缩的触发点至今未定位（PROGRESS.md §8.3），
            # 而 score≡0 平台看起来是**吸收态** —— 已观察到 epoch 4 塌缩后连续 82 个
            # epoch 没有恢复。把轨迹记下来，等于每训一个 fold 就免费得到一个塌缩时点
            # 的样本；攒够了就能看出触发条件（是否与某个 epoch 区间/损失尖峰相关）。
            # 塌缩是吸收态（已观察到 epoch 4 塌缩后连续 82 个 epoch 未恢复），所以
            # 「最后一个健康 epoch 的下一个」就是塌缩时点。不要用 max(history) 当平台
            # 基准 —— epoch 0 的初始损失可能高于平台（实测是平台的 1.179 倍），会误判。
            good = [i for i, v in enumerate(cb.history) if v <= 4 * cb.best]
            first_bad = (good[-1] + 1) if good and good[-1] + 1 < len(cb.history) else None
            if first_bad is not None and last > 4 * cb.best:
                print(f"[fourier_diff] 塌缩时点: epoch {first_bad} / {len(cb.history)}"
                      f"（此后未恢复）")
            print("[fourier_diff] 逐 epoch loss: "
                  + " ".join(f"{v:.3g}" for v in cb.history))

        if use_cuda:
            score_model = score_model.cuda()
        score_model.eval()

        self._model = score_model
        self._use_cuda = use_cuda
        self._fitted = True

        # --- 健康门禁：score 塌缩检测 --------------------------------------
        # 2026-07-27: 37 个 fold 训练「成功」退出后其实是死的 —— score 模型塌缩到
        # 恒输出 0。DSM 损失 (1/tr(Σ⁻¹))·||s + Σ^{-1/2}z||² 在 s≡0 处是一个有限的
        # 平台（退化成与 t 无关的常数），所以塌缩既不报错也不出 NaN，损失看着还挺小。
        # 采样时 score≡0 抽掉了反向 SDE 的去噪漂移，X ← X(1+½βΔt)+噪声 指数发散，
        # 样本 std 从 0.16 涨到 36，MIA 于是给出假的 AUC=1.000。
        # 详见 PROGRESS.md §8.3。这里在保存之前就把它拦下来。
        ratio = self._collapse_ratio(dm.X_train)
        if ratio > float(cfg.get("collapse_ratio_max", 0.5)):
            raise RuntimeError(
                f"fourier_diff 训练塌缩：DSM loss 已达 score≡0 平台的 {ratio:.3f} 倍 "
                f"(阈值 0.5, 1.0 = 完全塌缩)。这个模型是死的 —— 采样会指数发散、"
                f"MIA 会给出假的 AUC≈1.0。不保存。"
                f"\n  seed={self.seed} d_model={d_model} lr_max={lr_max} "
                f"max_epochs={max_epochs}"
                f"\n  换个种子重训，或见 PROGRESS.md §8.3。")
        return self

    def _collapse_ratio(self, X_std: "Any", n: int = 512) -> float:
        """本模型的 DSM 损失 / score≡0 的平台损失。1.0 = 完全塌缩，越小越健康。

        实测：健康 fold ≈ 0.05，塌缩 fold = 1.0000（停在平台上）。
        """
        import torch
        from fdiff.utils.losses import get_sde_loss_fn
        from fdiff.utils.dataclasses import DiffusableBatch

        m = self._model
        xb = X_std[:n]
        if self._use_cuda:
            xb = xb.cuda()
        lf = get_sde_loss_fn(scheduler=m.noise_scheduler, train=False,
                             likelihood_weighting=False)

        class _Zero(torch.nn.Module):
            """恒输出 0 的 score —— 平台基准。"""
            def __init__(s):
                super().__init__()
                s.noise_scheduler = m.noise_scheduler

            def forward(s, b):
                return torch.zeros_like(b.X)

        with torch.no_grad():
            # 同一个种子 -> 两次用同一批噪声/时间步，比值才有意义
            torch.manual_seed(0)
            mine = float(lf(m, DiffusableBatch(X=xb, y=None, timesteps=None)))
            torch.manual_seed(0)
            plateau = float(lf(_Zero(), DiffusableBatch(X=xb, y=None, timesteps=None)))
        return mine / plateau if plateau > 0 else float("inf")

    def sample(self, n: int, sample_cfg: Dict[str, Any] | None = None
               ) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("FourierDiffGenerator.sample before fit")
        _ensure_vendor_on_path()
        import torch
        from fdiff.sampling.sampler import DiffusionSampler

        cfg = dict(sample_cfg or {})
        num_diffusion_steps = int(cfg.get("num_diffusion_steps", 1000))
        sample_batch = int(cfg.get("sample_batch", 128))
        n = int(n)

        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

        self._model.eval()
        sampler = DiffusionSampler(
            score_model=self._model, sample_batch_size=sample_batch)

        pieces = []
        remaining = n
        with torch.no_grad():
            while remaining > 0:
                b = min(sample_batch, remaining)
                sampler.sample_batch_size = b
                x = sampler.sample(
                    num_samples=b, num_diffusion_steps=num_diffusion_steps)
                pieces.append(x.detach().cpu())
                remaining -= b

        fakes = torch.cat(pieces, dim=0)[:n].numpy().astype(np.float32)
        # Un-standardize (training used standardize=True): X * std + mean
        fakes = fakes * self._feat_std[None] + self._feat_mean[None]
        return fakes.astype(np.float32)

    def save(self, path: str) -> None:
        if not self._fitted:
            raise RuntimeError("nothing to save")
        import torch

        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        torch.save(self._model.state_dict(), p / "score_model.pt")
        np.savez(
            p / "meta.npz",
            feat_mean=self._feat_mean,
            feat_std=self._feat_std,
            T=np.int64(self.T),
            C=np.int64(self.C),
            d_model=np.int64(self.params.get("d_model", 36)),
            num_layers=np.int64(self.params.get("num_layers", 5)),
            n_head=np.int64(self.params.get("n_head", 6)),
        )

    def load(self, path: str) -> "FourierDiffGenerator":
        _ensure_vendor_on_path()
        import torch
        from fdiff.models.score_models import ScoreModule
        from fdiff.schedulers.sde import VPScheduler

        p = Path(path)
        meta = np.load(p / "meta.npz")
        self._feat_mean = meta["feat_mean"].astype(np.float32)
        self._feat_std = meta["feat_std"].astype(np.float32)
        T = int(meta["T"]); C = int(meta["C"])

        use_cuda = (self.device == "cuda") and torch.cuda.is_available()
        noise_scheduler = VPScheduler(
            beta_min=0.1, beta_max=20.0, fourier_noise_scaling=False, eps=1e-5)
        noise_scheduler.set_noise_scaling(T)
        # num_training_steps only affects the LR schedule (training); any
        # positive value is fine for inference-only reconstruction.
        score_model = ScoreModule(
            n_channels=C,
            max_len=T,
            noise_scheduler=noise_scheduler,
            fourier_noise_scaling=False,
            d_model=int(meta["d_model"]),
            num_layers=int(meta["num_layers"]),
            n_head=int(meta["n_head"]),
            num_training_steps=1000,
            lr_max=1.0e-3,
            likelihood_weighting=False,
        )
        state = torch.load(p / "score_model.pt", map_location="cpu")
        score_model.load_state_dict(state)
        if use_cuda:
            score_model = score_model.cuda()
        score_model.eval()

        self.T = T
        self.C = C
        self._model = score_model
        self._use_cuda = use_cuda
        self._fitted = True
        return self
