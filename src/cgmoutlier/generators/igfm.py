"""IG-FM generator adapter (wraps vendor/IG-FM).

IG-FM is a flow-matching model with an encoder-bottleneck-decoder Transformer
velocity network, trained with a dual generation/imputation objective. It is the
only non-diffusion, non-VAE member of the baseline set.

Everything model-side -- the architecture, the losses, the EMA, the sampler --
comes from ``vendor/IG-FM/igfm_core.py``, which is upstream's ``train_r10.py``
copied verbatim. This adapter only supplies data and a training loop, so the
comparison against the other baselines stays fair and the vendored file can be
re-synced from upstream with a plain copy.

Two adaptations are required, both forced by our data rather than chosen:

**We cannot use upstream's data loader.** It reads a continuous CSV and cuts
stride-1 windows. Our 73,404 chunks are non-overlapping gap-free days drawn from
402 different subjects; a stride-1 cutter over their concatenation would
manufacture windows spanning day and subject boundaries. We feed the chunks
directly, already z-clipped to ~[-1, 1] -- the space upstream expects and, more
importantly, the identical input every other baseline receives.

**Batch size must shrink, and be compensated.** Self-attention is O(T^2) and our
T=288 is 4.5x upstream's 64, so the attention matrix is ~20x larger and
upstream's batch of 256 needs over 20 GB. We use a micro-batch with gradient
accumulation so the *effective* batch remains 256; the optimiser therefore sees
the same batch statistics upstream tuned for.

Capacity is set by ``hidden``. Calibrated for T=288, D=3, enc=4/dec=4/heads=8:
2M->144, 4M->200, 6M->248, 8M->288, 10M->320. ``hidden`` must be divisible by
``n_heads``, which makes the grid coarse near 2M (136 = 1.87M, 144 = 2.09M); we
take 144 at +4.6% rather than alter the published architecture to chase a closer
match. Other baselines are calibrated to +-3%, so this is a documented exception.

Needs torch >= 2.1 (``torch.optim.swa_utils.get_ema_multi_avg_fn``).
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np

from .base import GeneratorBase

_VENDOR = Path(__file__).resolve().parents[3] / "vendor" / "IG-FM"
_CORE = None


def _core():
    """Import the vendored upstream module once, without running its CLI."""
    global _CORE
    if _CORE is None:
        p = _VENDOR / "igfm_core.py"
        if not p.exists():
            raise FileNotFoundError(f"vendored IG-FM not found at {p}")
        spec = importlib.util.spec_from_file_location("igfm_core", p)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["igfm_core"] = mod
        spec.loader.exec_module(mod)
        _CORE = mod
    return _CORE


class IGFMGenerator(GeneratorBase):
    """Flow-matching baseline. params: hidden, n_enc_layers, n_dec_layers, n_heads."""

    def __init__(self, T: int, C: int, params: Dict[str, Any] | None = None,
                 device: str = "cuda", seed: int = 2026):
        super().__init__(T=T, C=C, params=params, device=device, seed=seed)
        self._model = None
        self._ema = None

    # -- helpers ---------------------------------------------------------
    def _build(self):
        core = _core()
        p = self.params
        return core.VelocityNet(
            feature_size=self.C,
            hidden=int(p.get("hidden", 144)),
            n_enc_layers=int(p.get("n_enc_layers", 4)),
            n_dec_layers=int(p.get("n_dec_layers", 4)),
            n_heads=int(p.get("n_heads", 8)),
            window=self.T,
            # 改动二:瓶颈里划给「水平」的通道数。0 = 关
            level_dims=int(p.get("level_dims", 0)),
        )

    # -- core API --------------------------------------------------------
    def fit(self, X: np.ndarray, train_cfg: Dict[str, Any] | None = None) -> "IGFMGenerator":
        import torch
        core = _core()
        # 构造参数必须先并进来。run_loo.py 把 --params 交给构造函数,然后调 fit() 时
        # 不传 train_cfg(loo/train.py),所以只读 train_cfg 的话 --params 全被静默丢弃 ——
        # 训练会用 300000 的默认值跑完并正常退出,而 meta.json 里记的是我们要的 30000。
        # dimts.py 和 maven.py 是对的写法(cfg = dict(self.params)),这里跟上。
        cfg = dict(self.params); cfg.update(train_cfg or {})
        X = self._check_X(X)

        # `steps` 是本项目通用的名字(dimts/maven 都用它);`total_iters` 是上游的名字。
        # 两个都认,免得再出一次「名字对不上就静默用默认值」。
        total_iters = int(cfg.get("steps", cfg.get("total_iters", 300000)))
        eff_batch = int(cfg.get("batch_size", 256))          # upstream's effective batch
        micro = int(cfg.get("micro_batch", 32))              # what actually fits at T=288
        accum = max(1, eff_batch // micro)
        lr = float(cfg.get("lr", 5e-4))
        wd = float(cfg.get("weight_decay", 1e-4))
        clip = float(cfg.get("grad_clip", 1.0))
        ema_decay = float(cfg.get("ema_decay", 0.999))
        sched = cfg.get("lambda_schedule", "100000:1,200000:2,300000:4")
        schedule = [(int(b), float(v)) for b, v in
                    (piece.split(":") for piece in sched.split(","))]
        decor_a = float(cfg.get("decor_alpha", 0.1))
        corrmap_a = float(cfg.get("corrmap_alpha", 0.1))
        # IDG conditioning-mask schedule. training_step takes these positionally with
        # no defaults, so they must be supplied; values are upstream's CLI defaults.
        # 三个隐私模块的开关。默认全关 -> 训练路径和原版逐位一致,
        # 之前那个质量 0.0394 仍可复现。开关含义见 vendor/IG-FM/igfm_core.py。
        # ⚠️ 改动三只在一天窗口(T=288)上验过前提,七天上被证伪,所以这里硬拦一道。
        iso_on = bool(cfg.get("iso_weight", False))
        if iso_on and self.T != 288:
            raise ValueError(
                f"iso_weight(改动三)的前提只在 T=288 上验过,七天窗口上被证伪;"
                f"当前 T={self.T}。要在别的窗口长度上用,先把前提重测一遍。")
        mod_kw = dict(
            coord_mask=bool(cfg.get("coord_mask", False)),
            p_block=float(cfg.get("p_block", 0.35)),
            p_level=float(cfg.get("p_level", 0.25)),
            block_lo=float(cfg.get("block_lo", 0.15)),
            block_hi=float(cfg.get("block_hi", 0.55)),
            level_dims=int(self.params.get("level_dims", 0)),
            order_dims=int(self.params.get("order_dims", 0)),
            iso_weight=iso_on,
            iso_bucket=int(cfg.get("iso_bucket", 12)),
            iso_knn=int(cfg.get("iso_knn", 5)),
            iso_power=float(cfg.get("iso_power", 1.0)),
        )
        mask_kw = dict(
            p_pure_gen=float(cfg.get("p_pure_gen", 0.10)),
            p_cond_low=float(cfg.get("p_cond_low", 0.40)),
            cond_low_lo=float(cfg.get("cond_low_lo", 0.10)),
            cond_low_hi=float(cfg.get("cond_low_hi", 0.30)),
            cond_high_lo=float(cfg.get("cond_high_lo", 0.30)),
            cond_high_hi=float(cfg.get("cond_high_hi", 0.80)),
        )

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        dev = torch.device(self.device if torch.cuda.is_available() else "cpu")
        self._model = self._build().to(dev)
        n_par = sum(q.numel() for q in self._model.parameters())
        print(f"[igfm] VelocityNet {n_par/1e6:.2f}M params (T={self.T}, D={self.C}, "
              f"hidden={self.params.get('hidden', 144)})", flush=True)
        print(f"[igfm] micro_batch={micro} x accum={accum} -> effective batch {micro*accum} "
              f"(upstream {eff_batch}); attention is O(T^2) and T={self.T}", flush=True)

        opt = torch.optim.AdamW(self._model.parameters(), lr=lr, weight_decay=wd)
        self._ema = torch.optim.swa_utils.AveragedModel(
            self._model,
            multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(decay=ema_decay))

        # --- periodic checkpointing + resume ---------------------------------
        # 100K iters is ~15 h; a kill without this loses all of it. diffusion_ts lost
        # 16 GPU-hours to exactly that in July, and the fix there was not "add
        # checkpoints" but "put them somewhere that survives" -- its periodic saves
        # were going to /tmp/<pid>/, which a reboot clears and a new pid cannot find.
        # So this writes into the fold's own checkpoint directory, and keeps the two
        # newest (one is enough to resume; the second guards a half-written file).
        save_every = int(cfg.get("save_every", 2000))
        keep = int(cfg.get("keep_checkpoints", 2))
        # 默认落在 workdir 里。不给默认值的话周期性存档整段是死代码 ——
        # 一次 walltime 杀死就是全丢,而且事后无法挽救(IGFMGenerator 从不调
        # _checkpoint,所以没有 milestone 目录可以重采样)。
        resume_dir = cfg.get("resume_dir") or cfg.get("workdir")
        start_it = 0
        if resume_dir:
            rd = Path(resume_dir); rd.mkdir(parents=True, exist_ok=True)
            cks = sorted(rd.glob("iter-*.pt"),
                         key=lambda q: int(q.stem.split("-")[1]))
            if cks:
                try:
                    ck = torch.load(cks[-1], map_location=dev, weights_only=False)
                    self._model.load_state_dict(ck["model"])
                    self._ema.load_state_dict(ck["ema"])
                    opt.load_state_dict(ck["opt"])
                    start_it = int(ck["iter"])
                    print(f"[igfm] resuming from {cks[-1].name}: {start_it}/{total_iters} "
                          f"done, {total_iters - start_it} to go", flush=True)
                except Exception as e:          # corrupt checkpoint -> restart, don't die
                    print(f"[igfm] checkpoint {cks[-1].name} unreadable ({e}); "
                          "training from scratch", flush=True)
                    start_it = 0
        if start_it >= total_iters:
            print("[igfm] checkpoint shows training already complete, skipping", flush=True)
            self._fitted = True
            return self

        def _save(it_done):
            if not resume_dir:
                return
            rd = Path(resume_dir)
            tmp = rd / f".iter-{it_done}.pt.tmp"
            torch.save({"model": self._model.state_dict(), "ema": self._ema.state_dict(),
                        "opt": opt.state_dict(), "iter": it_done}, tmp)
            tmp.rename(rd / f"iter-{it_done}.pt")     # atomic: no half-written file
            olds = sorted(rd.glob("iter-*.pt"), key=lambda q: int(q.stem.split("-")[1]))
            for o in olds[:-keep]:
                o.unlink(missing_ok=True)

        data = torch.from_numpy(X).to(dev)
        N = data.shape[0]
        M_obs = torch.ones(micro, self.T, self.C, device=dev)
        self._model.train()
        # 按任务累计填充误差。改动一的可证伪预测就靠这个读 ——
        # 之前这里是 loss, _met = ...(metrics 被丢掉),结果训完只能事后拿权重补测。
        task_acc = {}
        for it in range(start_it, total_iters):
            lam = core.lambda_schedule(it, schedule)
            opt.zero_grad(set_to_none=True)
            for _ in range(accum):
                idx = torch.randint(0, N, (micro,), device=dev)
                loss, met = core.training_step(
                    self._model, data[idx], M_obs, lam,
                    decor_alpha=decor_a, corrmap_alpha=corrmap_a, **mask_kw, **mod_kw)
                s = task_acc.setdefault(met["task"], [0.0, 0])
                s[0] += met["L_imp"]; s[1] += 1
                (loss / accum).backward()
            if clip > 0:
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), clip)
            opt.step()
            self._ema.update_parameters(self._model)
            if (it + 1) % save_every == 0:
                _save(it + 1)
            if (it + 1) % 5000 == 0:
                per = " ".join(f"{k}={v[0]/max(v[1],1):.5f}(n={v[1]})"
                               for k, v in sorted(task_acc.items()))
                print(f"[igfm] iter {it+1}/{total_iters} loss={float(loss):.5f} "
                      f"lam={lam} | per-task L_imp: {per}", flush=True)
                task_acc.clear()
        _save(total_iters)
        self._fitted = True
        return self

    def sample(self, n: int, sample_cfg: Dict[str, Any] | None = None) -> np.ndarray:
        import torch
        core = _core()
        if not self._fitted:
            raise RuntimeError("IGFMGenerator.sample before fit")
        # 同上:sample() 也收不到 sample_cfg,不并进构造参数的话 sampling_steps 会被
        # 静默忽略,采样退回 200 步,而我们的参照集是 500 步的。
        cfg = dict(self.params); cfg.update(sample_cfg or {})
        dev = torch.device(self.device if torch.cuda.is_available() else "cpu")
        net = self._ema.module if self._ema is not None else self._model
        net.eval()
        out = core.sample_unconditional(
            net, n_samples=int(n), T=self.T, D=self.C,
            n_steps=int(cfg.get("sampling_steps", 200)),
            batch=int(cfg.get("sample_batch", 64)), device=dev)
        return np.asarray(out, dtype=np.float32)

    def save(self, path: str) -> None:
        import torch
        d = Path(path); d.mkdir(parents=True, exist_ok=True)
        torch.save({"model": self._model.state_dict(),
                    "ema": (self._ema.state_dict() if self._ema is not None else None),
                    "params": self.params, "T": self.T, "C": self.C, "seed": self.seed},
                   d / "igfm.pt")

    def load(self, path: str) -> "IGFMGenerator":
        import torch
        ck = torch.load(Path(path) / "igfm.pt", map_location="cpu", weights_only=False)
        self.params = dict(ck.get("params") or self.params)
        dev = torch.device(self.device if torch.cuda.is_available() else "cpu")
        self._model = self._build().to(dev)
        self._model.load_state_dict(ck["model"])
        if ck.get("ema"):
            self._ema = torch.optim.swa_utils.AveragedModel(self._model)
            self._ema.load_state_dict(ck["ema"])
        self._fitted = True
        return self
