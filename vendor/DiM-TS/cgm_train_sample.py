#!/usr/bin/env python
"""Train + sample ONE DiM-TS size cell on the shared metabonet_perkg chunks.

Runs INSIDE the dedicated env ``project_general_DIMTS`` (torch 2.12/cu13 + the
prebuilt ``selective_scan_cuda_oflex_rh`` Mamba kernel). DiM-TS is a Diffusion-TS
fork, so we treat it IDENTICALLY to our diffusion_ts baseline for fairness:

* train on the SAME (N, 288, 3) z-scored chunks (already in ~[-0.37, 1.0]),
  fed directly with NO extra normalization (``auto_norm=False``);
* same step budget as diffusion_ts: 100k optimizer steps, grad-accum 2, EMA;
* sample from the EMA model; outputs stay in the z-scored space the eval expects
  (the [-1,1] x_start clamp is harmless since the data already lives in range).

Writes ``all_samples.npy`` (n, 288, 3) to --out_npy. The env-agnostic
m7mia.eval.quality (run separately in project_cgm_CLModel1) then scores it with
the same --eval-seed/K as the rest of the sweep.

Invoke with cwd=<DiMTS repo>, PYTHONPATH=<repo>/Models/interpretable_diffusion:<repo>,
and LD_LIBRARY_PATH=<env>/lib so the kernel .so loads.

RESAMPLING WITHOUT RETRAINING
-----------------------------
``--skip_train`` restores a checkpoint from --results_folder and goes straight to the
sampling loop. Two questions need it and neither needs 3.4 GPU-hours of retraining:

* a run whose training finished but whose sampling pass was killed (the sampling loop is
  ~40% of the bill, so this is a real and recurring loss);
* how many denoising steps the released set actually needs -- ``--sampling_timesteps``
  below 500 switches DiM_TS.generate_mts to its DDIM path, which is a sampling-time
  choice the trained weights are indifferent to.

The draw is NOT bit-identical to what an uninterrupted run would have produced: training
advances the global RNG before sampling starts, so a restored model samples from a
different position in the stream. Same weights, same distribution, different draw.
"""
import argparse
import os
import re
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from Models.interpretable_diffusion.gaussian_diffusion import DiM_TS
from engine.solver import Trainer
from Utils.io_utils import seed_everything


class _ChunkDataset(Dataset):
    """Yields (T, C) float32 tensors from a pre-windowed (N, T, C) array.
    Mirrors the attributes Trainer/main.py read off the dataset object."""

    def __init__(self, arr, window, var_num):
        self.samples = arr
        self.window = int(window)
        self.var_num = int(var_num)
        self.auto_norm = False  # data already in model space; do NOT unnormalize

    def __len__(self):
        return self.samples.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(self.samples[idx]).float()


def latest_milestone(results_folder, seq_length):
    """The highest checkpoint-<n>.pt Trainer wrote, or None.

    Trainer appends _<seq_length> to results_folder and numbers checkpoints by
    milestone, not by step, so 100k steps at save_cycle=10k ends at checkpoint-10.
    Picking the max rather than assuming max_steps/save_cycle is what makes this work
    on a run that was killed mid-training as well as one that finished.
    """
    d = f"{results_folder}_{int(seq_length)}"
    if not os.path.isdir(d):
        return None
    found = [int(m.group(1)) for m in
             (re.fullmatch(r"checkpoint-(\d+)\.pt", f) for f in os.listdir(d)) if m]
    return max(found) if found else None


def restore(trainer, milestone, device):
    """Load model + EMA weights from a checkpoint, without the optimiser.

    Deliberately not `trainer.load()`: that also restores optimiser state, which
    sampling does not use, and it calls torch.load with this torch's `weights_only`
    default. The vendored solver stays byte-identical either way -- see PROVENANCE.md.
    """
    path = os.path.join(f"{trainer.results_folder}", f"checkpoint-{milestone}.pt")
    data = torch.load(path, map_location=device, weights_only=False)
    trainer.model.load_state_dict(data["model"])
    # generate_mts samples from the EMA copy, so this is the load that matters; a
    # checkpoint without it would silently sample the raw weights instead.
    trainer.ema.load_state_dict(data["ema"])
    trainer.step = int(data.get("step", 0))
    trainer.milestone = int(milestone)
    print(f"[dimts] restored {path} (step {trainer.step:,})", flush=True)
    return trainer


def build_config(hidden_size, T, C, max_steps, save_cycle, results_folder,
                 mmd_alpha=None, sampling_timesteps=None):
    # mmd_alpha weights a loss term that matches the CROSS-CHANNEL correlation
    # distribution between real and generated batches. It is undefined for C=1: the
    # term is built from the off-diagonal entries of the C x C correlation matrix, so
    # a single channel has no pairs at all. torch.corrcoef then returns a 0-d scalar
    # and eval_utils.cross_correlation_distribution dies with
    #     IndexError: too many indices for tensor of dimension 0
    # The 0.0008 below is inherited from the previous project's 3-channel
    # metabonet_perkg cohort. 0 is not a tuning choice for single-channel data, it is
    # the only defined value.
    if mmd_alpha is None:
        mmd_alpha = 0.0 if int(C) < 2 else 0.0008
    # sampling_timesteps < timesteps is what flips DiM_TS.fast_sampling on, and with the
    # default eta=0 that is deterministic DDIM. It is a property of the SAMPLE, not of
    # the weights, so it must never differ between a pair's two models -- the attack
    # compares released sets, and a pair sampled at different step counts would differ
    # for a reason that has nothing to do with membership.
    return {
        "model": {
            "params": dict(
                seq_length=T, feature_size=C, timesteps=500,
                sampling_timesteps=int(sampling_timesteps or 500),
                loss_type="l2", beta_schedule="cosine", n_heads=4, mlp_hidden_times=4,
                attn_pd=0.0, resid_pd=0.0, kernel_size=1, padding_size=0,
                mmd_alpha=float(mmd_alpha), l_loss=1.0, sample_type="batch",
                hidden_size=int(hidden_size), n_encoder=1, n_decoder=3,
                feature_last=True, input_shape=[T, C], mlp_ratio=4.0,
                d_state=1, d_conv=3, model_type="DiM", conv_num=1,
            )
        },
        "solver": {
            "base_lr": 1.0e-4,
            "max_epochs": int(max_steps),           # train_num_steps (gradient steps)
            "results_folder": results_folder,        # Trainer appends _<seq_length>
            "gradient_accumulate_every": 2,
            "save_cycle": int(save_cycle),
            "ema": {"decay": 0.995, "update_interval": 10},
            "scheduler": {
                "target": "engine.lr_sch.ReduceLROnPlateauWithWarmup",
                "params": {
                    "factor": 0.5, "patience": 10000, "min_lr": 1.0e-5,
                    "threshold": 1.0e-1, "threshold_mode": "rel",
                    "warmup_lr": 8.0e-4, "warmup": 500, "verbose": False,
                },
            },
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_npy", required=True, help="(N,288,3) z-scored chunks")
    ap.add_argument("--out_npy", required=True, help="where to write all_samples.npy")
    ap.add_argument("--results_folder", required=True, help="checkpoint dir (Trainer adds _<T>)")
    ap.add_argument("--hidden_size", type=int, required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max_steps", type=int, default=100000)
    ap.add_argument("--save_cycle", type=int, default=10000)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--K", type=int, default=10000, help="num synthetic samples")
    ap.add_argument("--sample_batch", type=int, default=1000, help="size_every")
    ap.add_argument("--mmd_alpha", type=float, default=None,
                    help="cross-channel correlation MMD weight. Undefined for C=1 "
                         "and defaulted to 0 there; 0.0008 for multi-channel data")
    ap.add_argument("--skip_train", action="store_true",
                    help="restore a checkpoint from --results_folder and sample only")
    ap.add_argument("--load_milestone", type=int, default=None,
                    help="which checkpoint-<n>.pt to restore; default the highest")
    ap.add_argument("--sampling_timesteps", type=int, default=None,
                    help="denoising steps at SAMPLE time; default 500 (= training "
                         "timesteps, the full loop). Below that generate_mts takes the "
                         "DDIM path")
    args = ap.parse_args()

    seed_everything(args.seed)
    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")

    X = np.load(args.data_npy).astype(np.float32)   # (N, T, C)
    N, T, C = X.shape
    print(f"[dimts] data {X.shape} range [{X.min():.3f},{X.max():.3f}] "
          f"hidden_size={args.hidden_size} gpu={args.gpu}", flush=True)

    cfg = build_config(args.hidden_size, T, C, args.max_steps, args.save_cycle,
                       args.results_folder, mmd_alpha=args.mmd_alpha,
                       sampling_timesteps=args.sampling_timesteps)
    print(f"[dimts] mmd_alpha={cfg['model']['params']['mmd_alpha']} (C={C})", flush=True)
    model = DiM_TS(**cfg["model"]["params"]).to(device)
    nparam = sum(p.numel() for p in model.parameters())
    print(f"[dimts] model params = {nparam:,}", flush=True)

    bs = max(1, min(int(args.batch_size), N))
    ds = _ChunkDataset(X, T, C)
    dl = DataLoader(ds, batch_size=bs, shuffle=True, num_workers=0,
                    pin_memory=True, drop_last=(N > bs))

    targs = SimpleNamespace(name="dimts", save_dir=args.results_folder)
    trainer = Trainer(config=cfg, args=targs, model=model,
                      dataloader={"dataloader": dl, "dataset": ds}, logger=None)

    if args.skip_train:
        ms = args.load_milestone
        if ms is None:
            ms = latest_milestone(args.results_folder, T)
            if ms is None:
                raise SystemExit(
                    f"[dimts] --skip_train but no checkpoint-*.pt under "
                    f"{args.results_folder}_{T}; there is nothing to resample")
        restore(trainer, ms, device)
    else:
        print(f"[dimts] training {args.max_steps} steps (grad-accum 2)...", flush=True)
        trainer.train()

    # sample from the EMA copy -- the one just trained, or the one just restored.
    sb = max(1, min(int(args.sample_batch), args.K))
    st = cfg["model"]["params"]["sampling_timesteps"]
    print(f"[dimts] sampling K={args.K} (size_every={sb}, sampling_timesteps={st}, "
          f"{'DDIM' if st < 500 else 'full'})...", flush=True)
    samples = trainer.sample(num=args.K, size_every=sb, shape=[T, C])
    samples = np.asarray(samples)[:args.K].astype(np.float32)  # auto_norm=False -> no unnormalize

    os.makedirs(os.path.dirname(args.out_npy), exist_ok=True)
    np.save(args.out_npy, samples)
    print(f"[dimts] wrote {args.out_npy} {samples.shape} "
          f"range [{float(samples.min()):.3f},{float(samples.max()):.3f}] "
          f"nan_frac={float(np.isnan(samples).mean()):.4f}", flush=True)


if __name__ == "__main__":
    main()
