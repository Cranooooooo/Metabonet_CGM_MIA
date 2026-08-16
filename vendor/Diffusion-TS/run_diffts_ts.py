#!/usr/bin/env python3
"""Diffusion-TS wrapper for our 4-dataset baseline comparison.

Loads <ds>_norm_truth_64_train.npy from Baseline_Results/_truth/window_64/,
trains Diffusion-TS using paper-default per-dataset hyperparameters (seq_length=64),
samples N samples, saves to Baseline_Results/Diffusion-TS/window_64/seed<seed>/<ds>_fake.npy.

Bypasses their CSV pipeline and yaml config — uses a custom NpyDataset that
wraps our pre-windowed truth array. Reuses their Model + Trainer + scheduler.

Usage:
    python -u run_diffts_ts.py --dataset stocks --seed 2023 --gpu 1
"""
import argparse
import json
import os
import random
import sys
import time
from types import SimpleNamespace

def _peek(arg):
    for i, a in enumerate(sys.argv[1:], start=1):
        if a == arg and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
        if a.startswith(arg + '='):
            return a.split('=', 1)[1]
    return None

_gpu = _peek('--gpu')
if _gpu is not None:
    os.environ['CUDA_VISIBLE_DEVICES'] = _gpu

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from Models.interpretable_diffusion.gaussian_diffusion import Diffusion_TS  # noqa: E402
from engine.solver import Trainer  # noqa: E402
from Utils.io_utils import seed_everything  # noqa: E402


BASELINE_RESULTS = os.environ.get('BASELINE_RESULTS', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results'))

def truth_paths(window):
    truth_dir = os.path.join(BASELINE_RESULTS, '_truth', f'window_{window}')
    return {
        ds: os.path.join(truth_dir, f'{ds}_norm_truth_{window}_train.npy')
        for ds in ('stocks', 'etth', 'energy', 'kddcup')
    }


def resolve_truth(dataset, window):
    """Locate the ground-truth windows, tolerating where make_truth_npy.py wrote them.

    Tries, in order: BASELINE_RESULTS/_truth, the repo's data/truth/ (make_truth's
    default), and the shared eval/results/_truth tree. Raises a clear, actionable
    error if none exist instead of a bare FileNotFoundError.
    """
    fname = f'{dataset}_norm_truth_{window}_train.npy'
    repo_root = os.path.abspath(os.path.join(HERE, '..', '..'))
    candidates = [
        os.path.join(BASELINE_RESULTS, '_truth', f'window_{window}', fname),
        os.path.join(repo_root, 'data', 'truth', f'window_{window}', fname),
        os.path.join(repo_root, 'eval', 'results', '_truth', f'window_{window}', fname),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    searched = '\n  '.join(candidates)
    raise FileNotFoundError(
        f"No ground-truth windows for dataset={dataset!r} window={window}.\n"
        f"Searched:\n  {searched}\n"
        f"Build them first:  cd {os.path.join(repo_root, 'data')} && "
        f"python make_truth_npy.py --window {window}\n"
        f"or set BASELINE_RESULTS to a tree containing _truth/window_{window}/."
    )


DATASETS = ('stocks', 'etth', 'energy', 'kddcup')

# Per-dataset hyperparameters (extrapolated from their Config/*.yaml + scale-up for KDDCup)
DATASET_HP = {
    # ds -> dict(n_layer_enc, n_layer_dec, d_model, timesteps, max_epochs, batch_size,
    #            patience, base_lr, warmup, save_cycle)
    'stocks': dict(n_layer_enc=2, n_layer_dec=2, d_model=64, timesteps=500,
                   max_epochs=10000, batch_size=64,
                   patience=2000, base_lr=1.0e-5, warmup=500, save_cycle=1000),
    'etth':   dict(n_layer_enc=3, n_layer_dec=2, d_model=64, timesteps=500,
                   max_epochs=18000, batch_size=64,
                   patience=4000, base_lr=1.0e-5, warmup=500, save_cycle=1800),
    'energy': dict(n_layer_enc=4, n_layer_dec=3, d_model=96, timesteps=1000,
                   max_epochs=25000, batch_size=64,
                   patience=5000, base_lr=1.0e-5, warmup=500, save_cycle=2500),
    # KDDCup D=59 (no paper config); use energy-like scale, slightly bigger d_model
    'kddcup': dict(n_layer_enc=4, n_layer_dec=3, d_model=128, timesteps=1000,
                   max_epochs=25000, batch_size=64,
                   patience=5000, base_lr=1.0e-5, warmup=500, save_cycle=2500),
}


class NpyDataset(torch.utils.data.Dataset):
    """Returns torch (T, D) tensor — matches CustomDataset's __getitem__."""
    def __init__(self, arr):
        # arr: (N, T, D) float32
        self.samples = arr

    def __len__(self):
        return self.samples.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(self.samples[idx]).float()


def build_config(hp, results_folder, seq_length, feature_size):
    """Construct a minimal config dict matching their Trainer's expectations."""
    return {
        'solver': {
            'max_epochs': hp['max_epochs'],
            'gradient_accumulate_every': 2,
            'save_cycle': hp['save_cycle'],
            'results_folder': results_folder,
            'base_lr': hp['base_lr'],
            'ema': {'decay': 0.995, 'update_interval': 10},
            'scheduler': {
                'target': 'engine.lr_sch.ReduceLROnPlateauWithWarmup',
                'params': {
                    'factor': 0.5,
                    'patience': hp['patience'],
                    'min_lr': 1.0e-5,
                    'threshold': 1.0e-1,
                    'threshold_mode': 'rel',
                    'warmup_lr': 8.0e-4,
                    'warmup': hp['warmup'],
                    'verbose': False,
                },
            },
        },
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True, choices=DATASETS)
    p.add_argument('--window', type=int, default=64)
    p.add_argument('--seed', type=int, default=2023)
    p.add_argument('--gpu', default='0')
    p.add_argument('--max_epochs', type=int, default=None,
                   help='override default iters')
    p.add_argument('--batch_size', type=int, default=None,
                   help='override default batch')
    p.add_argument('--sample_batch', type=int, default=256)
    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()
    seed_everything(args.seed)
    set_seed(args.seed)

    hp = dict(DATASET_HP[args.dataset])  # copy
    if args.max_epochs is not None:
        hp['max_epochs'] = args.max_epochs
    if args.batch_size is not None:
        hp['batch_size'] = args.batch_size

    window_tag = f'window_{args.window}'
    out_dir = os.path.join(BASELINE_RESULTS, 'Diffusion-TS', window_tag, f'seed{args.seed}')
    os.makedirs(out_dir, exist_ok=True)
    results_folder = os.path.join(HERE, 'checkpoints', f'diffts_{args.dataset}_seed{args.seed}_T{args.window}')

    # Load truth
    truth_src = resolve_truth(args.dataset, args.window)
    arr = np.load(truth_src).astype(np.float32)  # (N, T, D)
    N, T, D = arr.shape
    print(f'[diffts] truth loaded {truth_src} shape={arr.shape}', flush=True)

    # Build model with per-dataset HP
    model = Diffusion_TS(
        seq_length=T,
        feature_size=D,
        n_layer_enc=hp['n_layer_enc'],
        n_layer_dec=hp['n_layer_dec'],
        d_model=hp['d_model'],
        timesteps=hp['timesteps'],
        sampling_timesteps=hp['timesteps'],
        loss_type='l1',
        beta_schedule='cosine',
        n_heads=4,
        mlp_hidden_times=4,
        attn_pd=0.0,
        resid_pd=0.0,
        kernel_size=1,
        padding_size=0,
    ).cuda()
    n_params = sum(p.numel() for p in model.parameters())
    print(f'[diffts] model: T={T} D={D} d_model={hp["d_model"]} '
          f'enc={hp["n_layer_enc"]} dec={hp["n_layer_dec"]} '
          f'timesteps={hp["timesteps"]} params={n_params:,}', flush=True)

    # Dataloader
    ds = NpyDataset(arr)
    dataloader = torch.utils.data.DataLoader(
        ds, batch_size=hp['batch_size'], shuffle=True,
        num_workers=2, pin_memory=True, drop_last=True,
    )
    dataload_info = {'dataloader': dataloader, 'dataset': ds}

    # Build Trainer
    cfg = build_config(hp, results_folder, T, D)
    trainer_args = SimpleNamespace(
        name=f'diffts_{args.dataset}_seed{args.seed}',
        save_dir=out_dir,
    )
    trainer = Trainer(config=cfg, args=trainer_args, model=model,
                      dataloader=dataload_info, logger=None)

    print(f'[diffts] START train iters={hp["max_epochs"]} batch={hp["batch_size"]} '
          f'lr={hp["base_lr"]}', flush=True)
    t0 = time.time()
    trainer.train()
    train_time = time.time() - t0
    print(f'[diffts] train DONE in {train_time:.1f}s', flush=True)

    # Sample N samples (size_every for memory)
    print(f'[diffts] sampling N={N} (size_every={args.sample_batch})', flush=True)
    t1 = time.time()
    fakes = trainer.sample(num=N, size_every=args.sample_batch, shape=(T, D))
    fakes = np.asarray(fakes)[:N].astype(np.float32)  # (N, T, D)
    sample_time = time.time() - t1
    print(f'[diffts] sampled in {sample_time:.1f}s shape={fakes.shape} '
          f'range=[{fakes.min():.3f},{fakes.max():.3f}]', flush=True)

    # Save
    fake_path = os.path.join(out_dir, f'{args.dataset}_fake.npy')
    np.save(fake_path, fakes)
    print(f'[diffts] fake -> {fake_path}', flush=True)

    cfg_out = {**vars(args), **hp, 'D': D, 'T_seq': T, 'N': N,
               'n_params': n_params, 'train_time_sec': train_time,
               'sample_time_sec': sample_time, 'fake_path': fake_path}
    with open(os.path.join(out_dir, f'{args.dataset}_runcfg.json'), 'w') as fh:
        json.dump(cfg_out, fh, indent=2, default=str)
    print(f'[diffts] DONE total={time.time()-t0:.1f}s', flush=True)


if __name__ == '__main__':
    main()
