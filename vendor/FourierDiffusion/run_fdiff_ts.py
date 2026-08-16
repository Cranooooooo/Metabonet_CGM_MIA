#!/usr/bin/env python3
"""FourierDiffusion wrapper for our 4-dataset baseline comparison.

Bypasses hydra/wandb; constructs Datamodule + ScoreModule + Trainer directly.
Trains in time domain (fourier_transform=False) for comparability with other
baselines that operate in the time domain.

Usage:
    python -u run_fdiff_ts.py --dataset stocks --seed 2023 --gpu 1
"""
import argparse
import json
import os
import random
import sys
import time

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
import pytorch_lightning as pl
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'src'))

from fdiff.models.score_models import ScoreModule  # noqa: E402
from fdiff.schedulers.sde import VPScheduler  # noqa: E402
from fdiff.sampling.sampler import DiffusionSampler  # noqa: E402
from fdiff.dataloaders.datamodules import Datamodule  # noqa: E402
from fdiff.utils.dataclasses import collate_batch  # noqa: E402

# Their DiffusionDataset is in dataloaders module
from fdiff.dataloaders.datamodules import DiffusionDataset  # noqa: E402


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

# Paper-default training: 200 epochs per their conf/trainer/default.yaml
DATASET_HP = {
    'stocks': dict(d_model=72, num_layers=10, n_head=12, lr_max=1e-3,
                   batch_size=64, max_epochs=200),
    'etth':   dict(d_model=72, num_layers=10, n_head=12, lr_max=1e-3,
                   batch_size=64, max_epochs=200),
    'energy': dict(d_model=84, num_layers=10, n_head=12, lr_max=1e-3,
                   batch_size=64, max_epochs=200),
    'kddcup': dict(d_model=120, num_layers=10, n_head=12, lr_max=1e-3,
                   batch_size=64, max_epochs=200),
}


class NpyDatamodule(Datamodule):
    """Minimal Datamodule subclass that loads from a fixed npy file."""
    _npy_path = ''

    def __init__(self, npy_path, batch_size, random_seed):
        self._npy_path = npy_path
        super().__init__(
            data_dir=os.path.dirname(npy_path),
            random_seed=random_seed,
            batch_size=batch_size,
            fourier_transform=False,
            standardize=True,  # standardize -> SDE prior matches; un-standardize post-sample
        )

    @property
    def dataset_name(self) -> str:
        return os.path.basename(self._npy_path).split('_')[0]

    def prepare_data(self) -> None:  # override to skip mkdir/download
        pass

    def download_data(self) -> None:
        pass

    def setup(self, stage='fit'):
        arr = np.load(self._npy_path).astype(np.float32)  # (N, T, D)
        self.X_train = torch.from_numpy(arr).float()
        # No test split needed (we sample N samples after training)
        self.X_test = self.X_train[:1]  # dummy

    def val_dataloader(self):
        # avoid val loop (skip metrics that need ref)
        return None


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True, choices=DATASETS)
    p.add_argument('--window', type=int, default=64)
    p.add_argument('--seed', type=int, default=2023)
    p.add_argument('--gpu', default='0')
    p.add_argument('--max_epochs', type=int, default=None)
    p.add_argument('--batch_size', type=int, default=None)
    p.add_argument('--lr_max', type=float, default=None,
                   help='override per-dataset default lr_max (for stability at large T)')
    p.add_argument('--sample_batch', type=int, default=128)
    p.add_argument('--num_diffusion_steps', type=int, default=1000)
    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()
    set_seed(args.seed)
    pl.seed_everything(args.seed, workers=True)

    hp = dict(DATASET_HP[args.dataset])
    if args.max_epochs is not None:
        hp['max_epochs'] = args.max_epochs
    if args.batch_size is not None:
        hp['batch_size'] = args.batch_size
    if args.lr_max is not None:
        hp['lr_max'] = args.lr_max

    window_tag = f'window_{args.window}'
    out_dir = os.path.join(BASELINE_RESULTS, 'FourierDiffusion', window_tag, f'seed{args.seed}')
    ckpt_dir = os.path.join(HERE, 'checkpoints', f'fdiff_{args.dataset}_seed{args.seed}_T{args.window}')
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    # Skip if this cell's fake already exists (idempotent restart)
    fake_path = os.path.join(out_dir, f'{args.dataset}_fake.npy')
    if os.path.exists(fake_path):
        print(f'[fdiff] SKIP {args.dataset} seed{args.seed} T{args.window}: '
              f'fake already exists at {fake_path}', flush=True)
        return

    # Datamodule
    dm = NpyDatamodule(resolve_truth(args.dataset, args.window), hp['batch_size'], args.seed)
    dm.prepare_data()
    dm.setup('fit')
    N, T, D = dm.X_train.shape
    print(f'[fdiff] truth shape={dm.X_train.shape}', flush=True)
    num_training_steps = (N // hp['batch_size']) * hp['max_epochs']

    # Noise scheduler
    noise_scheduler = VPScheduler(beta_min=0.1, beta_max=20.0,
                                  fourier_noise_scaling=False, eps=1e-5)

    # Score model
    score_model = ScoreModule(
        n_channels=D,
        max_len=T,
        noise_scheduler=noise_scheduler,
        fourier_noise_scaling=False,
        d_model=hp['d_model'],
        num_layers=hp['num_layers'],
        n_head=hp['n_head'],
        num_training_steps=num_training_steps,
        lr_max=hp['lr_max'],
        likelihood_weighting=False,
    )
    n_params = sum(p.numel() for p in score_model.parameters())
    print(f'[fdiff] model: T={T} D={D} d_model={hp["d_model"]} '
          f'layers={hp["num_layers"]} heads={hp["n_head"]} params={n_params:,}', flush=True)

    # Trainer
    trainer = pl.Trainer(
        max_epochs=hp['max_epochs'],
        accelerator='gpu',
        devices=1,
        gradient_clip_val=1.0,
        logger=False,
        enable_progress_bar=True,
        enable_checkpointing=False,
        num_sanity_val_steps=0,
        limit_val_batches=0,
    )
    print(f'[fdiff] START train epochs={hp["max_epochs"]} batch={hp["batch_size"]} '
          f'training_steps={num_training_steps} lr={hp["lr_max"]}', flush=True)
    t0 = time.time()
    trainer.fit(model=score_model, datamodule=dm)
    train_time = time.time() - t0
    print(f'[fdiff] train DONE in {train_time:.1f}s', flush=True)

    # Sample
    print(f'[fdiff] sampling N={N}', flush=True)
    t1 = time.time()
    score_model.eval()
    if torch.cuda.is_available():
        score_model = score_model.cuda()
    sampler = DiffusionSampler(score_model=score_model, sample_batch_size=args.sample_batch)
    out_pieces = []
    remaining = N
    while remaining > 0:
        b = min(args.sample_batch, remaining)
        # Their sampler.sample takes num_samples but does the batching itself; call with batch_size
        sampler.sample_batch_size = b
        x = sampler.sample(num_samples=b, num_diffusion_steps=args.num_diffusion_steps)
        out_pieces.append(x.detach().cpu())
        remaining -= b
    fakes = torch.cat(out_pieces, dim=0)[:N]  # (N, T, D) torch tensor
    # Un-standardize (we set standardize=True): X * std + mean
    if dm.standardize:
        feat_mean, feat_std = dm.feature_mean_and_std
        fakes = fakes * feat_std + feat_mean
    fakes = fakes.numpy().astype(np.float32)
    sample_time = time.time() - t1
    print(f'[fdiff] sampled in {sample_time:.1f}s shape={fakes.shape} '
          f'range=[{fakes.min():.3f},{fakes.max():.3f}]', flush=True)

    fake_path = os.path.join(out_dir, f'{args.dataset}_fake.npy')
    np.save(fake_path, fakes)
    print(f'[fdiff] fake -> {fake_path}', flush=True)

    cfg_out = {**vars(args), **hp, 'D': D, 'T_seq': T, 'N': N,
               'n_params': n_params, 'num_training_steps': num_training_steps,
               'train_time_sec': train_time, 'sample_time_sec': sample_time}
    with open(os.path.join(out_dir, f'{args.dataset}_runcfg.json'), 'w') as fh:
        json.dump(cfg_out, fh, indent=2, default=str)
    print(f'[fdiff] DONE total={time.time()-t0:.1f}s', flush=True)


if __name__ == '__main__':
    main()
