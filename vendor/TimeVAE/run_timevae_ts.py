#!/usr/bin/env python3
"""TimeVAE wrapper for our 4-dataset baseline comparison.

Loads <ds>_norm_truth_64_train.npy from Baseline_Results/_truth/window_64/,
trains a TimeVAE (paper defaults), samples N prior samples, saves to
Baseline_Results/TimeVAE/window_64/seed<seed>/<ds>_fake.npy.

Usage:
    python -u run_timevae_ts.py --dataset stocks --seed 2023 --gpu 1 \\
        --max_epochs 1000
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

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, 'src')
sys.path.insert(0, SRC)

from vae.vae_utils import instantiate_vae_model, train_vae, get_prior_samples  # noqa: E402


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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True, choices=DATASETS)
    p.add_argument('--window', type=int, default=64)
    p.add_argument('--seed', type=int, default=2023)
    p.add_argument('--gpu', default='0')
    p.add_argument('--max_epochs', type=int, default=1000,
                   help='TimeVAE paper default: 1000')
    p.add_argument('--batch_size', type=int, default=16,
                   help='TimeVAE paper default: 16')
    p.add_argument('--latent_dim', type=int, default=8)
    p.add_argument('--hidden_layer_sizes', type=int, nargs='+',
                   default=[50, 100, 200])
    p.add_argument('--reconstruction_wt', type=float, default=3.0)
    p.add_argument('--use_residual_conn', action='store_true', default=True)
    p.add_argument('--trend_poly', type=int, default=0)
    p.add_argument('--verbose', type=int, default=0,
                   help='1 prints per-epoch loss; we use periodic prints from outer loop')
    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()
    set_seed(args.seed)

    window_tag = f'window_{args.window}'
    out_dir = os.path.join(BASELINE_RESULTS, 'TimeVAE', window_tag, f'seed{args.seed}')
    os.makedirs(out_dir, exist_ok=True)

    # Load truth
    truth_src = resolve_truth(args.dataset, args.window)
    arr = np.load(truth_src).astype(np.float32)  # (N, T, D)
    N, T, D = arr.shape
    print(f'[timevae] loaded truth {truth_src} shape={arr.shape}', flush=True)

    # Instantiate (this internally hardcodes torch.manual_seed(123); we override after)
    vae = instantiate_vae_model(
        vae_type='timeVAE',
        sequence_length=T,
        feature_dim=D,
        batch_size=args.batch_size,
        latent_dim=args.latent_dim,
        hidden_layer_sizes=list(args.hidden_layer_sizes),
        reconstruction_wt=args.reconstruction_wt,
        use_residual_conn=args.use_residual_conn,
        trend_poly=args.trend_poly,
        custom_seas=None,
    )
    # Re-seed AFTER instantiate to override their hardcoded seed=123
    set_seed(args.seed)
    n_params = sum(p.numel() for p in vae.parameters())
    print(f'[timevae] model: latent={args.latent_dim} hidden={args.hidden_layer_sizes} '
          f'recon_wt={args.reconstruction_wt} trend_poly={args.trend_poly} '
          f'residual={args.use_residual_conn} params={n_params:,}', flush=True)
    print(f'[timevae] data: N={N} T={T} D={D} epochs={args.max_epochs} batch={args.batch_size}',
          flush=True)

    # Train
    print(f'[timevae] START train dataset={args.dataset} seed={args.seed}', flush=True)
    t0 = time.time()
    train_vae(vae=vae, train_data=arr, max_epochs=args.max_epochs, verbose=args.verbose)
    train_time = time.time() - t0
    print(f'[timevae] train DONE in {train_time:.1f}s', flush=True)

    # Sample N prior samples
    print(f'[timevae] sampling N={N}', flush=True)
    t1 = time.time()
    vae.eval()
    with torch.no_grad():
        fakes = get_prior_samples(vae, num_samples=N)  # ndarray (N, T, D)
    # get_prior_samples already returns numpy via .cpu().numpy() in their util
    if not isinstance(fakes, np.ndarray):
        fakes = fakes.detach().cpu().numpy()
    fakes = fakes.astype(np.float32)
    sample_time = time.time() - t1
    print(f'[timevae] sampled in {sample_time:.1f}s shape={fakes.shape} '
          f'range=[{fakes.min():.3f},{fakes.max():.3f}]', flush=True)

    # Save
    fake_path = os.path.join(out_dir, f'{args.dataset}_fake.npy')
    np.save(fake_path, fakes)
    print(f'[timevae] fake -> {fake_path}', flush=True)

    cfg = {**vars(args), 'D': D, 'T_seq': T, 'N': N,
           'n_params': n_params, 'train_time_sec': train_time,
           'sample_time_sec': sample_time, 'fake_path': fake_path}
    with open(os.path.join(out_dir, f'{args.dataset}_runcfg.json'), 'w') as fh:
        json.dump(cfg, fh, indent=2)
    print(f'[timevae] DONE total={time.time()-t0:.1f}s', flush=True)


if __name__ == '__main__':
    main()
