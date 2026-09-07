"""Context-FID (with missingness — real-only mask).

Generated samples are assumed to be fully observed at every (n, t, c). To
prevent TS2Vec from latching onto the zero-pattern as a side channel that
trivially separates real from fake, we draw a synthetic fake-mask by
resampling patterns from the real_mask pool and zero-fill fake at those
positions. Both sides therefore enter TS2Vec under the same missingness
distribution, with no auxiliary mask channel.

Falls back exactly to the no-missing version when ori_mask is fully observed
(parity guarantee).

Inputs : real, fake of shape (N, T, C); ori_mask of shape (N, T, C) bool
         (True = observed). ori_mask may be None.
Returns: scalar FID in the TS2Vec embedding space.
"""
from __future__ import annotations

import numpy as np
import scipy

from utils.ts2vec_loader import get_TS2Vec_class
from utils.mask_utils import is_fully_observed, sanitize, resample_masks
from metrics.context_fid_woMissing import context_fid as _context_fid_clean


def _calculate_fid(act1: np.ndarray, act2: np.ndarray, eps: float = 1e-6) -> float:
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
    ssdiff = float(np.sum((mu1 - mu2) ** 2.0))
    sigma1 = sigma1 + eps * np.eye(sigma1.shape[0])
    sigma2 = sigma2 + eps * np.eye(sigma2.shape[0])
    covmean = scipy.linalg.sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean))


def context_fid_masked(ori_data: np.ndarray, generated_data: np.ndarray,
                       ori_mask: np.ndarray | None = None,
                       seed: int = 0) -> float:
    if is_fully_observed(ori_mask):
        return _context_fid_clean(ori_data, generated_data)

    Mo = np.asarray(ori_mask).astype(bool)
    real = sanitize(ori_data, Mo)

    # Synthesize a fake-mask by resampling real-mask patterns, then zero-fill fake
    # at those positions so both sides share the same missingness regime.
    rng = np.random.default_rng(seed)
    Mg = resample_masks(generated_data.shape[0], Mo, rng=rng)
    fake = sanitize(generated_data, Mg)

    TS2Vec = get_TS2Vec_class()
    model = TS2Vec(input_dims=real.shape[-1], device=0, batch_size=8, lr=0.001,
                   output_dims=320, max_train_length=3000)
    model.fit(real, verbose=False)
    ori_repr = model.encode(real, encoding_window='full_series')
    gen_repr = model.encode(fake, encoding_window='full_series')
    idx = np.random.permutation(real.shape[0])
    return _calculate_fid(ori_repr[idx], gen_repr[idx])
