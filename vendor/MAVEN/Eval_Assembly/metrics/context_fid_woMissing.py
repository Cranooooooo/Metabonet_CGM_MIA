"""Context-FID (no missingness).

TS2Vec full-series embedding -> Frechet distance between real and synthetic
embedding distributions. Identical protocol to
eval_ref_1/woMissing_scripts/Utils/context_fid.py.

Inputs : real, fake of shape (N, T, C).
Returns: scalar FID in the TS2Vec embedding space.
"""
from __future__ import annotations

import numpy as np
import scipy

from utils.ts2vec_loader import get_TS2Vec_class


def _calculate_fid(act1: np.ndarray, act2: np.ndarray) -> float:
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
    ssdiff = float(np.sum((mu1 - mu2) ** 2.0))
    covmean = scipy.linalg.sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean))


def context_fid(ori_data: np.ndarray, generated_data: np.ndarray) -> float:
    TS2Vec = get_TS2Vec_class()
    model = TS2Vec(input_dims=ori_data.shape[-1], device=0, batch_size=8, lr=0.001,
                   output_dims=320, max_train_length=3000)
    model.fit(ori_data, verbose=False)
    ori_repr = model.encode(ori_data, encoding_window='full_series')
    gen_repr = model.encode(generated_data, encoding_window='full_series')
    return _calculate_fid(ori_repr, gen_repr)
