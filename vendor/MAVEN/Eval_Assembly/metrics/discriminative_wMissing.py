"""Discriminative score (with missingness — real-only mask).

Generated samples are assumed to be fully observed. Following the EHR-Safe /
HealthGen / EHR-M-GAN protocol family, we draw a synthetic fake-mask by
resampling patterns from the real_mask pool, then zero-fill fake at those
positions. Both sides therefore enter the GRU classifier under the same
missingness distribution — the classifier cannot use "is this position zero?"
as a discriminative leak. No mask channel is concatenated; the input to the
GRU stays at C channels (same architecture as the no-missing path).

Falls back exactly to the no-missing version when ori_mask is fully observed.

Inputs : real, fake of shape (N, T, C); ori_mask of shape (N, T, C) bool.
Returns: (discriminative_score, fake_acc, real_acc).
"""
from __future__ import annotations

import numpy as np

from utils.mask_utils import is_fully_observed, sanitize, resample_masks
from metrics.discriminative_woMissing import discriminative_score as _disc_clean


def discriminative_score_masked(ori_data, generated_data,
                                ori_mask=None,
                                iterations: int = 2000, batch_size: int = 128,
                                seed: int = 0):
    if is_fully_observed(ori_mask):
        return _disc_clean(ori_data, generated_data,
                           iterations=iterations, batch_size=batch_size)

    Mo = np.asarray(ori_mask).astype(bool)
    real = sanitize(ori_data, Mo)

    # Resample real-mask patterns onto fake so the missingness regime matches.
    rng = np.random.default_rng(seed)
    Mg = resample_masks(generated_data.shape[0], Mo, rng=rng)
    fake = sanitize(generated_data, Mg)

    return _disc_clean(real, fake, iterations=iterations, batch_size=batch_size)
