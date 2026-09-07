"""Shared evaluation utilities (display + train/test split + time helpers).

Mirrors eval_ref_1/{wo,w}Missing_scripts/Utils/metric_utils.py for the basics
so the new physiological eval suite reports scores in the same format.
"""
from __future__ import annotations

import numpy as np
import scipy.stats


def display_scores(results) -> tuple[float, float]:
    """Print 'mean ± 95% CI (t-distribution)' over the iteration scores.

    Returns the (mean, half-width) tuple for downstream aggregation.
    """
    arr = np.asarray(results, dtype=float)
    mean = float(np.mean(arr))
    if arr.size <= 1:
        sigma = 0.0
    else:
        sem = scipy.stats.sem(arr)
        sigma = float(sem * scipy.stats.t.ppf((1 + 0.95) / 2.0, max(arr.size - 1, 1)))
    print(f"Final Score: {mean} ± {sigma}")
    return mean, sigma


def train_test_divide(data_x, data_x_hat, data_t, data_t_hat, train_rate: float = 0.8):
    """Divide train and test data for both original and synthetic data."""
    no = len(data_x)
    idx = np.random.permutation(no)
    train_idx = idx[: int(no * train_rate)]
    test_idx = idx[int(no * train_rate):]
    train_x = [data_x[i] for i in train_idx]
    test_x = [data_x[i] for i in test_idx]
    train_t = [data_t[i] for i in train_idx]
    test_t = [data_t[i] for i in test_idx]

    no = len(data_x_hat)
    idx = np.random.permutation(no)
    train_idx = idx[: int(no * train_rate)]
    test_idx = idx[int(no * train_rate):]
    train_x_hat = [data_x_hat[i] for i in train_idx]
    test_x_hat = [data_x_hat[i] for i in test_idx]
    train_t_hat = [data_t_hat[i] for i in train_idx]
    test_t_hat = [data_t_hat[i] for i in test_idx]

    return train_x, train_x_hat, test_x, test_x_hat, train_t, train_t_hat, test_t, test_t_hat


def extract_time(data) -> tuple[list[int], int]:
    """Returns each-sequence length list and the maximum."""
    time = []
    max_seq_len = 0
    for i in range(len(data)):
        max_seq_len = max(max_seq_len, len(data[i][:, 0]))
        time.append(len(data[i][:, 0]))
    return time, max_seq_len


def reduce_per_channel(arr: np.ndarray) -> dict:
    """Standard reporting: per-channel array + channel-mean float."""
    arr = np.asarray(arr, dtype=float)
    return {"per_channel": arr.tolist(), "mean": float(np.nanmean(arr))}
