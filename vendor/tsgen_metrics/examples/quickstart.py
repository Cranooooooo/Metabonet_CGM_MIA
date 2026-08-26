"""Minimal usage example."""
import numpy as np
from tsgen_metrics import evaluate

# (N, T, C) = windows, timesteps, channels — real & fake in the SAME value space.
rng = np.random.default_rng(0)
real = rng.standard_normal((500, 64, 3)).astype("float32")
fake = (real + 0.1 * rng.standard_normal(real.shape)).astype("float32")

res = evaluate(real, fake, subsample=500, n_seeds=3, epochs=20, verbose=True)
for name, m in res["metrics"].items():
    print(f"{name:<16} {m['value']:.5f}   [{m['status']}]  ({m['reference']})")
