#!/usr/bin/env python
"""Is TimeVAE using the GPU, and is batch_size=16 why it is slow?

    python scripts/probe_timevae.py            # inside a GPU job

Two questions, both answered by measurement rather than by reading the vendored source:

1. **Is the card actually being used?** `torch.cuda.is_available()` is checked with a
   matmul as well as the flag, because the operating guide records that the flag
   returns True on a wheel built without the card's architecture, where every kernel
   launch then fails.

2. **Where does the time go at batch_size=16?** The vendored default was set for a
   dataset around a hundredth of this cohort. At 176,445 windows it is 11,028
   iterations per epoch, and a batch of 16 x 288 x 1 is far too small to occupy an
   A100 -- the loop becomes launch-bound, which looks exactly like "running on CPU"
   from the outside: one core at 100%, the card idle.

Timing a few epochs at several batch sizes separates those two explanations and says
what a sane batch would cost, which is the number needed to decide whether the run in
flight is worth waiting for.
"""
import sys
import time

import numpy as np

from cgmoutlier._env import check as _envcheck                      # noqa: E402
_envcheck()


def main():
    import torch

    print(f"torch {torch.__version__}", flush=True)
    print(f"cuda.is_available() = {torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        print(f"device = {torch.cuda.get_device_name(0)}", flush=True)
        # The flag alone is not proof; the guide's check is an actual kernel.
        a = torch.randn(2048, 2048, device="cuda", dtype=torch.float32)
        torch.cuda.synchronize(); t0 = time.time()
        for _ in range(10):
            a = a @ a.T / 2048.0
        torch.cuda.synchronize()
        print(f"matmul check OK, {(time.time()-t0)/10*1000:.1f} ms per 2048^3",
              flush=True)

    sys.path.insert(0, "vendor/TimeVAE/src")
    from vae.vae_utils import instantiate_vae_model, train_vae

    N, T, C = 176445, 288, 1
    X = np.random.default_rng(0).standard_normal((20000, T, C)).astype(np.float32)
    print(f"\ntiming on {len(X):,} windows (the real cohort is {N:,})", flush=True)
    print(f"{'batch':>8}{'s/epoch':>12}{'iters/epoch':>14}"
          f"{'h for 1000 ep @176k':>22}", flush=True)

    for bs in (16, 64, 256, 1024):
        vae = instantiate_vae_model(
            vae_type="timeVAE", sequence_length=T, feature_dim=C, batch_size=bs,
            latent_dim=8, hidden_layer_sizes=[50, 100, 200],
            reconstruction_wt=3.0, use_residual_conn=True, trend_poly=0,
            custom_seas=None)
        p = next(vae.parameters())
        t0 = time.time()
        train_vae(vae=vae, train_data=X, max_epochs=1, verbose=0)
        dt = time.time() - t0
        # scale to the real cohort: iterations grow with N, cost per iteration does not
        full = dt * (N / len(X)) * 1000 / 3600
        print(f"{bs:>8}{dt:>12.1f}{int(np.ceil(N/bs)):>14,}{full:>22.1f}"
              f"   (params on {p.device})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
