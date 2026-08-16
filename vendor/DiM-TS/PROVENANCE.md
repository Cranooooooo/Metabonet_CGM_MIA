# DiM-TS

Vendored from the working copy at `project_general/Model_Joint_woMissing/DiMTS`,
**source only**. Excluded: `Data/datasets/` (bundled benchmark CSVs), `_grid_work/`
(checkpoints), `Figure/`, and every compiled artefact under `kernels/*/build/` — the
upstream tree is 74 GB, almost all of it weights and data.

## The compiled kernel is not here, and cannot be

`kernels/` ships the **sources** for the Mamba selective-scan and depthwise-conv CUDA
extensions. The `.so` files are deliberately excluded: they are linked against a
specific torch build, CUDA version and libstdc++, and a binary built here would fail on
any other machine in a way that looks like a code bug. Build them in the DiM-TS
environment — `docs/DIMTS.md` has the steps and the two failure messages you are most
likely to hit.

## Why it needs its own environment

The selective-scan kernel requires a torch build that will not coexist with the other
baselines'. Everything else in this repository runs on one environment; DiM-TS needs a
second, and the adapter therefore invokes it as a subprocess rather than importing it.

## Scale

The official configs are small: Stocks and ETTh1 are 2.65 M parameters; only Energy, at
28 channels, reaches 10.3 M. Ported to T=288, C=1, the 2 M preset here is 0.8x published
Stocks. Anything above ~4 M is an over-capacity condition, not a recommendation.
