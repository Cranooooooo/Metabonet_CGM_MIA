# DiM-TS

DiM-TS is the eighth baseline and the single biggest obstacle to "clone and run".

## What is vendored, and what is not

The **source is here**, under `vendor/DiM-TS/` (see its `PROVENANCE.md`). The compiled
CUDA extensions are not: `kernels/` ships their sources, and a `.so` built on this
machine is linked against one specific torch, CUDA and libstdc++, so shipping it would
produce failures elsewhere that look like code bugs.

DiM-TS needs a Mamba selective-scan kernel built against a torch that will not coexist
with the other baselines'. Everything else here runs in one environment; DiM-TS needs a
second, and `generators/dimts.py` therefore drives it as a subprocess instead of
importing it. Importing `DiMTSGenerator` works in the ordinary environment; only `fit()`
needs the other one, and it checks for the kernel before doing any work.

## Setting it up

```bash
conda create -p /path/to/envs/dimts python=3.10 -y && conda activate /path/to/envs/dimts
pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cu118

# requirements.txt pins torch and triton itself; installing those again from PyPI
# pulls the default build over the cu118 one just installed.
grep -v -E '^(torch|triton)==' vendor/DiM-TS/requirements.txt > /tmp/reqs.txt
pip install -r /tmp/reqs.txt

# Build the kernel from the vendored sources. `pip install selective_scan_cuda_oflex_rh`
# does NOT work -- that name is the MODULE the build produces, not a package on PyPI;
# it was an internal wheel in the previous project. setup.py's MODE=oflexrh emits
# exactly that module. nvcc must be >= 11.8 and must match the torch build.
module load cuda/11.8.0                    # or your site's equivalent
export TORCH_CUDA_ARCH_LIST="8.0"          # A100; naming it avoids building for all
pip install --no-build-isolation ./vendor/DiM-TS/kernels/selective_scan

# the kernel links against the env's libstdc++, not the system one
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
```

On ASPIRE2A this is `scripts/pbs/05_dimts_env.pbs`; do not run it on the login node.

Verify:

```bash
python -c "import selective_scan_cuda_oflex_rh; print('kernel ok')"
```

`ImportError: ... CXXABI_1.3.15 not found` means `LD_LIBRARY_PATH` is not picking up
the environment's `libstdc++`.

## Scale, measured

Official configs: Stocks is `feature_size=6, seq_length=64, hidden_size=128` and ETTh1
`7 x 64` at the same width, both **2.65 M** parameters; only Energy, at 28 channels,
reaches 10.3 M.

**Fewer channels does not make our model smaller.** `C` enters the parameter count only
through `Linear(C, H)` and `Linear(H, C)` — about 1.3 k parameters of difference between
`C=6` and `C=1`. What scales with the data shape is `T`, through `Linear(T, H)`,
`fc_feature = Linear(H, T)` and two `LearnablePositionalEncoding(max_len=T)`, roughly
`4*T*H`. Our `T=288` is 4.5x Stocks' 64, so at equal width our model is slightly
*larger* than the published one. Measured on this cohort (`T=288, C=1`):

| hidden_size | parameters | |
|---|---|---|
| 32 | 223,859 | |
| 40 | 329,027 | |
| 48 | 452,371 | |
| **52** | **522,907** | **closest to 500 k (1.05x)** |
| 56 | 596,707 | |
| 64 | 758,707 | the adapter's default |
| 80 | 1,142,867 | |
| 96 | 1,604,851 | closest to 2 M (0.80x) |
| 128 | 2,762,291 | published Stocks width; 1.04x its 2.65 M |

Treat anything above ~4 M as an over-capacity condition, not as a recommended setting.
Note the other direction too: 500 k against 178 k training windows is deliberately
under-parameterised, which suppresses memorisation and therefore biases the attack
towards a null result. That is a legitimate condition to run, but it is a condition,
not a neutral default.

## Invoking it

Through the ordinary registry, from the ordinary environment:

```python
from cgmoutlier.generators.registry import get
gen = get("dimts")(T=288, C=1, params={"python": "/path/to/envs/dimts/bin/python"})
gen.fit(X).sample(50_000)
```

or set `DIMTS_PYTHON` once and omit `params`. `fit()` writes the training array to a
work directory, runs `vendor/DiM-TS/cgm_train_sample.py` under that interpreter, and
reads the samples back in the same normalised space every other generator produces --
so downstream stages never learn which generator made a file.

`cgm_train_sample.py` came from the previous project, where the cohort was 3-channel
`metabonet_perkg`; this cohort is single-channel CGM.

### ⚠️ It was NOT channel-agnostic, and the first run proved it

An earlier version of this note claimed the script was already channel-agnostic because
it reads `T` and `C` off the array it is given. That was wrong. `build_config` also
inherited `mmd_alpha=0.0008`, which weights a loss term matching the **cross-channel**
correlation distribution between real and generated batches. That term is built from
the off-diagonal entries of the `C x C` correlation matrix, so at `C=1` there are no
channel pairs at all: `torch.corrcoef` returns a 0-d scalar and training dies on the
first step with

```
File "vendor/DiM-TS/eval_utils.py", line 18, in cross_correlation_distribution
    toreturn.append(corr_matrix[index])
IndexError: too many indices for tensor of dimension 0
```

`mmd_alpha` is now an argument, defaulted to **0 when `C < 2`** and to 0.0008
otherwise, and `generators/dimts.py` passes it. Zero is not a tuning choice here — it
is the only defined value for one channel.

Which is what "it has not been run since the port, so treat the first run as
unverified" was there to warn about. Everything downstream of a first run on a new
cohort should still be read that way.
