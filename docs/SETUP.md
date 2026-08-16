# Setting up on a new machine

## The short version

```bash
git clone <this repo> && cd CGM-OutlierMIA
conda create -n cgmoutlier python=3.10 -y && conda activate cgmoutlier
pip install -r requirements.txt
pip install -e .
export PYTHONNOUSERSITE=1

make test       # 35 tests, ~7 s
make smoke      # whole pipeline on synthetic data, CPU, ~2 min
make outliers   # the real 875-subject cohort, ~40 min, CPU
```

No download is needed for any of that. The cohort is in the repository.

## `PYTHONNOUSERSITE=1` is not optional

A numpy in `~/.local/lib/python3.10/site-packages` takes priority over your conda
environment — `conda activate` does not override it — and scikit-learn compiled against
numpy 1.x then fails with a message about dtype sizes that names neither package. The
scripts check for this and refuse to start; `make` exports it. Export it yourself if
you invoke a module directly. Full description in `docs/PITFALLS.md`.

## Threads on a shared machine

numba sizes its pool from `NUMBA_NUM_THREADS` and **ignores** `OMP_NUM_THREADS`, so on
a many-core box the DTW kernel will take the whole machine by default. `make` sets both
from `THREADS` (default 8):

```bash
make outliers THREADS=16
```

## GPU

The outlier stage is CPU-only except `D12`, which is not in the default set. The GPU is
needed for:

- the TS2Vec encoder used by `C9` and `D11` — it falls back to CPU and takes longer
- the generator stage, which is not yet wired up

Weights and the encoded cohort are cached under `results/_encoder_cache/` (225 MB,
gitignored, rebuilt from the cohort in a few minutes).

## torch

Not pinned in `requirements.txt`, because the right build depends on your CUDA. Install
it first:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128   # CUDA 12.8
```

Results in `results/` came from torch 2.7.0+cu128 on an NVIDIA L40S.

## DiM-TS

Needs its own environment and a CUDA kernel built ahead of time. It is the one
generator that will not work from the shared environment above; see `docs/DIMTS.md`.

## Rebuilding the cohort

Only needed to change the construction rules — a different day floor, subject count or
draw. Requires the raw parquet, which is not in the repository:

```bash
python scripts/fetch_data.py --check                  # what is already here
python scripts/fetch_data.py --from /path/to/metabonet_public.parquet
python scripts/build_cohort.py --config configs/data.yaml
```

`build_cohort.py` refuses to overwrite a non-empty cohort directory. Every score in
`results/` was computed against some cohort and nothing in a score file records which,
so overwriting one in place makes the existing results unattributable.

## After changing a method

`tests/test_regression.py` pins the outlier list the shipped cohort produces. When you
change a method deliberately, rerun `make outliers` and regenerate the fixture in the
same commit, so the diff shows exactly which subjects moved:

```bash
cp results/outliers/consensus.json \
   tests/fixtures/consensus_metabonet875_seed2026.json
```
