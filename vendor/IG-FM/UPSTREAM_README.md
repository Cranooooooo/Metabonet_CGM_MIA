# IG-FM (ICDE 2027) — code & reproducible baseline benchmark

This repository contains the code for **IG-FM** (Imputation-Guided Flow Matching for Multivariate
Time-Series Generation) together with a fully reproducible benchmark against six baselines, all
scored with **one shared metric implementation** for a fair comparison.

This is a **code-only** repository: model checkpoints and generated samples are intentionally NOT
included. The 4 dataset CSVs are provided under `data/`.

## Repository layout

```
data/        Dataset CSVs (stocks, etth, energy, kddcup) + make_truth_npy.py
baselines/   Five baselines, code only:
             DiffWave/  TimeVAE/  Diffusion-TS/  FourierDiffusion/  PaD-TS/
eval/        setup_eval.sh (fetches DiM-TS) + run_eval.py + launch_evals.sh + build_summary_csv.py
IG_FM/       Our model: train_r10.py + run_*.sh launchers
```

**Benchmark:** 6 baselines × 4 datasets (stocks, etth, energy, kddcup) × 3 seeds (2023, 1, 2),
windows {64, 128, 256}. 4 metrics, **all lower-is-better**: Context-FID, Cross-correlation,
Discriminative, Predictive.

> **Why DiM-TS isn't in `baselines/`.** DiM-TS (the 6th baseline, 2026) *also* provides the metric
> code we evaluate everything with. To keep the metrics byte-identical to the original — and to
> avoid redistributing someone else's code — we fetch DiM-TS straight from its upstream repo at a
> pinned commit (Step 1). That single checkout gives you **both** the evaluator **and** the DiM-TS
> baseline.

---

# Quick start (verify one result end-to-end)

This is the smallest path that proves the pipeline works. Copy-paste, top to bottom.

```bash
# 0) Environment (Python 3.10+). TensorFlow + tf-keras are REQUIRED — DiM-TS's
#    discriminative & predictive metrics run in tf.compat.v1 graph mode, whose
#    legacy RNN cells were removed from Keras 3 (the default in TF >= 2.16).
#    run_eval.py sets TF_USE_LEGACY_KERAS=1 for you; tf-keras must be installed.
#    Tip: the CPU build of TensorFlow is enough (run_eval hides the GPU from TF)
#    and avoids CUDA conflicts with torch — `tensorflow-cpu` instead of
#    `tensorflow` is recommended.
pip install torch tensorflow-cpu tf-keras scipy scikit-learn numpy pandas

# 1) Fetch DiM-TS (the evaluator + the 6th baseline) at the pinned commit.
cd eval
./setup_eval.sh                 # clones upstream DiM-TS into eval/DiMTS/ (git-ignored)
cd ..

# 2) Build the ground-truth windows for window length 64.
cd data
python make_truth_npy.py --window 64
cd ..
#    -> data/truth/window_64/<ds>_norm_truth_64_train.npy   (stocks, etth, energy, kddcup)

# 3) Generate samples from ONE baseline (DiffWave) on ONE dataset (stocks), seed 2023.
cd baselines/DiffWave
python run_diffwave_ts.py --dataset stocks --seed 2023 --gpu 0 --window 64
cd ../..
#    -> writes a <...>_fake.npy of shape (N, 64, D); the script prints its exact path.
#    The wrappers auto-discover the truth from data/truth/window_<T>/ (Step 2's default
#    output), so no env var is needed for this quick start. For the full grid below you
#    set BASELINE_RESULTS to keep truth + samples in one shared tree.

# 4) Evaluate that sample against the truth — all 4 metrics, 3 iterations each.
cd eval
python run_eval.py \
    --real ../data/truth/window_64/stocks_norm_truth_64_train.npy \
    --fake <PATH_PRINTED_IN_STEP_3> \
    --iterations 3
#    -> prints "Final Score: <mean> ± <ci>" for Context-FID, Cross-correlation,
#       Discriminative, and Predictive (in that order).
```

That's the whole loop: **fetch evaluator → make truth → make samples → score**. Everything below
just scales this up to the full 6×4×3 grid (and to IG-FM).

---

# Full reproduction

### Step 1 — Fetch DiM-TS (evaluator + 6th baseline)

```bash
cd eval
./setup_eval.sh
# Pinned to commit 8b9fc1a by default. To track upstream main instead:
#   DIMTS_COMMIT=main ./setup_eval.sh
cd ..
```

This clones `https://github.com/yzh8221/DiMTS` into `eval/DiMTS/` (git-ignored, not redistributed).
`run_eval.py` auto-detects it; override with `DIMTS_ROOT=/path/to/DiMTS` if you put it elsewhere.

### Step 2 — Build ground-truth windows

```bash
cd data
TRUTH_OUT_ROOT=../eval/results/_truth python make_truth_npy.py --window 64
TRUTH_OUT_ROOT=../eval/results/_truth python make_truth_npy.py --window 128
TRUTH_OUT_ROOT=../eval/results/_truth python make_truth_npy.py --window 256
cd ..
```

Setting `TRUTH_OUT_ROOT` writes the truth `.npy` straight into the layout the orchestrator expects:
`eval/results/_truth/window_<T>/<ds>_norm_truth_<T>_train.npy`. (Preprocessing: MinMax→[0,1],
stride-1 windows, deterministic seed-2023 shuffle; ETTh's first/date column is dropped.)

### Step 3 — Generate samples from the 5 vendored baselines

Point each baseline's output at the shared `eval/results/` tree with `BASELINE_RESULTS`, so samples
land exactly where the evaluator looks: `eval/results/<Baseline>/window_<T>/seed<S>/<ds>_fake.npy`.

```bash
export BASELINE_RESULTS="$PWD/eval/results"   # run from the repo root
export PYTHON=python                          # or your interpreter / conda env python

# Each baseline's launch_all.sh runs all 4 datasets for one (gpu, seed, window):
#   ./launch_all.sh <gpu> <seed> [window]
for B in DiffWave TimeVAE Diffusion-TS FourierDiffusion PaD-TS; do
  for S in 2023 1 2; do
    for T in 64 128 256; do
      ( cd "baselines/$B" && BASELINE_RESULTS="$BASELINE_RESULTS" PYTHON="$PYTHON" \
            ./launch_all.sh 0 "$S" "$T" )
    done
  done
done
```

Single cell instead (one baseline / dataset / seed / window):

```bash
cd baselines/DiffWave
BASELINE_RESULTS="$PWD/../../eval/results" python run_diffwave_ts.py \
    --dataset stocks --seed 2023 --gpu 0 --window 64
```

Wrappers: `DiffWave/run_diffwave_ts.py`, `TimeVAE/run_timevae_ts.py`,
`Diffusion-TS/run_diffts_ts.py`, `FourierDiffusion/run_fdiff_ts.py`, `PaD-TS/run_padts_ts.py`
(shared flags `--dataset --seed --gpu --window`).

### Step 4 — Generate samples from DiM-TS (6th baseline)

DiM-TS was fetched in Step 1. Run it from its own checkout (`main.py` + `Config/` YAMLs — no
`run_*_ts.py` wrapper) and put its `<ds>_fake.npy` under
`eval/results/DiMTS/window_<T>/seed<S>/`. See `eval/DiMTS/README.md` for its exact commands.

### Step 5 — Generate samples from IG-FM (our model)

```bash
cd IG_FM
# one cell (writes truth + fake .npy under outputs/<name>/):
python -u train_r10.py --dataset GENERAL_Stocks --name igfm_stocks_seed2023 --seed 2023 --gpu 0
# whole grid (4 datasets × ratios for one seed, across 4 GPUs):
./run_seed.sh 2023
cd ..
```

Dataset names are `GENERAL_Stocks`, `GENERAL_Etth`, `GENERAL_Energy`, `GENERAL_KDDCup`. Full flag
list (`--total_iters --batch_size --lr --hidden_dim --lambda_schedule --sampling_steps
--out_dir --resume`, …) is in `train_r10.py`. To score IG-FM, point `run_eval.py` directly at the
`*_fake.npy` and the matching truth it writes (see Step 6), or copy the fake into
`eval/results/IG_FM/window_64/seed<S>/<ds>_fake.npy` to include it in the grid.

### Step 6 — Evaluate everything and build the summary table

After Steps 2–5 have populated `eval/results/`, run the orchestrator (4 GPU workers in parallel
over the whole grid), then aggregate:

```bash
cd eval
./launch_evals.sh                       # evaluates every <baseline,window,dataset,seed> cell found
# (override the grid with env vars, e.g.:  WINDOWS="64" SEEDS="2023" ./launch_evals.sh )
python build_summary_csv.py             # writes eval/results/eval_summary.csv
```

Or score a single pair directly (works for any model, including IG-FM):

```bash
cd eval
python run_eval.py \
    --real results/_truth/window_64/stocks_norm_truth_64_train.npy \
    --fake results/DiffWave/window_64/seed2023/stocks_fake.npy \
    --iterations 3
```

`eval_summary.csv` columns: `baseline, dataset, window, seed, cfid_mean, cfid_ci, cc_mean, cc_ci,
disc_mean, disc_ci, pred_mean, pred_ci, status`.

---

## Requirements

Python 3.10+. Install per the part of the pipeline you run:

**Core — truth building, IG-FM, and evaluation** (covers Steps 2, 5, 6 and the Quick start):

```bash
pip install torch torchvision tensorflow-cpu tf-keras scipy scikit-learn numpy pandas
```

- **`tensorflow` + `tf-keras` are required for evaluation** — DiM-TS's discriminative & predictive
  metrics use `tf.compat.v1` graph-mode RNN cells, which Keras 3 (the default in TF >= 2.16)
  removed. `run_eval.py` sets `TF_USE_LEGACY_KERAS=1` automatically, but the `tf-keras` package
  must be installed or those two metrics crash with *"GRUCell is not available with Keras 3"*.
- Prefer **`tensorflow-cpu`**: `run_eval.py` hides the GPU from TF and runs the post-hoc RNNs on
  CPU (fast enough at these sizes), which also avoids CUDA-library clashes with torch in the same
  process. The GPU build works too, but only if its CUDA version matches torch's.

**Extra dependencies for individual baselines** (Step 3) — install on top of Core as needed:

| Baseline            | Extra packages                                              |
|---------------------|------------------------------------------------------------|
| DiffWave            | — (Core only)                                              |
| TimeVAE             | — (Core only)                                              |
| Diffusion-TS        | `einops`, `ema-pytorch`                                    |
| FourierDiffusion    | `pytorch_lightning`, `diffusers`, `hydra-core`, `einops`  |
| PaD-TS              | `timm`  (pulls in `torchvision`)                          |

```bash
# everything needed to run all five vendored baselines:
pip install einops ema-pytorch pytorch_lightning "diffusers[torch]" hydra-core timm
```

See `eval/requirements_eval.txt` for the evaluation environment.

## Note on excluded artifacts

Checkpoints, generated `.npy` samples, truth windows, logs and result CSVs are intentionally NOT
committed — this is a code-only repository. The fetched DiM-TS checkout (`eval/DiMTS/`) is
git-ignored. Regenerate everything with the steps above.
