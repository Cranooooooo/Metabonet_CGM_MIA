# MAVEN — Multi-Resolution Visual-View Fusion for Time-Series Generation

Each multivariate time-series window is mapped to 2-D **images** via
exactly-invertible transforms (delay embedding + complex STFT), generated with
per-view **flow matching** in image space, and fused into one sample by a
**retrieval-guided manifold-editing** module. Fusion is post-hoc.

**Latest model design — MAVEN-MaskAE** (`train_maven_maskae.py`): each view's
velocity network is a *mask-conditioned autoencoder transformer* with an explicit
**masked-imputation auxiliary task** (mask whole channels, regenerate them from the
observed ones) and an AE hidden-dim schedule giving coarse (bottleneck) + fine
representations, with **deep-supervision reconstruction loss at both**. See
**`DESIGN.md`** for the design, current empirical status, and the ablation ladder.
`train_maven.py` is the base per-view FM generator (no mask) kept as a reference.

Code-only release: model checkpoints and generated samples are intentionally NOT
included (see `.gitignore`); the four dataset CSVs and the full evaluation suite
(including the TS2Vec used by Context-FID) are.

## Layout
```
train_maven_maskae.py  # LATEST DESIGN: mask-conditioned AE velocity net + masked-imputation task
train_maven.py         # base per-view flow-matching generator (no mask; reference baseline)
visual_transforms.py   # invertible delay-embedding (A) and complex-STFT (B) transforms
fusion_nview.py        # N-view retrieval-guided editor (post-hoc fusion of the per-view samples)
fusion_editor.py       # 2-view editor + shared losses (mmd2/corr_loss/...) imported by fusion_nview
DESIGN.md              # design rationale, empirical status, ablation ladder
datasets/              # stock_data.csv, ETTh.csv, energy_data.csv, kddcup.csv
Eval_Assembly/         # eval suite: metrics/ + utils/ + eval_ref_1/.../ts2vec (Context-FID)
```

## Environment
Python 3.10, PyTorch (CUDA), `torchaudio`, `numpy`, `pandas`, `scikit-learn`.
```
pip install torch torchaudio numpy pandas scikit-learn
```
Single GPU is sufficient (models range 1.7M–52M params; see capacity table).

## Datasets (T = window length; D = channels)
| dataset key      | D  | file                  |
|------------------|----|-----------------------|
| GENERAL_Stocks   | 6  | datasets/stock_data.csv |
| GENERAL_Etth     | 7  | datasets/ETTh.csv     |
| GENERAL_Energy   | 28 | datasets/energy_data.csv |
| GENERAL_KDDCup   | 59 | datasets/kddcup.csv   |

Per-channel min-max scaling to [-1,1], stride-1 sliding windows. Main length T=64
(also T=128, 256). Evaluation uses the [0,1] convention and the FULL sample count.

## Per-dataset capacity & transform parameters
Backbone (same for all views of a dataset): **Stocks 128/4 · ETTh 256/6 · Energy
256/8 · KDDCup 512/8** (hidden_dim / n_layers). Transforms (fixed, no learnable params):
- **Fine views**: delay `tau=4, m=8`  +  STFT `n_fft=16, hop=8`
- **Coarse views**: delay `tau=8, m=16`  +  STFT `n_fft=32, hop=16`

The full model (V3.2) fuses 4 views: fine-delay, fine-STFT, coarse-delay, coarse-STFT.

## A. Reproduce from the provided checkpoints (fast)
Sample a branch from a checkpoint (0 training steps) and evaluate:
```bash
# sample branch A (delay) of Stocks from the trained model
python train_maven.py --dataset GENERAL_Stocks --name repro_stocks --gpu 0 --seed 1 \
  --window 64 --hidden_dim 128 --n_layers 4 --total_iters 150000 \
  --resume checkpoints/maven_stocks_indep.pt --fuse_mode gate --gamma 0 --no_share_noise

# evaluate (4 canonical metrics, lower=better)
python Eval_Assembly/eval_woMissing.py \
  --real outputs/repro_stocks/stocks_norm_truth_64_train.npy \
  --fake outputs/repro_stocks/repro_stocks_branchA_fake.npy \
  --iterations 3 --metrics context_fid cross_correlation discriminative predictive
```

## B. Train from scratch and produce the full (V3.2) result
Branches are independent (`--gamma 0 --no_share_noise`). Train the two **fine**
views and the two **coarse** views (each `train_maven.py` run trains both A and B
and samples paired branch fakes with shared noise):
```bash
H=128; L=4; DS=GENERAL_Stocks; T=64        # (use the capacity table per dataset)
# fine views -> outputs/<name>/<name>_branch{A,B}_fake.npy
python train_maven.py --dataset $DS --name fine  --gpu 0 --seed 1 --window $T \
  --batch_size 256 --hidden_dim $H --n_layers $L --total_iters 150000 \
  --gamma 0 --no_share_noise --fuse_mode gate
# coarse views
python train_maven.py --dataset $DS --name coarse --gpu 0 --seed 1 --window $T \
  --batch_size 256 --hidden_dim $H --n_layers $L --total_iters 150000 \
  --gamma 0 --no_share_noise --fuse_mode gate \
  --delay_tau 8 --delay_m 16 --stft_n_fft 32 --stft_hop 16
```
Fuse the 4 views with the N-view editor (base = strongest fine view by Context-FID),
then evaluate:
```bash
python fusion_nview.py \
  --base   outputs/fine/fine_branchA_fake.npy \
  --others outputs/fine/fine_branchB_fake.npy,outputs/coarse/coarse_branchA_fake.npy,outputs/coarse/coarse_branchB_fake.npy \
  --real   outputs/fine/stocks_norm_truth_64_train.npy \
  --out    outputs/fine/v32_fused.npy --gpu 0 --corr_weight 10

python Eval_Assembly/eval_woMissing.py --real outputs/fine/stocks_norm_truth_64_train.npy \
  --fake outputs/fine/v32_fused.npy --iterations 3 \
  --metrics context_fid cross_correlation discriminative predictive
```
For very large configs (e.g. KDDCup at T=256) add `--micro_batch_size 16` to bound
memory (gradient accumulation; effective batch stays 256). A single branch can also
be trained in isolation with `--only_branch A|B` (independent runs, fuse later).

Notes: `seed=1` throughout. `--iterations` repeats the stochastic metrics
(discriminative/predictive/Context-FID) and reports mean ± 95% CI; cross-correlation
is deterministic.

## Checkpoint inventory (`checkpoints/`, `.pt` = final 150k-iter model)
Naming: `maven_<dataset>[_t<T>]_{indep,coarse}.pt` — `indep` = fine views (delay+STFT),
`coarse` = coarse views. T omitted = T=64. Each file holds `model`, `ema_module`, `iter`.
Branch-parallel runs use `_fine{A,B}`/`_coarse{A,B}` (one branch each).
- T=64: stocks/etth/energy/kddcup × {indep, coarse}  (full V3.2 reproducible for all 4)
- T=128: stocks/etth/energy × {indep, coarse}; kddcup (indep)
- T=256: stocks/etth × {indep, coarse}; energy (indep); kddcup (fineA)
(Some long-window KDDCup/Energy coarse models are still training and can be added.)
