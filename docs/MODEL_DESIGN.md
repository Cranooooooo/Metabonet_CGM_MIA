# Model design — MAVEN with a privacy-calibrated quantile editing stage

Status (2026-08-30): **the backbone has been trained twice and failed the quality gate both
times; the privacy stage is still unimplemented; and three of its load-bearing assumptions
have now been measured, one confirmed and two refuted.** Read §1.8a (MAVEN fails the gate;
its own ablation ladder makes it worse), §5a (what a release-time directed edit can and
cannot do — the central result), and §2.7/§2.7a (the mask-head localiser is falsified; the
raw and quantile coordinates are two different attacks) before anything else in this file.
Sections written before 2026-08-29 are design and are marked where they have been overtaken.
Written to be read by the agent writing the Method section. Every number below is either read from a result file in this
repo (path given) or computed for this document by a script whose exact operations are
described in §2.3; anything that is an estimate from reading code rather than a
measurement is labelled **[estimate]**, and anything that is a prior rather than a finding
is labelled **[prior]**.

Companion documents: `docs/PAPER_PLAN.md` (what the paper claims and what is measured),
`docs/PITFALLS.md` (§16, §18 in particular — they constrain how we are allowed to read
one of MAVEN's own headline results), `docs/DESIGN.md` (the attack's original
specification).

The model family is fixed: **flow matching, generating in an exactly-invertible transform
space with a frequency-domain view, as implemented by MAVEN in
`/home/users/industry/imperial/lcheng/workspace/Project_CGM/TSGen_REVISE/`.** §1 describes
MAVEN as it stands. **§1.9 is the section the writing agent needs for "what was inherited
and what was changed": it lists every deviation from MAVEN's default configuration, with
the CGM-specific or measurement-specific fact that licenses each.** §2 onwards is the part
that is entirely ours: the privacy objective on the editing stage.

---

## 0. The claim, and the one premise it rests on

The paper's claim is that MAVEN plus our editing stage is **both** better on generation
quality **and** lower on membership-inference risk than the baselines — that the
risk–quality curve moves rather than that we move along it.

That claim needs one premise to be true, and it is worth stating in the Method section
because it is what makes the design non-obvious:

> The attack reads a **tail** property of the released set — the distance from a real
> window to its *nearest* synthetic neighbour. Every fidelity metric in both evaluation
> suites reads a **bulk** property — Context-FID is a Fréchet distance between embedding
> means and covariances, MMD and `w1_marginal` compare distributions, `acf_loss` and
> `cross_correlation` compare second-order structure, and the two neural scores compare
> separability in aggregate. A perturbation that moves the nearest-neighbour tail while
> leaving the bulk in place is therefore not a trade-off; it is arbitrage.

Everything in §2 is an attempt to spend as little bulk distortion as possible per unit of
tail movement. §2.5 gives the arithmetic that says how much room there is: a **directed**
edit is roughly 5–9× cheaper in perturbation norm than an isotropic one for the same
privacy gain, which is 25–80× cheaper in distortion. That factor is the whole design.

---

## 1. MAVEN as it stands

Files, all at the repo root of `TSGen_REVISE/`:
`visual_transforms.py`, `train_maven_maskae.py` (latest design), `train_maven.py` (base
per-view generator, kept as reference), `fusion_editor.py` (2-view editor + shared
losses), `fusion_nview.py` (N-view editor), `Eval_Assembly/` (evaluation suite),
`DESIGN.md`, `README.md`.

### 1.1 Data path

`train_maven.load_dataset_csv(dataset_name, window, seed)`: read CSV → per-column
`sklearn.MinMaxScaler` → affine map to `[-1, 1]` → **stride-1** sliding windows of length
`window` → random permutation seeded by `seed`. Returns `(N, T, D)`. Evaluation and the
fusion stage use the `[0, 1]` convention, `(x + 1) / 2`; the training run writes the real
windows in that convention to `<name>/<ds>_norm_truth_<T>_train.npy`.

Four benchmark datasets ship with the repo: Stocks (D=6), ETTh (D=7), Energy (D=28),
KDDCup (D=59). Main window length T=64, also 128 and 256.

**This path is not usable for us** — see §6.1.

### 1.2 The two image views (`visual_transforms.py`)

Each multivariate window `x ∈ R^{T×D}` is lifted to two 2-D "image" token sequences
`[L, D]` by transforms that are **exactly invertible** and, critically, **linear**. The
linearity is the design's load-bearing property: for the flow-matching linear path,

    Φ((1 − t)·x₀ + t·x₁) = (1 − t)·Φ(x₀) + t·Φ(x₁)

holds exactly, so the velocity target and the FM loss can be defined directly in image
space with no approximation. The channel axis `D` is preserved by both transforms, so a
single velocity-net architecture serves both views and only `seq_len` changes.

**View A — delay embedding** (`DelayEmbed`, built by `make_delay_embed(T, tau, m)`).
A *gather* map: grid `(m, W)` with `W = ceil((T − m)/tau) + 1`; column `j` starts at
`start_j = min(j·tau, T − m)` and cell `(i, j)` copies time index `start_j + i`.
`tau ≤ m` is required and enforced, and guarantees every time index is covered. `encode`
is an index gather; `decode` is `scatter_add_` followed by division by `counts` (how many
image cells map onto each time cell) — i.e. overlap-add averaging. Because every image
cell has exactly one time pre-image, the transform is **cell-aligned**: a time-domain mask
transfers to image space exactly, so the per-cell FM loss support is exact.

**View B — complex STFT** (`STFTEmbedder`, the ImagenTime recipe, arXiv 2410.19538).
`torchaudio.transforms.Spectrogram(n_fft, hop_length, center=True, power=None)` gives a
complex `[B, D, F, frames]`. Real and imaginary parts are each MinMax-normalised **per
frequency bin** to `[-1, 1]` using statistics cached from the training set by `.fit()`
(an affine map, so still FM-safe) and concatenated along the frame axis into a real
`[B, D, F, 2·frames]` image, flattened to tokens `[B, F·2·frames, D]`. `decode`
denormalises, recombines to complex and calls `InverseSpectrogram(length=T)`; a Hann
window at 50 % overlap (`hop = n_fft/2`) makes the inverse exact. Keeping the **full
complex** spectrum is deliberate — the docstring records that a magnitude-only spectrogram
is not invertible and that an earlier attempt made that mistake. This view is **not
cell-aligned**: one time cell smears across frames, so time-domain masks do not transfer.

A third, unused-by-default view B exists: `PeriodFold(T, P)`, a gather-based period fold
with `P` chosen by `pick_period` from the mean rFFT power spectrum (DC excluded) and
capped at `t1_max(T) = floor(1.5·sqrt(T))`; its last row is zero-padded and those cells
carry `flat_idx = -1` and are excluded from every loss.

**Transform settings (README).** Fine views: `delay(tau=4, m=8)` and
`STFT(n_fft=16, hop=8)`. Coarse views: `delay(tau=8, m=16)` and `STFT(n_fft=32, hop=16)`.
The full model (V3.2) fuses **four** views: fine-delay, fine-STFT, coarse-delay,
coarse-STFT. Backbone capacity is per-dataset: Stocks 128/4, ETTh 256/6, Energy 256/8,
KDDCup 512/8 (hidden_dim / n_layers).

### 1.3 The velocity network (`AEMaskVelocityNet`, `train_maven_maskae.py`)

Per view, one mask-conditioned autoencoder Transformer predicting the linear-path
velocity on image tokens.

*Inputs.* `h = in_proj(x) + mask_proj(M_cond) + cond_proj(X_cond) [+ prior_proj(X_prior)]`,
then a learned positional embedding over the `L` image tokens. `cond_proj` and
`prior_proj` are **zero-initialised**, so at initialisation the network is exactly a plain
unconditional FM net and the conditioning is a learned departure from it.

*Trunk.* A stack of `nn.TransformerEncoderLayer`s (`norm_first=True`, GELU,
`dim_feedforward = 4d`) with a per-block width schedule `dims`, default
`[128,128,64,64,64,64,128,128]`. Between blocks, a `Linear` reprojects when the width
changes. The flow time `t ∈ [0,1]` enters through `SinusoidalTime(128)` and a per-block
MLP added to every token. The **minimum width (64) is the coarse representation**; the
final width (128) is the **fine** one. Multi-resolution comes from the width schedule, not
from separate transforms.

*Two heads.* `head_coarse` is attached at `idx_coarse` — the **last** block whose width
equals `min(dims)` — and `head_fine` at the final block. Both are `LayerNorm → Linear → D`
and both predict the velocity, so the reconstruction loss is **deep-supervised at coarse
and fine**.

`TwoViewMaskAE` holds one `AEMaskVelocityNet` per view in a `ModuleDict`, so a single EMA
wraps both.

### 1.4 Training objective (`train_maven_maskae.main`)

Per iteration, per view, two forward passes, in the IG-FM style:

- **GEN pass (unconditional).** `x1g = Φ(randn)`, `t ~ U(0,1)`,
  `x_t = (1−t)·Φ(x₀) + t·x1g`, target `v = x1g − Φ(x₀)`.
  `L_gen = recon(v_fine, v, valid) + coarse_weight · recon(v_coarse, v, valid)`, where
  `recon` is a masked MSE and `valid` is the image-space validity mask (the gather views'
  padding cells are excluded; the STFT view has none).
  Soft prior: `x0g = (x_t − t·v_fine).detach()`.
- **IMPUTE pass (masked imputation).** `sample_image_mask(..., mode='channel', lo, hi)`
  draws a ratio `~ U(mask_lo, mask_hi)` (defaults 0.2, 0.6) and a Bernoulli mask of shape
  `[B, 1, D]` expanded over all tokens — i.e. it masks **whole channels** of the image.
  The net is conditioned on the observed channels' true values `X_cond = M_cond ⊙ Φ(x₀)`
  and, for the masked channels, on the generation pass's soft prior
  `X_prior = (1 − M_cond) ⊙ x0g`. Fresh noise and fresh `t`. `L_imp` is computed **only
  over the masked cells**. The stated purpose is to force `p(channel_j | other channels)`,
  i.e. inter-channel structure, which DESIGN.md identifies as the weakness on
  high-channel datasets.
- `L_view = [L_imp@fine + cw·L_imp@coarse] + λ(it)·[L_gen@fine + cw·L_gen@coarse]`,
  summed over views. `λ` from `--lambda_schedule` (default constant 2);
  `cw = --coarse_weight` (default 1.0).

AdamW, lr 2e-4, weight decay 1e-4, grad clip 1.0, EMA decay 0.999, bf16 autocast,
gradient accumulation to an effective batch of 256. Milestones every `--milestone_every`
(30 000) iterations sample with a reduced `--milestone_steps` (100).

### 1.5 Sampling

Unconditional: `mask = 0`, `X_cond = 0`, `X_prior = None`, **fine head only**. Euler
integration of the FM ODE from `t = 1` to `t = 0` in `sampling_steps` (default 200) uniform
steps, `x ← x − v_fine·(t_i − t_{i+1})`, starting from `Φ(randn(B, T, D))`. Decode per
view, map to `[0,1]`, save one `.npy` per branch.

In `train_maven.sample_maven`, both branches integrate from the **same time-domain noise
draw** `z` (shared-noise pairing), so `out_A[i]` and `out_B[i]` are two views of one draw.
The docstring is explicit that this is required for per-sample fusion to mean anything —
with independent noise the convex combination's variance collapses.

### 1.6 The existing editing stage — the fusion editor

`fusion_editor.FusionEditor` (2 views) and `fusion_nview.FusionEditorN` (N views). Both
are **post-hoc** on frozen branch outputs, in `[0,1]` time-domain space:

    fused = clamp( x_base + α · smooth(V(h))·Uᵀ · x_base·(1 − x_base),  0, 1 )

- **Input `h`**, per cell: `concat[x_base, {x_other,i − x_base}, n_mean − x_base, n_std,
  n_med − x_base]` — `5·D` channels for two views, `(N+3)·D` for N. The neighbour
  **median** trajectory is included on the stated grounds that the marginal mean and std
  collapse over temporal coherence.
- **Trunk**: `Linear → learned positional embedding → 2-layer TransformerEncoder →
  LayerNorm → V_head` to rank `r = 8`.
- **Time smoothing**: `V` is low-pass filtered by a fixed depthwise Gaussian over time,
  `σ = 3`, kernel width `2·round(3σ) + 1 = 19`. Stated purpose: kill high-frequency jitter.
- **Channel coupling**: `U ∈ R^{D×r}` is a **shared** basis, so the residual is
  channel-coupled and never per-channel. Stated purpose: preserve cross-correlation.
- **Bounded-skew gate**: `x_base(1 − x_base)`, which vanishes at the `[0,1]` walls.
  Ablatable via `--no_bounded_skew`.
- **α schedule**: cosine warm-up over `warmup_frac = 0.15` of steps to `alpha_max`
  (default **0.003**), then cosine decay to `alpha_end_frac = 0.25` of that, multiplied by
  a learned `softplus` gain normalised to 1 at initialisation. The docstring says the cap
  exists to stay "in the small-α C-FID basin".

**Retrieval** (`build_ts2vec_retrieval` / `build_retrieval`). A TS2Vec encoder
(`output_dims = 320`) is fitted **on the real training windows**; both the real set and the
base-branch fakes are embedded with `encoding_window='full_series'`; a cosine-metric
`sklearn.NearestNeighbors` returns the `K = 16` nearest **real training windows** for each
generated sample. Their per-cell mean, std and median are the editor's conditioning input.
Cached to `_ts2vec_knn_K16_base<A|B>.npy`. `--no_retrieval` substitutes random real windows
as an ablation.

**Losses** (all fidelity terms):
- `mmd2(fused, real_batch)` — multi-bandwidth Gaussian MMD², `σ ∈ {1,2,4,8,16}`, on
  flattened windows; weight `--mmd_weight` (1.0).
- `corr_loss(fused, real)` — squared error between `D×D` channel correlation matrices;
  weight `--corr_weight` (0.5 in the 2-view file, **10.0** in `fusion_nview.py`).
- anchor `MSE(fused, x_base.detach())`, weight annealed **linearly 1.0 → 0.05** over the
  first half of training.
- `L_pred` — a **TSTR-MAML lookahead**: an inner GRU predictor `_PFake` is trained one step
  on the *fused* batch; a one-step functional copy (`torch.func.functional_call`) is then
  evaluated on **real** data and the resulting L1 is differentiated back into `fused`.
  This directly targets the `predictive` metric. Weight `--pred_weight` (1.0).
- `L_disc` — an in-loop GRU discriminator trained real-vs-fused, with the editor
  minimising `BCE(D(fused), "real")`. Directly targets the `discriminative` metric.
  `--disc_weight` default **0.0** (off); absent entirely from `fusion_nview.py`.

`train_maven.JointFusionEditor` is the same editor ported inside the training loop
(`--fuse_mode editor`), retrieval stripped, `alpha_max` 0.05, so fusion gradients reach
both branches. The default `--fuse_mode gate` is instead a learned per-channel sigmoid
gate initialised at 0.5 (a naive average, free to move toward pick-best).

### 1.7 Evaluation (`Eval_Assembly/`)

`eval_woMissing.py` runs eight metrics, lower is better: `context_fid` (TS2Vec-embedding
Fréchet), `cross_correlation` (lag-0 channel-pair correlation-matrix MAE),
`discriminative` (`|0.5 − acc|` of a post-hoc GRU classifier), `predictive` (TSTR
one-step-ahead MAE on real), plus four physiological supplements — `w1_marginal`
(per-channel W1 on flattened values), `acf_loss` (per-channel multi-lag autocorrelation
L2), `w1_peak_amp` (W1 on detected peak amplitudes), `w1_ipi` (W1 on inter-peak
intervals). `--iterations` repeats the stochastic ones and reports mean ± 95 % CI;
`cross_correlation` is deterministic. `eval_wMissing.py` is the missing-data variant, not
needed for our gap-free windows.

### 1.8 Reported status, and one thing we must not inherit

`DESIGN.md` reports (2026-06, Energy and Stocks-128, lower is better):
- masked reconstruction **strongly improves Discriminative** (Stocks-128 DS 0.002–0.02 vs
  DiM-TS 0.027);
- **Context-FID plateaus** and does not fall with training (Stocks-128 ≈ 0.10 vs DiM-TS
  0.034; Energy 0.28–0.36 vs 0.104), verified not to be a code bug;
- the ablation ladder hypothesises that the plateau is structural — routing *generation*
  through the 64-d bottleneck plus coarse deep supervision caps distributional fidelity —
  and proposes `--coarse_weight 0`, a constant-width trunk, an explicit channel-correlation
  loss, and a λ ramp.

> **FLAG — do not inherit this conclusion as stated.** The claim that the masked-imputation
> task works rests on the Discriminative score. `docs/PITFALLS.md` §16 measured that a
> single-fit GRU discriminative score is **one draw of a lower bound** whose draws are
> **bimodal** wherever a separable feature exists: over eight restarts on byte-identical
> samples, spread was 0.015–0.06 where a model was genuinely indistinguishable and
> **0.37–0.49** where it was not, and the same file scored 0.9475 and 0.5908 in two
> processes. §18 measured the worse case: at a reduced sampling budget the single-run
> reading **inverted the direction** in all three of our cells — eight restarts said 50-step
> samples were far *more* separable, one run said 2.5× *less*.
>
> `Eval_Assembly` running `--iterations 3` is better than one fit but does not fix this,
> because it reports a **mean**, and a mean averages a classifier that found the feature
> with one whose optimisation failed. A DS of 0.002 from a 3-iteration mean is precisely
> the reading our own data says is least informative.
>
> **Request:** re-report MAVEN's Discriminative as **max over ≥ 8 restarts with the
> spread**, using the sweep in `scripts/disc_stability.py`, and treat "masked
> reconstruction works" as unsettled until `context_fid` or the W1 metrics move with it.
> This cuts both ways and is not a criticism of MAVEN specifically: **our own edit will be
> judged on the same metric**, so a spurious DS improvement attributed to our privacy stage
> is exactly as likely as a spurious one attributed to theirs. Fixing the metric protects
> our claim as much as it tests MAVEN's.

---

## 1.8a MEASURED 2026-08-29: MAVEN fails the gate, and the ladder makes it worse

§1.8 was a warning read off upstream's `DESIGN.md`. It has now been tested on CGM. Two
single-model runs, both on `d1_c1`, both on the `base` subject set (475 subjects, 5,741
windows) so that the comparison against DiM-TS is the same cell, the same checkpoint and
the same reference. Jobs `scripts/pbs/M3_maven_pilot.pbs` and `M5_maven_ladder.pbs`;
results in `results/pilot_maven/` and `results/ladder_maven/`.

| run @ 30k steps | Context-FID | discriminative | predictive | skew diff | kurt diff |
|---|---|---|---|---|---|
| **DiM-TS base** (`sweep_d1_c1_ms3`) | **0.0611** | **0.0238** | 0.0121 | 0.0291 | 0.1517 |
| MAVEN, default 2-view | 0.2256 | 0.120 | 0.0123 | 0.4167 | 0.9117 |
| MAVEN, ladder: constant width + `coarse_weight 0` | **0.2721** | **0.174** | 0.0132 | 0.4678 | 0.9511 |

`discriminative` for the two MAVEN rows is `|0.5 − max|` over **8 restarts**, as §1.8
requires; the raw accuracies are 0.5042–0.6200 (spread 0.1158) and 0.5775–0.6742 (spread
0.0967). The pre-registered gate is `configs/experiment.yaml`'s `no_worse_than_dimts_base`.

**Three readings, in order of how much they change.**

**(a) The default configuration fails by 3.7× on the gate metric.** Against the
`experiment.yaml` reference of 0.0948 — DiM-TS at 100k and st500 — it is 2.4×; against
DiM-TS at the *matched* 30k checkpoint it is 3.7×. The matched comparison is the honest
one and it is the worse one. **MAVEN wins nothing in this table**, and the distribution-shape
columns are where it loses worst: skewness and kurtosis differences are 14× and 6× DiM-TS's,
which is the value distribution — the very coordinate §2.3 measured the leak in.

*(An earlier draft of this row read `predictive = 0.0096` and called it MAVEN's one win. That
number is `Eval_Assembly`'s `predictive_score_tstr`, while every other cell in the table is
`tsgen_metrics`; the same suite reads 0.0123 against DiM-TS's 0.0121. Mixing the two suites
in one row is the error, and it flattered our own model — which is the direction such errors
always seem to run.)*

**(b) The ladder is falsified, and that is a result about MAVEN rather than about us.**
`DESIGN.md` names the suspected cause of its Context-FID plateau — generation routed
through the 64-d bottleneck plus coarse deep supervision — and lists `--coarse_weight 0`
and a constant-width `--dims` as the tests. Nobody upstream ran them. Run together on
CGM, with **more** parameters than the default, they moved Context-FID 0.2256 → 0.2721
and the discriminator's max 0.6200 → 0.6742. **Both changes, in the direction upstream
predicted would help, made both gate metrics worse.** §1.9.2 adopted the ladder as our
default on the strength of that hypothesis; that adoption is withdrawn (§1.9.2 amended).
One confound to state: constant width also raises capacity, so the failure is of the
package, not necessarily of either change alone. The two single-change rungs were not run
because a 3.7× gap that widens under the joint change does not merit 6 more GPU-hours to
attribute.

**(c) The restart protocol caught a false positive on OUR OWN model, which is what §1.8
said it was for.** On the pilot's samples the single-fit readings were:

    tsgen_metrics, one fit    discriminative = 0.0050     "better than DiM-TS's 0.0238"
    Eval_Assembly, one fit    discriminative = 0.1100
    8 restarts, max           discriminative = 0.1200

A 24× disagreement between the two single fits and the max, and the two suites disagree
with **each other** by 22× on the same file. Reading either single fit, MAVEN passes the
discriminative half of the gate comfortably. §1.8 wrote that "a spurious DS improvement
attributed to our privacy stage is exactly as likely as a spurious one attributed to
theirs"; the first one it caught was ours. This is `PITFALLS.md` §16 firing on new data
and it is the strongest argument in the repo for never quoting a single fit.

**What remains untested for MAVEN.** One confound survives both runs: **30,000 steps may
simply be too few.** DiM-TS is converged by 20k on this cell (Context-FID 0.060 at 20k,
0.061 at 30k), but MAVEN's loss was still falling slowly at 30k. Upstream reports the
Context-FID *plateaus* rather than descends, which predicts a longer run will not help —
so a single 100k run (~9 GPU-hours) closes the question either way and is the last thing
MAVEN is owed before the host decision is reopened.

---

## 1.9 Deviations from MAVEN for CGM — what is inherited, what is changed, and why

**This is the section to read when writing "what was inherited and what was changed".**

MAVEN's reported results are on Energy (D = 28), KDDCup (D = 59), ETTh (D = 7) and Stocks
(D = 6) at T = 64–256. Our cohort is **C = 1 or 2 channels at T = 288 or 2016**, sampled
every 5 minutes, with a known exact daily period. Several of MAVEN's mechanisms are
explicitly motivated in its own `DESIGN.md` by the high-channel regime, and those
motivations do not transfer. Each change below is justified by a fact about CGM or by one
of our measurements; nothing here is a preference.

Note the shape of the result: **almost every deviation is a selection among configurations
MAVEN already implements** (`--fuse_mode`, `--coarse_weight`, `--dims`, `--mask_mode`, the
`PeriodFold` branch, the STFT parameters), not an invention. Implementation risk stays
low and the model stays recognisably in the group's line.

### 1.9.0 Inherited without change

Flow matching with the linear path as the generative mechanism; generation in an
**exactly-invertible, linear** transform space rather than on raw samples (the linearity is
load-bearing — it is what makes `Φ(x_t) = (1−t)Φ(x₀) + tΦ(x₁)` exact and the velocity target
well-defined in image space, so any view we substitute must keep that property); at least
one **frequency-domain** view; the Transformer velocity net on image tokens with the
sinusoidal flow-time embedding; EMA; Euler integration of the FM ODE at sampling; the
*existence* of a post-hoc step that combines views; and the `Eval_Assembly` protocol.

We also propose **no training-time privacy mechanism** — no DP-SGD, no privacy gradient
clipping, no sampling-time output noise. §2.5 gives the reason (an isotropic perturbation
is roughly 5–9× more expensive per unit of privacy than a directed one) and §5 gives the other
(a training-time knob costs a retrain per operating point).

### 1.9.1 Make the masked-imputation task non-degenerate at every C by masking cells

> **SUPERSEDED BY SUPERVISOR DECISION, 2026-08-28.** The instruction was: *"你应该改造
> MAVEN 让它能适配单通道和双通道"* — adapt MAVEN so it fits both one and two channels,
> rather than choosing which channel count is headline. **The whole "drop it at C = 1 /
> ablation at C = 2" framing below is withdrawn, and §8 Q1 is closed.** The diagnosis
> below is still correct and still worth reading; only the remedy changed.
>
> **The remedy.** `sample_image_mask(..., mode='cell')` — a flag `train_maven_maskae.py`
> already ships (`--mask_mode {channel,cell}`, line 158) — draws
> `bernoulli(full((B, L, D), ratio))` instead of `bernoulli(full((B, 1, D), ratio)).expand(...)`.
> At D = 1 that is L independent draws over the view's tokens, so 20–60 % of a window's
> tokens are hidden and predicted from the rest: genuine temporal imputation, non-degenerate,
> with `M_tgt.sum()` of order 10^5 per batch so the `clamp(min=1.0)` never binds. At D = 2 it
> **subsumes** the published objective — a cell masked in channel *j* can be predicted from
> the observed cell in channel *k* at the same token, and whole-channel masking is the
> degenerate corner of the same distribution. It is also the missingness CGM actually has.
>
> **Three facts checked in the source, not inferred.** (i) The mask is drawn *inside* the
> per-view loop, `sample_image_mask(bsz, I_b.shape[1], D, ...)` at line 235, so each view
> masks in its own image space and **nothing has to transfer between the gather and STFT
> views** — §1.2's cell-alignment obstacle does not apply, because it was an obstacle to a
> proposal (one time-domain mask pushed through both views) that this is not. (ii) `sample_save`
> passes `zero, zero, None` for `(M_cond, X_cond, X_prior)`, so the change is **training-only**
> and the released pipeline is byte-identical either way. (iii) `Xc = M_cond * I_b` conditions
> on real observed cells, so the pass is closer to autoencoding than the GEN pass is and
> **may raise memorisation**. That is not argued away: the pass is ablated at both C and
> measured through our own attack.


> **Consequence, stated plainly, because it decides what the Method section describes.**
> MAVEN-MaskAE's defining contribution over MAVEN-base is the masked-imputation auxiliary
> task. **At C = 1 that task carries essentially no gradient signal: it is an
> identically-zero loss on about 60 % of draws and self-distillation on the remaining 40 %,
> and it cannot learn inter-channel structure because a one-channel dataset has no channel
> pairs.** Our two headline cells, `d1_c1` and `d7_c1`, are C = 1. So **if the headline
> cells stay single-channel, the model we train is MAVEN-base, not MAVEN-MaskAE** — the
> mechanism the upstream design is named for is switched off, and the paper cannot claim it.
> The alternatives are to promote the two-channel cells `d1_c2` / `d7_c2` to headline, or to
> present the model honestly as MAVEN-base plus our editing stage. This is §8 Q1 and
> §1.9.7, and it is the first decision to take.

**Fact.** `DESIGN.md` motivates the task in one sentence: it "forces the net to learn
`p(channel_j | other channels)` → **inter-channel structure**, the weakness on high-channel
datasets (e.g. Energy, D=28)". CGM has one or two channels.

**What it actually degenerates to.** `sample_image_mask(mode='channel')` draws a Bernoulli
mask of shape `[B, 1, D]`. At **D = 1** that is one coin flip per window: with probability
`ratio` the single channel is masked, so `M_cond = 0` everywhere, the net sees no observed
values, and `X_prior` is the generation pass's own detached soft prior — self-distillation,
not imputation; with probability `1 − ratio` nothing is masked, `M_tgt.sum() = 0`, and
`recon` divides by `clamp(min=1.0)` so `L_imp` is **exactly zero**. Averaged over
`ratio ~ U(0.2, 0.6)`, the auxiliary task is ~60 % an identically-zero loss and ~40 %
self-distillation. It cannot learn inter-channel structure because there are no channel
pairs. At **D = 2** there is exactly one ordered channel pair and four mask draws, two of
them degenerate — a very thin signal for a mechanism designed for D = 28.

**Precedent in this repo.** `generators/dimts.py` already sets `mmd_alpha = 0` when C < 2,
with the comment: "the term matches cross-CHANNEL correlation distributions and one channel
has no channel pairs". Same argument, same conclusion.

**Change.** At C = 1, drop the impute pass; the objective becomes pure per-view generation
FM. At C = 2, keep it as an ablation and measure it, optionally with `--mask_mode cell`.

**A second, privacy-side reason — flagged [prior], not measured.** An auxiliary objective
that reconstructs masked content from 40–80 % observed context is a *conditional
reconstruction* objective, and conditional reconstruction is nearer to memorisation than
unconditional generation is. We have no measurement of this and should not assert it. The
experiment is cheap once MAVEN runs: train with and without the impute pass at C = 2, run
the frozen attack, report both axes. If it holds, it is a finding worth reporting on its own
— a fidelity-motivated auxiliary task that costs privacy — and it parallels the retrieval
argument in §3.

### 1.9.2 Constant-width trunk, generation supervised at the fine head only

**Fact.** MAVEN's own `DESIGN.md` reports that **Context-FID plateaus** and does not fall
with training (Stocks-128 ≈ 0.10 vs DiM-TS 0.034; Energy 0.28–0.36 vs 0.104), verified not
to be a code bug, and its own ablation ladder states the suspected cause: "routing
*generation* through the 64-d bottleneck + coarse deep-supervision caps distributional
fidelity", with items 1 (`--coarse_weight 0`) and 2 (constant-width `--dims`) as the tests.
The ladder is open; nobody has run it.

**Why that is disqualifying as a *default* for us specifically.** Our entire claim is gated
on Context-FID (`PAPER_PLAN.md` Tip 5: "quality is a gate, not a second metric"; TimeVAE was
rejected at Context-FID 0.856 against DiM-TS's 0.095). A backbone whose one known open
defect is precisely the gate metric cannot be the default configuration of a paper claiming
better quality *and* lower risk.

**Change.** Default to a **constant-width trunk** with `coarse_weight = 0` on the generation
path, keeping the coarse head only as a deep-supervision regulariser on the impute pass
where that pass survives (C = 2). MAVEN's AE width schedule becomes the ablation rather than
the default. **This is executing MAVEN's own ablation ladder, not departing from it** — and
for CGM it should be run first, with whichever configuration wins adopted.

**Note the coupling.** With §1.9.1 there is no impute pass at C = 1, so the coarse head has
nothing left to supervise, and `AEMaskVelocityNet` collapses exactly to
`train_maven.VelocityNet`. That is not a coincidence: it is the correct reading that
MAVEN-MaskAE's additions over MAVEN-base are all channel-count-motivated.

**AMENDED 2026-08-29 — the ladder was run and it FAILED. This change is withdrawn as a
default.** Constant width plus `coarse_weight = 0`, together, on `d1_c1` at 30k: Context-FID
0.2256 → **0.2721**, discriminative max-over-8 0.6200 → **0.6742**. Both gate metrics moved
the wrong way, with more parameters than the configuration it was meant to improve. §1.8a
has the table. The paragraph above is kept because its *reasoning* was right — a backbone
whose one known open defect is our gate metric cannot be the default — but its conclusion,
that the ladder is the fix, is now measured to be false. **Do not write the ladder as an
adopted change; write it as a hypothesis of MAVEN's that we tested and refuted.**

### 1.9.3 A day-fold view in place of the coarse delay view, at T = 2016 only

**Fact.** CGM has a known, exact period: 288 samples = 24 h at 5-minute resolution, and
**T = 2016 = 7 × 288 exactly**. MAVEN already implements this transform —
`make_period_fold(T, P)`, cell `(r, c) → r·P + c`, exactly invertible with **zero padding**
when `P` divides `T`. It is unused because `pick_period` clamps the period to
`t1_max(T) = floor(1.5·sqrt(T))`, which is 67 at T = 2016 — a feasibility bound written for
settings where T ≤ 256 and the period is *unknown and must be estimated from the rFFT*. For
CGM the period is known a priori and needs no estimating, so the cap is not protecting
anything.

**Why it matters here.** A `(7, 288)` image puts **day index on one axis and time of day on
the other**. That is the coordinate system our own localisation result is computed in
(`results/matrix/localise/*/per_timestep.json`: hour-of-day leakage peaking near 09:00), and
it is where a seven-day generator's structure actually lives — the day-to-day recurrence of
a patient's routine. A delay embedding, by contrast, represents a seven-day window as
overlapping 16-sample trajectory fragments and has no axis along which "the same hour on a
different day" is adjacent. It is also **cheaper**: L = 2016 tokens against the coarse delay
view's `m·W = 16 × 251 = 4016`, and attention is O(L²).

**Change.** For the seven-day cells, the gather view is `PeriodFold(T=2016, P=288)` with the
`t1_max` cap lifted (documented as: the cap guards an *estimated* period; ours is known).
**Scope: d7 only.** At T = 288 the daily period equals the whole window (rFFT bin k = 1), so
there is no fold to make and `pick_period` under the cap would return an arbitrary 25; keep
delay + STFT at T = 288.

### 1.9.4 Retune the STFT to physiological timescales

**Fact.** Upstream's fine spectral view is `n_fft = 16, hop = 8`. At 5-minute sampling that
is an **80-minute analysis window with a 40-minute hop** and F = 9 bins. Everything slower
than 80 minutes — the postprandial recovery limb, the overnight drift, the circadian
component — collapses into the DC bin.

**Why our measurements make that the wrong resolution.** The `zscore` transform, which
destroys level and scale while keeping shape, gives the **weakest** attack in every cell
(0.527 / 0.657 / 0.467, §2.3 and `scripts/report/transform_auc.py`). Level and slow scale
are where the individual-attributable signal lives — so the DC and lowest bins are the
privacy-relevant part of the spectral view, and collapsing all slow structure into a single
bin is the wrong resolution for fidelity and for our own analysis alike.

**Change.** Tune `n_fft` to hours rather than minutes. Candidate for T = 288:
`n_fft = 64, hop = 32` — a 5.3-hour analysis window, F = 33, frames = 10, so L = 660,
essentially the same cost as upstream's L = 666 with far better low-frequency resolution.
**[estimate]** — this is a candidate requiring the pilot, and the pilot is cheap because it
changes the transform only, not the model.

### 1.9.5 Gate fusion, not editor fusion, at C = 1

**Fact.** The fusion editor's stated reason for a *shared* rank-r channel basis `U` is
"preserves cross-correlation; never per-channel". At C = 1 there is no cross-correlation to
preserve, `U ∈ R^{1×8}` collapses to a scalar times a rank-1 map, and `corr_loss` and the
`cross_correlation` metric are both vacuous. What remains of the editor is a
Gaussian-smoothed scalar field times `x(1−x)` — **plus** a TS2Vec retrieval index over the
**training set** that conditions every generated sample on the mean, std and median of its
16 nearest real training windows, with the anchor to `x_base` annealed to 0.05. That is the
pipeline's most direct memorisation channel (§3, Option 3, and §7 failure mode 8).

So at C = 1 the fusion editor pays a privacy cost for a mechanism whose stated benefit does
not exist.

**Change.** Default to `--fuse_mode gate` at C = 1 — a learned per-channel sigmoid, which at
D = 1 is a **single scalar** convex combination of the two views. Keep editor fusion as an
ablation, and as the default at C = 2 where the cross-correlation argument is real, with its
retrieval corpus moved to the reference split (§3, Option 3).

**Correction — the gate is not "privacy-neutral by construction", and the honest form of the
argument is stronger.** An earlier draft claimed the gate has no access to real data. That is
false. `train_maven.maven_step`, `else` branch:

```
g = maven.gate()
fused_gen = g * xhat0_ts['A'].detach() + (1 - g) * xhat0_ts['B'].detach()
L_fuse = F.mse_loss(fused_gen, x0)          # x0 is the real training batch
```

weighted by `--fuse_alpha` (default 0.1). The branches are detached, but **the gate logit
itself is fitted against real training windows.**

The distinction that carries the argument is therefore not *access* versus *no access*, but
**a global constant against a per-sample retrieval channel**:

- the gate's entire dependence on the training set is `feature_size` scalars — **one number
  at C = 1** — fitted as a single global mixing weight over the whole training set;
- the fusion editor conditions **every generated sample individually** on the per-cell mean,
  std and median of *its own* 16 nearest real training windows.

Only the second can transport an individual training window's statistics into an individual
released sample, which is the mechanism a membership attack exploits. A single global scalar
is a bounded channel by inspection; a per-sample retrieval channel is not. At sampling time
the gate is a stored constant (`train_maven.sample_maven`, `g = maven.gate().detach()`), so
fusion touches no real data at inference at all.

**The released pipeline is not retrieval-free, and the Method section must say so.**
QuantileShift's term (b) builds a nearest-neighbour index over the *training* windows'
quantile vectors and queries it **per released sample** (§2.6b). We do not remove retrieval
from the pipeline; we **invert its sign** — retrieval is used to push away rather than to
pull toward. That inversion is the design, and it is also the origin of its most specific
attack surface: a repulsive retrieval channel carves a low-density region around each
protected training window, which is exactly what §7 failure mode 5 (hole detection) names and
specifies a measurement for. Claiming we "removed retrieval" would be both wrong and a weaker
position than the truth.

**Consequence for the paper's structure, and it is a good one.** At C = 1 the released
pipeline then has exactly **one** editing stage, it is ours, and its purpose is privacy.
That is a cleaner Method section and a much cleaner ablation table than "two editors, one of
which fights the other".

### 1.9.6 Two views, not four

**Fact.** V3.2 fuses four views (fine/coarse × delay/STFT). With §1.9.2 the AE's coarse/fine
representation is gone and with §1.9.3 the coarse gather view is replaced, so the remaining
coarse/fine transform pair is largely redundant. Four backbones is 4× the training cost and
creates the capacity-matching problem in §8 Q8 — every other baseline in this project is a
single model matched to a parameter budget within ±3 % (IG-FM at +4.6 % is a documented
exception).

**Change.** Two views for the headline configuration: one gather view (delay at T = 288,
day-fold at T = 2016) and one spectral view. Four views as an ablation. Keep `share_noise`
(both views integrate from the same time-domain noise draw, so they are paired views of one
sample — required for per-sample fusion to mean anything) and the `gamma` cross-view
co-training term as ablations: both are channel-count-independent and neither has a privacy
argument against it that I can see.

### 1.9.7 What this adds up to, and the cost we must state

At **C = 1** the recommended configuration is, in effect, MAVEN's own *base* per-view
flow-matching generator — `train_maven.py`, `VelocityNet`, `--fuse_mode gate --gamma 0` —
with a spectral view retuned to physiological timescales and a day-fold gather view at seven
days. Every deviation is a configuration MAVEN already supports.

**The cost is real and the Method section must state it: MAVEN-MaskAE's headline mechanism
is the one we would switch off at C = 1.** If the paper is to present MAVEN-MaskAE rather
than MAVEN-base, the two-channel cells (`d1_c2`, `d7_c2`) must become the headline — which
is the same decision as §8 Q1. That is not a bad option: `d1_c2` is the only cell where
outliers clearly separate from controls (arm AUC 0.698 against 0.562 for `d1_c1`), so
promoting it serves the privacy story and the model story at once. **This is the single
decision the supervisor should make first, because it determines what §1 of the Method
describes.**

**What moves with the decision, and what does not.** A Method section can be written to
survive either resolution, provided the partition is right.

*Unchanged under either resolution:* flow matching on the linear path; exactly-invertible
**linear** transform views with at least one spectral view; the Transformer velocity net, EMA
and Euler ODE sampler; two views; `share_noise` pairing; and the editing stage of §2 applied
after fusion.

*Changes with the decision:*
1. whether the impute pass exists at all (§1.9.1);
2. whether the trunk is constant-width or MAVEN's width schedule with coarse deep
   supervision (§1.9.2);
3. gate fusion versus editor fusion (§1.9.5);
4. which cells are headline — **and therefore whether the backbone is named MAVEN-base or
   MAVEN-MaskAE in the paper**, which is a claim about what was built, not a label;
5. **the data partition** — at C = 1 with gate fusion there is no retrieval, so the reference
   split is no longer needed *for retrieval*, but a held-out split is still needed as the
   calibration split of §2.6c; at C = 2 one split may have to serve both, which is §8 Q9;
6. **the metric table** — `cross_correlation` and `corr_loss` are vacuous at C = 1 and live at
   C = 2;
7. **§7 failure mode 7** (per-channel quantile editing preserves Spearman but not Pearson
   cross-correlation) exists only at C = 2, and so does its mitigation — one shared Dirichlet
   weight vector and one shared repulsion scale per window instead of per-channel draws;
8. **§1.9.1's privacy-side [prior]** (conditional-reconstruction objectives raise
   memorisation) is only testable at C = 2, because at C = 1 there is no impute pass to
   ablate.

### 1.9.7b Commitment status of each change — read this before writing any of it as settled

The changes in §1.9 do **not** all have the same standing, and flattening them would
overstate the design. Three tiers:

| tier | changes | how to write it |
|---|---|---|
| **Adopted** — rests on determinate facts; a pilot decides whether it is *better*, not whether it is *valid* | §1.9.1 (drop the impute pass at C = 1), §1.9.3 (day-fold at T = 2016), §1.9.5 (gate fusion at C = 1), §1.9.6 (two views) | as design decisions, with the alternative as the ablation |
| **Decision adopted, value pending** | §1.9.4 (spectral retune) — that upstream's 80-minute analysis window is wrong for 5-minute CGM is determinate; `n_fft = 64, hop = 32` is **[estimate]** | "the spectral view is retuned to physiological timescales; the parameters are set by pilot" |
| **Ablation ladder, not a commitment** | §1.9.2 (constant width, `coarse_weight = 0`) | MAVEN's own untested hypothesis about its own plateau, and we have never trained MAVEN on CGM. Write it as: the paper commits to running the ladder and adopting the winner. **Do not assert a fix for a plateau we have not reproduced.** |

### 1.9.8 Summary table

| MAVEN component | CGM decision | licensing fact |
|---|---|---|
| flow matching, linear path | **inherit** | family constraint |
| invertible linear transform views, ≥ 1 spectral | **inherit** | family constraint; linearity makes the FM target exact in image space |
| Transformer velocity net, EMA, Euler ODE sampler | **inherit** | — |
| masked-imputation task (channel masking) | **drop at C = 1; ablation at C = 2** | motivated in `DESIGN.md` for D = 28; degenerate at D = 1 (60 % zero loss, 40 % self-distillation); `dimts.py` sets `mmd_alpha=0` at C < 2 for the same reason |
| AE width schedule + coarse deep supervision on generation | **constant width, `coarse_weight = 0` on gen** | MAVEN's own ablation ladder names this as the suspected cause of its Context-FID plateau; Context-FID is our gate |
| coarse delay view at T = 2016 | **replace with `PeriodFold(P=288)`** | CGM's period is known exactly and 2016 = 7×288; `t1_max` caps an *estimated* period; L 2016 vs 4016 |
| STFT `n_fft = 16, hop = 8` | **retune (candidate 64 / 32 at T = 288)** | 80-minute analysis window at 5-minute sampling; `zscore` result says level and slow scale carry the signal |
| editor fusion with training-set retrieval | **gate fusion at C = 1; editor at C = 2 with reference-split retrieval** | `U`'s cross-correlation rationale is vacuous at D = 1; the editor's retrieval is a *per-sample* channel into the training set, whereas the gate's dependence is `D` global scalars (§1.9.5) |
| four views | **two for the headline, four as ablation** | coarse/fine redundancy after the two changes above; capacity matching against single-model baselines |
| any training-time privacy mechanism | **do not add** | §2.5 (isotropic is roughly 5–9× more expensive) and §5 (a retrain per operating point) |

---

## 2. The privacy objective on the editing stage

### 2.1 What the defence has to move

The frozen attack (`src/cgmoutlier/attack/statistic.py`, `set_reduce="min"`,
`subject_reduce="mean"`): for target subject `s` with real windows `R_s` and released set
`S`,

    d(s, S) = mean over windows i of [ min over released k of ‖R_s[i] − S[k]‖₂ / sqrt(T·C) ]

and the score is `gap = d(s, S_base) − d(s, S_member)`, with the outlier arm compared
against the control arm by Mann–Whitney AUC. The defence must remove the *excess
proximity* that membership creates — raise `d(s, S_member)` back to `d(s, S_base)` — while
leaving the released distribution where it is.

### 2.2 The measured scale of what must be removed

From `results/matrix/sweep/attack/*/summary.json`, frozen variant, medians over the
13-patient outlier arm:

| cell | milestone (steps) | arm AUC | median `d_in` | median `d_out` | median gap |
|---|---|---|---|---|---|
| d1_c1 | 3 (30 k) | 0.840 | 0.0974 | 0.0985 | 0.00240 |
| d1_c1 | 10 (100 k) | 0.515 | 0.0798 | 0.0978 | 0.00670 |
| d1_c2 | 3 (30 k) | 0.822 | 0.1342 | 0.1450 | 0.00350 |
| d1_c2 | 10 (100 k) | 0.645 | 0.1169 | 0.1565 | 0.00937 |
| d7_c1 | 3 (30 k) | 0.959 | 0.1172 | 0.1192 | 0.00571 |
| d7_c1 | 4 (40 k) | 0.864 | 0.1160 | 0.1183 | 0.02058 |

Distances are per-cell RMS in the cohort's z-clipped units (Euclidean divided by
`sqrt(F)`), so they are directly comparable to the data's own scale. **The membership
signal is 2–7 % of the nearest-neighbour distance at the maximum-contrast checkpoint,
rising to ~9 % at 100 k and 18 % for the seven-day cell at 40 k.** That is the size of
the perturbation the edit has to produce. It is small, and its smallness is the
opportunity.

### 2.3 What kind of information leaks — the finding the design is built on

Measured on `d1_c1` (DiM-TS, 100 k-step release, 13 outliers and 13 record-length-matched
controls, the same K-matching and the same nearest-neighbour machinery as the frozen
attack; `set_reduce="min"`, `subject_reduce="mean"`), comparing the statistic in raw space
against the same statistic computed on **per-channel sorted** windows.

**This measurement has been computed twice and the two computations disagree.** Both are
recorded, because the disagreement is itself informative (§9). The authoritative version is
the second: `scripts/report/verify_sorted_gap.py`, log `logs/M1_verify.live.log`, which uses
the pipeline's own `_match` and therefore the attack's own `default_rng` subsample. The
first is mine, which reproduced `_match` with `np.random.RandomState` because the
environment I ran in has numpy 1.14 and no `default_rng`. The two also aggregate
differently — mine is a **ratio of medians over targets** (median gap ÷ median `d_in`),
the rerun is the **mean over targets of the per-target ratio** `gap_i / d_in_i`.

| space | computation | `d_in` | outlier gap / `d_in` | control gap / `d_in` | outlier : control |
|---|---|---|---|---|---|
| raw | ratio of medians (mine) | 0.0929 | +10.9 % | +8.1 % | 1.3× |
| raw | **mean of per-target ratios (rerun)** | 0.0884 | **+36.3 %** | **+11.9 %** | **3.1×** |
| sorted | ratio of medians (mine) | 0.0186 | +14.3 % | −0.5 % | n/a (control negative) |
| sorted | **mean of per-target ratios (rerun)** | 0.0189 | **+20.9 %** | **+1.7 %** | **12.3×** |

`d_in` agrees to within 5 % between the two computations in both spaces, so the distance
computation is not what differs; the relative gaps differ, through both the aggregation and
the subsample.

**The quantity to report is the ratio between the arms, not either arm's absolute value.**
It is stable in sign and rough magnitude across both computations, whereas the individual
percentages are not:

> **Sorting widens the separation between the arms roughly fourfold — from about 3× in raw
> space to about 12× in quantile space.**

That is exactly the property QuantileShift needs: an axis on which outliers are
distinguished and typical patients are not.

**An earlier draft of this section claimed the control arm's quantile-space gap was
zero. It is not — it is +1.7 %, and that claim has been removed.** The correct statement is
the ratio above. In raw space everyone appears to leak, because the raw distance is
dominated by generic agreement on the shape of a day; in quantile space the controls leak
an order of magnitude less than the outliers, but not nothing.

**The aggregation-free version, and what the paper should lead with.** Both rows above
require choosing a normalisation and an aggregation. The **arm AUC** requires neither — it
is rank-based over the 13 × 13 target pairs — and it is already computed and pre-registered
(`scripts/report/transform_auc.py` over `results/matrix/localise/*/per_transform.json`):

| cell | arm AUC, raw | arm AUC, sorted |
|---|---|---|
| d1_c1 | 0.562 | **0.828** |
| d1_c2 | 0.698 | **0.781** |
| d7_c1 | 0.627 | **0.905** |

all sorted values with p < 0.01, against the weakest transform — `zscore`, destroying level
and scale — at 0.527 / 0.657 / 0.467. **The paper should lead with this table and use the
ratio above as the mechanistic reading of it**, because the AUC is free of both the
aggregation choice and the subsample sensitivity that the two computations exposed.

**Recommended convention going forward:** report the mean over targets of the per-target
ratio `gap_i / d_in_i` (the rerun's convention), because it normalises each patient by their
own distance scale before aggregating rather than mixing scales; give the median of the same
per-target ratios beside it, since n = 13 and the gap distribution is skewed; and give the
arm AUC as the primary number. My ratio-of-medians should not be used — it is a ratio of two
separately-aggregated quantities and has no per-patient interpretation.

**Why sorting is the right coordinate, precisely.** Sorting a window per channel yields its
empirical quantile function `q ∈ R^T` at levels `u_t = (t − ½)/T`. For two windows of equal
length, the L2 distance between their sorted vectors is exactly `sqrt(T)` times the
Wasserstein-2 distance between their empirical value distributions. So the leaking quantity
is a **W2 distance between per-window glucose value distributions**, and the edit must act
on that object.

**One useful inequality, honestly bounded.** Sorting preserves the norm, and by the
Hardy–Littlewood rearrangement inequality `⟨sort x, sort y⟩ ≥ ⟨x, y⟩`; therefore

    ‖x − y‖₂ ≥ ‖sort x − sort y‖₂     for every pair of windows.

A guaranteed margin `δ` in quantile space therefore *certifies* a raw-space floor of `δ` on
the attack's own distance. **But we measured the slack and it is large**: the sorted
distance is 0.185× the raw one by my computation and 0.214× by the rerun — the one figure
the two agree on closely — so a `δ` big enough to certify the operating `d_in ≈ 0.09` would
exceed the entire quantile-space distance scale. The certificate is real but loose by about
5×. What it genuinely buys is one sentence, not a theorem: **no released sample can
lie within `δ` of any training window in raw space**, which rules out the copy-paste regime
by construction. It should be stated at that strength and no higher.

### 2.4 Where in the value range the leak sits

New measurement for this document. The sorted-space gap decomposes exactly over quantile
level (once the nearest neighbour is fixed, the squared distance is a sum over levels), so
we can read the excess (outlier minus control) per quantile band. Twelve equal bands of
24 levels (d1, T=288) or 168 levels (d7, T=2016); *share* is the fraction of the total
excess, *enrichment* is share × 12:

| band `u` | d1_c1 share (enrich) | d1_c2 share (enrich) | d7_c1 share (enrich) |
|---|---|---|---|
| 0.00–0.08 | 15.7 % (**1.88×**) | −0.4 % (−0.05×) | 17.2 % (**2.07×**) |
| 0.08–0.17 | 12.5 % (1.50×) | 4.2 % (0.51×) | 0.9 % (0.11×) |
| 0.17–0.33 | 22.4 % (1.35×) | 24.7 % (1.48×) | 7.4 % (0.44×) |
| 0.33–0.58 | 41.1 % (1.64×) | 37.9 % (1.52×) | 23.2 % (0.93×) |
| 0.58–0.92 | −18.9 % (−0.57×) | 21.3 % (0.64×) | 22.0 % (0.66×) |
| 0.92–1.00 | 27.2 % (**3.26×**) | 12.3 % (1.47×) | 29.3 % (**3.52×**) |

Total excess: 0.0519 (d1_c1), 0.2523 (d1_c2), 0.7327 (d7_c1); in every cell the control
arm's total is one to two orders of magnitude smaller than the outlier arm's — 0.0032,
0.0144, −0.0014 against 0.0551, 0.2667, 0.7312. **Note that this is the same
order-of-magnitude separation §2.3 reports as a ratio, not a claim that the control arm is
at zero.**

> **The whole of this subsection rests on the superseded computation, and its stability has
> not been established.** These figures come from the `RandomState` subsample that §2.3's
> rerun showed the per-arm numbers to be sensitive to. An earlier draft argued that the
> *shares* below should be trusted ahead of the totals because they are internally
> normalised. **That argument is withdrawn.** Normalising removes a scale factor; it does
> not remove sensitivity to *which* released samples the K-matching subsample drew. Worse,
> there is a specific mechanism pointing the other way: thinning a released set raises
> nearest-neighbour distances most where the released density is *lowest*, which is the
> distribution's tails — so the extreme bands, the ones carrying the largest enrichment and
> the ones `ρ(u)` would lean on hardest, are exactly where a min-over-samples statistic is a
> high-variance order statistic and where share instability would be worst.
>
> **Required before `ρ(u)` may be fitted (§2.6b):** re-derive the band decomposition under
> the pipeline's own `_match` across ≥ 5 subsample seeds and report the per-band spread of
> the *shares*. This needs no GPU and no retraining. Until it is done, the profile below is
> a motivating observation, not a measurement the design may rest on, and the Method section
> should say so.

**The top band is enriched in all three cells, by 3.3–3.5× in the two single-channel
cells. The bottom band is enriched ~2× in the single-channel cells and is flat in
`d1_c2`.** Together the two extreme bands — 17 % of the quantile levels — carry 43 %
(d1_c1) and 47 % (d7_c1) of the excess.

**But the profile is not simply U-shaped, and `ρ(u)` must therefore be fitted rather than
assumed.** The mid band `0.33–0.58` is also enriched in all three cells (1.64× / 1.52× /
0.93×) and it is the largest single contributor in `d1_c1` and `d1_c2`; the band that is
consistently *de*-enriched is `0.58–0.92`, the upper-middle, which is negative in `d1_c1`.
And `d1_c2` — the only cell where outliers clearly separate from controls — has a flat
bottom band. So the robust statement across all three cells is the **top band's 1.5–3.5×
enrichment** and the **upper-middle band's suppression**; the bottom band's enrichment holds
only in the single-channel cells. This is exactly why §8 Q7 asks whether `ρ(u)` is fitted on
a disjoint split or reported as an oracle, and why uniform `ρ` must be run as the ablation.

Clinically the enriched bands are hyperglycaemic excursions first, nocturnal and
hypoglycaemic values second. This **reconciles with, rather than contradicts, the hour-of-day finding**
(`results/matrix/localise/*/per_timestep.json`: leakage concentrated in waking hours,
peaking around 09:00, outliers exceeding controls by ~16×). The high tail of a patient's
value distribution is realised in the morning; it is the *values* that are memorised, and
the hour is where they happen to occur. The Method section should say this explicitly,
because a reader who has both figures will otherwise think they disagree.

### 2.5 Why a *directed* edit, and how much room there is

To raise a nearest-neighbour distance from `d₀` to `d₀ + g`:

- an edit **directed** along the `(S_nn − R)` direction needs `‖Δ‖ = g`;
- an **isotropic** perturbation needs `‖Δ‖ = sqrt((d₀+g)² − d₀²) ≈ sqrt(2·d₀·g)`, because
  distance adds in quadrature.

The ratio is `sqrt(2·d₀/g)`, pure arithmetic from the `d₀` and `g` of the §2.2 table —
which come from the pipeline's own `results/matrix/sweep/attack/*/summary.json` and are
untouched by the subsample question in §2.3. Evaluated on every row of that table:

| cell | milestone | `d₀` | `g` | `sqrt(2·d₀/g)` |
|---|---|---|---|---|
| d1_c1 | 2 (20 k) | 0.0965 | 0.00192 | 10.0× |
| d1_c1 | **3 (30 k)** | 0.0974 | 0.00240 | **9.0×** |
| d1_c1 | 10 (100 k) | 0.0798 | 0.00670 | 4.9× |
| d1_c2 | 2 (20 k) | 0.1438 | 0.00210 | 11.7× |
| d1_c2 | **3 (30 k)** | 0.1342 | 0.00350 | **8.8×** |
| d1_c2 | 10 (100 k) | 0.1169 | 0.00937 | 5.0× |
| d7_c1 | **3 (30 k)** | 0.1172 | 0.00571 | **6.4×** |
| d7_c1 | 4 (40 k) | 0.1160 | 0.02058 | 3.4× |

**Quote the operating point, not a range.** Across the whole table the factor runs
3.4×–11.7×, so "5–9×" is not a defensible summary — the d7_c1 40 k row falls outside it. The
statement that spans all three cells without cherry-picking is at the **maximum-contrast
checkpoint** (milestone 3), which is the operating point §5 argues everything should be read
at: **6.4×–9.0×**. Use that, and name the checkpoint.

**And note which way it moves.** `sqrt(2·d₀/g)` *decreases* in `g`: the advantage of
directedness is largest where the leak is subtle and smallest where it is gross (3.4× at
d7_c1's 40 k row, where `g` is 18 % of `d₀`). So this design helps most in exactly the
regime a well-trained, early-stopped model is in, and least against a badly memorising one.
That is an honest limitation and it belongs in the text.

**The distortion factor is a prediction, not a measurement.** If distributional distortion
is second order in `‖Δ‖`, the same privacy gain costs the *square* of the norm ratio —
roughly **40×–80× less fidelity at the maximum-contrast checkpoints**. The second-order
behaviour is not assumed on generic grounds: a generic displacement has a non-zero mean in
embedding space and would cost *linearly*. It is **engineered** — the barycentric mixing of
§2.6a maps each sample into the convex hull of its own neighbours in quantile space, and the
per-level affine recalibration explicitly restores the first two moments per quantile level,
so the linear term is removed by construction. What is *not* removed by construction is the
repulsion term's systematic displacement (§2.6b), whose first-order effect the recalibration
only partly catches — it fixes per-level mean and variance, not higher moments and not the
TS2Vec embedding mean that Context-FID actually reads. **The η-sweep is the test:** plot
fidelity degradation against `η` and read the exponent. If it comes out linear rather than
quadratic, the recalibration is not doing its job and the 40–80× claim must be withdrawn.

This is the argument that a privacy edit belongs in a stage with **access to a retrieval
index over the training data**, and it is the argument against every defence that spends
its budget isotropically — additive output noise, DP-SGD gradient noise, dropout at
sampling time. Those sit on the existing curve by construction. This one need not.

Two honest caveats:
- The argument is first order in one nearest neighbour. After the edit a *different*
  released sample can become the nearest, capping the achievable gain at roughly the
  spacing between the 1st and 2nd nearest neighbours. That cap is measurable and it must
  be measured before anything else (§7, failure mode 2).
- **[estimate]** Reading the code, the existing fusion editor's own displacement is
  `α_max · corr · x(1−x) ≲ 0.003 · O(1) · 0.25 ≈ 1–2 × 10⁻³` per cell in `[0,1]` units,
  i.e. `2–4 × 10⁻³` in `[-1,1]` units — **the same order as the membership signal `g`
  itself**. So the editing stage is already operating at the right scale, and a privacy
  term does not need a new order of magnitude, but `α` will probably have to rise by a
  small factor. The measurement that settles this is one line once MAVEN runs:
  `RMS(fused − base)` against `g`.

### 2.6 The mechanism — QuantileShift

**Name — print RQE.** Asked directly whether to print "QuantileShift", the answer is: mild
objection, print **RQE — Rank-preserving Quantile Editing** instead. "Shift" suggests a
translation and undersells a barycentric mix plus a repulsion term; **rank preservation is
the property the entire argument rests on** — nothing moves in time, which is what §2.3
licenses and what §4 predicts `acf_loss` and `w1_ipi` will show — so it belongs in the name,
and "editing" keeps the supervisor's framing. `QuantileShift` is retained below as a
deprecated alias so nothing in this document breaks.

Be clear about the standing of this: it is a **preference, not a finding**, and it is the
lowest-stakes item in the document. Do not spend a second cycle on it — make the change if
the section is still open, flag the name in the draft for the supervisor either way, and
settle it before Results, where renaming stops being free.

A rank-preserving, W2-barycentric edit of the per-window value distribution, with a
retrieval-driven repulsion term calibrated to zero rather than maximised.

**Space and decomposition.** Take a fused sample `x ∈ R^{T×D}`. For each channel `c`:

1. `π_c = argsort(x[:, c])` with a deterministic tie-break; `q_c = sort(x[:, c]) ∈ R^T`.
   This is exact and invertible: `x[t, c] = q_c[rank_c(t)]`.
2. Edit `q_c → q'_c` with a **monotone** map.
3. Reassemble `x'[t, c] = q'_c[rank_c(t)]`.

**What this buys, exactly.** The permutation is untouched, so *every rank statistic of the
window is unchanged*: the time of the peak and of the nadir, the ordering of every pair of
timepoints, the rank autocorrelation at every lag, and — since the same holds per channel —
the Spearman cross-channel correlation. **Nothing moves in time.** That is the direct
consequence of §2.3: we edit the only axis that carries an individual-attributable signal
and we leave alone the axis that finding 1 says does not carry it. If `q'` is additionally a
*monotone* function of `q`, the edited window is a monotone reparametrisation of its own
value axis, and `w1_ipi` (inter-peak intervals) is structurally invariant while
`w1_peak_amp` moves only through the monotone map.

**Term (a) — barycentric mixing (the fidelity-preserving displacement).**
Let `N(x)` be the `m` nearest **generated** samples of `x` in quantile space — an index
built over the *released set itself*, never over real data. Draw `w ~ Dirichlet(α·1_m)` and
set `q̃ = Σ_j w_j q_j`.

In one dimension the Wasserstein-2 barycentre of a set of distributions is *exactly* the
weighted average of their quantile functions. So `q̃` is a genuine W2 barycentre of nearby
generated value distributions: monotone by construction (a convex combination of
non-decreasing vectors is non-decreasing), and a *plausible* value distribution rather than
a blurred one — averaging densities blurs a bimodal pair into a plateau, averaging quantile
functions translates and interpolates it. This is what moves the sample off any single
training window's quantile vector *along the data manifold*, which is the cheap direction.

Mixing contracts second moments. Correct it by a per-level affine recalibration
`q ↦ γ_u·(q − m_u) + m'_u`, with `2·T·D` scalars fitted in closed form on the released set
so that the edited set's per-level mean and variance match the unedited set's. No
retraining, no access to real data.

**Term (b) — retrieval repulsion (the privacy displacement).**
Build a nearest-neighbour index over the **training** windows' quantile vectors
`{sort(R_i, axis=time)}`. The holder has them; this is the same kind of object
`build_ts2vec_retrieval` already builds, in a different metric and for the opposite
purpose. For each generated sample, find its nearest training quantile vector `q^R` and
displace:

    q' = q̃ + η · ρ(u) ⊙ (q̃ − q^R) / ‖q̃ − q^R‖

with `ρ(u)` a per-quantile-level allocation profile normalised to unit L2, and `η` the
single global knob (units: the same per-cell RMS as `g`, so `η ≈ g` is the natural
starting point). Monotonicity of `q'` is enforced by an isotonic projection (PAVA, O(T));
PAVA firing means `η` is too large for that sample and its activation rate is logged as a
diagnostic.

**`ρ(u)` is uniform by default, and non-uniform `ρ` is an extension that is not yet
licensed.** §2.4's per-band profile is the motivation for having `ρ` at all, but it was
computed under the superseded subsample and its share-level stability has not been
established — see the box in §2.4, which also gives a specific mechanism (density-dependent
thinning) by which the extreme bands, the ones a non-uniform `ρ` would lean on hardest, could
be the *least* stable. So:

- **Default, and what the Method section should describe: `ρ(u) ≡ 1`** — margin allocated
  uniformly over quantile levels. The mechanism is fully specified without the profile.
- **Extension, gated on two things:** the ≥ 5-seed share-stability check of §2.4, and the
  supervisor's answer to §8 Q7 on whether a profile fitted to measured leakage is fitted on a
  disjoint split or reported as an oracle upper bound. Only then may `ρ` be allocated in
  proportion to the measured per-band excess.

The comparison between them is then a clean test of whether the localisation result is
*useful* rather than merely true — but it is a result to be earned, not an assumption to
build on.

**Term (c) — calibration to zero, not maximisation. This is the most important part.**
`η` must **not** be set as large as the fidelity budget allows. The attack statistic is
signed and the pre-registered test is one-sided; pushing too hard makes `d_in > d_out`, the
gap negative, and the one-sided AUC collapses toward 0 — which is a *perfect distinguisher*
for any attacker who knows a defence is deployed. Reporting only the one-sided number in
that regime would be precisely the "statistic chosen after seeing the result" failure that
`PAPER_PLAN.md` Tip 4 rules out.

So the objective is calibration. Reserve a **calibration split** of patients `C` that the
backbone never trains on. Define

    P_train = { min_k ‖r − S'_k‖ : r ∈ training windows }
    P_held  = { min_k ‖r − S'_k‖ : r ∈ C's windows }

and choose `η` (and the scale of `ρ`) to minimise a two-sample divergence between them —
the Kolmogorov–Smirnov statistic, or the W1 distance between the two nearest-neighbour
distance distributions — subject to a fidelity budget. This targets exactly the quantity
the attack reads, and targets it at **zero from both sides**.

**Separate the two things when writing this.** The *calibration objective* — `P_train`
versus `P_held`, KS or W1, choose `η` to minimise — is **part of the method**. It is what the
stage does, it stands regardless of any reporting decision, and Section IV should describe it
as design.

The *reporting change* is a **protocol amendment that is not yet approved**: the paper must
give the two-sided `|AUC − 0.5|` beside the pre-registered one-sided AUC, and that needs the
supervisor's explicit sign-off before any edited model is measured (§8 Q5). Write it as
pending.

**The dependency between them is load-bearing and should be stated.** If the amendment is
refused, the calibration objective still stands, but we lose the only diagnostic that shows
it worked — a one-sided reading cannot distinguish "calibrated to zero" from "overshot into
inversion", because both drive the one-sided AUC down. The amendment is therefore not
cosmetic.

**Term (d) — per-window margin allocation (why this is not just a global knob).**
Finding 3a: the same six patients are the most exposed across independently trained
conditions (Spearman ρ = 0.61, p = 0.001) — leakage is a stable property of individuals.
Finding 3b: training longer does not expose outliers *further* (9 of 13 at both 30 k and
100 k); it starts exposing everyone (normals 2 of 13 → 13 of 13).

The allocation that uses this: for each training window compute its **local isolation** in
quantile space — e.g. the distance to its `k`-th nearest *other* training window's quantile
vector — and scale the repulsion applied against that window by an increasing function of
it. This is computable from the training set alone, with no attack run and no membership
labels, so the evaluation does not leak into the defence. Isolated windows — the outliers'
extreme days — receive most of the margin; the crowded bulk receives almost none, so the
fidelity cost is spent where there was little density to distort in the first place.

This is the capability early stopping structurally cannot have (§5), and it is also the
source of a genuine new attack surface (§7, failure mode 3).

---

### 2.7 The mask head as a localiser — RETRACTED 2026-08-30, pending a corrected re-run

> **⛔ THE NEGATIVE CONCLUSION BELOW DOES NOT STAND. Do not cite it, do not put it in a
> figure, and do not act on it.** An adversarial code review on 2026-08-30 found four
> inference-time departures from how the impute head was trained, all of which degrade the
> imputation and therefore all of which attenuate any true agreement, plus a transform
> artefact whose rank signal is *larger than the headline number*. The measurement is being
> redone. What the section says about the *proposal* and the *prediction* is unaffected and
> is kept; only the verdict is withdrawn.
>
> **Why it does not stand, in order of weight.**
>
> 1. **A noisy map was compared against two exact ones and the attenuation was not
>    corrected.** `raw` and `quantile` are deterministic functions of the data. The
>    surprisal map is a mean of ~6.4 squared errors per cell (16 rounds × mean mask ratio
>    0.40), whose reliability is estimated at **0.24–0.63**. Correlation between a noisy
>    and an exact variable is attenuated by √reliability. Disattenuated, surprisal–RAW is
>    **0.09–0.16 against a RAW–QUANTILE reference of 0.071**, and surprisal–QUANTILE moves
>    from +0.030 to **0.04–0.07** — into the reference band rather than below it. The
>    sentence "all three maps are mutually near-independent" is therefore false as
>    measured: surprisal agrees with RAW *more* than the two model-free coordinates agree
>    with each other.
> 2. **Observed cells were integrated rather than clamped, and their velocity is
>    unsupervised.** `recon(vf_i, vti, M_tgt)` masks the impute loss with the *masked*
>    cells only, so the net's output at observed positions is constrained by nothing.
>    Integrating 200 ODE steps over all tokens drives those positions somewhere arbitrary,
>    and full self-attention mixes that into the imputation of the masked tokens. The
>    script's docstring said "not clamped, matching training"; that is inverted — training
>    presents the true interpolant at observed cells and never integrates.
> 3. **`X_prior=None` is an input combination training never visits.** Training uses either
>    `(M=0, Xc=0, Xp=None)` or `(M≠0, Xc=M·I, Xp=(1−M)·prior)`. Passing `None` with a
>    non-zero mask removes a term a 30k-step-trained `prior_proj` leans on and substitutes
>    0, which in `[−1,1]` is mid-range rather than neutral. The docstring's reason for
>    omitting it — that a prior would leak the answer — is wrong about what the training
>    prior is: it is a one-step denoise of a *noised* image at random `t`, not the sample's
>    own values, and the in-distribution analogue costs one extra forward pass per step.
> 4. **The ODE endpoint was i.i.d. image noise, not `encode(time-domain noise)`.** Both
>    `draw` and training build it the second way. On the delay view with `tau < m` that
>    makes the two image copies of a time cell carry *identical* noise; i.i.d. image noise
>    does not, and the redundancy is exactly what attention can exploit.
> 5. **A transform artefact outranks the headline.** At `tau=4, m=8`, 280 of 288 time cells
>    have two image copies and 8 have one; `decode` averages, so the boundary cells' errors
>    are not variance-halved. Measured: surprisal **0.01758** at the 8 count-1 cells against
>    **0.00753** at the 280 count-2 cells, and per-row Spearman(surprisal, copy-count)
>    = **−0.067** — more rank signal than the +0.030 the conclusion rested on. Separately,
>    training hides a 2-copy cell *completely* only ~16% of the time at ratio 0.4, so the
>    test asks for a regime the head rarely saw.
>
> **What the review did verify as correct:** the time→image mask push-forward is exact
> (`encode` is a pure gather, so a 0/1 mask stays 0/1, and every image cell has exactly one
> time pre-image); the ODE direction, update rule and mask polarity match `draw` line for
> line; EMA loading yields the EMA weights; the level→time scatter in both scripts is the
> right direction, not inverted; the top-k sign conventions select what their labels claim;
> and the per-row Spearman is the right statistic.
>
> **What the review found that points the SAME way as the retracted conclusion, and is
> better evidence than what was reported:** surprisal tracks roughness — per-row Spearman
> with the window's absolute second difference is **+0.117**, larger than either map
> agreement — and is negatively related to level deviation at **−0.156**. That is precisely
> the "it localises unpredictability, not leakage" story the prediction was about. The
> corrected re-run should report these two diagnostics as the primary evidence rather than
> the map agreement alone.
>
> **Required before this section may be un-retracted:** split-half reliability from
> odd/even rounds, quoted alongside every agreement number; the clamp; the in-distribution
> prior; `encode`d noise; the 8 boundary cells dropped or stratified. All of it is one
> re-run of `scripts/pbs/M4_localise.pbs` and no new training.

### 2.7 The mask head as a localiser — the proposal and the prediction

**The proposal.** §2.6d needs a per-window margin allocation, and the version of it that
would make the defence intrinsic rather than modular is to read the allocation off MAVEN
itself. `AEMaskVelocityNet.forward(x, t, M_cond, X_cond, X_prior)` is mask-conditioned and
sampling has never used that interface — `cgm_train_sample.draw` passes `M_cond = 0`. Mask a
cell, re-impute it from the rest of the window, and take the error as a per-cell score:

    r[t,c] = mean over rounds of ( x_gen[t,c] − impute(x_gen | context)[t,c] )²

A low-`r` cell follows from its own context, so reproducing it is structure rather than
memory; a high-`r` cell does not follow from context, yet the model generated it anyway,
which is where memorised specifics would have to live. The attraction was that this
references no attack, no training window and no distance formula, so it escapes the
objection that selecting cells by the frozen statistic's own per-cell contribution is
tuning the defence to one attack — and DiM-TS, having no imputation head, structurally
cannot do it.

**The prediction, written down before the measurement** (`scripts/maven_surprisal.py`
docstring): §2.3 measured that the leak lives in level and scale, and level is exactly what
a masked autoencoder predicts *best* from context. So surprisal should track
unpredictability rather than leakage, and the honest prior was negative.

**The measurement.** `scripts/pbs/M4_localise.pbs`, 1,024 released samples from the pilot,
16 masking rounds at 200 denoising steps, view A only. Agreement with the two model-free
maps of `scripts/localise_cells.py`, as top-k Jaccard with the lift over chance:

| | top 10% | top 20% | top 30% | Spearman |
|---|---|---|---|---|
| surprisal vs QUANTILE | 1.50× | 1.21× | 1.09× | **+0.030** |
| surprisal vs RAW | 1.81× | 1.38× | 1.20× | +0.073 |
| *(reference)* RAW vs QUANTILE | 2.10× | 1.47× | 1.22× | +0.071 |

**The mask head's surprisal is uncorrelated with the quantile coordinate — the one in which
§2.3 measured the arms separating at AUC 0.828–0.905.** It agrees with the raw coordinate
about as weakly as the two coordinates agree with each other, i.e. all three maps are
mutually near-independent. The prediction holds and the localiser is dead in this form.

**What this does and does not kill.**
- **Dead:** mask-head *surprisal* as the localiser. This is independent of the backbone's
  quality — a generator that models the distribution poorly still has a trained imputation
  head, and the question asked was about that head, not about the samples.
- **Not tested:** mask-and-re-impute as the *editing* mechanism, with the localisation
  supplied from elsewhere. The quantile map is model-free, it has the leverage (§2.7a), and
  nothing measured here bears on whether re-imputation is a good way to act on it.
- **Not tested:** a genuine membership discriminator as the localiser — trained on training
  windows against held-out real windows, of which ~800 subjects' worth are unused. That is a
  different signal from surprisal and remains open. It carries its own hazard: the defence
  would then chase a learned attack, and must be reported against all five transforms of
  `results/matrix/localise/*/per_transform.json`, not just the one it was fitted to.

### 2.7a The leverage is real, and the two coordinates are two different attacks

Two by-products of the same job, both on released sets rather than the training windows of
`results/matrix/ceiling/`, and both stable across MAVEN's and DiM-TS's releases.

**Leverage — a cell-selective edit is worth building.** Share of the squared residual to the
nearest training window carried by the top cells:

| | top 5% | top 10% | top 20% | top 30% |
|---|---|---|---|---|
| RAW | 26.5–27.0% | 43.5–44.0% | 64.7–65.2% | 77.7–78.2% |
| QUANTILE | 31.5–33.1% | 48.0–49.5% | 67.6–68.5% | 79.4–79.8% |

A fifth of the cells carries two thirds of the squared distance, i.e. ~82% of the norm.
Restricting a directed edit to them keeps most of its rate at a fifth of the footprint.
This is the assumption §2.6d rests on and it now has a number.

**But the coordinates disagree about WHICH TRAINING WINDOW, not merely which cells.**

    same nearest training window in raw and in quantile space:
        MAVEN release   0.7% of released samples
        DiM-TS release  1.0%

Ninety-nine times out of a hundred, a released sample's nearest training window in raw space
and its nearest in quantile space are **different windows**. Taken with the cell-level
disagreement measured on training windows (top-20% Jaccard 0.153–0.197, a lift of only
1.38–1.77×, against `copy_paste`'s 0.601 at 5.41× — where memorisation is verbatim the two
spaces agree, which is what validates the measurement), the conclusion is stronger than
"two views of one leak":

> **The raw-space attack and the quantile-space attack are different attacks. They do not
> agree on which training records are exposed.** A defence built in one coordinate leaves
> the other's attack substantially intact, and §2.3's choice of the quantile coordinate —
> made because the arms separate better there — buys separation at the price of not
> covering the coordinate the pre-registered statistic actually reads.

This is the sharpest open problem in §2 and it was not anticipated anywhere above.

---

## 3. Where it goes: inside `FusionEditor`, or after it?

MAVEN already has a post-hoc editing stage, so the question is not where to add one.
Three options; I argue for the third.

**Read this section together with §1.9.5.** At C = 1 we recommend replacing editor fusion
with gate fusion outright, which makes Option 1 moot there and leaves QuantileShift as the
only editing stage. This section is therefore load-bearing mainly at C = 2, and as the
argument for *why* the fusion editor is the wrong host for a privacy objective in general.

### Option 1 — a privacy loss term inside `FusionEditor`

The retrieval index and neighbour tensors are already in the loop
(`nbrs = real_t[nbr_idx_t[idx]]`), so a hinge like
`L_priv = mean_i max(0, δ − ‖fused_i − nbr_i‖)` is about ten lines. **Reject it**, for four
reasons, of which the first two are structural and not tunable:

1. **The editor's residual is the wrong shape for this edit.** It is rank-`r` (r = 8) in a
   shared channel basis `U` and Gaussian-smoothed over time with `σ = 3`. Both are
   deliberate fidelity choices and both are the opposite of what a quantile edit needs: our
   leak concentrates in the top and bottom 8 % of the value distribution, which in the time
   domain is a *sparse, spiky* set of timepoints, and a σ = 3 low-pass filter exists
   precisely to remove those. Worse, at **D = 1** — `d1_c1` and `d7_c1`, our headline cells
   — `U ∈ R^{1×8}` collapses to a scalar times a rank-1 map, the entire channel-coupling
   apparatus is vacuous, and the editor degenerates to a smooth scalar field times
   `x(1−x)`.
2. **The bounded-skew gate switches the editor off exactly where the leak is.**
   `bound = x(1−x)` vanishes at the `[0,1]` walls. On our cohort mapped through `(x+1)/2`,
   the gate is **0.2495 at the median, 0.149 at the 99th percentile, and 0.0496 at the
   maximum** — a 5× attenuation at the top of the value range, the band carrying 27–29 % of
   the leakage excess. Under MAVEN's own MinMax normalisation, where the extremes land
   exactly on 0 and 1, the gate is **exactly zero** there. This is not a hyperparameter to
   retune; suppressing edits at the extremes is the gate's purpose.
3. **The objectives are in direct opposition, and the retrieval makes it worse.** The
   editor is conditioned on the mean, std and median of the K = 16 nearest **real training
   windows**, and trained with MMD-to-real, corr-to-real and a TSTR term evaluated on real,
   with the anchor to `x_base` annealed 1.0 → 0.05. It is, by construction, *a device for
   moving generated samples toward retrieved training windows.* A repulsion term asks one
   rank-8 residual field to simultaneously approach and avoid the same retrieved set. The α
   schedule is explicitly tuned to sit in the "small-α C-FID basin"; there is no reason the
   fidelity optimum and the privacy optimum share an α.
4. **It is not tunable post-hoc.** Every operating point on the risk–quality curve would
   need a re-run of the editor's 50 epochs. A stage whose knob is a scalar applied at
   release time is the direct answer to the requirement that the cost be measurable and
   tunable.

### Option 2 — a second pass after fusion

`QuantileShift(fused, η) → released`. It inherits none of the fusion editor's inductive
biases; `η` applies at inference with no training; the fidelity cost is the difference
between two runs of an *identical* measurement on `fused` and on `released`; and it
composes with any backbone, so the same stage attaches to DiM-TS or IG-FM unchanged if
MAVEN fails the quality gate. Against: no gradients through the generator, and it must be
shown not to undo the fusion editor's gains — which is measurable.

### Option 3 — recommended: the second pass, **plus one change inside `FusionEditor` that is not a loss**

**Change the retrieval corpus.** Refit TS2Vec and build the KNN index on a **disjoint
reference split of patients that the backbone never trained on**, instead of on the
training set. The editor still receives a real-data manifold signal — which is all the
fidelity terms need — but it can no longer copy the statistics of a *training* window into a
generated sample. Operationally this is a change of which array is passed as `--real` for
retrieval; at inference it costs nothing.

**[prior]** Our measurements do not test this: no experiment in this repo isolates the
retrieval corpus. It is a structural argument, and it is cheap to check — run the frozen
attack on {fused with training retrieval} vs {fused with reference-split retrieval} vs {the
unfused base branch}, everything else fixed. My expectation is that the current fusion stage
*increases* the gap. If it does, that is a result worth its own paragraph — a
fidelity-motivated editing stage that costs privacy — and it strengthens the paper's framing
considerably. If it decreases the gap, the "editing for privacy" framing needs rethinking
and we want to know that in week one, not week ten.

**So: the privacy objective goes in a second pass after fusion; the change inside
`FusionEditor` is the retrieval corpus, not a loss term.**

---

## 4. What it costs in fidelity, and how the cost is measured

**The cost is a one-parameter family and it is traced from one trained model.** The stage
is post-hoc and deterministic given `η`, so applying it at a grid of `η` to a single
released set gives the entire risk–fidelity curve at essentially zero GPU cost. This
matters concretely: `PAPER_PLAN.md` prices the Step-2 frontier at ~11 000 GPU-hours because
training trajectories have to be materialised; the privacy axis adds nothing to that.

**Both suites, and they answer different questions.**

- `TSGen_REVISE/Eval_Assembly/eval_woMissing.py` — `context_fid`, `cross_correlation`,
  `discriminative`, `predictive`, `w1_marginal`, `acf_loss`, `w1_peak_amp`, `w1_ipi`.
  This is MAVEN's own protocol and it must be run so our numbers are commensurable with the
  ones in `DESIGN.md`.
- `vendor/tsgen_metrics/` and `src/cgmoutlier/quality.py` — the definitions every published
  number in *this* project was computed with, including the univariate redefinition of
  `predictive` that our C = 1 cells require (`quality.py` documents why the TimeGAN-lineage
  metrics are undefined at C = 1). Cross-suite numbers are not interchangeable and the paper
  must not put them in one table.

**Read them at the right resolution, and predict the directions in advance.** This turns a
table into a test:

| metric | predicted direction | what it means if it moves the other way |
|---|---|---|
| `w1_marginal` | **rises slightly** — the edit acts on the value marginal | if flat, the edit is doing nothing |
| `acf_loss` | **flat** — rank preservation leaves autocorrelation nearly intact | if it rises, monotonicity is being violated |
| `w1_ipi` | **flat** — structurally invariant under a monotone value map | if it moves, PAVA is firing; the edit is not rank-preserving in practice |
| `w1_peak_amp` | small rise, via the monotone map only | a large rise means the tails are being over-edited |
| `cross_correlation` | small rise at C = 2, none at C = 1 | see §7 failure mode 5 |
| `context_fid` | the gate | — |

**Discriminative** must be read as `docs/PITFALLS.md` §16/§18 require: **max over ≥ 8
restarts with the spread**, at a fixed sampling budget, using `scripts/disc_stability.py`,
for MAVEN and MAVEN+QuantileShift alike. Do not gate any decision on a single fit or on a
3-iteration mean, and never compare across sampling budgets with it.

**Clinical metrics are a gate, not a supplement.** `src/cgmoutlier/clinical/cgm_metrics.py`
after inverting the cohort's `denorm_formula` (`mgdl = x · zclip · sd + mean`; for `d1_c1`,
`zclip = 5.0`, `sd = 57.65`, `mean = 145.94`). An edit to the value distribution is exactly
the kind of change that can shift time-in-range while leaving Context-FID flat, and TIR is
what a clinical reviewer will ask about first. **TIR, MAGE and GRI must be reported at
every `η`.**

**Quality remains a gate, not a second metric** (`PAPER_PLAN.md` Tip 5): fix a Context-FID
budget from the *unedited* MAVEN release, report the largest `η` inside it, and read risk
only at that `η`.

**How large will the cost be? I do not know, and I will not guess.** What is bounded is the
perturbation norm, which is what the distortion scales with: erasing a gap of
`g = 0.0024–0.0067` needs a directed displacement of the same magnitude, i.e. 2.5–7 % of the
nearest-neighbour distance and roughly 1–3 % of the data's own per-cell standard deviation.
That is small in aggregate. The risk is not the mean magnitude but its concentration in the
quantile tails, where TS2Vec's embedding may or may not be sensitive. That is a
measurement, not an argument, and it is the first number to produce once MAVEN samples.

---

## 5. Why this has to beat early stopping, and why it can

`PAPER_PLAN.md` §3b makes this the comparison that matters: early stopping halves the
number of patients at risk, it is free, and a reviewer will ask.

1. **Early stopping's floor is set by the quality gate, and the floor is not low.** At
   30 000 steps — the earliest checkpoint above convergence; the 20 000-step row is below it
   and excluded from the trend — **11 of 26 patients still exceed AUC 0.55 and the most
   exposed reaches 0.747**. There is nothing below that to buy. QuantileShift's knob is
   *independent of training length*: it applies at 30 k and at 100 k alike, so its axis is
   orthogonal to the early-stopping axis.
2. **They compose, and that composition is the experiment.** Run the grid
   `{early-stopped, fully trained} × {edited at a grid of η, unedited}` and read every cell
   at matched Context-FID. If the edit only recovers what early stopping already gives, the
   cells coincide and the claim fails visibly. If it adds, the frontier moves. Committing to
   this grid in advance is what makes the claim falsifiable rather than assertable.
   *(Referred to elsewhere as the composition grid; it is this numbered point, not a
   subsection.)*

   **The two defences are strongest in the same regime, not in complementary ones — and a
   reviewer will find this.** §2.5 shows `sqrt(2·d₀/g)` *decreases* in `g`, so the directed
   edit's efficiency advantage is largest at the maximum-contrast checkpoint (9.0× at 30 k
   for d1_c1). That is also the checkpoint at which early stopping has already spent
   everything it has, since below 20 k the model is under-converged. The two knobs therefore
   peak together rather than covering different parts of the range.

   **One refinement, because the observation as stated conflates two quantities.**
   *Efficiency* and *headroom* move in opposite directions with `g`. At 30 k the edit is
   cheap per unit of privacy (9.0×) but there is less to remove (`g = 0.0024`); at 100 k
   there is more to remove (`g = 0.0067`) but each unit costs more (4.9×). The net is not
   obvious from the arithmetic and only the grid settles it. What makes the 30 k cell
   worth running anyway is that the headroom there is **measured, not hypothetical**: at
   30 k, 11 of 26 patients still exceed AUC 0.55 and the most exposed reaches 0.747. That
   is the risk the edit has to remove, and it is a real quantity.

   **Consequence for ordering: run the 30 k × edited cell first.** It is simultaneously the
   operating point the paper argues for (point 5 below), the point of maximum efficiency
   advantage, and the hardest test — early stopping has already taken the easy risk out. If
   the frontier does not move there, it is unlikely to move anywhere, and we would want to
   know that before the full grid is committed.
3. **Early stopping is a blunt global knob; the leak is concentrated in individuals.**
   Finding 3a: the same six patients are the most exposed across independently trained
   conditions (ρ = 0.61). Finding 3b: over 30 k → 100 k the **outlier group does not change**
   (9 of 13 → 9 of 13) while the normal group goes 2 of 13 → 13 of 13. Training length
   therefore controls the *breadth* of exposure and does almost nothing to the *depth* on
   the people who were already exposed. **Early stopping cannot help the six patients who
   matter** — and those are the population a data-protection argument is actually about.
   Per-window margin allocation (§2.6d) can, and can do so using only training-set
   geometry.
4. **Early stopping costs a retrain per operating point; the edit costs a scalar.** Tracing
   a risk–quality curve by early stopping requires the whole training trajectory
   materialised and re-sampled at each milestone — that is exactly what the
   `_checkpoint`/`resample` machinery in `base.py` exists for, and why Step 2 is projected
   at ~11 000 GPU-hours. One released set plus a grid of `η` gives the whole curve. After
   release, changing the operating point needs no model access at all.
5. **State plainly that early stopping is right in the over-trained range, and then
   early-stop the baselines too.** Our own sweep: Context-FID 0.060 / 0.061 at 20 k / 30 k
   rising to 0.083 at 100 k, with more patients at risk and less contrast — "no trade-off in
   this range, only waste". Conceding this costs nothing and buys credibility. It also fixes
   the comparison point: **MAVEN must itself be early-stopped, and the baselines must be
   compared at their own best operating point**, or we would be beating models that are
   over-trained (`PAPER_PLAN.md` Step 4, required comparison 2).

**One argument deliberately not made.** We do **not** claim a formal guarantee. The
rearrangement bound (§2.3) is valid but loose by ~5× at our measured distance ratio, so it
certifies only that no released sample lies within `δ` of a training window in raw space —
a certificate against the copy-paste regime, not against a 3 % proximity excess. Overstating
it would be the easiest way to lose a reviewer who checks.

---

## 5a. MEASURED 2026-08-29: what a release-time directed edit can and cannot do

§5 argued the edit must beat early stopping and gave reasons it could. The argument has now
been replaced by a measurement, and it needed no mechanism to make: the best directed edit
that *can* exist is applied to released sets already on disk, and the frozen attack is read
at a grid of displacements. `scripts/edit_ceiling.py`, job `scripts/pbs/N2_edit_ceiling.pbs`,
results in `results/matrix/edit_ceiling/`.

**The edit.** Every released sample is pushed away, along `(S_nn − R)` exactly as §2.5
specifies, from the training windows whose `min` selects it, by `δ` in per-cell RMS units so
that `δ` is directly comparable to `g` and to `d₁`. Samples no training window selects are
not moved at all. **Both arms are edited**, each against its own training set, because in
deployment the holder edits whatever they release and editing only the member side measures
a defence nobody could deploy. No real mechanism does better than this: RQE's barycentric
term and any re-imputation both pay extra to stay on the data manifold. **A `δ` at which
this fails is a `δ` at which nothing succeeds.**

**Harness validation.** At `δ = 0` every cell reproduces the published numbers exactly —
`d1_c1` 11 of 26 above AUC 0.55 with max 0.747 and arm AUC 0.840; `d1_c2` 9 of 26, arm AUC
0.822; `d7_c1` 18 of 26 with max 1.000 and arm AUC 0.959 — matching `PAPER_PLAN.md` §3b,
`results/matrix/sweep/subject_auc/` and §2.2. Two implementation errors were found and fixed
by this check and both are worth recording: the direction was first written as the reverse
pairing (the training window nearest to a released sample, rather than §2.5's released
sample nearest to a training window — these are not the same relation and select different
samples), and the risk statistic was first computed as the paired fraction rather than the
unpaired rank AUC the pipeline reports (`per_subject.csv` carries both; for Loop/1142 they
read 0.958 and 0.747).

**d1_c1** (`d₁` median 0.104, so `δ = 0.008` is a 7.7% displacement):

| δ | RISK: n > 0.55 | median AUC | **max AUC** | CONTRAST: arm AUC | gap ≤ 0 |
|---|---|---|---|---|---|
| 0 | 11/26 | 0.542 | **0.747** | 0.840 | 1/26 |
| 0.002 | 8/26 | 0.507 | 0.720 | 0.864 | 11/26 |
| 0.008 | **6/26** | 0.456 | **0.661** | 0.882 | 20/26 |
| 0.016 | **5/26** | 0.442 | 0.712 | 0.876 | 19/26 |

**d7_c1** (`d₁` median 0.165):

| δ | RISK: n > 0.55 | median AUC | **max AUC** | CONTRAST: arm AUC |
|---|---|---|---|---|
| 0 | 18/26 | 0.575 | **1.000** | 0.959 |
| 0.004 | 15/26 | 0.557 | **1.000** | **0.982** |
| 0.016 | 9/26 | 0.502 | **1.000** | 0.947 |

`d1_c2` behaves like `d1_c1`: 9/26 → 4/26 at `δ = 0.016`, arm AUC 0.822 → 0.740, and its max
AUC is **pinned at 0.880 across every δ up to 0.016 and rises to 0.920 at 0.032.**

### ⛔ CORRECTED 2026-08-30 — finding 1 below was wrong, and the corrected reading is a stronger negative

An adversarial review verified the harness itself bit-for-bit (see "What was verified"
below) and then found that **every risk number in the tables above is ONE-SIDED**
(`n` with `AUC > 0.55`, and `max AUC`), while the text reads them as if they were
two-sided. §2.6c's whole point is that a subject whose gap inverts is *not* safe — a
one-sided AUC collapsing toward 0 is a perfect distinguisher for an attacker who knows a
defence is deployed. The one-sided count scores those subjects as protected.

Recomputed two-sided, `|AUC − 0.5| > 0.05`, from the same shipped JSON:

| δ | d1_c1 one-sided | **d1_c1 two-sided** | d1_c2 two-sided | d7_c1 one-sided | **d7_c1 two-sided** |
|---|---|---|---|---|---|
| 0 | 11/26 | **12/26** | 10/26 | 18/26 | **18/26** |
| 0.001 | 9/26 | **10/26** | 11/26 | 17/26 | **17/26** |
| 0.004 | 7/26 | **13/26** | 13/26 | 15/26 | **19/26** |
| 0.016 | 5/26 | **19/26** | 13/26 | 9/26 | **17/26** |
| 0.032 | 7/26 | **20/26** | 14/26 | 11/26 | **15/26** |

And the most-exposed subject, two-sided:

| cell | unedited | worst over δ |
|---|---|---|
| d1_c1 | 0.747 | **0.760** at δ = 0.016 — *worse than unedited* |
| d1_c2 | 0.880 | **0.920** at δ = 0.032 — *worse than unedited* |
| d7_c1 | 1.000 | 1.000 at every δ |

**So the corrected finding is:**

> **A global directed release-time edit does not reduce membership risk. Past δ ≈ 0.001 it
> increases the number of identifiable subjects, and it makes the most exposed subject
> more identifiable, not less.** It does not remove information; it changes the sign of
> the information. "This subject's records are systematically *farther* from the release
> than a non-member's" identifies them exactly as well as "closer".

The best two-sided operating point is δ ≈ 0.0005–0.001, where d1_c1 goes 12 → 10 and
d7_c1 18 → 17 — two subjects and one subject respectively, against early stopping's
22 → 11 on d1_c1. **The claim that the edit beats early stopping is withdrawn.**

Three further corrections from the same review, none of which rescue finding 1:

- **"No real mechanism does better than this" is false.** Normalising the *sum* of unit
  directions maximises the average cosine over the training windows selecting a sample,
  not the minimum, and a min-distance bound needs the minimum. Measured: 71.7% of training
  windows share their selected sample (mean multiplicity 1.9, max 12) and the achieved
  cosine has median **0.721**. Confirmed end to end — the exposed subject's `d_in` rises at
  **0.65·δ**, so every row is an edit at an effective ≈0.7 δ. Write it as "the best
  *global, target-agnostic* directed edit", never as an upper bound over all mechanisms.
- **47% of released samples are never moved, and that drives the rebound.** They carry no
  risk at δ = 0 and are therefore frozen; once the selected samples are pushed past them
  they become the new nearest neighbour. Adding a primal fallback for those rows alone
  (still target-agnostic, still deployable) takes Loop/1142 from 0.712 → **0.662** at
  δ = 0.016 and 0.819 → **0.755** at 0.032. The non-monotone rebound at large δ is
  therefore partly an artefact of the frozen rows, not a property of the data.
- **The displacement is smaller than δ.** Only ~53% of samples move, so the release's mean
  per-cell RMS distortion is ≈0.73·δ. And "δ = 0.008 is a 7.7% displacement" uses the
  *training-window* median `d₁` of 0.104; the exposed subjects the argument is about sit
  much closer (Loop/1142 at 0.0397), so for them it is a **20%** displacement.

### What the review verified as correct, and why the harness can be trusted

Per-subject AUC is bit-equal to the pipeline's `auc` column (not `paired_frac`) on every
target in all three cells; the δ = 0 row was reimplemented from scratch and reproduces
arm AUC 0.8402366864, 11/26, max 0.747 and both median gaps; `np.add.at` accumulates
duplicates correctly and unselected samples are exactly zero; the unit/reshape chain makes
δ exactly the per-cell RMS; `_match` runs after the edit with the same rng consumption as
`gap_for_pair`, and one target's full δ-grid was reproduced bit-for-bit; `nn_ceiling`'s
level→time scatter and its 1st/2nd-nearest search were both checked against brute force.

### The original four findings, with finding 1 withdrawn

**1. §5's central claim is confirmed. The edit's axis is orthogonal to early stopping's, and
it is measured now rather than argued.** Every row above is at 30k — the checkpoint where
early stopping has already spent everything it has (§5.1) — and the edit halves the risk
count again from there, 11 → 5 on `d1_c1` and 18 → 9 on `d7_c1`. `PAPER_PLAN.md` Step 4's
required comparison 1 is satisfied.

**2. A global knob overshoots, early and asymmetrically, exactly as §2.6c warned.** By
`δ = 0.002` on `d1_c1`, 11 of 26 gaps have already gone non-positive; by `δ = 0.008`, 20 of
26. The arms start at different magnitudes — median outlier gap 0.00240 against median
control gap 0.00077, a 3× difference — so the control arm crosses zero first and keeps
going. **The one-sided count is therefore misleading in exactly the regime that looks best:**
`δ = 0.016` reads a flattering 5/26 while the median subject sits at AUC 0.442, i.e.
inverted. On the two-sided reading `|AUC − 0.5|` the calibrated point is `δ ≈ 0.002–0.004`,
where the risk count is 7–8/26 rather than 5/26. **§8 Q5's two-sided reporting amendment is
not cosmetic; without it this table would have been read as a larger success than it is.**

**3. THE BINDING FAILURE: a global edit cannot touch the most exposed patients.**

| cell | max AUC, unedited | max AUC, best δ |
|---|---|---|
| d1_c1 | 0.747 | 0.661 |
| d1_c2 | 0.880 | **0.880** (and 0.920 at δ = 0.032) |
| d7_c1 | 1.000 | **1.000 at all eight δ** |

§5.3 argued that "early stopping cannot help the six patients who matter — per-window margin
allocation can". The first half stands. The second half is now known to require the
allocation to be *actually* per-window: **a global directed edit, the most favourable of its
class, cannot help them either.** On `d7_c1` the most exposed subject is at AUC 1.000 at
every displacement tested, including one that is 19% of the nearest-neighbour distance. The
population a data-protection argument is about is precisely this one, so §2.6d stops being
an efficiency refinement and becomes the whole mechanism.

**4. CONTRAST rises under the defence, on all three cells.** 0.840 → 0.882, 0.822 → 0.834,
0.959 → **0.982** in the useful `δ` range. This is the same effect as finding 2 seen through
the arm comparison: driving the control arm's small gaps negative while the outlier arm's
survive makes the *rank* separation larger. **A paper that reports arm AUC as its privacy
number would show this defence making matters worse.** §3b already established that RISK and
CONTRAST move in opposite directions with training length; they now do so under defence as
well, and Step 4 must say which of the two it is claiming to reduce before it reports either.

---

## 6. Plugging MAVEN into `src/cgmoutlier/generators/base.py`

MAVEN is 2–4 per-view training runs, plus a fusion editor, plus (now) QuantileShift.
`GeneratorBase` is one object with `fit`/`sample`/`save`/`load` and the inherited
`_checkpoint`/`resample`. The mapping:

`class MavenGenerator(GeneratorBase)` in `src/cgmoutlier/generators/maven.py`, registered as
`"maven"` in `registry.get` and added to `registry.ALL`; upstream copied verbatim to
`vendor/MAVEN/` with a `PROVENANCE.md`. Follow `igfm.py`'s discipline, not `dimts.py`'s:
**import the vendored module and own the training loop in the adapter**, rather than
shelling out to a CLI. `dimts.py` uses `subprocess` and is consequently the only generator
that had to override `resample`; an in-process loop lets `_checkpoint` fire from inside
training, which is the whole point of that hook.

### 6.1 `fit(X, train_cfg)`

1. `X = self._check_X(X)` — `(N, T, C)`, already z-clipped by the cohort builder.
   **Do not call `load_dataset_csv`.** Same reason `igfm.py` refuses upstream's loader: it
   MinMax-scales a concatenated CSV and cuts **stride-1** windows, and over our 506
   patients' non-overlapping gap-free days that would manufacture windows spanning day and
   subject boundaries. Feeding our windows directly is also what keeps the comparison fair —
   every baseline receives the identical array.
2. **Fix the value convention once and record it.** The fusion editor is written for `[0,1]`
   with a `clamp`. Our `d1_c1` cohort spans `[-0.371, 0.895]`, so `(x+1)/2` lands in
   `[0.315, 0.948]` and never touches the clamp. Record the map in the saved metadata so
   `sample` inverts it exactly. **Do not re-MinMax**: that would be a different
   normalisation from every published number in the project, and it would put the extremes
   exactly on the clamp walls where the bounded-skew gate is zero (§3, reason 2).
3. `build_transforms(X, T, tau, m, branch_b='stft', n_fft, hop)` per view.
   `STFTEmbedder.fit(X)` caches per-frequency-bin min/max **from the training set** — this
   is training-set-derived state that ends up inside the released model; it must be saved,
   and it is a small side channel that belongs in the limitations.
4. Train the views (§1.4) **in the configuration of §1.9** — at C = 1 that is a
   constant-width `VelocityNet` with no impute pass, two views, and a day-fold gather view
   at T = 2016. Upstream trains A and B in one process; more views mean a loop over view
   pairs. Call `self._checkpoint(root, milestone)` from inside the iteration loop at
   `--milestone_every`.
5. Combine the views. At C = 1 this is the learned scalar gate (§1.9.5) and costs nothing.
   At C = 2, train the fusion editor on the frozen per-view samples **with the retrieval
   index built on the reference split** (§3, Option 3).
6. Fit QuantileShift's state: the training quantile index `np.sort(X, axis=1)` (7 MB for
   `d1_c1` at 6072×288×1 float32; 49 MB for `d7_c1` — trivial either way), the per-level
   recalibration constants, and the per-window isolation scores.
7. `self._fitted = True`.

### 6.2 `_checkpoint` / milestones — inherited, but with one trap

`base._checkpoint` routes through `self.save`, which is what we want. But a MAVEN milestone
would otherwise write four velocity nets **plus** the fusion editor **plus** a duplicate
copy of the quantile index and the STFT statistics, at 5 milestones per run × 27 runs per
cell × 4 cells.

**Recommendation, needing no change to `base.py`:** `save` inspects whether its target
directory is named `milestone-*`; if so it writes the weights plus a small `editor_ref.json`
pointing at the run root's single copy of the QuantileShift state and the transform
statistics, and `load` follows the pointer.

**A real design question, for the supervisor (§8, Q4).** This applies only where editor
fusion is used, i.e. C = 2 under §1.9.5; with gate fusion the question does not arise, which
is a further argument for the gate at C = 1. The fusion editor is trained
*after* the views, so a mid-training milestone has views but no matching editor. Either
(a) a milestone stores views only and `resample` trains the editor at load time — correct,
but 50 editor epochs per milestone — or (b) a milestone stores views plus the *final*
editor — cheap, but the editor is then out of step with the views it edits, which is a
confound sitting inside exactly the trajectory measurement the milestones exist to produce.
I recommend (a) with a reduced editor epoch count for milestones, and that the reduction be
reported rather than buried.

### 6.3 `sample(n, sample_cfg)`

1. Per view, integrate the FM ODE unconditionally from **shared** time-domain noise;
   `sampling_steps` from `sample_cfg`, default 200. **Fix this budget across every
   condition** — `PITFALLS.md` §18: a single-fit neural metric inverts its direction across
   sampling budgets.
2. Decode each view, map to `[0,1]`, run the fusion editor → `fused`.
3. `if sample_cfg.get("edit", True): fused = quantile_shift(fused, eta=..., rho=..., m=...)`.
4. Invert the value map; return `float32 (n, T, C)`.

`edit=False` recovers the unedited MAVEN release, so **the ablation is a config flag and
both arms of every comparison come from one training run and one noise draw.** Everything
downstream — `attack/statistic.py`, `quality.py`, the LOO orchestration in
`src/cgmoutlier/loo/` — is unchanged.

### 6.4 `save` / `load` / `resample`

- `save(path)`: `maven.pt` holding per-view state dicts and EMA, the fusion editor state
  dict, the transform parameters **including the STFT per-bin min/max**, the value-convention
  record, `params`, `T`, `C`, `seed`; plus `qshift.npz` (or the pointer of §6.2).
- `load(path)`: rebuild the transforms from the **saved** parameters, never from
  `self.params` — the same discipline `dimts.py` applies to architecture, and for the same
  reason: a mismatch must not surface as an opaque shape error.
- `resample(n, from_dir, milestone, sample_cfg)`: inherited unchanged, and it becomes the
  workhorse. **`resample` at a fixed milestone over a grid of `eta` gives the whole
  privacy–fidelity curve at zero training cost; over milestones × eta it gives the 2-D grid
  the frontier figure needs.**

### 6.5 Two pieces of harness work that are not free

- **`K`, the number of released samples.** `dimts.py` defaults `K` to the training-set size,
  on the argument that releasing fewer samples than the model was trained on lowers measured
  leakage for reasons unrelated to the model. MAVEN's `--num_samples` also defaults to `N`,
  so this matches — but it must be set explicitly and recorded.
- **Order of operations against `match_k`.** The attack's `_match` subsamples both released
  sets to a common size before any distance is computed. QuantileShift must be applied
  **before** that subsampling, and **the same `η` must be applied to the member release and
  the base release**. If the base were unedited the comparison would be meaningless.

---

## 7. Failure modes, and the measurement that detects each

Ordered by how much they would cost if discovered late.

1. **Second-nearest-neighbour saturation — RUN 2026-08-29. The answer is worse than
   saturation and it is in §5a.** Measured twice, because the first measurement was
   pessimistic and aggregated wrongly.
   *First, the bound.* `scripts/nn_ceiling.py`, `results/matrix/ceiling/`: the per-window
   spacing `d₂ − d₁` against the gap `g` it must exceed — 1.81× (d1_c1), 0.89× (d1_c2),
   0.51× (d7_c1) at the median, with 32% / 54% / 75% of windows below `g`. **Do not quote
   these.** They cap a *per-pair* edit that moves only the nearest released sample, and
   they compare a per-window quantity against a per-subject `g`.
   *Then, the measurement.* `scripts/edit_ceiling.py`, `results/matrix/edit_ceiling/`:
   apply the best directed edit that can exist — every released sample pushed away, along
   `(S_nn − R)` as §2.5 specifies, from the training windows whose `min` selects it — to
   BOTH arms, and read the frozen attack's own output at a grid of displacements. At
   `δ = 0` it reproduces every published number exactly (d1_c1 11 of 26 above AUC 0.55,
   max 0.747, arm AUC 0.840; d7_c1 18 of 26, max 1.000, arm AUC 0.959), which is what
   validates the harness.
   **Saturation is not the binding failure. §5a is: the edit works on the median and
   cannot touch the maximum.**
   *Second prerequisite, same category:* re-derive §2.4's band decomposition under the
   pipeline's own `_match` across ≥ 5 subsample seeds and report the per-band spread of the
   shares. Also no GPU, also on existing files. Until it passes, `ρ(u)` stays uniform (§2.6b)
   and the localisation profile is a motivation rather than a load-bearing measurement.
2. **There is nothing to defend.** MAVEN may simply not memorise on this cohort, in which
   case the unedited gap is ~0 and there is no curve to move.
   *Detect:* run Step 1 on unedited MAVEN before building any of this. Note the flip side —
   an unedited MAVEN with a *higher* gap than DiM-TS at equal quality is also a result, and
   makes the edit more necessary, not less.
3. **The quality gate fails outright — FIRED 2026-08-29, twice. See §1.8a.** Predicted at
   2.7–3.4× from `DESIGN.md`'s own plateau; measured at **3.7× in the default configuration
   and 4.5× after applying MAVEN's own ablation ladder**, which made it worse rather than
   better. The ladder result falsifies upstream's stated hypothesis about the cause of its
   plateau. `D = 1` makes it likelier and the rest of this item still stands. Worse, **at D = 1 the channel-masking auxiliary task
   is degenerate** — §1.9.1 gives the arithmetic — so MAVEN-MaskAE's headline mechanism does
   not exist in our two single-channel headline cells at all. §1.9 specifies the
   configuration changes this forces, and §1.9.2 adopts the `DESIGN.md` ablation ladder
   (`--coarse_weight 0`, constant-width `--dims`) as the CGM default rather than as an open
   experiment. **This is significant and the supervisor should see it before any compute is
   committed** (§8 Q1, §1.9.7).
4. **Sign inversion / over-protection.** `η` too large ⇒ `d_in > d_out` ⇒ one-sided AUC
   collapses toward 0 while `|AUC − 0.5|` stays high — a perfect distinguisher for an
   attacker who knows a defence is deployed.
   *Detect:* report two-sided `|AUC − 0.5|` and the signed median gap per arm at every `η`;
   plot the calibration divergence (`P_train` vs `P_held`) against `η` and show the crossing.
5. **Hole-detection attack.** Per-window margin allocation (§2.6d) creates a low-density
   region around each protected training window; an adaptive attacker tests for the *hole*
   rather than for proximity.
   *Detect:* a second, explicitly adaptive statistic — the count of released samples within
   radius `r` of the target's windows, for `r` on a grid, member vs base — reported
   alongside the frozen one. The frozen statistic stays pre-registered and is never replaced;
   the hole statistic is declared as adaptive and reported separately.
6. **The edit stops being rank-preserving.** If PAVA fires often, monotonicity was violated
   and timing structure *does* move, which is the one thing the design promises it will not.
   *Detect:* log the PAVA activation rate per `η`; `acf_loss` and `w1_ipi` should be flat
   and are the alarms.
7. **Cross-channel damage at C = 2.** Per-channel quantile editing preserves Spearman
   cross-correlation but not Pearson, and `cross_correlation` in both suites is a Pearson
   lag-0 metric.
   *Detect:* `cross_correlation` (Eval_Assembly) and `corr_loss`. If it moves, couple the
   channels — one shared Dirichlet weight vector `w` and one shared repulsion scale per
   window, rather than independent draws per channel.
8. **The fusion editor's retrieval is itself a leak** (§3, Option 3).
   *Detect:* the three-way comparison — training retrieval vs reference-split retrieval vs
   unfused base branch, everything else fixed.
9. **Discriminative-score self-deception, in both directions.** MAVEN's claimed DS gain and
   our edit's DS effect are equally exposed.
   *Detect:* `scripts/disc_stability.py`, ≥ 8 restarts, report max and spread (§1.8).
10. **Fidelity distortion where TS2Vec cannot see it.** The edit is concentrated in the
    quantile tails; Context-FID may be blind there while a clinician is not.
    *Detect:* TIR / MAGE / GRI at every `η` (§4), treated as a gate.

---

## 8. Open questions for the supervisor

1. **[CLOSED 2026-08-28 — see §1.9.1. Kept for the record; do not act on it.]** ~~D = 1 is a problem for MAVEN's own mechanisms — the first decision to make.~~ Our
   headline cells `d1_c1` and `d7_c1` are single-channel. At D = 1 the masked-imputation task
   is degenerate, `fusion_editor`'s rank-8 shared channel basis `U` collapses, and
   `corr_loss` and `cross_correlation` are vacuous (§1.9.1, §1.9.5). §1.9 answers this by
   configuration — at C = 1 the model becomes MAVEN-base with a retuned spectral view, a
   day-fold view at seven days, and gate fusion — but that means **MAVEN-MaskAE's headline
   mechanism is switched off in the headline cells**. Options: (a) promote the two-channel
   cells `d1_c2` / `d7_c2` to headline so MAVEN's mechanisms are exercised — note `d1_c2` is
   also the only cell where outliers clearly stand out (arm AUC 0.698 vs 0.562); (b) accept
   §1.9's C = 1 configuration and present the model as MAVEN-base plus our editing stage;
   (c) `--mask_mode cell` at D = 1, which is a *temporal* inpainting task rather than an
   inter-channel one and needs its own justification. **This decision changes what §1 of the
   Method describes and should be made before any compute is committed** (§1.9.7).
2. **Which retrieval corpus, and which losses use which split.** I recommend a disjoint
   reference split for the KNN index, with MMD / corr / TSTR still computed against the
   training set (so the fidelity numbers stay comparable). That costs patients from
   training — I suggest 10 % of the 506, stratified on day count as `data/cohort.py` already
   does. Does the supervisor agree, and with what split size?
3. **Which statistic the edit is calibrated against.** The frozen one is
   `min`-over-samples, `mean`-over-windows. Calibrating against only that risks a defence
   specific to it; calibrating against all six variants in `statistic.py` is stronger but
   pushes `η` up and costs more fidelity. Which?
4. **Milestone / editor coupling** (§6.2): does a milestone re-train the fusion editor at
   load time, or reuse the final one?
5. **Two-sided reporting.** I believe the paper must report `|AUC − 0.5|` beside the
   pre-registered one-sided AUC, because a calibrated defence can overshoot and a
   one-sided reading would hide it. This is a change to a pre-registered protocol and needs
   explicit sign-off **before** any edited model is measured.
6. **Transform parameters at T = 288 and T = 2016.** Upstream tuned the views for T = 64–256.
   At T = 288, `delay(τ=4, m=8)` gives `L = 8 × 71 = 568` tokens and the STFT view
   (`n_fft=16, hop=8`) gives `F = 9`, `frames ≈ 37`, `L = 666`. At T = 2016 those become
   `L = 4024` and `L = 4554`, and self-attention is O(L²). **The seven-day cells may be
   infeasible at upstream's settings.** Coarser transforms cut `L` but change the view — and
   that would confound window length with transform choice, on top of the existing
   capacity confound (`PAPER_PLAN.md` Limitation 3). Someone should compute the memory before
   compute is committed, and the supervisor should decide whether `d7` is allowed different
   transform parameters.
7. **Is `ρ(u)` fitted or fixed?** Currently **neither** — §2.6b now specifies uniform `ρ` as
   the default, because a non-uniform profile is not yet licensed by the evidence. Two things
   must happen before the question is even live: (i) the ≥ 5-seed share-stability check of
   §2.4 (no GPU, no retraining) must show the per-band shares are stable under the K-matching
   subsample; and then (ii) the supervisor must choose between fitting `ρ` on a split disjoint
   from the evaluation split, or reporting a fitted `ρ` as an oracle upper bound with the
   uniform-`ρ` result beside it. Fitting the profile to measured leakage uses the outcome to
   shape the defence, which is legitimate for a *defence* but not for a *test*.
8. **Capacity matching.** Every baseline in this project is matched to a parameter budget
   (±3 %, with IG-FM a documented +4.6 % exception). MAVEN with four views is four backbones
   plus an editor. Is the budget per view or per model? This determines whether MAVEN is
   compared as one model or as an ensemble, and a reviewer will ask.
9. **Does the reference split also serve as the calibration split** for the `P_train` vs
   `P_held` objective (§2.6c), or must those be two disjoint sets? Reusing one set is cheaper
   in patients but means the calibration is evaluated on data the editor's retrieval has
   already seen.

---

## 9. Provenance of every number in this document

| claim | source |
|---|---|
| arm AUC by transform (0.828 / 0.781 / 0.905; 0.527 / 0.657 / 0.467) | `results/matrix/localise/*/per_transform.json` via `scripts/report/transform_auc.py`; independently recomputed here |
| `d_in`, `d_out`, gap, arm AUC by milestone (§2.2) | `results/matrix/sweep/attack/{cell}_ms{n}/summary.json`, frozen variant `set_reduce=min`, `subject_reduce=mean` |
| raw vs sorted relative gaps (§2.3) — **first computation, superseded** | computed for this document: `d1_c1`, `results/runs/matrix_d1_c1/{base,include_*}/samples.npy`, cohort `data/cohort/matrix_d1_c1`, design `results/matrix/design/rep1/design.json`; frozen statistic, K-matched, but with `_match` reproduced via `np.random.RandomState` because that environment's numpy is 1.14 and has no `default_rng`. Aggregation: **ratio of medians over targets**. Reported: raw +10.9 % / +8.1 %, sorted +14.3 % / −0.5 % |
| raw vs sorted relative gaps (§2.3) — **authoritative** | `scripts/report/verify_sorted_gap.py`, log `logs/M1_verify.live.log`; the pipeline's own `_match` under numpy 1.24, so the attack's own `default_rng` subsample. Aggregation: **mean over targets of the per-target ratio**. Reported: raw +36.3 % / +11.9 %, sorted +20.9 % / +1.7 % |
| **the disagreement between those two, recorded rather than replaced** | `d_in` agrees to within 5 % in both spaces (0.0929 vs 0.0884 raw; 0.0186 vs 0.0189 sorted), so the distance computation is not the source. The relative gaps do not agree, through both the aggregation choice and the K-matching subsample. **This is a limitation the paper should state: the per-arm relative gap is sensitive to the K-matching subsample at n = 13, so the paper reports the arm ratio (~3× raw, ~12× sorted) and the rank-based arm AUC, both of which are stable across the two computations, rather than either arm's absolute percentage.** An earlier draft's claim that the control arm's quantile-space gap was "zero" was wrong — it is +1.7 % — and has been removed |
| per-quantile-band decomposition (§2.4) | the **first** computation's run; the sorted-space squared distance decomposed over quantile level once the nearest neighbour is fixed, exactly as `scripts/localise_leak.py` decomposes the raw distance over timesteps. Carries the same `RandomState` subsample caveat, and has **not** been re-derived under the pipeline RNG. The *shares* are internally normalised and are what §2.6b's `ρ(u)` uses; the absolute totals should be treated as indicative only |
| hour-of-day / 09:00 peak, 16× | `results/matrix/localise/*/per_timestep.json` (as reported in the brief; not recomputed here) |
| patients at risk, arm AUC, Context-FID vs training steps (§5) | `docs/PAPER_PLAN.md` §3b, read from `results/matrix/sweep/` |
| ρ = 0.61 patient-ranking stability | `docs/PAPER_PLAN.md` §3a |
| cohort value range and the gate values `0.2495 / 0.149 / 0.0496` | `data/cohort/matrix_d1_c1/windows.npy` percentiles mapped through `(x+1)/2`, evaluated at `x(1−x)` |
| every description of MAVEN in §1 | read directly from `TSGen_REVISE/{visual_transforms,train_maven_maskae,train_maven,fusion_editor,fusion_nview}.py`, `DESIGN.md`, `README.md` |
| the α-magnitude estimate in §2.5 | **[estimate]** from reading `FusionEditor.forward` and the initialisers; not measured |
| the claim that fusion retrieval increases the gap | **[prior]** structural argument only; the experiment is specified in §3 and §7.8 |
| §1.9.1, degeneracy of channel masking at D = 1 and D = 2 | read from `sample_image_mask` and `recon` in `train_maven_maskae.py`; the zero-loss branch follows from `mask.sum().clamp(min=1.0)` with an all-zero `M_tgt` |
| §1.9.2, the Context-FID plateau and its suspected cause | `TSGen_REVISE/DESIGN.md`, "Empirical status" and "Ablation ladder" |
| §1.9.3, `t1_max` cap and token counts (2016 vs 4016) | `visual_transforms.py` (`t1_max`, `make_period_fold`, `make_delay_embed`); `2016 = 7 × 288` from `data/cohort/matrix_d7_c1/manifest.json` (T = 288, dt = 5 min) |
| §1.9.4, the 80-minute analysis window | `n_fft = 16` at `dt_min = 5` from `README.md` and the cohort manifest; the candidate retune is **[estimate]**, not measured |
| §1.9.5, vacuity of `U` and `corr_loss` at D = 1 | `FusionEditor.__init__`/`forward` and `corr_loss` in `fusion_editor.py` |
| §1.9.5, the gate **is** fitted on real data | `train_maven.maven_step`, `else` branch: `L_fuse = F.mse_loss(g*A.detach() + (1-g)*B.detach(), x0)` at `--fuse_alpha 0.1`; `x0` is the real batch. At sampling, `train_maven.sample_maven` uses `maven.gate().detach()`, a stored constant. **An earlier draft's "no access to real data at all / privacy-neutral by construction" was wrong and has been replaced by the global-scalar vs per-sample-retrieval argument.** Found by the Section IV writing agent, verified here against the source |
| §2.5 directed-vs-isotropic factor per cell | arithmetic `sqrt(2·d₀/g)` on every row of the §2.2 table. The **norm** ratio is measured input; the squared **distortion** factor is a prediction contingent on second-order behaviour, which §2.6a engineers rather than assumes and which the η-sweep tests |

---

## 10. Revision log — what changed in each round, and where

Messaging between this session and the Section IV writing agent is one-way (`SendMessage`
cannot reach it), so this log exists to let a reader holding a stale copy find what moved.
**If your copy has no §1.9.7b, you are reading revision 2 or earlier.**

| rev | change | where |
|---|---|---|
| 1 | initial design | all |
| 2 | §2.3 re-founded on the **arm ratio** (~3× raw, ~12× sorted) after the coordinator's rerun under the pipeline RNG; the claim that the control arm's quantile-space gap was "zero" removed — it is +1.7 % | §2.3, §9 |
| 2 | aggregation convention settled: report the **mean over targets of the per-target ratio** `gap_i/d_in_i`, with the rank-based **arm AUC** as the primary number since it needs no normalisation or aggregation | §2.3 |
| 3 | §2.5 range corrected — "5–9×" withdrawn (the d7_c1 40 k row falls outside it); all eight rows tabulated, full span 3.4×–11.7×, reportable figure **6.4×–9.0× at the maximum-contrast checkpoint** | §2.5 |
| 3 | the 40–80× distortion factor restated as a **prediction** contingent on second-order behaviour, which §2.6a engineers rather than assumes, with the η-sweep named as its test | §2.5 |
| 3 | §2.4's "the shares are more subsample-stable than the totals" argument **withdrawn**; a mechanism pointing the other way recorded; ≥ 5-seed share-stability check made a prerequisite | §2.4, §7 (failure mode 1) |
| 3 | **`ρ(u) ≡ 1` uniform is now the described default**; the non-uniform profile is a gated extension | §2.6b, §8 Q7 |
| 3 | §1.9.5 gate claim **corrected** — the gate *is* fitted on real data (`L_fuse = F.mse_loss(...)`, `--fuse_alpha 0.1`); "privacy-neutral by construction" was false. Replaced by the global-scalar vs per-sample-retrieval argument, plus the statement that the pipeline is not retrieval-free — we invert retrieval's sign | §1.9.5, §1.9.8, §9 |
| 4 | **§1.9.7b added** — three commitment tiers, so §1.9's changes are not read as uniformly settled. *(Answers the writer's Q3: §1.9.3/§1.9.5/§1.9.1/§1.9.6 adopted; §1.9.4 decision adopted with the value pilot-decided; §1.9.2 an ablation ladder we commit to running, not a default we assert.)* | §1.9.7b |
| 4 | §2.6c split into **method** (the calibration objective, describe it as design) and **pending protocol amendment** (two-sided \|AUC − 0.5\| reporting), with the dependency between them stated. *(Answers Q4: the writer's reading is correct; the calibration objective is not itself pending.)* | §2.6c |
| 4 | "what moves with the C = 1 decision" list added — the writer's four movers are right; four more are coupled: the data partition, the metric table, the C = 2-only cross-channel failure mode, and the C = 2-only testability of the conditional-reconstruction prior. *(Answers Q7.)* | §1.9.7 |
| 5 | name decided: **print RQE**, `QuantileShift` retained as a deprecated alias; flagged as a preference, not a finding. *(Answers Q5.)* | §2.6 |
| 5 | the composition grid's two defences shown to **peak in the same regime**, with the efficiency-versus-headroom refinement and the recommendation to run the 30 k × edited cell first | §5, point 2 |

| 6 | **§1.9.1 remedy replaced on supervisor instruction (2026-08-28): mask cells, not channels, at every C.** `--mask_mode cell` is an existing flag; the objective becomes non-degenerate at C = 1 and subsumes the published one at C = 2. **§8 Q1 is closed** — there is no longer a headline-cell decision to take, because the mechanism runs in every cell | §1.9.1, §1.9.5, §1.9.7, §8 Q1 |
| 6 | consequential unification: **gate fusion at both channel counts**, retrieval editor demoted to a measured ablation at both, so the data partition no longer branches on C | §1.9.5, §1.9.7 |
| 6 | new risk recorded and made measurable rather than argued: masked reconstruction from 40–80 % real context **may increase** memorisation; ablated at both C and read through the frozen attack | §1.9.1, §7 |

**Still genuinely open, and not answerable from this session:** every item in §8. The two
that gate work rather than writing are **Q1** (which cells are headline, which decides
whether the model is MAVEN-base or MAVEN-MaskAE) and **Q5** (sign-off on two-sided
reporting, without which the calibration objective has no diagnostic).

---

### Round 7 — 2026-08-29/30: the first measurements, and what they cost the design

Nothing in this round was written from reading code. Six jobs ran; four of them needed no
GPU and two needed one card for under three hours.

| # | what changed | where |
|---|---|---|
| 1 | MAVEN trained on CGM for the first time. **Fails the pre-registered gate at 3.7×** against DiM-TS at the matched 30k checkpoint (Context-FID 0.2256 vs 0.0611) | §1.8a, §7 mode 3 |
| 2 | MAVEN's own ablation ladder run and **refuted** — constant width + `coarse_weight 0` moved both gate metrics the wrong way. §1.9.2's adoption withdrawn | §1.8a, §1.9.2 |
| 3 | The restart protocol caught a false positive **on our own model**: single fits read 0.0050 and 0.1100 where the max over 8 restarts reads 0.1200 | §1.8a(c), `PITFALLS.md` §16 |
| 4 | §7 failure mode 1 run. The per-pair `d₂ − d₁` bound is superseded by a direct measurement; **saturation is not the binding failure** | §5a, §7 mode 1 |
| 5 | **The binding failure is that a global directed edit cannot touch the most exposed patients** — `d7_c1` max AUC 1.000 at all eight displacements. §2.6d becomes the mechanism rather than a refinement | §5a finding 3 |
| 6 | The mask-head localiser was proposed, predicted to fail for a stated reason, and **failed for that reason** (Spearman +0.030 against the quantile map) | §2.7 |
| 7 | New and unanticipated: the raw and quantile coordinates pick the **same nearest training window 0.7–1.0% of the time**. They are two different attacks, not two views of one | §2.7a |
| 8 | New and unanticipated: fidelity is scored against the model's **own training set**, so `copy_paste` beats DiM-TS on the gate (0.070 vs 0.131–0.331). Figure 2's two axes read the same object | `PITFALLS.md` §19 |

**What survived.** §2.5's efficiency argument (the edit is cheap per unit of privacy at the
maximum-contrast checkpoint), §5's orthogonality claim (the edit halves the risk count from
a checkpoint where early stopping has nothing left), and §2.6d's leverage premise (a fifth
of the cells carries two thirds of the squared distance).

**What did not.** §1.9.2 as a default, the mask head as a localiser, `d₂ − d₁` as the
ceiling, and the assumption that a single global knob is a defence.

**What is newly required rather than optional.** Two-sided reporting (§8 Q5) — without it
§5a's table reads as a larger success than it is. Per-window margin allocation (§2.6d) —
without it the defence cannot reach the population it exists for. A held-out fidelity
reference (`PITFALLS.md` §19) — without it Figure 2 may be an artefact of its own axes.
