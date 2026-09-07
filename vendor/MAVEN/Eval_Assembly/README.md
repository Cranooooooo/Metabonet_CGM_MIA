# Multi-Channel Physiological Time-Series Generation Evaluation

A focused, non-redundant evaluation suite for **generative models on multi-channel physiological time series** (ECG, CGM, EEG, multi-lead biosignals, …). **Eight fidelity metrics**, two evaluation modes (with vs. without missingness in real data), one consistent CLI.

The suite extends the four canonical TimeGAN / Diffusion-TS metrics (Context-FID, Cross-correlation, Discriminative, Predictive) with four physiology-tailored additions covering marginal distribution, autocorrelation, peak-amplitude, and inter-peak-interval views. Together they answer eight independent questions about how well a generator reproduces the structure that physiological signals actually depend on.

> **Asymmetric assumption.** This suite assumes that **generated (fake) samples are fully observed at every (n, t, c)**, while **real (training) data may have missing values**. Only `--real_mask` is exposed on the CLI. When a metric needs both sides to share the same missingness regime (Context-FID, Discriminative, Predictive), it draws a synthetic fake-mask internally by resampling patterns from the real-mask pool — this is an implementation detail, not a user-facing concept.

---

## Table of Contents

1. [Why these eight metrics](#1-why-these-eight-metrics)
2. [Folder layout](#2-folder-layout)
3. [Quickstart](#3-quickstart)
3.5 [How to use these files (5 common workflows)](#35-how-to-use-these-files-5-common-workflows)
4. [Entry scripts](#4-entry-scripts)
5. [Metric reference (1 of 8 → 8 of 8)](#5-metric-reference)
6. [The two-mode design (wo-missing vs. w-missing)](#6-the-two-mode-design)
7. [Output schema](#7-output-schema)
8. [Multi-channel guidance](#8-multi-channel-guidance)
9. [Pitfalls and assumptions](#9-pitfalls-and-assumptions)
10. [Extending the suite](#10-extending-the-suite)

---

## 1. Why these eight metrics

Each metric answers one independent question about the generator. They were picked so that **dropping any one removes a real piece of information** — no redundancy, no padding.

| #  | Metric              | Question answered                                                            | Family       |
|----|---------------------|------------------------------------------------------------------------------|--------------|
| 1  | `context_fid`       | Does the global semantic embedding distribution match?                       | Classic (FID)|
| 2  | `cross_correlation` | Are inter-channel coupling relationships preserved? *(multi-channel core)*   | Classic      |
| 3  | `discriminative`    | Can a supervised classifier still tell real from fake?                       | Classic      |
| 4  | `predictive`        | Does a forecaster trained on synthetic generalize to real?                   | Classic (TSTR)|
| 5  | `w1_marginal`       | Are per-channel value distributions correct?                                 | Physiology   |
| 6  | `acf_loss`          | Is per-channel temporal autocorrelation / rhythm correct?                    | Physiology   |
| 7  | `w1_peak_amp`       | Are per-channel peak amplitudes correct?                                     | Physiology   |
| 8  | `w1_ipi`            | Are per-channel inter-peak intervals (rhythm at the event level) correct?    | Physiology   |

The four classics dominate the time-series-GAN/Diffusion literature. The four supplements close gaps that no classic alone can detect — *e.g.* a generator that gets Context-FID and Discriminative right can still produce blood-glucose traces with the wrong post-meal peak height, or ECG beats with a slightly off RR interval distribution. Peak-amp / IPI will catch those.

> **Why no `psd_w1`?** A Welch-PSD metric was considered and dropped. Welch requires a continuous, equally-sampled signal; under missingness, neither zero-fill nor linear interpolation gives a faithful spectrum (zero-fill injects spurious low-frequency power; interpolation kills high-frequency content). Spectral content is largely captured by `acf_loss` over multiple lags anyway, so the marginal value of an unreliable PSD metric did not justify its risk of being misleading. If you have fully-observed data and specifically need a frequency-domain check, see §10.5 for a Welch-PSD-W1 sketch.

---

## 2. Folder layout

```
.
├── README.md                  ← this file
├── eval_doc.md                ← original taxonomy notes (Chinese)
├── eval_woMissing.py          ← entry script: no-missingness pipeline
├── eval_wMissing.py           ← entry script: real-only-missingness pipeline
│
├── utils/
│   ├── mask_utils.py          ← mask helpers (sanitize, resample, pairwise-complete corr,
│   │                            per_sample_observed_indices)
│   ├── metric_utils.py        ← display_scores, train_test_divide, reduce_per_channel
│   └── ts2vec_loader.py       ← shim that registers eval_ref_1/ts2vec as Models.ts2vec.*
│
├── metrics/                   ← 16 files = 8 metrics × 2 modes
│   ├── context_fid_woMissing.py        cross_correlation_woMissing.py
│   ├── context_fid_wMissing.py         cross_correlation_wMissing.py
│   ├── discriminative_woMissing.py     predictive_woMissing.py
│   ├── discriminative_wMissing.py      predictive_wMissing.py
│   ├── w1_marginal_woMissing.py        acf_loss_woMissing.py
│   ├── w1_marginal_wMissing.py         acf_loss_wMissing.py
│   ├── w1_peak_amp_woMissing.py        w1_ipi_woMissing.py
│   └── w1_peak_amp_wMissing.py         w1_ipi_wMissing.py
│
├── eval_ref_1/                ← reference: Diffusion-TS-style 4-metric suite
└── eval_ref_2/                ← reference: ECG fidelity / TSTR / privacy suite
```

`eval_ref_1/` and `eval_ref_2/` are **kept as references**, not as runtime dependencies — except that Context-FID reuses `eval_ref_1/wMissing_scripts/ts2vec/` via `utils/ts2vec_loader.py`.

### Which file does what

Use this map to find the right place to read, edit, or extend.

| File / folder                                 | Role                                             | When to touch it |
|-----------------------------------------------|--------------------------------------------------|------------------|
| `eval_woMissing.py`                           | CLI entry — runs all 8 metrics on full data      | Day-to-day evaluation |
| `eval_wMissing.py`                            | CLI entry — runs all 8 metrics with `--real_mask`| Day-to-day evaluation under missing real data |
| `metrics/<name>_woMissing.py`                 | One metric, one function, no-missing path        | To audit / tweak the algorithm |
| `metrics/<name>_wMissing.py`                  | Same metric, mask-aware path; falls back to wo   | To change how missingness is handled |
| `utils/mask_utils.py`                         | All mask plumbing                                | Adding a new mask convention |
| `utils/metric_utils.py`                       | Score reporting, splits, per-channel reduction   | Changing reporting format |
| `utils/ts2vec_loader.py`                      | Imports the TS2Vec package from `eval_ref_1`     | Moving the ts2vec package |

**Naming convention.** `<metric>_woMissing.py` = without missingness; `<metric>_wMissing.py` = with missingness in real (only). Each pair always exposes one function — `<metric>(...)` in the wo file, `<metric>_masked(...)` in the w file. The w version always begins with `if is_fully_observed(ori_mask): return _wo_version(...)` so the two paths agree exactly when there is no missingness.

---

## 3. Quickstart

### Inputs

Every metric takes two NumPy arrays:

```python
real:  np.ndarray   # shape (N, T, C), float32 — may have missing positions
fake:  np.ndarray   # shape (N, T, C), float32 — assumed fully observed
```

For the with-missing mode, you additionally provide:

```python
real_mask: np.ndarray   # shape (N, T, C), bool — True = observed
```

There is **no** `fake_mask` argument.

### Run

```bash
# No missingness
python eval_woMissing.py \
    --real  data/real.npy  \
    --fake  data/fake.npy  \
    --fs 250               \
    --iterations 3         \
    --out_json results_wo.json

# With missingness in real
python eval_wMissing.py \
    --real      data/real.npy        \
    --real_mask data/real_mask.npy   \
    --fake      data/fake.npy        \
    --fs 250                         \
    --iterations 3                   \
    --out_json results_w.json

# Run only a subset
python eval_woMissing.py --real real.npy --fake fake.npy \
       --metrics context_fid w1_peak_amp w1_ipi
```

### Programmatic use

```python
from metrics.w1_peak_amp_woMissing import w1_peak_amp
from metrics.cross_correlation_wMissing import cross_correlation_score_masked

amp_result = w1_peak_amp(real, fake,
                         peak_kwargs={"distance": 30, "prominence": 0.1})
cc_score   = cross_correlation_score_masked(real, fake, ori_mask=real_m)
```

Each metric module exposes one top-level function with a signature matching its docstring.

---

## 3.5. How to use these files (5 common workflows)

### Workflow A. "I just generated some samples — give me one number per metric"

```bash
# 1. Save your data (one-time setup, in your generation pipeline)
import numpy as np
np.save("real.npy", real_array)   # shape (N, T, C), float32
np.save("fake.npy", fake_array)   # shape (N, T, C), float32

# 2. Evaluate
python eval_woMissing.py \
    --real real.npy --fake fake.npy \
    --fs 250 --iterations 3 \
    --out_json results.json
```

### Workflow B. "My real data has missing values"

Provide a boolean mask matching `real.shape`:

```bash
python eval_wMissing.py \
    --real      real.npy \
    --real_mask real_mask.npy \
    --fake      fake.npy \
    --fs 250 --iterations 3 \
    --out_json results.json
```

For Context-FID / Discriminative / Predictive, the script internally resamples patterns from `real_mask` (with replacement) to project the same missingness regime onto fake — so neither side benefits from a "this position is zero" cheat. You do not provide or see this synthetic mask.

If `--real_mask` is omitted (or is fully True), every metric short-circuits to the no-missing path and produces *bit-equal* results to `eval_woMissing.py` on the same inputs.

### Workflow C. "I only want a subset of the metrics"

```bash
# Fast iteration: distance metrics only (~seconds on CPU)
python eval_woMissing.py --real real.npy --fake fake.npy --fs 250 \
       --metrics w1_marginal acf_loss w1_peak_amp w1_ipi cross_correlation

# Slow / heavy: neural metrics (use GPU; ~minutes)
python eval_woMissing.py --real real.npy --fake fake.npy --fs 250 \
       --iterations 3 \
       --metrics context_fid discriminative predictive
```

### Workflow D. "I want to call one metric from my own Python code"

```python
import sys, numpy as np
sys.path.insert(0, "/path/to/Eval_Assembly")

from metrics.w1_peak_amp_woMissing import w1_peak_amp
from metrics.context_fid_woMissing import context_fid

real = np.load("real.npy")
fake = np.load("fake.npy")

amp = w1_peak_amp(real, fake,
                  peak_kwargs_per_channel=[
                      {"distance": 30, "prominence": 0.3},   # ECG
                      {"distance": 200, "prominence": 0.05}, # respiration
                  ])
print(amp["per_channel"], amp["mean"])

fid = context_fid(real, fake)
```

```python
# With real-only missingness
from metrics.w1_peak_amp_wMissing import w1_peak_amp_masked

real_m = np.load("real_mask.npy").astype(bool)

amp = w1_peak_amp_masked(real, fake,
                         ori_mask=real_m,
                         peak_kwargs={"distance": 30, "prominence": 0.3})
```

The naming is mechanical:

| File                             | Function                              |
|----------------------------------|---------------------------------------|
| `metrics/<name>_woMissing.py`    | `<name>(real, fake, **kw)`            |
| `metrics/<name>_wMissing.py`     | `<name>_masked(real, fake, ori_mask=None, **kw)` |

Exception: `discriminative` and `cross_correlation` use slightly different verb forms (`discriminative_score`, `cross_correlation_score`); see §5.

### Workflow E. "I want to compare several generators side-by-side"

```bash
for tag in baseline ddpm diffusion_ts ours; do
    python eval_woMissing.py \
        --real  data/real.npy --fake data/fake_${tag}.npy \
        --fs 250 --iterations 3 \
        --out_json results_${tag}.json
done
```

```python
import json, pandas as pd
rows = []
for tag in ["baseline", "ddpm", "diffusion_ts", "ours"]:
    with open(f"results_{tag}.json") as f:
        r = json.load(f)
    rows.append({
        "model":             tag,
        "context_fid":       r["context_fid"]["mean"],
        "cross_correlation": r["cross_correlation"]["mean"],
        "discriminative":    r["discriminative"]["mean"],
        "predictive":        r["predictive"]["mean"],
        "w1_marginal":       r["w1_marginal"]["mean"],
        "acf_loss":          r["acf_loss"]["mean"],
        "w1_peak_amp":       r["w1_peak_amp"]["mean"],
        "w1_ipi":            r["w1_ipi"]["mean"],
    })
pd.DataFrame(rows).to_csv("leaderboard.csv", index=False)
```

For per-channel debugging, swap `r["<metric>"]["mean"]` for `r["<metric>"]["per_channel_mean"]`.

### Common gotchas (read this before your first run)

1. **`--fs` must match your data.** Default is `1.0`. For an ECG sampled at 250 Hz, pass `--fs 250` or `w1_ipi` will be in samples (not seconds).
2. **Channel order in `real` and `fake` must match exactly.** All per-channel metrics compare channel `c` of `real` against channel `c` of `fake`.
3. **Tune `--peak_kwargs_json` per dataset.** `find_peaks` with no kwargs picks up noise spikes. At minimum set `distance` and `prominence`. For per-channel kwargs, use the programmatic API (Workflow D) — the CLI flag only sets a global default.
4. **The four neural metrics need a GPU to be fast.** On CPU, expect minutes per metric per iteration. Use `--metrics` to skip them during fast iteration.
5. **Make sure `eval_ref_1/wMissing_scripts/ts2vec/` exists.** Context-FID imports it via `utils/ts2vec_loader.py`. If you delete the reference folder, Context-FID will fail to import.

---

## 4. Entry scripts

Both entry scripts share the same CLI surface. The with-missing version adds `--real_mask` and `--min_overlap`.

### Common flags

| Flag                  | Default          | Notes |
|-----------------------|------------------|-------|
| `--real`              | (required)       | `.npy` of real data, shape `(N, T, C)` |
| `--fake`              | (required)       | `.npy` of synthetic data |
| `--iterations`        | `3`              | Repeats for stochastic metrics; deterministic ones run once internally |
| `--seed`              | `0`              | Sets `numpy` and `torch` seeds; also seeds the synthetic fake-mask resampler |
| `--fs`                | `1.0`            | Sampling rate (Hz); used by IPI metric |
| `--max_lag`           | `64`             | ACF max lag |
| `--peak_kwargs_json`  | `None`           | JSON string forwarded to `scipy.signal.find_peaks` for peak / IPI metrics |
| `--out_json`          | `None`           | Dump full result dict to JSON |
| `--metrics`           | all 8            | Subset; choices below |

`--metrics` choices: `context_fid cross_correlation discriminative predictive w1_marginal acf_loss w1_peak_amp w1_ipi`.

### Extra flags for `eval_wMissing.py`

| Flag             | Default | Notes |
|------------------|---------|-------|
| `--real_mask`    | `None`  | `.npy` bool mask matching `real.shape`. Omit (or pass an all-True mask) to short-circuit to the no-missing pipeline. |
| `--min_overlap`  | `30`    | Pairwise-complete correlation min overlap for cross-correlation |

If `--real_mask` is missing or fully True, every metric short-circuits to its no-missing implementation — guaranteeing **bit-equal results to `eval_woMissing.py` on fully-observed inputs**.

---

## 5. Metric reference

Each subsection below describes one metric: what it measures, the algorithm in one sentence, the inputs/outputs, the relevant parameters, and how to interpret the score.

### 5.1 `context_fid` — Context-FID

**What it measures.** Global semantic similarity in a learned embedding space. The "FID" of time-series.

**Algorithm.** Train TS2Vec on the real data, encode both real and fake into a 320-d vector per sample, compute the Frechet distance between the two empirical Gaussians.

**Inputs / output.** `real, fake : (N, T, C)` → `float`. **Lower is better. 0 = indistinguishable embeddings.**

**With missingness.** A synthetic fake-mask is drawn by resampling `real_mask` patterns (with replacement) and applied to fake (zero-fill). Both real and fake then enter TS2Vec under the **same missingness distribution**, with no auxiliary mask channel — the encoder cannot use "is this position zero?" as a side signal.

**Parameters.** Internal TS2Vec defaults (output_dims=320, max_train_length=3000) — not exposed as CLI flags; tweak in code if needed.

**Interpretation.** A few units of magnitude are expected on physiological data; track the *delta vs. baseline generators*, not the absolute number.

**File.** `metrics/context_fid_{wo,w}Missing.py`

---

### 5.2 `cross_correlation` — Cross-channel correlation matrix MAE

**What it measures.** Whether the **inter-channel coupling structure** at lag 0 is preserved (e.g., HR ↔ respiration coupling, multi-lead ECG synchrony). **The single most important metric in the multi-channel setting.**

**Algorithm.** Per-side, compute the `C×C` Pearson cross-correlation matrix at lag 0 (via `cacf_torch` on z-standardized data); take the L1 distance between real and fake matrices over the lower-triangular indices, divided by 10 for scale parity with the eval_ref_1 protocol.

**Inputs / output.** `real, fake : (N, T, C)` → `float`. **Lower is better.** **Returns 0 if C=1** (no off-diagonal pairs).

**With missingness.** Real's `C×C` matrix is computed with `pairwise_complete_corr` — for each `(i, j)` channel pair, the Pearson correlation uses only timesteps where **both** channels are observed in the same sample. Fake (fully observed) is computed straight on the full signal. Pairs with overlap < `min_overlap` (default 30) on real yield NaN and are excluded from the score.

**Parameters.** `--min_overlap` (with-missing only, default 30). `max_lag=1` is hard-coded inside the metric (lag-0 only) — easily extendable.

**Interpretation.** A score near 0 means the generator preserves how channels move together. Spikes in this metric usually mean a generator that produces channels **independently** instead of jointly.

**File.** `metrics/cross_correlation_{wo,w}Missing.py`

---

### 5.3 `discriminative` — GRU real-vs-fake classifier

**What it measures.** Whether a supervised model can still tell real from synthetic.

**Algorithm.** Single-layer GRU(hidden = `dim/2`) → Linear(1) trained with Adam (lr=1e-3, BCE) for 2000 iterations on batches of 128. 80 / 20 split. Score = `|0.5 - test_accuracy|`.

**Inputs / output.** `real, fake : (N, T, C)` → `(score, fake_acc, real_acc)`. **Lower is better. 0 = totally indistinguishable.**

**With missingness.** Following the EHR-Safe / HealthGen / EHR-M-GAN protocol family: a synthetic fake-mask is drawn by resampling `real_mask` patterns (with replacement) and applied to fake (zero-fill). Both sides then enter the **same GRU architecture as the no-missing path** — input stays at `C` channels, no mask channel is concatenated. Because both sides have zeros at positions drawn from the same distribution, the classifier cannot exploit "is this zero?" as a discriminative leak.

**Parameters.** `iterations=2000, batch_size=128` hard-coded inside (matches eval_ref_1).

**Interpretation.** 0.05 is "very good", 0.15+ usually means the classifier exploits something obvious (range, noise floor, distributional shift).

**File.** `metrics/discriminative_{wo,w}Missing.py`

---

### 5.4 `predictive` — TSTR one-step forecaster

**What it measures.** Train-on-Synthetic-Test-on-Real for forecasting. Whether the temporal structure transfers.

**Algorithm.** Single-layer GRU(hidden = `dim/2`) trained on synthetic (5000 iterations, batch 128) to predict the next-step value of the **last** channel from the previous `dim-1` channels' history. Final score = MAE on real.

**Inputs / output.** `real, fake : (N, T, C)` → `float MAE`. **Lower is better.**

**With missingness.** A synthetic fake-mask is resampled from `real_mask` and applied to fake before training, so the predictor sees the same gap regime at train time as it will encounter at test time on real. Input is `2*(dim-1)` channels (history features ⊕ history mask — mask channel is auxiliary input here, not a discriminative leak, since this is regression). Loss is masked L1; test MAE is computed only on positions where the target is observed.

**Single-channel fallback.** When `C=1` the protocol reduces to single-channel autoregression with mask channel (input = `(x_t, m_t)`, target = `x_{t+1}` scored only when `m_{t+1} = 1`).

**Parameters.** `iterations=5000, batch_size=128` hard-coded.

**Interpretation.** Compare against:
- A **real-trained** model (lower bound — same architecture, trained on real).
- A **naive last-value** baseline (upper bound — `pred = x_t`).

If your synthetic-trained MAE is close to real-trained MAE and beats naive, the temporal structure transferred.

**File.** `metrics/predictive_{wo,w}Missing.py`

---

### 5.5 `w1_marginal` — Per-channel value distribution W1

**What it measures.** The basic "are the value ranges and densities correct" check, **per channel independently**.

**Algorithm.** For each channel `c`, flatten `real[..., c]` and `fake[..., c]` and compute `scipy.stats.wasserstein_distance(r, f)`. Report the per-channel array and the channel-mean.

**Inputs / output.** `real, fake : (N, T, C)` → `{per_channel: list[C], mean: float}`. **Lower is better.** Same units as the input signal.

**With missingness.** Only observed (and finite) values from real contribute to the real-side empirical distribution. Fake uses every value. W1 does not require equal sample sizes on the two sides.

**Interpretation.** A blowup in `w1_marginal` for a single channel often means the generator collapsed that channel to the wrong mean / variance — easy fix, but invisible to global metrics like Context-FID.

**File.** `metrics/w1_marginal_{wo,w}Missing.py`

---

### 5.6 `acf_loss` — Per-channel multi-lag autocorrelation L2

**What it measures.** **Continuous-time rhythmic structure** within each channel — how strongly the signal correlates with shifted versions of itself.

**Algorithm.** For each channel, compute the sample-mean autocorrelation at lags `0..max_lag-1`, then take the L2 norm of the (real − fake) difference vector. Per-channel array + channel-mean.

**Inputs / output.** `real, fake : (N, T, C)` → `{per_channel: list[C], mean: float}`. **Lower is better.**

**With missingness.** Real's per-lag autocorrelation uses **pairwise-complete observations** only — for each lag `l`, only `(t, t+l)` pairs where both are observed are included. Lags with no valid pair on the real side become NaN and are dropped from the per-channel L2. Fake's ACF uses the standard sample-mean estimator.

**Parameters.** `--max_lag` (default 64). Set to ≈ one full period of your signal for best signal/noise.

**Interpretation.** Complements `cross_correlation` (which is *between* channels at lag 0). `acf_loss` is *within* a channel, *across* lags. Both can spike independently. Multi-lag ACF also indirectly captures most of what a frequency-domain metric would tell you, which is why this suite drops PSD entirely.

**File.** `metrics/acf_loss_{wo,w}Missing.py`

---

### 5.7 `w1_peak_amp` — Per-channel peak amplitude W1

**What it measures.** **Discrete peak heights** per channel — R-wave amplitude, CGM postprandial peak height, respiratory tidal volume, EEG event amplitude.

**Algorithm.** Per `(sample, channel)`, run `scipy.signal.find_peaks` to get peak indices, take the signal values at those indices, pool across all samples for that channel, then `wasserstein_distance(amp_real, amp_fake)`.

**Inputs / output.** `real, fake : (N, T, C)` → `{per_channel: list[C], mean: float}`. **Lower is better.** Units = signal units.

**With missingness.** Real-side peaks are detected only on **maximal contiguous observed segments** of each `(sample, channel)`. A peak adjacent to a missing block is discarded to avoid spurious extrema at discontinuities. Fake-side peaks are detected on the full signal.

**Parameters.**
- `--peak_kwargs_json`: JSON dict forwarded to `find_peaks`. Recommended keys: `height`, `distance`, `prominence`, `width`.
- Programmatically you can pass `peak_kwargs_per_channel: list[dict]` to give each channel its own settings (highly recommended for multi-modal data, e.g. ECG + respiration on the same array).

**Interpretation.** Often the most informative single metric for "does it look right physiologically?". A small `w1_peak_amp` means the generator gets event magnitudes right.

**File.** `metrics/w1_peak_amp_{wo,w}Missing.py`

---

### 5.8 `w1_ipi` — Per-channel inter-peak interval W1

**What it measures.** **Event-level rhythm**: the distribution of inter-peak intervals (in seconds if `fs` is set, otherwise in samples). Captures RR-intervals, breath-to-breath intervals, postprandial peak spacing — anything ACF sees only as smeared continuous rhythm.

**Algorithm.** Per `(sample, channel)`, find peaks → `np.diff(peak_idx) / fs` → pool across samples → `wasserstein_distance`.

**Inputs / output.** `real, fake : (N, T, C)` → `{per_channel: list[C], mean: float}`. **Lower is better.** Units = seconds (if `fs` set).

**With missingness.** Real-side peaks are detected within each contiguous observed segment; **only intervals that fall entirely within one segment are counted** — never across a missing block (those would be mask-artifacts, not real periods). Fake-side IPIs are computed across the full signal.

**Parameters.** Same as `w1_peak_amp` (`--peak_kwargs_json` / `peak_kwargs_per_channel`); plus `--fs` to convert intervals to seconds.

**Interpretation.** Together with `w1_peak_amp`, these two cover the "discrete event-level" view that ACF smears out. Very different from `acf_loss`: a generator can have great ACF but terrible IPI distribution if it spaces beats too regularly (no jitter).

**File.** `metrics/w1_ipi_{wo,w}Missing.py`

---

## 6. The two-mode design

Each of the eight metrics is implemented twice — once for the no-missingness case (`*_woMissing.py`) and once mask-aware (`*_wMissing.py`). The split serves two purposes:

1. **Speed and clarity for the common case.** The no-missing path is a straight, unconditional algorithm — easier to read, audit, and benchmark.
2. **Bit-equal parity.** Every with-missing metric calls `is_fully_observed(ori_mask)` first; if true, it dispatches to the no-missing implementation **literally**. This guarantees that `eval_wMissing.py` with no `--real_mask` (or an all-True mask) returns *exactly* the same numbers as `eval_woMissing.py`.

### Mask conventions

- `M : (N, T, C) bool` — `True = observed`, `False = missing`.
- Missing values in `X` may be NaN, Inf, or any sentinel — `sanitize` zero-fills them.
- For metrics that find events (peaks, IPIs), processing is restricted to maximal contiguous observed runs (`per_sample_observed_indices`).
- For correlation-style metrics, real uses `pairwise_complete_corr` (pair `(i, j)` correlations on timesteps where both are observed).
- For metrics that need both sides to share the same missingness regime (Context-FID, Discriminative, Predictive), real-mask patterns are resampled with replacement onto fake (`resample_masks`).

### Mask-aware adaptations per metric

| Metric                | wo-missing                              | w-missing adaptation                                                                  |
|-----------------------|-----------------------------------------|---------------------------------------------------------------------------------------|
| `context_fid`         | TS2Vec on `(N,T,C)`                     | Resample real_mask onto fake; zero-fill both; TS2Vec on `(N,T,C)` (no mask channel)   |
| `cross_correlation`   | `cacf_torch` lag-0 matrix               | Real → `pairwise_complete_corr`; fake → standard correlation                          |
| `discriminative`      | GRU on `C` channels                     | Resample real_mask onto fake; zero-fill both; reuse no-missing GRU (no mask channel)  |
| `predictive`          | History → next-step MAE                 | Resample real_mask onto fake; train with masked L1; mask channel kept as input        |
| `w1_marginal`         | All values per channel                  | Real: observed values only; fake: full                                                |
| `acf_loss`            | Sample-mean ACF per lag                 | Real: pairwise-complete ACF; fake: standard                                           |
| `w1_peak_amp`         | `find_peaks` on full signal             | Real: `find_peaks` per contiguous observed run; fake: full                            |
| `w1_ipi`              | `diff` of peaks on full signal          | Real: `diff` of peaks **within** each observed run; fake: full                        |

---

## 7. Output schema

### Scalar metrics (`context_fid`, `cross_correlation`, `discriminative`, `predictive`)

```jsonc
{
  "per_iter": [0.123, 0.118, 0.121],   // one entry per --iterations
  "mean":     0.1206,
  "ci95":     0.0034                   // 95% CI half-width via t-distribution
}
```

### Per-channel metrics (`w1_marginal`, `acf_loss`, `w1_peak_amp`, `w1_ipi`)

```jsonc
{
  "per_iter":         [[0.05, 0.07, 0.04]],   // [iter][channel]
  "per_channel_mean": [0.05, 0.07, 0.04],     // averaged across iters, length C
  "mean":             0.0533,                  // channel-mean of per_channel_mean
  "ci95":             0.0
}
```

`--out_json results.json` dumps the full `{metric_name: result_dict}` map.

---

## 8. Multi-channel guidance

This suite is built for the multi-channel case. Three concrete recommendations:

### 8.1 Always check the per-channel breakdown

Channel-mean is convenient for leaderboards but hides which channel is the actual problem. Inspect `per_channel_mean` for `w1_marginal`, `acf_loss`, `w1_peak_amp`, `w1_ipi` whenever the mean looks bad.

### 8.2 Channel ordering must be consistent

Every per-channel metric is computed channel-by-channel. If your generator outputs channels in a different order than `real`, **all per-channel metrics will look catastrophically bad even if the data is fine**. Align channel indexing before evaluating.

### 8.3 Per-channel peak-finder kwargs

For multi-modal arrays (e.g., ECG + respiration in the same `(N,T,C)`), the right peak threshold is wildly different per channel. Use the programmatic `peak_kwargs_per_channel` argument:

```python
w1_peak_amp(real, fake, peak_kwargs_per_channel=[
    {"distance": 30, "prominence": 0.3},   # ECG R-peaks
    {"distance": 200, "prominence": 0.05}, # respiration
])
```

The CLI flag `--peak_kwargs_json` only sets a global default for all channels.

### 8.4 Predictive's last-channel target

The predictive score in the eval_ref_1 protocol uses the **last** channel as the forecast target. If channel ordering matters semantically (e.g., glucose is the channel you actually care about), put glucose last. Otherwise, run predictive multiple times with different channel orderings and average — see §10.2.

---

## 9. Pitfalls and assumptions

### 9.1 `fs` is needed for `w1_ipi` units

The default is `1.0`. If your ECG is sampled at 250 Hz and you forget `--fs 250`, IPIs are reported in samples, not seconds. (PSD is no longer in this suite — see §1's note on why.)

### 9.2 Peak finding is sensitive to its kwargs

`scipy.signal.find_peaks` with no kwargs returns local maxima of any height, which is rarely what you want for physiology. **Always tune `distance`, `prominence`, and/or `height` per channel.** Otherwise `w1_peak_amp` and `w1_ipi` will be dominated by noise spikes.

### 9.3 Discriminative / Predictive run neural networks

These two metrics train RNNs (2000–5000 iterations, batch 128) and are by far the slowest. On CPU each takes minutes; on GPU each takes seconds. Use `--metrics` to skip them during fast iteration.

### 9.4 `iterations` only applies to stochastic metrics

`context_fid`, `cross_correlation`, `discriminative`, `predictive` are stochastic (TS2Vec / GRU training) and benefit from multiple runs to estimate confidence. The four distance-based metrics are deterministic and run once internally regardless of `--iterations`.

### 9.5 Context-FID needs the eval_ref_1 ts2vec package

`utils/ts2vec_loader.py` registers `eval_ref_1/wMissing_scripts/ts2vec/` as both `ts2vec` and `Models.ts2vec.*`. If you move or remove `eval_ref_1`, Context-FID breaks. Either copy the package locally or update the path in `ts2vec_loader.py`.

### 9.6 Single-channel quirks

- `cross_correlation` returns `0` (no off-diagonal pairs to compare).
- `predictive` falls back to single-channel autoregression (`x_t → x_{t+1}`).
- All other metrics work the same as the multi-channel case (the per-channel array just has length 1).

### 9.7 Mask-aware bit-parity

When `--real_mask` is omitted or fully True, each `_masked` function dispatches to its no-missing twin. **Both pipelines therefore return identical numbers on the same fully-observed inputs** — this is enforced by `is_fully_observed(ori_mask)` short-circuits and is the suite's core correctness guarantee.

### 9.8 The asymmetry assumption

This suite assumes fake is always fully observed. If your generator literally outputs a mask alongside its samples, you should **not** use the with-missing path with that mask — instead, use the *intersection* of fake_mask and real_mask as your effective real_mask before passing it to the script. Re-introducing a user-facing fake_mask was deliberately rejected here because it is rare in practice and adds protocol complexity.

---

## 10. Extending the suite

### 10.1 Add a 9th metric

1. Create `metrics/<name>_woMissing.py`. Expose one function `<name>(real, fake, **kw) -> float | dict`.
2. Create `metrics/<name>_wMissing.py`. Implement `<name>_masked(real, fake, ori_mask=None, **kw)`. Start with `if is_fully_observed(ori_mask): return _woMissing(...)`.
3. Add the import + a `_run_scalar` or `_run_dict` entry in both `eval_woMissing.py` and `eval_wMissing.py`.
4. Add the name to the `--metrics` `default` list and `choices` list in both entry scripts.

### 10.2 Rotate predictive target channel

The eval_ref_1 protocol forecasts only the last channel. To average across all channels:

```python
from metrics.predictive_woMissing import predictive_score

scores = []
for c in range(C):
    perm = list(range(C))
    perm.remove(c); perm.append(c)   # move target c to last position
    scores.append(predictive_score(real[..., perm], fake[..., perm]))
mean_pred = float(np.mean(scores))
```

The CLI does not do this by default to stay protocol-compliant.

### 10.3 Use a different embedding for Context-FID

In `metrics/context_fid_woMissing.py`, swap `TS2Vec` for any `(N, T, C) → (N, D)` encoder of your choice; the `_calculate_fid` helper is encoder-agnostic.

### 10.4 Re-introduce a frequency-domain metric (PSD-W1) for fully-observed data

If you have only the no-missing case to evaluate, you can add a Welch-PSD-W1 metric as a 9th metric (see §10.1 for the recipe). Sketch:

```python
from scipy.signal import welch
from scipy.stats  import wasserstein_distance

def psd_w1(real, fake, fs, nperseg=256):
    N, T, C = real.shape
    out = np.empty(C)
    for c in range(C):
        f, P_r = welch(real[..., c], fs=fs, nperseg=nperseg, axis=1)
        _, P_f = welch(fake[..., c], fs=fs, nperseg=nperseg, axis=1)
        P_r = P_r.mean(0); P_r /= P_r.sum()
        P_f = P_f.mean(0); P_f /= P_f.sum()
        out[c] = wasserstein_distance(f, f, u_weights=P_r, v_weights=P_f)
    return {"per_channel": out.tolist(), "mean": float(out.mean())}
```

Do **not** add a missing-aware variant unless you switch to Lomb-Scargle / `mne.psd_array_welch` with mask support — see §1's note on why a naive PSD under missingness is unreliable.

### 10.5 Add cross-spectral coherence (optional)

If your physiology has known cross-channel frequency-band coupling (e.g., respiration → HR), `scipy.signal.coherence` per channel pair gives `(C, C, freq)`. Per-pair-per-band W1 against real coherence yields a coherence-domain analogue of `cross_correlation`. Sketch:

```python
from scipy.signal import coherence
def coherence_w1(real, fake, fs):
    N, T, C = real.shape
    out = np.zeros((C, C))
    for i in range(C):
        for j in range(i+1, C):
            f_r, c_r = coherence(real[..., i], real[..., j], fs=fs)
            f_f, c_f = coherence(fake[..., i], fake[..., j], fs=fs)
            out[i, j] = wasserstein_distance(f_r, f_f, u_weights=c_r, v_weights=c_f)
    return out
```

---

## Reference protocols

- **TimeGAN** — Yoon et al. (2019). The discriminative & predictive scores originate here.
- **Diffusion-TS / Context-FID** — Yuan & Qiao (2024). Context-FID and the four-metric reporting convention.
- **Mask-aware variants** — adapted from `eval_ref_1/wMissing_scripts/` in this repo, with the simplifying assumption that fake is fully observed (so the user-facing `fake_mask` argument is dropped).
- **Physiology supplements** — peak / IPI / per-channel ACF / W1-marginal designed in this suite to close gaps the four classics leave open on physiological signals.

## License & attribution

This suite reuses code conventions and the TS2Vec package from the `eval_ref_1/` reference. Cite the underlying TimeGAN, Diffusion-TS, and TS2Vec papers when reporting these metrics.
