# tsgen_metrics

Reusable **generation-quality metrics for time series**. Given a real set and a
generated set — both arrays shaped `(N, T, C)` = (windows, timesteps, channels) in
the **same value space** — it computes **10 metrics, all lower = better**, with a
single call.

```python
from tsgen_metrics import evaluate
res = evaluate(real, fake)                 # real/fake: float32 (N, T, C)
print(res["metrics"]["fdds"]["value"])
```

CLI:

```bash
python -m tsgen_metrics --real real.npy --fake fake.npy --out scores.json
# or, after `pip install -e .`:
tsgen-metrics --real real.npy --fake fake.npy
```

## Install

```bash
pip install -e .        # deps: numpy, scipy, torch
```

Neural metrics use a GPU if available (CPU works, slower).

## Metrics & provenance

10 metrics, all **lower = better**. Each result carries `abbrev`, `name`, `status`
and `reference`. Four provenance tiers:

| key | abbrev | metric | status | reference |
|---|---|---|---|---|
| `context_fid` | Context-FID | Context-FID | **reference-aligned** | DiM-TS `Context_FID` (vendored official **TS2Vec**) |
| `correlational` | CC | Cross-Correlation Difference | **reference-validated** | Diffusion-TS `CrossCorrelLoss` |
| `vds` | VDS | Value Distribution Shift | **reference-validated** | PaD-TS `VDS_Naive` (Li et al. 2025) |
| `fdds` | FDDS | Functional Dependency Distribution Shift | **reference-validated** | PaD-TS `BMMD_Naive(cross_correlation_distribution)` (Li et al. 2025) |
| `discriminative` | DS | Discriminative Score | **recipe-matched** | DiM-TS/TimeGAN GRU (hidden=C//2, iters=2000, batch=128) |
| `predictive` | PS | Predictive Score (TSTR) | **recipe-matched** | DiM-TS/TimeGAN GRU (hidden=C//2, iters=5000, batch=128) |
| `mdd` | MDD | Windowed MMD | own | RBF-MMD² over flattened windows (median-heuristic bandwidth) |
| `acd` | ACD | AutoCorrelation Difference | own | L2 of mean autocorrelation curves |
| `skewness_diff` | SD | Skewness Difference | own | per-channel skewness gap |
| `kurtosis_diff` | KD | Kurtosis Difference | own | per-channel kurtosis gap |

- **reference-validated** (`correlational`, `vds`, `fdds`): reproduces the published
  reference on identical input — verified by `validate_against_reference.py`
  (`correlational`/`fdds` match exactly; `vds` within the reference's 10000-sample
  stochasticity).
- **reference-aligned** (`context_fid`): uses the **official TS2Vec** encoder
  (vendored at `tsgen_metrics/_vendor/ts2vec`) exactly as DiM-TS's `Context_FID`.
  Faithful up to TS2Vec's own training randomness (the official also averages runs).
  On the metabonet sweep, our values vs the official `run_eval.py` were 8.9 vs 8.3,
  0.13 vs 0.70, 22.6 vs 20.5 — **same ranking**, ~10% magnitude (across a different
  normalization too).
- **recipe-matched** (`discriminative`, `predictive`): the official GRU
  hyper-parameter recipe (hidden, iterations, batch, Adam lr). These are **stochastic
  trained nets** and TF↔PyTorch internals differ, so they track the official ranking
  but are **not bit-identical** to the official TF numbers. For the exact official
  numbers, run DiM-TS/PaD-TS `run_eval.py`.
- **own** (`mdd`, `acd`, `skewness_diff`, `kurtosis_diff`): self-contained, standard
  statistics; no single canonical published implementation to match.

## Validate against the published references

```bash
python validate_against_reference.py --real real.npy --fake fake.npy \
    --diffts-dir /path/to/Diffusion-TS \
    --padts-dir  /path/to/PaD-TS
```

This recomputes `correlational` (Diffusion-TS `CrossCorrelLoss`), `vds` and `fdds`
(PaD-TS `MMD.py`) with the original code on the same data and prints the relative
difference. Only the deterministic metrics are validated (the neural ones cannot be,
by construction — see above).

## API

`evaluate(real, fake, *, subsample=2000, seed=0, n_seeds=3, epochs=40, device=None,
feature_names=None, include_neural=True, verbose=False) -> dict`

- `subsample`: equal per-side window count (`None` = all, capped at `min(N_real, N_fake)`).
- `n_seeds`: statistical metrics are averaged over this many independent subsamples
  (`value` = mean, `std` reported). Neural metrics use `seed` only.
- `include_neural=False`: skip the proxy neural metrics (deterministic-only, fast, CPU).

Returns `{"metrics": {<name>: {value, lower_is_better, status, reference, std?, ...}},
"per_channel": {...}, "window", "channels", ...}`.

**Important**: `real` and `fake` must already be in the same normalization. Metrics
compare them directly; if you MinMax one and z-score the other, the numbers are
meaningless.

## Layout

```
tsgen_metrics/
  __init__.py        # evaluate, PROVENANCE
  core.py            # evaluate() orchestration + subsampling
  statistical.py     # correlational, mmd, acd, skewness, kurtosis, vds, fdds
  neural.py          # context_fid, discriminative, predictive (proxies)
  __main__.py        # CLI
validate_against_reference.py
examples/quickstart.py
```

## Provenance of the code itself

`statistical.py` / `neural.py` originated in the `naive_MIA_testing/metrics` engine of
the Model_7 MIA framework. The deterministic metrics were aligned to, and validated
against, the published **Diffusion-TS** (`Utils/cross_correlation.py`) and **PaD-TS**
(`eval_utils/MMD.py`) implementations. **Context-FID uses the official TS2Vec encoder**,
vendored verbatim at `tsgen_metrics/_vendor/ts2vec` (from DiM-TS). The Discriminative /
Predictive GRUs are PyTorch reimplementations of the official recipe (matched
hyper-parameters), so they track the official ranking but are stochastic — for the exact
official TF numbers, run DiM-TS / PaD-TS `run_eval.py`.

`_vendor/ts2vec` is third-party code (TS2Vec, MIT) included for faithful Context-FID.
