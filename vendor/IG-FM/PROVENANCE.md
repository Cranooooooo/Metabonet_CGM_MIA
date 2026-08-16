# IG-FM — vendored

Copied from `Cranooooooo/IGFM_ICDE2027` at commit `3f9a246`
(local checkout: `/home/ling/workspace/project_general/Model_Joint_woMissing/IGFM_ICDE2027`), on 2026-08-02.

`igfm_core.py` is `IG_FM/train_r10.py` verbatim. **Do not edit it.** Adapting the
model to our data is done entirely in `src/m7mia/generators/igfm.py`, so this file
can be re-synced from upstream with a plain copy.

## What upstream expects vs what we have

| | upstream | ours |
|---|---|---|
| input | continuous CSV, stride-1 windows | 73,404 pre-windowed daily chunks |
| window T | 64 | **288** |
| channels D | 6–59 | **3** |
| normalisation | MinMaxScaler to [-1,1] | already z-clipped to ~[-1,1] |

Two consequences the adapter handles:

1. **The CSV windower cannot be used.** Our chunks come from 402 different
   subjects and are non-overlapping gap-free days. A stride-1 cutter over their
   concatenation would manufacture windows spanning day and subject boundaries,
   corrupting both the data and the subject structure the membership evaluation
   depends on.
2. **Batch size must drop.** Attention is O(T^2); at T=288 vs 64 the attention
   matrix is ~20x larger, and upstream's batch 256 needs >20 GB. We use a smaller
   micro-batch with gradient accumulation so the *effective* batch stays 256 and
   the training dynamics match upstream.

## Environment

Needs `torch >= 2.1` for `torch.optim.swa_utils.get_ema_multi_avg_fn`; the main
project env has torch 2.0.1 and will fail on it. Use the DIMTS env
(`/home/ling/.conda/envs/project_general_DIMTS`, torch 2.12), which we verified is
numpy-2 clean for our loader. `python-dateutil` was installed there to repair a
broken pandas (upstream imports pandas at module scope).
