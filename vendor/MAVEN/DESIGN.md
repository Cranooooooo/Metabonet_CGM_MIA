# MAVEN-MaskAE — design notes

## Pipeline
1. Each multivariate TS window `x ∈ [B,T,D]` is mapped to **two image views** by
   exactly-invertible transforms (`visual_transforms.py`):
   - **A** = delay embedding, **B** = complex STFT. Tokens `[B,L,D]`; the channel
     axis `D` is preserved by both transforms.
2. Per view, a **mask-conditioned autoencoder velocity net** (`AEMaskVelocityNet`
   in `train_maven_maskae.py`) runs flow matching (FM) in image space.
3. The two per-view samples are decoded to TS and **fused** post-hoc by the
   retrieval-guided editor (`fusion_nview.py`, base = strongest view).
4. Evaluate with `Eval_Assembly/` (Context-FID, Cross-correlation, Discriminative,
   Predictive; lower = better).

## The model (latest design)
`AEMaskVelocityNet` has an **autoencoder hidden-dim schedule**
(default `[128,128,64,64,64,64,128,128]`): the width shrinks to a 64-d
**bottleneck (the coarse representation)** then expands to 128 (**fine**). It has
**two output heads** — `head_coarse` at the last bottleneck block, `head_fine` at
the final block — and the velocity-reconstruction loss is computed at **both**
(deep supervision at coarse + fine). Multi-resolution therefore comes from the AE
dims, not from separate fine/coarse transforms.

### Masked-imputation auxiliary task (per step, per view)
- **GEN pass**: unconditional FM (`mask=0`) → soft prior `x0_pred_g = x_t - t·v`.
- **IMPUTE pass**: `sample_image_mask` masks **whole channels** of the image;
  the net is conditioned on the observed channels (`X_cond = M_cond·image`) and
  the gen soft prior for the masked ones (`X_prior = (1−M_cond)·x0_pred_g.detach()`);
  the reconstruction loss `L_imp` is over the **masked cells only**. This forces the
  net to learn `p(channel_j | other channels)` → **inter-channel structure**, the
  weakness on high-channel datasets (e.g. Energy, D=28).
- `L = [L_imp@fine + cw·L_imp@coarse] + λ·[L_gen@fine + cw·L_gen@coarse]`.
- Conditioning projections (`cond_proj`/`prior_proj`) are **zero-init**, so the net
  reduces to a plain unconditional FM at init.
- **Sampling** is unconditional (`mask=0`) using the **fine head**.

## Empirical status (2026-06, Energy & Stocks-128, lower=better)
- The masked reconstruction **strongly improves Discriminative** (e.g. Stocks-128
  DS 0.002–0.02 vs DiMTS 0.027) — the inter-channel objective works.
- **Context-FID plateaus** and does not drop with training (Stocks-128 CFID ~0.10
  vs DiMTS 0.034; Energy CFID ~0.28–0.36 vs DiMTS 0.104). Verified **not a code bug**.

## Ablation ladder (open questions to explore)
A code review argued the CFID plateau is structural — routing *generation* through
the 64-d bottleneck + coarse deep-supervision caps distributional fidelity (note:
IG-FM's enc→bottleneck→dec is actually **constant-width**; the bottleneck there only
taps an intermediate `z` for helper losses). Suggested ablations, expose via flags:
1. **`--coarse_weight 0`** — drop deep-supervision on the generation path (keep it
   only as DS regularizer / on the impute pass). Cheapest test.
2. **Constant-width trunk** — `--dims 128,128,128,128,128,128,128,128` (no
   dimensional squeeze); the bottleneck becomes a tap, not a choke on generation.
3. **Add a channel-correlation loss** (corrmap / decor on a full-width `z`) to
   supervise inter-channel structure directly, instead of via the bottleneck.
4. **λ schedule** — try the ramp `40000:1,80000:2,120000:4` and lower `--mask_hi`
   so imputation stays a light auxiliary, prioritising the generation objective.

## Files
- `train_maven_maskae.py` — the model + training (this design).
- `visual_transforms.py` — delay/STFT invertible image transforms.
- `fusion_nview.py` — N-view retrieval-guided editor fusion (post-hoc).
- `fusion_editor.py` — 2-view editor + shared losses imported by `fusion_nview`.
- `train_maven.py` — the base per-view FM generator (no mask; reference baseline).
- `Eval_Assembly/` — the 4-metric evaluation suite (incl. TS2Vec for Context-FID).
- `datasets/` — the four GENERAL benchmark CSVs.
