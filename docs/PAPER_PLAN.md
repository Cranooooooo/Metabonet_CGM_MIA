# Paper plan — membership-inference risk in CGM synthesis, and a generator that reduces it

## Thesis

Synthetic CGM data carries membership-inference risk. We quantify that risk across
generative baselines, localise where it comes from, and design a generator — flow
matching with a time-series editing stage — that lowers it **without paying for it in
generation quality**.

---

## Step 1 — Establish that the risk exists

### What it does

Prove, on a design that admits no alternative explanation, that membership in a
generator's training set can be inferred from its released samples.

### Why it has to come first, and why it has to be this strict

Everything downstream is conditional on the risk being real. The design therefore has to
close each escape route a reviewer would reach for:

- **Symmetric pairing.** Controls are drawn *out* of the background before the base is
  built, so every pair is `(background + target)` against `(background)` and both arms
  share one base. A per-run offset enters both arms with the same sign and cancels.
- **A frozen statistic.** `min × mean` was fixed in `configs/experiment.yaml` before any
  result was seen. Six grid cells exist; choosing the best one after the fact is not a
  measurement. On `d1_c1` the frozen cell reads 0.562 while `mean × mean` reads 0.651 —
  the difference between a null and a headline.
- **Controls matched on exposure.** Each control has *exactly* the same window count as
  its outlier (maximum relative gap 0.000), so "people with longer records are easier to
  identify" cannot explain the contrast.
- **Symmetric screening.** Targets are the intersection over four detector seeds (13
  subjects). Controls come from the 390 subjects **no method flagged in any seed**. The
  background drops anyone flagged in any seed. Screening targets strictly while screening
  controls loosely would put outliers in the control arm — the error that makes both arms
  leak and hides the effect.

### Status: 3 of 4 cells complete, `d7_c2` due 08-30

All four cells hold the same 506 subjects and the same 6,072 `(subject, day)` blocks,
h=256, 100,000 steps, 1,054 epochs. Only window length and channel count vary.

| cell | window | channels | arm AUC | p | `p_within` out / norm | max per-subject AUC | >0.55 out / norm |
|---|---|---|---|---|---|---|---|
| `d1_c1` | 1 day | CGM | 0.562 | 0.304 | 1.2e-04 / 1.2e-04 | 0.960 | 11/13 · 13/13 |
| `d1_c2` | 1 day | +basal/kg | 0.698 | 0.045 | 1.2e-04 / 1.7e-03 | 1.000 | 10/13 · 5/13 |
| `d7_c1` | 7 days | CGM | 0.627 | 0.141 | 1.2e-04 / 2.4e-04 | 1.000 | 13/13 · 12/13 |
| `d7_c2` | 7 days | +basal/kg | *training* | | | | |

### Finding

**The risk is real and it is not selective at convergence.** `p_within` is on the order
of 1e-4 in *both* arms of every finished cell: adding any subject moves the generator
toward that subject. One subject reaches a per-subject AUC of **1.000** — identified with
certainty.

The prior hypothesis that outliers are preferentially exposed does **not** hold at this
training budget; only `d1_c2` clears p < 0.05 on the arm contrast. Step 3b explains why.

### Deliverables

Table 1 (design) · Table 2 (four cells) · Figure 1 (per-subject AUC, outlier vs control)

---

## Step 2 — The risk–quality frontier across baselines

### What it does

Six generators × four cells × a full training-length curve each, plotted as generation
quality against membership risk.

### Why a frontier, and why a curve per baseline

Two reasons, and the second is not obvious:

1. **A frontier is what makes the contribution legible.** "Better privacy" and "same
   quality" are one claim, not two, and only a frontier states it. Against a ranking, a
   reviewer can answer that our model is simply worse and therefore safer — and be right,
   because a ranking cannot distinguish moving *off* the frontier from sliding *along*
   it. That distinction is the contribution.
2. **Single points are not comparable across baselines.** Step 3b shows risk is an
   **inverted U in training length** — 0.840 at 30k steps, 0.515 at 100k. Comparing
   models at one arbitrary budget compares each one at a random position on its own
   curve. The comparable quantity is **peak risk**, which requires the curve.

### Status

| generator | state |
|---|---|
| DiM-TS | 4 cells + a 12-point training-length curve — **complete** |
| copy_paste | positive control; ceiling of 0.816 measured, not assumed |
| TimeVAE | 1 base model; **fails the quality gate** — restart max 0.963, spread 0.461, context-FID 0.856 (9× DiM-TS). Trained in 9 minutes |
| PaD-TS, Diffusion-TS, DiffWave, FourierDiffusion, IG-FM | *wait for Step 2 prerequisites* |

### Two blockers, both prerequisites rather than options

1. **Only DiM-TS can resample from an intermediate checkpoint.** `GeneratorBase` has no
   `resample` in its interface; the other five generators cannot produce a
   training-length curve at all. Checkpoint-and-resample support has to be added to each.
2. **Per-model cost is unmeasured and varies by 30×.** TimeVAE trains in 9 minutes,
   DiM-TS in 4.6 hours. **Each baseline gets one base model measured first**, which is
   how TimeVAE was ruled out in 9 minutes rather than 124 GPU-hours.

A baseline whose base fails the quality gate stops there. That is a result — *this
architecture cannot produce usable CGM windows at this data scale* — not a gap.

### Cost

At DiM-TS's measured per-model times (4.6 / 7.9 / 22.9 / 26.0 h for the four cells),
one baseline over all four cells is **1,658 GPU-hours**; five baselines is **8,289**, and
the sampling for six training-length curves adds **2,640**. Roughly **10,900 GPU-hours**,
or 28 days at 16 cards. This is an upper bound set by the most expensive architecture;
the per-baseline probes in blocker 2 exist to bring it down.

### Deliverables

Figure 2 (frontier: context-FID against peak arm AUC, one curve per baseline) ·
Table 3 (peak location and peak risk per baseline)

---

## Step 3 — Localise the leak

### 3a — Which subjects — **complete**

**What it does.** Identify which individuals are repeatedly exposed, and show it is not
sampling noise.

**Why.** This is the step from *risk exists* to *risk is predictable*. If a different set
of people leaked each time, there would be nothing for a defence to act on.

**Result.** The **same six outliers** occupy the top of both `d1_c1` and `d1_c2` —
independently trained models, different channel counts, different seeds. Over all 26
targets, **Spearman ρ = +0.608, p = 0.001**.

### 3b — Which conditions amplify it — **complete**

**What it does.** Separate the effects of training length, window length and channel
count.

**Why.** A defence has to act where the risk is produced. If training length dominates,
early stopping is the cheapest possible defence and our model has to beat it.

**Result.** Training length is the strongest variable, and the mechanism is the opposite
of what we expected:

| steps | arm AUC | p | outlier > 0.55 | control > 0.55 |
|---|---|---|---|---|
| 20,000 | 0.633 | 0.130 | 8/13 | 7/13 |
| **30,000** | **0.840** | **0.002** | 9/13 | **2/13** |
| 40,000 | 0.757 | 0.014 | 10/13 | 2/13 |
| 60,000 | 0.681 | 0.062 | 9/13 | 4/13 |
| 80,000 | 0.550 | 0.341 | 11/13 | 10/13 |
| 100,000 | 0.515 | 0.459 | 9/13 | **13/13** |

*(`d1_c1`; the 20,000-step row is below convergence — restart max 0.647 — and is read as
such.)*

**The outlier arm barely moves. The control arm goes from 2/13 to 13/13.** Over-training
does not expose outliers further; it starts exposing everyone, and the contrast is what
disappears.

This has a direct consequence for Step 4: **early stopping preserves the contrast but not
the absolute risk** — at 30k steps 9 of 13 outliers still exceed 0.55. That gap is where
our model has to live.

Generation quality degrades over the same range (context-FID 0.060 → 0.083), so
over-training is not a privacy-for-quality trade: it costs both.

### 3c — Which of a person's information leaks — **designed, not started**

**What it does.** For the individuals located in 3a, determine *what* was memorised —
which hours of the day, and which kind of structure.

**Why this is the bridge to Step 4.** The proposed defence is time-series editing.
**Editing needs to know what to edit.** If the leak is a characteristic overnight
hypoglycaemic excursion, the module edits that; if it is overall variability, that is a
different operation entirely. Without 3c the editing module is guesswork.

#### Design — both questions answered with the **frozen statistic**, not a new attack

A separately-trained attack would produce a localisation that does not correspond to the
number reported in Step 1.

**(i) Where in time — an exact decomposition.**

The frozen statistic is `d(t, S) = mean_i ( min_k ‖R_i − S_k‖ / √F )`, and Euclidean
distance is additive over the time axis:

```
‖R_i − S_k‖² = Σ_h (R_i[h] − S_k[h])²
```

So: take the nearest neighbours exactly as the attack does — `k*_in = argmin` over
`include_t`'s samples, `k*_out` over `base`'s — then hold them fixed and split the squared
distance per timestep. Reporting `mean_i (c_out[i,h] − c_in[i,h])` against `h` gives a
curve whose sum **is** the quantity reported in Step 1. It is a decomposition, not an
approximation.

**(ii) What kind of structure — the same statistic in transformed spaces.**

Rather than `attack/panel.py`, whose ten features compare *attack strength* rather than
*information type* — and whose pooled protocol carries a known LOSO bias — recompute
`min × mean` in five spaces:

| transform | preserves | destroys |
|---|---|---|
| raw | everything | — |
| first difference | rate of change, variability | absolute level |
| sorted values | the distribution of glucose values | all timing |
| hourly means (288 → 24) | coarse shape | fine structure |
| per-window z-score | shape | level and scale |

If the gap survives on **sorted values**, the leak is distributional and editing must
change the distribution. If it needs the **raw** space, it is tied to specific events at
specific times and editing can be local. If it survives on **first differences**, it is a
dynamics signature. Each row maps onto a different editing operation.

#### The control that keeps this honest

A time curve will rise wherever glucose is most variable — post-prandial excursions
inflate every distance, membership or not. So **the control arm's curve is plotted
alongside**. Only where the outlier curve exceeds the control curve is there leakage;
where both rise together, it is variance. The same applies to the transform table: the
readable quantity is the **difference between the arms' gaps**, not either gap alone.

#### Cost

No new training. Distance recomputation on existing samples: one full attack takes ~10
minutes, so five transforms plus the time decomposition is roughly **1 CPU-hour per
cell**.

### Deliverables

Figure 3 (hour-of-day gap curves for the six most-exposed subjects, with the control
arm) · Figure 4 (arm-difference by transform space)

---

## Step 4 — Our generator

*Wait for Steps 2–3.*

**Proposal.** Flow-matching backbone plus a time-series editing stage.

**Why the combination is defensible** (to be argued in the paper):

- Flow matching samples iteratively but in far fewer steps than diffusion, which is what
  makes a per-baseline training-length curve affordable at all.
- Editing is **post hoc**, so its quality cost is directly measurable and controllable.
  DP-SGD-style defences inject noise during training, where the quality cost is neither.

**Required comparisons.**

1. **Against early stopping.** 30,000 steps already pushes controls to 2/13. The editing
   module has to beat *training less*, or the obvious question goes unanswered.
2. **Against DP-SGD**, as the standard defence baseline.
3. **Off the frontier, not along it** — this is the whole claim.

---

## Limitations to state explicitly

1. **One replicate.** The 13 targets share a single base model, so their gaps are
   correlated and every p-value here is a **screen, not a test**. Three replicates are
   required before submission. This is the most serious outstanding gap.
2. **The main campaign trained 3.3× past peak contrast.** Under the framing adopted here
   this becomes a finding rather than a defect, but it must be stated as the reason the
   headline cell numbers are lower than the peak.
3. **Capacity is not matched between window lengths.** Counting checkpoints: `d1_c1` has
   10,511,511 parameters at T=288 and `d7_c1` has 12,282,711 at T=2016 — 7× the sequence
   for 16.9% more parameters, all of it in two positional embeddings and the input/output
   projections, with a byte-identical backbone. Per timestep that is 36,498 against
   6,093. The window-length effect therefore has a capacity effect inside it and the two
   cannot be separated without varying the architecture, which introduces a third
   variable.

---

## Execution order

| # | task | cost | blocks |
|---|---|---|---|
| 1 | Add checkpoint + resample to five generators | ~2 days engineering | Step 2 |
| 2 | One base model per baseline: measure cost and quality | ~1 day | may remove baselines from Step 2 |
| 3 | **Step 3c localisation** | ~1 CPU-hour per cell, no training | Step 4's design |
| 4 | Step 2 full matrix | up to 28 days, less after (2) | Figure 2 |
| 5 | Replicates 2 and 3 | parallel with (4) | every p-value |

Task 3 needs no new compute and directly determines what the editing module in Step 4
operates on. It should run first.

---

## Tips for reading and maintaining this document

1. **It is a living document.** Every number in it is read from an artefact under
   `results/` and checked against that artefact before the document is committed. Nothing
   here is typed from memory or carried over from an earlier draft.

2. **Not-yet-measured is written as not-yet-measured.** Where a quantity has not been
   computed the row says *wait* or *training*, never an estimate. Cost projections are
   the one exception and are labelled as extrapolations from measured per-model times.

3. **Read the discriminator as a maximum over restarts, with its spread.** A single
   discriminative accuracy is one draw of a lower bound, and the draws are bimodal
   wherever a separable feature exists — the classifier either finds it or does not. A
   tight cluster at 0.53 and a cluster spanning 0.50–0.84 can have similar medians and
   mean opposite things. `docs/PITFALLS.md` §16 and §18 record two occasions when a
   single-run reading gave the wrong answer, once in the wrong direction entirely.

4. **Every p-value here is a screen, not a test, until replicates exist.** The 13 targets
   in a cell share one base model, so their gaps are correlated and the effective
   independent unit is the replicate, not the target. This is stated in Limitations and
   applies to every p in every table above.

5. **A statistic chosen after seeing the result is not a measurement.** `min × mean` was
   frozen in `configs/experiment.yaml` before any result existed. Where a non-frozen grid
   cell reads higher — `mean × mean` gives 0.651 against the frozen 0.562 on `d1_c1` —
   the frozen number is the one reported.

6. **Quality is a gate, not a co-equal metric.** An MIA number from a generator whose
   samples fail the quality gate answers nothing, so quality is established first and the
   leakage numbers are read only for models that pass. This is why TimeVAE has no MIA
   entry.
