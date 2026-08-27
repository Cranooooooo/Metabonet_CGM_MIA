# Paper plan — membership-inference risk in CGM synthesis, and a generator that reduces it

## Thesis

Synthetic continuous glucose monitoring (CGM) data carries membership-inference risk. We quantify that risk across
generative baselines, localise where it comes from, and design a generator — flow
matching with a time-series editing stage — that lowers it **without paying for it in
generation quality**.

---

## Step 1 — Establish that the risk exists

Show that membership in the training set can be inferred from a generator's released
synthetic data. Everything downstream depends on this, so the design is
built to close the alternative explanations a reviewer would reach for:

- **Both groups are compared against the same reference model.** For every patient we
  train one generator that includes them and compare it with a single shared generator
  trained without any of the tested patients. Because that one model serves as the
  non-member side for outliers and controls alike, any idiosyncrasy of that training run
  affects both groups equally and cancels.
- **The attack statistic was pre-registered.** Several reasonable choices exist and they
  disagree.
- **Controls are matched to outliers on record length**, so record length cannot
  explain the result.
- **Outliers and controls are screened equally strictly.** Thirteen outlier-detection
  methods score every patient on glucose distribution, waveform shape and learned
  representation. An **outlier** is a patient that the consensus of those methods flags
  under all four random seeds; a **control** (labelled *normal* in the tables) is drawn
  only from patients that no method flagged under any seed. Screening one side loosely
  would put outliers into the control group and hide the effect.

### Status: 3 of 4 conditions complete, the fourth finishes 08-30

Four conditions, identical except for the two variables under study: **window length** and
**number of channels**.

| condition | window length | channels | median AUC, outliers | median AUC, normals | outliers with AUC > 0.55 | normals with AUC > 0.55 | arm AUC | Context-FID |
|---|---|---|---|---|---|---|---|---|
| `d1_c1` | 1 day | 1 (glucose) | 0.800 | 0.635 | 11 of 13 | **13 of 13** | 0.562 | 0.095 |
| `d1_c2` | 1 day | 2 (+ insulin) | 0.600 | 0.520 | 10 of 13 | **5 of 13** | **0.698** | 0.160 |
| `d7_c1` | 7 days | 1 (glucose) | 0.760 | 0.679 | 13 of 13 | **12 of 13** | 0.627 | 0.331 |
| `d7_c2` | 7 days | 2 (+ insulin) | *training* | | | | | |

**Columns.** *Median AUC* is per patient — the chance an attacker correctly decides
whether that patient was used; 0.5 is a coin flip. The attack scores a patient by the
distance from their real records to the nearest synthetic sample, with and without them in
the training set. *> 0.55* counts how many of the 13 in
each group are identifiable at all. *Arm AUC* compares the two groups as a whole and is
what the study's original hypothesis was about; 0.5 means outliers and normals are equally
exposed. *Context-FID* is generation quality, lower is better, measured on each
condition's reference model. For scale: the values below run 0.095 to 0.331, and the one
generator we have rejected scored 0.856. It is a gate rather than a trade-off (Tip 5).

### Finding

**The risk is real, and at this training length it is not selective.** Read the two
`AUC > 0.55` columns: in the single-channel conditions almost every *normal* patient is
identifiable too — 13 of 13 and 12 of 13, and one patient reaches AUC 1.00. Adding the insulin channel is the only change that affects this: it pushes normals back
down to 5 of 13 while leaving outliers where they were, which is why that condition is the
only one where outliers clearly stand out.

The hypothesis we began with — that outliers are the ones at risk — does **not** hold at
the training length we used. Step 3b explains why, and the explanation turns out to be
the most interesting result so far.

### Deliverables

Table 1 (design) · Table 2 (four conditions) · Figure 1 (per-patient risk, outliers vs
controls)

---

## Step 2 — The risk–quality frontier across baselines

Six generators, each on all four conditions, each measured across its whole training
trajectory rather than at a single endpoint, plotted as generation quality against privacy
risk. Two reasons for that shape, and the second is not obvious:

1. **A ranking cannot tell a better model from a worse one.** Plot every generator as a
   point — quality on one axis, privacy risk on the other — and the best points trace a
   Pareto frontier: the lowest risk currently achievable at each level of quality.
   Reducing risk by giving up quality moves *along* that frontier, and anything does it:
   train fewer steps, shrink the model, add noise. Reducing risk *at the same quality*
   moves the frontier itself. A ranking reports one number and so cannot distinguish
   them, which leaves a reviewer free to answer that our model is simply worse — and
   they would have no way to be shown otherwise. Moving the frontier is the claim.
2. **Single points are not comparable across baselines.** Step 3b shows risk rises and
   then falls with training length. Comparing models at one fixed budget compares each at
   an arbitrary point on its own trajectory. The comparable quantity is each model's
   **peak** risk, which requires the trajectory.

### Status

| generator | state |
|---|---|
| DiM-TS | all four conditions plus a full training trajectory — **complete** |
| copy-paste | positive control: a generator that replays training data verbatim — **complete** |
| TimeVAE | **fails the quality gate.** Its samples are trivially distinguishable from real data and its Context-FID is 0.856, 9× that of DiM-TS. Ruled out after a single-model pilot |
| PaD-TS, Diffusion-TS, DiffWave, FourierDiffusion, IG-FM | not started |

### Two prerequisites

1. **Only DiM-TS can sample from intermediate training checkpoints.** The other five need
   that before any training trajectory can be measured.
2. **Training cost varies about thirtyfold between architectures** — nine minutes per
   model for the fastest, nearly five hours for DiM-TS. Each baseline therefore gets a
   single-model pilot before we commit to the full set. That is how TimeVAE was ruled out in
   minutes rather than days.

A baseline whose first model fails the quality gate stops there. That is itself a result
— *this architecture cannot produce usable CGM data at this scale* — not a gap.

### Cost

Extrapolating from measured times, the full plan is roughly **11,000 GPU-hours**, about a
month on the hardware we have. This is an upper bound set by the slowest architecture;
the single-model probes above exist to bring it down.

### Deliverables

Figure 2 (frontier: quality against peak risk, one trajectory per baseline) · Table 3
(where each baseline peaks, and how high)

---

## Step 3 — Localise the leak

### 3a — Which patients — **complete**

Identify which individuals are repeatedly exposed, and show it is not chance. This is the
step from *risk exists* to *risk is predictable*: if a different set of patients leaked
each time, a defence would have nothing to act on.

**Result.** The **same six patients** are the most exposed in two independently trained
conditions, with different channel counts and different random seeds. Across all 26 tested
patients the two rankings agree strongly (Spearman ρ = 0.61, p = 0.001).

### 3b — What amplifies it — **complete**

Separate the effects of training length, window length and number of channels. A defence
has to act where the risk is produced, and if training length dominates then simply
training less is the cheapest possible defence — one our model has to beat.

**Result. Risk rises and then collapses as training continues.**

| training steps | epochs | median AUC, outliers | median AUC, normals | outliers with AUC > 0.55 | normals with AUC > 0.55 | arm AUC | Context-FID |
|---|---|---|---|---|---|---|---|
| 20,000 | 223 | 0.562 | 0.554 | 8 of 13 | 7 of 13 | 0.633 | 0.060 |
| **30,000** | **334** | 0.625 | 0.517 | 9 of 13 | **2 of 13** | **0.840** | 0.061 |
| 40,000 | 446 | 0.625 | 0.520 | 10 of 13 | 2 of 13 | 0.757 | 0.065 |
| 60,000 | 669 | 0.724 | 0.520 | 9 of 13 | 4 of 13 | 0.680 | 0.072 |
| 80,000 | 892 | 0.688 | 0.571 | 11 of 13 | 10 of 13 | 0.550 | 0.074 |
| **100,000 (what we ran)** | **1,115** | 0.679 | 0.633 | 9 of 13 | **13 of 13** | **0.515** | 0.083 |

*Same columns as Step 1. The 20,000-step row is below convergence and is excluded from
the trend.*

**The outlier group barely changes. The normal group goes from 2 of 13 exposed to all
13.** Over-training does not expose outliers further — it starts exposing everyone, and
what disappears is the *contrast* between the groups.

Two consequences:

- **Our main experiment ran roughly three times past the point of maximum contrast.** The
  headline numbers in Step 1 therefore understate how distinguishable outliers can be.
- **Training less preserves the contrast but not the absolute risk** — even at the peak,
  9 of 13 outliers are still identifiable. That gap is where our model has to live.

Generation quality degrades over the same range, so over-training is not a
privacy-for-quality trade: it costs both.

### 3c — What about a patient leaks — **designed, not started**

For the patients identified in 3a, determine *what* was memorised: which hours of the
day, and what kind of structure. The Step 4 defence edits the time series, so it needs
a target: a characteristic overnight low and elevated overall variability call for
different editing operations.

Two analyses, both reusing the attack from Step 1 rather than introducing a new one, so
that what we localise corresponds to the number we report:

- **Which hours.** Decompose the attack distance over hour of day, giving a curve whose
  total is the Step 1 statistic exactly.
- **What kind of structure.** Recompute the same attack after transformations that each
  destroy one kind of information — timing, absolute level, fine detail, variability. If
  the leak survives when all timing is destroyed, it is distributional and editing must
  change the distribution; if it needs the original timing, it is tied to specific events
  and editing can be local.

Both are plotted **alongside the control patients**, because any distance measure rises
where glucose is most variable — after meals, for instance — whether or not anyone is
being identified. Only where outliers exceed controls is there leakage.

This needs no new training, only re-analysis of data we already have: about an hour.

### Deliverables

Figure 3 (hour-of-day leakage for the six most exposed patients, against controls) ·
Figure 4 (which kind of information carries the leak)

---

## Step 4 — Our generator

*Waiting on Steps 2 and 3.*

**Proposal.** A flow-matching backbone with a time-series editing stage.

**Why this combination.** Flow matching generates in far fewer steps than diffusion, which
is what makes measuring a whole training trajectory affordable for every baseline.
Editing is applied after generation, so its cost in quality is directly measurable and
controllable — unlike defences that inject noise during training, where it is neither.

**Required comparisons.**

1. **Against simply training less**, which Step 3b shows is already an effective defence.
2. **Against DP-SGD**, the standard privacy baseline.
3. **Off the frontier, not along it** — this is the whole claim.

---

## Limitations to state explicitly

1. **One replicate.** All 13 outliers in a condition share a single reference model, so
   their measurements are correlated and the effective sample size is one, not 13. Every
   p-value above is therefore a **screen, not a test**. Three replicates are required
   before submission. This is the most serious outstanding gap.
2. **The main experiment trained past the point of maximum contrast.** Under the framing
   we have adopted this becomes a finding rather than a defect, but it must be stated as
   the reason the headline numbers are lower than the peak.
3. **Window length is confounded with model capacity.** The seven-day generator has 7×
   the sequence length and 17% more parameters, so the window-length effect includes a
   capacity effect. Separating them would require changing the architecture, which
   introduces a third variable.

---

## Execution order

| # | task | cost | unblocks |
|---|---|---|---|
| 1 | Add checkpoint sampling to five generators | ~2 days engineering | Step 2 |
| 2 | Single-model pilot per baseline: cost and quality | ~1 day | may remove baselines from Step 2 |
| 3 | **Step 3c localisation** | ~1 hour, no training | Step 4's design |
| 4 | Step 2 full matrix | up to a month, less after (2) | Figure 2 |
| 5 | Replicates 2 and 3 | in parallel with (4) | every p-value |

Task 3 needs no new compute and directly determines what the editing module in Step 4
operates on. It should run first.

---

## Tips for reading and maintaining this document

1. **It is a living document.** Every number in it is read from a result file and checked
   against that file before the document is committed — nothing is typed from memory.

2. **Not-yet-measured is written as not-yet-measured**, never filled in with an estimate.
   Cost projections are the one exception and are labelled as extrapolations.

3. **Every p-value here is a screen, not a test.** See Limitation 1.

4. **A statistic chosen after seeing the result is not a measurement.** The attack was
   fixed in advance. Where a different reasonable choice would give a higher number, the
   pre-registered one is what we report.

5. **Quality is a gate, not a second metric to trade against.** A privacy number from a
   generator whose output is obviously fake answers nothing, so quality is established
   first and risk is read only for models that pass. This is why TimeVAE has no risk
   number.

6. **The discriminative score is read over repeated fits, not one.** A single classifier
   fit may miss the discrepancy; we refit repeatedly and report the best fit and the
   spread. A single fit has twice given the wrong answer, once with the sign reversed.
