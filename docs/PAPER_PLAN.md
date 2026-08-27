# Paper plan — membership-inference risk in CGM synthesis, and a generator that reduces it

## Thesis

Synthetic CGM data carries membership-inference risk. We quantify that risk across
generative baselines, localise where it comes from, and design a generator — flow
matching with a time-series editing stage — that lowers it **without paying for it in
generation quality**.

---

## Step 1 — Establish that the risk exists

Show that you can tell, from a generator's released synthetic data, whether a particular
patient was in its training set. Everything downstream depends on this, so the design is
built to close the alternative explanations a reviewer would reach for:

- **Both arms are compared against the same reference model.** Any quirk of that
  particular training run then affects both arms equally and drops out of the comparison.
- **The attack statistic was fixed before we saw any result.** Several reasonable choices
  exist and they do not agree; picking the flattering one afterwards would not be a
  measurement.
- **Each control patient has exactly as much data as the outlier they are matched to.**
  So "patients with longer records are easier to identify" cannot explain the result.
- **Outliers and controls are screened equally strictly.** Outliers are the patients
  every detector flagged under every random seed; controls are drawn only from patients
  no detector ever flagged. Screening one side loosely would put outliers into the
  control group and hide the effect.

### Status: 3 of 4 conditions complete, the fourth finishes 08-30

Four conditions, identical in every respect except the two variables under study — how
long a window the generator produces, and how many signals it produces at once.

| condition | window | signals | **how identifiable is a typical outlier** | **a typical normal patient** | **outliers at risk** | **normals at risk** | **do outliers stand out** | **generation quality** |
|---|---|---|---|---|---|---|---|---|
| `d1_c1` | 1 day | glucose | 0.80 | 0.64 | 11 of 13 | **13 of 13** | 0.56 | 0.095 |
| `d1_c2` | 1 day | + insulin | 0.60 | 0.52 | 10 of 13 | **5 of 13** | **0.70** | 0.160 |
| `d7_c1` | 7 days | glucose | 0.76 | 0.68 | 13 of 13 | **12 of 13** | 0.63 | 0.331 |
| `d7_c2` | 7 days | + insulin | *training* | | | | | |

**How to read the columns.**
*Identifiability* is per patient: the chance an attacker correctly decides whether that
patient was used, where 0.5 is a coin flip. The two medians are the typical outlier and
the typical normal patient. *At risk* counts how many of the 13 in each group exceed 0.55,
i.e. are identifiable at all. *Do outliers stand out* compares the two groups as a whole —
0.5 means outliers and normals are equally exposed, and it is the number the study's
original hypothesis was about. *Generation quality* is Context-FID, lower is better; it is
reported for the reference model of each condition and is a gate, not a trade-off (Tip 5).

### Finding

**The risk is real, and at this training length it is not selective.** Read the two
"at risk" columns: in the glucose-only conditions almost every *normal* patient is
identifiable too — 13 of 13 and 12 of 13. At least one patient is identified with
certainty. Adding the second signal is the one thing that changes this: it pushes normal
patients back down to 5 of 13 while leaving outliers where they were, which is why that
condition is the only one where outliers clearly stand out.

The hypothesis we began with — that outliers are the ones at risk — does **not** hold at
the training length we used. Step 3b explains why, and the explanation turns out to be
the most interesting result so far.

### Deliverables

Table 1 (design) · Table 2 (four conditions) · Figure 1 (per-patient risk, outliers vs
controls)

---

## Step 2 — The risk–quality frontier across baselines

Six generators, each on all four conditions, each measured across its whole training
trajectory rather than at one endpoint, plotted as generation quality against privacy
risk. Two reasons for that shape, and the second is not obvious:

1. **A frontier is what makes the contribution legible.** "Better privacy" and "same
   quality" are one claim, not two. Against a plain ranking a reviewer can answer that
   our model is simply worse and therefore safer — and be right, because a ranking cannot
   distinguish moving *off* the frontier from sliding *along* it. That distinction is the
   contribution.
2. **Single points are not comparable across baselines.** Step 3b shows risk rises and
   then falls with training length. Comparing models at one fixed budget compares each at
   an arbitrary point on its own trajectory. The comparable quantity is each model's
   **peak** risk, which requires the trajectory.

### Status

| generator | state |
|---|---|
| DiM-TS | all four conditions plus a full training trajectory — **complete** |
| copy-paste | positive control: a generator that simply replays training data |
| TimeVAE | **fails the quality gate.** Its samples are easily told from real data, and its Context-FID is 9× DiM-TS. Ruled out after one model |
| PaD-TS, Diffusion-TS, DiffWave, FourierDiffusion, IG-FM | not started |

### Two prerequisites

1. **Only DiM-TS can currently produce samples from a partly-trained model.** The other
   five would need that capability added before any training trajectory can be measured.
2. **Training cost varies about thirtyfold between architectures** — nine minutes per
   model for the fastest, nearly five hours for DiM-TS. Each baseline therefore gets one
   model measured before we commit to the full set. That is how TimeVAE was ruled out in
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
conditions, with different signals and different random seeds. Across all 26 tested
patients the two rankings agree strongly (Spearman ρ = 0.61, p = 0.001).

### 3b — What amplifies it — **complete**

Separate the effects of training length, window length and number of signals. A defence
has to act where the risk is produced, and if training length dominates then simply
training less is the cheapest possible defence — one our model has to beat.

**Result. Risk rises and then collapses as training continues.**

| training | typical outlier | typical normal | outliers at risk | normals at risk | do outliers stand out | generation quality |
|---|---|---|---|---|---|---|
| shortest | 0.56 | 0.55 | 8 of 13 | 7 of 13 | 0.63 | 0.060 |
| **early-middle** | 0.63 | 0.52 | 9 of 13 | **2 of 13** | **0.84** | 0.061 |
| middle | 0.63 | 0.52 | 10 of 13 | 2 of 13 | 0.76 | 0.065 |
| long | 0.72 | 0.52 | 9 of 13 | 4 of 13 | 0.68 | 0.072 |
| longer | 0.69 | 0.57 | 11 of 13 | 10 of 13 | 0.55 | 0.074 |
| **full (what we ran)** | 0.68 | 0.63 | 9 of 13 | **13 of 13** | **0.51** | 0.083 |

*Same columns as Step 1. The shortest row is below convergence — its generator has not
finished learning — and is read as such.*

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
day, and what kind of structure. This is the bridge to Step 4 — the proposed defence
edits the time series, and **editing needs to know what to edit**. If the leak is a
characteristic overnight low, the module edits that; if it is overall variability, that
is a different operation. Without 3c the editing module is guesswork.

Two analyses, both reusing the attack from Step 1 rather than introducing a new one, so
that what we localise corresponds to the number we report:

- **Which hours.** Split the reported distance across the time axis, giving a
  curve over the day whose total is exactly the quantity reported in Step 1.
- **What kind of structure.** Recompute the same attack after transformations that each
  destroy one kind of information — timing, absolute level, fine detail, variability. If
  the leak survives when all timing is destroyed, it is distributional and editing must
  change the distribution; if it needs the original timing, it is tied to specific events
  and editing can be local. Each outcome maps to a different editing operation.

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
3. **Model capacity is not matched across window lengths.** The seven-day generator has
   seven times the sequence to model and only marginally more capacity to do it with, so
   the window-length effect has a capacity effect inside it. Separating them would require
   changing the architecture, which introduces a third variable.

---

## Execution order

| # | task | cost | blocks |
|---|---|---|---|
| 1 | Give five generators the ability to sample from a partly-trained model | ~2 days | Step 2 |
| 2 | One model per baseline: measure cost and quality | ~1 day | may remove baselines from Step 2 |
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

3. **Every p-value here is a screen, not a test**, until we have replicates. See
   Limitation 1 — this applies to every p in every table above.

4. **A statistic chosen after seeing the result is not a measurement.** The attack was
   fixed in advance. Where a different reasonable choice would give a higher number, the
   pre-registered one is what we report.

5. **Quality is a gate, not a second metric to trade against.** A privacy number from a
   generator whose output is obviously fake answers nothing, so quality is established
   first and risk is read only for models that pass. This is why TimeVAE has no risk
   number.

6. **Judging whether synthetic data is distinguishable takes several attempts, not one.**
   A classifier trained once may or may not find the tell; we train it repeatedly and read
   the best attempt and the spread. A single run has twice given us the wrong answer, once
   in the wrong direction entirely.
