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

### Status: COMPLETE — all four conditions, the last landed 2026-08-30

Four conditions, identical except for the two variables under study: **window length** and
**number of channels**.

| condition | window length | channels | median AUC, outliers | median AUC, normals | outliers with AUC > 0.55 | normals with AUC > 0.55 | arm AUC | Context-FID |
|---|---|---|---|---|---|---|---|---|
| `d1_c1` | 1 day | 1 (glucose) | 0.800 | 0.635 | 11 of 13 | **13 of 13** | 0.562 | 0.095 |
| `d1_c2` | 1 day | 2 (+ insulin) | 0.600 | 0.520 | 10 of 13 | **5 of 13** | **0.698** | 0.160 |
| `d7_c1` | 7 days | 1 (glucose) | 0.760 | 0.679 | 13 of 13 | **12 of 13** | 0.627 | 0.331 |
| `d7_c2` | 7 days | 2 (+ insulin) | 0.642 | **0.488** | 12 of 13 | **1 of 13** | **0.935** | 0.284 |

**Columns.** *Median AUC* is per patient — the chance an attacker correctly decides
whether that patient was used; 0.5 is a coin flip. The attack scores a patient by the
distance from their real records to the nearest synthetic sample, with and without them in
the training set. *AUC > 0.55* counts how many of the 13 in each
group are identifiable at all. *Arm AUC* compares the two groups as a whole and is
what the study's original hypothesis was about; 0.5 means outliers and normals are equally
exposed. *Context-FID* is generation quality, lower is better, measured on each
condition's reference model. For scale: the values below run 0.095 to 0.331, and the one
generator we have rejected scored 0.856. It is a gate rather than a trade-off (Tip 5).

### Finding — REWRITTEN 2026-08-30 when the fourth condition landed, and it changes the claim

**The risk is real. Whether it is *selective* depends entirely on the channel count, and
with all four cells in hand the pattern has no ambiguity.**

| | normals with AUC > 0.55 | arm AUC | p |
|---|---|---|---|
| glucose only, 1 day | 13 of 13 | 0.562 | 0.304 |
| glucose only, 7 days | 12 of 13 | 0.627 | 0.141 |
| **+ insulin, 1 day** | **5 of 13** | 0.698 | 0.045 |
| **+ insulin, 7 days** | **1 of 13** | **0.935** | **< 0.001** |

**With glucose alone, everyone leaks and the arms do not separate. Add the insulin
channel and the ordinary patients go quiet while the outliers stay exposed** — 1 of 13
at seven days, with the normal arm's median AUC at **0.488**, below chance. `d7_c2` is
the only cell whose arm separation is unambiguous, and it is the strongest by a wide
margin.

**So the study's original hypothesis was not wrong — it was missing a precondition.**
"Outliers are the ones at risk" holds *only when the generator sees insulin as well as
glucose*. On glucose alone it fails, and it fails because everyone is identifiable, not
because outliers are safe.

**And the quality cost we attributed to the second channel does not survive the seven-day
cell.** At one day the insulin channel cost Context-FID 0.095 → 0.160. At seven days
`d7_c2` reads **0.284 against `d7_c1`'s 0.331** — the two-channel cell is *better*. The
"insulin buys contrast at the price of fidelity" framing holds at one day only and must
not be stated generally.

**Every p-value in the table above is still a screen, not a test** (Limitation 1): the 13
outliers in a cell share one base model, so their gaps are correlated and the effective
sample size is one. `d7_c2`'s p < 0.001 is the most suggestive number in the study and it
is exactly the one that most needs the replicates.

---

**The original reading, kept because Step 3b explains it and it is still true of the
single-channel cells.**

**At this training length the risk is not selective in the glucose-only conditions.** Read the two
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

1. **Privacy and generation quality trade off against each other, and the contribution is
   to break that trade-off.** Plotting every generator as a point — quality on one axis,
   privacy risk on the other — should trace a curve: better privacy currently costs
   quality. Buying privacy with quality moves *along* the curve, and anything does it —
   train fewer steps, shrink the model, add noise. Achieving lower risk *at the same
   quality* moves the curve itself, and that is the claim. A ranking reports one number
   and cannot distinguish the two, so it leaves a reviewer free to answer that our model
   is simply worse.

   Note that the curve itself is a hypothesis at this point: we have one generator
   measured, so Step 2's first job is to establish that the trade-off exists and is
   monotone before anything can be said about breaking it. If it turns out not to exist —
   if some architecture is both better and safer — that is a more interesting result,
   because it would mean existing methods are merely unoptimised rather than up against
   something fundamental.
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

**Result — CORRECTED 2026-08-30. The number was right; the sentence describing it was
wrong, and the full picture is sharper than the original claim.**

The ρ = 0.61 came from **`d1_c1` against `d7_c1`** — two conditions that differ in *window
length*, not in channel count. Both are single-channel. The earlier text said "different
channel counts and different random seeds", which is not what was compared. With all four
cells finished the whole matrix can be reported, and it says something the single pair did
not (26 shared targets, Spearman on the per-subject AUC, 10,000-permutation test):

| pair | differs in | ρ | p |
|---|---|---|---|
| `d1_c2` vs `d7_c2` | window length | **+0.651** | **0.0003** |
| `d1_c1` vs `d7_c1` | window length | **+0.611** | **0.0012** |
| `d7_c1` vs `d7_c2` | channels | +0.416 | 0.038 |
| `d1_c1` vs `d7_c2` | both | +0.326 | 0.109 |
| `d1_c1` vs `d1_c2` | channels | +0.227 | 0.270 |
| `d1_c2` vs `d7_c1` | both | +0.084 | 0.685 |

**Who leaks is reproducible across window length and is not reproducible across channel
count.** The two same-channel pairs are the only ones that clear ρ = 0.6, and the two
cross-channel pairs at matched window length sit at 0.416 and 0.227.

That is still enough for the claim this step exists to make — leakage is a stable property
of individuals, so a defence has something to act on — but it constrains it: **the stable
thing is "who leaks given a fixed set of channels", not "who leaks" in general.** Adding the
insulin channel does not merely quiet the ordinary patients (Step 1); it substantially
reshuffles which individuals are exposed at all. A per-patient defence calibrated on one
channel configuration should not be assumed to transfer to another.

**Provenance note, because this breaks Tip 1.** The 0.61 sat in this document with no result
file behind it and no script that produced it; it was found again only by recomputing all
six pairs from `results/matrix/subject_auc/*/per_subject.csv`. Any remaining number here
without a path should be treated the same way until it is re-derived.

### 3b — What amplifies it — **complete**

Separate the effects of training length, window length and number of channels. A defence
has to act where the risk is produced, and if training length dominates then simply
training less is the cheapest possible defence — one our model has to beat.

**Result. Two different quantities move in opposite directions, and conflating them is
easy.** *Risk* is whether an individual can be identified — the per-patient AUC.
*Contrast* is whether outliers are identified more than normals — the arm AUC, which is
what the study's original hypothesis was about. Training more raises risk and destroys
contrast.

| training steps | epochs | **RISK**: patients with AUC > 0.55 | **RISK**: highest AUC | **CONTRAST**: arm AUC | Context-FID |
|---|---|---|---|---|---|
| 20,000 | 223 | 15 of 26 | 0.688 | 0.633 | 0.060 |
| 30,000 | 334 | **11 of 26** | 0.747 | **0.840** | 0.061 |
| 40,000 | 446 | 12 of 26 | 0.800 | 0.757 | 0.065 |
| 60,000 | 669 | 13 of 26 | 0.960 | 0.680 | 0.072 |
| 80,000 | 892 | 21 of 26 | 0.940 | 0.550 | 0.074 |
| **100,000 (what we ran)** | **1,115** | **22 of 26** | 0.960 | **0.515** | 0.083 |

*⚠️ **The claim that the 20,000-step row is "below convergence" has no result file behind
it and the quality data contradicts it** (flagged 2026-08-30). On Context-FID, 20k is the
**best** of the six milestones in two of three cells — 0.0602 on `d1_c1` and 0.1306 on
`d7_c1` — which is the opposite of what "below convergence" predicts. Only `d1_c2` has an
interior minimum (40k, 0.0992). Either the footnote is judging convergence by something
other than generation quality, or it is wrong. `scripts/pbs/L4_base_earlystop.pbs` adds
the 10k and 50k milestones on `base` to make the shape below 20k visible for the first
time. Until that lands, do not exclude the 20k row on this footnote's authority.*

*The 20,000-step row was excluded from the trend on the reading above. Breaking the
26 patients into the two groups shows why contrast collapses:*

| training steps | outliers with AUC > 0.55 | normals with AUC > 0.55 |
|---|---|---|
| 30,000 | 9 of 13 | **2 of 13** |
| 100,000 | 9 of 13 | **13 of 13** |

**The outlier group does not change. The normal group goes from 2 of 13 to 13 of 13.**
Over-training does not expose outliers further — it starts exposing everyone, which is
what destroys the contrast while raising the risk.

Three consequences:

- **Our main experiment ran roughly three times past the point of maximum contrast**, so
  the Step 1 numbers understate how distinguishable outliers can be.
- **Between 30,000 and 100,000 steps, training longer is worse on every axis measured** —
  more patients at risk, less contrast, and worse generation quality. There is no
  trade-off in this range, only waste.
- **Early stopping is therefore a real defence**, halving the number of patients at risk.
  Our model has to beat it, not merely beat the over-trained endpoint.


> **⛔ SUPERSEDED IN PART, 2026-08-31 — the transform table below answers a different
> question than the one this step is named for, and the right measurement now exists.**
>
> The table reports **arm AUC**: how separable the outlier arm's gaps are from the control
> arm's. That is *"what distinguishes outliers from ordinary patients"*, not *"what makes
> an individual identifiable"* — and only the second one tells a defence what to act on.
> The two are not the same, and reading the first as the second is the error.
>
> The per-arm numbers make it concrete: sorting **lowers** the outlier arm's own gap
> (+36.3% → +20.9% of `d_in`) and lowers the control arm's more (+11.9% → +1.7%). Sorting
> does not reveal more leakage; it strips a component that was inflating both arms — the
> generic agreement on the shape of a day — which is why in raw space everyone appears to
> leak.
>
> **The right measurement is `scripts/leak_locus.py`**, which recomputes each subject's own
> identifiability (the same unpaired rank AUC `subject_auc.py` reports) under each
> transform, with a shuffle floor control. Its answer is different and is below.

### 3c-bis — What actually carries the leak — measured 2026-08-31

Per-subject identifiability, **median over 26 subjects**, 0.5 = cannot be told apart. The
median is used rather than a count because the count's floor is 7–12 of 26
(`PITFALLS.md` §20); the median's floor is 0.503–0.516 and separates cleanly.

| destroyed | d1_c1 | d1_c2 | d7_c1 | d7_c2 |
|---|---|---|---|---|
| nothing | **0.653** | **0.575** | **0.691** | **0.542** |
| absolute level | 0.549 | 0.514 | 0.543 | 0.504 |
| all temporal ordering | 0.536 | 0.519 | 0.600 | 0.600 |
| sub-hour detail | 0.692 | 0.580 | 0.724 | 0.548 |
| level + scale | 0.560 | 0.560 | 0.651 | 0.532 |
| **level AND ordering** | **0.487** | **0.490** | **0.506** | **0.520** |
| **level + scale AND ordering** | **0.520** | **0.507** | **0.508** | **0.525** |
| *shuffle floor* | *0.503* | *0.503* | *0.509* | *0.516* |

**Three findings, and the first is the design input.**

1. **The leak is carried jointly by the absolute level and the temporal ordering.
   Destroying both together lands on the floor in all four cells.** Destroying either one
   alone removes only about half of it.
2. **Sub-hour detail carries nothing** — destroying it leaves identifiability unchanged or
   slightly higher (0.653 → 0.692). A defence should not spend distortion there.
3. **No single-coordinate edit can be sufficient.** This is measured, not argued: level
   alone leaves 0.549, ordering alone leaves 0.536, both together reach 0.487.

**Consequence for Step 4.** The mechanism must disturb **both** the level and the ordering.
The earlier design note — that the defence should edit the value distribution (the quantile
function) — targets only one of the two: sorting *preserves* the value distribution exactly
and still removes half the signal, so the quantile function cannot be the whole story.

### 3c — What about a patient leaks — **complete 2026-08-27**

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

**Result — and it is the most consequential of the three localisation findings.** Both
analyses ran 2026-08-27 (`results/matrix/localise/{d1_c1,d1_c2,d7_c1}/`); this section
carried a stale "not started" label until 2026-08-30 because the document was committed two
minutes before the jobs landed.

**What kind of structure.** The attack recomputed after five transforms, each destroying one
kind of information. Arm AUC, all *sorted* values at p < 0.01:

| transform | destroys | d1_c1 | d1_c2 | d7_c1 |
|---|---|---|---|---|
| raw | nothing | 0.562 | 0.698 | 0.627 |
| **sorted** | **all timing** | **0.828** | **0.781** | **0.905** |
| diff | absolute level | 0.598 | 0.533 | 0.704 |
| hourly | detail below one hour | 0.538 | 0.746 | 0.627 |
| zscore | level and scale | 0.527 | 0.657 | 0.467 |

**Destroying every trace of timing does not remove the leak — it strengthens it, by up to
3.2×.** Destroying level and scale is what removes it. So what identifies a patient is the
*distribution of their glucose values*, not when those values occur; formally, the leaking
quantity is a Wasserstein-2 distance between per-window value distributions. **A defence must
edit the value axis, and it may leave timing untouched.**

**Which hours.** Leakage decomposed over hour of day, outlier arm against control arm, since
any distance measure rises where glucose is variable whether or not anyone is identified:

| | 07:00 | 08:00 | **09:00** | 10:00 | whole day |
|---|---|---|---|---|---|
| d1_c1, outlier ÷ control | 5.0× | 4.8× | **16.3×** | 13.3× | 3.4× |
| d1_c2, outlier ÷ control | — | 17.7× | **20.5×** | 3.6× | 5.4× |
| d7_c1, outlier ÷ control | 4.6× | 6.7× | 3.6× | 3.9× | 3.9× |

Read the *ratio*, never the outlier curve's own peak: on d1_c1 that peak is at 15:00, where
controls are high too and the ratio is only 3.4×. Hours whose control value is near zero or
negative (d1_c2 at 00, 02, 07, 17) are unstable and are not quoted.

**The two findings reconcile, and the Method section must say how, because a reader with both
figures will otherwise think they disagree.** The high tail of a patient's value distribution
is realised in the morning; **it is the values that are memorised, and the hour is merely
where they happen to occur.** Hour-of-day is the symptom, the value distribution is the
cause — which is why Figure 3 is evidence and Figure 4 is the design input, and why no
defence should act on the hour. Corroborated: cells selected by raw-space (time-domain)
contribution and by quantile-space contribution overlap at a Jaccard lift of only 1.4–1.8×
over chance (`MODEL_DESIGN.md` §2.7a).

### Deliverables

Figure 3 (hour-of-day leakage for the six most exposed patients, against controls) ·
Figure 4 (which kind of information carries the leak)

---

## Step 4 — Our generator

### Status 2026-08-30: the backbone failed its pilot, and the defence's shape is now measured rather than proposed

Six jobs ran on 08-29/30. `MODEL_DESIGN.md` §1.8a, §5a and §2.7 carry the detail; this is
what changes for the paper.

**The proposed backbone does not pass the gate.** MAVEN trained on `d1_c1` at the 30k
operating point scores Context-FID **0.2256** against DiM-TS's **0.0611** on the same cell
and the same checkpoint — 3.7×, where Tip 5 makes quality a gate rather than a trade. Running
MAVEN's *own* published ablation ladder made it worse (0.2721), which refutes upstream's
stated hypothesis about the cause of its plateau. One confound is untested: 30k may be too
few steps for MAVEN, and a single 100k run settles it.

**What a release-time defence can do, measured without building one.** The best directed edit
that can exist, applied to released sets already on disk and read through the frozen attack:

| | identifiable, unedited | identifiable, best δ | **most exposed, unedited → worst δ** | arm AUC |
|---|---|---|---|---|
| d1_c1 | 12/26 | 10/26 | 0.747 → **0.760** | 0.840 → 0.882 |
| d1_c2 | 10/26 | 10/26 | 0.880 → **0.920** | 0.822 → 0.740 |
| d7_c1 | 18/26 | 17/26 | 1.000 → 1.000 | 0.959 → **0.982** |

All counts are **two-sided** (`|AUC − 0.5| > 0.05`). A first version of this table used the
one-sided count and read 11 → 5, 9 → 4 and 18 → 9; those numbers score a subject whose gap
has *inverted* as protected, which is precisely the failure §2.6c exists to prevent. The
correction was found by code review, not by us, and it reverses the finding.

Three consequences the paper has to carry:

1. **A global directed release-time edit does not reduce risk.** Past a displacement of
   ~1% of the nearest-neighbour distance it *increases* the number of identifiable
   subjects, and on two of three cells it makes the most exposed subject more identifiable
   than leaving the data alone. The mechanism does not remove information — it flips the
   sign of it, and "systematically farther from the release than a non-member" identifies
   a person exactly as well as "closer". **The claim that this beats early stopping is
   withdrawn**; early stopping alone takes d1_c1 from 22/26 to 11/26, which no edit here
   approaches.
2. **A global knob cannot reach the patients the argument is about.** d7_c1's most exposed
   subject sits at AUC 1.000 at every displacement tested. Per-window margin allocation
   stops being an efficiency refinement and becomes the mechanism.
3. **The defence raises CONTRAST while lowering the one-sided risk count, on all three
   cells.** It drives the control arm's small gaps negative before the outlier arm's, so
   the rank comparison between arms widens. §3b found RISK and CONTRAST moving oppositely
   with training length; they do so under defence too. **Step 4 must state which of the two
   it claims to reduce before it reports either number**, and must report two-sided
   `|AUC − 0.5|` throughout — this table is the demonstration of why.

**One mechanism proposed, and its refutation RETRACTED.** Localising the leak with MAVEN's
own mask head would make the defence intrinsic to the backbone rather than a portable
module. A first measurement read Spearman +0.030 against the coordinate the leak lives in
and was written up as a falsification; a code review on 2026-08-30 withdrew it. The map was
a noisy estimate compared against two exact ones with no attenuation correction, and the
imputation departed from training in four ways that all push the agreement down. Corrected
numbers are pending; nothing here should be cited until then. See `MODEL_DESIGN.md` §2.7.
A membership-discriminator localiser is a different signal and remains untested.

**One new obstacle, not anticipated anywhere.** A released sample's nearest training window
in raw space and in quantile space are the same window **0.7–1.0% of the time**. The two
coordinates are two different attacks that disagree about which records are exposed, so a
defence built in one leaves the other substantially intact.

### The proposal



**Proposal.** A flow-matching backbone with a time-series editing stage.

**Why this combination.** Flow matching generates in far fewer steps than diffusion, which
is what makes measuring a whole training trajectory affordable for every baseline.
Editing is applied after generation, so its cost in quality is directly measurable and
controllable, and it can be tuned after the generator is trained rather than requiring a
retrain per setting.

**Required comparisons.**

1. **Against simply training less**, which Step 3b shows halves the number of patients at
   risk. This is the comparison that matters: it is free, it is obvious, and a reviewer
   will ask for it.
2. **Against the baselines at their own best operating point**, not at a fixed budget —
   otherwise we would be beating models that are themselves over-trained.
3. **At equal generation quality**, so the claim is breaking the trade-off rather than
   moving along it.

---

## Limitations to state explicitly

1. **One replicate.** All 13 outliers in a condition share a single reference model, so
   their measurements are correlated and the effective sample size is one, not 13. Every
   p-value above is therefore a **screen, not a test**. Three replicates are required
   before submission. This is the most serious outstanding gap.
2. **The main experiment trained past the point of maximum contrast.** Under the framing
   we have adopted this becomes a finding rather than a defect, but it must be stated as
   the reason the headline numbers are lower than the peak.
3. **Generation quality is scored against each model's own training set.** This does *not*
   make the two axes redundant — the attack reads a tail property and fidelity reads a bulk
   one (`MODEL_DESIGN.md` §0), and `copy_paste` is separated correctly by the pair (0.070
   quality against 0.816 risk, versus DiM-TS's 0.095 and 0.653). What it does is bias the
   ruler **against any defence that moves released samples away from training windows**,
   which is every mechanism we propose, so "lower risk at equal quality" is judged on ground
   that is not neutral; and it scores reproduction of the training sample rather than capture
   of the population, which is the question a data-release paper actually asks. The fix needs
   no retraining and can be applied at any time, including after the matrix runs.

4. **Window length is confounded with model capacity.** The seven-day generator has 7×
   the sequence length and 17% more parameters, so the window-length effect includes a
   capacity effect. Separating them would require changing the architecture, which
   introduces a third variable.

---

## Training-budget protocol — adopted 2026-08-30

**Early stopping is decided on generation quality, on the `base` model only, and the
chosen step count is then used for every model in that cell.**

1. Train `base` to the full budget, banking a checkpoint every 10k steps.
2. Resample `base` at each checkpoint and read Context-FID. The minimum is the cell's
   training budget.
3. **Train all 27 models of that cell to exactly that step count.**

**Fixing the budget is required for the attack to be valid, not merely cheaper.** §3b
measured a large effect of training length on the membership gap, so if `include_t` and
`base` ran to different lengths the gap would confound membership with training length —
the one confound the whole design exists to exclude. A per-model stopping rule would
reintroduce it.

**Deciding it on `base` also keeps the protocol clear of the targets.** `base` never sees
any of the 26, so nothing about them enters the choice of budget.

**Do NOT stop on the training loss.** Measured on `d7_c2` (both shards agree): the loss
falls monotonically to 100k and is still improving 0.6–0.8% per 10k steps at the end
(0.01695 → 0.01351 at 30k → 0.01210 at 100k), while over the same range Context-FID
doubles (0.166 → 0.327) and the median per-subject AUC rises (0.575 → 0.640). **The loss
and the two quantities the paper is about move in opposite directions**, so a
loss-plateau rule would run to the worst point on both axes. A "no new minimum in 5
epochs" rule is worse still: one epoch is ~90 steps here, so five epochs is ~450 steps,
and single-step loss scatters between 0.007 and 0.021 — the rule would fire on noise.

**Cost.** One resample-and-score pass on one model is roughly 20 minutes, so bracketing
the minimum costs a few GPU-hours per cell against the ~125–700 GPU-hours the 27-model
campaign costs. It is bought back many times over: 30k instead of 100k is a 3.3x saving
on the campaign itself.

---

## Execution order

**Reordered 2026-08-30.** The old order put the Step 2 matrix — a month of GPU — ahead of
two cheap checks that change what that matrix means.

| # | task | cost | unblocks |
|---|---|---|---|
| 1 | Re-score fidelity against held-out subjects (Limitation 3) | hours, no GPU, no retraining | removes a bias against Step 4's own claim; can run any time, including after (4) |
| **2** | **Add checkpoint sampling to `padts`, `igfm`, `fourier_diff`, `timevae`** (`diffusion_ts` and `diffwave` are partly done) | **~2 days engineering** | **the actual critical path — no trajectories, no Figure 2** |
| 3 | Single-model pilot per baseline: cost and quality | ~1 day | may remove baselines from Step 2 |
| 4 | Step 2 full matrix | up to a month, less after (3) | Figure 2 |
| 5 | Replicates 2 and 3 | in parallel with (4) | every p-value |
| — | ~~Step 3c localisation~~ | **done 2026-08-27** | Step 4's design |

**CORRECTED 2026-08-30 — task 1 is worth doing but it does NOT gate task 4, and the
previous version of this paragraph said it did.** The claim was that Figure 2's two axes
read the same quantity, so the matrix must not run before the fix. That is wrong twice. The
axes are different functionals: the attack reads a **tail** property (nearest-neighbour
distance) and every fidelity metric reads a **bulk** one (`MODEL_DESIGN.md` §0) — and that
difference is the premise Step 4's entire mechanism rests on. And `copy_paste` is separated
correctly by the pair as they stand: Context-FID 0.070 with per-subject AUC 0.816, against
DiM-TS's 0.095 and 0.653. The memoriser is better on quality and far worse on risk, which is
exactly what the two axes are for.

**The real critical path is task 2**: four of the six baselines cannot sample from a
mid-training checkpoint, and without that there are no trajectories, and without
trajectories Figure 2 cannot be drawn at all. Step 3b is why trajectories are required —
risk rises and then falls with training length, so comparing generators at one fixed budget
compares each at an arbitrary point on its own curve.

**Open, and not on this list because it is a decision rather than a task:** whether Step 4
still has a backbone. MAVEN failed its pilot twice and is owed one 100k run before the
question is reopened; if that fails, the choices are another flow-matching host (IG-FM is the
same family) or accepting that the defence is backbone-independent — which is a weaker paper
and should be argued rather than defaulted into.

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
