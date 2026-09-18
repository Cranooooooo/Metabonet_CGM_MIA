# Paper structure — npj Digital Medicine

Modelled section-for-section on the supplied reference (Rafeletou et al. 2026,
doi 10.1038/s41746-026-03254-5). This file is the agreed skeleton;
`PAPER_PLAN.md` holds the working notes, open action items and claims discipline.

**Scope: one-day windows only (`d1_c1` CGM, `d1_c2` CGM + basal). Seven-day work is absent.**

---

## What we are deliberately NOT doing

The reference is a biomedical paper about **patients**. It is not organised around a
method. Five habits to suppress:

| ✗ Computer-science habit | ✓ What the reference does |
|---|---|
| A `Related Work` section | Background sits in the (unlabelled) introduction; comparison to prior work sits in the Discussion |
| A `Proposed Method` section before experiments | The method appears **inside Results**, as a measured finding, and its recipe goes to Methods at the very end |
| Noun-phrase subheadings (`Experimental setup`, `Ablation study`) | Every Results subheading is a **declarative sentence stating a finding** |
| `Experiments` / `Ablation` / `Implementation details` in the body | One `Methods` section, last, with noun-phrase subheadings |
| Leading with architecture and benchmark tables | Leading with **who is at risk**, and what that means clinically |

The arc of the reference is **landscape → discovery → mechanism → consequence →
validation**. Ours follows it.

---

## Title

Reference pattern: *"A novel multiomics machine learning signature **identifies** rapid
progression **in** clinically low risk prostate cancer"* — instrument, active verb,
clinical finding, patient population.

1. **Membership inference identifies concentrated re-identification risk in synthetic
   continuous glucose monitoring data** ← recommended
2. Atypical glycaemic profiles carry the re-identification risk in synthetic continuous
   glucose monitoring
3. A paired membership-inference framework identifies level-carried re-identification risk
   in shared synthetic continuous glucose monitoring

---

## Abstract — 150–220 words, one paragraph, **no printed heading**

No subheadings, no "Background/Methods/Results". Beats, in the reference's proportions
(problem → what was built → what was found, with numbers → what it means clinically):

1. Synthetic CGM is being shared, and journals and regulators now **require** depositing it.
2. We built a paired membership-inference design over a 506-subject cohort — a background
   generator and 26 one-subject-different generators — and applied it to seven generators
   across two data compositions.
3. Risk is **not uniform**: the strongest published generator exposes **24 of 26** patients
   above AUC 0.55 against that release's own permutation floor of **7**.
4. The signal is carried by **absolute glucose level** and **temporal ordering**, shown by
   ablating each.
5. A generator whose bottleneck cannot store absolute level sits **at the permutation
   floor in both compositions** while achieving the best fidelity of any learned model.
6. Level can be stripped and re-offset at release, preserving shape-based glycaemic metrics.

---

## (Introduction) — **no printed heading**, ~600 words, 4 paragraphs

Opens straight into the field, like *"Prostate cancer (PCa) remains the second most
frequently diagnosed…"*.

1. CGM is among the most valuable and least shareable physiological records; synthetic data
   is the proposed route, and this collection **requires public deposition** of it.
2. Privacy of synthetic health data is usually argued from nearest-neighbour heuristics or
   from the absence of copied records. Membership inference against *generative* models for
   physiological time series is barely measured, and CGM is assumed benign.
3. Aggregate privacy metrics average away the people who matter. Rare physiology is both
   what sharing is meant to preserve and what is most identifying — the sparse-population
   fairness question.
4. What we did, what we found (3–4 sentences, with the headline numbers), ending on the
   custodian-facing implication. **No roadmap paragraph.**

---

## Results — six subsections, each a declarative claim

### R1. Membership signal in synthetic CGM is bounded between a memorisation ceiling and a permutation floor
*Fig. 1 — design schematic; ceiling and floor on one axis.*

Establishes that the measurement resolves at all, and corrects how such numbers are read:
"patients above AUC 0.55" has a floor of **7–12 of 26, not ~1**, and the floor is computed
**per released dataset**. A copy-paste generator, which memorises by construction, sets the
ceiling. *This is a finding about the instrument, not a methods paragraph.*

### R2. Membership is recoverable for nearly every patient from glucose alone, and most confidently for those with atypical profiles
*Table 1 — seven generators × two compositions × (fidelity, arm AUC, exposed vs floor).*

The headline, and **not** the one an earlier draft assumed. DiM-TS exposes **24 of 26**
from glucose alone against its own floor of **7** — but of those 24, **13 of 13 are matched
controls** and only 11 of 13 are atypical. By count, exposure is *universal*, not
concentrated. The concentration is in **severity**: median per-patient AUC **0.80** in the
atypical arm against **0.635** in the control arm, where permuted membership gives 0.51 and
0.50. Most exposed patient reaches **0.96**. A threshold count alone hides this.

### R3. The membership signal is carried by absolute glucose level and by temporal ordering
*Fig. 2 — ablation of each coordinate.*

Destroying absolute level drops risk 0.65 → 0.55; destroying temporal order drops it too.
Two separable coordinates. This is the mechanism, and it licenses both interventions below.

### R4. Adding basal insulin protects typical patients but leaves atypical patients exposed
*Fig. 3 — real-data identifiability, measured exposure by arm, and fidelity, on one panel.*

The composition change separates the arms that glucose alone left indistinguishable:
matched controls fall **13/13 → 5/13**, atypical targets only **11/13 → 10/13**. The
protection accrues almost entirely to patients who were already typical.

This is the **opposite** of what the real recordings predict: one day of basal alone
identifies its subject **29.3 %** of the time (chance 0.11 %) against **1.5–2.6 %** for
glucose. Basal is a programmed schedule — person-stable and close in character to the
absolute level R3 names as the carrier.

The resolution is on the fidelity axis: **every** generator fits two channels worse, the
release is noisier, and membership signal is lost along with fidelity. What survives is the
signal that was strongest to begin with — the atypical patients. So adding an intrinsically
more identifying channel lowers measured exposure by degrading the release, and leaves the
most vulnerable patients where they were.

⚠️ Any two of the three quantities support the wrong conclusion. Fig. 3 must carry all three.
### R5. Removing the model capacity that stores absolute level returns membership signal to the permutation floor
*Fig. 4 — the two modules and their effect.*

The method, presented as a measured result. Two structural changes: masking whole
coordinates (contiguous time blocks; the level) instead of scattered cells, and pooling the
level subspace over time so it **structurally cannot** hold a per-timepoint trace — capacity
removed, not penalised. Result: **at the floor in both compositions** (10→10, 5→5) with the
best fidelity of any learned model (0.0383 / 0.0551). Report the honest magnitudes: the
level-task prediction holds by 12–24 %, while the block-masking half moves 2–5×.

### R6. Stripping and re-offsetting absolute level preserves shape-based glycaemic metrics
*Table 2 — glycaemic metrics before/after, per-chunk distributions with IQR.*

The release-time version of R3, for custodians who cannot retrain: remove the absolute
level, re-offset with a random start. Which clinical metrics survive (TIR, CV, MAGE) and
which cannot by construction (mean glucose, %time < 54).

---

## Discussion — **no subheadings**, ~800 words

Continuous prose, in this order: what the numbers license · comparison with prior work
(this is where "related work" lives) · **the disagreement we must report** — on `d1_c1` the
arm AUC and the floor-adjusted count point opposite ways for our own model, on one
replicate · whether the destroyed signal is the clinically useful signal · limitations
(one replicate, one cohort, one-day windows, no per-channel fidelity, outliers defined by
one metric family) · what a custodian should do.

---

## Methods — **last**, ~1,200 words, noun-phrase subheadings

1. **Sex as biological variable** — Nature Portfolio mandatory; the reference opens with it.
2. Cohort and data acquisition
3. Window construction and subject key *(the key is `(source_file, id)`; `id` alone is unique only within a study)*
4. Definition of atypical glycaemic profiles
5. Control matching
6. Paired membership-inference design
7. Generative models and training budgets
8. Membership statistic *(incl. K matching between arms)*
9. Permutation floor
10. Generation fidelity metrics
11. Glycaemic utility metrics
12. Statistical analysis

---

## Back matter — in the reference's order

```
Data availability          <- must name the deposited synthetic dataset + DOI
Code availability          <- the public GitHub repository
References                 <- BEFORE the acknowledgements
Acknowledgements
Author contributions
Funding
Competing interests
Supplementary information
```

---

## Display items

| | |
|---|---|
| Fig. 1 | Design, memorisation ceiling, permutation floor |
| Fig. 2 | Level and ordering ablation |
| Fig. 3 | Real-data identifiability vs synthetic leakage vs fidelity, by channel |
| Fig. 4 | The two structural modules and their effect on the floor |
| Table 1 | Seven generators × two compositions × three columns |
| Table 2 | Glycaemic utility after level-stripping |

Everything else → Supplementary.

---

## Proportions to hold to

Reference measured at **Results : Discussion : Methods ≈ 4 : 1 : 2.5**. Results dominates;
the Discussion is short. Main text 3,000–4,000 words.
