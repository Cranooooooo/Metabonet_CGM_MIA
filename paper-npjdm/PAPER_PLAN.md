# Paper plan — npj Digital Medicine

**Target.** *npj Digital Medicine*, Collection: **"Synthetic Clinical Data and
Privacy-Preserving Frameworks for Trustworthy Health AI"**
(<https://www.nature.com/collections/gggjgihcjj>), submission deadline **2027-07-01**.

**Scope decision (2026-09-18).** This paper is built on the **one-day window only** —
`d1_c1` (CGM) and `d1_c2` (CGM + basal insulin). The seven-day cells are **out of scope
for this submission**; they are still training and will not be mentioned. Everything below
is sized to what is already on disk for `d1`.

---

## 0. What the collection demands, and what that costs us

Three requirements from the call, each with an action:

| Requirement | Action |
|---|---|
| **Authors describing a synthetic clinical dataset MUST deposit it in a public repository at submission**, with a persistent identifier and an open licence (PhysioNet, Synapse, UK Data Service, …) | We must pick a repository, generate the release from the final model, and get a DOI **before** submission. This is not a formality — it is a hard gate, and it takes weeks (PhysioNet review is slow). **Start this early.** |
| Novelty **paired with practical clinical applicability** — bridging method and deployment | The paper cannot be only a benchmark. It has to end with something a data custodian can do. Our §Mitigation is that ending. |
| Fairness/bias with **particular attention to sparse populations and rare diseases** | This is our outlier design, and it is the single best fit we have. Foreground it — the whole study is "are the unusual patients the ones who get exposed?" |

**The deposition requirement is also our hook.** The collection requires releasing a
synthetic dataset; our paper measures what that release leaks. Say so in the first
paragraph of the Introduction — it makes the work constitutive of the collection rather
than adjacent to it.

---

## 1. Format (measured from two real npj Digital Medicine articles)

Sources: Rafeletou et al. 2026, *A novel multiomics machine learning signature identifies
rapid progression in clinically low risk prostate cancer*, doi 10.1038/s41746-026-03254-5
(the reference supplied 2026-09-18); and PMC12081667. Nature Portfolio order.
**There is no Related Work section** — background lives in the Introduction, comparison to
prior work lives in the Discussion.

```
Title / authors
(abstract)               NO heading printed. One paragraph, unstructured.
(introduction)           NO heading printed. Starts straight into the field.
Results                  WITH subheadings   <- first visible heading in the paper
Discussion               NO subheadings
Methods                  WITH subheadings   <- at the END
Data availability
Code availability
References               <- note: BEFORE the acknowledgements
Acknowledgements
Author contributions
Funding
Competing interests
Supplementary information
```

**Abstract length: 150–220 words.** The two articles measured 138 and 220, so this is not
a tight constraint. (A secondary source claiming a 70-word limit is wrong for Articles.)

**Proportions, by length in the reference: Results : Discussion : Methods = 4 : 1 : 2.5.**
Results dominates; the Discussion is short.

**Results subheadings are DECLARATIVE SENTENCES that state the finding, not noun-phrase
labels.** From the reference: *"NCCN prostate cancer risk groups are associated with
distinct genomic profiles and survival outcomes"*, *"Machine learning identifies ZNF268 as
a novel independent prognostic factor"*, *"Epigenetic dysregulation of ZNF268 is linked to
metabolic reprogramming"*. Every subheading is a falsifiable claim. Our §5 subheadings are
written this way below.

Main text ~3,000–4,000 words. Display items: aim for **4 figures + 2 tables**; push
everything else to Supplementary.

---

## 2. Title (candidates)

1. *Who gets exposed when synthetic CGM is shared? Membership inference against generative
   models for continuous glucose monitoring*
2. *Outlier patients carry the membership risk in synthetic continuous glucose monitoring*
3. *Structural capacity limits, not loss penalties, remove membership signal from synthetic
   CGM*

Pick 1 or 2 for a clinical readership — they name the patient, not the method.

---

## 3. Abstract (150-220 words, one paragraph, no printed heading)

Beats, in order:

1. Synthetic CGM is being released to enable sharing; collections and regulators now
   **require** deposition. What that release leaks is unmeasured.
2. We build a paired membership-inference design over **N subjects**, 13 consensus
   outliers matched to 13 controls, against **seven generators** on two settings
   (CGM alone; CGM + basal insulin).
3. Headline 1: risk is **not uniform** — it concentrates in outliers, and the strongest
   baseline exposes **24 of 26** patients above AUC 0.55 against a shuffle floor of 7.
4. Headline 2: the leak is carried by **absolute level and temporal ordering**, shown by
   ablating each.
5. Headline 3: a generator with **two structural modules** — removing the capacity that
   stores level, and masking whole coordinates rather than scattered cells — sits at the
   **shuffle floor on both settings** while giving the best generation fidelity of any
   learned model tested.
6. Implication for custodians: level can be stripped and re-offset at release time.

**Do not put a single unqualified "N× better than SOTA" in the abstract** — see §9.

---

## 4. Introduction (~600 words, 4 paragraphs)

1. **The pull.** CGM data is valuable and shareable only with difficulty; synthetic data is
   the proposed answer; journals/collections and regulators now require public deposition.
   State the collection's own requirement here.
2. **The gap.** Privacy of synthetic health data is usually argued from distance-to-nearest-
   neighbour heuristics or asserted from the fact that no real record is copied. Membership
   inference against *generative* models for physiological time series is barely measured,
   and CGM is assumed benign ("it's just glucose").
3. **Why outliers.** Aggregate privacy metrics average away the people who matter. Rare
   physiology is exactly what a sharing collection wants preserved and exactly what is most
   identifying. Frame as the fairness/sparse-population question the collection asks for.
4. **What we did and what we found** — three or four sentences, ending with the
   custodian-facing implication. No "the rest of this paper is organised as…".

---

## 5. Results — the section plan

Six subsections. Each names its display item and the numbers it must carry.

### 5.1 A paired design with a memorisation ceiling and a shuffle floor resolves membership signal *(Fig. 1)*

Establish the instrument before any claim. Content: the symmetric design (base on 475
background subjects; 26 `include` models each adding exactly one target); the statistic
`gap = d_OUT − d_IN`; and the two controls that make it readable:

- **`copy_paste` ceiling** — a generator that memorises by construction must separate, or
  nothing downstream means anything.
- **the permutation floor** — "how many of 26 exceed AUC 0.55" has a floor of **3–11, median 7**,
  and the floor is computed **per release**.

**Fig. 1** = design schematic + the two controls on one axis.

> ⚠️ This subsection is load-bearing. Reviewers of privacy papers reject on "your attack is
> weak", and the floor is what stops us over-claiming in the other direction.

### 5.2 Membership risk is not uniform across patients and concentrates in outliers *(Table 1)*

The main benchmark table across seven arms × two settings. Report **three** columns per
setting, not one:

| | Context-FID | arm AUC (outlier vs control) | patients > 0.55, raw → shuffle floor |
|---|---|---|---|

Numbers on disk (`d1_c1` / `d1_c2`):

| arm | Context-FID | arm AUC | raw → floor |
|---|---|---|---|
| `copy_paste` (ceiling) | 0.0335 / 0.0425 | 0.562 / 0.680 | — ⚠️ not yet computed |
| **IG-FM + 2 modules** | **0.0383 / 0.0551** | 0.698 / 0.479 | **10 → 10 / 5 → 5** |
| IG-FM (stock) | ⚠️ missing / 0.0552 | 0.515 / 0.598 | 6 → 5 / 7 → 6 |
| DiM-TS | 0.0948 / 0.1590 | 0.562 / 0.698 | **24 → 7 / 15 → 8** |
| FourierDiffusion | 0.1000 / 0.1943 | 0.710 / 0.485 | 7 → 5 / 6 → 3 |
| DiffWave | 0.3675 / 0.4905 | 0.639 / 0.740 | 12 → 8 / 13 → 11 |
| Diffusion-TS | 0.4224 / 2.0576 | 0.828 / 0.604 | 7 → 8 / 8 → 4 |

**The story is the third column, not the second.** DiM-TS at 24→7 is +17 patients over its
own floor; IG-FM + 2 modules is at ±0 in both settings.

### 5.3 The membership signal is carried by absolute glucose level and temporal ordering *(Fig. 2)*

The ablation. Destroying the **absolute level** drops risk 0.65 → 0.55; destroying **time
ordering** also drops it. Two independent coordinates, two different attacks.

This subsection is the bridge to the design and to the mitigation — without it, §5.5 is an
unmotivated architecture choice.

### 5.4 Basal insulin is the identifying channel in real data, yet adding it does not raise measured leakage from the synthetic release *(Fig. 3)*

**This answers the question Nick raised in the 09-01 meeting and Lucas said we had not
answered.** The claim "adding basal insulin makes normal subjects harder to identify" is
ambiguous between *privacy improved* and *the generator got worse*. We can now separate
them because we have fidelity for both settings.

⚠️ **Check the direction before writing this section — it is mixed, and an earlier draft of
this plan over-claimed it.** Measured c1 -> c2 (arm AUC / patients above own floor):

| arm | arm AUC | above floor |
|---|---|---|
| IG-FM + 2 modules | 0.698 -> 0.479 (down) | 0 -> 0 |
| IG-FM stock | 0.515 -> 0.598 (up) | +1 -> +1 |
| DiffWave | 0.639 -> 0.740 (up) | +4 -> +2 (down) |
| FourierDiffusion | 0.710 -> 0.485 (down) | +2 -> +3 (up) |
| Diffusion-TS | 0.828 -> 0.604 (down) | -1 -> +4 (up) |

No arm moves consistently, and the two readings disagree within arms. So the defensible
claim is NOT "insulin raises leakage" and NOT "insulin protects". It is: **basal is far
more identifying than glucose in the REAL data, yet that does not show up as more leakage
from the synthetic release — and fidelity falls in every arm when the second channel is
added, which is the confound that has to be reported alongside.**

Must also carry the identifiability result that explains the direction: on **real** windows,
one day of **basal alone** identifies a subject **29.3%** of the time (61× chance) against
**1.5–2.6%** for CGM alone. Basal is the identifying channel; CGM is close to anonymous.
That is the mechanism, and it supports Nick's clinical intuition that basal is
person-stable.

> ⚠️ Gap: we have **no per-channel fidelity** — Context-FID is computed over all channels
> jointly. Either add a per-channel decomposition or state the limitation explicitly.

### 5.5 Removing the capacity that stores level, rather than penalising its use, returns membership signal to the floor *(Fig. 4)*

The method, presented as a result because its effect is measured.

- **Coordinate masking.** Training hides **whole coordinates** — contiguous time blocks, or
  the level — instead of scattered cells. Scattered-cell masking is solvable by
  interpolation, and interpolation is exactly local correlation, which is where identity
  lives.
- **Level-subspace pooling.** The first 48 bottleneck channels are **averaged over time and
  broadcast back**, so they are structurally incapable of carrying a per-timepoint trace.
  *The capacity is removed, not discouraged.*

Report the falsifiable prediction and its **honest outcome**: `level`-task imputation error
should exceed `cell`-task error. It does, in every setting, but **by 12–24%, not
dramatically** — while the `block` task runs **2–5× higher**. So the contiguous-block half
of the masking module carries most of the effect. Say this; do not round it up.

> ⚠️ Also state that the companion group-decorrelation loss **cannot** constrain the
> quantity the pooling preserves (it z-scores along time, removing exactly the mean the
> pooling broadcasts). The pooling does the work. A reviewer who reads the code will find
> this.

### 5.6 Stripping and re-offsetting absolute level preserves shape-based clinical metrics *(Table 2)*

Strip the absolute level and re-offset with a random starting point (±10%). This is the
deployable version of §5.3, it was endorsed as clinically sensible in the 09-01 meeting,
and it is what makes the paper answer the collection's "practical clinical applicability".

**Needs the clinical-utility side**: which CGM metrics survive the strip (TIR, CV, MAGE)
and which do not (mean glucose, %time<54 by construction). Per-chunk distributions with
IQR, not a single mean delta — see §8/F1.

---

## 6. Discussion (~800 words)

1. **What the numbers license.** Risk is concentrated in outliers; the leak is level +
   order; structural capacity removal reaches the floor at no fidelity cost.
2. **Comparison to prior work** (this is where the old "Related Work" content goes).
3. **The uncomfortable one — report it.** On `d1_c1` the two readings disagree: IG-FM + 2
   modules has arm AUC **0.698 (p=0.045)**, worse-looking than stock (0.515), yet its
   per-subject count is **exactly at its own floor (10 → 10)** while stock is +1. With 13
   vs 13 the arm AUC is discrete (0.698 = 118/169) and within one replicate it is a screen,
   not a test. **We have one replicate.** Either run a second, or state this plainly as a
   limitation. Do not report only the flattering reading.
4. **Clinical meaning of what we destroy** — is the identifying signal the clinically
   useful signal? Partly: mean glucose is both. Level-strip-and-reoffset is defensible
   because the *shape* metrics survive.
5. **Limitations.** One replicate; one cohort; one-day windows only (do **not** promise the
   seven-day result); no per-channel fidelity; the outlier set is defined by one metric
   family, so "outlier" means "outlier by those metrics".
6. **What a custodian should do on Monday.**

---

## 7. Methods (~1,200 words, at the END)

Subsections: **Sex as biological variable** (a Nature Portfolio MANDATORY reporting item —
the reference opens its Methods with it; state the cohort sex composition and that our
analyses are not sex-stratified, if that is the case) · Cohort and windowing · Outlier definition (13 metrics, ≥7 votes, 4 seeds,
intersection) · Control matching · The paired design and the 27 training jobs · Attack
statistic and the frozen variant · The shuffle floor · Generators and budgets · Generation
quality metrics · Statistical treatment.

Two things that **must** be in Methods or a reviewer will catch them:

- **The subject key is `(source_file, id)`.** `id` alone is unique only within a source
  study; 27% of subjects on the naive key are composites of 2–9 people. State the corrected
  key.
- **`K` differs between arms by design** (an `include` model releases its extra subject's
  windows) and the attack **matches K** before computing any distance.

---

## 8. Open action items before writing (from the 2026-09-01 meeting)

| # | Item | Cost | Blocking? |
|---|---|---|---|
| **F1** | Clinical metrics as **per-chunk distributions with IQR/95% CI** + distribution distance (KS/Wasserstein), not a single mean delta | small | **Yes — §5.6** |
| **A2** | Per-channel fidelity, so "insulin helps privacy" vs "insulin is generated badly" is fully separable | medium | No, but weakens §5.4 |
| **B1–B3** | Correlate per-subject vulnerability with the outlier score, over a **fairer outlier/non-outlier mix**; use the more principled vulnerability ranking | medium | **Yes if we claim "who is at risk"** |
| **E1** | Human Turing test: ours vs real vs strongest baseline, randomised pairwise, rank by preference | medium | No — strong addition |
| — | `copy_paste` shuffle floor (4 cells) | ~0 | Only if we quote its counts |
| — | IG-FM stock `d1_c1` generation quality (never scored) | ~0 | **Yes — Table 1 hole** |
| — | **Deposit the synthetic dataset, get a DOI** | weeks | **Yes — hard submission gate** |

Deferred (seven-day work, out of scope here): window-length ladder and the link to the
30-day re-identification findings; training-budget/over-training curves beyond the `d1_c1`
result.

---

## 9. Claims discipline

Three things we will be tempted to write and must not, in this form:

1. **"N× better generation quality than SOTA."** Every such ratio compares our model at its
   budget against DiM-TS **at its 100,000-step endpoint**. Against DiM-TS's *best*
   milestone the margin is much smaller. Either quote best-against-best or say "endpoint".
2. **"Our modules remove the leak."** What is measured is *at the shuffle floor in two
   settings, one replicate*. That is a floor result, not a proof of no leakage.
3. **"CGM is safe to share."** Our own data says basal is 20× more identifying than CGM —
   so the safe-sounding claim is about the channel, not the modality.
