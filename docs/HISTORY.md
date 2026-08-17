# How the study got here

A chronological record: what was run at each stage, what it returned, and what that
result made us do next. Numbers are from the artefacts named beside them, not from
recollection. `docs/PLAN.md` holds the design derivations; `docs/MEMO.md` holds the
costings and the decisions still open; this file holds the sequence.

---

## Stage 1–3 · The single-channel study is set up (to 2026-08-07)

**The question.** Not "does this cohort leak on average" — an earlier population-level
study on this data found nothing — but **whether the unusual subjects are the exposed
ones**.

**The cohort.** MetaboNet's no-DUA public release: 154,842,077 rows, 1,291 subjects, 14
studies. Keep a (subject, day) only when all 288 five-minute cells carry a reading, keep
subjects with ≥30 such days, draw stratified on day count. → **875 subjects, 182,597
one-day CGM windows**.

**Who is an outlier.** Fourteen methods in five families — clinical consensus metrics,
curve shape under CID-DTW, whole-distribution MMD and sliced Wasserstein, learned
TS2Vec embeddings, and two negative controls. A subject is an outlier when it lands in
the top 5% of **≥7 of the 13 candidates**. The whole stack is re-run under four base
seeds and only subjects flagged under **all four** are carried forward: 28 → 24 → **20**.

**The design.** Per replicate: 20 outliers, 20 day-matched controls, 835 background.

```
base       trained on the 835 background        -> d_OUT for all 40 targets
include_t  trained on the background plus t     -> d_IN  for target t
gap        d_OUT - d_IN, frozen as min x mean
question   does the OUTLIER arm's gap exceed the CONTROL arm's
```

Three replicates with disjoint control draws, because within one replicate all 40
targets share a base and a p-value over them is overstated. **41 models × 3 = 123.**

---

## Stage 4 · The main campaign (2026-08-07 → 08-10)

123 DiM-TS models at `hidden_size=128`, 655 GPU·h, 63.8 h wall clock, zero failures.

```
                  rep1     rep2     rep3
AUC              0.680    0.690    0.670
p_between        0.027    0.021    0.034

mean 0.680   SD 0.010   95% CI [0.655, 0.705]   one-sided p ~ 0.0005
```

**Outliers do leak more than day-matched controls, and it replicates.**

The replicates earned their cost: the *within-arm* p-values swing four orders of
magnitude across them (1.8e-05, 9.5e-07, 0.053) while the *between-arm* AUC moves by
0.02. A single run's offset dominates a within-arm reading and cancels in the
comparison — which is what the symmetric pairing was designed to do.

---

## Stage 5–9 · What 0.680 is worth (to 2026-08-13)

An effect that replicates is not the same as an effect that matters. Three checks:

**Per-subject risk.** 79 of 80 subjects have a per-subject AUC of **0.50–0.52**. One
does not: subject **1142**, at 0.643 paired and 0.717 under the single-model panel,
**+0.152 net of its own control** and 17.8 SD above the control mean.

**A realistic attacker.** Eight supervised (classifier × feature) attackers under
leave-one-subject-out reach at best **+0.014** between the groups, and half are
negative.

**Does the instrument work at all?** The same panel on `copy_paste`, which memorises by
construction, reads **0.82 against an arithmetic ceiling of 0.816**, and 0.50 on every
negative control. So the null is a measurement, not a failure to measure.

**The publishable claim became a bounded safety result:** leakage is measurable,
systematic and materially negligible — for everyone except one person.

**Decision.** The remaining doubt was the data itself. One day of CGM is a single
smooth, homeostatically bounded channel; perhaps there is simply nothing to leak. That
sent us back to the raw file.

---

## Stage 6 · What is actually in the raw data (2026-08-14)

**37 columns, not 3.** The time-varying numeric ones, with the fraction of non-null
cells that are not exactly zero:

| channel | non-null | non-zero | reading |
|---|---|---|---|
| CGM | 0.898 | 1.000 | the trace we had |
| insulin | 0.965 | 0.767 | **exactly basal + bolus**, verified row by row |
| basal | 0.936 | 0.751 | a second continuous trace |
| bolus | 0.944 | **0.066** | an event channel, 93% zeros |
| carbs | 0.710 | **0.016** | an event channel, 98% zeros |
| heartrate / steps | 0.007 | — | 13 and 12 subjects. Unusable |

**The premise for single-channel was false.** `cohort.py` justified dropping insulin
because it "gates out 63% of the subjects that have usable CGM". Measured under the
module's own rule: CGM 875, CGM+basal+bolus **834**. The cost is 4.7%.

---

## Stage 7 · Three data defects, found while acting on that (2026-08-14)

**`id` is unique only within a study.** Each of the 14 studies numbers its participants
from 1 and the consolidated release did not re-key. 241 of 1,291 ids (18.7%) appear
under two or more; subject "102" spans five studies and reports two CGM devices at the
same minute with different values. **238 of the 875 published subjects (27.2%) are
composites of 2–9 people, holding 41% of the windows**, and 9,055 windows (5.0%)
interleave two people sample by sample.

Checked before assuming the worst: the consensus outliers are *depleted* of composites
(2 of 24 against a 27.2% base rate), subject 1142 is single-study, and composites are
not more identifiable. **The published findings sit on single-study subjects.**

Re-keying on `(source_file, id)` is a correction that also **enlarges** the cohort:
875 → **1,329** subjects at one channel, **1,253** at three.

Two smaller defects, both mine, both now guarded: a float32 axis-0 reduction that
stalled and wrote CGM mean 96.05 where the truth is 145.29 (`docs/PITFALLS.md` §14),
and subject ids containing `/` breaking every path built from them.

---

## Stage 8 · The ceiling, measured without any generator (2026-08-14)

Before spending more GPU on attacks, ask what there is to find. Leave-one-day-out
subject retrieval on **real** windows: hold out a day, rank every subject by the
distance from that day to their nearest other day, record where the true subject lands.
It bounds any membership attack from above and costs one CPU job.

**Every day is ranked only against subjects from its own study**, which holds units,
device, pump model, protocol and era fixed. Top-1 / lift over that study's chance rate:

| space | all 3ch | CGM only | **basal only** | bolus only |
|---|---|---|---|---|
| quantile | 17.3% / 36× | 1.5% / 3.1× | **29.3% / 61×** | 1.2% / 2.6× |
| level | 14.9% / 31× | 1.1% / 2.3× | 14.3% / 30× | 2.6% / 5.4× |
| raw | 6.5% / 14× | 2.6% / 5.5× | 17.3% / 36× | 2.2% / 4.6× |

**A day of CGM is close to anonymous; a day of basal is not** — about 20× more
identifying. A basal profile is a programmed schedule; glucose is homeostatically
bounded.

On the single-channel cohort the same measurement **reproduces the outlier finding with
no generator at all**: consensus outliers have median normalised rank 0.134 against
0.326 (p < 0.0001), and subject 1142 sits at z = −4.0.

**Decision.** The real data holds a large outlier-vs-normal gap that the membership
pipeline converts into 0.50–0.52. The signal is lost in the pipeline, not missing from
the data — and basal is where identity lives. Move to three channels.

---

## Stage 9 · Outlier detection on three channels (2026-08-15)

The stage was made multichannel. Group A (the four clinical methods) stays CGM-only —
its metrics are defined on glucose and a basal rate has no time in range — so four of
thirteen votes are identical between runs and the denominator is unchanged, which is
what makes the two lists comparable.

Same 1,253 subjects, same 172,119 days, four seeds, only the channels differ:

```
CGM only              37 stable outliers
CGM+basal+bolus       20 stable outliers
in both 15   Jaccard 0.357
```

Per method, adding channels replaces **76–97%** of each top-5% set. The self-check
passed: group A and E13 moved not at all.

**A finding that had to be reported rather than papered over:** the vote-count trough
that justified "≥7 of 13" does not exist on this cohort — single-channel troughs at 5,
three-channel at 4 with no second mode. The threshold is now a convention held fixed
for comparability, and the paper must say so.

---

## Stage 10 · The three-channel campaign, and two negative results (2026-08-15 → 08-16)

Launched at h128 on the corrected cohort. **Stopped at 44 of 123 models**, for two
reasons.

**It could not have finished.** Three channels cost **8.71 h/model against the
single-channel 5.35** — 1.63×, entirely in training (ms/step 39 → 98; ms/sample
unchanged). The launcher's walltimes were budgeted from 5.35 and every lane was killed.
The earlier "channel count is nearly free" claim came from the *parameter* count, where
C=1 and C=3 differ by 514 parameters. Parameter count is not compute.

**Generation quality is worse and nothing fixes it.** A classifier separates real from
synthetic at **0.65–0.68**, where the single-channel models sat at 0.50.

- *Not training length*: the loss is converged — 0.01218 at 30–40k steps, **0.01198 at
  both 80–90k and 90–100k**.
- *Not capacity*: one model per width, 128 → 256 (3.8× the parameters), discriminative
  accuracy 0.595 → 0.582 with no trend. The spread across five widths (0.061) is **half
  the spread across three models at one width** (0.124).
- *The channel*: `bolus` is 95% exact zeros. A single feature — the share of near-zero
  bolus cells per window — separates real from synthetic at **0.74**, better than the
  full discriminator's 0.60. A Gaussian diffusion process cannot emit a repeated exact
  constant, and it under-produces the spikes.

---

## Stage 11 · Where the membership signal actually is (2026-08-16)

The same attack on the same models, restricted to one channel subset at a time. Nothing
retrained. Per-subject AUC over each subject's own windows:

| channels | outlier mean | control mean | outliers excl. 1142 | Loop/1142 |
|---|---|---|---|---|
| CGM | 0.5131 | 0.4989 | 0.5029 | **0.6358** |
| basal | 0.5002 | 0.5164 | 0.5004 | 0.4979 |
| bolus | 0.4735 | 0.4983 | 0.4709 | 0.5045 |
| CGM+basal | 0.5033 | 0.4955 | 0.5029 | 0.5084 |
| all three | 0.5023 | 0.5019 | 0.5009 | 0.5182 |

**One subject, one channel.** `Loop/1142` reads 0.636 on CGM, CI [0.615, 0.658],
p = 1e-31 — the same person at the same strength as the published study's 0.643,
reproduced across a rebuilt cohort, a corrected key and a changed channel set. He is
unremarkable on basal and bolus. Every other outlier sits between 0.479 and 0.516.

**Adding channels dilutes the one real signal.** 1142 falls from 0.636 to 0.508 on
CGM+basal — outside his own confidence interval. The statistic is a Euclidean distance
over the flattened window and `min` picks the nearest sample *in the joint space*, so
288 uninformative dimensions change which sample is nearest.

**High identifiability is not high memorisability.** basal is 20× more identifying than
CGM in the real data and contributes nothing to membership leakage. A basal profile is
stereotyped — the generator learns the population's few templates without memorising
anyone. A CGM trace is idiosyncratic noise, and reproducing one means remembering it.

---

## Stage 12 · Is one day too short? No (2026-08-16)

`copy_paste` bootstraps K rows with replacement, so 63.2% of a target's windows come
back verbatim: it measures what the **attack** can detect when leakage is maximal and
known, with the generator's fitting ability removed.

| window | per-subject AUC | median gap |
|---|---|---|
| 1 day (published) | 0.8187 | 0.0680 |
| 7 days, contiguous | 0.8147 | 0.0917 |
| 14 days, concat | 0.8229 | 0.0951 |
| 21 days, concat | 0.8236 | 0.1047 |
| arithmetic ceiling | 0.816 | |

**Flat, 0.811–0.824, against per-subject intervals of ±0.06.** The gap magnitude rises
54% — a memorised long window stands out by a larger margin — but the *fraction*
memorised is 63.2% at every length and a rank statistic sees only the fraction.

"One day is too short for a membership signal to exist" is therefore not the
explanation for the flat results. The attack works identically at 288 and 6,048
dimensions.

A constraint discovered here and worth carrying: complete days are mostly **not
consecutive**. The median subject's longest unbroken run is 8 days. Calendar-contiguous
weeks exist for 809 subjects, fortnights for 40, three-week runs for 11.

---

## Stage 13 · Running now

**Seven-day pilot.** 10 single-CGM outliers (including 1142) + 10 day-matched controls
+ 1 base = 21 DiM-TS models on calendar-contiguous weeks, CGM only, `hidden_size=96`
(indistinguishable from 128 on every quality metric at 81% of the cost), 3,700 steps
(the published 100,000 is 64.6 epochs over 197,970 one-day windows; the same 64.6
epochs over 7,001 seven-day windows is 3,700). If a difference appears, the full design
is 20 outliers and 60 controls.

**The confound this pilot cannot remove:** seven-day windows are not only longer, there
are 27× fewer of them, because a week needs seven consecutive complete days. Fewer
training examples means more memorisation per example. A difference found here is
"longer windows AND a smaller training set".

---

## The through-line, in five sentences

1. Outliers leak more than matched controls, AUC 0.680, replicated three times — but
   every individual's absolute risk is negligible except one person's.
2. That negative is trustworthy: the same instrument reads 0.82 on a generator that
   memorises and 0.50 on every negative control.
3. The data is not the limitation — real days carry a large outlier-vs-normal
   identifiability gap, and window length does not move the attack's ceiling.
4. Nor is the generator's size or training length; three channels made generation worse
   for a reason specific to one sparse channel.
5. What survives every change of cohort, key, channel and statistic is **one subject**.
