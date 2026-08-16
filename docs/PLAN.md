# The experiment plan, and where it currently stands

This file is the durable record of *what we are running and why*. It exists because the
plan lived in a chat session and a machine restart took it. Anything load-bearing that
is decided in conversation belongs here the same day.

**Status as of 2026-08-10 12:15 (+08).** Stages 1–5 complete. The campaign ran
2026-08-07 19:59 → 2026-08-10 11:45, 63.8 h wall clock, **123/123 models, zero
failures, zero requeues**. Result below.

---

## The result

Frozen statistic `min × mean`, K matched, 20 outliers vs 20 day-matched controls per
replicate.

```
                     rep1       rep2       rep3
auc                 0.6800     0.6900     0.6700
p_between           0.0266     0.0206     0.0339
median gap outlier  0.000300   0.000387   0.000177
median gap control  0.000198   0.000151  -0.000028
p_within outlier    1.8e-05    9.5e-07    0.0527
p_within control    0.0181     0.0120     0.7738
```

**Replicate-level test** — the three draws are disjoint, so their backgrounds, bases and
arm differences are independent, and these three numbers are the units a p-value may
rest on:

```
mean auc 0.680   SD 0.010   SE 0.0058   t(2) = 31.2
95% CI [0.655, 0.705]                   one-sided p ~ 0.0005
```

### What the replicates showed that a single run could not

The **within-arm** quantities swing wildly across replicates while the **between-arm**
comparison barely moves:

```
                  rep1      rep2      rep3     spread
p_within outlier  1.8e-05   9.5e-07   0.0527   4 orders of magnitude
p_within control  0.0181    0.0120    0.7738   64x
auc               0.6800    0.6900    0.6700   0.02
```

In rep3 the control arm's gap is *negative* and not significant at all, where in rep1
and rep2 it was significantly positive. The same statistic, read within an arm, swings
to nearly the opposite conclusion — and the arm comparison is unmoved.

That is the symmetric design paying off exactly as derived. A single training run's
offset `δ` is large enough to dominate the within-arm reading; under the symmetric form
it enters both arms with the same sign and cancels in the comparison. Under the older
asymmetric form it would have entered with **opposite** signs, and at the magnitude rep3
shows, that alone could manufacture or erase the entire effect.

The K artefact is small and measured, not asserted: unmatched K gives rep1 auc 0.6975
against 0.6800 matched, so released-set size accounts for 0.02 of the 0.18.

### How strong is the attack on each arm, and for which attacker

AUC 0.680 is a **between-arm** number — P(a random outlier's gap exceeds a random
control's). There is no separate "outliers' AUC" and "normals' AUC" to compare; 0.680
*is* that comparison. What can be asked per arm is how well a membership attack does on
that group, and the answer depends entirely on what the attacker is assumed to hold.

```
arm          paired acc   sign p   unpaired AUC   paired, per replicate
outlier          0.850    0.0000       0.518      [0.85, 1.00, 0.70]
control          0.650    0.0137       0.513      [0.80, 0.65, 0.50]
                                                  chance = 0.50
```

**Paired** — the attacker holds *both* models and asks which one contains t; correct
when `gap > 0`. Outliers 85%, controls 65%: a 20-point difference, and the ordering the
study set out to test. But note the second row is *also* above chance (p = 0.014):
ordinary subjects leak too, just less. The claim is "outliers leak **more**", never
"only outliers leak".

This setting is an **upper bound**. No real attacker is handed a model trained
deliberately without their target — that counterfactual is the study's instrument, not
a threat model.

**Unpaired** — the attacker holds one released set and scores membership by absolute
distance. Both arms land at ~0.51 against an SE of ≈0.053 at n=60: **chance**. The
paired signal does not transfer for free, because between-subject variation in absolute
distance swamps the within-subject membership shift — which is exactly why the design
pairs.

Boundary on that: this is the *naive* absolute-distance attacker. A better-calibrated
unpaired attacker (per-subject normalisation, a learned score) might do better. What is
established is that the paired signal does not transfer trivially, not that it cannot.

The per-replicate paired accuracies swing again — `[0.85, 1.00, 0.70]` for outliers,
20/20 correct in rep2 and 14/20 in rep3 — while the between-arm AUC stays at 0.67–0.69.
Same pattern as everywhere else in this study: within-arm readings are unstable, the arm
comparison is not.

### Who inside the outlier arm — `results/attack/dimts_h128_subject_risk/`

Gaps standardised against their own replicate's control arm (median/MAD), so the three
bases' offsets leave. Null = 60 controls, one measurement each, since the draws are
disjoint; outliers are the same 20 measured three times.

```
        z_mean   z_sd   pct vs null   n_above(0-3)   days
 569      8.58   4.67       100%           3          223
1142      6.29   2.22       100%           3          363
 186      4.92   2.82        98%           3           84
 973      2.11   2.75        93%           3           67
 ...
1163     -0.72   1.14        20%           1           31
null: median 0.00, p95 +2.60
```

**Heavy tail AND broad shift, not one or the other.** Only 3/20 clear the null's 95th
percentile — against 1/20 expected, binomial p ≈ 0.08, which is not significant, so
"a few individuals cross a threshold" is not the story either. But dropping the three
most exposed still leaves **AUC 0.625** (per replicate 0.624 / 0.638 / 0.615; t(2)=18.3,
one-sided p ≈ 0.0015), so they do not carry the effect on their own. Read that as
"still 0.625", not "fell by 0.055" — removing the top-ranked positives lowers AUC
mechanically.

**Per-subject risk is only partly reproducible.** Spearman across replicate pairs on the
20 outliers: +0.674 (p 0.001), +0.426 (p 0.061), +0.203 (p 0.391) — one pair of three.
569, 1142 and 186 sit at the top in every replicate and 10/20 stay above their
replicate's control median in all three, but the middle and tail do not rank
consistently, and even 569's own z varies with SD 4.67.

> Reportable: *"569, 1142 and 186 are reproducibly high-exposure individuals."*
> Not reportable: a per-subject risk ranking. Three replicates made the per-subject
> question **askable**; they did not make it answered.

Two checks that came back clean:

- **No day-count confound.** Spearman(z, n_windows) = +0.374, p = 0.104 over the 20
  outliers; +0.065, p = 0.622 over the 60 controls. The diagnostic that disqualified
  `subject_reduce=min` passes on the frozen variant. *(First run of this reported
  p = 0.023 for the outlier arm by correlating over all 60 rows — 20 people counted
  three times each, with `n_windows` constant within a person. The control arm's 60
  rows are 60 distinct people, so only one of the two numbers was wrong, which is what
  made it easy to miss.)*
- **The drawn borderline control behaved normally.** 467, flagged by 2 of 4 detection
  seeds and kept on purpose, sits at the null's 63rd percentile. Keeping and tagging
  rather than removing is what made that checkable at all.

### One AUC per subject — `results/attack/dimts_h128_subject_auc/`

The attack is window-level underneath, so each subject has its own AUC:
`P(d_out > d_in)` over that person's own windows, Mann-Whitney, 0.5 = no signal. Only
the per-subject *reduction* reaches `gaps.parquet`, so the window vectors are recomputed
(`scripts/subject_auc.py`, 5 min on qdev). 20 outliers averaged over their 3 replicates
+ 60 controls = **80 numbers**, each with a paired bootstrap CI.

```
group       n   AUC med  AUC mean          IQR      max   >0.55
outlier    20     0.505     0.513   [0.501, 0.514]  0.643     1
control    60     0.501     0.502   [0.498, 0.506]  0.518     0
```

**79 of 80 subjects sit at 0.50–0.518: individually unattackable.** The single exception
is **1142** at AUC 0.643, CI [0.628, 0.660], 363 windows (the longest record in the
cohort), Wilcoxon p = 4.5e-60 — the one person a per-record attack would actually
identify.

No record-length confound: Spearman(AUC, n_windows) = +0.093, p = 0.411.

**Detectable and substantial are different, and here they separate cleanly:**

```
                 AUC > 0.52      Wilcoxon p < 0.05
outlier (n=20)     1  (5%)          14  (70%)
control (n=60)     0  (0%)          17  (28%)
```

70% of outliers leak *detectably*; one leaks *materially*. The most extreme control, 34,
has AUC 0.518 at p = 3e-5 — overwhelming significance, negligible magnitude. The
controls' 28% is also far above the 5% chance rate, so ordinary subjects leak too,
matching the 65% paired accuracy above.

**So "outliers are more attackable" holds at the group level and fails at the individual
level.** Group: AUC-of-AUCs 0.668, p = 0.013 (secondary/exploratory — a second test on
data whose primary test is the frozen `min × mean`). Individual: 19 of 20 outliers are
as unattackable as controls. The group effect is a systematic sub-percent shift in
per-subject AUC (median 0.505 vs 0.501), not individual attackability.

#### Two leakage modes, found because the two statistics disagree

```
target   days   gap-z    AUC    per-window paired hit-rate
 569     223    8.58    0.515          0.604
1142     363    6.29    0.643          0.856
```

569 ranks first on the frozen gap statistic and is unremarkable on AUC; 1142 is the
reverse. That is not a contradiction — the two measure different things:

* **569: a few days memorised.** A large shift in *mean* nearest-neighbour distance
  (z = 8.58) carried by a minority of windows; only 60% of its windows favour the member
  model.
* **1142: the whole person shifted slightly.** 86% of windows favour the member model
  with a modest per-window shift.

Different mechanisms, and they would need different mitigations — per-sample outlier
suppression against the first, distribution-level regularisation against the second.
Neither statistic alone would have shown this.

### Generation quality — the check stage 4 shipped without

Stage 4 released 123 models and stage 5 measured leakage on them with nobody having
measured whether the samples were any good. That is a hole: a generator emitting noise
gives d_IN ≈ d_OUT for everyone, the attack reads 0.5, and "no leakage" is
indistinguishable from "no model". Measured 2026-08-10 on the three h=128 bases
(`scripts/eval_quality.py`, `results/quality/`):

```
base       Context-FID   discriminative   predictive MAE (TSTR)
rep1            0.0465           0.0042             0.0146
rep2            0.0645           0.0292             0.0159
rep3            0.0634           0.0083             0.0160
```

**The samples are not noise — they are close to indistinguishable.** The discriminative
score is |accuracy − 0.5| for a post-hoc GRU told to separate real from synthetic: it
reaches **50.4 %–52.9 %**. One-step-ahead MAE is 0.015 against a data span of ~1.375,
about 1 %.

So the headline result is not an artefact of a broken generator. The interesting
combination is the opposite one: **a generator whose samples a classifier cannot pick
out still leaks membership at AUC 0.68.** High fidelity and measurable leakage co-occur
here; neither "it didn't learn" nor "it overfit and is also bad" describes it.

The three bases span 0.0042–0.0292 on the discriminative score. That was first read as
run-to-run instability in the models, by analogy with the within-arm attack readings.
**It is not — see the next subsection, which measures it.**

#### How much of that spread is the metric rather than the model (2026-08-12)

The three numbers above vary in two things at once: the model, and nothing else, because
the evaluation seed was 2026 for all of them. So one `base` — `dimts_h128_rep1` — was
re-scored at three further seeds, changing only the 3,000-window subsample and the
classifier initialisation:

```
                        Context-FID   discriminative   predictive
same model, seed 7           0.0652         0.0017         0.0147
same model, seed 101         0.0551         0.0183         0.0154
same model, seed 999         0.0389         0.0433         0.0136
three DIFFERENT models,
      all at seed 2026    0.0465-0.0645  0.0042-0.0292  0.0146-0.0160
```

**One model measured three times varies at least as much as three models measured
once.** The discriminative spread is 0.0017–0.0433 within a single set of weights,
wider than the 0.0042–0.0292 across three training runs. The earlier attribution was
therefore wrong: most of that dispersion is the estimator — a 3,000-window subsample of
176,445 and a small GRU's initialisation — not the generator.

Consequences, both of which bind on anything downstream:

* **A single quality number carries an error bar of roughly its own size.** Comparing
  two configurations on one score each cannot resolve a difference smaller than about
  0.04 on the discriminative score or 0.026 on Context-FID.
* **These three metrics cannot currently rank capacities.** Doing so needs the seed
  averaged over — several evaluation seeds per model, which is minutes of GPU each and
  far cheaper than more training runs.

The within-arm attack instability documented above is a genuinely different phenomenon:
there the estimator is exact and the *models* move. Here the models are fixed and the
estimator moves. The two were conflated on 2026-08-10 and should not be again.

#### Capacity vs quality, once the seed is averaged out (2026-08-12)

Every model scored at all four seeds {2026, 7, 101, 999}; h=128 at all three bases, so
seed variance and model variance are separable there. h=52 has one model, so its model
variance is not estimable — the limit on everything below.

```
        params      h/model   vs h=128    Context-FID      discriminative    predictive
h=52     522,907     3.30      0.62x     0.1132 +- 0.0290  0.0283 +- 0.0223  0.0150
h=96   1,604,851     4.39      0.83x     0.0549 +- 0.0049  0.0306 (see below) 0.0148
h=128  2,762,291     5.31      1.00x     0.0575 +- 0.0109  0.0130            0.0147
```

Measurements: h=52 one model x 4 seeds, h=96 one model x 16, h=128 three models x 4
(rep1 also at 16). Seed SD at h=128 (0.0094) exceeds its model-to-model SD (0.0062),
which is why every comparison below averages the seed out.

**Quality saturates between h=52 and h=96; it does not track width across the range.**

```
Context-FID, one-sided Mann-Whitney
  h=52  vs h=128    p = 0.0002     clearly worse
  h=96  vs h=128    p = 0.85       indistinguishable (if anything, lower)
```

All four h=52 measurements (0.0791–0.1432) sit above all sixteen h=128 measurements;
h=52's mean is 1.93x h=128's, 8.8 model-SDs above it. h=96 lands on top of h=128.

The single-seed reading of 2026-08-10 got the h=52 gap wrong by ~40% in the direction
that mattered: seed 2026 is h=52's *best* of four (0.0791 against a mean of 0.1132).
One measurement per configuration was not enough to rank anything here.

#### The discriminative signal at h=96 that did not replicate

Worth recording as a method note, because the first three readings all pointed the
wrong way. h=96 produced 0.1842 at seed 999 — the largest value in a 20-measurement
grid whose next largest was 0.0667. Two follow-ups made it look real: a second value
above 0.08 appeared at a new seed, and a paired sign test over 8 seeds gave 6/8 in the
same direction (p = 0.063).

It does not survive the only test that is not circular. The hypothesis was generated by
the extreme value, so the seeds gathered *before* it was noticed cannot test it:

```
                              pairs   h=96 higher   median diff   sign-test p
discovery set (not a test)      4        3/3          +0.0167        0.125
OUT OF SAMPLE (the test)       12        7/12         +0.0033        0.387
pooled (selection-biased)      16       10/15         +0.0058        0.151
```

The effect halved as n doubled and the out-of-sample direction is 7/12 — chance.
**h=96 is not more discriminable than h=128**; the 0.1842 was this estimator's tail,
which the same page already documents as wide (0.0017–0.0433 on one fixed model).

The general rule this is an instance of: an estimator with SD comparable to its mean
will hand you an extreme value roughly every twenty measurements, and the seeds you
collected before noticing it are part of how you noticed it.

#### The first cross-generator point: diffusion_ts, 2026-08-13

One base at each generator's own default, scored at the same four seeds. `diffusion_ts`
is the first to finish and the first cross-generator quality number this project has.

```
                 h/model   vs h=128    Context-FID       discriminative   predictive
diffusion_ts      1.68      0.32x     1.193 +- 0.14      0.140            0.0226
DiM-TS h=128      5.31      1.00x     0.0575 +- 0.0109   0.019            0.0147
```

**A third of the cost and 20.7x the Context-FID**, with no overlap at all: the lowest of
the four `diffusion_ts` seeds (1.052) is above the highest of the sixteen h=128
measurements (0.0715). Its discriminative score of 0.140 is a classifier accuracy of
64% against DiM-TS's 51.9% -- these samples are separable from real data, where DiM-TS's
are close to not being.

**What this does NOT establish.** `diffusion_ts` ran at its authors' default, which is
**12,000 optimiser steps and 223,719 parameters** against DiM-TS's 100,000 steps and
2,762,291. Two variables moved together, and the capacity sweep above already showed
this cohort is sensitive to one of them (h=52 is 1.93x h=128's Context-FID). Per-step
cost is nearly identical (0.039 s against 0.039 s x 2 grad-accum), so **the saving is
entirely the shorter training budget, not a faster generator.**

The comparison that separates them costs 2.6 GPU.h: `diffusion_ts` at a matched 100,000
steps is 1.08 h of training plus 1.55 h of sampling, still half of DiM-TS's 5.31 h. If
quality closes, the result is a genuine 2x speed-up attributable to budget; if it does
not, `diffusion_ts` is weaker on this cohort and the claim is finally clean.

Until that runs, the honest statement is **"diffusion_ts at its published defaults is
much cheaper and much worse here"**, which answers neither "is it a worse generator"
nor "is 12k steps enough".

#### What this settles

* **h=52 is not usable** — a measurable loss of distributional fidelity for a 1.6x
  saving.
* **h=96 costs nothing measurable and saves 17% of wall clock** (4.39 h vs 5.31 h per
  model, ~38 GPU.h per 41-model replicate). It is the fastest configuration that no
  metric here separates from the published width.
* **`hidden_size=128` remains the defensible default for the reported result**, now on
  this project's own evidence rather than only as the authors' choice.

**The boundary on all three.** h=52 and h=96 are one training run each. h=128's
model-SD of 0.0062 is small only because three of them were measured, and neither of
the other widths has an equivalent. A second h=96 model (~4.4 GPU.h) is what would turn
"indistinguishable" into "indistinguishable across models". Units are not comparable to
published tables, so "1.93x" is a ratio within this project's own estimator.

#### These metrics had to be rebuilt, and why that matters

None of the three vendored implementations runs on this cohort, for three independent
reasons — documented in `cgmoutlier.quality`:

1. `discriminative_metric.py` and `predictive_metric.py` are **TensorFlow 1**; neither
   environment has TF, and installing things is how this project once replaced torch
   2.7.0+cu126 with a cu130 build and silently lost CUDA. Reimplemented in torch.
2. `context_fid.py` imports `Models.ts2vec.ts2vec`; the vendored TS2Vec is at
   `ts2vec/ts2vec.py` **and its own files import the stale path too**. Nothing in the
   tree calls Context_FID, so it had never been exercised. Repaired with a `sys.modules`
   alias so the vendored source stays byte-identical to PROVENANCE.md.
3. **Both TimeGAN-lineage metrics are undefined at C=1** — the third appearance of this
   vendored code's multivariate assumption, after `mmd_alpha`:
   `predictive_metric.py:57` builds a `dim-1` = 0-feature input, and
   `discriminative_metric.py:74` sets `hidden_dim = int(dim/2)` = 0 units. The
   discriminative width is now a named argument; the predictive score is **redefined**
   as univariate one-step-ahead, which is a declared change of definition, not a port.

Consequence: these numbers compare **our configurations to each other**. They are not
comparable to the DiM-TS or TimeGAN papers.

### What this supports, and what it does not

**Supported.** DiM-TS at its published width (`hidden_size=128`, 2,762,291 parameters),
releasing K = the training-set size, leaks membership *more* for outlier subjects than
for day-count-matched controls: AUC ≈ 0.68, stable across three independent replicates.

**Supported, and it is the sharper claim.** That group difference is a systematic
sub-percent shift, **not** individual attackability. 79 of 80 subjects have a per-subject
AUC of 0.50–0.52; exactly one (1142) reaches 0.64. Membership leakage is statistically
detectable for 70% of outliers and 28% of controls while remaining materially negligible
for all but one person. Any statement of the form "outlier subjects can be identified"
is unsupported; "outlier subjects leak measurably more, and one of them leaks enough to
matter" is what the data shows.

**Not supported.** n = 3, so the SD estimate is itself unstable and the t(2) tails are
heavy. The outlier arm is the **same 20 subjects** in all three replicates — only 20
survive all four detection seeds — so this contains no outlier-sampling variance. One
generator, one capacity, one denoising-step setting.

---

## The supervised single-model attacker, and the power check (2026-08-13)

The result above is computed from `d_OUT − d_IN`, which needs BOTH models. That is the
study's instrument, not an adversary. This section asks the operational question
instead — **what can someone holding one released set do?** — and then asks the question
that makes any negative answer worth reading: **could the attacker have found leakage if
it were there?**

### The attacker

A classifier scores one window against one release; membership is never a difference of
two releases at inference. Trained across subjects, leave-one-subject-out, so it must
generalise to a person it has never seen — which is the situation a real adversary is
in, since nobody has labelled data for the person they want to test.

```
row   = (subject u, release include_v, window x)      label = (u == v)
LOSO  = the held-out subject is removed from training in EVERY replicate
score = per-subject AUC over that subject's own windows, member vs non-member condition
```

`base` is excluded from the releases used: all 40 targets are non-members of it, so
"is this base" would predict the label without any membership information.

### What it found on DiM-TS: nothing, in eight cells

```
classifier x features    outlier   control    diff     negative control
C1_logreg x euc_min       0.5138   0.5001   +0.0137   0.5013 / 0.5003
C1_logreg x raw10         0.5079   0.4987   +0.0092   0.4968 / 0.4966
C2_tree   x raw10         0.5034   0.4997   +0.0037   0.5039 / 0.5007
C4_hgb    x raw10         0.5016   0.5007   +0.0009   0.5090 / 0.5033
C5_svm    x raw10         0.4983   0.5003   -0.0020   0.5048 / 0.5008
C3_forest x raw10         0.4974   0.5026   -0.0053   0.5034 / 0.5018
C4_hgb    x cheap4        0.4941   0.5013   -0.0072   0.5083 / 0.5030
C6_mlp    x raw10         0.4877   0.5014   -0.0138   0.4961 / 0.4997
```

Best group difference +0.0137; half the cells are negative. **Complexity does not help
and mostly hurts** — one feature with logistic regression beats every tree, kernel and
network on ten features, because the signal is small enough that capacity buys
overfitting rather than sensitivity.

Only subject **1142** stands out, and it does so in six of the eight cells: 0.717 at
best (C1 x euc_min), against its own negative control of 0.565, so **+0.152 net**. The
best control subject reaches 0.568.

### The power check: `copy_paste`, and why 0.82 is a pass

A negative control proves the panel does not produce false POSITIVES. It says nothing
about false negatives, and "eight attackers found nothing" is evidence of absence only
if those attackers can find something when it is present.

`copy_paste` releases training rows verbatim. Same design, same features, same
classifiers, same LOSO — only the generator differs, so this measures the INSTRUMENT.

```
                        outlier   control    negative control
C1_logreg x euc_min      0.8268   0.8236     0.5003 / 0.5002
C2_tree   x raw10        0.8230   0.8218     0.4997 / 0.5000
C5_svm    x raw10        0.8218   0.8261     0.5006 / 0.5001
C6_mlp    x raw10        0.8196   0.8212     0.4991 / 0.5011
C4_hgb    x raw10        0.8187   0.8180     0.5000 / 0.5023
C3_forest x raw10        0.8186   0.8236     0.5012 / 0.5031
C4_hgb    x cheap4       0.8180   0.8164     0.5070 / 0.4970
C1_logreg x raw10        0.8171   0.8233     0.4988 / 0.4988
```

**0.82 is not "only 82%" — it is this control's ceiling.** `copy_paste` samples K rows
WITH replacement (`rng.integers`), so a bootstrap covers `1 − e⁻¹ = 63.2%` of the
training windows. A subject's windows are therefore verbatim-present 63.2% of the time
(distance ~5e-5, perfectly separable) and absent otherwise (chance):

```
predicted AUC = 0.632 x 1.0 + 0.368 x 0.5 = 0.816      measured 0.817-0.827
```

The panel reads the leakage that is theoretically present, no more and no less. Reading
1.0 would have meant something ELSE was leaking.

The two arms are also equal on `copy_paste` (0.8187 vs 0.8180), as they must be: a
generator that memorises indiscriminately separates neither arm. **One control
establishes sensitivity and specificity at once.**

```
sensitivity   0.82 on known leakage, against a 0.816 ceiling
specificity   0.50 on every negative control
DiM-TS        0.50
```

**The DiM-TS null is a real null, not a blind instrument.**

### Scaling up would not change it

The 60 controls give the null distribution of per-subject AUC: mean 0.5018, SD 0.0079.
The expected maximum of n draws is `mean + SD x Φ⁻¹(1 − 1/(n+1))`:

```
n = 60    predicted max 0.5187    observed 0.5182     (the model checks out)
n = 875   predicted max 0.5259
```

**Measuring the whole cohort would need 875 include models — about 4,600 GPU.h — to
move the most exposed normal subject from 0.518 to roughly 0.526.** Not worth running.

1142 at 0.6427 is **17.8 SD** above the control mean. That is not an extreme draw from
the null; no cohort size produces it by sampling. Whatever 1142 is, it is not the tail
of the normal distribution.

### The three claims this study can make

1. **Outlier subjects leak more than day-matched controls.** AUC 0.68, three
   independent replicates, replicate-level p ≈ 0.0005. AUC is invariant to any monotone
   transform of the gap, so this is a statement about ordering and overlap — it carries
   no magnitude.
2. **Both groups' absolute risk is negligible.** 79 of 80 subjects sit at per-subject
   AUC 0.50–0.52; a realistic single-model attacker reads ~0.51 on both arms; eight
   supervised attackers reach at best +0.014 between groups. The magnitude behind the
   0.68 is a median gap of 0.27% of the absolute distance for outliers and 0.11% for
   controls.
3. **One exception, and it is not sampling noise.** Subject 1142: per-subject AUC 0.643
   paired, 0.717 under the single-model panel, +0.152 net of its own null, first in all
   three replicates, 17.8 SD above the control mean.

Read together: **membership leakage here is measurable, systematic, and materially
negligible — for everyone except one person.** That is a bounded safety result, and the
power check is what makes the bound worth quoting.

---

---

## The question

Does a generative model trained on CGM data leak *membership of the subjects who are
outliers*? Not "does the cohort leak on average" — a previous population-level study on
this data found nothing — but whether the unusual subjects are the ones exposed.

## The design, in one box

Per replicate, 875 subjects split into 20 stable outliers, 20 drawn controls and an 835
background:

```
base       trained on the 835 background          -> d_OUT for all 40 targets
include_t  trained on the background plus t       -> d_IN  for target t
statistic  gap = d_OUT - d_IN, frozen as min x mean
question   does the OUTLIER arm's gap exceed the CONTROL arm's (AUC vs 0.5)
```

41 models per replicate, **3 replicates, 123 models**. Full derivation in
`docs/DESIGN.md`.

## The four decisions that shape it

| decision | value | why not otherwise |
|---|---|---|
| symmetric pairing | controls drawn OUT of the background; every pair is `(include_t, base)` | reusing one base as OUT for outliers and IN for controls puts its single-run offset `δ` into the arms with **opposite sign**, and `δ` is shared within an arm so 20 targets do not average it out. `docs/DESIGN.md` |
| 3 replicates | disjoint control draws → independent background, base and arm difference | within one replicate all 40 targets share a base, so a p-value over them is overstated. Replicate-level differences are the independent units |
| capacity | `hidden_size = 128` → **2,762,291** parameters | it is the width DiM-TS's authors published (2.65 M on Stocks). 522,907 was our own "≈500 k" choice and `docs/DIMTS.md` says that regime is deliberately under-parameterised, i.e. biased toward a null |
| borderline subjects | tagged in `design.json`, **not** removed from the control pool | removing them leaves a hand-picked "most normal" control set, which tightens the null and makes separation *easier* to find |

## Two bugs this round, both of which would have wasted the campaign

- **`match_controls(seed=...)` was never read.** The draw was deterministic greedy, so
  three replicates varying only the seed would have shared one control set and measured
  only training noise — ~290 GPU·h for far less than it appeared to buy. Fixed:
  `n_candidates=k` picks uniformly among the k nearest *available* by day count, which
  randomises the draw without giving up the day matching. Regression test:
  `test_seed_actually_moves_the_draw`.
- **The `δ` sign flip above.** Not caught by the module's own docstring, which listed
  two other asymmetries and described base reuse as a pure saving.

### Two more, found launching the follow-up sweep on 2026-08-10

- **PBS `-v` strips double quotes from the value.** `-v PARAMS='{"hidden_size": 96}'`
  arrives in the job as `{hidden_size: 96}` and `json.loads` dies on
  `Expecting property name enclosed in double quotes` — visible in
  `logs/19_h{52,96}_base.log`, two jobs dead inside a second. Pass structure through a
  file or through separate scalar variables, never as inline JSON.
- **`run_loo.py` fed DiM-TS's params to whatever generator was named.** Switching
  `--generator timevae` still inherited `hidden_size=128`, `save_cycle`, `sample_batch`
  from the DiM-TS config — a cross-generator comparison silently run at one generator's
  hyperparameters. Now params are **not** inherited when the generator differs from the
  config's, and the job prints what it dropped:

  ```
  [loo] --generator timevae differs from the config's 'dimts'; NOT inheriting its params
        (['batch_size', 'hidden_size', 'sample_batch', 'save_cycle', 'steps'])
  ```

  This is the same multi-channel/multi-config assumption that produced `mmd_alpha` and
  the C=1 quality metrics: configuration written for one model reused for another
  without anything checking that it applies.

## Cost — measured, `results/probe/capacity.json`, 2026-08-07

```
hidden   params      ms/step   ms/sample   h/model   vs h=52
    52    522,907      32.4       30.60      3.33      1.00x
    96  1,604,851      33.2       46.96      4.18      1.26x
   128  2,762,291      39.1       59.89      5.16      1.55x
```

**Width is cheap; denoising is not.** 5.3× the parameters is **1.21×** the optimiser
step — 522 k parameters at batch 64 leaves an A100 latency-bound, so most of the extra
width is free. Sampling carries the increase (1.96×) and is 58% of the bill at h=128,
set by 500 denoising steps rather than by size.

The h=52 probe said 3.33 h and the real `base` run took 3.49 h (+5%), so budget
**5.4 h/model** at h=128.

```
123 models x 5.4 h = 635 GPU.h
12 cards, busiest shard holds 4 models = ~21.6 h per wave, 3 waves = ~65 h
storage    ~1.1 GB/model (train + samples x2 + 10 checkpoints at ~46 MB) = ~135 GB
```

## Queue plan

`g3` is congested (64 queued against 8 running, checked 2026-08-07 19:00), so the 12
cards come from queues with real capacity instead. Shape picks the queue; never `-q`.

| lane | shape | lands in | cards | shards | models/card | needs |
|---|---|---|---|---|---|---|
| a | `ngpus=4 ncpus=64` / 30 h | `glong` | 4 | 0–3 | 4 | 21.6 h |
| b | same | `glong` | 4 | 4–7 | 4, 3, 3, 3 | 21.6 h |
| c | `ngpus=3 ncpus=48` / 20 h | `g2` | 3 | 8–10 | 3 | 16.2 h |
| d | `ngpus=1 ncpus=16` / 20 h | `g1` | 1 | 11 | 3 | 16.2 h |

41 jobs over 12 shards puts 4 models on shards 0–4 and 3 on the rest, so the walltimes
are not uniform: `g2`/`g1` cap at 24 h and 20 h leaves those lanes 23% of headroom,
while `glong` starts at 24 h so 30 h is the shortest honest ask.

**`mem` is not ours to choose.** A GPU node is 4 cards / 128 cpus / 440 gb, and PBS
pins memory at **110 gb per card** whatever the request says: `mem=120gb:ngpus=3` was
admitted as 330 gb. Asking for less does not improve placement, so the figure in the
request is cosmetic. (Measured need is far below it either way — the `base` run peaked
at 1.7 GB RSS.)

**One wave per replicate**, chained with `-W depend=afterany`. 123 models cannot be
spread over 12 equal shards — 10–11 models per shard is ~38 h, past `g2`/`g1`'s 24 h
ceiling. A wave is the already-validated 41-model / 12-card / ~14 h shape, and each
wave that finishes is a *complete usable replicate* rather than a third of three.

## Where it stands

| stage | state |
|---|---|
| 1–3 cohort, 14 methods, consensus, seed stability (20/24) | done |
| 4 design | `results/design_sym/rep{1,2,3}`, 41 jobs each, 60 distinct controls, day matching 0.7/0.7/1.0% median and 0 over tolerance |
| 4 capacity | measured, table above; `hidden_size=128` committed to `configs/experiment.yaml` |
| 4 run | **complete**. 123/123 models, 36 shard tasks, `FAILED=0` throughout, every lane `Exit_status=0`. 132 GB in `results/runs/dimts_h128_rep{1,2,3}/` |
| 5 attack | complete for all three replicates → `results/attack/dimts_h128_rep{1,2,3}/` plus `_unmatched` companions |
| 6 quality | complete. h=52/96/128 at 4-16 seeds each, four alternative generators at 4 → `results/quality/` |
| 7 capacity/generator sweep | **complete 2026-08-13.** Frontier is monotone; `hidden_size=96` is the cheapest configuration no metric separates from the published width |
| 8 single-model attacker panel | complete → `results/attack_panel/table_pooled.json`. Eight classifier x feature cells, LOSO, best group difference +0.014 |
| 9 power check | **complete and passed.** Same panel reads 0.82 on `copy_paste` against a 0.816 ceiling, 0.50 on every negative control → `results/attack_panel/table_pooled_copypaste.json` |

### Campaign accounting

```
                    budgeted            actual
per model           5.4 h               5.3 h        (fit 314.3-329.7 min across 123)
total               635 GPU.h           ~655 GPU.h
wall clock          ~65 h               63.8 h
```

All three within 3% of the capacity probe's projection. Lane walltimes: `glong`
21:08–21:17, `g2` 15:55–16:05, `g1` 15:52–16:11 — every lane finished inside its
request with hours to spare.

### rep1 verification, 2026-08-08

```
41 job directories, exactly the design's 41           41 DISTINCT subject fingerprints
1 x 835 subjects (base) + 40 x 836 (include_t)        10 checkpoints per model
nan_frac 0 everywhere                                 41 GB
fit 317.8 / 319.4 / 329.4 min  (min/median/max, 3.6% spread)
sample_range lower bound -1.000 / -0.568 / -0.429     upper bound 1.000 throughout
```

One model's minimum sample touches the -1.0 clip boundary; that is one value out of
176 k and a floor, not a bias. The h=52 base hit -1.000 as its ordinary minimum, so the
wider model tracks the data range (-0.375 .. 1.000) noticeably better — a fidelity
signal to confirm properly with the vendored Context-FID / discriminative / predictive
metrics once all three replicates exist.

Timing held: 5.32 h/model measured against 5.4 h budgeted from the capacity probe.

**Lanes chain independently**, so waves overlap: lanes c and d were 5 hours into rep2
before lanes a and b finished rep1. The idle cost is therefore confined to within a
lane — shards 5, 6 and 7 hold 3 models against shard 4's 4, so three cards sat out
5.3 h waiting for their job to exit. About 16 GPU·h a wave, 7.5%. Not worth fixing:
letting an idle card pick up a neighbour's shard needs a cross-process lock, and
without one two cards would train the same job.
| tests | 62 passing |

The h=52 `base` in `results/runs/dimts/base/` belongs to the superseded asymmetric
design and is **not** reusable. `loo.train` now fingerprints the job's subject list into
`meta.json` and refuses to treat a finished directory as done when the fingerprint (or,
for pre-fingerprint runs, `n_subjects`) disagrees — every design has a job called
`base`, and silently reusing the wrong one is otherwise invisible downstream.

### Watching it

```bash
qstat -u $USER                                    # Q queued, H held on a dependency, R running
qstat -xf <jobid> | grep -E 'job_state|Exit_status|comment'
ls results/runs/dimts_h128_rep1/*/samples.npy | wc -l     # 41 when the wave is done
tail -f logs/13_<jobid>_shard<N>.log
```

A log whose mtime is advancing is the only reliable liveness check — `nvidia-smi` on the
login node reports zero busy GPUs on a full machine.

## After the run: no more training

The frozen statistic is raw-space Euclidean (`attack/statistic.py:window_distances`),
not TS2Vec — the encoder cache belongs to stage 2's detection methods. The attack is
BLAS on CPU; copy_paste's 49 models / 48 pairs took 20 min on `qdev`, so 3 replicates
is about an hour there. **Nothing downstream of stage 4 needs a GPU.**

## The follow-up sweep, 2026-08-10 — launched, and mostly lost

Follow-ups 1 and 3 below were launched together on the evening of 2026-08-10, deliberately
as **quality-only probes rather than full campaigns**: one base per configuration
(2 capacity + 4 generators = 6 models, ~40 GPU·h) to draw the quality-vs-cost frontier
first, and only then decide which configurations are worth a 41-model MIA campaign
(350+ GPU·h each). That ordering still stands.

### What survived

| run | shape | outcome |
|---|---|---|
| `dimts_h52_rep1/base` | `g2` 2 GPU / 8 h | **complete** — fit 198.5 min, `samples.npy`, `nan_frac=0`, range [-1, 1] |
| `dimts_h96_rep1/base` | same job | **trained, not sampled** — `ckpt_288/checkpoint-10.pt` at 100 k steps, killed ~26 % through the sampling loop |
| `timevae_base/base` | `gdev` 1 GPU / 2 h | walltime kill, `walltime 7242 exceeded limit 7200`. No artefacts |
| `fourier_diff_base/base` | `g1` 1 GPU / 20 h | log stops 4 min in, at the first training step. No artefacts, no cause recorded |
| `diffwave` + `diffusion_ts` | `g2` 2 GPU / 8 h | never started; queued behind our own capacity job, gone by 2026-08-12 |

Only the `timevae` death is diagnosed, and it is ours: `gdev`'s 2 h ceiling is below what
TimeVAE actually needs on 176 k windows. The other three wrote no PBS epilogue at all,
and by 2026-08-12 `qstat -x` had aged them out — **the capacity job had 8 h and died at
4 h 11 m, so it did not hit its own walltime**. That points at a site-side event on the
evening of 2026-08-10 rather than at anything in the code, but no evidence survives to
confirm it. Treat "the job vanished" as a possible outcome and re-check artefacts, not
just `qstat`.

### The one piece of luck, and the path it forces

h=96's **training is complete** — the loss was only the sampling pass. Recovering it by
retraining would cost 3.4 GPU·h for weights that already exist on disk.

`DiMTSGenerator.load()` could not do it: it restored `all_samples.npy` and nothing else,
so a run directory holding ten checkpoints but no sample file was unrecoverable. That is
the *same* missing path follow-up 2 needs — `sampling_timesteps` 500 → 50 is answered by
re-sampling trained weights, never by retraining. Building it once serves both:

```
scripts/resample.py --from-run <dir> --out <dir> [--sampling-timesteps N]
```

**Declared, because it is a real difference:** a rescued sample set is drawn from the
same EMA weights but not from the same RNG position an uninterrupted run would have
reached, since training consumes the stream before sampling. The K windows are i.i.d.
draws from the same model, so nothing downstream is biased — but h=96's released set is
not bit-reproducible from the original command line, and its `meta.json` records that.

## Declared follow-ups, in the order they are likely to be asked for

1. **Capacity sweep.** Reviewers will ask what capacity does regardless of which value
   we defend, so it is a declared variable rather than a defended constant. **One**
   replicate per extra width (41 models), not three — the replicates establish the
   effect at the primary capacity; the sweep only has to show the trend. This is why
   run directories carry the width: `results/runs/dimts_h128_rep1/`.
   It should run **before** the second generator, because it settles what a
   cross-generator comparison has to hold fixed: DiM-TS was run at its *authors'*
   published width, and `diffusion_ts`'s own default is 224 k parameters — matching on
   "each author's default" and matching on parameter count are different experiments
   and only one of them can be run.
2. **Sampling steps.** `sampling_timesteps` 500 → 50 via `fast_sample`. Answerable by
   **re-sampling a trained checkpoint**, which is why `loo.train` forces a durable
   workdir — ~25 GPU·h, no retraining.
3. **A second generator.** Costs are already measured: diffwave ≥23, diffusion_ts 109,
   fourier_diff 629 GPU·h per replicate-equivalent. Answers "is this DiM-TS-specific".
4. The editor component, out of scope here.

## The real-data identifiability ceiling, 2026-08-14

No generator anywhere. Leave-one-day-out subject retrieval on real windows
(`scripts/identifiability.py`): hold out a day, rank every subject by the distance from
that day to their nearest other day, record where the true subject lands. It bounds any
membership attack from above — a generator cannot leak an identity the data does not
carry — and it costs one CPU job rather than 40 trained models.

**Every day is ranked only against subjects from its own study.** Recording units, CGM
device, pump model, protocol and era are constant within a study and differ between
them, so cross-study retrieval can identify the *study* instead of the person. The
control costs a factor of 6 in apparent lift and almost nothing in top-1, which is
itself the evidence that it was needed. Cohort `metabonet_sid_c3`, 1,253 subjects on the
corrected `(source_file, id)` key, 30 days each. Top-1 / lift over the study's own
chance rate:

| space | all 3ch | CGM only | **basal only** | bolus only |
|---|---|---|---|---|
| quantile | 17.3% / 36× | 1.5% / 3.1× | **29.3% / 61×** | 1.2% / 2.6× |
| level | 14.9% / 31× | 1.1% / 2.3× | 14.3% / 30× | 2.6% / 5.4× |
| raw | 6.5% / 14× | 2.6% / 5.5× | 17.3% / 36× | 2.2% / 4.6× |
| spectrum | 7.5% / 16× | 1.3% / 2.7× | 10.0% / 21× | 2.4% / 5.0× |
| shape | 6.4% / 13× | 1.7% / 3.6× | 12.3% / 26× | 2.3% / 4.8× |
| acf | 2.3% / 4.8× | 0.8% / 1.6× | 8.1% / 17× | 1.4% / 2.9× |

**A day of CGM is close to anonymous; a day of basal is not.** basal alone is about 20×
more identifying than CGM alone, and beats all three channels together — the feature
matrix is column-standardised, so weak channels dilute a strong one. A basal profile is
a programmed schedule; glucose is homeostatically bounded and its effective
dimensionality is far below its 288 samples.

Two things this number depends on, both of which changed it:

- **ties are mid-ranked.** bolus is 95% exact zeros, so many subjects sit at exactly
  equal distance. Counting a tie as rank 0 reported **62% top-1 on bolus quantile**;
  after the fix, 1.2%. Almost all of bolus's apparent signal was that.
- **mixed recording units were ruled out, not assumed.** Every study's median CGM is
  140–163 mg/dL; there is no mmol/L cluster.

On the single-channel cohort the same measurement reproduces the outlier finding without
any generator: consensus outliers have median normalised rank 0.134 against 0.326 for
the rest (p < 0.0001), and subject 1142 sits at z = −4.0 on `level`. So the real data
holds a large outlier-versus-normal identifiability gap that the membership pipeline
converts into per-subject AUC 0.50–0.52. **The signal is lost in the pipeline, not
missing from the data** — which is the case against buying more generator capacity
before explaining where it goes.

## Open, not yet decided

- Whether a null at h=128 warrants going above the published width. That would leave
  the authors' envelope and needs its own justification; it is not the default.
- **Where the identifiability gap is destroyed.** Three candidates: (a) the generator
  never memorises the subject, (b) the attack statistic reads the wrong feature space —
  identifiability lives in `level`/`quantile`/`spectrum` while the panel uses Euclidean
  `min × mean`, (c) at N = 875 one subject is 0.11% of the training set. (b) needs no
  GPU at all: the 142 GB of synthetic samples under `results/runs/` are still on disk.
  (c) is answerable by training at small N, where one subject is a percent of the data.
- **Whether to retrain on basal rather than on more CGM capacity.** The ceiling says a
  perfect attack on CGM has 3–5× chance to work with, and on basal it has 61×.
