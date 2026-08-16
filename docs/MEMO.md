# Memo: three-channel rerun vs longer windows — which is cheaper to answer

2026-08-14. Costing two proposals for extending the membership study, against measured
numbers rather than estimates. Sources: `results/probe/capacity.json` (GPU cost),
`results/channel_coverage_studyid/consecutive.csv` (contiguity), `qstat -Qf` (queue
limits, re-probed today), and the campaign accounting in `docs/PLAN.md`.

**Recommendation: run proposal 1 next, with no prerequisite. Proposal 2 only at
L = 7 days, if at all.** Proposal 1 costs the same GPU time as the campaign already run
(655 GPU·h, ~64 h wall clock) and needs one module changed. Proposal 2 cannot be run at
the fortnight or month the clinical claim is about, because the data does not contain
those windows. A third step this memo originally proposed as a free prerequisite — a
feature-space recomputation of the existing attack — has been **withdrawn**; its premise
did not survive checking. See the last section.

---

## Proposal 1 — rerun on CGM + basal + bolus

### GPU cost: 1.6x the run already completed — this section was wrong

**The estimate below replaces one that said "identical to the run already completed".**
That estimate came from the parameter count: `C` enters it only through `Linear(C, H)`
and `Linear(H, C)`, so C=1 and C=3 differ by **514 parameters** out of 2,762,805 —
`docs/DIMTS.md` measured this and it is correct. Parameter count is not compute. The
Mamba scan and the convolutions process three times the channel data per step, and the
cross-channel MMD term (`mmd_alpha` 0.0 → 0.0008 at C>1) does not execute at all at C=1.

Measured on the campaign itself, 24 completed models against the published 41:

```
                    three-channel        single-channel
per model            8.71 h               5.35 h            1.63x
ms/step              98.1                 39.1              2.51x
ms/sample            60.3                 59.9              1.01x
```

**The whole increase is in training; sampling is unchanged.** At C=1 the split was
2.17 h training / 2.99 h sampling; at C=3 it is 5.45 / 2.81. So "width is cheap,
denoising is expensive" — true of the single-channel table — does not hold here.

```
123 models x 8.71 h       = 1,071 GPU.h
12 cards                  = ~89 h wall clock
storage                   ~2.4 GB/model     = ~295 GB
```

### Window length is not the bottleneck: the attack ceiling is flat from 1 day to 21

`copy_paste` bootstraps K rows from its training set with replacement, so `1 - e^-1` =
63.2% of a target's windows come back verbatim and the rest are chance. It is the
instrument, not the subject: it measures what the ATTACK can detect when the leakage is
maximal and known, with the generator's fitting ability removed from the question.

Per-subject AUC_t, 40 targets each, `results/subject_auc_multiday/`:

| window | per-subject AUC | range | median gap (min×mean) | windows/subject |
|---|---|---|---|---|
| 1 day (published) | 0.8187 | — | 0.0680 | ~137 |
| 7 days, contiguous | 0.8147 | 0.699–1.000 | 0.0917 | 35 |
| 7 days, concat | 0.8111 | 0.755–0.879 | 0.0907 | 58 |
| 14 days, concat | 0.8229 | 0.668–0.933 | 0.0951 | 29 |
| 21 days, concat | 0.8236 | 0.665–1.000 | 0.1047 | 19 |
| arithmetic ceiling | 0.816 | | | |

**Every length lands in 0.811–0.824, a span of 1.3 points, against per-subject
confidence intervals of roughly ±0.06.** There is no trend.

The gap magnitude, meanwhile, rises monotonically: 0.068 → 0.105, up 54% from one day to
three weeks. The two diverge for a reason worth stating: a longer window makes a
memorised window stand out by a **larger absolute margin**, but the **fraction** of
windows released verbatim is 63.2% at every length, and a rank statistic sees only the
fraction.

**What this rules out.** "One day is too short for a membership signal to exist" is not
the explanation for the flat DiM-TS results. If a single day could not carry a
detectable identity, copy_paste could not reach 0.82 on single days — it does, and three
weeks does no better. The attack machinery works identically at 288 and at 6,048
dimensions.

**What it cannot answer.** copy_paste's ceiling is fixed by its bootstrap, so it says
nothing about whether a REAL generator memorises more from longer windows. That question
runs the other way — a 6,048-dimensional window is far harder to memorise than a
288-dimensional one, but leaks far more once memorised — and needs DiM-TS trained at
those lengths.

**Method note: the concat construction is validated.** Calendar-contiguous weeks exist
for only 573 subjects with ≥4 windows, fortnights and three-week runs for far fewer, so
14 and 21 days use `concat` — each subject's complete days in date order, chopped into
blocks of L, skipping gaps. 90% of concat windows at L=7 contain at least one seam and
about a third of all day-to-day transitions in them are fabricated. At L=7, where both
constructions exist, they read 0.8147 and 0.8111. The seams do not matter to copy_paste,
which is why the 14- and 21-day rows are usable.

### The three-channel campaign was stopped on 2026-08-16, at 44 of 123 models

Two reasons, and the second one was structural.

**It could not have finished as launched.** Every lane was killed by walltime:

```
lane a/b   4 models per card x 8.71 h = 34.8 h   against a 30 h request
lane c/d   3 models per card x 8.71 h = 26.1 h   against a 20 h request
```

The `LANES` table in `scripts/pbs/launch_campaign.sh` budgets from the single-channel
5.35 h/model. It was reused unchanged for a campaign whose models take 8.71 h, so all
twelve jobs were under-budgeted from submission. **Fix the walltimes before relaunching
anything from that script at C=3.**

**And the question it was answering had already been answered by replicate 1.** See the
per-channel results below: the attack finds one subject, in one channel, and adding
channels dilutes even that.

Kept on disk: `results/runs/dimts_c3_h128_rep1` (32 of 41 models) and `_rep2` (12).
Nine of replicate 1's pairs have no member model, so its arms are 13 outliers against
18 controls and are no longer day-matched — every arm-level number from it is
provisional. Per-subject AUCs are not affected: each uses only that subject's own
windows.

### Where the membership signal is, by channel

Per-subject AUC_t = P(d_out > d_in) over the subject's own windows, replicate 1,
`results/subject_auc_c3/`. 0.5 means the attack cannot tell, for that person, which
model saw him. Every channel subset uses the SAME 31 targets and the SAME models, so
the comparison between rows is clean even though the absolute values are provisional.

| channels | outlier mean | control mean | outliers excl. 1142 | Loop/1142 |
|---|---|---|---|---|
| CGM | 0.5131 | 0.4989 | 0.5029 | **0.6358** |
| basal | 0.5002 | 0.5164 | 0.5004 | 0.4979 |
| bolus | 0.4735 | 0.4983 | 0.4709 | 0.5045 |
| CGM+basal | 0.5033 | 0.4955 | 0.5029 | 0.5084 |
| CGM+bolus | 0.5016 | 0.4993 | 0.4986 | 0.5370 |
| basal+bolus | 0.5016 | 0.5005 | 0.5016 | 0.5009 |
| all three | 0.5023 | 0.5019 | 0.5009 | 0.5182 |

**One subject, one channel.** `Loop/1142` reads 0.636 on CGM with a 95% interval of
[0.615, 0.658] and p = 1e-31 — the same person, at the same strength, as the published
single-channel study's 0.643, reproduced across a rebuilt cohort, a corrected subject
key and a changed channel set. He is unremarkable on basal (0.498) and bolus (0.504).
Every other outlier sits between 0.479 and 0.516, indistinguishable from the controls.

**Adding channels dilutes the one real signal.** 1142 falls from 0.636 on CGM alone to
0.508 on CGM+basal — outside his own CGM confidence interval, so this is an effect and
not noise. The attack statistic is a Euclidean distance over the flattened window, and
`set_reduce=min` picks the nearest released sample **in the joint space**: adding 288
dimensions that carry no membership information changes which sample is nearest, and it
is chosen for matching basal rather than for matching CGM.

The consequence is a statement about the statistic, not about the data: **a
multi-channel attack should be run per channel and reported per channel.** Flattening
channels into one distance assumes they are homogeneous, and they are not — the same
conclusion the identifiability measurement reached from the other direction.

### More capacity does not close the three-channel quality gap

One `base` model per width on the c3 cohort, campaign budget (100k steps, batch 64),
`results/quality_c3_width/`. The h128 row is the campaign's own base, not a retrain.

| hidden | params | ContextFID | discriminative accuracy | predictive MAE |
|---|---|---|---|---|
| 128 | 2.76 M | 0.1044 | 0.595 | 0.0200 |
| 160 | 4.23 M | 0.0839 | 0.621 | 0.0196 |
| 192 | 6.01 M | 0.1139 | 0.617 | 0.0196 |
| 224 | ~7.9 M | 0.0628 | 0.643 | 0.0197 |
| 256 | 10.51 M | 0.0904 | 0.582 | 0.0195 |

**No trend.** 3.8× the parameters moves discriminative accuracy from 0.595 to 0.582.
The scale to read that against: three h128 models from the campaign itself scored 0.677,
0.553 and 0.653 — a spread of **0.124 at one width**, twice the 0.061 spread across all
five widths. The width effect is not separable from run-to-run noise. ContextFID has no
trend either, and predictive MAE is flat.

Together with the loss trace — converged by 20k steps, 1.6% gained over the next 70k —
**both of the obvious explanations for the gap are now excluded: not duration, not
capacity.** There is no reason to rerun the campaign at a larger width, which is what
this table was commissioned to decide.

The remaining candidate is the data. `bolus` is 95% exact zeros with occasional spikes;
a Gaussian diffusion process is built for smooth trajectories and a sparse spike train
is the opposite of its prior, so the classifier separating real from synthetic may be
reading that one channel. Testable with a single model: train CGM+basal only, two
continuous channels, and see whether the accuracy returns to 0.50. ~7 GPU·h.

### Cost by width at three channels

`results/probe_c3/capacity_c3.json`, measured 2026-08-15. The h128 row is a control:
the probe predicts 8.26 h/model where the campaign measures 8.71, a 5% error, which is
what makes the other rows usable as costs.

| hidden | params | ms/step | ms/sample | h/model | vs h128 | 123 models | 12-card days |
|---|---|---|---|---|---|---|---|
| **128** | 2,762,805 | 98.1 | 60.3 | 8.26 | 1.00× | 1,016 h | 3.5 |
| 160 | 4,231,669 | 104.8 | 78.4 | 9.47 | 1.15× | 1,165 h | 4.0 |
| 192 | 6,011,829 | 112.4 | 95.8 | 10.71 | 1.30× | 1,317 h | 4.6 |
| 256 | 10,506,037 | 133.8 | 143.2 | 14.10 | 1.71× | 1,734 h | 6.0 |

Width stays cheap even at 10.5 M parameters: 3.8× the parameters costs 1.71×, an
effective exponent of 0.40 — the same as the 96→128 step at one channel. The growth
that does appear is in **sampling** (60 → 143 ms), which is 500 denoising steps times
the width and has nothing to do with parameter count.

### The cohort already exists

`data/cohort/metabonet_sid_c3` — 1,253 subjects, 172,119 windows, verified, 0 duplicate
(subject, day). Built on the corrected `(source_file, id)` key, which also grows the
usable cohort from 875 to 1,253 and is a correction that has to happen regardless of
which proposal runs.

### What has to be written: the outlier stage, and only it

| stage | multichannel today | why |
|---|---|---|
| cohort | **yes** | `scripts/build_cohort_multi.py`, done and verified |
| outlier detection | **no** | `outliers/run.py:35` and `clinical.py:30` take `X[:, :, 0]` |
| design / matching | yes | operates on subject ids and day counts, not on windows |
| generator | yes | DiM-TS takes `C` from the data; the adapter passes it through |
| attack | **yes** | `attack/statistic.py:_flat` and `attack/panel.py:_flat` reshape `(N, T, C) → (N, T·C)`; every feature space is already channel-agnostic |

Only the outlier stage needs work, and it splits cleanly:

- **Group A (A1–A4), 4 of 14 methods: cannot be extended, and should not be.** They are
  clinical metrics — CV > 36%, TIR, MAGE, GRI — defined on glucose in mg/dL. There is no
  "time in range" for a basal rate. Run them on CGM as they are.
- **Groups B, C, D (B5–B7b, C8–C10, D11–D12), 9 methods: extend by flattening.** They
  consume a raw 288-vector or a TS2Vec embedding; a 288×3 window flattens to 864 the
  same way the attack already does. TS2Vec takes a channel dimension natively.
- **Group E (E13, E14): E13 is window count, channel-agnostic. E14 is MMD, flattens.**

**The consensus denominator moves, and that is a known trap** (`docs/PITFALLS.md` §6).
The current rule is "flagged by ≥ 7 of 13 candidates". If group A stays CGM-only while
the rest see three channels, the methods are no longer measuring the same object and
the vote is not comparable to the published one. Two defensible options, and the choice
should be made before running, not after seeing the outlier list:

1. **run the consensus twice** — 13 candidates on CGM (reproducing the published 20) and
   9–10 on three channels — and report the overlap as a result in its own right;
2. **drop group A** and run a 9–10 method consensus on three channels only, re-deriving
   the threshold from seed stability rather than reusing "≥ 7".

Option 1 is more work in analysis and no more in compute, and it answers "does adding
channels change who the outliers are", which is the first thing a reviewer will ask.

### Estimated effort

```
outlier stage multichannel + tests        ~0.5 day
consensus + seed stability (CPU)          hours, qdev
design (CPU)                              minutes
generator campaign                        655 GPU.h / 63.8 h wall clock
attack + panel (CPU)                      as before
```

**Time to a first answer: about three days**, of which 2.7 are the generator campaign
running unattended.

---

## Proposal 2 — a week / fortnight / month as one window

### The data does not contain the windows the clinical claim is about

Measured today, `scripts/consecutive_days.py`. Subjects holding at least one run of L
**consecutive** complete days, among those clearing the 30-complete-day bar:

| L (days) | CGM only | CGM+basal+bolus | non-overlapping windows (3ch) |
|---|---|---|---|
| 1 | 1,329 (100%) | 1,253 (100%) | 172,119 |
| **7** | 809 (60.9%) | **761 (60.7%)** | 6,514 |
| **14** | 40 (3.0%) | **33 (2.6%)** | 88 |
| **30** | 5 (0.4%) | **2 (0.2%)** | 4 |

Median longest run per subject: **8 days**. p90: **9 days**.

The cohort's days are complete *individually*; nothing had checked that they are
contiguous, and mostly they are not. The clinical reading that motivates this proposal —
about fifteen days pins an individual down — needs a fortnight, and **33 subjects have
one**. A design needing 20 outliers and 20 day-matched controls cannot be drawn from 33
subjects, let alone from the 2 with a month.

**Imputing the gaps is not available here.** Filling missing hours injects a model's
estimate of the subject into the data a membership measurement is about to read. The
measurement asks how much a generator knows about a subject; pre-loading the answer into
its input makes the result uninterpretable in the direction that flatters the
hypothesis.

So the only runnable length is **L = 7**, one day past the median run, and it keeps 61%
of subjects but only **6,514 non-overlapping windows — 26% of the data** by timestep
count (13.1 M against 49.6 M). Sliding at stride 1 gives 15,211 windows, but neighbours
share six of seven days: the effective sample size is still the non-overlapping count.

### GPU cost at L = 7: about 1.6x proposal 1

`T` is what this architecture scales with, and the training budget is a **fixed 100,000
optimiser steps** (`generators/dimts.py:116`), not a fixed number of epochs. The Mamba
selective scan is linear in `T`, so:

```
                        T=288 (measured)      T=2016 (projected, linear in T)
ms/step                    39.1                   ~274
training  100k steps        1.1 h                  ~7.6 h
K (= training set size)  ~178,000 windows        6,514 windows
ms/sample                  59.9                   ~419
sampling                    3.0 h                  ~0.8 h
per model                   5.3 h (actual)         ~8.4 h
123 models                655 GPU.h              ~1,030 GPU.h
```

Sampling gets *cheaper* because there are far fewer windows to generate; training gets
much more expensive because the step count is fixed. Note the second-order problem in
that same fact: 100 k steps at batch 64 over 6,514 windows is **983 epochs** against
about 36 today. That is a different memorisation regime, not the same experiment at a
longer window — and `docs/PITFALLS.md` §11 is precisely about budgets that are quietly
a property of the dataset. The step count would have to be re-derived, which makes the
cost above a floor rather than an estimate.

Also unwritten: a contiguity-aware cohort builder, and a check that `T = 2016` fits the
model's positional encodings and memory at batch 64.

### Estimated effort

```
contiguity-aware builder + windowing      ~0.5 day
re-derive the training budget for T=2016  a capacity probe, ~2 GPU.h
generator campaign                        ~1,030 GPU.h / ~86 h wall clock
```

**Time to a first answer: about five days**, and the answer is only about a week, not
about the fortnight the proposal is motivated by.

---

## Can both run at the same time? No.

Queue limits, re-probed 2026-08-14 with `qstat -Qf`:

| queue | walltime | jobs running per user |
|---|---|---|
| `gdev` | ≤ 2 h | 10 — but a **16-card pool for the whole machine** |
| `g1` | 2–24 h | **1** |
| `g2` | 2–24 h | **1** |
| `g3` | 2–24 h | **1** |
| `glong` | 24–120 h | **2** |

Above two hours a user can hold **five jobs total**. The validated campaign shape already
spends four of them — `glong`×2 at 4 cards, `g2` at 3, `g1` at 1 = 12 cards — and it was
built that way because `g3` was congested. Two campaigns at once would want ~24 cards
across eight lanes; the ceiling is 16 cards across five, and only if `g3` cooperates.

**Running both concurrently does not increase throughput, it splits it.** The same
cards do the same total work; each campaign would take roughly twice as long, and both
would land later than running them one after the other. `gdev` does not help: a 5–8 h
model does not fit a 2 h walltime, and chunking training across `gdev` jobs would be a
new checkpoint-resume mechanism nobody has validated.

The honest schedule is **sequential**, and that is an argument for starting with the
cheaper one.

---

## Recommendation

**Run proposal 1 first.** It costs GPU time already known to fit (655 GPU·h, 63.8 h),
reuses a cohort that is built and verified, needs one module extended, and answers the
question directly: with three channels, who are the outliers, do outliers and normals
differ in leakage, and is the difference significant.

There is also independent evidence that it is the right channel to spend on. The
real-data identifiability ceiling (`docs/PLAN.md`, 2026-08-14, within-study control):
one day of **basal** identifies its subject 29.3% of the time, 61× chance; one day of
**CGM** manages 1.5%, 3.1× chance. A perfect attack on CGM has 3–5× chance to work with;
on basal it has 61×. Proposal 1 puts that channel in front of the attack.

**Hold proposal 2.** At L = 14 and L = 30 it is not runnable on this data, and at L = 7
it costs 1.6× more to answer a question the clinical literature does not actually make
about seven days. If it is run later, run it *after* proposal 1, on the same cohort, so
the two differ only in window length.

### Withdrawn: recomputing the existing attack in another feature space

An earlier draft of this memo proposed, as a free prerequisite, recomputing the attack
statistic on the single-channel samples already on disk (`results/runs/`, 142 GB, no
retraining) using `quantile` or `spectrum` distances instead of the Euclidean
`min × mean` the panel uses. **The premise was wrong and the step is not worth taking.**

The claim was that identifiability lives in `level`/`quantile`/`spectrum` while the
attack reads Euclidean. Checking the within-study CGM numbers that claim was drawn from:

```
raw       top1 3.70%   AUC 0.698    <- what the attack already uses
quantile       2.13%       0.692
shape          2.45%       0.624
level          1.86%       0.667
spectrum       1.84%       0.670
acf            1.39%       0.575
```

On CGM the raw Euclidean space is the **best** of the six, not a poor choice. Two
different quantities had been conflated: `level`/`quantile`/`spectrum` are where the
*outlier-versus-normal gap* is widest, which is a question about which feature separates
two groups of people. Which distance an attack should use is a question about absolute
retrieval, and raw wins that one.

What survives: the extra feature transforms are worth carrying **inside** proposal 1
rather than ahead of it. `attack/statistic.py:gap_for_pair` already sweeps a grid
(`set_reduce` × `subject_reduce`, all six combinations on every run), so adding a
feature transform to that grid is marginal cost on a pass that is happening anyway — and
it would then be evaluated on basal, where `quantile` reaches 29.3% top-1 against CGM's
1.5%, which is the only place it has a plausible reason to matter.
