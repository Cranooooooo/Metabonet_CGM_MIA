# The paired include/exclude design

## What is measured

For a target subject `t`, two generators that differ by **exactly one subject**. The
20 controls are drawn out of the normal pool *before* the base is built, so both arms
are constructed the same way:

```
B         background: normals minus the 20 drawn controls   -> d_OUT for every target
B + t     background plus t                                 -> d_IN  for target t
```

With 875 subjects, 20 stable outliers and 20 controls: `|B| = 835`, one base model,
20 outlier include models and 20 control include models — **41 models per replicate**.

The statistic is `d_OUT − d_IN`. If including a subject pulls the released samples
closer to that subject, the gap is positive.

### Why the base must sit on the same side of every pair

The obvious saving is to leave the controls in the background and make the control arm
`(M0, M0 − c)`, reusing one base as the OUT model for outliers and the IN model for
controls. That is what this design did until 2026-08-07, and it is wrong in a way that
is not visible in the model count.

The base is **one training run**. If it happens to release samples a little further from
everything — call that offset `δ` — then

```
outlier gap = d(s, S_M0)   − d(s, S_M0+s)   shifts by  +δ
control gap = d(c, S_M0−c) − d(c, S_M0)     shifts by  −δ
```

The arms move apart by `2δ` for a reason that belongs to one training run rather than to
membership, and `δ` is **shared within each arm**, so averaging over 20 targets does not
remove it. It is the same quantity the study is trying to measure. With the controls
drawn out first, the base is the OUT model for all 40 pairs, `δ` shifts both arms
together and cancels in the arm comparison.

Two smaller asymmetries go with it: the arms compared `N+1` vs `N` and `N` vs `N−1`
(now both `|B|+1` vs `|B|`), and each `M0 − c` had its own distinct background of `N−1`
subjects, so the control arm's 20 models did not even share one background (now all 41
runs share `B`).

`build_design.py --asymmetric` still builds the old form, so `results/design/` stays
rebuildable.

## Replicates

Within one replicate every target shares one base model, so the 40 gaps are correlated
and a p-value that treats them as independent is overstated. That is not fixable inside
a replicate — it is what a replicate *is*.

The study runs **three**, each with a disjoint control draw, hence its own background,
its own base and its own arm difference. Those three differences are the independent
units a p-value can rest on. 3 × 41 = **123 models**.

The draw is randomised by `match_controls(seed=..., n_candidates=k)`: the target's `k`
nearest still-available candidates by day count, one picked uniformly. `k = 1` is the
old deterministic greedy draw. **`seed` was accepted and never read before
2026-08-07** — three replicates varying only the seed would have shared one control set
and measured only training noise.

What replicates do **not** give: there are only 20 stable outliers, so the outlier arm is
the same 20 people every time. They estimate the variance of the model and of the
control draw, not of the outlier sample.

## Which design directory is which

| directory | form | outliers | jobs | what it is |
|---|---|---|---|---|
| `results/design/` | asymmetric | 24 | 49 | the seed-2026 consensus. What `results/runs/copy_paste` and `results/attack/copy_paste` were computed against; kept so those stay attributable |
| `results/design_stable/` | asymmetric | 20 | 41 | superseded — built before the `δ` problem above was found. Not used for a result |
| `results/design_sym/rep{1,2,3}/` | symmetric | 20 | 41 each | **what the generator stage runs** |

The outlier count was 28 before `outliers/common.method_rng` gave each method its own
RNG stream (docs/PITFALLS.md §5), then 24, and is 20 after seed stability: 732, 1152,
819 and 909 are flagged under only some base seeds and each would cost two models.

Ten further subjects were flagged by some seed but not all — 303, 1120, 1153, 467, 254,
929 and the four above. They stay in the **control pool** and are tagged in
`design.json` rather than removed: removing them would leave the controls a hand-picked
"most normal" subset, which tightens the empirical null and makes separation *easier* to
find, not harder. Whether a drawn one behaves differently is then something the analysis
can check.

Nothing in a sample file records which design produced it, so designs are never
overwritten in place — the same reason `build_cohort.py` refuses to overwrite a cohort.
Each replicate also gets its own `results/runs/<generator>_rep<r>/`, because `run_loo.py`
skips any job that already has a `samples.npy` and every replicate has a job named
`base`.

## What the design can show

Everything except membership is held identical — the other 846 subjects, the
architecture, the schedule, the sampling seed. A difference in `d` is therefore
attributable to that one subject's presence and nothing else. That is stronger than
the population-level test it replaces, which could only ask whether a cohort leaked on
average.

## What it cannot show

- **One subject in 848 is 0.12% of the training data.** This is the realistic setting
  — a custodian trains on everything they have — and it is also the regime in which a
  previous population-level study on this data found nothing. A null result here is a
  statement about that regime, not about generative models.
- **Each subject gets one training run**, so the gap includes training noise. Nothing
  here separates a real effect from a lucky initialisation; that needs repeats, which
  are deliberately deferred until there is an effect worth hardening.
- **The outliers share `M0`.** Their comparisons are correlated, which matters for a
  p-value and not for whether two groups separate on a plot.

## Controls must be matched on day count

The 24 consensus outliers have a median of 129 complete days against the pool's 174.
Day count drives nearest-neighbour distance independently of membership, and it also
changes how many chances an attacker has. Drawing controls at random would confound
atypicality with data volume; they are matched on day count instead.

The realised matching is in `results/design/design.json` under `matching`: median
relative gap 0.0, worst 0.0045, none outside the 15% tolerance. Read it rather than
assume it — if the outliers had sat at the extreme of the day distribution there would
have been no close match to find, and the report is what says whether that happened.

## The released-set size is a second confound

`include_s` trains on `n_s` more windows than `base`, and with `K` = training-set size
it also *releases* `n_s` more samples — about 0.1%. Nearest-neighbour distance falls
with K on its own, so that alone pushes `d_IN` down and the gap up, in exactly the
direction that looks like leakage. `attack.statistic` cuts both released sets to the
same size before computing any distance (`match_k`, on by default). It is undone at
the attack, not at the source: `K` = training-set size is what a custodian would
actually release, and an attack cannot see a memorised window that was never released.

## The distance

`d` is chosen from a grid rather than asserted:

    set_reduce      min | mean     how one real window scores against the released set
    subject_reduce  min | q10 | mean   how a subject's windows are combined

`attack.statistic` computes all six on every run. Which one the study uses was fixed on
`copy_paste` — a generator that memorises by construction, where the answer is known
before the run — and is now frozen in `configs/experiment.yaml`. Choosing it on the
real generator's output would be selecting the test on the outcome.

**The choice: `min` × `mean`.** Two things decided it, neither of them "the biggest
gap" (`results/attack/copy_paste/variant_choice.json`):

- **Detection.** Every `set_reduce=mean` variant fails outright: adding one subject to
  178k released samples does not move the mean distance to the released set (gap
  ~1e-5, Wilcoxon n.s.), where `min` gives p = 6e-8 in both arms.
- **Day-count dependence.** `subject_reduce=min` correlates with the subject's window
  count at Spearman −0.64, p = 0.0008, *within the control arm*. It takes the single
  most exposed window, so more windows means more draws and a lower minimum whether or
  not the subject was a member — the exact confound the matched controls exist to
  remove. `q10` and `mean` are both non-significant on that test; `mean` has the
  smaller coefficient and the larger effect-over-spread.

A note on both arms: an outlier's pair is `(include_s, base)` and a control's is
`(base, exclude_c)`, so **both are membership pairs**. A generator that memorises
indiscriminately makes both arms strongly positive and separates neither, which is
exactly what `copy_paste` does (AUC 0.54). "The arms did not separate" is therefore not
a failed sanity check on that run; the between-arm comparison only becomes informative
on a generator that does not memorise everything.

**What the design can resolve.** On copy_paste the outlier-minus-control difference is
0.21 pooled standard deviations. Two arms of 24 need roughly 0.8 to reach 80% power, so
this design can only resolve a large difference. That is a statement about one training
run per subject, not about the statistic: a smaller real effect needs repeats, not more
targets.
