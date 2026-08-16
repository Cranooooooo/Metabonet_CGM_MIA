# The fourteen outlier methods

A subject is "atypical" only relative to a choice of representation, a way of pooling
that subject's days, a reference to compare against, and a score. Those four choices
are independent, and the legal combinations number in the thousands. Picking one and
calling its output *the* outliers would be a result about the choice.

So fourteen are computed instead, spread deliberately across the design space, and a
subject is carried forward only when a **majority of them agree**: in the top 5% under
at least 7 of the 13 candidates. Two of the fourteen are controls and never vote.

Agreement across methods that share no machinery is the argument. A1 sees eight
clinical summary numbers; B5 sees the shape of days under an elastic distance; C8 sees
the whole distribution of windows without ever averaging them; D11 sees a learned
320-d embedding. A subject at the extreme of all four is not an artefact of any one.

Configuration lives in `configs/methods.yaml`; the table below is the reasoning.

## What each method is

| key | group | method | representation | subject aggregation | reference | score |
|-----|-------|--------|----------------|---------------------|-----------|-------|
| `A1` | A | Consensus-8 Mahalanobis | clinical (consensus 8) | mean | other subjects | robust Mahalanobis |
| `A2` | A | CV > 36% (consensus rule) | clinical (CV only) | mean | fixed threshold | threshold |
| `A3` | A | Cluster-representative Mahalanobis | clinical (5 cluster reps) | mean | other subjects | robust Mahalanobis |
| `A4` | A | Consensus-8 Isolation Forest | clinical (consensus 8) | mean | other subjects | Isolation Forest |
| `B5` | B | CID-DTW distance to cohort medoid | raw curve shape | mean over days | cohort medoid | CID-DTW |
| `B6` | B | Glucotype share vector | raw curve shape | share vector | cohort centroid | Euclidean |
| `B7a` | B | Raw k-NN distance (mean) | raw 288-vector | mean over days | other subjects' windows | k-NN distance |
| `B7b` | B | Raw k-NN distance (max) | raw 288-vector | max over days | other subjects' windows | k-NN distance |
| `C8` | C | MMD vs cohort (raw space) | raw 288-vector | none (set kept) | pooled cohort windows | MMD^2 (RBF) |
| `C9` | C | MMD vs cohort (TS2Vec space) | TS2Vec 320-d | none (set kept) | pooled cohort windows | MMD^2 (RBF) |
| `C10` | C | Sliced Wasserstein vs cohort | raw 288-vector | none (set kept) | pooled cohort windows | sliced Wasserstein-2 |
| `D11` | D | TS2Vec + Local Outlier Factor | TS2Vec 320-d | mean | other subjects | LOF |
| `D12` | D | Autoencoder error (max over days) | raw 288-vector | max over days | fitted model | reconstruction error |
| `E13` | E | Window count (negative control) | none | count | none | n_days |
| `E14` | E | Leave-self-out MMD (raw) | raw 288-vector | none (set kept) | cohort minus self | MMD^2 (RBF) |

## Why these fourteen
### Group A — Clinical, literature-grounded

Interpretable to a clinician and anchored in the international consensus on CGM metrics.
**`A1` Consensus-8 Mahalanobis**  
The eight metrics clinical papers actually report. An outlier here is explainable to a clinician in one sentence, which no embedding-space method can manage.

<sub>How: Per window compute the eight consensus metrics in mg/dL; average over the subject's days; scale each metric by median and IQR; fit a Minimum Covariance Determinant estimator on the cohort and take the Mahalanobis distance. MCD rather than the sample covariance because the outliers we are looking for would otherwise inflate the covariance they are measured against.</sub>
**`A2` CV > 36% (consensus rule)**  
Not our invention: the international consensus already defines unstable glycaemia this way. A zero-parameter, zero-fit baseline. If the fitted methods select the same people, the construct is credible; if they do not, that disagreement is itself a result.

<sub>How: Subject CV = mean over days of (SD / mean x 100). Score = CV - 36, so positive means unstable by the consensus definition.</sub>
**`A3` Cluster-representative Mahalanobis**  
Runs beside A1 to test whether picking metrics by discriminant ratio beats picking them by clinical consensus. The 32-metric battery collapses to five clusters, one holding sixteen metrics, so some reduction is forced; which reduction is a choice worth measuring.

<sub>How: As A1, on the highest-DR member of each correlation cluster.</sub>
**`A4` Consensus-8 Isolation Forest**  
Mahalanobis assumes an ellipse. Isolation Forest assumes nothing about shape, so a subject that is unusual in a corner of the space rather than far from the centre shows up here and not in A1.

<sub>How: 400 trees on the same standardised eight metrics; score is the negated average path length to isolation.</sub>
### Group B — Curve shape, raw space

Follows the glucotypes route: compare the shape of days directly, with no learned encoder and no feature engineering.
**`B5` CID-DTW distance to cohort medoid**  
The dissimilarity Hall et al. validated on CGM specifically. It captures fluctuation magnitude, rate of change and frequency at once, which a feature vector has to approximate one coordinate at a time.

<sub>How: Complexity-invariant DTW (Sakoe-Chiba band) from each of the subject's days to a set of cohort medoid days found by k-medoids on a window subsample; subject score is the mean of the per-day minima.</sub>
**`B6` Glucotype share vector**  
Reproduces a published CGM phenotype construct rather than inventing one. The output is a share over Low / Moderate / Severe glycaemic signatures, which is directly interpretable and already in the literature.

<sub>How: Spectral clustering on a CID-DTW dissimilarity matrix over a window subsample gives three signatures; every window is assigned to its nearest signature medoid; the subject becomes a 3-vector of time shares; the score is the distance from the cohort mean share.</sub>
**`B7a` Raw k-NN distance (mean)**  
The fit-free baseline the benchmark literature insists on. Together with B7b it isolates the aggregation choice with everything else held fixed.

<sub>How: For each of the subject's days, Euclidean distance to its k-th nearest window among OTHER subjects' windows; subject score is the mean.</sub>
**`B7b` Raw k-NN distance (max)**  
Identical to B7a except for the aggregation. If the two disagree about who the outliers are, the aggregation stage matters more than the choice of representation or score -- and it is the stage nobody reports.

<sub>How: As B7a, taking the maximum over the subject's days instead of the mean.</sub>
### Group C — Distribution-level, set preserved

The 2% of the design space that never averages a subject's days into one point -- the only route that can see a single unusual day.
**`C8` MMD vs cohort (raw space)**  
Compares the subject's whole distribution of days against the cohort's. A subject whose average day is ordinary but whose spread of days is not is invisible to every mean-aggregated method and visible here.

<sub>How: RBF-kernel MMD^2 between the subject's windows and a fixed reference sample of cohort windows; bandwidth by the median heuristic, computed once and shared so every subject is scored against the same kernel.</sub>
**`C9` MMD vs cohort (TS2Vec space)**  
Same statistic as C8 in a learned space. The pair isolates the effect of the representation with the score held fixed -- the only clean way to ask whether the embedding buys anything.

<sub>How: As C8, on frozen TS2Vec embeddings.</sub>
**`C10` Sliced Wasserstein vs cohort**  
MMD's sensitivity is set by a kernel bandwidth and saturates on far-apart distributions; Wasserstein moves mass and keeps growing. Two divergences that disagree tell you the difference is in the tail.

<sub>How: 200 random 1-D projections; per projection the W2 distance between sorted samples; score is the mean over projections.</sub>
### Group D — Learned representation

Deliberately thin: the benchmark literature finds deep anomaly detectors' advantage is largely an evaluation artefact.
**`D11` TS2Vec + Local Outlier Factor**  
Density relative to a subject's own neighbours rather than to the cohort centre. A small, sparse cluster of similar subjects scores high here and near zero on any centre-distance method.

<sub>How: Frozen TS2Vec embedding per window, mean-pooled to one vector per subject, LOF with k = 20 over subjects.</sub>
**`D12` Autoencoder error (max over days)**  
The only method in the set that targets a single unusual day directly. Memorisation attaches to a specific window, so this is the aggregation that matches the mechanism -- and the one no published CGM phenotyping work uses.

<sub>How: 1-D conv autoencoder trained on all windows unsupervised; per-window MSE; subject score is the maximum over that subject's days, with the 95th percentile also stored as a less brittle variant.</sub>
### Group E — Controls

Not candidates. They exist to catch the two confounds that would make everything else meaningless.
**`E13` Window count (negative control)**  
A subject with 700 days has more chances to look extreme on any per-day statistic, and more chances to be hit by a membership attack later. If the outlier lists track this, the study is measuring data volume. It is in the set to be checked against, not to be believed.

<sub>How: Number of CGM-complete days. No modelling.</sub>
**`E14` Leave-self-out MMD (raw)**  
C8 scores every subject against a reference pool that contains that subject's own windows, so a prolific subject drags the reference towards itself and looks more typical than it is. The gap between C8 and E14 is the size of that bias.

<sub>How: As C8, with the subject's own windows removed from the reference sample.</sub>

## The two controls

`E13` is the **window count**. Nothing about a subject's glucose enters it. If a real
method's flag list overlaps E13's, that method is partly detecting how long someone
wore the sensor. Measured overlap is near zero except for the two max-aggregating
methods (`B5`, `B7b` at 0.17), which is the expected direction: a maximum over days
has more chances to be large when there are more days.

`E14` is `C8` with the subject **removed from its own reference pool**. C8 leaves each
subject in the cohort it is compared against, which shrinks the apparent distance for
subjects contributing many windows. E14 measures how much that self-inclusion matters.

Neither control votes. Their job is to invalidate the others, not to add to them.

## `D12` is implemented but not run

The autoencoder method exists in `outliers/learned.py` and is excluded from the default
`--only` list. It needs a GPU and 20 epochs of training, and its output would be a
fifteenth vote on a list where the learned group is already represented by `C9` and
`D11`. Run it with `--only D12` if you want it; the consensus bar would then be 8 of 14.

## What the consensus cut is, and is not

`top 5%` and `7 of 13` are choices. The 5% cut is the conventional tail; the 7-vote
bar is the midpoint of the vote scale, and the vote histogram has a trough between 3
and 6 votes, so the bar sits in a flat region rather than on a slope. Neither was tuned
against any downstream result — the attack experiment did not exist when they were set,
which is the only thing that makes them honest.

Sensitivity to the cut is a one-line change (`--top-pct`, `--min-methods`) and belongs
in the paper as a table, not in this file as a claim.

## Seeds

Several methods subsample: 30 days per subject for the DTW pass, a few thousand
reference windows for the MMD and Wasserstein passes. Each method derives its own
stream from `(method key, base seed)`, so rerunning one method reproduces it exactly
regardless of what else the run contains — see `common.method_rng`. An earlier version
shared one generator across all methods in run order, which made every score depend on
the run's composition.

Whether a subject survives a change of seed is measured, not assumed:

```bash
python scripts/seed_stability.py --seeds 2026,7,101,999
```

Only subjects flagged under every seed should be carried into the attack experiment.

### What the sweep found (2026-08-07, `results/seed_stability/stability.json`)

Per-seed consensus sizes are 24, 24, 22 and 28 under seeds 2026, 7, 101 and 999.
**20 of the 24 seed-2026 outliers are flagged under all four**; 732 (3/4), 1152 (2/4),
819 (1/4) and 909 (1/4) are not, and six further subjects appear under some seeds only.
`results/design_stable/` is built on the 20 and is the design the generator stage uses.

### ⚠️ `B5` and `B6` do not survive a reseed, and they still vote

| method | median pairwise Spearman | share of a seed's top-5% that every other seed also flags |
|---|---|---|
| `B6` | 0.453 | **16%** |
| `B5` | 0.360 | **39%** |
| `B7b` | 0.982 | 74% |
| `D11` | 0.983 | 92% |
| everything else | ≥ 0.996 | ≥ 97% |

Both see 30 days per subject rather than all of them, which is what makes the elastic
distance affordable; at the 5% cut that subsample decides most of the list. They are
two of the thirteen votes, and at the ≥7 bar they can move a subject across it.

They are **deliberately still in the vote**. Dropping them turns "7 of 13" into "7 of
11" — the same threshold against a different denominator, which is a different claim
(`docs/PITFALLS.md` §6), and it would invalidate every score, the consensus and the
pinned regression fixture in one step. The instability is handled where it does damage
instead: by requiring a subject to survive all four seeds before it costs two models.

Anyone rerunning the consensus without `B5`/`B6` should treat it as a separate,
declared sweep with its own `--min-methods`, not as a correction to this one.
