"""The 14 outlier-scoring methods, with the reason each is in the set.

Selected from a design space of 4,810 single methods (13 window representations x 74
legal aggregation/score pairs x 5 references). Three principles drove the cut:

 1. COVER THE AGGREGATION AXIS. Averaging a subject's ~176 days answers "is this
    person unusual on average"; memorisation attaches to one specific window, so
    "does this person have one unusual day" is the question that matches the
    mechanism. Five of the fifteen score columns here do not average it away.
 2. EVERY GROUP CARRIES A FIT-FREE BASELINE. Wu & Keogh showed that time-series
    anomaly benchmarks are flawed badly enough that deep methods' apparent advantage
    is often an artefact of the evaluation; without a trivial baseline in the set
    there is no way to tell a real gain from a broken comparison.
 3. CLINICAL READABILITY IS NOT OPTIONAL. Group A and B6 produce outliers a
    diabetologist can argue with. Groups C and D can only say "atypical in some
    embedding", which a JBHI reviewer will not accept as the whole story.

Literature the choices rest on:
  * Battelino et al., Diabetes Care 2019 -- international consensus on CGM metrics.
    Fixes the eight metrics of group A and, in CV > 36%, gives a threshold for
    "unstable glycaemia" that clinicians already use. Also sets the data-sufficiency
    bar we inherit: >= 14 days at >= 70% capture.
  * Hall et al., PLOS Biology 2018 ("glucotypes") -- CID-DTW dissimilarity plus
    spectral clustering on CGM windows. The validated shape-based route on this exact
    modality, and the basis of B5/B6.
  * Wu & Keogh; TimeSeAD -- why the deep end of the set is deliberately thin.
"""
from __future__ import annotations

# Consensus metrics (Battelino 2019). Keys match m7mia.clinical.cgm_metrics.
CONSENSUS = ["tir_70_180", "tbr_70", "t_lt_54", "tar_180", "t_gt_250",
             "mean", "gmi", "cv"]
# Highest-DR representative of each of the five correlation clusters at |r_s| >= .85
CLUSTER_REPS = ["conga1", "grade_hypo", "grade_hyper", "cv", "gri"]
CV_UNSTABLE = 36.0          # consensus threshold for unstable glycaemia

METHODS = [
    dict(key="A1", group="A", name="Consensus-8 Mahalanobis",
         repr="clinical (consensus 8)", agg="mean", ref="other subjects", score="robust Mahalanobis",
         tail="mean",
         why="The eight metrics clinical papers actually report. An outlier here is "
             "explainable to a clinician in one sentence, which no embedding-space "
             "method can manage.",
         how="Per window compute the eight consensus metrics in mg/dL; average over the "
             "subject's days; scale each metric by median and IQR; fit a Minimum "
             "Covariance Determinant estimator on the cohort and take the Mahalanobis "
             "distance. MCD rather than the sample covariance because the outliers we "
             "are looking for would otherwise inflate the covariance they are measured "
             "against."),
    dict(key="A2", group="A", name="CV > 36% (consensus rule)",
         repr="clinical (CV only)", agg="mean", ref="fixed threshold", score="threshold",
         tail="mean",
         why="Not our invention: the international consensus already defines unstable "
             "glycaemia this way. A zero-parameter, zero-fit baseline. If the fitted "
             "methods select the same people, the construct is credible; if they do "
             "not, that disagreement is itself a result.",
         how="Subject CV = mean over days of (SD / mean x 100). Score = CV - 36, so "
             "positive means unstable by the consensus definition."),
    dict(key="A3", group="A", name="Cluster-representative Mahalanobis",
         repr="clinical (5 cluster reps)", agg="mean", ref="other subjects",
         score="robust Mahalanobis", tail="mean",
         why="Runs beside A1 to test whether picking metrics by discriminant ratio "
             "beats picking them by clinical consensus. The 32-metric battery collapses "
             "to five clusters, one holding sixteen metrics, so some reduction is "
             "forced; which reduction is a choice worth measuring.",
         how="As A1, on the highest-DR member of each correlation cluster."),
    dict(key="A4", group="A", name="Consensus-8 Isolation Forest",
         repr="clinical (consensus 8)", agg="mean", ref="other subjects",
         score="Isolation Forest", tail="mean",
         why="Mahalanobis assumes an ellipse. Isolation Forest assumes nothing about "
             "shape, so a subject that is unusual in a corner of the space rather than "
             "far from the centre shows up here and not in A1.",
         how="400 trees on the same standardised eight metrics; score is the negated "
             "average path length to isolation."),

    dict(key="B5", group="B", name="CID-DTW distance to cohort medoid",
         repr="raw curve shape", agg="mean over days", ref="cohort medoid",
         score="CID-DTW", tail="mean",
         why="The dissimilarity Hall et al. validated on CGM specifically. It captures "
             "fluctuation magnitude, rate of change and frequency at once, which a "
             "feature vector has to approximate one coordinate at a time.",
         how="Complexity-invariant DTW (Sakoe-Chiba band) from each of the subject's "
             "days to a set of cohort medoid days found by k-medoids on a window "
             "subsample; subject score is the mean of the per-day minima."),
    dict(key="B6", group="B", name="Glucotype share vector",
         repr="raw curve shape", agg="share vector", ref="cohort centroid",
         score="Euclidean", tail="mean",
         why="Reproduces a published CGM phenotype construct rather than inventing one. "
             "The output is a share over Low / Moderate / Severe glycaemic signatures, "
             "which is directly interpretable and already in the literature.",
         how="Spectral clustering on a CID-DTW dissimilarity matrix over a window "
             "subsample gives three signatures; every window is assigned to its nearest "
             "signature medoid; the subject becomes a 3-vector of time shares; the score "
             "is the distance from the cohort mean share."),
    dict(key="B7a", group="B", name="Raw k-NN distance (mean)",
         repr="raw 288-vector", agg="mean over days", ref="other subjects' windows",
         score="k-NN distance", tail="mean",
         why="The fit-free baseline the benchmark literature insists on. Together with "
             "B7b it isolates the aggregation choice with everything else held fixed.",
         how="For each of the subject's days, Euclidean distance to its k-th nearest "
             "window among OTHER subjects' windows; subject score is the mean."),
    dict(key="B7b", group="B", name="Raw k-NN distance (max)",
         repr="raw 288-vector", agg="max over days", ref="other subjects' windows",
         score="k-NN distance", tail="max",
         why="Identical to B7a except for the aggregation. If the two disagree about "
             "who the outliers are, the aggregation stage matters more than the choice "
             "of representation or score -- and it is the stage nobody reports.",
         how="As B7a, taking the maximum over the subject's days instead of the mean."),

    dict(key="C8", group="C", name="MMD vs cohort (raw space)",
         repr="raw 288-vector", agg="none (set kept)", ref="pooled cohort windows",
         score="MMD^2 (RBF)", tail="set",
         why="Compares the subject's whole distribution of days against the cohort's. A "
             "subject whose average day is ordinary but whose spread of days is not is "
             "invisible to every mean-aggregated method and visible here.",
         how="RBF-kernel MMD^2 between the subject's windows and a fixed reference "
             "sample of cohort windows; bandwidth by the median heuristic, computed once "
             "and shared so every subject is scored against the same kernel."),
    dict(key="C9", group="C", name="MMD vs cohort (TS2Vec space)",
         repr="TS2Vec 320-d", agg="none (set kept)", ref="pooled cohort windows",
         score="MMD^2 (RBF)", tail="set",
         why="Same statistic as C8 in a learned space. The pair isolates the effect of "
             "the representation with the score held fixed -- the only clean way to ask "
             "whether the embedding buys anything.",
         how="As C8, on frozen TS2Vec embeddings."),
    dict(key="C10", group="C", name="Sliced Wasserstein vs cohort",
         repr="raw 288-vector", agg="none (set kept)", ref="pooled cohort windows",
         score="sliced Wasserstein-2", tail="set",
         why="MMD's sensitivity is set by a kernel bandwidth and saturates on far-apart "
             "distributions; Wasserstein moves mass and keeps growing. Two divergences "
             "that disagree tell you the difference is in the tail.",
         how="200 random 1-D projections; per projection the W2 distance between sorted "
             "samples; score is the mean over projections."),

    dict(key="D11", group="D", name="TS2Vec + Local Outlier Factor",
         repr="TS2Vec 320-d", agg="mean", ref="other subjects", score="LOF",
         tail="mean",
         why="Density relative to a subject's own neighbours rather than to the cohort "
             "centre. A small, sparse cluster of similar subjects scores high here and "
             "near zero on any centre-distance method.",
         how="Frozen TS2Vec embedding per window, mean-pooled to one vector per subject, "
             "LOF with k = 20 over subjects."),
    dict(key="D12", group="D", name="Autoencoder error (max over days)",
         repr="raw 288-vector", agg="max over days", ref="fitted model", score="reconstruction error",
         tail="max",
         why="The only method in the set that targets a single unusual day directly. "
             "Memorisation attaches to a specific window, so this is the aggregation that "
             "matches the mechanism -- and the one no published CGM phenotyping work uses.",
         how="1-D conv autoencoder trained on all windows unsupervised; per-window MSE; "
             "subject score is the maximum over that subject's days, with the 95th "
             "percentile also stored as a less brittle variant."),

    dict(key="E13", group="E", name="Window count (negative control)",
         repr="none", agg="count", ref="none", score="n_days", tail="control",
         why="A subject with 700 days has more chances to look extreme on any per-day "
             "statistic, and more chances to be hit by a membership attack later. If the "
             "outlier lists track this, the study is measuring data volume. It is in the "
             "set to be checked against, not to be believed.",
         how="Number of CGM-complete days. No modelling."),
    dict(key="E14", group="E", name="Leave-self-out MMD (raw)",
         repr="raw 288-vector", agg="none (set kept)", ref="cohort minus self",
         score="MMD^2 (RBF)", tail="set",
         why="C8 scores every subject against a reference pool that contains that "
             "subject's own windows, so a prolific subject drags the reference towards "
             "itself and looks more typical than it is. The gap between C8 and E14 is "
             "the size of that bias.",
         how="As C8, with the subject's own windows removed from the reference sample."),
]

GROUPS = {
    "A": dict(name="Clinical, literature-grounded",
              why="Interpretable to a clinician and anchored in the international "
                  "consensus on CGM metrics."),
    "B": dict(name="Curve shape, raw space",
              why="Follows the glucotypes route: compare the shape of days directly, "
                  "with no learned encoder and no feature engineering."),
    "C": dict(name="Distribution-level, set preserved",
              why="The 2% of the design space that never averages a subject's days into "
                  "one point -- the only route that can see a single unusual day."),
    "D": dict(name="Learned representation",
              why="Deliberately thin: the benchmark literature finds deep anomaly "
                  "detectors' advantage is largely an evaluation artefact."),
    "E": dict(name="Controls",
              why="Not candidates. They exist to catch the two confounds that would "
                  "make everything else meaningless."),
}
