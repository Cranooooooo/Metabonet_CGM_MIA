#!/usr/bin/env python
"""Stage 6b: one membership-inference AUC per SUBJECT, not one per arm.

    python scripts/subject_auc.py --runs dimts_h128 --replicates 3

Writes results/attack/<runs>_subject_auc/{per_subject.csv,summary.json}. CPU only --
this is the same BLAS the attack already runs, roughly 20 min per replicate on qdev.

WHY THIS HAS TO BE RECOMPUTED RATHER THAN READ OFF gaps.parquet
---------------------------------------------------------------
The attack is window-level underneath: for target t it builds

    d_in[i]  = distance from t's window i to the nearest sample released by include_t
    d_out[i] = the same against base

and then `reduce_subject` collapses those n_t-vectors to one number per subject. Only
the collapsed number reaches the parquet. A per-subject AUC needs the vectors, so they
are recomputed here and the ranks are taken instead of the mean:

    AUC_t = P(d_out > d_in) over t's own windows,  Mann-Whitney, 0.5 = no signal

That is the standard per-record membership score, and it is what "how attackable is
this person" means as a number.

THIS IS A NEW STATISTIC, AND IT DOES NOT RE-OPEN THE FROZEN ONE
---------------------------------------------------------------
`min x mean` was frozen on copy_paste and governs the headline arm comparison. AUC_t is
a different reduction over windows answering a different question -- per-subject
attackability rather than arm separation. `set_reduce=min` is kept because that part of
the frozen choice is about which released sample to measure against, and changing it
here would be variant-shopping. Any arm comparison computed FROM AUC_t is secondary and
exploratory by construction: it is a second test on data whose primary test is already
reported.

THREE THINGS THAT WILL MISLEAD IF READ WITHOUT CARE
---------------------------------------------------
* **n_windows spans 31..363.** AUC_t from 31 windows is far noisier than from 363, so a
  bare ranking of 80 numbers puts short-record subjects at both ends for reasons that
  have nothing to do with exposure. Every AUC therefore carries a bootstrap CI, and the
  correlation between AUC and n_windows is reported rather than hoped away.
* **AUC and detectability are different questions.** AUC_t = 0.55 over 300 windows is a
  reliably detectable leak; AUC_t = 0.70 over 31 windows may not be. Effect size
  (AUC_t) is what compares across people; the p-value is what says whether an attacker
  would notice. Both are written out.
* **The arms are not shaped the same.** Outliers appear in all 3 replicates and get 3
  AUCs each; controls are disjoint draws and get 1. The 80-subject table uses the
  outliers' mean over replicates, so an outlier's number is a 3-run average and a
  control's is a single run -- the outlier column is the quieter of the two.

And as everywhere in this study: this is the PAIRED counterfactual, where the attacker
is handed a model trained deliberately without the target. It is an upper bound on
attacker capability, not a threat model.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from cgmoutlier._env import check as _envcheck                      # noqa: E402
_envcheck()
from cgmoutlier.data.cohort import load as load_cohort              # noqa: E402
from cgmoutlier.attack.statistic import window_distances, _match    # noqa: E402


def auc_ci(d_out, d_in, n_boot=2000, seed=2026):
    """AUC over one subject's windows, with a paired bootstrap CI.

    Resampling WINDOWS, not the two vectors independently: d_out[i] and d_in[i] are the
    same window measured against two models and are strongly correlated, so an
    unpaired bootstrap would inflate the interval.
    """
    n = len(d_in)
    u = stats.mannwhitneyu(d_out, d_in, alternative="two-sided").statistic
    point = float(u / (n * n))
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        o, i_ = d_out[idx], d_in[idx]
        boot[b] = stats.mannwhitneyu(o, i_, alternative="two-sided").statistic / (n * n)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return point, float(lo), float(hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="dimts_h128")
    ap.add_argument("--design", default="results/design_sym")
    ap.add_argument("--cohort", default="data/cohort/metabonet875")
    ap.add_argument("--replicates", type=int, default=3)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--chunk", type=int, default=8192)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", default=None)
    ap.add_argument("--channels", default="all",
                    help="'all', or a comma-separated list of channel names. Restricts "
                         "the distance to a channel subset of the SAME models; nothing "
                         "is retrained.")
    ap.add_argument("--skip-missing", action="store_true",
                    help="pass over pairs whose model has not finished, instead of "
                         "stopping. For reading a campaign that is still running -- the "
                         "arms are then unbalanced and no longer day-matched, so the "
                         "arm comparison is provisional even though each subject's own "
                         "AUC is final.")
    a = ap.parse_args()
    reps = list(range(1, a.replicates + 1))
    out = Path(a.out or f"results/attack/{a.runs}_subject_auc")
    out.mkdir(parents=True, exist_ok=True)

    X, sids, man = load_cohort(a.cohort)
    sids = np.asarray([str(s) for s in sids])

    names = man.get("channels") or [man.get("channel", "CGM")]
    if a.channels == "all":
        cols = list(range(len(names)))
    else:
        want = a.channels.split(",")
        bad = [c for c in want if c not in names]
        if bad:
            raise SystemExit(f"cohort holds {names}, asked for {bad}")
        cols = [names.index(c) for c in want]
    chan_label = "+".join(names[i] for i in cols)
    print(f"[auc] channels {chan_label} (columns {cols} of {names})")

    rows = []
    for r in reps:
        design = json.loads((Path(a.design) / f"rep{r}" / "design.json").read_text())
        runs = Path("results/runs") / f"{a.runs}_rep{r}"
        cache = {}

        # the base is the nonmember of every pair, so it is sliced once and kept; the
        # members are read once each and are not cached
        reuse = {pr["nonmember"] for pr in design["pairs"]}

        def samples(name):
            if name in cache:
                return cache[name]
            p = runs / name / "samples.npy"
            if not p.exists():
                raise SystemExit(f"missing {p}")
            S = np.load(p, mmap_mode="r")
            if len(cols) != S.shape[-1]:
                S = np.ascontiguousarray(np.asarray(S)[..., cols])
            if name in reuse:
                cache[name] = S
            return S

        for n, pr in enumerate(design["pairs"], 1):
            t = str(pr["target"])
            if a.skip_missing and not all(
                    (runs / pr[k] / "samples.npy").exists()
                    for k in ("member", "nonmember")):
                continue
            R = np.ascontiguousarray(np.asarray(X)[sids == t][..., cols])
            S_in, S_out, k = _match(samples(pr["member"]), samples(pr["nonmember"]),
                                    t, a.seed, True)
            d_in = window_distances(R, S_in, set_reduce="min", chunk=a.chunk)
            d_out = window_distances(R, S_out, set_reduce="min", chunk=a.chunk)

            point, lo, hi = auc_ci(d_out, d_in, a.n_boot, a.seed)
            # Wilcoxon on the paired per-window differences: does this ONE subject leak
            # detectably at all. Distinct from AUC, which is the effect size.
            try:
                w_p = float(stats.wilcoxon(d_out, d_in, alternative="greater").pvalue)
            except ValueError:                       # all differences zero
                w_p = 1.0
            rows.append(dict(
                replicate=r, target=t, group=pr["group"], n_windows=int(len(R)),
                auc=point, auc_lo=lo, auc_hi=hi,
                paired_frac=float((d_out > d_in).mean()), wilcoxon_p=w_p,
                d_in_mean=float(d_in.mean()), d_out_mean=float(d_out.mean()),
                k_matched=int(k)))
            print(f"[auc] rep{r} {n:>2}/{len(design['pairs'])} {pr['group']:<7} "
                  f"{t:<6} n={len(R):<4} AUC {point:.3f} [{lo:.3f},{hi:.3f}] "
                  f"p={w_p:.2e}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(out / "per_replicate.csv", index=False)

    # ---- the 80-subject table: outliers averaged over their 3 replicates ---------
    per_subject = (df.groupby(["target", "group"])
                     .agg(auc=("auc", "mean"), auc_sd=("auc", "std"),
                          n_measurements=("auc", "size"),
                          n_windows=("n_windows", "first"),
                          paired_frac=("paired_frac", "mean"),
                          min_wilcoxon_p=("wilcoxon_p", "min"))
                     .reset_index()
                     .sort_values("auc", ascending=False))
    per_subject.to_csv(out / "per_subject.csv", index=False)

    o = per_subject[per_subject.group == "outlier"]
    c = per_subject[per_subject.group == "control"]
    print(f"\n{len(per_subject)} subjects: {len(o)} outliers (mean of "
          f"{int(o.n_measurements.iloc[0])} replicates each), {len(c)} controls (1 each)")

    print(f"\n{'group':<9}{'n':>4}{'AUC med':>10}{'AUC mean':>10}{'IQR':>18}"
          f"{'max':>8}{'>0.55':>8}")
    for name, d in (("outlier", o), ("control", c)):
        q1, q3 = np.percentile(d.auc, [25, 75])
        print(f"{name:<9}{len(d):>4}{d.auc.median():>10.3f}{d.auc.mean():>10.3f}"
              f"   [{q1:.3f}, {q3:.3f}]{d.auc.max():>8.3f}"
              f"{int((d.auc > 0.55).sum()):>8}")

    u = stats.mannwhitneyu(o.auc, c.auc, alternative="greater")
    arm_auc = float(u.statistic / (len(o) * len(c)))
    print(f"\nSECONDARY, EXPLORATORY -- outlier vs control on this new statistic:")
    print(f"  arm-level AUC-of-AUCs {arm_auc:.3f}, Mann-Whitney p {u.pvalue:.4f}")
    print(f"  the frozen result stays AUC 0.680 from min x mean; this is a second\n"
          f"  test on the same data and is reported as a decomposition, not a claim")

    rho_all, p_all = stats.spearmanr(per_subject.auc, per_subject.n_windows)
    print(f"\nSpearman(AUC, n_windows) over all {len(per_subject)} subjects: "
          f"rho {rho_all:+.3f}, p {p_all:.3f}")
    print("  a strong positive here would mean the ranking partly measures record\n"
          "  length rather than exposure")

    print(f"\ntop 10 by AUC:")
    print(f"{'target':>8}{'group':>9}{'AUC':>8}{'95% CI':>18}{'days':>7}{'wilcox p':>11}")
    for _, x in per_subject.head(10).iterrows():
        sub = df[df.target == x.target]
        lo, hi = sub.auc_lo.mean(), sub.auc_hi.mean()
        print(f"{x.target:>8}{x.group:>9}{x.auc:>8.3f}   [{lo:.3f}, {hi:.3f}]"
              f"{x.n_windows:>7.0f}{x.min_wilcoxon_p:>11.1e}")

    summary = dict(
        runs=a.runs, replicates=reps, n_subjects=int(len(per_subject)),
        statistic="per-subject Mann-Whitney AUC over windows, set_reduce=min",
        note=("secondary/exploratory; the frozen primary statistic is min x mean and "
              "its result (arm AUC 0.680) is unchanged by anything here"),
        by_group={g: dict(n=int(len(d)), median=float(d.auc.median()),
                          mean=float(d.auc.mean()), max=float(d.auc.max()),
                          q1=float(np.percentile(d.auc, 25)),
                          q3=float(np.percentile(d.auc, 75)),
                          n_over_0_55=int((d.auc > 0.55).sum()))
                  for g, d in (("outlier", o), ("control", c))},
        exploratory_arm_auc_of_aucs=arm_auc,
        exploratory_arm_p=float(u.pvalue),
        spearman_auc_vs_n_windows=dict(rho=float(rho_all), p=float(p_all)),
    )
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {out}/per_subject.csv, per_replicate.csv and summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
