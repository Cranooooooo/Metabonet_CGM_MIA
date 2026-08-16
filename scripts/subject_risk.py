#!/usr/bin/env python
"""Stage 6: is the arm difference a property of the outlier GROUP, or of a few people?

    python scripts/subject_risk.py --runs dimts_h128 --replicates 3

Reads results/attack/<runs>_rep<r>/gaps.parquet for each replicate and writes
results/attack/<runs>_subject_risk/{subjects.csv,summary.json}. No models are trained
and no GPU is touched -- everything here is already on disk.

WHY THIS IS A SEPARATE QUESTION FROM THE HEADLINE AUC
-----------------------------------------------------
AUC 0.68 says the outlier arm sits above the control arm. It does NOT say whether all
20 outliers are somewhat exposed or three of them are very exposed and the rest are
indistinguishable from controls. Those are different claims about who is at risk and
they call for different mitigations, so the decomposition is reported rather than
assumed.

THE TWO ARMS ARE NOT SHAPED THE SAME, AND IT MATTERS HERE
---------------------------------------------------------
The replicate draws are disjoint by construction, so:

    outliers   the SAME 20 subjects in all 3 replicates  -> 3 measurements each
    controls   60 DISTINCT subjects, one replicate each  -> 1 measurement each

That asymmetry is what makes this analysis possible on one side and impossible on the
other. Per-subject stability can be asked of an outlier and cannot be asked of a
control. In exchange the control arm gives a 60-point null built from 60 independent
people rather than 20 people measured three times.

STANDARDISATION, AND WHY POOLING RAW GAPS WOULD BE WRONG
--------------------------------------------------------
Each replicate has its own base model, and docs/PLAN.md shows that single run's offset
is large enough to swing a within-arm reading by orders of magnitude. Raw gaps from
different replicates are therefore not on one scale. Every gap is standardised against
its OWN replicate's control arm,

    z = (gap - median(control gaps in that replicate)) / MAD(control gaps in that
                                                            replicate)

which is the same cancellation the symmetric design uses, applied per subject: the
replicate's offset and spread leave, and what remains is "how far above this
replicate's typical control does this subject sit". MAD rather than SD because n=20
per replicate and one extreme control would otherwise set the scale.

FOUR THINGS ARE REPORTED
------------------------
1. Per-outlier exposure: mean z over its 3 replicates, and its percentile against the
   60-control null. Plus `n_above`, how many of its 3 replicates put it above that
   replicate's control median -- 0..3, deliberately the same shape as the 0..4 seed
   count in results/seed_stability, because it answers the same kind of question.
2. Stability: Spearman between replicate pairs over the 20 outliers. If a subject's
   exposure does not rank consistently across independent runs, per-subject risk is
   noise and must not be reported, whatever the group-level AUC says.
3. Concentration: the arm AUC recomputed with the top-k most exposed outliers removed,
   k = 1..5. If AUC collapses to 0.5 after dropping two people, the finding is about
   those people.
4. Confound checks: Spearman(gap, n_windows) inside each arm -- the diagnostic that
   disqualified `subject_reduce=min`, re-run on the frozen variant -- and where the
   drawn borderline controls (flagged by some detection seeds, kept on purpose) sit in
   the null.
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


def load(runs: str, reps, config="configs/experiment.yaml"):
    """-> one frame of the frozen variant's rows, tagged by replicate."""
    frozen = {}
    p = Path(config)
    if p.exists():
        import yaml
        frozen = (yaml.safe_load(p.read_text()) or {}).get("attack", {}) or {}
    sr = frozen.get("set_reduce", "min")
    su = frozen.get("subject_reduce", "mean")

    frames = []
    for r in reps:
        f = Path(f"results/attack/{runs}_rep{r}/gaps.parquet")
        if not f.exists():
            raise SystemExit(f"missing {f}; run 15_attack_dimts.pbs for replicate {r}")
        d = pd.read_parquet(f)
        d = d[(d.set_reduce == sr) & (d.subject_reduce == su)].copy()
        d["replicate"] = r
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    print(f"[risk] frozen variant {sr} x {su}: {len(df)} rows over {len(reps)} "
          f"replicates")
    return df, f"{sr} x {su}"


def standardise(df):
    """z against the subject's OWN replicate's control arm. See the module docstring."""
    out = []
    for r, d in df.groupby("replicate"):
        ctrl = d.loc[d.group == "control", "gap"]
        med = float(ctrl.median())
        mad = float(stats.median_abs_deviation(ctrl, scale="normal"))
        if not mad > 0:
            raise SystemExit(f"replicate {r}: control MAD is {mad}, cannot standardise")
        d = d.copy()
        d["z"] = (d.gap - med) / mad
        d["ctrl_median"] = med
        d["ctrl_mad"] = mad
        out.append(d)
        print(f"[risk] replicate {r}: control median {med:+.3e}, MAD {mad:.3e}")
    return pd.concat(out, ignore_index=True)


def auc(pos, neg):
    """P(a random positive exceeds a random negative), ties at 0.5."""
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    u = stats.mannwhitneyu(pos, neg, alternative="two-sided").statistic
    return float(u / (len(pos) * len(neg)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="dimts_h128")
    ap.add_argument("--replicates", type=int, default=3)
    ap.add_argument("--design", default="results/design_sym",
                    help="read borderline_controls from each replicate's design.json")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    reps = list(range(1, a.replicates + 1))
    out = Path(a.out or f"results/attack/{a.runs}_subject_risk")
    out.mkdir(parents=True, exist_ok=True)

    df, variant = load(a.runs, reps)
    df = standardise(df)

    out_arm = df[df.group == "outlier"]
    ctl_arm = df[df.group == "control"]

    n_out = out_arm.target.nunique()
    n_ctl = ctl_arm.target.nunique()
    print(f"[risk] {n_out} outliers x {len(reps)} replicates, "
          f"{n_ctl} distinct controls x 1")
    if n_ctl != len(ctl_arm):
        raise SystemExit("a control appears in more than one replicate; the draws were "
                         "supposed to be disjoint -- check build_design.py --replicates")

    null = ctl_arm.z.to_numpy()                       # 60 independent people

    # ---- 1. per-outlier exposure ------------------------------------------------
    rows = []
    for t, d in out_arm.groupby("target"):
        zs = d.sort_values("replicate").z.to_numpy()
        rows.append(dict(
            target=t, n_windows=int(d.n_windows.iloc[0]),
            **{f"z_rep{r}": float(v) for r, v in zip(sorted(d.replicate), zs)},
            z_mean=float(zs.mean()), z_sd=float(zs.std(ddof=1)),
            pct_vs_null=float((null < zs.mean()).mean() * 100),
            n_above=int((zs > 0).sum()),              # 0..3, like the 0..4 seed count
        ))
    subj = pd.DataFrame(rows).sort_values("z_mean", ascending=False)
    subj.to_csv(out / "subjects.csv", index=False)

    p95 = float(np.percentile(null, 95))
    n_over_p95 = int((subj.z_mean > p95).sum())
    n_all3 = int((subj.n_above == 3).sum())

    print(f"\n{'target':>8}{'z_mean':>9}{'z_sd':>8}{'pct_vs_null':>13}{'n_above':>9}"
          f"{'days':>7}")
    for _, r in subj.iterrows():
        print(f"{r.target:>8}{r.z_mean:>9.2f}{r.z_sd:>8.2f}{r.pct_vs_null:>12.0f}%"
              f"{r.n_above:>9.0f}{r.n_windows:>7.0f}")
    print(f"\nnull (60 controls): median {np.median(null):+.2f}, p95 {p95:+.2f}")
    print(f"{n_over_p95}/{n_out} outliers exceed the null's 95th percentile")
    print(f"{n_all3}/{n_out} are above their replicate's control median in all 3")

    # ---- 1b. how strong is the attack WITHIN each arm? ---------------------------
    # The headline AUC is a BETWEEN-arm number: P(a random outlier's gap exceeds a
    # random control's). There is no "the outliers' AUC" and "the normals' AUC" to
    # compare -- 0.68 IS that comparison. What can be asked per arm is how well a
    # membership attack does on that group, and the answer depends entirely on what
    # the attacker is assumed to hold:
    #
    #   paired      they have BOTH models and ask which one contains t. The attack is
    #               right when d_OUT > d_IN, i.e. gap > 0. Chance is 0.5. This is the
    #               study's counterfactual and it is an UPPER BOUND -- no real attacker
    #               is handed a model trained without their target.
    #   unpaired    they hold one released set and score membership by absolute
    #               distance. AUC over (d_OUT as non-member, d_IN as member) pooled
    #               within the arm. d_IN and d_OUT for one subject are nearly equal
    #               while different subjects sit at very different absolute distances,
    #               so between-subject variation swamps the membership signal here.
    #               That is precisely why the design pairs, and reporting the number
    #               makes the gap between the two threat models explicit.
    arms = {}
    for name, arm in (("outlier", out_arm), ("control", ctl_arm)):
        paired, unpaired = [], []
        for r in reps:
            d = arm[arm.replicate == r]
            if not len(d):
                continue
            paired.append(float((d.gap > 0).mean()))
            unpaired.append(auc(d.d_out, d.d_in))
        n_pos = int((arm.gap > 0).sum())
        sign_p = float(stats.binomtest(n_pos, len(arm), 0.5,
                                       alternative="greater").pvalue)
        arms[name] = dict(
            paired_accuracy=float(np.mean(paired)), paired_per_replicate=paired,
            unpaired_auc=float(np.mean(unpaired)), unpaired_per_replicate=unpaired,
            n_correct=n_pos, n=int(len(arm)), sign_test_p=sign_p)
    print("\nattack strength WITHIN each arm (the between-arm AUC is the headline "
          "0.68):")
    print(f"{'arm':<10}{'paired acc':>12}{'sign p':>10}{'unpaired AUC':>14}   "
          f"per-replicate paired")
    for k, v in arms.items():
        print(f"{k:<10}{v['paired_accuracy']:>12.3f}{v['sign_test_p']:>10.4f}"
              f"{v['unpaired_auc']:>14.3f}   "
              f"{[round(x, 3) for x in v['paired_per_replicate']]}")
    print("  paired   = attacker holds both models; 0.5 is chance. An UPPER BOUND.")
    print("  unpaired = attacker holds one released set and uses absolute distance.")

    # ---- 2. is per-subject exposure stable across independent runs? --------------
    wide = out_arm.pivot_table(index="target", columns="replicate", values="z")
    pairs = {}
    for i, j in [(1, 2), (1, 3), (2, 3)]:
        if i in wide.columns and j in wide.columns:
            rho, p = stats.spearmanr(wide[i], wide[j])
            pairs[f"rep{i}_vs_rep{j}"] = dict(rho=float(rho), p=float(p))
    print("\nper-subject rank stability across replicates (n=20 each):")
    for k, v in pairs.items():
        print(f"  {k}: rho {v['rho']:+.3f}  p {v['p']:.3f}")
    print("  a rho near 0 means per-subject risk is not reproducible and only the\n"
          "  group-level AUC may be reported")

    # ---- 3. is the effect carried by a few people? ------------------------------
    order = subj.target.tolist()                      # most exposed first
    drops = []
    for k in range(0, 6):
        excl = set(order[:k])
        per_rep = []
        for r in reps:
            d = df[df.replicate == r]
            pos = d[(d.group == "outlier") & (~d.target.isin(excl))].z
            neg = d[d.group == "control"].z
            per_rep.append(auc(pos, neg))
        drops.append(dict(dropped=k, removed=order[:k],
                          auc_per_replicate=[round(x, 4) for x in per_rep],
                          auc_mean=float(np.mean(per_rep))))
    print("\nAUC with the k most exposed outliers removed:")
    for d in drops:
        print(f"  k={d['dropped']}  AUC {d['auc_mean']:.3f}   "
              f"per replicate {d['auc_per_replicate']}")

    # ---- 4. confounds -----------------------------------------------------------
    # ONE ROW PER SUBJECT, not per (subject, replicate). The outlier arm's 60 rows are
    # 20 people measured three times and n_windows is a property of the person, so
    # correlating over all 60 counts each subject three times and reports a p-value for
    # an n it does not have. The control arm's 60 rows ARE 60 distinct people, so there
    # the two agree -- which is exactly why the mistake is easy to miss.
    conf = {}
    for name, arm in (("outlier", out_arm), ("control", ctl_arm)):
        per_subject = arm.groupby("target").agg(z=("z", "mean"),
                                                n_windows=("n_windows", "first"))
        rho, p = stats.spearmanr(per_subject.z, per_subject.n_windows)
        conf[name] = dict(rho=float(rho), p=float(p), n_subjects=int(len(per_subject)),
                          n_rows=int(len(arm)))
    print("\nSpearman(z, n_windows), one row per SUBJECT -- the check that disqualified"
          "\nsubject_reduce=min:")
    for k, v in conf.items():
        print(f"  {k:<8} rho {v['rho']:+.3f}  p {v['p']:.3f}  "
              f"({v['n_subjects']} subjects from {v['n_rows']} rows)")

    borderline = {}
    for r in reps:
        dj = Path(a.design) / f"rep{r}" / "design.json"
        if dj.exists():
            borderline.update(json.loads(dj.read_text()).get("borderline_controls", {}))
    bl = ctl_arm[ctl_arm.target.isin(borderline)]
    if len(bl):
        print("\ndrawn borderline controls (flagged by some detection seeds, kept on "
              "purpose):")
        for _, r in bl.iterrows():
            print(f"  {r.target} (rep{r.replicate}, flagged by "
                  f"{borderline[r.target]} seeds): z {r.z:+.2f}, "
                  f"{(null < r.z).mean()*100:.0f}th pct of the null")
    else:
        print("\nno borderline subject was drawn as a control")

    summary = dict(
        variant=variant, runs=a.runs, replicates=reps,
        n_outliers=n_out, n_controls=n_ctl,
        null=dict(median=float(np.median(null)), p95=p95, n=len(null)),
        n_outliers_over_null_p95=n_over_p95, n_outliers_above_in_all_replicates=n_all3,
        arm_attack_strength=arms,
        rank_stability=pairs, auc_after_dropping_top_k=drops,
        spearman_z_vs_n_windows=conf,
        borderline_controls={k: dict(seeds=v,
                                     z=float(bl.loc[bl.target == k, "z"].iloc[0]))
                             for k, v in borderline.items() if (bl.target == k).any()},
    )
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {out}/subjects.csv and {out}/summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
