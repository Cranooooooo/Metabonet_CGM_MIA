#!/usr/bin/env python
"""One attack classifier shared by every subject; one AUC per subject.

    python scripts/attack_panel_pooled.py --panel results/attack_panel/dimts_h128

WHY THIS REPLACES THE PER-SUBJECT PROTOCOL
------------------------------------------
`attack_panel_table.py` trains a classifier on one subject's rows alone. Within a
subject there is exactly ONE release containing them, so "which release is this" is a
perfect predictor of the label and the classifier takes it. Measured on the real data:
every cell's negative control landed within 0.007 of its main number -- 0.763 against
0.769 for the strongest -- i.e. membership contributed nothing and the whole table was
release identity. That is a property of the protocol, not a bug.

Pooling across subjects removes both shortcuts at once:

    row = (subject u, release include_v, window x),  label = (u == v)

    * every release holds exactly one member and 39 non-members -> release identity
      carries no information about the label
    * every subject is a member in exactly one release -> subject identity carries
      none either

`base` is dropped: all 40 targets are non-members of it, so "is this base" would be a
third shortcut of the same kind.

What is left that can predict the label is the RELATIONSHIP between a subject and a
release, which is membership. The classifier now has to find something that transfers
from 79 people to the 80th -- a harder task, and the one an actual adversary faces,
since nobody has labelled data for the person they want to test.

THREAT MODEL IS UNCHANGED. The attacker still holds one released set at inference: a
row is one window against one release. Training the attacker offline on other people is
what makes it deployable, not what makes it stronger.

LEAVE ONE SUBJECT OUT
---------------------
A subject held out is removed from training in EVERY replicate, not just the one being
scored. The 20 outliers appear in all three; leaving only the scored replicate out
would let the classifier memorise them from the other two.

THE NEGATIVE CONTROL
--------------------
Same trained classifier, but a release the held-out subject is also NOT in stands in as
the member release. Its true AUC is 0.5. It is the only evidence that the main number
is membership rather than an artefact -- the per-subject protocol looked strong and was
not, and nothing but this column showed it.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from cgmoutlier._env import check as _envcheck                      # noqa: E402
_envcheck()
from cgmoutlier.attack.panel import FEATURE_SETS                    # noqa: E402


def classifiers():
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
    from sklearn.svm import SVC
    from sklearn.neural_network import MLPClassifier
    # class_weight where the estimator supports it: one member release against four
    # non-member ones is 20% positive, and the majority class would otherwise set the
    # threshold on its own.
    return {
        "C1_logreg": (lambda: LogisticRegression(max_iter=2000,
                                                 class_weight="balanced"), 60000),
        "C2_tree": (lambda: DecisionTreeClassifier(max_depth=4, random_state=0,
                                                   class_weight="balanced"), 60000),
        "C3_forest": (lambda: RandomForestClassifier(n_estimators=100, random_state=0,
                                                     class_weight="balanced",
                                                     n_jobs=-1), 40000),
        "C4_hgb": (lambda: HistGradientBoostingClassifier(max_iter=100,
                                                          random_state=0), 60000),
        # SVC is O(n^2); it gets a smaller cap rather than a different protocol.
        "C5_svm": (lambda: SVC(kernel="rbf", C=1.0, class_weight="balanced"), 8000),
        "C6_mlp": (lambda: MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=200,
                                         early_stopping=True, random_state=0), 40000),
    }


def auc(pos, neg):
    """P(pos > neg), ties at half."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if not len(pos) or not len(neg):
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = np.argsort(allv, kind="mergesort")
    raw = np.empty(len(order), float)
    raw[order] = np.arange(1, len(order) + 1)
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    ranks = (np.bincount(inv, weights=raw) / cnt)[inv]
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2)
                 / (len(pos) * len(neg)))


def build(panel: Path):
    """-> records, one per (replicate, subject).

    Each carries the member block and the non-member blocks, `base` excluded.
    """
    recs = []
    for f in sorted(panel.glob("rep*.npz")):
        z = np.load(f, allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        for t in meta["targets"]:
            negs = [s for s in meta["negatives"][t] if s != "base"]
            if not negs:
                continue
            recs.append(dict(
                target=t, rep=meta["rep"], group=meta["groups"][t],
                n_windows=meta["n_windows"][t],
                pos=z[f"{t}|{meta['positive'][t]}"],
                negs=[z[f"{t}|{s}"] for s in negs]))
    if not recs:
        sys.exit(f"no rep*.npz under {panel}")
    return recs


def _scores(clf, Z):
    return (clf.predict_proba(Z)[:, 1] if hasattr(clf, "predict_proba")
            else clf.decision_function(Z))


def run_cell(recs, make_clf, cap, cols, *, seed=0):
    """LOSO over people. -> (main rows, control rows)."""
    from sklearn.preprocessing import StandardScaler

    people = sorted({r["target"] for r in recs})
    rng = np.random.default_rng(seed)
    main, ctrl = [], []
    for person in people:
        tr = [r for r in recs if r["target"] != person]
        X = np.vstack([b[:, cols] for r in tr for b in [r["pos"]] + r["negs"]])
        y = np.concatenate([np.r_[np.ones(len(r["pos"])),
                                  np.zeros(sum(len(b) for b in r["negs"]))]
                            for r in tr])
        if len(X) > cap:                       # stratified thin-out, not a different
            keep = np.concatenate([            # protocol for the expensive models
                rng.choice(np.flatnonzero(y == c),
                           size=max(1, int(cap * (y == c).mean())), replace=False)
                for c in (0, 1)])
            X, y = X[keep], y[keep]
        sc = StandardScaler().fit(X)
        clf = make_clf().fit(sc.transform(X), y)

        for r in [r for r in recs if r["target"] == person]:
            row = dict(target=person, rep=r["rep"], group=r["group"],
                       n_windows=r["n_windows"])
            s_pos = _scores(clf, sc.transform(r["pos"][:, cols]))
            s_neg = [_scores(clf, sc.transform(b[:, cols])) for b in r["negs"]]
            main.append(dict(row, auc=auc(s_pos, np.concatenate(s_neg))))
            # control: a release the subject is ALSO not in plays the member
            if len(s_neg) > 1:
                ctrl.append(dict(row, auc=auc(s_neg[0], np.concatenate(s_neg[1:]))))
    return main, ctrl


def summarise(rows, label):
    """Group means and three views of "who is most exposed".

    THE OBSERVATION UNIT IS (subject, replicate), NOT the subject. The 20 outliers are
    measured in all three replicates and the 60 controls in one each, so a top-three
    over observations can be one outlier three times while a control can never occupy
    more than one slot. That asymmetry makes the two groups' top-THREE incomparable as
    sets, though their maxima remain comparable. Hence three views:

      top3_obs     highest (subject, replicate) readings. Shows reproducibility --
                   the same person leading in all three replicates is a statement
                   that averaging hides.
      top3_by_max  distinct people, ranked by their best replicate. Same noise on
                   both sides, so comparable between groups.
      top3_by_mean distinct people, ranked by the mean over their replicates. The
                   better estimate of a person, but outlier means are averages of
                   three and control means of one, so the groups are NOT comparable
                   here -- an outlier's maximum is smoothed and a control's is not.
    """
    import pandas as pd
    d = pd.DataFrame(rows).dropna(subset=["auc"])
    o, c = d[d.group == "outlier"], d[d.group == "control"]

    def top_obs(x):
        return [(round(a, 3), t, int(w), int(r)) for a, t, w, r in
                sorted(zip(x.auc, x.target, x.n_windows, x.rep), reverse=True)[:3]]

    def top_people(x, how):
        g = x.groupby("target").agg(auc=("auc", how), w=("n_windows", "first"),
                                    n=("auc", "size"))
        g = g.sort_values("auc", ascending=False).head(3)
        return [(round(r.auc, 3), t, int(r.w), int(r.n)) for t, r in g.iterrows()]

    return dict(attacker=label,
                outlier_mean=float(o.auc.mean()), control_mean=float(c.auc.mean()),
                diff=float(o.auc.mean() - c.auc.mean()),
                outlier_n=int(len(o)), control_n=int(len(c)),
                outlier_people=int(o.target.nunique()),
                control_people=int(c.target.nunique()),
                outlier_top3=top_obs(o), control_top3=top_obs(c),
                outlier_top3_by_max=top_people(o, "max"),
                control_top3_by_max=top_people(c, "max"),
                outlier_top3_by_mean=top_people(o, "mean"),
                control_top3_by_mean=top_people(c, "mean"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="results/attack_panel/dimts_h128")
    ap.add_argument("--out", default="results/attack_panel/table_pooled.json")
    ap.add_argument("--cells", default=None)
    a = ap.parse_args()

    recs = build(Path(a.panel))
    names = json.loads(str(np.load(sorted(Path(a.panel).glob("rep*.npz"))[0],
                                   allow_pickle=False)["meta"]))["feature_names"]
    clfs = classifiers()
    plan = a.cells.split(",") if a.cells else [
        "C1_logreg!F1_min", "C1_logreg!F3_raw10", "C2_tree!F3_raw10",
        "C3_forest!F3_raw10", "C4_hgb!F3_raw10", "C5_svm!F3_raw10",
        "C6_mlp!F3_raw10", "C4_hgb!F2_cheap4",
    ]
    print(f"[pooled] {len(recs)} (replicate, subject) records, "
          f"{len({r['target'] for r in recs})} distinct people", flush=True)

    out, allrows = [], {}
    for cell in plan:
        cname, fname = cell.split("!")
        make_clf, cap = clfs[cname]
        cols = [names.index(n) for n in FEATURE_SETS[fname]]
        t0 = time.time()
        m, c = run_cell(recs, make_clf, cap, cols)
        label = f"{cname} x {fname}"
        # Two different things share the word "control" and must not share a field
        # name: `control_*` is the CONTROL GROUP, the day-matched normal subjects, and
        # `negctrl_*` is the NEGATIVE CONTROL, the same cell run with a release the
        # subject is also not a member of. One is an arm of the study, the other is
        # the validity check on the measurement.
        s, neg = summarise(m, label), summarise(c, label + " [negative control]")
        s.update(negctrl_outlier_mean=neg["outlier_mean"],
                 negctrl_control_mean=neg["control_mean"],
                 negctrl_diff=neg["diff"],
                 # the null's extreme: what an individual reading has to beat
                 negctrl_outlier_top3_by_max=neg["outlier_top3_by_max"],
                 negctrl_control_top3_by_max=neg["control_top3_by_max"],
                 seconds=round(time.time() - t0, 1))
        out.append(s)
        # Every per-subject reading is kept, so any later view -- a different ranking,
        # a significance test, a subset by window count -- is a read of this file
        # rather than another run of the panel.
        allrows[label] = dict(main=m, negative_control=c)
        print(f"{label:22} 异常 {s['outlier_mean']:.4f}  "
              f"正常 {s['control_mean']:.4f}  差 {s['diff']:+.4f}   "
              f"[负对照 异常 {neg['outlier_mean']:.4f} "
              f"正常 {neg['control_mean']:.4f}]  {s['seconds']:.0f}s", flush=True)
        print(f"    按观测    异常 {s['outlier_top3']}", flush=True)
        print(f"              正常 {s['control_top3']}", flush=True)
        print(f"    按人-最大 异常 {s['outlier_top3_by_max']}", flush=True)
        print(f"              正常 {s['control_top3_by_max']}", flush=True)
        print(f"    按人-均值 异常 {s['outlier_top3_by_mean']}", flush=True)
        print(f"              正常 {s['control_top3_by_mean']}", flush=True)
        print(f"    负对照最大 异常 {neg['outlier_top3_by_max']}  "
              f"正常 {neg['control_top3_by_max']}", flush=True)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))
    rows_path = Path(a.out).with_name(Path(a.out).stem + "_rows.json")
    rows_path.write_text(json.dumps(allrows, indent=1))
    print(f"\nwrote {a.out}\nwrote {rows_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
