#!/usr/bin/env python
"""Stage B of the supervised attacker panel: train the classifiers, emit the table.

    python scripts/attack_panel_table.py --panel results/attack_panel/dimts_h128

One attacker per (classifier, feature set). For each subject a classifier is trained on
that subject's own windows and tested on windows it never saw, so nothing has to
generalise across people and "how unusual this person is" cancels by construction --
it is identical in the subject's positive and negative rows.

WHAT ONE CELL IS
----------------
For subject t in one replicate:

    positives   phi(x, S_include_t)          x ranges over t's day windows
    negatives   phi(x, S_u)                  S_u drawn from the sets t is NOT in
    split       BY WINDOW -- a window's positive and negative rows travel together,
                so a held-out row is never the other half of a training row
    repeat      5 times, redrawing both the split and the negative sets
    score       AUC between the test windows' member-condition and non-member-
                condition scores; 0.5 is no signal

The attacker never holds two releases at once: each row is features of one window
against one release. The two conditions meet only in the evaluation, which is
unavoidable -- measuring membership leakage requires both worlds to exist somewhere.

THE NEGATIVE CONTROL COLUMN
---------------------------
`--control` reruns each cell with a set the subject is ALSO not a member of standing
in as the positive. The true AUC is then 0.5 by construction, so whatever the column
reads is the panel's residual bias -- from release-identity, from the classifier
overfitting a small subject, from anything. A main number is only interpretable
against it.

WHY THERE IS NO POPULATION-REFERENCE FEATURE SET HERE
-----------------------------------------------------
A feature such as d(x, real cohort) is identical in a window's positive and negative
rows, because it does not involve the released set at all. Constant across the
conditions means zero membership information to a per-subject classifier. Such
features matter only in a cross-subject design, where they let one decision boundary
serve subjects of different typicality.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

from cgmoutlier._env import check as _envcheck                      # noqa: E402
_envcheck()
from cgmoutlier.attack.panel import FEATURE_SETS                    # noqa: E402
from cgmoutlier.attack.panel import normalise as P_normalise        # noqa: E402


def classifiers():
    """The model axis. Each returns a fresh, unfitted estimator.

    Kept to families that train on a few hundred rows: the smallest subject here has
    31 windows, i.e. 43 training rows after the split. A learned encoder with its own
    classification head is in the plan and is reported blank for that reason -- see
    the table's footnote rather than a silently different protocol.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
    from sklearn.svm import SVC
    from sklearn.neural_network import MLPClassifier
    return {
        "C1_logreg": lambda: LogisticRegression(max_iter=2000),
        "C2_tree": lambda: DecisionTreeClassifier(max_depth=3, random_state=0),
        "C3_forest": lambda: RandomForestClassifier(n_estimators=200, random_state=0,
                                                    n_jobs=1),
        "C4_hgb": lambda: HistGradientBoostingClassifier(max_iter=100,
                                                         random_state=0),
        "C5_svm": lambda: SVC(kernel="rbf", C=1.0),
        "C6_mlp": lambda: MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=800,
                                        random_state=0),
    }


def _scores(clf, Z):
    if hasattr(clf, "predict_proba"):
        return clf.predict_proba(Z)[:, 1]
    return clf.decision_function(Z)


def auc(pos, neg):
    """P(pos > neg), ties counted half. Mann-Whitney without the scipy round-trip."""
    pos, neg = np.asarray(pos), np.asarray(neg)
    if not len(pos) or not len(neg):
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = np.argsort(allv, kind="mergesort")
    raw = np.empty(len(order), float)
    raw[order] = np.arange(1, len(order) + 1)
    # Ties get the average rank. Tree classifiers emit a handful of distinct
    # probabilities, so a cell where the model learned nothing produces mostly ties;
    # without this it would read 0 or 1 rather than 0.5.
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    ranks = (np.bincount(inv, weights=raw) / cnt)[inv]
    r1 = ranks[:len(pos)].sum()
    return float((r1 - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def subject_auc(pos_feat, neg_feats, make_clf, *, cols, n_repeat=5, train_frac=0.7,
                seed=0):
    """Mean test AUC over `n_repeat` window splits for one subject."""
    from sklearn.preprocessing import StandardScaler

    n = len(pos_feat)
    if n < 8:
        return float("nan"), 0
    rng = np.random.default_rng(seed)
    out = []
    for rep in range(n_repeat):
        idx = rng.permutation(n)
        ntr = max(4, int(round(train_frac * n)))
        tr, te = idx[:ntr], idx[ntr:]
        if len(te) < 2:
            continue
        # one negative row per window, its source redrawn each repeat, so no single
        # release can carry the label
        pick = rng.integers(0, len(neg_feats), size=n)
        neg = np.stack([neg_feats[pick[i]][i] for i in range(n)])

        Xtr = np.vstack([pos_feat[tr][:, cols], neg[tr][:, cols]])
        ytr = np.r_[np.ones(len(tr)), np.zeros(len(tr))]
        sc = StandardScaler().fit(Xtr)
        clf = make_clf().fit(sc.transform(Xtr), ytr)
        out.append(auc(_scores(clf, sc.transform(pos_feat[te][:, cols])),
                       _scores(clf, sc.transform(neg[te][:, cols]))))
    return (float(np.mean(out)) if out else float("nan")), len(out)


def load_panel(panel: Path):
    reps = {}
    for f in sorted(panel.glob("rep*.npz")):
        z = np.load(f, allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        reps[meta["rep"]] = (z, meta)
    if not reps:
        sys.exit(f"no rep*.npz under {panel}; run attack_panel_features.py first")
    return reps


def block(z, meta, t, s, *, normalised=True):
    """Subject t's features against release s, in that release's own reference units.

    Without this the raw features carry whatever is idiosyncratic about the release,
    and since a subject's positives all come from ONE release that is a shortcut to
    the label. Measured on a pilot: the negative control read 0.667 raw, where it must
    read 0.5. `REF|<set>|<K>` holds the mean and sd of the same fixed background
    windows scored against that release at that K -- background subjects are members
    of every release, so the reference is membership-free.
    """
    F = z[f"{t}|{s}"]
    if not normalised:
        return F
    key = f"REF|{s}|{meta['k'][t]}"
    if key not in z:
        raise KeyError(f"{key} missing; re-run attack_panel_features.py -- this "
                       f"panel predates per-release normalisation")
    mu, sd = z[key]
    return P_normalise(F, mu, sd)


def run_cell(reps, make_clf, cols, *, control=False, n_repeat=5, seed=0,
             normalised=True):
    """-> list of dicts, one per (subject, replicate)."""
    rows = []
    for r, (z, meta) in sorted(reps.items()):
        for t in meta["targets"]:
            negs = list(meta["negatives"][t])
            if control:
                # A set the subject is ALSO not in stands in as the positive. An
                # include_u, not base: the main cell's positive is an include_*, and a
                # control whose positive is the one structurally distinct release
                # would measure a different thing from the cell it is controlling.
                inc = [s for s in negs if s != "base"]
                if not inc:
                    continue
                pos_name = inc[0]
                negs = [s for s in negs if s != pos_name]
            else:
                pos_name = meta["positive"][t]
            if not negs:
                continue
            pos = block(z, meta, t, pos_name, normalised=normalised)
            neg_feats = [block(z, meta, t, s, normalised=normalised) for s in negs]
            a, k = subject_auc(pos, neg_feats, make_clf, cols=cols,
                               n_repeat=n_repeat, seed=seed + r)
            rows.append(dict(target=t, rep=r, group=meta["groups"][t],
                             n_windows=meta["n_windows"][t], auc=a, n_splits=k))
    return rows


def summarise(rows, label):
    """Group means, their difference, and the top three of each group.

    Both groups are summarised from SINGLE-replicate observations. The 20 outliers are
    measured in all three replicates and the 60 controls in one each, so averaging the
    outliers first would give their maxima less noise than the controls' -- and a
    top-three comparison between a smoothed maximum and a raw one is biased toward the
    controls. Means are unaffected either way; maxima are not.
    """
    import pandas as pd
    d = pd.DataFrame(rows).dropna(subset=["auc"])
    g = {k: v for k, v in d.groupby("group")}
    o, c = g.get("outlier"), g.get("control")
    top = lambda x: sorted(zip(x.auc, x.target, x.n_windows, x.rep), reverse=True)[:3]
    return dict(
        attacker=label,
        outlier_mean=float(o.auc.mean()), control_mean=float(c.auc.mean()),
        diff=float(o.auc.mean() - c.auc.mean()),
        outlier_n=int(len(o)), control_n=int(len(c)),
        outlier_people=int(o.target.nunique()), control_people=int(c.target.nunique()),
        outlier_top3=[(round(a, 3), t, int(w), int(r)) for a, t, w, r in top(o)],
        control_top3=[(round(a, 3), t, int(w), int(r)) for a, t, w, r in top(c)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="results/attack_panel/dimts_h128")
    ap.add_argument("--out", default="results/attack_panel/table.json")
    ap.add_argument("--n-repeat", type=int, default=5)
    ap.add_argument("--cells", default=None,
                    help="comma-separated <clf>!<featset>; default the planned panel")
    ap.add_argument("--also-raw", action="store_true",
                    help="run every cell a second time on UNNORMALISED features. The "
                         "pair is the evidence that per-release normalisation was "
                         "needed: the raw negative control reads well above 0.5")
    a = ap.parse_args()

    reps = load_panel(Path(a.panel))
    names = json.loads(str(next(iter(reps.values()))[0]["meta"]))["feature_names"]
    clfs = classifiers()

    plan = a.cells.split(",") if a.cells else [
        "C1_logreg!F1_min",                       # single-feature linear baseline
        "C1_logreg!F3_raw10",                     # does combining features help
        "C2_tree!F3_raw10",
        "C3_forest!F3_raw10",
        "C4_hgb!F3_raw10",
        "C5_svm!F3_raw10",
        "C6_mlp!F3_raw10",
        "C4_hgb!F2_cheap4",                       # 4 features vs 10, same classifier
    ]

    out = []
    for cell in plan:
        cname, fname = cell.split("!")
        cols = [names.index(n) for n in FEATURE_SETS[fname]]
        for norm in ([True, False] if a.also_raw else [True]):
            label = f"{cname} x {fname}" + ("" if norm else " [raw]")
            main_rows = run_cell(reps, clfs[cname], cols, n_repeat=a.n_repeat,
                                 normalised=norm)
            ctrl_rows = run_cell(reps, clfs[cname], cols, n_repeat=a.n_repeat,
                                 control=True, normalised=norm)
            s = summarise(main_rows, label)
            cs = summarise(ctrl_rows, label + " [control]")
            s["normalised"] = norm
            s["control_outlier_mean"] = cs["outlier_mean"]
            s["control_control_mean"] = cs["control_mean"]
            s["control_diff"] = cs["diff"]
            out.append(s)
            print(f"{label:30} 异常 {s['outlier_mean']:.4f}  "
                  f"正常 {s['control_mean']:.4f}  差 {s['diff']:+.4f}   "
                  f"[负对照 异常 {cs['outlier_mean']:.4f} "
                  f"正常 {cs['control_mean']:.4f}]", flush=True)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
