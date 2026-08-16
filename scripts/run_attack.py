#!/usr/bin/env python
"""Stage 5: d_OUT - d_IN for every pair in the design, and the two-arm comparison.

    python scripts/run_attack.py --generator copy_paste

Reads results/runs/<generator>/<job>/samples.npy for every model the design names and
writes results/attack/<generator>/{gaps.parquet,summary.json}.

Run this on copy_paste FIRST. It memorises by construction, so its outlier arm must
separate from its control arm; if it does not, the statistic is wrong and no result
from a real generator would mean anything. The distance variant is chosen there and
frozen -- picking it on the real generator's output would be selecting the test on the
outcome (see the module docstring of cgmoutlier.attack.statistic).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

from cgmoutlier._env import check as _envcheck                     # noqa: E402
_envcheck()
from cgmoutlier.data.cohort import load as load_cohort                # noqa: E402
from cgmoutlier.attack.statistic import gap_for_pair, summarise       # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", default="results/design")
    ap.add_argument("--cohort", default="data/cohort/metabonet875")
    ap.add_argument("--runs", default=None, help="default results/runs/<generator>")
    ap.add_argument("--out", default=None, help="default results/attack/<generator>")
    ap.add_argument("--generator", default="copy_paste")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--chunk", type=int, default=8192)
    ap.add_argument("--no-match-k", action="store_true",
                    help="do NOT cut both released sets to the same size. Only for "
                         "measuring how large that artefact is; see statistic.py")
    ap.add_argument("--config", default="configs/experiment.yaml",
                    help="read the frozen distance variant from here")
    ap.add_argument("--channels", default="all",
                    help="'all', or a comma-separated list of channel names. Runs the "
                         "SAME attack on a channel subset of the same models, which is "
                         "what says whether the membership signal lives in one channel. "
                         "The distance is Euclidean over the flattened window, so "
                         "dropping a channel drops its contribution and nothing else; "
                         "no model is retrained.")
    a = ap.parse_args()

    # Every variant is always computed -- they are nearly free once the distance
    # matrices exist, and a later run should not have to be repeated to see one. The
    # config says which is THE result; it is marked rather than filtered so that
    # reopening the choice on new output is a visible act, not a silent one.
    frozen = {}
    cfgp = Path(a.config)
    if cfgp.exists():
        import yaml
        frozen = (yaml.safe_load(cfgp.read_text()) or {}).get("attack", {}) or {}

    design = json.loads((Path(a.design) / "design.json").read_text())
    runs = Path(a.runs) if a.runs else Path("results/runs") / a.generator
    out = Path(a.out) if a.out else Path("results/attack") / a.generator
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
            sys.exit(f"cohort holds {names}, asked for {bad}")
        cols = [names.index(c) for c in want]
    chan_label = "+".join(names[i] for i in cols)
    print(f"[attack] channels {chan_label} (columns {cols} of {names})")

    # Every pair reads the base model, and the released sets are hundreds of MB, so
    # they are cached by name rather than reloaded 48 times.
    cache = {}

    # Every pair reuses the SAME nonmember (the base model) under the symmetric design,
    # so that one is sliced once and kept; the members are used once each and are not
    # cached, because 40 materialised channel slices would be tens of gigabytes.
    reuse = {pr["nonmember"] for pr in design["pairs"]}

    def samples(name):
        if name in cache:
            return cache[name]
        p = runs / name / "samples.npy"
        if not p.exists():
            sys.exit(f"missing {p}\n  run scripts/run_loo.py --job {name} "
                     f"--generator {a.generator}")
        S = np.load(p, mmap_mode="r")
        if len(cols) != S.shape[-1]:
            S = np.ascontiguousarray(np.asarray(S)[..., cols])
        if name in reuse:
            cache[name] = S
        return S

    missing = [pr for pr in design["pairs"]
               if not (runs / pr["member"] / "samples.npy").exists()
               or not (runs / pr["nonmember"] / "samples.npy").exists()]
    if missing:
        print(f"[attack] {len(missing)} of {len(design['pairs'])} pairs are missing a "
              f"model; computing the rest", file=sys.stderr)

    rows = []
    for i, pr in enumerate(design["pairs"], 1):
        t = str(pr["target"])
        if not (runs / pr["member"] / "samples.npy").exists() \
           or not (runs / pr["nonmember"] / "samples.npy").exists():
            continue
        R = np.ascontiguousarray(np.asarray(X)[sids == t][..., cols])
        r = gap_for_pair(R, samples(pr["member"]), samples(pr["nonmember"]),
                         target=t, seed=a.seed, match_k=not a.no_match_k,
                         chunk=a.chunk)
        for row in r:
            row.update(group=pr["group"], member=pr["member"], channels=chan_label,
                       nonmember=pr["nonmember"])
        rows.extend(r)
        print(f"[attack] {i}/{len(design['pairs'])} {pr['group']:<7} {t:<6} "
              f"n={len(R)} windows", flush=True)

    if not rows:
        sys.exit("no pair had both of its models; nothing computed")

    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_parquet(out / "gaps.parquet")

    S = summarise(rows)
    if frozen.get("set_reduce") and frozen.get("subject_reduce"):
        S.insert(2, "frozen", (S.set_reduce == frozen["set_reduce"])
                 & (S.subject_reduce == frozen["subject_reduce"]))
    S.to_json(out / "summary.json", orient="records", indent=2)

    n_by_group = df.groupby("group")["target"].nunique().to_dict()
    print(f"\n{df.target.nunique()} targets: "
          f"{n_by_group.get('outlier', 0)} outliers / "
          f"{n_by_group.get('control', 0)} controls\n")
    cols = ["set_reduce", "subject_reduce", "frozen", "median_gap_outlier",
            "median_gap_control", "p_within_outlier", "p_within_control",
            "auc", "p_between"]
    print(S[[c for c in cols if c in S.columns]].to_string(index=False))
    print(f"\nwrote {out/'gaps.parquet'} and {out/'summary.json'}")
    print(
        "p_within_*  is the gap positive at all, per arm. BOTH arms are membership\n"
        "            pairs -- (include_t, base) whichever arm t is in -- so a generator\n"
        "            that memorises indiscriminately makes both small and separates\n"
        "            neither. That is what copy_paste is expected to do.\n"
        "auc         does the OUTLIER arm exceed the CONTROL arm; 0.5 is no\n"
        "            separation. This is the study's question, not the sanity check.\n"
        "Both p-values treat the targets as independent when they share one base model.\n"
        "Within ONE replicate read them as a screen, not a test; the test is the spread\n"
        "of the arm difference ACROSS replicates, which have independent draws, and\n"
        "independent backgrounds and bases as a result.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
