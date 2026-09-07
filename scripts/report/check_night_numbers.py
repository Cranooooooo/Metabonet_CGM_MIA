#!/usr/bin/env python
"""Every number the 2026-08-29/30 round wrote into the docs, checked against its file.

PAPER_PLAN.md Tip 1 says the documents are living and that every number in them is read
from a result file and checked before the document is committed. This is that check for
the round recorded in MODEL_DESIGN.md 1.8a, 2.7, 2.7a, 5a, 7 mode 1 and PITFALLS.md 19.

Two traps it exists to catch, both of which it did catch on first run:

  - `results/*/tsgem` is a FILE with no extension, so a `*.json` glob silently finds
    nothing and every comparison against it reports None rather than failing loudly.
  - the transform arm AUCs are DERIVED, not stored: per_transform.json holds per-target
    gaps and the AUC is a Mann-Whitney over the two arms. Quoting them requires
    recomputing them, which is what this does.

It also caught a real documentation error: the pilot's `predictive` was quoted from
Eval_Assembly (0.0096) inside a table whose every other cell came from tsgen_metrics,
where the same quantity reads 0.0123 against DiM-TS's 0.0121 -- turning "MAVEN's one win"
into "MAVEN wins nothing". Mixing suites within one row is the error to watch for.

    python scripts/report/check_night_numbers.py
"""
import json
import sys

import numpy as np

FAIL = []


def chk(label, got, want, tol=0.0006):
    good = got is not None and abs(got - want) <= tol
    print(f"  {'OK ' if good else 'BAD'}  {label:46s} doc={want}  file={got}")
    if not good:
        FAIL.append(label)


def mw(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(((a[:, None] > b[None, :]).sum()
                  + 0.5 * (a[:, None] == b[None, :]).sum()) / (len(a) * len(b)))


print("=== MODEL_DESIGN 1.8a: MAVEN pilot and ladder against DiM-TS ===")
for tag, qdir, cf, dmax, pred in [("pilot", "results/pilot_maven", 0.2256, 0.6200, 0.0123),
                                  ("ladder", "results/ladder_maven", 0.2721, 0.6742, 0.0132)]:
    v = json.load(open(f"{qdir}/tsgem"))          # a file, not a directory
    v = list(v.values())[0] if isinstance(list(v.values())[0], dict) else v
    chk(f"{tag} context_fid", v["context_fid"], cf, 0.0001)
    chk(f"{tag} predictive (tsgem, NOT Eval_Assembly)", v["predictive"], pred, 0.0001)
    m = list(json.load(open(f"{qdir}/disc_stability.json")).values())[0]["max"]
    chk(f"{tag} discriminative max over 8", m, dmax, 0.001)

b = json.load(open("results/matrix/sweep/quality_tsgem/d1_c1_ms3.json"))[
    "results/runs/sweep_d1_c1_ms3/base"]
chk("DiM-TS d1_c1 @30k context_fid", b["context_fid"], 0.0611)
chk("DiM-TS d1_c1 @30k discriminative", b["discriminative"], 0.0238)
chk("DiM-TS d1_c1 @30k predictive", b["predictive"], 0.0121)

print("\n=== PITFALLS 19: the memoriser wins the gate ===")
chk("copy_paste d7 context_fid",
    json.load(open("results/quality_cp_d7/cp_d7_contiguous_rep1__base.json"))["context_fid"],
    0.070, 0.001)
for ms, want in (("ms2", 0.131), ("ms6", 0.287)):
    v = json.load(open(f"results/matrix/sweep/quality_tsgem/d7_c1_{ms}.json"))[
        f"results/runs/sweep_d7_c1_{ms}/base"]["context_fid"]
    chk(f"DiM-TS d7_c1 {ms} context_fid", v, want, 0.001)

print("\n=== MODEL_DESIGN 5a: what a directed release-time edit does ===")
ROWS = {"d1_c1": [(0.0, 11, 0.747, 0.840), (0.002, 8, 0.720, 0.864),
                  (0.008, 6, 0.661, 0.882), (0.016, 5, 0.712, 0.876)],
        "d1_c2": [(0.0, 9, 0.880, 0.822), (0.016, 4, 0.880, 0.740)],
        "d7_c1": [(0.0, 18, 1.000, 0.959), (0.004, 15, 1.000, 0.982),
                  (0.016, 9, 1.000, 0.947)]}
for cell, rows in ROWS.items():
    S = {r["delta"]: r for r in
         json.load(open(f"results/matrix/edit_ceiling/{cell}.json"))["summary"]}
    for d, n55, mx, arm in rows:
        chk(f"{cell} d={d} risk>0.55", S[d]["risk_n_above_055"], n55, 0)
        chk(f"{cell} d={d} max AUC", S[d]["risk_max_auc"], mx, 0.001)
        chk(f"{cell} d={d} arm AUC", S[d]["arm_auc"], arm, 0.001)

print("\n=== MODEL_DESIGN 7 mode 1: the superseded per-pair bound ===")
for cell, ratio, frac in (("d1_c1", 1.81, 0.322), ("d1_c2", 0.89, 0.539),
                          ("d7_c1", 0.51, 0.751)):
    j = json.load(open(f"results/matrix/ceiling/{cell}.json"))
    chk(f"{cell} ceiling/g", j["d2m1_median"] / j["gap"], ratio, 0.01)
    chk(f"{cell} frac of windows below g", j["frac_ceiling_below_gap"], frac, 0.001)

print("\n=== PAPER_PLAN 3c: arm AUC per transform (DERIVED, not stored) ===")
WANT = {"d1_c1": {"raw": 0.562, "sorted": 0.828, "diff": 0.598, "hourly": 0.538, "zscore": 0.527},
        "d1_c2": {"raw": 0.698, "sorted": 0.781, "diff": 0.533, "hourly": 0.746, "zscore": 0.657},
        "d7_c1": {"raw": 0.627, "sorted": 0.905, "diff": 0.704, "hourly": 0.627, "zscore": 0.467}}
for cell, w in WANT.items():
    j = json.load(open(f"results/matrix/localise/{cell}/per_transform.json"))
    for t, want in w.items():
        o = [r[t] for r in j.values() if r["group"] == "outlier"]
        c = [r[t] for r in j.values() if r["group"] == "control"]
        chk(f"{cell} {t} arm AUC", mw(o, c), want, 0.002)

print(f"\n==== {'ALL OK' if not FAIL else str(len(FAIL)) + ' MISMATCHES: ' + ', '.join(FAIL)} ====")
sys.exit(1 if FAIL else 0)
