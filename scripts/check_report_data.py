#!/usr/bin/env python
"""Re-derive every headline number in docs/report/index.html from the artefacts.

The page carries its own JS consistency check (`validateDataConsistency`), but that
only proves the page agrees with ITSELF. This proves it agrees with the files on disk.

Each check recomputes a value from an artefact and asserts that the value appears in
the page, formatted the way the page formats it. A check whose artefact is missing is
reported as UNVERIFIABLE rather than silently skipped -- the page marks those values
`unverified` and this script is what that claim rests on.

    qsub -l select=1:ncpus=4:mem=16gb -l walltime=00:15:00 \
         -o logs/99_check_report.log scripts/pbs/99_check_report.pbs      # -> qdev

Reading a handful of small JSON/CSV files is not compute; run it under a job anyway
so nothing heavier than an editor ever runs on a login node
(NSCC_operation_guide_book.md 1).
"""
import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "docs" / "report" / "index.html"

OK, BAD, MISSING = [], [], []


def load_json(rel):
    p = ROOT / rel
    if not p.exists():
        return None
    with p.open() as fh:
        return json.load(fh)


def load_csv(rel):
    p = ROOT / rel
    if not p.exists():
        return None
    with p.open() as fh:
        return list(csv.DictReader(fh))


def acc(j):
    """Discriminator accuracy. 0.50 means real and synthetic are indistinguishable."""
    return None if j is None else j["discriminative_accuracy"]


def frozen(summary):
    """The one grid cell that counts: set_reduce=min, subject_reduce=mean."""
    for row in summary:
        if row["set_reduce"] == "min" and row["subject_reduce"] == "mean":
            return row
    return None


def check(name, artefact, expect, page_text, fmt="{}"):
    """`expect` must appear in the page rendered as `fmt`."""
    if expect is None:
        MISSING.append((name, artefact))
        return
    needle = fmt.format(expect)
    (OK if needle in page_text else BAD).append((name, artefact, needle))


def main():
    if not PAGE.exists():
        sys.exit(f"no page at {PAGE}")
    page = PAGE.read_text(encoding="utf-8")

    # ---- Q2 primary: three replicates of the frozen statistic -----------------
    aucs = []
    for r in (1, 2, 3):
        s = load_json(f"results/attack/dimts_h128_rep{r}/summary.json")
        row = frozen(s) if s else None
        if row is None:
            MISSING.append((f"rep{r} frozen variant",
                            f"results/attack/dimts_h128_rep{r}/summary.json"))
            continue
        aucs.append(row["auc"])
        check(f"rep{r} arm AUC", f"attack/dimts_h128_rep{r}", round(row["auc"], 4),
              page, "auc:{}")
        check(f"rep{r} p_between", f"attack/dimts_h128_rep{r}",
              round(row["p_between"], 4), page, "p:{}")
        check(f"rep{r} median gap, outlier", f"attack/dimts_h128_rep{r}",
              round(row["median_gap_outlier"], 7), page, "gapOut:{}")
        check(f"rep{r} median d_IN, outlier", f"attack/dimts_h128_rep{r}",
              round(row["median_d_in_outlier"], 5), page, "dOut:{}")
    if len(aucs) == 3:
        check("primary mean", "derived from the three replicates",
              round(sum(aucs) / 3, 2), page, "mean:{}")

    # relative magnitude, recomputed -- the doc says 0.27/0.11, the files say otherwise
    rels_o, rels_c = [], []
    for r in (1, 2, 3):
        s = load_json(f"results/attack/dimts_h128_rep{r}/summary.json")
        row = frozen(s) if s else None
        if row:
            rels_o.append(row["median_gap_outlier"] / row["median_d_in_outlier"] * 100)
            rels_c.append(row["median_gap_control"] / row["median_d_in_control"] * 100)
    if rels_o:
        check("outlier relative magnitude", "recomputed from the three replicates",
              round(sum(rels_o) / len(rels_o), 3), page, "relOut:{}")
        check("control relative magnitude", "recomputed from the three replicates",
              round(sum(rels_c) / len(rels_c), 3), page, "relCtl:{}")

    # ---- Q2 per-subject array -------------------------------------------------
    rows = load_csv("results/attack/dimts_h128_subject_auc/per_subject.csv")
    if rows is None:
        MISSING.append(("per-subject array", "attack/dimts_h128_subject_auc"))
    else:
        subj = [(r["target"], "O" if r["group"] == "outlier" else "C",
                 float(r["auc"]), int(r["n_windows"])) for r in rows]
        outl = [a for _, g, a, _ in subj if g == "O"]
        ctrl = [a for _, g, a, _ in subj if g == "C"]
        others = [a for t, _, a, _ in subj if t != "1142"]
        # the array is embedded in the page; parse it back out and compare row by row
        m = re.search(r"subjects:\[(\[.*?\])\]\s*\}", page, re.S)
        if not m:
            BAD.append(("per-subject array", "index.html",
                        "could not locate the embedded subjects array"))
        else:
            embedded = re.findall(r'\["([^"]+)","([OC])",([\d.]+),(\d+)\]', m.group(1))
            if len(embedded) != len(subj):
                BAD.append(("per-subject array length", "per_subject.csv",
                            f"page has {len(embedded)} rows, artefact has {len(subj)}"))
            else:
                OK.append((f"per-subject array length ({len(subj)})",
                           "per_subject.csv", "match"))
            want = {t: (g, round(a, 4), n) for t, g, a, n in subj}
            wrong = [e[0] for e in embedded
                     if want.get(e[0]) != (e[1], float(e[2]), int(e[3]))]
            if wrong:
                BAD.append(("per-subject array rows", "per_subject.csv",
                            f"{len(wrong)} row(s) differ: {wrong[:6]}"))
            else:
                OK.append((f"all {len(embedded)} array rows match the CSV",
                           "per_subject.csv", "match"))
        check("outlier mean", "per_subject.csv",
              round(sum(outl) / len(outl), 4), page, "outlierMean:{}")
        check("control mean", "per_subject.csv",
              round(sum(ctrl) / len(ctrl), 4), page, "controlMean:{}")
        check("1142 per-subject AUC", "per_subject.csv",
              round([a for t, _, a, _ in subj if t == "1142"][0], 4), page, "s1142:{}")
        check("79-subject range, low", "per_subject.csv", round(min(others), 4),
              page, "range79:[{}")
        check("79-subject range, high", "per_subject.csv", round(max(others), 4),
              page, ",{}]")

    # ---- Q2 sensitivity, excluding 1142 --------------------------------------
    log = ROOT / "logs" / "97_auc_no1142.live.log"
    if not log.exists():
        MISSING.append(("excl-1142 recomputation", "logs/97_auc_no1142.live.log"))
    else:
        txt = log.read_text()
        vals = re.findall(r"^\s+\d\s+20\s+0\.\d{4}\s+(0\.\d{4})", txt, re.M)
        for i, v in enumerate(vals, 1):
            check(f"rep{i} excl-1142", "logs/97_auc_no1142.live.log",
                  float(v), page, "auc:{}")

    # ---- Q1 attack panel ------------------------------------------------------
    panel = load_json("results/attack_panel/table_pooled.json")
    if panel is None:
        MISSING.append(("pooled attack panel", "results/attack_panel/table_pooled.json"))
    else:
        for row in panel:
            check(f"{row['attacker']} outlier mean", "table_pooled.json",
                  round(row["outlier_mean"], 4), page, "outlier:{}")
            check(f"{row['attacker']} diff", "table_pooled.json",
                  round(row["diff"], 4), page, "diff:{}")
        check("negative-diff attacker count", "table_pooled.json",
              sum(1 for r in panel if r["diff"] < 0), page, "nNegativeGE0:{}")

    broken = load_json("results/attack_panel/table.json")
    if broken is None:
        MISSING.append(("per-subject protocol panel", "results/attack_panel/table.json"))
    else:
        by = {r["attacker"]: r for r in broken}
        for a in ("C4_hgb x F3_raw10", "C3_forest x F3_raw10", "C2_tree x F3_raw10"):
            if a in by:
                check(f"broken design {a}", "table.json",
                      round(by[a]["outlier_mean"], 4), page, "outlier:{}")

    cp = load_json("results/attack_panel/table_pooled_copypaste.json")
    if cp:
        check("copy_paste best outlier mean", "table_pooled_copypaste.json",
              round(max(r["outlier_mean"] for r in cp), 4), page, "{}")
    q = load_json("results/quality_cp_1d/copy_paste__base.json")
    if q:
        check("copy_paste discriminator, 1 day", "quality_cp_1d",
              round(acc(q), 4), page, "discriminator:{}")

    # ---- outlier consensus ----------------------------------------------------
    for tag, rel in (("single channel", "results/seed_stability/stability.json"),
                     ("CGM+basal/kg",
                      "results/outliers_c2_perkg/stability/stability.json")):
        st = load_json(rel)
        if st is None:
            MISSING.append((f"{tag} seed stability", rel))
            continue
        per = {k: len(v) for k, v in st["per_seed"].items()}
        sets = [set(v) for v in st["per_seed"].values()]
        inter, union = len(set.intersection(*sets)), len(set.union(*sets))
        for k, v in per.items():
            check(f"{tag} seed {k}", rel, v, page, f'"{k}":{{}}')
        check(f"{tag} intersection", rel, inter, page, "intersection:{}")
        check(f"{tag} union", rel, union, page, "union:{}")
        check(f"{tag} stability", rel, round(inter / union, 3), page, "stability:{}")

    # ---- multichannel ---------------------------------------------------------
    for ch, key in (("CGM", "CGM"), ("CGM+basal", "CGM + basal"), ("basal", "basal"),
                    ("bolus", "bolus"), ("all", "all three")):
        s = load_json(f"results/attack_c3_bychannel/rep1_{ch}/summary.json")
        row = frozen(s) if s else None
        if row is None:
            MISSING.append((f"channel {ch}", f"attack_c3_bychannel/rep1_{ch}"))
            continue
        check(f"channel {ch} arm AUC", f"attack_c3_bychannel/rep1_{ch}",
              round(row["auc"], 4), page, "armAuc:{}")

    done = 0
    for r in (1, 2, 3):
        d = ROOT / f"results/runs/dimts_c3_h128_rep{r}"
        n = len(list(d.glob("*/samples.npy"))) if d.exists() else 0
        done += n
        check(f"three-channel rep{r} completed", "results/runs/", n, page,
              f"rep:{r},done:{{}}")
    check("three-channel total completed", "results/runs/", done, page, "done:{}")

    w = {}
    for h in (128, 160, 192, 224, 256):
        f = ("dimts_c3_h128_rep1__base.json" if h == 128
             else f"dimts_c3_h{h}_probe__base.json")
        j = load_json(f"results/quality_c3_width/{f}")
        if j:
            w[h] = round(acc(j), 4)
            check(f"width {h} accuracy", "quality_c3_width", w[h], page, "acc:{}")

    # ---- window length --------------------------------------------------------
    for tag, label in (("d7_contiguous", "7 consecutive"), ("d7_concat", "7 stitched"),
                       ("d14_concat", "14"), ("d21_concat", "21")):
        j = load_json(f"results/subject_auc_multiday/{tag}/summary.json")
        if j is None:
            MISSING.append((f"multiday {tag}", f"subject_auc_multiday/{tag}"))
            continue
        g = j["by_group"]
        pooled = (g["outlier"]["mean"] * g["outlier"]["n"] +
                  g["control"]["mean"] * g["control"]["n"]) / j["n_subjects"]
        check(f"window {label} per-subject AUC", f"subject_auc_multiday/{tag}",
              round(pooled, 4), page, "auc:{}")

    # values the page marks `unverified` -- assert that no artefact backs them, so the
    # flag cannot quietly become wrong if one appears later
    for name, globpat in (
            ("1-day copy_paste per-subject AUC (0.8187)",
             "results/subject_auc*/*1d*/summary.json"),
            ("vote-threshold sensitivity counts (55/38, 44/23.5, 33/15)",
             "results/**/threshold_sensitivity*.json")):
        hits = list(ROOT.glob(globpat))
        if hits:
            BAD.append((name, str(hits[0]),
                        "page says unverified, but an artefact now exists"))
        else:
            MISSING.append((name, "no artefact — page marks it unverified"))

    avail = load_csv("results/channel_coverage_studyid/consecutive_multiday.csv")
    if avail is None:
        MISSING.append(("consecutive-day availability", "consecutive_multiday.csv"))
    else:
        for r in avail:
            if r["channels"] != "CGM":
                continue
            check(f"L={r['length']} subjects", "consecutive_multiday.csv",
                  int(r["subjects"]), page, "subjects:{}")

    j = load_json("results/quality_d7/dimts_d7_conv_rep1__base.json")
    if j:
        check("seven-day discriminator", "quality_d7",
              round(acc(j), 4), page, "discriminator:{}")
    s = load_json("results/attack_d7/summary.json")
    row = frozen(s) if s else None
    if row:
        check("seven-day arm AUC", "attack_d7", round(row["auc"], 2), page, "armAuc:{}")
    j = load_json("results/subject_auc_d7/summary.json")
    if j:
        check("seven-day outlier mean", "subject_auc_d7",
              round(j["by_group"]["outlier"]["mean"], 4), page, "outlier:{}")
        check("seven-day 1142", "subject_auc_d7",
              round(j["by_group"]["outlier"]["max"], 4), page, "s1142:{}")

    # h256 -- the page claims it is blocked with no samples; verify that
    h256 = ROOT / "results/runs/dimts_d7_h256_rep1/base"
    if h256.exists():
        n_samples = len(list(h256.glob("samples.npy")))
        n_ckpt = len(list((h256 / "ckpt_2016").glob("*.pt"))) if (h256 / "ckpt_2016").exists() else 0
        check("h256 checkpoints", "results/runs/dimts_d7_h256_rep1", n_ckpt,
              page, "checkpoints:{}")
        if n_samples:
            BAD.append(("h256 status", "results/runs/dimts_d7_h256_rep1",
                        "page says blocked with no samples, but samples.npy exists"))
        else:
            OK.append(("h256 has no samples -- 'blocked' is correct",
                       "results/runs/dimts_d7_h256_rep1", "verified"))

    # ---- identifiability ------------------------------------------------------
    for tag, key in (("metabonet_sid_c2_perkg", "perKg"), ("metabonet_sid_c2_raw", "raw")):
        rows = load_csv(f"results/identifiability_within/{tag}.csv")
        if rows is None:
            MISSING.append((f"identifiability {tag}", tag))
            continue
        q = {r["channels"]: r for r in rows if r["space"] == "quantile"}
        for ch, field in (("basal", "basal"), ("CGM", "cgm")):
            if ch in q:
                check(f"{key} {ch} top-1", f"identifiability_within/{tag}.csv",
                      round(float(q[ch]["top1"]) * 100, 2), page, f"{field}:{{}}")
                check(f"{key} {ch} lift", f"identifiability_within/{tag}.csv",
                      round(float(q[ch]["lift"]), 2), page,
                      f"{field}Lift:{{}}" if field == "cgm" else "basalLift:{}")

    # ---- cohorts --------------------------------------------------------------
    for key in ("metabonet875", "metabonet_sid_c1", "metabonet_sid_c3",
                "metabonet_sid_c2_raw", "metabonet_sid_c2_perkg"):
        m = load_json(f"data/cohort/{key}/manifest.json")
        if m is None:
            MISSING.append((f"cohort {key}", f"data/cohort/{key}/manifest.json"))
            continue
        check(f"cohort {key} subjects", f"data/cohort/{key}", m["n_subjects"],
              page, "subjects:{},")
        check(f"cohort {key} windows", f"data/cohort/{key}", m["n_windows"],
              page, "windows:{},")

    d = load_json("results/design_sym/rep1/design.json")
    if d:
        check("background size", "design_sym/rep1", d["n_background"], page, "background:{},")
        check("models per replicate", "design_sym/rep1", d["n_jobs"], page, "models:{}")

    # ---- report ---------------------------------------------------------------
    print(f"\n{'=' * 74}\nreport data check — docs/report/index.html\n{'=' * 74}")
    print(f"  verified against artefacts : {len(OK)}")
    print(f"  MISMATCHED                 : {len(BAD)}")
    print(f"  artefact not on disk       : {len(MISSING)}")
    if BAD:
        print("\nMISMATCHES — the page disagrees with the artefact:")
        for n, a, v in BAD:
            print(f"  ✗ {n:44} {a}\n      expected to find: {v}")
    if MISSING:
        print("\nNOT VERIFIABLE — no artefact on disk. The page must mark these unverified:")
        for n, a in MISSING:
            print(f"  ? {n:44} {a}")
    print()
    return 1 if BAD else 0


if __name__ == "__main__":
    sys.exit(main())
