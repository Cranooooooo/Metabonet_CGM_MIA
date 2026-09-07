# -*- coding: utf-8 -*-
"""临床 CGM 指标电池:真实 vs 生成,按「差多少」排。

    Metric | Real | Synthetic | Δ | |Δ|/real

Δ 越小 = 这个模型把这条临床统计量复现得越好。
表头那几条是临床报告的招牌指标,其余按相对差距从大到小排 —— 排在最上面的
就是这个模型最没学像的地方。

    qsub -v TASK=scripts/report/clinical_table.py scripts/pbs/dev/fetch.pbs
"""
import json, sys
from pathlib import Path
import numpy as np

R = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(R / "src")); sys.path.insert(0, str(R / "scripts"))
from cgmoutlier.data.cohort import load as load_cohort, channel_raw     # noqa: E402
from cgm_clinical_metrics import full_battery                           # noqa: E402

# 招牌指标先列,不参与排序
HEAD = ["平均血糖", "SD", "CV %", "GMI %", "%T70-180", "%T<70", "%T>180"]

SETS = [("DiM-TS", "matrix_{c}"), ("IG-FM", "pilot_igfm_{c}"),
        ("照抄(自检)", "cp_matrix_{c}")]


def mgdl(X, man):
    return np.asarray(channel_raw(X, man, "CGM"), dtype=np.float64)


out = {}
for c in ("d1_c1", "d1_c2", "d7_c1", "d7_c2"):
    coh = R / "data" / "cohort" / f"matrix_{c}"
    man = json.loads((coh / "manifest.json").read_text())
    Xr, _, _ = load_cohort(str(coh))
    real = {n: float(np.mean(v)) for n, g, d, v in full_battery(mgdl(Xr, man))}
    meta = {n: (g, d) for n, g, d, v in full_battery(mgdl(Xr, man))}

    got = {}
    for disp, tmpl in SETS:
        p = R / "results" / "runs" / tmpl.format(c=c) / "base" / "samples.npy"
        if p.exists():
            got[disp] = {n: float(np.mean(v))
                         for n, g, d, v in full_battery(mgdl(np.load(p), man))}
    out[c] = dict(real=real, meta={k: list(v) for k, v in meta.items()}, **got)

    for disp, syn in got.items():
        rows = []
        for n, rv in real.items():
            sv = syn[n]
            rel = abs(sv - rv) / abs(rv) * 100 if rv else float("nan")
            rows.append((n, rv, sv, sv - rv, rel))
        head = [r for r in rows if r[0] in HEAD]
        rest = sorted([r for r in rows if r[0] not in HEAD], key=lambda r: -r[4])
        print(f"\n{'='*76}\n{c} · 真实 vs {disp}   (Δ 越小越像真的)")
        print(f"{'指标':<15}{'真实':>10}{'生成':>11}{'Δ':>10}{'|Δ|/真实':>10}")
        print("-" * 76)
        for tag, block in (("── 招牌指标 ──", head), ("── 其余,按相对差距排 ──", rest)):
            print(tag)
            for n, rv, sv, d, rel in block:
                print(f"{n:<15}{rv:>10.3f}{sv:>11.3f}{d:>+10.3f}{rel:>9.1f}%")

(R / "results" / "clinical").mkdir(parents=True, exist_ok=True)
(R / "results" / "clinical" / "battery.json").write_text(
    json.dumps(out, ensure_ascii=False, indent=1))
print("\n==== CLINICAL OK ====")
