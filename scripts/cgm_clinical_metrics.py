# -*- coding: utf-8 -*-
"""临床 CGM 指标:真实数据 vs 生成数据。

为什么要有这一套
----------------
面板此前只有 Context-FID 一个质量轴。它是个分布距离,数值本身没有临床含义 ——
「0.0394 比 0.0948 好」说明不了假数据能不能替真数据做研究。
Moscardó et al., Diabetes Technol Ther 2020;22(10):719-726 给了一组标准指标,
而且给了每个指标的**判别比 (DR)**:区分不同病人的能力。DR 高的指标,
生成模型没学像的代价也更大。所以下面按论文的 DR 从高到低排。

论文的三条推荐:MAG(变异性,DR 2.98)、%T<54(低血糖)、%T 70-180(达标时间)。

⚠️ 与论文的一处偏离,必须写明:论文按每人 12 天的窗口算,并说 GV 指标要 12 天
才稳定。我们的窗口是 1 天(T=288)或 7 天(T=2016),所以单窗口的读数比论文的抖。
但这里比的不是「某个人的 MAG 是多少」,而是「真假两批窗口的 MAG 分布像不像」,
两边窗口长度完全相同,所以比较本身是公平的。

⚠️ 自检:同一套指标会算在 copy_paste 上。它就是训练窗口有放回重采样,
所以每个指标都必须和真实值几乎一致。对不上的那个指标就是实现错了。

    qsub -v TASK=scripts/cgm_clinical_metrics.py scripts/pbs/dev/fetch.pbs
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cgmoutlier.data.cohort import load as load_cohort, channel_raw   # noqa: E402

MGDL_PER_MMOL = 18.0182
DT_MIN = 5.0


# ── 变异性指标 ────────────────────────────────────────────────────────────────
def mag(G):
    """Mean Absolute Glucose,mg/dL 每小时。论文里 DR 最高的一个 (2.98)。
    定义:相邻两点差的绝对值求和,除以总时长(小时)。Hermanides 2010。
    注意:在等间隔无缺口采样下,它和 AARC(平均绝对变化率)是同一个数 ——
    Σ|ΔG| / 总小时数 == mean|ΔG| / (5/60)。所以这里不再单列 AARC,免得
    同一个量占两列、看起来像两个独立证据。"""
    d = np.abs(np.diff(G, axis=1))
    hours = (G.shape[1] - 1) * DT_MIN / 60.0
    return d.sum(1) / hours


def gvp(G):
    """Glycemic Variability Percentage,%。曲线长度比一条平线长出多少。Peyser 2018。"""
    dg = np.diff(G, axis=1)
    L = np.sqrt(DT_MIN ** 2 + dg ** 2).sum(1)
    L0 = DT_MIN * (G.shape[1] - 1)
    return (L / L0 - 1.0) * 100.0


def conga(G, hours=1):
    """CONGA-n:与 n 小时前的差值的标准差。McDonnell 2005。"""
    k = int(hours * 60 / DT_MIN)
    d = G[:, k:] - G[:, :-k]
    return d.std(1, ddof=1)


def mage(G):
    """MAGE:超过 1 个标准差的波动的平均幅度。Service 1970。
    ⚠️ 这个指标有名地依赖实现(怎么找转折点、算升还是算降)。这里用
    「相邻极值之间、幅度超过窗口自身 SD 的那些波动取平均」,并且升降都算。
    因为真假两边用的是同一份实现,所以对比仍然成立;但它的绝对值不要
    拿去和别的文献直接比。"""
    out = np.empty(len(G))
    for i, g in enumerate(G):
        sd = g.std(ddof=1)
        d = np.diff(g)
        s = np.sign(d)
        s = s[s != 0]
        if len(s) < 2:
            out[i] = 0.0
            continue
        turn = np.flatnonzero(np.diff(np.sign(np.diff(g))) != 0) + 1
        idx = np.concatenate(([0], turn, [len(g) - 1]))
        amp = np.abs(np.diff(g[idx]))
        big = amp[amp > sd]
        out[i] = big.mean() if len(big) else 0.0
    return out


def modd(G):
    """MODD:与前一天同一时刻的差的绝对值的均值。只有多天窗口才有定义。"""
    k = int(24 * 60 / DT_MIN)
    if G.shape[1] <= k:
        return None
    return np.abs(G[:, k:] - G[:, :-k]).mean(1)


# ── 风险 / 质量指标 ───────────────────────────────────────────────────────────
def _kovatchev_f(G):
    """Kovatchev 的对称化变换。LBGI/HBGI/ADRR 都建立在它上面。Kovatchev 1997。"""
    # ⚠️ 生成器偶尔吐出非生理甚至负的血糖值,而 ln(BG)^1.084 在 BG<1 时是
    # 负数的非整数次幂 -> nan,再取均值整列就全毁了,而且不报错只给个 warning。
    # 先夹到 20-600 mg/dL 这个生理区间,和临床软件的惯例一致。
    return 1.509 * (np.log(np.clip(G, 20.0, 600.0)) ** 1.084 - 5.381)


def lbgi_hbgi(G):
    f = _kovatchev_f(G)
    r = 10.0 * f ** 2
    return np.where(f < 0, r, 0.0).mean(1), np.where(f > 0, r, 0.0).mean(1)


def adrr(G):
    """ADRR:每天取当日最大低血糖风险 + 最大高血糖风险,再对天数平均。Kovatchev 2006。"""
    per_day = int(24 * 60 / DT_MIN)
    days = G.shape[1] // per_day
    if days < 1:
        return None
    f = _kovatchev_f(G[:, :days * per_day]).reshape(len(G), days, per_day)
    r = 10.0 * f ** 2
    lr = np.where(f < 0, r, 0.0).max(2)
    hr = np.where(f > 0, r, 0.0).max(2)
    return (lr + hr).mean(1)


def m_value(G, ref=100.0):
    """M-value(M100):对偏离参考值的三次方惩罚。Schlichtkrull 1965。"""
    return (np.abs(10.0 * np.log10(np.clip(G, 1e-3, None) / ref)) ** 3).mean(1)


def grade(G):
    """GRADE 及其低/达标/高三段占比。Hill 2007。单位要先换成 mmol/L。"""
    g = np.clip(G, 1e-3, None) / MGDL_PER_MMOL
    gr = np.clip(425.0 * (np.log10(np.log10(np.clip(g, 1.0001, None))) + 0.16) ** 2, 0, 50)
    tot = gr.sum(1)
    tot = np.where(tot == 0, 1e-12, tot)
    lo = np.where(g < 3.9, gr, 0.0).sum(1) / tot * 100.0
    hi = np.where(g > 7.8, gr, 0.0).sum(1) / tot * 100.0
    return np.median(gr, axis=1), lo, hi


def igc(G, ult=140.0, llt=80.0, a=1.1, b=2.0, d=30.0):
    """IGC = 高血糖指数 + 低血糖指数,用论文说的默认参数(IGC1)。Rodbard 2009。"""
    n = G.shape[1]
    hgi = (np.clip(G - ult, 0, None) ** a).sum(1) / (n * d)
    lgi = (np.clip(llt - G, 0, None) ** b).sum(1) / (n * d)
    return hgi + lgi


def j_index(G):
    """J-index = 0.001 x (均值 + 标准差)^2,mg/dL。Wojcicki 1995。"""
    return 0.001 * (G.mean(1) + G.std(1, ddof=1)) ** 2


def pct(G, lo=None, hi=None):
    m = np.ones_like(G, dtype=bool)
    if lo is not None:
        m &= G >= lo
    if hi is not None:
        m &= G < hi
    return m.mean(1) * 100.0


# 按论文表 1/表 2 的判别比从高到低排。DR 是论文的值,不是我们算的。
def all_metrics(G):
    lb, hb = lbgi_hbgi(G)
    gr, grlo, grhi = grade(G)
    out = [
        ("MAG",          "变异性", 2.98, mag(G)),
        ("GVP %",        "变异性", 2.20, gvp(G)),
        ("J-index",      "变异性", 1.85, j_index(G)),
        ("CONGA-1",      "变异性", 1.73, conga(G, 1)),
        ("SD",           "变异性", 1.62, G.std(1, ddof=1)),
        ("MAGE",         "变异性", 1.56, mage(G)),
        ("CV %",         "变异性", 1.33, 100.0 * G.std(1, ddof=1) / G.mean(1)),
        ("%T<54",        "低血糖", 2.15, pct(G, hi=54)),
        ("LBGI",         "低血糖", 1.93, lb),
        ("%T<70",        "低血糖", 1.51, pct(G, hi=70)),
        ("M-value",      "总体",   2.00, m_value(G)),
        ("IGC",          "总体",   1.92, igc(G)),
        ("GRADE",        "总体",   1.67, gr),
        ("%T70-180",     "总体",   1.63, pct(G, 70, 180)),
        ("HBGI",         "高血糖", 1.83, hb),
        ("%T>180",       "高血糖", 1.59, pct(G, lo=180)),
        ("%T>140",       "高血糖", 1.54, pct(G, lo=140)),
        ("%GRADEhyper",  "高血糖", 1.50, grhi),
        ("平均血糖",      "参考",   None, G.mean(1)),
        ("%T>250",       "参考",   None, pct(G, lo=250)),
    ]
    md = modd(G)
    if md is not None:
        out.append(("MODD", "变异性", 1.47, md))
        out.append(("ADRR", "变异性", 1.81, adrr(G)))
    return out


# ── 论文之外、但临床报告里常见的几个,一并列出便于对照 ──────────────────────────
def gmi(G):
    """Glucose Management Indicator %,由平均血糖估算的 HbA1c。Bergenstal 2018。"""
    return 3.31 + 0.02392 * G.mean(1)


def lability_index(G):
    """Lability Index:间隔 1 小时的两点差的平方,除以间隔小时数。Ryan 2004。
    单位 (mg/dL)^2/h,数值量级大,和其他指标不可直接比大小。"""
    k = int(60 / DT_MIN)
    d = G[:, k:] - G[:, :-k]
    return (d ** 2).mean(1) / 1.0


def gri(G):
    """Glucose Risk Index,由五个时间占比线性组合。Klonoff 2023。0-100。"""
    t54 = pct(G, hi=54)
    t70 = pct(G, 54, 70)
    t250 = pct(G, lo=250)
    t180 = pct(G, 180, 250)
    return np.clip(3.0 * t54 + 2.4 * t70 + 1.6 * t250 + 0.8 * t180, 0, 100)


def bgri(G):
    """Blood Glucose Risk Index = LBGI + HBGI。Kovatchev 1997。"""
    lb, hb = lbgi_hbgi(G)
    return lb + hb


def pgs_approx(G):
    """Personal Glycemic State 的近似。Peyser 2018 的原式由四个分量相乘/相加,
    且含事件计数项。这里用其中可从单窗口稳定算出的三项:
    平均血糖项 + 变异性项(GVP) + 达标时间项,**明确标为近似**,
    只用于「真假两边是否一致」,不要拿去和文献里的 PGS 数值比。"""
    f_mg = 1.0 + 9.0 / (1 + np.exp(-0.049 * (G.mean(1) - 127.0)))
    f_gvp = 1.0 + 9.0 / (1 + np.exp(-0.047 * (gvp(G) - 38.6)))
    f_tir = 1.0 + 9.0 / (1 + np.exp(0.0833 * (pct(G, 70, 180) - 55.0)))
    return f_mg * f_gvp * f_tir / 10.0


def aarc(G):
    """Average Absolute Rate of Change,mg/dL 每分钟。
    ⚠️ 等间隔无缺口时它就是 MAG/60 —— 同一个量换了单位。两个都列是为了和
    文献对齐,但读的时候要知道它们不是两条独立证据。"""
    return mag(G) / 60.0


EXTRA = [
    ("GMI %",        "参考", None, gmi),
    ("中位血糖",      "参考", None, lambda G: np.median(G, axis=1)),
    ("Lability Index", "变异性", 2.11, lability_index),
    ("AARC",         "变异性", 1.95, aarc),
    ("CONGA-2",      "变异性", None, lambda G: conga(G, 2)),
    ("CONGA-4",      "变异性", None, lambda G: conga(G, 4)),
    ("PGS(近似)",    "总体",  1.71, pgs_approx),
    ("GRI",          "总体",  1.87, gri),
    ("BGRI",         "总体",  None, bgri),
    ("%T70-140",     "总体",  1.58, lambda G: pct(G, 70, 140)),
    ("%T70-160",     "总体",  1.63, lambda G: pct(G, 70, 160)),
    ("%T50-140",     "总体",  None, lambda G: pct(G, 50, 140)),
    ("%GRADEhypo",   "低血糖", 1.79, lambda G: grade(G)[1]),
]


def full_battery(G):
    """论文那组 + 临床报告常见的那几个。"""
    return list(all_metrics(G)) + [(n, g, d, f(G)) for n, g, d, f in EXTRA]
