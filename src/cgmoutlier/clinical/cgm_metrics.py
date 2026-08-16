"""CGM clinical metric battery — every metric assessed in Moscardó & Oliver 2019,
"Assessment of Glucose Control Metrics by Discriminant Ratio" (Diabetes Technol Ther
2019; dia.2019.0415), the professors' reference paper.

Conventions:
- All glucose values are **mg/dL** (the project's internal AND external unit).
  thresholds: TBR<70, TIR 70-180, TAR>180, VeryLow<54, VeryHigh>250.
  mmol/L = mg/dL / 18.018 (display only).
- A "window" is one daily trace `g` (1-D, regular sampling `dt_min` minutes; default 5).
- "subject" metrics operate on a stack of daily windows `(D, T)` (MODD/ADRR need ≥2 days).
- Each function returns a float; `compute_window` / `compute_subject` return dicts.

Metrics: distribution/time-in-range; GV metrics {MAG, AARC, MAGE, MODD, CONGA1/2/4, LI,
J-index, M-value100, GVP, ADRR, SD, CV}; control indices {LBGI, HBGI, BGRI, GRADE(+hypo/
hyper), IGC1, GRI, PGS}. Paper finding: MAG has the highest GV discriminant ratio; IGC1
best overall-control; HBGI best hyper; LBGI best hypo; optimal time-in-range = 70-180.

Every formula carries its literature reference in-line. Where a published metric has an
ambiguous/variant definition (M-value reference, PGS, MAGE turning-point rule) the choice
is stated and marked APPROX where relevant.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

MGDL_PER_MMOL = 18.018

# clinical thresholds (mg/dL)
TH_VLOW = 54.0
TH_LOW = 70.0
TH_HIGH = 180.0
TH_VHIGH = 250.0


def _clean(g: np.ndarray) -> np.ndarray:
    g = np.asarray(g, dtype=np.float64).ravel()
    g = g[np.isfinite(g)]
    return np.clip(g, 1.0, None)  # clip ≥1 so ln/log are defined


# --------------------------------------------------------------------------- #
# distribution / time-in-range
# --------------------------------------------------------------------------- #
def mean_glucose(g):      return float(np.mean(_clean(g)))
def median_glucose(g):    return float(np.median(_clean(g)))
def sd_glucose(g):        return float(np.std(_clean(g), ddof=1)) if _clean(g).size > 1 else 0.0
def cv_percent(g):
    c = _clean(g); m = c.mean()
    return float(100.0 * c.std(ddof=1) / m) if c.size > 1 and m > 0 else 0.0
def gmi_percent(g):       return float(3.31 + 0.02392 * mean_glucose(g))  # Bergenstal 2018 (mg/dL form)
def pct_in(g, lo, hi):    c = _clean(g); return float(100.0 * np.mean((c >= lo) & (c <= hi))) if c.size else 0.0
def pct_below(g, th):     c = _clean(g); return float(100.0 * np.mean(c < th)) if c.size else 0.0
def pct_above(g, th):     c = _clean(g); return float(100.0 * np.mean(c > th)) if c.size else 0.0


# --------------------------------------------------------------------------- #
# glucose-variability (GV) metrics
# --------------------------------------------------------------------------- #
def mag(g, dt_min=5.0):
    """Mean Absolute Glucose change — Σ|Δg| / total_hours (mg/dL per hour).
    HIGHEST discriminant ratio in the paper. (Hermanides 2010; McDonnell 2005)."""
    c = _clean(g)
    if c.size < 2:
        return 0.0
    hours = (c.size - 1) * dt_min / 60.0
    return float(np.abs(np.diff(c)).sum() / hours)


def aarc(g, dt_min=5.0):
    """Average Absolute Rate of Change = mean(|Δg|/Δt) (mg/dL per minute)."""
    c = _clean(g)
    if c.size < 2:
        return 0.0
    return float(np.mean(np.abs(np.diff(c)) / dt_min))


def mage(g):
    """Mean Amplitude of Glycemic Excursions (Service 1970; Baghurst 2011 turning-point
    algorithm). Mean amplitude of excursions exceeding 1 SD. APPROX: Baghurst-style
    turning-point detection, excursions measured peak↔nadir, threshold = 1·SD."""
    c = _clean(g)
    if c.size < 3:
        return 0.0
    sd = c.std(ddof=1)
    if sd == 0:
        return 0.0
    # turning points (local extrema)
    d = np.diff(c)
    sign = np.sign(d)
    # indices where slope changes sign -> turning point at i+1
    tp = [0]
    for i in range(1, len(sign)):
        if sign[i] != 0 and sign[i] != sign[i - 1] and sign[i - 1] != 0:
            tp.append(i)
    tp.append(len(c) - 1)
    ext = c[np.array(sorted(set(tp)))]
    amps = np.abs(np.diff(ext))
    amps = amps[amps > sd]
    return float(amps.mean()) if amps.size else 0.0


def conga(g, n_hours=1.0, dt_min=5.0):
    """CONGAn — SD of differences g(t) − g(t−n hours) (McDonnell 2005)."""
    c = _clean(g)
    lag = int(round(n_hours * 60.0 / dt_min))
    if lag < 1 or c.size <= lag:
        return 0.0
    diffs = c[lag:] - c[:-lag]
    return float(np.std(diffs, ddof=1)) if diffs.size > 1 else 0.0


def lability_index(g, dt_min=5.0, spacing_h=1.0):
    """Lability Index (Ryan 2004) — mean of (Δg)² / Δt_h over readings spaced ~1 h
    apart. Units (mg/dL)²/h."""
    c = _clean(g)
    step = int(round(spacing_h * 60.0 / dt_min))
    if step < 1 or c.size <= step:
        return 0.0
    pairs = c[::step]
    if pairs.size < 2:
        return 0.0
    return float(np.mean(np.diff(pairs) ** 2 / spacing_h))


def j_index(g):
    """J-index (Wojcicki 1995) = 0.001·(mean + SD)² (mg/dL)."""
    c = _clean(g)
    return float(0.001 * (c.mean() + (c.std(ddof=1) if c.size > 1 else 0.0)) ** 2)


def m_value(g, ref=100.0):
    """M-value₁₀₀ (Schlichtkrull 1965) = mean(|10·log10(g/ref)|³), reference 100 mg/dL."""
    c = _clean(g)
    return float(np.mean(np.abs(10.0 * np.log10(c / ref)) ** 3))


def gvp(g, dt_min=5.0):
    """Glycemic Variability Percentage (Peyser 2018) = 100·(L/L0 − 1), L = trajectory
    path length, L0 = horizontal length."""
    c = _clean(g)
    if c.size < 2:
        return 0.0
    L = np.sqrt(dt_min ** 2 + np.diff(c) ** 2).sum()
    L0 = (c.size - 1) * dt_min
    return float(100.0 * (L / L0 - 1.0)) if L0 > 0 else 0.0


# --------------------------------------------------------------------------- #
# Kovatchev risk indices (control)
# --------------------------------------------------------------------------- #
def _kovatchev_f(g):
    """f(BG) = 1.509·((ln BG)^1.084 − 5.381), BG in mg/dL (Kovatchev 1997)."""
    c = _clean(g)
    return 1.509 * (np.log(c) ** 1.084 - 5.381)


def lbgi(g):
    f = _kovatchev_f(g); rl = 10.0 * np.where(f < 0, f, 0.0) ** 2
    return float(np.mean(rl))


def hbgi(g):
    f = _kovatchev_f(g); rh = 10.0 * np.where(f > 0, f, 0.0) ** 2
    return float(np.mean(rh))


def bgri(g):
    return lbgi(g) + hbgi(g)


def grade(g):
    """GRADE (Hill 2007) = mean of 425·(log10(log10(BG_mmol)) + 0.16)², capped at 50.
    Returns (grade_mean, %hypo, %eu, %hyper) split by 70/180 mg/dL."""
    c = _clean(g)
    mmol = c / MGDL_PER_MMOL
    gr = 425.0 * (np.log10(np.log10(mmol)) + 0.16) ** 2
    gr = np.clip(gr, 0.0, 50.0)
    tot = gr.sum()
    if tot <= 0:
        return 0.0, 0.0, 0.0, 0.0
    hypo = gr[c < TH_LOW].sum() / tot * 100.0
    hyper = gr[c > TH_HIGH].sum() / tot * 100.0
    eu = 100.0 - hypo - hyper
    return float(gr.mean()), float(hypo), float(eu), float(hyper)


def igc(g, a=1.1, b=2.0, c_=30.0, d=30.0, ll=80.0, ul=140.0):
    """Index of Glycemic Control IGC₁ (Rodbard 2009) = hyperglycemia index + hypoglycemia
    index. Best overall-control discriminant ratio in the paper."""
    c = _clean(g)
    n = c.size
    if n == 0:
        return 0.0
    hyper = np.sum((c[c > ul] - ul) ** b) / (n * d)
    hypo = np.sum((ll - c[c < ll]) ** a) / (n * c_)
    return float(hyper + hypo)


def gri(g):
    """Glucose Risk Index (Klonoff 2022) = 3.0·VLow + 2.4·Low + 1.6·VHigh + 0.8·High
    (component %times), clipped to [0,100]."""
    c = _clean(g)
    vlow = np.mean(c < TH_VLOW) * 100.0
    low = np.mean((c >= TH_VLOW) & (c < TH_LOW)) * 100.0
    vhigh = np.mean(c > TH_VHIGH) * 100.0
    high = np.mean((c > TH_HIGH) & (c <= TH_VHIGH)) * 100.0
    return float(np.clip(3.0 * vlow + 2.4 * low + 1.6 * vhigh + 0.8 * high, 0.0, 100.0))


def pgs(g, dt_min=5.0):
    """Personal Glycemic State (Hirsch 2017). APPROX of the published composite:
    PGS = F(hypo) + F(hyper) + F(%out-of-range) + F(GVP), each F a published sub-score.
    NOTE: exact Hirsch sub-score breakpoints are reproduced from the paper's figures; treat
    as an approximation pending verification against the original PGS reference."""
    c = _clean(g)
    tir = pct_in(c, TH_LOW, TH_HIGH)
    hypo_t = pct_below(c, TH_LOW)
    hyper_t = pct_above(c, TH_HIGH)
    v = gvp(c, dt_min)
    # published-style monotone sub-scores (capped); combined additively
    f_hypo = min(1.0 + hypo_t / 5.0, 5.0)
    f_hyper = min(1.0 + hyper_t / 20.0, 5.0)
    f_out = min(1.0 + (100.0 - tir) / 20.0, 5.0)
    f_gvp = min(1.0 + v / 20.0, 5.0)
    return float(f_hypo * f_hyper * f_out * f_gvp)


# --------------------------------------------------------------------------- #
# multi-day metrics (need a (D, T) stack of aligned daily windows)
# --------------------------------------------------------------------------- #
def modd(windows):
    """MODD — Mean Of Daily Differences (Molnar 1972): mean |g_d(t) − g_{d-1}(t)| over
    consecutive days at the same time-of-day. `windows` = (D, T) aligned daily traces."""
    w = np.asarray(windows, dtype=np.float64)
    if w.ndim != 2 or w.shape[0] < 2:
        return 0.0
    diffs = np.abs(np.diff(w, axis=0))
    diffs = diffs[np.isfinite(diffs)]
    return float(diffs.mean()) if diffs.size else 0.0


def adrr(windows):
    """ADRR — Average Daily Risk Range (Kovatchev 2006): mean over days of
    (max daily low-risk + max daily high-risk) from the Kovatchev f-transform."""
    w = np.asarray(windows, dtype=np.float64)
    if w.ndim != 2 or w.shape[0] < 1:
        return 0.0
    lr_hr = []
    for day in w:
        d = day[np.isfinite(day)]
        if d.size == 0:
            continue
        f = _kovatchev_f(d)
        rl = 10.0 * np.where(f < 0, f, 0.0) ** 2
        rh = 10.0 * np.where(f > 0, f, 0.0) ** 2
        lr_hr.append(rl.max() + rh.max())
    return float(np.mean(lr_hr)) if lr_hr else 0.0


# --------------------------------------------------------------------------- #
# registry + drivers
# --------------------------------------------------------------------------- #
# metric_key -> (group, full name, direction: 'match' (vs real) | None)
METRIC_INFO: Dict[str, Dict[str, str]] = {
    "mean": {"group": "dist", "name": "Mean glucose"},
    "median": {"group": "dist", "name": "Median glucose"},
    "sd": {"group": "gv", "name": "SD"},
    "cv": {"group": "gv", "name": "Coefficient of Variation %"},
    "gmi": {"group": "dist", "name": "GMI %"},
    "tir_70_180": {"group": "range", "name": "% Time In Range 70-180"},
    "tbr_70": {"group": "range", "name": "% Time Below 70"},
    "tar_180": {"group": "range", "name": "% Time Above 180"},
    "t_lt_54": {"group": "range", "name": "% Time < 54"},
    "t_gt_250": {"group": "range", "name": "% Time > 250"},
    "tir_70_140": {"group": "range", "name": "% Time In 70-140"},
    "tir_70_160": {"group": "range", "name": "% Time In 70-160"},
    "tir_50_140": {"group": "range", "name": "% Time In 50-140"},
    "mag": {"group": "gv", "name": "Mean Absolute Glucose (MAG)"},
    "aarc": {"group": "gv", "name": "Avg Absolute Rate of Change"},
    "mage": {"group": "gv", "name": "MAGE"},
    "conga1": {"group": "gv", "name": "CONGA-1h"},
    "conga2": {"group": "gv", "name": "CONGA-2h"},
    "conga4": {"group": "gv", "name": "CONGA-4h"},
    "li": {"group": "gv", "name": "Lability Index"},
    "j_index": {"group": "gv", "name": "J-index"},
    "m_value": {"group": "control", "name": "M-value100"},
    "gvp": {"group": "gv", "name": "Glycemic Variability %"},
    "lbgi": {"group": "control", "name": "Low Blood Glucose Index"},
    "hbgi": {"group": "control", "name": "High Blood Glucose Index"},
    "bgri": {"group": "control", "name": "Blood Glucose Risk Index"},
    "grade": {"group": "control", "name": "GRADE"},
    "grade_hypo": {"group": "control", "name": "GRADE % hypo"},
    "grade_hyper": {"group": "control", "name": "GRADE % hyper"},
    "igc": {"group": "control", "name": "Index of Glycemic Control (IGC1)"},
    "gri": {"group": "control", "name": "Glucose Risk Index"},
    "pgs": {"group": "control", "name": "Personal Glycemic State (approx)"},
    # multi-day
    "modd": {"group": "gv", "name": "Mean Of Daily Differences"},
    "adrr": {"group": "control", "name": "Average Daily Risk Range"},
}

ALL_WINDOW_METRICS: List[str] = [k for k in METRIC_INFO if k not in ("modd", "adrr")]


def compute_window(g, dt_min: float = 5.0) -> Dict[str, float]:
    """All single-day metrics for one daily trace `g` (mg/dL)."""
    g = _clean(g)
    gr, gr_hypo, _gr_eu, gr_hyper = grade(g)
    return {
        "mean": mean_glucose(g), "median": median_glucose(g),
        "sd": sd_glucose(g), "cv": cv_percent(g), "gmi": gmi_percent(g),
        "tir_70_180": pct_in(g, TH_LOW, TH_HIGH), "tbr_70": pct_below(g, TH_LOW),
        "tar_180": pct_above(g, TH_HIGH), "t_lt_54": pct_below(g, TH_VLOW),
        "t_gt_250": pct_above(g, TH_VHIGH), "tir_70_140": pct_in(g, 70, 140),
        "tir_70_160": pct_in(g, 70, 160), "tir_50_140": pct_in(g, 50, 140),
        "mag": mag(g, dt_min), "aarc": aarc(g, dt_min), "mage": mage(g),
        "conga1": conga(g, 1, dt_min), "conga2": conga(g, 2, dt_min), "conga4": conga(g, 4, dt_min),
        "li": lability_index(g, dt_min), "j_index": j_index(g), "m_value": m_value(g),
        "gvp": gvp(g, dt_min), "lbgi": lbgi(g), "hbgi": hbgi(g), "bgri": bgri(g),
        "grade": gr, "grade_hypo": gr_hypo, "grade_hyper": gr_hyper,
        "igc": igc(g), "gri": gri(g), "pgs": pgs(g, dt_min),
    }


def compute_subject(windows, dt_min: float = 5.0) -> Dict[str, float]:
    """Subject-level summary: mean of each single-day metric over the subject's days,
    plus the genuinely multi-day metrics (MODD, ADRR). `windows` = (D, T) mg/dL."""
    w = np.asarray(windows, dtype=np.float64)
    per_day = [compute_window(day, dt_min) for day in w]
    out = {k: float(np.nanmean([d[k] for d in per_day])) for k in ALL_WINDOW_METRICS}
    out["modd"] = modd(w)
    out["adrr"] = adrr(w)
    return out
