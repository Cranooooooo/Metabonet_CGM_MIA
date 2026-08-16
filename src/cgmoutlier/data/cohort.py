"""Parquet -> single-variate CGM cohort.

Single channel. This module's own justification for that used to read: "the insulin
channels cost 0.0% of a generator's parameters and gate out 63% of the subjects that
have usable CGM, so dropping them is free on the model side and more than doubles the
cohort." THE 63% IS WRONG. Measured on the released file by
`scripts/channel_coverage.py`, under this module's own complete-day rule:

    CGM                       875 subjects   (exactly what this builder produces)
    CGM + basal               839            95.9%
    CGM + basal + bolus       834            95.3%
    CGM + basal + bolus+carbs 805            92.0%

The cost of the insulin channels is 4.7%, not 63%, so the first half of the argument
(they are free on the model side) stands while the half that decided the design does
not. The multi-channel builder is `scripts/build_cohort_multi.py`; this module is kept
single-channel because every published result was computed with it.

TWO THINGS THAT MUST NOT BE INHERITED FROM ANOTHER COHORT
---------------------------------------------------------
1. The normalisation constants. Every clinical metric is defined in mg/dL and is
   recovered by inverting this transform. Reusing another cohort's mean and sd leaves
   every TIR / MAGE / GRI value wrong-but-plausible, with nothing raised. They are
   recomputed here and written into the manifest beside the array.
2. The subject draw. A subject's day count drives its nearest-neighbour distance
   independently of membership, so if outliers happened to be the prolific subjects
   the study would measure data volume. The draw is stratified on day count.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd

T = 288                 # 24 h at 5-minute resolution
DT_MIN = 5
ZCLIP = 5.0
DEMO_COLS = ["age", "gender", "ethnicity", "weight", "height", "age_of_diagnosis",
             "is_pregnant", "treatment_group", "insulin_delivery_modality", "cgm_device"]


@dataclass
class Manifest:
    source: str
    channel: str
    C: int
    T: int
    dt_min: int
    zclip: float
    n_subjects: int
    n_windows: int
    min_days: int
    seed: int
    stratified_on: str
    mean_mgdl: float
    sd_mgdl: float
    denorm_formula: str = "mgdl = x * zclip * sd_mgdl + mean_mgdl"
    note: str = ("Constants are specific to this cohort. Do not reuse them for a "
                 "different subject set; every clinical metric depends on them.")


def complete_days(df: pd.DataFrame, t: int = T) -> pd.DataFrame:
    """(id, day) pairs where all `t` five-minute cells carry a CGM reading."""
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"])
    d["day"] = d["date"].dt.floor("D")
    cnt = d.groupby(["id", "day"])["CGM"].count()
    return cnt[cnt >= t].reset_index()[["id", "day"]]


def stratified_draw(n_days: pd.Series, n: int, seed: int, strata: int = 4) -> np.ndarray:
    """Equal numbers from each quartile of day count, so the cohort is not all
    high-volume subjects."""
    rng = np.random.default_rng(seed)
    q = pd.qcut(n_days, strata, labels=False, duplicates="drop")
    per = n // (q.max() + 1)
    picked = []
    for s in range(q.max() + 1):
        pool = n_days.index[q.values == s].to_numpy()
        picked.append(rng.choice(pool, size=min(per, pool.size), replace=False))
    picked = np.concatenate(picked)
    if picked.size < n:
        rest = np.setdiff1d(n_days.index.to_numpy(), picked)
        picked = np.concatenate([picked, rng.choice(rest, n - picked.size, replace=False)])
    return np.sort(picked[:n])


def build(parquet: str | Path, out: str | Path, n_subjects: int | None = None,
          min_days: int = 30, seed: int = 2026, verbose: bool = True):
    import pyarrow.parquet as pq
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    cols = ["id", "date", "CGM"] + DEMO_COLS
    have = set(pq.read_schema(parquet).names)
    df = pq.read_table(parquet, columns=[c for c in cols if c in have]).to_pandas()
    if verbose:
        print(f"[cohort] {len(df):,} rows, {df['id'].nunique():,} subjects")

    full = complete_days(df)
    per = full.groupby("id").size().rename("n_days")
    elig = per[per >= min_days]
    if verbose:
        print(f"[cohort] {len(full):,} complete days over {per.size:,} subjects; "
              f"{elig.size:,} have >= {min_days}")
    picked = (stratified_draw(elig, n_subjects, seed) if n_subjects
              else np.sort(elig.index.to_numpy()))

    keep = df[df.id.isin(picked)].sort_values(["id", "date"])
    keep["day"] = pd.to_datetime(keep["date"]).dt.floor("D")
    ok = set(map(tuple, full[full.id.isin(picked)].values))
    wins, sids = [], []
    for (sid, day), g in keep.groupby(["id", "day"], sort=True):
        if (sid, day) not in ok:
            continue
        v = g["CGM"].to_numpy(np.float32)
        if v.size < T or not np.isfinite(v[:T]).all():
            continue
        wins.append(v[:T]); sids.append(sid)
    X = np.stack(wins)[:, :, None]
    sids = np.asarray(sids)

    mu, sd = float(X.mean()), float(X.std())
    Xn = (np.clip((X - mu) / sd, -ZCLIP, ZCLIP) / ZCLIP).astype(np.float32)
    if verbose:
        print(f"[cohort] {X.shape} windows | constants mean={mu:.4f} sd={sd:.4f} mg/dL")

    np.save(out / "windows.npy", Xn)
    np.save(out / "subject_ids.npy", sids)
    demo = (df[df.id.isin(picked)][["id"] + [c for c in DEMO_COLS if c in df.columns]]
            .drop_duplicates("id").set_index("id").loc[picked].reset_index())
    demo["n_days"] = demo.id.map(per)
    demo.to_parquet(out / "subjects.parquet")
    man = Manifest(source=str(parquet), channel="CGM", C=1, T=T, dt_min=DT_MIN,
                   zclip=ZCLIP, n_subjects=int(picked.size), n_windows=int(X.shape[0]),
                   min_days=min_days, seed=seed, stratified_on="n_days",
                   mean_mgdl=mu, sd_mgdl=sd)
    (out / "manifest.json").write_text(json.dumps(asdict(man), indent=2))
    return man


def load(path: str | Path):
    """-> (windows, subject_ids, manifest). The only supported way to read a cohort.

    Accepts either `windows.npy` (memory-mapped, what `build` writes) or the committed
    `cohort.npz`. The npz holds the SAME float32 values -- it is compression, not
    quantisation, so a run against either is the same run. It costs 8 MB more than a
    float16 pack would, which is not worth having the shipped data and the checked-in
    results disagree in the fourth decimal.
    """
    p = Path(path)
    man = json.loads((p / "manifest.json").read_text())
    if (p / "windows.npy").exists():
        X = np.load(p / "windows.npy", mmap_mode="r")
        s = np.load(p / "subject_ids.npy", allow_pickle=True)
    elif (p / "cohort.npz").exists():
        z = np.load(p / "cohort.npz", allow_pickle=True)
        X, s = z["windows"], z["subject_ids"]
        if X.dtype != np.float32:
            raise ValueError(f"{p}/cohort.npz holds {X.dtype}, expected float32")
    else:
        raise FileNotFoundError(f"no windows.npy or cohort.npz under {p}")
    if X.shape[0] != s.shape[0]:
        raise ValueError(f"windows {X.shape[0]} != subject_ids {s.shape[0]}")
    if X.shape[0] != man["n_windows"] or len(set(s.tolist())) != man["n_subjects"]:
        raise ValueError(
            f"{p} does not match its manifest: {X.shape[0]} windows / "
            f"{len(set(s.tolist()))} subjects, manifest says "
            f"{man['n_windows']} / {man['n_subjects']}")
    return X, s, man


def to_mgdl(x, man):
    return np.asarray(x) * man["zclip"] * man["sd_mgdl"] + man["mean_mgdl"]


def channel_raw(X, man, name="CGM"):
    """One named channel of a cohort, in its own original units.

    Reads either manifest shape: this module writes scalar `mean_mgdl`/`sd_mgdl`, while
    `scripts/build_cohort_multi.py` writes a per-channel `channel_constants` list and
    deliberately omits the scalars, so that `to_mgdl` — which would apply CGM's
    constants to basal — fails on a multichannel cohort instead of returning
    wrong-but-plausible numbers.

    Everything clinical is defined on glucose in mg/dL, so a multichannel run still
    takes group A from this channel and this channel only.
    """
    X = np.asarray(X)
    cc = man.get("channel_constants")
    if cc is None:
        have = man.get("channel", "CGM")
        if have != name:
            raise KeyError(f"cohort holds channel {have!r}, asked for {name!r}")
        return X[..., 0] * man["zclip"] * man["sd_mgdl"] + man["mean_mgdl"]
    names = [c["name"] for c in cc]
    if name not in names:
        raise KeyError(f"cohort holds channels {names}, asked for {name!r}")
    i = names.index(name)
    return X[..., i] * man["zclip"] * cc[i]["sd"] + cc[i]["mean"]
