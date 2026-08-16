"""Run the fourteen methods over a cohort and write one file per score column.

Per-method files rather than one table: a failure in the autoencoder should not cost
the two hours the clinical battery took, and a rerun should skip what exists.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..data.cohort import load
from . import clinical as CL, distribution as DI, learned as LE, shape as SH
from .common import SEED, subject_slices

# The default run. D12 is implemented and deliberately not in here: it needs a GPU and
# 20 epochs of training, the learned group is already carried by C9 and D11, and the
# published 875-subject result was computed over these fourteen. Adding it silently
# would change the denominator of the vote (7-of-13 becomes 7-of-14) without anyone
# choosing to. Run it with `--only D12` and pass a matching --min-methods.
ALL = ["A1", "A2", "A3", "A4", "B5", "B6", "B7a", "B7b",
       "C8", "C9", "C10", "D11", "E13", "E14"]
OPTIONAL = ["D12"]
CONTROLS = ["E13"]                      # not candidates; excluded from any consensus
N_CANDIDATES = len([k for k in ALL if k not in CONTROLS])   # 13


def run(cohort, out, only=None, dtw_per_subject=30, device="cuda", seed=SEED,
        channels="all", clinical_cache=None, verbose=True):
    """Score every subject. `channels` is "all", or a list of channel names.

    WHICH METHODS SEE WHICH CHANNELS
    --------------------------------
    Group A (A1-A4) is always CGM alone: its metrics are the Battelino consensus, all
    defined on glucose in mg/dL, and a basal rate has no time in range. So four of the
    thirteen votes are identical between a one-channel and a three-channel run, and the
    denominator stays 13 -- ">= 7 of 13" means the same thing in both, which is what
    makes the two outlier lists comparable at all. It also makes the comparison
    CONSERVATIVE: a third of the vote cannot move, so a difference between the lists is
    a floor on the effect of adding channels, not a ceiling.

    B7, C8, C10, E14 take the channels flattened to one T*C vector -- they are k-NN,
    MMD and sliced-Wasserstein, all of which only need a vector, and the cohort's
    per-channel z-score already puts the channels on a common scale.

    B5/B6 keep the channel axis: DTW aligns a path through C-dimensional space, and
    flattening would instead lay C days end to end and align that.

    C9/D11 embed with TS2Vec, which takes `input_dims` from the data. The encoder cache
    is keyed on a fingerprint that includes the array shape, so a three-channel run
    trains its own encoder rather than silently reusing the one-channel one.
    """
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    want = set(only or ALL)
    X, sids, man = load(cohort)
    X = np.asarray(X)
    names = man.get("channels") or [man.get("channel", "CGM")]
    if channels == "all":
        cols = list(range(X.shape[-1]))
    else:
        chans = channels.split(",") if isinstance(channels, str) else list(channels)
        missing = [c for c in chans if c not in names]
        if missing:
            raise ValueError(f"cohort holds {names}, asked for {missing}")
        cols = [names.index(c) for c in chans]
    cur = np.ascontiguousarray(X[..., cols], np.float32)          # (N, T, c)
    flat = np.ascontiguousarray(cur.reshape(cur.shape[0], -1), np.float32)
    subs, _ = subject_slices(sids)
    meta = {"channels": [names[i] for i in cols], "channel_dim": int(flat.shape[1])}
    if verbose:
        print(f"  channels {meta['channels']} -> flat dim {flat.shape[1]}", flush=True)

    def save(key, v, extra=None):
        pd.DataFrame({"id": subs, "score": np.asarray(v, float),
                      **(extra or {})}).to_parquet(out / f"{key}.parquet")
        if verbose:
            print(f"  -> {key}", flush=True)

    def todo(k):
        return k in want and not (out / f"{k}.parquet").exists()

    if want & {"A1", "A2", "A3", "A4"}:
        # Group A depends only on the CGM channel, so a one-channel and a three-channel
        # run over the SAME cohort produce a bit-identical table. Pointing both at one
        # cache saves recomputing 32 metrics over every window a second time; it is
        # only valid across runs on the same cohort, which is why it is an explicit
        # argument rather than a shared default path.
        cache = Path(clinical_cache) if clinical_cache else out / "_clinical.parquet"
        C = (pd.read_parquet(cache) if cache.exists()
             else CL.per_subject_metrics(X, sids, man, verbose=verbose))
        if not cache.exists():
            C.to_parquet(cache)
        C = C.loc[subs]
        for k, fn in [("A1", CL.A1), ("A2", CL.A2), ("A3", CL.A3), ("A4", CL.A4)]:
            if todo(k):
                save(k, fn(C), {"cv": C["cv"].to_numpy()} if k == "A2" else None)

    if todo("B5") or todo("B6"):
        b5, b6, share, m = SH.B5_B6(cur, sids, dtw_per_subject, seed=seed, verbose=verbose)
        meta["B5_B6"] = m
        if todo("B5"):
            save("B5", b5)
        if todo("B6"):
            save("B6", b6, {"share_L": share[:, 0], "share_M": share[:, 1],
                            "share_S": share[:, 2]})

    if todo("B7a") or todo("B7b"):
        a, b = SH.B7(flat, sids, seed=seed)
        if todo("B7a"):
            save("B7a", a)
        if todo("B7b"):
            save("B7b", b)

    if todo("C8"):
        v, m = DI.mmd_vs_cohort(flat, sids, key="C8", seed=seed); meta["C8"] = m; save("C8", v)
    if todo("E14"):
        v, m = DI.mmd_vs_cohort(flat, sids, key="E14", seed=seed, leave_self_out=True); meta["E14"] = m; save("E14", v)
    if todo("C10"):
        save("C10", DI.sliced_wasserstein(flat, sids, seed=seed))
    if todo("E13"):
        save("E13", DI.E13_day_count(sids))

    if todo("C9") or todo("D11"):
        from ..attack.learned_features import get_encoder, encode_real_cached
        model, em = get_encoder(cur)
        E = np.asarray(encode_real_cached(model, cur, em["fingerprint"]))
        meta["encoder"] = em
        if todo("D11"):
            save("D11", LE.D11_lof(E, sids))
        if todo("C9"):
            v, m = DI.mmd_vs_cohort(E, sids, key="C9", seed=seed); meta["C9"] = m; save("C9", v)

    if todo("D12"):
        mx, p95 = LE.D12_autoencoder_max(flat, sids, device=device, seed=seed, verbose=verbose)
        save("D12", mx, {"p95": p95})

    meta["seed"] = seed
    (out / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
    return out


def consensus(scores_dir, top_pct=5.0, min_methods=7, exclude=CONTROLS,
              expect=N_CANDIDATES):
    """Subjects in the top `top_pct` of at least `min_methods` candidate methods.

    Controls are excluded from the vote by name, not by hoping nobody selects them.
    The cut and the threshold are arguments because both change the answer: at the 5%
    cut the count-of-methods histogram has a trough at 3-6, which is what makes >=7 a
    reading of the data rather than a round number.

    `expect` guards the denominator. The candidate set is whatever score files are on
    disk, so a half-finished run or an extra optional method quietly turns "7 of 13"
    into "7 of 11" or "7 of 14" -- same threshold, different claim, no error. Pass
    expect=None to accept whatever is there.
    """
    d = Path(scores_dir)
    S = {p.stem: pd.read_parquet(p).set_index("id")["score"]
         for p in sorted(d.glob("[ABCDE]*.parquet"))}
    df = pd.DataFrame(S)
    cand = [c for c in df.columns if c not in exclude]
    if expect is not None and len(cand) != expect:
        raise ValueError(
            f"expected {expect} candidate methods, found {len(cand)}: {cand}\n"
            f"  in {d}\n"
            f"  A vote of >={min_methods} means something different against a different\n"
            f"  denominator. Finish the run, or pass expect={len(cand)} deliberately.")
    cut = max(1, int(len(df) * top_pct / 100))
    top = {k: set(df[k].nlargest(cut).index) for k in cand}
    cnt = pd.Series({i: sum(i in top[k] for k in cand) for i in set().union(*top.values())})
    sel = cnt[cnt >= min_methods].sort_values(ascending=False)
    return dict(
        outliers=[str(i) for i in sel.index],
        n_candidates=len(cand), candidates=cand, cut_pct=top_pct, cut_n=cut,
        min_methods=min_methods, n_subjects=int(len(df)),
        votes={str(i): int(v) for i, v in sel.items()},
        flagged_by={str(i): [k for k in cand if i in top[k]] for i in sel.index},
        histogram={str(k): int(v) for k, v in cnt.value_counts().sort_index().items()},
    )
