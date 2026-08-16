"""Properties every outlier method must have, checked on a small fake cohort so the
suite runs anywhere in about a minute."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
FAKE = ROOT / "data" / "cohort" / "fake60"


@pytest.fixture(scope="session")
def fake():
    if not FAKE.exists():
        subprocess.run([sys.executable, str(ROOT / "scripts" / "make_fake_data.py"),
                        "--out", str(FAKE), "--n-subjects", "40"], check=True, cwd=ROOT)
    from cgmoutlier.data.cohort import load
    X, sids, man = load(FAKE)
    assert man.get("fake") is True
    return np.ascontiguousarray(np.asarray(X)[:, :, 0], np.float32), sids, man


def test_method_rng_is_independent_of_run_composition():
    """The bug this replaces: one module-level generator shared by every method in run
    order, so `--only C8` and `--only C9,C8` produced different C8 scores and neither
    reproduced a full run."""
    from cgmoutlier.outliers.common import method_rng
    a = method_rng("C8").standard_normal(50)
    b = method_rng("C8").standard_normal(50)
    c = method_rng("C9").standard_normal(50)
    assert np.array_equal(a, b), "same key must give the same stream"
    assert not np.array_equal(a, c), "different keys must give different streams"
    assert not np.array_equal(a, method_rng("C8", seed=7).standard_normal(50))


def test_subject_slices_partition_every_window(fake):
    from cgmoutlier.outliers.common import subject_slices
    _, sids, _ = fake
    subs, sl = subject_slices(sids)
    assert sum(len(v) for v in sl.values()) == len(sids)
    assert set(np.concatenate(list(sl.values())).tolist()) == set(range(len(sids)))
    for s in subs:
        assert set(sids[sl[s]]) == {s}


@pytest.mark.parametrize("key", ["C8", "C10", "E14"])
def test_distribution_methods_are_reproducible(fake, key):
    from cgmoutlier.outliers import distribution as DI
    flat, sids, _ = fake
    if key == "C10":
        f = lambda: DI.sliced_wasserstein(flat, sids, n_proj=20, n_ref=500)
    else:
        f = lambda: DI.mmd_vs_cohort(flat, sids, n_ref=500, key=key,
                                     leave_self_out=(key == "E14"))[0]
    a, b = np.asarray(f()), np.asarray(f())
    assert np.allclose(a, b), f"{key} is not reproducible under a fixed seed"
    assert np.isfinite(a).all()
    assert (a >= 0).all() or key == "E14", "a divergence from the cohort is non-negative"


def test_e13_is_exactly_the_day_count(fake):
    """The negative control has to contain no glucose information at all, or it cannot
    do its job of catching methods that are really measuring wear time."""
    from cgmoutlier.outliers.distribution import E13_day_count
    from cgmoutlier.outliers.common import subject_slices
    _, sids, _ = fake
    subs, sl = subject_slices(sids)
    assert np.array_equal(np.asarray(E13_day_count(sids)),
                          np.array([len(sl[s]) for s in subs], float))


def test_clinical_metrics_are_physiological(fake):
    from cgmoutlier.outliers import clinical as CL
    from cgmoutlier.data.cohort import load
    X, sids, man = load(FAKE)
    C = CL.per_subject_metrics(X, sids, man, verbose=False)
    assert (C["tir_70_180"].between(0, 100)).all()
    assert (C["cv"] > 0).all() and (C["cv"] < 200).all()
    assert (C["mean"].between(40, 400)).all()


def test_consensus_excludes_controls(tmp_path):
    """E13 must never vote. If it did, the outlier list would partly be a list of
    people who wore the sensor longest."""
    import pandas as pd
    from cgmoutlier.outliers.run import CONTROLS, consensus
    ids = [f"S{i}" for i in range(100)]
    for k in ["A1", "A2", "B5", "C8", "C9", "C10", "D11", "E13"]:
        # E13 ranks the ids in the exact reverse of every real method
        v = np.arange(100, dtype=float)
        pd.DataFrame({"id": ids, "score": v[::-1] if k == "E13" else v}).to_parquet(
            tmp_path / f"{k}.parquet")
    c = consensus(tmp_path, top_pct=10.0, min_methods=5, expect=7)
    assert "E13" in CONTROLS and "E13" not in c["candidates"]
    assert set(c["outliers"]) == {f"S{i}" for i in range(90, 100)}


def test_consensus_refuses_a_changed_denominator(tmp_path):
    """">=7 of 13" and ">=7 of 11" are different claims. A run that died after eleven
    methods leaves eleven score files behind and the vote silently re-bases on them."""
    import pandas as pd
    from cgmoutlier.outliers.run import consensus
    for k in ["A1", "A2", "B5"]:
        pd.DataFrame({"id": ["a", "b"], "score": [1.0, 2.0]}).to_parquet(
            tmp_path / f"{k}.parquet")
    with pytest.raises(ValueError, match="expected 13 candidate methods, found 3"):
        consensus(tmp_path)
