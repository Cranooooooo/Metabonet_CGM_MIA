"""Smoke test: package imports and computes coherent metrics.

Runs the DETERMINISTIC metrics (no training, no GPU) so it is fast and stable in
CI. Optionally exercises the neural metrics when TSGEN_TEST_NEURAL=1.

Usable both directly (`python tests/test_smoke.py`) and under pytest.
"""
import os

import numpy as np

from tsgen_metrics import evaluate, PROVENANCE


def _data():
    rng = np.random.default_rng(0)
    real = rng.standard_normal((120, 32, 3)).astype("float32")
    fake = (real + 0.1 * rng.standard_normal(real.shape)).astype("float32")
    return real, fake


def test_deterministic():
    real, fake = _data()
    res = evaluate(real, fake, subsample=120, n_seeds=2, include_neural=False, device="cpu")
    m = res["metrics"]
    expected = {"correlational", "mdd", "acd", "skewness_diff", "kurtosis_diff", "vds", "fdds"}
    assert expected.issubset(m), f"missing metrics: {expected - set(m)}"
    for key, rec in m.items():
        assert np.isfinite(rec["value"]), f"{key} not finite"
        assert rec["lower_is_better"] is True
        assert rec["abbrev"] and rec["name"] and rec["status"]
        assert PROVENANCE[key]["abbrev"] == rec["abbrev"]
    # identical sets -> ~0 for deterministic metrics
    res0 = evaluate(real, real, subsample=120, n_seeds=1, include_neural=False, device="cpu")
    assert res0["metrics"]["correlational"]["value"] < 1e-6
    assert res0["metrics"]["fdds"]["value"] < 1e-6
    print("deterministic smoke OK:", {k: round(v["value"], 5) for k, v in m.items()})


def test_neural_optional():
    if os.environ.get("TSGEN_TEST_NEURAL") != "1":
        print("neural test skipped (set TSGEN_TEST_NEURAL=1 to run)")
        return
    real, fake = _data()
    res = evaluate(real, fake, subsample=120, include_neural=True)
    for key in ("context_fid", "discriminative", "predictive"):
        assert np.isfinite(res["metrics"][key]["value"])
    print("neural smoke OK")


if __name__ == "__main__":
    test_deterministic()
    test_neural_optional()
    print("ALL SMOKE TESTS PASSED")
