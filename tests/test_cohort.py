"""The cohort contract: shapes, ids, and the normalisation round-trip."""
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
COHORT = ROOT / "data" / "cohort" / "metabonet875"
pytestmark = pytest.mark.skipif(not COHORT.exists(), reason="real cohort not present")


@pytest.fixture(scope="module")
def coh():
    from cgmoutlier.data.cohort import load
    return load(COHORT)


def test_shapes_agree(coh):
    X, sids, man = coh
    assert X.ndim == 3 and X.shape[1] == man["T"] and X.shape[2] == man["C"] == 1
    assert len(sids) == X.shape[0] == man["n_windows"]
    assert len(set(sids)) == man["n_subjects"]


def test_ids_are_strings(coh):
    """A subject id is a label, not a number. Casting to int has bitten this code
    before: `int('P02')` raises, and `int('007') == 7` silently merges subjects."""
    _, sids, _ = coh
    assert all(isinstance(s, str) for s in np.unique(sids)[:20])


def test_normalisation_is_in_range(coh):
    X, _, man = coh
    assert np.abs(X).max() <= 1.0 + 1e-6, "clip is at +-zclip then divided by zclip"


def test_denormalises_to_physiological_glucose(coh):
    """The clinical metrics are defined in mg/dL, so this transform has to be right.
    A wrong sd here would move TIR without moving anything that looks broken."""
    from cgmoutlier.data.cohort import to_mgdl
    X, _, man = coh
    g = to_mgdl(X[:2000], man)
    assert 20 < g.min() < 90, g.min()
    assert 250 < g.max() < 600, g.max()
    assert 100 < g.mean() < 200, g.mean()


def test_manifest_constants_come_from_this_cohort(coh):
    """Normalisation constants must never be inherited from another cohort; the whole
    mg/dL layer silently shifts if they are."""
    X, _, man = coh
    recovered_mean = float(X.mean()) * man["zclip"] * man["sd_mgdl"] + man["mean_mgdl"]
    assert abs(recovered_mean - man["mean_mgdl"]) < 2.0


def test_every_subject_clears_the_day_floor(coh):
    _, sids, man = coh
    _, counts = np.unique(sids, return_counts=True)
    assert counts.min() >= man["min_days"]
