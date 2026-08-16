"""Pin the outlier list the shipped cohort produces.

This is the test that catches a port going subtly wrong. Every method here subsamples
or fits something, so "it ran without error" is a weak signal -- a broken reference
pool, a lost seed or a silently reordered subject index all produce a plausible list of
the wrong people. Comparing against a stored list is the only cheap check that the code
still computes what it computed.

The fixture is `tests/fixtures/consensus_metabonet875_seed2026.json`, and it is a
record of THIS code at THIS seed on THIS cohort. It is not ground truth about who is
atypical: change the seed, the day cap, the reference size or the cut and a different
list is equally correct. When you change a method on purpose, regenerate the fixture in
the same commit as the change, so the diff shows what moved.

Skipped unless the real cohort and a completed run are both present.
"""
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FIX = Path(__file__).parent / "fixtures" / "consensus_metabonet875_seed2026.json"
RUN = ROOT / "results" / "outliers" / "consensus.json"

pytestmark = pytest.mark.skipif(
    not (FIX.exists() and RUN.exists()),
    reason="needs the real cohort scored into results/outliers")


@pytest.fixture(scope="module")
def pair():
    return json.loads(FIX.read_text()), json.loads(RUN.read_text())


def test_same_denominator_and_cut(pair):
    fix, run = pair
    for k in ("n_subjects", "n_candidates", "cut_pct", "min_methods", "cut_n"):
        assert run[k] == fix[k], f"{k}: {run[k]} != {fix[k]}"
    assert run["candidates"] == fix["candidates"]


def test_same_outliers(pair):
    fix, run = pair
    a, b = set(fix["outliers"]), set(run["outliers"])
    assert b == a, (f"outlier list changed\n"
                    f"  lost:  {sorted(a - b)}\n"
                    f"  gained:{sorted(b - a)}")


def test_same_votes(pair):
    """A subject can stay on the list while the methods backing it change, which is a
    different code path breaking quietly."""
    fix, run = pair
    assert run["votes"] == fix["votes"]
    assert run["flagged_by"] == fix["flagged_by"]
