"""The paired design is pure bookkeeping, so it can be checked exactly."""
import numpy as np
import pytest

from cgmoutlier.loo import build, match_controls


@pytest.fixture
def toy():
    subs = [f"S{i:03d}" for i in range(100)]
    rs = np.random.default_rng(0)
    days = {s: int(d) for s, d in zip(subs, rs.integers(30, 300, 100))}
    return subs, days


def test_job_count_and_training_set_sizes(toy):
    """Symmetric: the controls leave the pool, so the base trains on 100 - 5 - 5."""
    subs, days = toy
    d = build(subs, outliers=subs[:5], days=days)
    assert len(d.jobs) == 1 + 5 + 5
    base = next(j for j in d.jobs if j.role == "base")
    assert base.n_subjects == 90
    assert [j.role for j in d.jobs].count("exclude") == 0
    for j in d.jobs:
        if j.role == "include":
            assert j.n_subjects == 91 and j.target in j.subjects
    assert {j.group for j in d.jobs} == {None, "outlier", "control"}


def test_asymmetric_keeps_the_old_shape(toy):
    subs, days = toy
    d = build(subs, outliers=subs[:5], days=days, symmetric=False)
    assert len(d.jobs) == 1 + 5 + 5
    assert next(j for j in d.jobs if j.role == "base").n_subjects == 95
    for j in d.jobs:
        if j.role == "include":
            assert j.n_subjects == 96 and j.target in j.subjects
        if j.role == "exclude":
            assert j.n_subjects == 94 and j.target not in j.subjects


@pytest.mark.parametrize("symmetric", [True, False])
def test_each_pair_differs_by_exactly_one_subject(toy, symmetric):
    """The whole design rests on this. If two training sets differ by anything else,
    the measured gap is not attributable to membership."""
    subs, days = toy
    d = build(subs, outliers=subs[:6], days=days, symmetric=symmetric)
    by_name = {j.name: set(j.subjects) for j in d.jobs}
    for p in d.pairs:
        a, b = by_name[p["member"]], by_name[p["nonmember"]]
        assert a - b == {p["target"]}, p
        assert b - a == set(), p


def test_symmetric_puts_the_base_on_the_same_side_of_every_pair(toy):
    """The reason the symmetric form exists. When the base is the non-member for
    outliers and the member for controls, its own offset enters the two arms with
    opposite sign and shifts them apart by twice that offset -- and it is shared within
    an arm, so 20 targets do not average it away."""
    subs, days = toy
    d = build(subs, outliers=subs[:6], days=days)
    assert {p["nonmember"] for p in d.pairs} == {"base"}
    assert "base" not in {p["member"] for p in d.pairs}
    assert {p["group"] for p in d.pairs} == {"outlier", "control"}

    old = build(subs, outliers=subs[:6], days=days, symmetric=False)
    sides = {p["group"]: ("member" if p["member"] == "base" else "nonmember")
             for p in old.pairs}
    assert sides == {"outlier": "nonmember", "control": "member"}


def test_symmetric_background_is_identical_across_every_run(toy):
    """All 1 + 2n training sets are one background plus at most one target, so nothing
    but the target differs between any two of them."""
    subs, days = toy
    d = build(subs, outliers=subs[:6], days=days)
    bg = set(d.background)
    assert not bg & set(d.outliers) and not bg & set(d.controls)
    for j in d.jobs:
        assert set(j.subjects) - {j.target} == bg, j.name


def test_controls_are_normals_and_disjoint_from_outliers(toy):
    subs, days = toy
    out = subs[:8]
    d = build(subs, outliers=out, days=days)
    assert not set(d.controls) & set(out)
    assert set(d.controls) <= set(d.normals)
    assert len(set(d.controls)) == len(d.controls) == 8, "no control used twice"


def test_matching_beats_random_on_day_count():
    """The reason controls are matched at all: outliers here are short-record subjects,
    and a random draw would put a day-count difference between the two arms."""
    rs = np.random.default_rng(1)
    subs = [f"S{i:03d}" for i in range(400)]
    days = {s: int(d) for s, d in zip(subs, rs.integers(120, 300, 400))}
    outliers = subs[:20]
    for s in outliers:                       # outliers have unusually few days
        days[s] = int(rs.integers(60, 110))

    controls, rep = match_controls(days, outliers, subs)
    matched_gap = abs(np.median([days[c] for c in controls])
                      - np.median([days[s] for s in outliers]))
    random_gap = abs(np.median([days[c] for c in rs.choice(subs[20:], 20, replace=False)])
                     - np.median([days[s] for s in outliers]))
    assert matched_gap < random_gap
    assert rep["max_rel_gap"] >= 0


def test_matching_reports_when_it_cannot_match(toy):
    """Silence would be the dangerous outcome: if no close control exists, the arms
    differ on day count and the experiment cannot tell that from membership."""
    subs = [f"S{i:03d}" for i in range(50)]
    days = {s: 200 for s in subs}
    for s in subs[:5]:
        days[s] = 20                         # nothing in the pool is near 20 days
    _, rep = match_controls(days, subs[:5], subs, tol=0.15)
    assert rep["n_over_tol"] == 5
    assert "warning" in rep


def test_seed_actually_moves_the_draw(toy):
    """Regression. `seed` was a parameter match_controls accepted and never read, so
    every call returned the same greedy draw. Replicates built by varying the seed
    would have shared one control set and measured only training noise."""
    subs, days = toy
    a, _ = match_controls(days, subs[:8], subs, seed=1, n_candidates=6)
    b, _ = match_controls(days, subs[:8], subs, seed=2, n_candidates=6)
    assert a != b


def test_k_one_is_still_the_deterministic_greedy_draw(toy):
    subs, days = toy
    a, _ = match_controls(days, subs[:8], subs, seed=1, n_candidates=1)
    b, _ = match_controls(days, subs[:8], subs, seed=999, n_candidates=1)
    assert a == b, "k=1 must not depend on the seed, or old designs stop rebuilding"


def test_exclude_makes_replicate_draws_disjoint(toy):
    subs, days = toy
    out = subs[:8]
    used, draws = [], []
    for r in range(3):
        c, _ = match_controls(days, out, subs, seed=2026 + r, n_candidates=6,
                              exclude=used)
        used += c
        draws.append(c)
    assert len(set(used)) == 24, "a subject was a control in two replicates"
    for c in draws:
        assert not set(c) & set(out)


def test_randomised_draw_still_matches_on_day_count():
    """Randomising the draw must not quietly become the random draw the matching exists
    to avoid: the candidates are the k nearest AVAILABLE, not a tolerance band, so a
    rare day count still gets its closest match.

    Note what is NOT asserted: that k=8 matches worse than k=1. The greedy draw is
    per-outlier optimal, not globally optimal, so leaving a subject for a later outlier
    can come out tighter overall. What must hold is that k=8 moves the draw and stays
    inside the tolerance.
    """
    rs = np.random.default_rng(3)
    subs = [f"S{i:03d}" for i in range(400)]
    days = {s: int(d) for s, d in zip(subs, rs.integers(30, 300, 400))}
    outliers = subs[:20]
    for s in outliers:                       # outliers have unusually few days
        days[s] = int(rs.integers(60, 110))

    c_rand, rand = match_controls(days, outliers, subs, seed=7, n_candidates=8)
    c_greedy, _ = match_controls(days, outliers, subs, n_candidates=1)
    c_plain = list(rs.choice(subs[20:], 20, replace=False))

    def median_day_gap(cs):
        return float(np.median([abs(days[c] - days[s]) for s, c in zip(outliers, cs)]))

    assert c_rand != c_greedy, "k=8 must actually move the draw"
    assert rand["max_rel_gap"] <= 0.15, "k=8 must still be a matched draw"
    assert median_day_gap(c_rand) < median_day_gap(c_plain)


def test_borderline_controls_are_recorded_not_removed(toy):
    """Dropping the subjects some detection seed flagged would leave a hand-picked
    'most normal' control set, which tightens the empirical null and makes separation
    easier to find. They are kept and tagged so the analysis can check them."""
    subs, days = toy
    out = subs[:8]
    d0 = build(subs, outliers=out, days=days)
    flagged = {c: 2 for c in d0.controls[:3]}
    d = build(subs, outliers=out, days=days, borderline=flagged)
    assert set(d.borderline) == set(flagged)
    assert set(d.borderline) <= set(d.controls), "tagged, and still drawn"
    assert "borderline_controls" in d.matching


def test_rejects_outliers_outside_the_cohort(toy):
    subs, days = toy
    with pytest.raises(ValueError, match="not in the cohort"):
        build(subs, outliers=["NOPE"], days=days)
