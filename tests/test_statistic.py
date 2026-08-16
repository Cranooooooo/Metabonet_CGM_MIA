"""d_OUT - d_IN.

The distances are checked against brute force rather than against remembered numbers,
and the sign convention is pinned on a copy-paste-shaped case: a released set that
literally contains the target's windows must give a positive gap. That direction is
the one thing in the pipeline it is easiest to get backwards and hardest to notice.
"""
import numpy as np
import pytest

from cgmoutlier.attack.statistic import (_match, gap_for_pair, reduce_subject,
                                         summarise, window_distances)


def brute(R, S, how):
    R, S = R.reshape(len(R), -1), S.reshape(len(S), -1)
    D = np.linalg.norm(R[:, None, :] - S[None, :, :], axis=2) / np.sqrt(R.shape[1])
    return D.min(1) if how == "min" else D.mean(1)


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.mark.parametrize("how", ["min", "mean"])
def test_window_distances_match_brute_force(rng, how):
    R = rng.standard_normal((7, 12, 1)).astype(np.float32)
    S = rng.standard_normal((40, 12, 1)).astype(np.float32)
    got = window_distances(R, S, set_reduce=how, chunk=8)   # chunk < len(S) on purpose
    assert np.allclose(got, brute(R, S, how), atol=1e-4)


def test_chunking_does_not_change_the_answer(rng):
    R = rng.standard_normal((5, 12, 1)).astype(np.float32)
    S = rng.standard_normal((37, 12, 1)).astype(np.float32)
    a = window_distances(R, S, set_reduce="min", chunk=1)
    b = window_distances(R, S, set_reduce="min", chunk=10_000)
    assert np.allclose(a, b)


def test_reduce_subject():
    d = np.array([1.0, 2.0, 3.0, 10.0])
    assert reduce_subject(d, "min") == 1.0
    assert reduce_subject(d, "mean") == 4.0
    assert reduce_subject(d, "q10") == pytest.approx(np.quantile(d, 0.1))
    with pytest.raises(ValueError):
        reduce_subject(d, "median")


def test_copy_paste_shape_gives_a_positive_gap(rng):
    """The member model released the target's own windows; the non-member did not."""
    R = rng.standard_normal((6, 12, 1)).astype(np.float32)
    other = rng.standard_normal((50, 12, 1)).astype(np.float32)
    S_member = np.concatenate([other, R])          # contains the target verbatim
    S_nonmember = other

    rows = gap_for_pair(R, S_member, S_nonmember, target="t", match_k=False)
    assert rows, "no variants computed"
    for r in rows:
        assert r["gap"] > 0, r
        if r["set_reduce"] == "min":
            # not exactly 0: the expanded form r2 + s2 - 2 r.s cancels to ~1e-9 in
            # float32 for an identical pair, and the sqrt turns that into ~5e-5.
            # See the note in statistic.window_distances -- it is a floor on d, four
            # orders of magnitude below the distances the study compares.
            assert r["d_in"] == pytest.approx(0.0, abs=1e-3)


def test_gap_is_zero_when_both_models_released_the_same_set(rng):
    R = rng.standard_normal((6, 12, 1)).astype(np.float32)
    S = rng.standard_normal((30, 12, 1)).astype(np.float32)
    for r in gap_for_pair(R, S, S, target="t", match_k=False):
        assert r["gap"] == pytest.approx(0.0, abs=1e-6)


def test_match_k_cuts_both_sets_to_one_size_and_is_reproducible(rng):
    A = rng.standard_normal((100, 4, 1)).astype(np.float32)
    B = rng.standard_normal((80, 4, 1)).astype(np.float32)
    a1, b1, k = _match(A, B, "t", 2026, True)
    a2, b2, _ = _match(A, B, "t", 2026, True)
    assert k == 80 and len(a1) == len(b1) == 80
    assert np.array_equal(a1, a2) and np.array_equal(b1, b2)
    # a different target draws a different subsample
    a3, _, _ = _match(A, B, "u", 2026, True)
    assert not np.array_equal(a1, a3)


def test_match_k_removes_the_larger_released_set_advantage(rng):
    """More released samples lowers the nearest-neighbour distance on its own.

    Both models here are the same distribution and neither saw the target, so the only
    difference is that one released more. Unmatched, that alone produces a positive
    gap; matched, it does not.
    """
    R = rng.standard_normal((20, 16, 1)).astype(np.float32)
    S_small = rng.standard_normal((200, 16, 1)).astype(np.float32)
    S_big = rng.standard_normal((4000, 16, 1)).astype(np.float32)

    unmatched = [r for r in gap_for_pair(R, S_big, S_small, target="t", match_k=False)
                 if r["set_reduce"] == "min" and r["subject_reduce"] == "mean"][0]
    matched = [r for r in gap_for_pair(R, S_big, S_small, target="t", match_k=True)
               if r["set_reduce"] == "min" and r["subject_reduce"] == "mean"][0]

    assert unmatched["gap"] > 0                       # the artefact this exists to kill
    assert abs(matched["gap"]) < unmatched["gap"]
    assert matched["k_matched"] == 200


def test_summarise_reports_separation():
    rows = ([dict(target=f"o{i}", group="outlier", set_reduce="min",
                  subject_reduce="mean", d_in=0, d_out=1, gap=1.0 + i)
             for i in range(8)] +
            [dict(target=f"c{i}", group="control", set_reduce="min",
                  subject_reduce="mean", d_in=0, d_out=0, gap=-1.0 - i)
             for i in range(8)])
    s = summarise(rows).iloc[0]
    assert s["auc"] == 1.0                       # every outlier gap beats every control
    assert s["p_between"] < 0.05
    assert s["p_within_outlier"] < 0.05          # outlier gaps are positive
    assert np.isnan(s["p_within_control"]) or s["p_within_control"] > 0.5


def test_summarise_separates_the_two_questions():
    """Both arms leaking equally must read as 'detected, not separated'.

    This is copy_paste: an outlier's pair is (include_s, base) and a control's is
    (base, exclude_c), so a generator that memorises everything makes both arms
    strongly positive and neither arm larger than the other.
    """
    rows = ([dict(target=f"o{i}", group="outlier", set_reduce="min",
                  subject_reduce="q10", d_in=0.0, d_out=0.07, gap=0.07 + 0.001 * i)
             for i in range(8)] +
            [dict(target=f"c{i}", group="control", set_reduce="min",
                  subject_reduce="q10", d_in=0.0, d_out=0.07, gap=0.07 + 0.001 * i)
             for i in range(8)])
    s = summarise(rows).iloc[0]
    assert s["p_within_outlier"] < 0.05 and s["p_within_control"] < 0.05
    assert s["auc"] == pytest.approx(0.5, abs=0.05)
