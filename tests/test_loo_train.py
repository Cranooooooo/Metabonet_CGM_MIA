"""Stage 4 bookkeeping. Exact tests: none of this involves a model.

What is checked is the property the whole design rests on -- that `include_s` and
`base` differ by exactly s's windows and by nothing else, including order.
"""
import json

import numpy as np
import pytest

from cgmoutlier.loo.train import _job_seed, training_set


@pytest.fixture
def cohort():
    """Three subjects with 2, 3 and 4 windows, values that identify their owner."""
    sids = np.array(["a"] * 2 + ["b"] * 3 + ["c"] * 4)
    X = np.arange(len(sids) * 4, dtype=np.float32).reshape(len(sids), 4, 1)
    return X, sids


def test_training_set_selects_exactly_the_named_subjects(cohort):
    X, sids = cohort
    Xtr, s = training_set(X, sids, ["a", "c"])
    assert Xtr.shape == (6, 4, 1)
    assert list(s) == ["a", "a", "c", "c", "c", "c"]


def test_training_set_is_cohort_ordered_not_argument_ordered(cohort):
    """The include/base pair must not differ in row order, only in membership."""
    X, sids = cohort
    a, _ = training_set(X, sids, ["c", "a"])
    b, _ = training_set(X, sids, ["a", "c"])
    assert np.array_equal(a, b)


def test_include_is_base_plus_exactly_the_target(cohort):
    X, sids = cohort
    base, _ = training_set(X, sids, ["a", "b"])
    incl, _ = training_set(X, sids, ["a", "b", "c"])
    assert len(incl) == len(base) + 4
    # every base row survives in the include set, unchanged
    assert np.array_equal(incl[np.isin(np.arange(len(incl)), np.arange(len(base)))],
                          base)


def test_unknown_subject_is_an_error_not_a_silent_smaller_training_set(cohort):
    X, sids = cohort
    with pytest.raises(ValueError, match="not in the cohort"):
        training_set(X, sids, ["a", "zzz"])


def test_job_seed_is_stable_and_distinct():
    assert _job_seed(2026, "base") == _job_seed(2026, "base")
    names = ["base", "include_569", "exclude_569", "include_602"]
    assert len({_job_seed(2026, n) for n in names}) == len(names)
    # a different base seed moves every job
    assert _job_seed(7, "base") != _job_seed(2026, "base")


def test_run_writes_a_readable_sample_file(tmp_path, monkeypatch, cohort):
    """End to end on the cheapest generator.

    The earlier version of this suite only tested the skip path, so it passed while
    `run` wrote nothing: np.save appends '.npy' to a path that lacks it, so the
    temporary file landed as 'samples.npy.tmp.npy' and the rename failed on every
    job. A test that asserts the artefact exists is what catches that.
    """
    from cgmoutlier.loo import train as T

    X, sids = cohort
    monkeypatch.setattr(T, "load_cohort", lambda *a, **k: (X, sids, dict(n_windows=len(X))))
    job = tmp_path / "base.json"
    job.write_text(json.dumps(dict(name="base", role="base", target=None,
                                   subjects=["a", "b"], n_subjects=2)))

    out = T.run(job, "unused", tmp_path / "runs", generator="copy_paste",
                device="cpu", verbose=False)
    S = np.load(out / "samples.npy")
    assert S.shape == (5, 4, 1)                       # K defaults to the training size
    assert not list(out.glob("*.tmp*")), "temporary file left behind"

    meta = json.loads((out / "meta.json").read_text())
    assert meta["K"] == 5 and meta["n_train_windows"] == 5
    assert meta["generator"] == "copy_paste"


def test_run_skips_a_finished_job(tmp_path, cohort):
    """A requeued job must cost only what it had not done (49 jobs, 1 slot/queue)."""
    import hashlib
    from cgmoutlier.loo import train as T

    out = tmp_path / "runs"
    (out / "base").mkdir(parents=True)
    np.save(out / "base" / "samples.npy", np.zeros((1, 4, 1), np.float32))
    # a genuinely finished job writes meta.json too. This fixture used to create only
    # samples.npy, which is the state a job KILLED between the two writes leaves behind
    # -- run() now refuses that rather than adopting a sample file it cannot attribute
    # to any design, so the fixture has to describe a real completion.
    fp = hashlib.sha1("a".encode()).hexdigest()[:16]
    (out / "base" / "meta.json").write_text(json.dumps(
        dict(job="base", subjects_sha1=fp, n_subjects=1)))
    job = tmp_path / "base.json"
    job.write_text(json.dumps(dict(name="base", role="base", target=None,
                                   subjects=["a"], n_subjects=1)))

    called = []
    T.load_cohort = lambda *a, **k: called.append(1)      # must never be reached
    assert T.run(job, "unused", out, verbose=False) == out / "base"
    assert not called


def test_run_refuses_samples_with_no_meta(tmp_path, cohort):
    """samples.npy with no meta.json cannot be attributed to a design, so it is not 'done'.

    This is the state a walltime kill or an OOM leaves: run() renames samples.npy into
    place and writes meta.json after it. The old guard read `m.get(...)` off an empty
    dict, found None on both branches, fell through and printed "already done, skipping".
    """
    import pytest
    from cgmoutlier.loo import train as T

    out = tmp_path / "runs"
    (out / "base").mkdir(parents=True)
    np.save(out / "base" / "samples.npy", np.zeros((1, 4, 1), np.float32))
    job = tmp_path / "base.json"
    job.write_text(json.dumps(dict(name="base", role="base", target=None,
                                   subjects=["a"], n_subjects=1)))
    with pytest.raises(ValueError, match="no way to tell which design"):
        T.run(job, "unused", out, verbose=False)


class _Restorable:
    """A generator that can resample, standing in for DiM-TS's subprocess."""

    seen = []

    def __init__(self, T, C, params=None, device="cpu", seed=0):
        self.T, self.C, self.params = T, C, dict(params or {})

    def resample(self, n, *, from_dir=None, out_dir=None, milestone=None,
                 sampling_timesteps=None):
        _Restorable.seen.append(dict(n=n, from_dir=str(from_dir),
                                     out_dir=str(out_dir), milestone=milestone,
                                     sampling_timesteps=sampling_timesteps))
        return np.zeros((n, self.T, self.C), np.float32)


def _trained_run(tmp_path, X, name="run"):
    """A run directory as a killed sampling pass leaves it: weights, no samples."""
    d = tmp_path / name
    (d / "ckpt_4").mkdir(parents=True)
    np.save(d / "train.npy", X)
    (d / "config.json").write_text(json.dumps(dict(hidden_size=96)))
    return d


@pytest.fixture
def resample_env(tmp_path, monkeypatch, cohort):
    from cgmoutlier.loo import train as T

    X, sids = cohort
    monkeypatch.setattr(T, "load_cohort",
                        lambda *a, **k: (X, sids, dict(n_windows=len(X))))
    monkeypatch.setattr(T, "get_generator", lambda name: _Restorable)
    _Restorable.seen = []
    job = tmp_path / "base.json"
    job.write_text(json.dumps(dict(name="base", role="base", target=None,
                                   subjects=["a", "b"], n_subjects=2)))
    return T, X, sids, job


def test_resample_writes_samples_and_declares_the_draw(tmp_path, resample_env):
    """The rescue case: the run gets the sample file its training already paid for."""
    T, X, sids, job = resample_env
    Xtr, _ = T.training_set(X, sids, ["a", "b"])
    run = _trained_run(tmp_path, Xtr)

    out = T.resample(run, job, "unused", device="cpu", verbose=False)
    assert out == run
    assert np.load(run / "samples.npy").shape == (5, 4, 1)
    assert not list(run.glob("*.tmp*"))

    meta = json.loads((run / "meta.json").read_text())
    assert meta["K"] == 5 and meta["subjects_sha1"]
    # The point of the whole record: this directory must not read as one that was
    # never interrupted.
    assert meta["bit_reproducible"] is False
    assert meta["resampled_from"] == str(run)
    assert meta["sampling_timesteps"] == 500


def test_resample_refuses_checkpoints_trained_on_a_different_job(tmp_path, resample_env):
    """Every design has a `base` and a checkpoint records nothing about its design, so
    the run's own train.npy is re-derived and compared rather than trusted."""
    T, X, sids, job = resample_env
    other, _ = T.training_set(X, sids, ["a"])        # a DIFFERENT job's training set
    run = _trained_run(tmp_path, other)

    with pytest.raises(ValueError, match="not the training set"):
        T.resample(run, job, "unused", device="cpu", verbose=False)
    assert not (run / "samples.npy").exists()


def test_resample_to_a_new_directory_leaves_the_original_alone(tmp_path, resample_env):
    """A step-count change is a different released set, not a repair: the 500-step
    samples must survive it or there is nothing to compare against."""
    T, X, sids, job = resample_env
    Xtr, _ = T.training_set(X, sids, ["a", "b"])
    run = _trained_run(tmp_path, Xtr)
    np.save(run / "samples.npy", np.ones((5, 4, 1), np.float32))

    out = T.resample(run, job, "unused", out=tmp_path / "st50", device="cpu",
                     sampling_timesteps=50, verbose=False)
    assert out == tmp_path / "st50"
    assert np.load(run / "samples.npy").all(), "the original release was overwritten"
    assert json.loads((out / "meta.json").read_text())["sampling_timesteps"] == 50
    assert _Restorable.seen[-1]["sampling_timesteps"] == 50


def test_resample_skips_a_run_that_already_has_its_samples(tmp_path, resample_env):
    T, X, sids, job = resample_env
    Xtr, _ = T.training_set(X, sids, ["a", "b"])
    run = _trained_run(tmp_path, Xtr)
    np.save(run / "samples.npy", np.ones((5, 4, 1), np.float32))

    T.resample(run, job, "unused", device="cpu", verbose=False)
    assert not _Restorable.seen, "resampled a run that was already released"


def test_resample_refuses_a_generator_that_cannot_restore(tmp_path, monkeypatch,
                                                          resample_env):
    """copy_paste keeps no checkpoint, so 'resample' there would silently mean
    'retrain' -- a different model, released as if it were the same one."""
    T, X, sids, job = resample_env
    Xtr, _ = T.training_set(X, sids, ["a", "b"])
    run = _trained_run(tmp_path, Xtr)

    class _NoRestore:
        def __init__(self, T, C, params=None, device="cpu", seed=0):
            pass

    monkeypatch.setattr(T, "get_generator", lambda name: _NoRestore)
    with pytest.raises(TypeError, match="cannot resample"):
        T.resample(run, job, "unused", device="cpu", verbose=False)


def test_run_refuses_a_finished_job_from_another_design(tmp_path, monkeypatch, cohort):
    """Every design has a job called `base`, so a run directory left by one design
    would let another skip straight past it and hand the attack a model trained on a
    different set of subjects. A sample file records nothing about its design, so this
    is the only place it can be caught."""
    from cgmoutlier.loo import train as T

    X, sids = cohort
    monkeypatch.setattr(T, "load_cohort",
                        lambda *a, **k: (X, sids, dict(n_windows=len(X))))
    out = tmp_path / "runs"
    job = tmp_path / "base.json"
    job.write_text(json.dumps(dict(name="base", role="base", target=None,
                                   subjects=["a", "b"], n_subjects=2)))
    T.run(job, "unused", out, generator="copy_paste", device="cpu", verbose=False)

    other = tmp_path / "base_other.json"                  # same NAME, different design
    other.write_text(json.dumps(dict(name="base", role="base", target=None,
                                     subjects=["a"], n_subjects=1)))
    with pytest.raises(ValueError, match="DIFFERENT job with the same name"):
        T.run(other, "unused", out, generator="copy_paste", device="cpu", verbose=False)
