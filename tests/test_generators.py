"""Every registered generator must at least be reachable in this environment.

The point is narrow and worth having: a generator that cannot be imported is not a
generator you can choose at experiment time, and the failure otherwise surfaces hours
into a run. `dimts` is included deliberately -- it needs a second environment to RUN,
but importing it must not."""
import numpy as np
import pytest

from cgmoutlier.generators.base import GeneratorBase
from cgmoutlier.generators.registry import ALL, get


@pytest.mark.parametrize("name", ALL)
def test_resolves_to_a_generator_class(name):
    cls = get(name)
    assert issubclass(cls, GeneratorBase), name


def test_unknown_name_is_an_error():
    with pytest.raises(KeyError, match="unknown generator"):
        get("not_a_generator")


def test_dimts_load_accepts_a_run_that_trained_but_never_sampled(tmp_path):
    """What a walltime kill during the sampling loop leaves behind.

    `load` used to require all_samples.npy, so a directory holding ten checkpoints --
    100k steps of training, the expensive 60% of the run -- raised FileNotFoundError
    and looked like nothing. It is the input `resample` exists to consume."""
    run = tmp_path / "base"
    (run / "ckpt_288").mkdir(parents=True)
    (run / "ckpt_288" / "checkpoint-10.pt").write_bytes(b"")

    g = get("dimts")(T=288, C=1).load(str(run))
    assert g.fitted
    with pytest.raises(RuntimeError, match="resample"):
        g.sample(1)                       # nothing to return YET, and it says so


def test_dimts_load_refuses_a_directory_with_neither(tmp_path):
    run = tmp_path / "base"
    run.mkdir()
    with pytest.raises(FileNotFoundError, match="nothing here to restore"):
        get("dimts")(T=288, C=1).load(str(run))


def test_dimts_resample_refuses_a_run_with_no_checkpoint(tmp_path):
    """The error must name the missing weights, not fail later inside the subprocess
    on an env check that has nothing to do with it."""
    run = tmp_path / "base"
    run.mkdir()
    np.save(run / "train.npy", np.zeros((4, 288, 1), np.float32))
    (run / "config.json").write_text('{"hidden_size": 96}')

    g = get("dimts")(T=288, C=1, params={"python": "/nonexistent"})
    g._preflight = lambda: None                       # the env is not what is on trial
    with pytest.raises(FileNotFoundError, match="no trained weights"):
        g.resample(4, from_dir=run)


def test_dimts_resample_refuses_a_run_with_no_config(tmp_path):
    """Architecture comes from the trained run, never from the caller: a guessed
    hidden_size surfaces as an opaque state_dict shape error hours later."""
    run = tmp_path / "base"
    (run / "ckpt_288").mkdir(parents=True)
    (run / "ckpt_288" / "checkpoint-10.pt").write_bytes(b"")
    np.save(run / "train.npy", np.zeros((4, 288, 1), np.float32))

    g = get("dimts")(T=288, C=1, params={"python": "/nonexistent"})
    g._preflight = lambda: None
    with pytest.raises(FileNotFoundError, match="architecture is unknown"):
        g.resample(4, from_dir=run)


def test_copy_paste_reproduces_its_training_data():
    """The leakage upper bound. If this control does not return training rows verbatim
    it cannot anchor the top of the attack's scale."""
    X = np.random.default_rng(0).standard_normal((200, 288, 1)).astype(np.float32)
    g = get("copy_paste")(T=288, C=1).fit(X)
    s = g.sample(50)
    assert s.shape == (50, 288, 1)
    rows = {r.tobytes() for r in X}
    assert all(r.tobytes() in rows for r in s), "samples are not verbatim training rows"


def test_dimts_imports_without_its_environment():
    cls = get("dimts")
    g = cls(T=288, C=1, params={})
    with pytest.raises(RuntimeError, match="needs its own environment"):
        g.fit(np.zeros((4, 288, 1), np.float32))
