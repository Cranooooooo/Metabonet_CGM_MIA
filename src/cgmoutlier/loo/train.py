"""One design job -> one trained generator and one released synthetic set.

ONE PROCESS, ONE JOB. `run` trains exactly one model and writes exactly one sample
file. That is not a style choice: an earlier sampler in this project produced the full
model plus every fold in a single call, so running one copy per GPU had each copy redo
the same work -- three GPUs for thirteen hours produced two folds (docs/PITFALLS.md
§3). With 49 jobs on a scheduler that admits one job per queue above two hours, the
property that matters is that a job can be requeued without redoing a finished one.

WHAT IS HELD IDENTICAL ACROSS A PAIR
------------------------------------
The design's whole claim is that two models differ by one subject and nothing else, so
everything else is fixed here rather than left to the caller:

* the generator, its params and the training seed come from one config, not per job;
* K, the number of released samples, defaults to the job's own training-set size --
  an attack can only see a memorised window if it is actually released, so a fixed
  small K would lower measured leakage for a reason unrelated to membership;
* the sampling seed is derived from the job name, so two jobs never share a draw and
  a rerun of one job is exact whatever else has run.

K DIFFERS BETWEEN THE TWO ARMS BY DESIGN, AND THE ATTACK MUST UNDO IT
--------------------------------------------------------------------
`include_s` trains on n_s more windows than `base`, so with K = training-set size it
also releases n_s more samples -- about 0.1%. Nearest-neighbour distance falls with K,
so that alone biases d_IN downwards and the gap upwards. The bias is small and it is
real, and it is removed on the attack side by matching K before any distance is
computed (see `attack.statistic`), not here: the released set is what a custodian
would actually release, and shrinking it at source would be a different experiment.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np

from ..data.cohort import load as load_cohort
from ..generators.registry import get as get_generator


def _job_seed(base_seed: int, job_name: str) -> int:
    """A distinct, reproducible sampling seed per job.

    Derived from the name rather than from a counter so that adding a job, or running
    them in a different order, does not shift every other job's draw -- the same
    failure `outliers.common.method_rng` exists to prevent (docs/PITFALLS.md §5).
    """
    h = hashlib.sha1(f"{base_seed}|{job_name}".encode()).digest()
    return int.from_bytes(h[:4], "big")


def training_set(X, sids, subjects):
    """The windows belonging to `subjects`, in cohort order.

    Cohort order, not the order of `subjects`: two jobs whose subject lists differ by
    one element must otherwise produce the same array up to that element, and relying
    on the caller's ordering to achieve that is how a silent difference gets in.
    """
    want = set(map(str, subjects))
    keep = np.array([str(s) in want for s in sids])
    missing = want - {str(s) for s in np.asarray(sids)[keep]}
    if missing:
        raise ValueError(f"{len(missing)} job subjects are not in the cohort: "
                         f"{sorted(missing)[:5]}")
    return np.ascontiguousarray(np.asarray(X)[keep]), np.asarray(sids)[keep]


def resample(from_run, job_file, cohort, out=None, generator=None, params=None,
             K=None, sampling_timesteps=None, milestone=None, seed=2026,
             device="cuda", overwrite=False, verbose=True):
    """Release a new sample set from an already-trained run, without retraining.

    Two questions are answered here and neither is worth a training run:

    * a run whose training finished and whose sampling pass was killed -- sampling is
      about 40% of the bill at h=128, so the other 60% is already on disk;
    * how many denoising steps the released set needs (`sampling_timesteps` 500 -> 50),
      which changes the sample and not the model. Point `out` somewhere new for that
      one: it is a different released set from the same model, not a repair.

    WHAT IS CHECKED BEFORE ANY OF IT
    --------------------------------
    A checkpoint directory carries no record of which design trained it, and every
    design has a job called `base`. So the run's own `train.npy` is compared against
    the training set this job names, element for element. That is stronger than the
    fingerprint `run` writes -- it re-derives the array rather than trusting a hash
    someone else recorded -- and it is the only thing standing between a rescued
    sample set and one drawn from the wrong model.

    WHAT IS DECLARED AFTERWARDS
    ---------------------------
    The draw is not the one an uninterrupted run would have made: training advances
    the RNG before sampling starts. Same weights, same distribution, different draw,
    so nothing downstream is biased and nothing is bit-reproducible from the original
    command line either. `meta.json` records that rather than leaving the directory
    indistinguishable from one that was never interrupted.
    """
    from_run = Path(from_run)
    out = Path(out) if out else from_run
    job = json.loads(Path(job_file).read_text())
    done = out / "samples.npy"
    fp = hashlib.sha1("\n".join(map(str, job["subjects"])).encode()).hexdigest()[:16]

    if done.exists() and not overwrite:
        if verbose:
            print(f"[loo] {job['name']}: already has {done}, skipping "
                  f"(--overwrite to replace)")
        return out

    X, sids, man = load_cohort(cohort)
    Xtr, _ = training_set(X, sids, job["subjects"])
    N, T, C = Xtr.shape
    k = int(K) if K else N

    saved = from_run / "train.npy"
    if not saved.exists():
        raise FileNotFoundError(
            f"{saved} is missing; without it there is no way to confirm the "
            f"checkpoints under {from_run} were trained on this job")
    Xsaved = np.load(saved)
    if Xsaved.shape != Xtr.shape or not np.array_equal(Xsaved, Xtr):
        raise ValueError(
            f"{saved} is not the training set of {job_file}: "
            f"{Xsaved.shape} vs {Xtr.shape}. These checkpoints belong to a different "
            f"job or a different design; resampling them would release a model "
            f"trained on the wrong subjects.")
    del Xsaved

    prev = from_run / "meta.json"
    m = json.loads(prev.read_text()) if prev.exists() else {}
    generator = generator or m.get("generator") or "dimts"
    p = dict(m.get("params") or {})
    p.update(params or {})
    p["workdir"] = str(from_run)
    job_seed = _job_seed(seed, job["name"])
    gen = get_generator(generator)(T=T, C=C, params=p, device=device, seed=job_seed)
    if not hasattr(gen, "resample"):
        raise TypeError(f"{generator} cannot resample: it has no checkpoint to "
                        f"restore, so a new sample set costs a full training run")

    if verbose:
        print(f"[loo] {job['name']}: resampling K={k:,} from {from_run} "
              f"(milestone={milestone or 'latest'}, "
              f"sampling_timesteps={sampling_timesteps or 500}) -> {out}", flush=True)

    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    S = np.asarray(gen.resample(k, from_dir=from_run, out_dir=out,
                                milestone=milestone,
                                sampling_timesteps=sampling_timesteps),
                   dtype=np.float32)
    t_sample = time.time() - t0

    if S.shape != (k, T, C):
        raise ValueError(f"generator returned {S.shape}, expected {(k, T, C)}")
    nan = float(np.isnan(S).mean())
    if nan:
        raise ValueError(f"{nan:.2%} of the released samples are NaN; refusing to "
                         f"write a sample file the attack would read as data")

    tmp = out / "samples.tmp.npy"
    np.save(tmp, S)
    tmp.rename(done)

    meta = dict(m)
    meta.update(job=job["name"], role=job["role"], target=job["target"],
                group=job.get("group"), job_file=str(job_file), subjects_sha1=fp,
                n_subjects=job["n_subjects"], n_train_windows=int(N), K=int(k),
                T=int(T), C=int(C), generator=generator, params=p,
                base_seed=int(seed), job_seed=int(job_seed),
                cohort=str(cohort), cohort_windows=int(man["n_windows"]),
                sample_seconds=round(t_sample, 2),
                sample_range=[float(S.min()), float(S.max())],
                resampled_from=str(from_run),
                resample_milestone=(int(milestone) if milestone is not None
                                    else "latest"),
                sampling_timesteps=int(sampling_timesteps or 500),
                bit_reproducible=False,
                resample_note=(
                    "Drawn from a restored checkpoint. Training advances the RNG "
                    "before sampling, so this is not the draw an uninterrupted run "
                    "would have made -- same weights and same distribution, "
                    "different draw. fit_seconds, if present, belongs to the "
                    "original training run."))
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    if verbose:
        print(f"[loo] {job['name']}: sample {t_sample/60:.1f} min -> {done}", flush=True)
    return out


def run(job_file, cohort, out, generator="copy_paste", params=None, K=None,
        seed=2026, device="cuda", overwrite=False, verbose=True):
    """Train one model, release one sample set. -> the directory it wrote."""
    job = json.loads(Path(job_file).read_text())
    out = Path(out) / job["name"]
    done = out / "samples.npy"
    fp = hashlib.sha1("\n".join(map(str, job["subjects"])).encode()).hexdigest()[:16]
    if done.exists() and not overwrite:
        # "Already done" must mean done FOR THIS JOB. Job names repeat across designs --
        # every design has a `base` -- so pointing a new design at a directory an older
        # one wrote would skip straight past it and hand the attack a model trained on a
        # different set of subjects. Nothing downstream could see that: a sample file
        # records nothing about the design that produced it, which is exactly why
        # designs are never rebuilt in place either.
        prev = out / "meta.json"
        m = json.loads(prev.read_text()) if prev.exists() else {}
        was, want = m.get("subjects_sha1"), fp
        if was is None:                       # written before fingerprints existed
            was, want = m.get("n_subjects"), job["n_subjects"]
        if was is not None and was != want:
            raise ValueError(
                f"{done} was written by a DIFFERENT job with the same name "
                f"({was} != {want}). This directory belongs to another design; point "
                f"--out somewhere else rather than mixing two designs in one tree.")
        if verbose:
            print(f"[loo] {job['name']}: already done, skipping ({done})")
        return out

    X, sids, man = load_cohort(cohort)
    Xtr, str_ = training_set(X, sids, job["subjects"])
    N, T, C = Xtr.shape
    k = int(K) if K else N
    job_seed = _job_seed(seed, job["name"])

    if verbose:
        print(f"[loo] {job['name']}: role={job['role']} target={job['target']} "
              f"{job['n_subjects']} subjects, {N:,} windows, K={k:,}, "
              f"generator={generator}, seed={job_seed}", flush=True)

    p = dict(params or {})
    p.setdefault("n_samples", k)
    # Checkpoints must outlive the process. DiM-TS trains and samples in one subprocess
    # call and defaults its work directory to tempfile.mkdtemp(), so without this the
    # trained weights are discarded the moment the job ends -- and any later question
    # that is answered by re-sampling rather than retraining (how many denoising steps
    # the released set needs, what a different K does) would cost the whole 164 GPU-hour
    # training run again. Adapters that do not take `workdir` ignore it.
    p.setdefault("workdir", str(out))
    gen = get_generator(generator)(T=T, C=C, params=p, device=device, seed=job_seed)

    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    gen.fit(Xtr)
    t_fit = time.time() - t0

    t0 = time.time()
    S = np.asarray(gen.sample(k), dtype=np.float32)
    t_sample = time.time() - t0

    if S.shape != (k, T, C):
        raise ValueError(f"generator returned {S.shape}, expected {(k, T, C)}")
    nan = float(np.isnan(S).mean())
    if nan:
        raise ValueError(f"{nan:.2%} of the released samples are NaN; refusing to "
                         f"write a sample file the attack would read as data")

    # The suffix must stay .npy: np.save appends ".npy" to any path that lacks it, so
    # "samples.npy.tmp" is written as "samples.npy.tmp.npy" and the rename below then
    # fails on a file that was never created.
    tmp = out / "samples.tmp.npy"
    np.save(tmp, S)
    tmp.rename(done)                                   # atomic: no half-written set

    meta = dict(job=job["name"], role=job["role"], target=job["target"],
                group=job.get("group"), job_file=str(job_file), subjects_sha1=fp,
                n_subjects=job["n_subjects"], n_train_windows=int(N), K=int(k),
                T=int(T), C=int(C), generator=generator, params=p,
                base_seed=int(seed), job_seed=int(job_seed),
                cohort=str(cohort), cohort_windows=int(man["n_windows"]),
                fit_seconds=round(t_fit, 2), sample_seconds=round(t_sample, 2),
                sample_range=[float(S.min()), float(S.max())])
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    if verbose:
        print(f"[loo] {job['name']}: fit {t_fit/60:.1f} min, sample {t_sample/60:.1f} "
              f"min -> {done}", flush=True)
    return out
