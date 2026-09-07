# Running the baseline campaign on a new machine

For someone picking up the Step 2 baseline matrix on hardware that is not the NSCC
cluster this was built on. `docs/SETUP.md` covers installing the package; this covers
running the experiments, and the traps that have already cost time here.

Read `docs/PAPER_PLAN.md` Step 2 for *why* the matrix has the shape it does. This
document assumes that and tells you how to execute it.

---

## 0. What a clone gives you, and what it does not

`git clone` is the whole transfer. About 190 MB, and it **includes the training data** —
each cohort ships as one lossless `cohort.npz` (float32, `npz/deflate`), and
`cgmoutlier.data.cohort.load` reads it directly:

```python
if (p / "windows.npy").exists():        # what build() writes locally
    ...
elif (p / "cohort.npz").exists():       # what the repository ships
    z = np.load(p / "cohort.npz", allow_pickle=True)
```

The docstring is explicit that this is compression and not quantisation, so **a run
against the npz is the same run**. There is no unpack step. The loader also validates
`n_windows` and `n_subjects` against the manifest, so a truncated transfer fails loudly.

| | size | |
|---|---:|---|
| `data/cohort/matrix_d1_c1` | 2.0 MB | 1 day, T=288, CGM only |
| `data/cohort/matrix_d1_c2` | 4.9 MB | 1 day, T=288, CGM + insulin |
| `data/cohort/matrix_d7_c1` | 13.1 MB | 7 days, T=2016, CGM only |
| `data/cohort/matrix_d7_c2` | 31.7 MB | 7 days, T=2016, CGM + insulin |
| `data/cohort/metabonet875` | 55.8 MB | the 875-subject pool the outlier list comes from |

**Not** in the clone: `data/raw/metabonet_public.parquet` (1.3 GB). You only need it to
cut *new* cohorts. Every experiment below runs without it.

Also not in the clone, by design: `logs/`, model weights (`results/**/*.pt`) and released
samples (`results/**/*.npy`). Scores, manifests and design files **are** committed —
a repository that ships the data but not what the data produced cannot be checked.

Check the source data's terms before moving it to another institution's hardware. That
is a governance question, not a technical one.

---

## 1. Environment

Do **not** copy the conda environment. It is 6.4 GB and prefix envs do not survive the
move. Rebuild:

```bash
conda create -p <prefix> python=3.10 -y && conda activate <prefix>
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cu126   # match YOUR CUDA
pip install -e .
export PYTHONNOUSERSITE=1
```

Three things that are not optional:

- **`PYTHONNOUSERSITE=1`.** A numpy in `~/.local/lib/python3.10/site-packages` takes
  priority over the conda env — `conda activate` does not override it — and scikit-learn
  compiled against numpy 1.x then fails with a message about dtype sizes that names
  neither package.
- **numpy stays below 2.0.** Pinned in `requirements.txt` deliberately; several vendored
  generators are compiled against the 1.x C ABI.
- **Install torch alone.** Pulling `torchvision` alongside it has silently replaced torch
  with a CPU build here, after which `torch.cuda.is_available()` is False and nothing says
  why.

---

## 2. Verify before you queue anything

```bash
make test     # 88 tests, ~100 s, CPU
make smoke    # whole pipeline on synthetic data, CPU, ~2 min
```

Then, **inside a GPU job**, confirm the GPU is real with a matmul and not just with the
flag — the default wheel trains happily on CPU and only the wall-clock tells you:

```python
import torch; assert torch.cuda.is_available()
a = torch.randn(4096, 4096, device="cuda"); (a @ a).sum().item()
print(torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).total_memory / 2**30)
```

Record that memory figure. **Every cost in this document was measured on 40 GB cards**,
and the seven-day cells sit right against that limit.

Finally, confirm the registry imports — this is where a fresh clone used to die:

```python
from cgmoutlier.generators.registry import ALL, get
for n in ALL: get(n)
```

---

## 3. The design, in one page

Four cells: `{1 day T=288, 7 days T=2016} × {CGM only, CGM + insulin}`.

Each cell is **27 models**: one `base` trained on 475 background subjects, plus 26
`include_<target>` models that each add exactly one target subject to that same
background. The 26 targets are 13 outliers and 13 length-matched ordinary subjects, so
no result can be explained by how much data a person contributed.

The design is fixed and committed at `results/matrix/design/rep1/` — 27 job files. Do
not regenerate it. `build_design.py` defaults `--n-candidates` to 1 for a single
replicate and to 8 when `--replicates > 1`, so re-running it with `--replicates 3`
produces a **different** rep1 and orphans every completed run.

**Equal budget is the axis being measured.** All generators see **6,400,000 training
sequences**. The conversion lives in `scripts/pbs/dev/baseline_base.body.sh`:

| generator | key | one unit feeds |
|---|---|---|
| Diffusion-TS | `max_epochs` — *iterations*, despite the name | `batch_size` |
| DiffWave | `total_iters` | `batch_size` |
| FourierDiffusion | `max_epochs` — a *real* epoch | 5741 (the whole set) |
| IG-FM | `steps` | 256 (8 accum × 32 micro) |
| DiM-TS | `steps` | 64 |

Getting `max_epochs` backwards between Diffusion-TS and FourierDiffusion is a factor of
about **90**. Diffusion-TS additionally defaults `gradient_accumulate_every=2`, so a
"step" feeds `2 × batch_size` unless you set it to 1 — the budget helper does.

---

## 4. Train one base, read the gate, and stop if it fails

Never start 27 models on an untested generator. One base first:

```bash
python -u scripts/run_loo.py \
    --design results/matrix/design/rep1 \
    --cohort data/cohort/matrix_d1_c1 \
    --out    results/runs/bl_<gen>_d1_c1 \
    --generator <gen> --job base --seed 2026 \
    --params '{"max_epochs":100000,"batch_size":64,"gradient_accumulate_every":1,"timesteps":500}'

python -u scripts/eval_quality_tsgem.py \
    --runs results/runs/bl_<gen>_d1_c1/base \
    --cohort data/cohort/matrix_d1_c1 \
    --design results/matrix/design/rep1 \
    --out results/bl_<gen>_d1_c1/tsgem.json --label bl_<gen>_d1_c1
```

The gate is registered in `configs/experiment.yaml` and is **relative**: no worse than
DiM-TS `base` in the same cell, on `context_fid` and on `discriminative` read as the
*max over 8 restarts* (`scripts/disc_stability.py` says why a single fit is not enough).

A generator whose base fails the gate **stops there and that is a result** — "this
architecture cannot produce usable CGM data at this scale" — not a gap to be filled by
spending 27 models on it. TimeVAE was ruled out this way in minutes (Context-FID 0.856).

Read the score from the parsed number, not from the presence of the key:
`eval_quality_tsgem` can write `"context_fid": null`, and a `grep -q` on the key
succeeds on it, reporting a failed measurement as a pass.

---

## 5. The full 27 models

Shard across whatever cards you have; each model trains on one GPU:

```bash
python -u scripts/run_loo.py --design results/matrix/design/rep1 \
    --cohort data/cohort/<cell> --out results/runs/<run> \
    --generator <gen> --params '<same params as the base>' \
    --shard $i --n-shards $N --seed 2026
```

`--list` prints the slice without running it. `run_loo` skips a job whose
`samples.npy` already exists, so a resubmit is idempotent at model granularity.

**Every one of the 27 must use identical parameters.** §3b measured a large effect of
training length on the membership gap, so if `base` and `include_t` ran to different
lengths the gap would confound membership with training length — the one confound the
whole design exists to exclude. The PBS bodies here fingerprint the params and refuse
to start a run that drifted from the cell's other models; keep that guard if you port
them.

Then the analysis, in this order:

```bash
python scripts/run_attack.py   ...   # the frozen min×mean statistic
python scripts/subject_auc.py  ...   # per-subject Mann-Whitney AUC
python scripts/leak_locus.py   ...   # the shuffle floor this release is judged against
```

A release is only "clean" when it reads what **its own** shuffled floor reads. Absolute
AUC near 0.5 is not the test; the floor is 0.50–0.52 depending on the cell.

---

## 6. Traps, each of which has already cost time here

1. **`--params` reaches the constructor, not `fit()`.** `run_loo` calls `gen.fit(Xtr)`
   with no `train_cfg`. An adapter that starts `cfg = dict(train_cfg or {})` silently
   discards the budget, trains on its own default, completes normally, and writes a
   `meta.json` recording the budget you asked for. Four adapters were in that state; if
   you add a fifth, start it `cfg = dict(self.params); cfg.update(train_cfg or {})`.
2. **`batch_size=64` OOMs at T=2016.** Measured: IG-FM needs micro-batch 8 (32 and 16
   both fail), Diffusion-TS needs 4. Smaller is not always slower — for IG-FM, 4 was no
   faster than 8, so it is compute-bound, not memory-bound, at that point.
3. **FourierDiffusion's `keep_best` callback allocates 46.5 GiB at T=2016.** It runs a
   forward pass on a hardcoded `dm.X_train[:512]` (`fourier_diff.py:158`). This is
   independent of `batch_size`, so it reads as "the model does not fit" at every batch
   size you try. It must stay **on** for real runs — it protects against a collapse that
   hit 35 folds — so the fix is to size that batch by T, not to disable it.
4. **The three new baseline adapters have no checkpoint/resume.** Their checkpoint path
   is read only from `train_cfg`, which `run_loo` does not pass. One model per job; two
   models in one job means a walltime kill on the first takes the second with it and you
   get zero partial credit.
5. **Bigger sampling batches can be slower.** Measured at T=2016: 800 windows per call
   took 969 s per 500 samples, 300 took 581 s. Memory-bandwidth bound. Do not assume.
6. **A cost probe reporting a negative per-step cost is measuring process warm-up**, not
   the model. The first `fit()` in a process pays CUDA context init and kernel autotuning;
   a two-point difference assumes both points pay it equally. Take the high point divided
   by its budget as a conservative upper bound instead.
7. **Sampling loops in the vendored code are `range(n // batch + 1)`** — they generate one
   batch too many and discard it. Ask for enough samples that the extra batch is a small
   fraction, or take two points and use the slope.
8. **Do not compute on the login node.** NSCC-specific, but the habit is worth keeping:
   route even a log scan or a page build through a zero-GPU queue.

---

## 7. What it costs

Per model, measured on 40 GB cards at the equal budget above.

| generator | d1_c1 | d1_c2 | d7_c1 | d7_c2 |
|---|---:|---:|---:|---:|
| DiM-TS | 4.59 h | 8.00 h | 22.93 h | 25.99 h |
| IG-FM (either arm) | 4.35 h | 4.38 h | 109.1 h | 109.4 h |
| Diffusion-TS (author default) | 0.59 h | 0.58 h | 18.0 h | ~18 h |
| Diffusion-TS (`solar`, capacity-matched) | 2.70 h | 2.73 h | not probed | not probed |
| DiffWave | 0.52 h | 0.52 h | ~3.7 h | ~3.7 h |
| FourierDiffusion | 1.48 h | 1.49 h | see trap 3 | see trap 3 |

Multiply by 27 for a full cell. The three cheap baselines are 14–40 GPU-hours per
one-day cell, which is why they are the sensible thing to run on new hardware first.

A capacity note worth carrying: Diffusion-TS's adapter default is `1/1/32` = 224k
parameters at T=288, and the authors' own eleven configs never go below `d_model=64` or
`n_layer_dec=2`. Every "Diffusion-TS is poor" result so far is at a capacity its authors
never used. Their `solar.yaml` (`4/4/96`) happens to be 1,999,513 parameters here, which
is 0.96× IG-FM — one config that is both the authors' own and capacity-matched. At
T=2016 the *default* is already 8.3M parameters, because the trend and Fourier components
scale with T, so the strawman problem exists only on the one-day cells.

---

## 8. If your scheduler is not PBS

`scripts/pbs/` is PBS-specific: `#PBS` directives, `-P <project>`, and a routing table
where the queue is chosen by `(ngpus, walltime)` rather than by a flag. What to carry
across when you translate:

- **Sharding.** The bodies split work with `CUDA_VISIBLE_DEVICES` inside one node. Across
  nodes you need `pbsdsh`/`srun`/MPI instead; the scripts cannot see another node.
- **The guards.** Parameter fingerprinting, the checkpoint-count check that verifies
  `iter-2000` exists *by name* (pruning is oldest-first, so "8 archives present" passes
  while the early ones a trajectory needs are gone), and the `fit_seconds` guard that
  marks a resumed run's timing as unusable for estimating a campaign.
- **`trap ... EXIT` tailing the log.** A job killed at the wall otherwise tells you
  nothing about where it was.

---

## 9. What to send back

Results are small. Per cell per generator: `results/<run>/tsgem.json`,
`results/matrix/attack/<cell>/summary.json`,
`results/matrix/subject_auc/<cell>/summary.json`, `results/matrix/leak_locus/<cell>.json`,
and the `meta.json` of each of the 27 models — that last one carries `fit_seconds` and
`sample_seconds`, which is how the cost table above was built and how the next estimate
will be.

Weights and samples do not need to travel. Send the `meta.json` files even for runs that
failed the quality gate; a failed gate is a result and its cost still informs planning.
