# Traps

Things that have already cost this project days of compute or produced a number nobody
could reproduce. Each is enforced in code where that is possible, documented here where
it is not.

## 1. `PYTHONNOUSERSITE=1`

A numpy in `~/.local/lib/python3.10/site-packages` (2.2.6) shadows the environment's
1.24.4. Python places user site ahead of the environment on `sys.path`, so
`conda activate` does not protect you. scikit-learn 1.1.2, compiled against the numpy
1.x C ABI, then dies with

```
ValueError: numpy.dtype size changed, may indicate binary incompatibility.
            Expected 96 from C header, got 88 from PyObject
```

which names neither numpy nor the shadowing. `src/cgmoutlier/_env.py` checks for this
and refuses to start; `make` exports the variable. If you invoke a module directly,
export it yourself.

```bash
export PYTHONNOUSERSITE=1
```

## 2. numba ignores `OMP_NUM_THREADS`

Its parallel pool is sized from `NUMBA_NUM_THREADS`, which defaults to the machine's
core count. On a 128-core shared box, twelve workers with `OMP_NUM_THREADS=3` set took
**50 cores and pushed the load average past 270**, because each had spawned 138
threads. The module that owns the parallel kernel calls `set_num_threads` at import,
so it applies however the process was started; set the variable as well when you drive
it yourself.

```bash
export NUMBA_NUM_THREADS=4
```

## 3. One process, one unit of work

An earlier sampler produced the full model plus every fold in a single call. Running
one copy per GPU therefore had each copy redo the same nine sample sets — three GPUs
for thirteen hours produced two folds. Every driver here takes exactly one unit of
work per process. If you add one, keep that property.

## 4. Never edit a running bash driver

Bash reads a script incrementally, so editing one mid-run corrupts it from the edit
point on. A watcher died this way at stage 4 of 6. All drivers here are Python and all
tuning lives in `configs/`.

## 5. One shared RNG makes every score depend on the run's composition

The previous implementation held a single module-level generator:

```python
RNG = np.random.default_rng(2026)          # consumed in code order by every method
```

Each method drew its reference pool from it in turn, so a method's numbers depended on
which *earlier* methods were selected in the same process. Measured on the 875-subject
cohort: rerunning `--only C8` against the archived C8 from the full run gives Spearman
**0.9981** and 42 of 44 subjects in common at the 5% cut — same code, same data, same
seed, different answer, because the full run had drawn for B7 first. Small differences,
compounded across fourteen methods, moved the consensus list by four subjects.

`common.method_rng(key, seed)` derives an independent stream from the method's own name,
so one method's rerun is exact whatever else the run contains. `tests/test_methods.py`
pins this.

The subsampling itself is unchanged and still matters — B5/B6 see 30 days per subject,
not all of them. That is now a *reproducible* approximation rather than an
irreproducible one, and how much of the outlier list survives a change of base seed is
measured rather than assumed:

```bash
python scripts/seed_stability.py --seeds 2026,7,101,999
```

## 6. The consensus denominator moves silently

`consensus()` reads whatever score files are on disk. A run that died after eleven
methods, or one that included the optional `D12`, quietly turns "at least 7 of 13" into
"7 of 11" or "7 of 14" — the same threshold against a different denominator, which is a
different claim, with no error raised. It now takes an `expect=` count and refuses to
proceed when the candidate set is not the size you said. `D12` is implemented but kept
out of the default `ALL` for the same reason.

## 7. ⛔ No `python` on the login node — not even the cheap kind

`~/NSCC_operation_guide_book.md` §1: no compute on the login node, and it names
`python` explicitly alongside `pip install` and `nohup`. NSCC **auto-flags it and
raises a support ticket against the project**, and the guide records that the detector
fired on *one CPU featurization run*.

On 2026-08-12 this repo's own agent ran, on the login node: a 54-second 1.6 GB feature
extraction (a CPU featurization run — the exact documented trigger), the full pytest
suite twice, and a dozen short analysis snippets over `gaps.parquet` and
`results/quality/`. `scripts/pbs/env.sh` carries the warning in its first three lines.
Reading the rule is not the same as having a place to put the work.

**The failure mode is that the rule sounds scoped to heavy jobs and is not.** A
three-line `python -c` that loads a parquet is still `python` on a shared login node,
and the difference between it and the run that trips the detector is not visible from
the prompt.

**How to comply without friction:** every python invocation goes in a `qdev` CPU job —
zero GPUs and walltime ≤ 2 h routes there at priority 100 with 100 concurrent jobs, so
it starts in about a minute. `scripts/pbs/30_attack_panel_table.pbs` exists to be that
place: it runs pytest and the analysis together, so "this is too small to be worth a
job" never has to be decided. The login shell is for `qsub`, `qstat`, `tail`, `cat`,
`grep` and `bash -n`.

## 8. A CPU job with no `mem=` gets the queue default and is OOM-killed

Three `qdev` jobs died as `cgroup/OOM: Killed because of memory limit` on 2026-08-12
against a measured 1.6 GB peak, because the submit asked for `select=1:ncpus=32` and
nothing else. `qstat` showed `10mb` in the Memory column before they ran; it was there
to be read.

**Why the habit does not transfer from the GPU lanes.** PBS pins ~110 gb per card on
the `g*` queues whatever the request says (`docs/PLAN.md`, queue plan), so every GPU
recipe in this repo omits `mem` and has never needed it. Carrying that shape to a CPU
queue silently drops the allocation by four orders of magnitude.

Ask for a measured figure, not a round one: the feature pass blocks its distance
matrices at 128 rows precisely so peak memory does not scale with the subject's window
count, which is what makes `mem=16gb` an honest ask rather than a guess.

## 9. An advancing log mtime proves liveness; a static one proves nothing

`~/NSCC_operation_guide_book.md` §7 gives the right rule — verify a run by a log whose
mtime is advancing, never by `pgrep` or `nvidia-smi` — but it has a precondition the
rule does not state: **the program has to print.**

On 2026-08-12 three generator base runs sat with logs untouched for an hour while
`diffusion_ts` wrote 3.7 MB of tqdm output beside them. They were not hung. PyTorch
Lightning and several of these adapters disable their progress bar when stdout is not a
terminal, so a healthy training loop writes nothing for hours. The 2026-08-10
`fourier_diff` failure had the same signature — log stopping seconds after the Lightning
dataloader warning — and was read as a hang on the strength of the same mistake.

**The liveness check that does not depend on the program's behaviour** is the
scheduler's own accounting:

```bash
qstat -f <jobid> | grep -aE 'resources_used.(cput|cpupercent|mem)'
```

`cput` advancing against `walltime` is work being done. Read `cpupercent` against the
job's shape before concluding anything from its size: a 3-task GPU job showed 2294%
(~23 cores, ~7.6 per task) and a 1-GPU job with `num_workers=0` showed 80% — one core
feeding the card — and both were healthy. `resources_used.ngpus` reads 0 on this
machine even for a running GPU job; it is not a signal.

## 10. Walltime boundaries are inclusive at the top, and they bite in both directions

Queue routing is by walltime band, and every band's upper edge belongs to that band:

```
<= 2:00:00     gdev / qdev   priority 100, 10 (GPU) or 100 (CPU) concurrent jobs
2:00:01 ..     g1-g3 / q*    priority 10, ONE running job per queue per user
.. 24:00:00    g1-g3         24:00:00 exactly is still g1, NOT glong
> 24:00:00     glong         2 concurrent jobs
```

Both edges have been hit here, in opposite directions:

* `2:00:01` silently drops out of the fast lane — documented in the operating guide,
  and the reason `18_quality.pbs` asks for exactly `02:00:00`.
* **`24:00:00` routes to `g1`, not `glong`.** A job that needs a long lane was sent to
  the most congested queue on the machine (43 queued) on 2026-08-13 because 24 h reads
  like "the start of the long band" and is in fact the end of the short one. `30:00:00`
  is what lands in `glong`.

The rule that survives both: **never ask for a band's boundary value.** Pick a walltime
that is unambiguously inside the band you want, and confirm the destination with
`qstat -u $USER` immediately after the submit — the Queue column is the only proof that
the shape did what you meant.

## 11. A generator's default budget is a property of the DATASET, not of the generator

`timevae`'s adapter defaults are `batch_size=16, max_epochs=1000`. On the probe's 20,000
windows that is minutes. On this cohort's 176,445 windows it is 11,028 iterations per
epoch and **11 million iterations, about 14.5 hours** — so the same "default" run was
killed at a 2 h walltime on 2026-08-10 and would have been killed again at 12 h on
2026-08-13 had the arithmetic not been done first.

Two costs are hidden in an epoch-count default: dataset size AND batch size. `timevae`
moves on both against the cost probe (8.8x the windows, a quarter of the batch), which
is why the probe's 14.5 seconds and the real run's 14.5 hours differ by 3,600x rather
than by the 8.8x the window count alone suggests.

Before submitting any generator at its published defaults, convert the budget into
iterations for THIS dataset and multiply by the probe's per-iteration cost. The number
that matters is never the one in the config.

## 12. A per-subject membership classifier learns which RELEASE it is looking at

Training an attack classifier on one subject's rows looks like the clean design: every
row is the same person, so "how unusual this person is" cancels by construction. It is
unusable, and the reason is structural rather than a bug.

Within one subject and one replicate there is **exactly one release containing them**.
So the positive class is one specific release and the negative class is a mixture, and
"is this that release" predicts the label perfectly. Two releases are two training runs
and differ by an offset this project has already measured as large enough to dominate
within-arm readings — so the classifier takes that instead of membership.

**Splitting train/test by window does not break it.** Held-out windows face the same two
releases. Cross-validation offers no protection against a confound that is constant
within the fold.

Measured on 2026-08-13, per-subject protocol, every cell within 0.007 of its own
negative control:

```
                     main    negative control      (the control must read 0.5)
C3_forest x raw10   0.7763        0.7782
C4_hgb    x raw10   0.7685        0.7630
C2_tree   x raw10   0.7478        0.7466
```

Without the control this reads as a strong positive: outliers 0.78 against controls
0.64, +0.14. **The entire table, including the group difference, was release identity.**

**Per-release normalisation made it worse, not better.** Z-scoring each feature against
background windows scored on the same release pushed the tree models' control from 0.55
to 0.78 — dividing by a near-constant reference SD amplifies exactly what it was meant
to remove.

**The fix is to pool across subjects**, so that neither release nor subject carries
label information: every release then holds one member and 39 non-members, and every
subject is a member in exactly one release. `scripts/attack_panel_pooled.py`. The
negative control returns to 0.500 and the power check on `copy_paste` reads 0.82.

The general form: **a confound that is constant within a training set cannot be
detected by any split of that training set.** It needs a control that varies it — here,
a positive release the subject is also not a member of.

## 13. ⛔ `id` is unique only WITHIN a source study

Measured 2026-08-14 on the released parquet (`scripts/check_id_collisions.py`):

```
1,291 distinct ids over 14 source files
ids appearing in more than one source file: 241 (18.7%)

subject "102": 479,352 rows, 5 studies  [DCLP3, Flair, Loop, PEDAP, ReplaceBG]
  2018-03-26: 576 rows, 288 distinct timestamps, from DCLP3 and Loop
    00:00  CGM 213.0  basal 0.091667  DCLP3  Dexcom G6
    00:00  CGM   NaN  basal 0.067083  Loop   Dexcom G5
```

Two CGM devices reporting the same minute, two basal profiles: these are **two people**.
Each study numbers its participants from 1, and the consolidated release did not
re-key them. **The subject key is (source_file, id).**

Everything in this repository keys on `id` alone, so:

- **238 of the 875 cohort subjects (27.2%) are composites** of 2–9 studies, holding
  **74,900 of 182,597 windows (41.0%)**
- `complete_days` counts 288 CGM cells for an (id, day) and passes days that hold two
  people's readings; `build` then takes the first 288 rows in date order, which
  **interleaves them sample by sample**. 9,055 windows (5.0% of the cohort) carry the
  signature — `acf(2) > acf(1)`, impossible for a smooth five-minute trace, against
  0.10% among single-study subjects
- where the second study contributes NaN CGM the interleave makes `isfinite` fail and
  the day is dropped, which is why the corruption is invisible in aggregate statistics

**What this does NOT explain**, checked before assuming the worst
(`scripts/check_collision_impact.py`): the consensus outliers are *depleted* of
composites (2 of 24, 8.3%, against a 27.2% base rate, Fisher p = 0.9955); subject 1142
is single-study (Loop), has no oversized day and no window above the interleave
threshold; and composite subjects are not more identifiable (p = 1). The headline
findings sit on single-study subjects.

The general form: **a key that is unique per source is not unique after a merge.** Test
it by counting sources per key, not by looking for duplicate rows — the rows differ, so
nothing looks duplicated.

## 14. ⚠️ Silent: a float32 reduction along axis 0 stalls, and the size that matters is the RATIO

`X.reshape(-1, C).mean(axis=0)` on a float32 array does **not** use pairwise summation.
Reducing a 2-D array along axis 0 accumulates row by row into a float32 output buffer,
so it is a plain sequential sum. Over 49.6M cells it wrote these constants into a cohort
manifest:

```
            written    true      error
CGM          96.05    145.29     -34%
basal         0.0708    0.0842   -19%
bolus         0.0659    0.0659     0%
```

Nothing raised. The array stays a valid invertible encoding, so only a clinical metric —
recovered by inverting the constants — shows the damage, and it shows it as
wrong-but-plausible numbers.

**The trap inside the trap: this is not a large-magnitude problem.** basal sums to only
3.5e6, comfortably under float32's 2^24 integer limit, and was still off by 19% — because
the float32 spacing at 3.5e6 is 0.25 while each increment is 0.07, so the additions
stopped registering. What matters is `running_sum / increment`, not the absolute size.
A channel of small numbers is exactly as exposed as a channel of large ones.

Why the single-channel path escaped it: an `(N, 1)` reduction is dispatched as a
contiguous 1-D pairwise sum, and `cohort.py` uses `X.mean()` over the whole array, which
is also pairwise. So the bug appeared only when a second channel was added, and the
single-channel result sitting right next to it looked correct.

```python
mu = flat.mean(axis=0, dtype=np.float64)      # dtype is load-bearing, not defensive
sd = flat.std(axis=0, dtype=np.float64)
if abs(mu[i] - np.median(flat[:, i])) > 5 * sd[i]:      # and assert it afterwards
    raise SystemExit(...)
```

## 15. ⚠️ Silent: two of the thirteen outlier votes do not survive a change of machine

Re-deriving the published consensus from scratch on ASPIRE2A gives **22 outliers, not
24** — a strict subset, losing `819` and `909` and gaining nobody. Per-method top-5%
sets against the shipped scores:

```
eleven methods   Jaccard 1.000   max |score diff| 1e-16 .. 6e-07   (round-off)
C9               Jaccard 0.955   max |score diff| 3.7e-02
D11              Jaccard 0.755   max |score diff| 1.6e-01
```

C9 and D11 are the only two methods downstream of the TS2Vec embedding, and both `819`
and `909` sat on exactly 7 votes and lost exactly `D11`.

**It is not a code change, and not a seed.** The encoder fingerprint is byte-identical
across the two runs (`a93691f85dfc6288`, `fitted: false` in both), so the same
checkpoint is loaded. What differs is the cached EMBEDDING array:
`_encoder_cache/real_emb_<fp>_182597.npy` was recomputed on this machine on 2026-08-07,
while `results/outliers/*.parquet` shipped with the repository and its `meta.json`
records `/home/ling/...` — a different machine. `encode_real_cached` returns the cached
array whenever it exists, so the device flag of the run reading it is irrelevant: the
divergence was baked in when the array was regenerated.

D11 is Local Outlier Factor in a 320-dimensional space. LOF is a function of
k-nearest-neighbour *ordering*, and ordering in 320 dimensions is not stable under the
last-bit differences that a different BLAS, GPU or torch build produces. A quarter of
its top-5% set moves; two subjects at the 7-of-13 boundary fall out.

**Consequences.**

- `tests/fixtures/consensus_metabonet875_seed2026.json` cannot be reproduced here. It
  is a record of that code on that machine, which is what its own docstring says it is.
- Any comparison of two outlier lists must have **both** lists computed on the same
  machine with the same embedding array. Comparing a new run against the shipped list
  measures the machine as much as the change.
- The 22 that reproduce are the robust core; `819` and `909` are marginal in the
  literal sense — they clear the bar under one embedding and not another.

The general form: **a cached intermediate makes a result reproducible only as far back
as the cache.** The fingerprint guards against reusing the wrong encoder; nothing
guards against the same encoder having been run through different hardware.

## 16. ⚠️ Silent: the discriminator's weights were never seeded, so the score depends on run ORDER

`quality.discriminative_score` seeded two of the three random things it uses. `seed`
controlled the subsample (`np.random.default_rng`) and the batch draw
(`torch.Generator`). It did not control the GRU's weight initialisation, which came from
torch's **global** rng — never seeded here, and advanced by whatever ran earlier in the
same process.

Measured on 2026-08-17, on a byte-identical `samples.npy` (mtime 2026-08-16 19:43,
unchanged between the two reads) with the same `seed=2026`:

```
seven-day h96 base, scored FIRST in its process    0.9475      05:36
seven-day h96 base, scored SECOND (after h256)     0.5908      21:34
seven-day h96 base, a third process                0.6483      23:0x
```

**The published "the seven-day generator did not fit, discriminative accuracy 0.9475"
rested on one draw of that.** So did every other discriminative accuracy in the project.

### The instability is structured, and the structure is the useful part

Eight restarts per model, one fixed subsample, only the initialisation moving
(`scripts/disc_stability.py`):

```
                             min    median     max    spread
copy_paste, 7 days         0.4950   0.5067   0.5100   0.0150
one-day single channel*    0.4817   0.5025   0.5433   0.0617
h256, 7 days, 22k steps    0.4875   0.5242   0.5508   0.0633
h96,  7 days, 34k steps    0.4792   0.6383   0.8442   0.3650
h96,  7 days, 22k steps    0.4967   0.5538   0.9883   0.4917
                                          *16 scorings across seedcheck_* dirs
```

Where the generator is genuinely indistinguishable the classifier lands at chance every
time and the spread is 0.015–0.06. Where a separable feature exists the draws are
**bimodal** — the classifier either finds the feature or it does not — and the spread is
0.37–0.49.

So the metric is a **lower bound on separability, sampled once**. A single low reading
is weak evidence of quality; a single high reading is strong evidence of separability.

**Report the maximum over restarts, with the spread beside it.** A mean would average a
successful classifier with a failed optimisation and mean nothing.

### What it changed, and what it did not

It did not change the direction of the seven-day finding: h96 reaches 0.844 and 0.988,
so those samples really are separable. It changed what that rests on, and it made the
h96-versus-h256 comparison possible at all — 0.4808 against 0.9475 looked like a
0.47 difference and is not, because 0.4808 was one draw from a distribution whose max
is 0.5508.

It also puts a caveat on numbers that are still quoted elsewhere: the three-channel
quality figures (0.553 / 0.653 / 0.677), the width sweep (0.582–0.643), and the
"spread of 0.124 across three models at one width" that was read as run-to-run model
variation. An unknown part of that 0.124 is the metric, not the models.

Fixed: `torch.manual_seed(seed if init_seed is None else init_seed)` before the network
is built, plus an `init_seed` argument so restarts can be requested deliberately.

The general form: **count the sources of randomness, then count the seeds.** Two out of
three looks seeded from every angle except the one that matters, and the symptom — a
number that moves when you score a different set of models together — looks like a data
problem rather than an RNG problem.

## Shared machines

Measure before scaling up. On this hardware the per-worker cost was **not** the
`OMP_NUM_THREADS` setting, and going from 6 to 12 workers moved total usage from 9.2
to 12.0 cores rather than doubling it — the first stage is I/O bound. Doubling worker
count on the assumption that CPU scales linearly is how trap 2 went unnoticed.

**`gdev`'s GPU pool is 16 cards for the whole machine**, not per user
(`max_run_res.ngpus=[o:PBS_ALL=16]`). The 10-job-per-user allowance is not the
constraint. On 2026-08-12 eight single-GPU `gdev` jobs were submitted at once for
eight evaluation seeds — half the machine's fast lane, held by one user, for work that
batches trivially: `scripts/eval_quality.py --runs` takes several run directories, and
`18_quality.pbs` passes `SEED`/`OUT` through, so N seeds over M models fit in far fewer
jobs. Batch first, then submit.
