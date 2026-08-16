# Running this repository on ASPIRE2A (NSCC)

Site rules are in `~/NSCC_operation_guide_book.md`; this file is only how *this*
project maps onto them. Probed 2026-08-06 — re-probe anything load-bearing.

## The four facts that shape everything here

1. **No compute on the login node**, and that includes `make test`, `pip install` and
   the conda env build. NSCC auto-flags it and opens a ticket against the project.
   The login shell is for `qsub`, `qstat` and reading logs.
2. **Walltime and GPU count pick the queue.** No `-q`. `<= 2 h` with 0 GPUs lands in
   `qdev` (priority 100, 100 concurrent jobs); `<= 2 h` with 1–4 GPUs lands in `gdev`
   (priority 100, 10 jobs, but a **16-card pool shared by the whole machine**).
   `2:00:01` silently drops you to priority 10 and one running job per queue.
3. **`$HOME`'s quota is enforced and unreadable.** The env (5–7 GB) and everything the
   generator stage writes go to `/data/projects/11704243/$USER`. Run
   `scripts/pbs/setup_links.sh` once to put the symlinks in place.
4. **`conda` does not exist in a batch shell** — the profile defines it as a lazy-init
   function. `scripts/pbs/env.sh` sources the module hook explicitly. Everything else
   sources `env.sh`.

Project code is **11704243** (from `id`; there is no `projects` command on this box).

## Order

```bash
bash scripts/pbs/setup_links.sh                 # once; no compute, login node is fine

qsub scripts/pbs/00_build_env.pbs               # qdev,  ~15 min
qsub scripts/pbs/01_test.pbs                    # qdev,  ~5 min   (35 tests + smoke)
qsub scripts/pbs/02_encoder.pbs                 # gdev,  ~10 min  (TS2Vec cache, 1 GPU)

for s in 7 101 999; do                          # qdev,  3 in parallel, ~1 h
  qsub -v SEED=$s -N cgm_stab_$s -o logs/03_stability_$s.log \
       scripts/pbs/03_stability_seed.pbs
done

qsub scripts/pbs/04_stability_report.pbs        # qdev,  ~2 min -> stability.json
```

Stage 4/5 on the known-positive generator, before any GPU is spent on DiM-TS:

```bash
qsub scripts/pbs/07_loo_copypaste.pbs           # qdev,  ~20 min, 49 models, no GPU
qsub scripts/pbs/08_attack_copypaste.pbs        # qdev,  ~20 min -> the variant choice
```

`copy_paste` releases the training windows verbatim, so **each arm's** gap must come
out strongly positive — both arms are membership pairs, so it is expected to separate
neither arm from the other. That run is what fixes the distance variant
(`set_reduce` x `subject_reduce`); it is frozen there and reused for every real
generator, because choosing it on DiM-TS output would be selecting the test on the
outcome. `docs/DESIGN.md` deliberately left the choice open for exactly this moment.

Each stage needs the previous one's Exit_status=0. Chain them with
`qsub -W depend=afterok:<jobid> ...` if you want it unattended.

## Reading job state without being misled

```bash
qstat -u $USER                     # queued / running
qstat -xf <jobid> | grep -E 'job_state|Exit_status|resources_used'
```

- `-x` is required or a finished job looks like an error.
- `job_state=F` means finished, **not** successful. Success is `Exit_status=0`.
- Logs stage back ~a minute late. An empty log is not a failed job.
- `nvidia-smi` on the login node reports zero busy GPUs on a full machine. Ask the
  scheduler, not the node.
- Verify a run is alive by a log file whose **mtime is advancing**, never by `pgrep`.

## Threads

`env.sh` sets `OMP_NUM_THREADS`, `MKL_NUM_THREADS` and `NUMBA_NUM_THREADS` from the
job's `NCPUS`. numba ignores `OMP_NUM_THREADS` and will otherwise size its pool from
the node's full core count — `docs/PITFALLS.md` §2. The first stage is I/O bound and
does not scale linearly, so 16 cpus is deliberate, not a placeholder.

## Stage 4 (generators) — not written yet, and why

The 49 include/exclude runs do not fit this ladder as-is:

- `gdev`'s 2 h ceiling is almost certainly shorter than one generator's training run,
  and its 16-card pool is shared with everyone on the machine.
- Above 2 h each `g*` queue admits **one running job per user**. The way to get
  throughput is to ask for a queue's whole GPU band and run one independent task per
  card inside the job — `g1`(1) + `g2`(3) + `g3`(4) + 2×`glong`(4) = 15 concurrent
  tasks. `g4` (>=5 GPUs on one node) is structurally unplaceable: 4 cards per node.
- One process, one model (`docs/PITFALLS.md` §3), one card per process (packing four
  workers onto an A100 measured ~40% *worse* than one). A multi-task job holds every
  card until its slowest task ends, so shard by equal-cost units.

## What one model costs — measured, 2026-08-07

A100-40GB, `T=288, C=1`, K = the training-set size (~177k). Training extrapolated to
each generator's own default budget from a two-point rate; sampling measured directly.
Raw numbers in `results/probe/generators.json` and `logs/10_*_card0_dimts.log`.

| generator | params | train / model | sample / model | 41 models |
|---|---|---|---|---|
| `timevae` | — | ~0 | ~0 | **~0 GPU·h** |
| `diffwave` | 2,718,977 | not resolved | 0.56 h | ≥ 23 GPU·h |
| `diffusion_ts` | 223,719 | 0.35 h | 2.31 h | 109 GPU·h |
| `dimts` (hidden 52) | 522,907 | 2.1 h | 1.9 h | **164 GPU·h** |
| `fourier_diff` | 786,887 | 6.5 h | 8.83 h | **629 GPU·h** |

**Sampling dominates for every diffusion model, and it is set by denoising steps, not
by size.** `diffusion_ts` has the fewest parameters of the four and the second-highest
sampling bill; `fourier_diff` denoises in 1000 steps against DiM-TS's 500 and costs 4.7x
as much per sample. Published defaults are not comparable across baselines and should
not be used to choose between them.

Two things that change the bill without changing the generator:

- **Sampling batch size.** DiM-TS measured 101.6 ms/sample at batch 64 and 39.1 ms at
  batch 256 — 2.6x for a knob that does not touch the model. The adapter's default
  `sample_batch` is 1000, so a real run is cheaper than this probe.
- **Denoising steps.** `gaussian_diffusion.py` has `fast_sample`; 500 -> 50 steps takes
  DiM-TS's 1.9 h to roughly 0.2 h. It changes the released distribution, so it is a
  declared condition, not a free saving.

### Rates need two budget points, not one

`train_rate` fits a line through two budgets so that fixed setup and per-step cost come
apart. It matters: `diffusion_ts` spends **434.7 s of setup** against 0.0125 s/step, so
a single 200-step point reads as 2.15 s/step — 170x high. `diffwave` came out with a
*negative* slope, meaning 561 s of setup swamped both budgets and its per-step cost was
simply not measured; the probe now labels that `unmeasured` instead of reporting the
number.
