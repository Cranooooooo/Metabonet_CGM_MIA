# CGM-OutlierMIA

Are subjects whose CGM patterns are atypical more exposed to membership inference
against a synthetic-data generator?

The repository builds a single-variate CGM cohort, scores every subject for
atypicality fourteen ways, takes the consensus outliers, trains a paired
include/exclude generator for each, and measures whether including one subject moves
the released samples measurably closer to that subject.

---

## Quickstart

```bash
conda create -n cgmoutlier python=3.10 -y && conda activate cgmoutlier
pip install -r requirements.txt && pip install -e .
export PYTHONNOUSERSITE=1          # see docs/PITFALLS.md -- this one is not optional

make test                          # 35 tests, ~7 s
make smoke                         # whole pipeline on synthetic data, CPU, ~2 min
```

`make smoke` needs no data access at all: it generates a fake cohort, runs all fifteen
methods on it and builds the paired design. It proves the code runs. It proves nothing
about CGM -- the fake subjects come from three hand-written archetypes, so "who is an
outlier" is true by construction.

The real cohort (875 subjects, 182,597 windows) is **packaged in this repository** as
one lossless 58 MB file, so the outlier stage runs on the real data straight after
cloning:

```bash
make outliers      # 14 methods over 875 subjects -> consensus list   (~30 min, CPU)
make stability     # the same under four base seeds; keep what survives all of them
make design        # consensus list -> 57 training runs, controls matched on day count
```

Only the generator stage needs a GPU and the raw parquet. See `docs/DATA.md`.

---

## Read these before running anything

| document | why |
|---|---|
| `docs/PITFALLS.md` | traps that have cost this project days of compute or an unreproducible number |
| `docs/DATA.md` | where the data comes from, what may be redistributed, and why normalisation constants must be recomputed per cohort |
| `docs/METHODS.md` | the fourteen outlier methods, why each is in the set, with citations |
| `docs/DESIGN.md` | the paired include/exclude design and what it can and cannot show |
| `docs/DIMTS.md` | DiM-TS needs its own environment and a prebuilt CUDA kernel |

## Layout

```
src/cgmoutlier/
  data/        parquet -> windows, normalisation, cohort draw
  outliers/    the 14 methods, one module per group (A/B/C/D/E)
  generators/  adapters over vendor/, one class per baseline
  loo/         paired include/exclude training and sampling
  attack/      subject embedding, aggregation, d_OUT - d_IN
  clinical/    the 32-metric CGM battery
configs/       every knob; nothing is hard-coded in the drivers
scripts/       one driver per stage, in order: fetch -> build_cohort ->
               run_outliers -> seed_stability -> build_design
tests/         35 tests; the design ones are exact, the method ones run on fake
               data, and one pins the outlier list the shipped cohort produces
vendor/        upstream generator code, verbatim, each with PROVENANCE.md
data/cohort/   the packaged 875-subject cohort (no raw data, ever)
results/       scores, votes and the design -- committed, so the shipped data and
               what it produced can be checked against each other
```

## Status

Stages 1-3 (cohort, outlier scoring, paired design) are implemented, tested and run.
The generator adapters under `src/cgmoutlier/generators/` and `vendor/` are ported and
importable; which generator the experiment uses is not yet decided, and no
include/exclude models have been trained. `src/cgmoutlier/attack/` currently holds the
feature extractors only -- the `d_OUT - d_IN` statistic is not written, because the
right distance is a choice to make once the models exist and the options can be
compared on the same runs.
