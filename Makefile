# Order matters: each target consumes the previous one's output.
PY      ?= python
# A numpy in ~/.local shadows the active environment and breaks scikit-learn's C ABI.
# scripts/ check for this and refuse to run; setting it here means make always wins.
export PYTHONNOUSERSITE = 1
# numba reads its own variable and ignores OMP_NUM_THREADS; without this it spawns one
# thread per core per process on a shared box.
THREADS ?= 8
export OMP_NUM_THREADS   = $(THREADS)
export NUMBA_NUM_THREADS = $(THREADS)
COHORT  ?= data/cohort/metabonet875
OUT     ?= results/outliers
SEED    ?= 2026

.PHONY: help install test smoke fake cohort outliers stability consensus design clean-results

help:
	@grep -E '^[a-z-]+:.*?## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-12s %s\n", $$1, $$2}'

install:  ## editable install + dev extras
	$(PY) -m pip install -e ".[dev]"

test:  ## unit tests + the 28-subject regression fixture
	$(PY) -m pytest tests -q

fake:  ## synthesise a fake cohort so the pipeline runs without the real data
	$(PY) scripts/make_fake_data.py --out data/cohort/fake60 --n-subjects 60

smoke: fake  ## whole pipeline on fake data, CPU only, no data access needed
	$(PY) scripts/run_outliers.py --cohort data/cohort/fake60 --out results/fake \
	      --device cpu
	$(PY) scripts/build_design.py --cohort data/cohort/fake60 \
	      --consensus results/fake/consensus.json --out results/fake/design

cohort:  ## build the real cohort from data/raw/ (see docs/DATA.md)
	$(PY) scripts/build_cohort.py --config configs/data.yaml

outliers:  ## score every subject fourteen ways
	$(PY) scripts/run_outliers.py --cohort $(COHORT) --out $(OUT) --seed $(SEED)

stability:  ## rerun under four seeds; only subjects flagged by all should be used
	$(PY) scripts/seed_stability.py --cohort $(COHORT) --seeds 2026,7,101,999

design:  ## outlier list -> the 57 training runs, with day-count-matched controls
	$(PY) scripts/build_design.py --cohort $(COHORT) --out results/design

consensus:  ## recompute the vote from existing scores (cheap; no rescoring)
	$(PY) -c "import json,sys; sys.path.insert(0,'src'); \
	from cgmoutlier.outliers.run import consensus as c; \
	r=c('$(OUT)'); print(len(r['outliers']),'outliers'); print(' '.join(r['outliers']))"

clean-results:  ## delete scores, keep the cohort
	rm -rf $(OUT) results/seed_stability
