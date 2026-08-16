#!/usr/bin/env python
"""Stage 4: train the paired include/exclude generators and release their samples.

One invocation trains one model:

    python scripts/run_loo.py --job include_569 --generator copy_paste

or a disjoint slice of the design, one process per GPU (docs/PITFALLS.md §3 -- the
slices never overlap, so two cards never redo each other's work):

    for i in 0 1 2 3; do
      CUDA_VISIBLE_DEVICES=$i python scripts/run_loo.py \
          --shard $i --n-shards 4 --generator dimts > logs/card$i.log 2>&1 &
    done
    wait

Finished jobs are skipped, so a requeued job costs only what it had not done. The
`base` job is put first in every shard order: 24 outlier pairs and 24 control pairs all
read it, and nothing downstream can start without it.
"""
import argparse
import json
import sys
from pathlib import Path

from cgmoutlier._env import check as _envcheck                     # noqa: E402
_envcheck()
from cgmoutlier.loo.train import run                                  # noqa: E402


def job_order(jobs_dir):
    """base first, then include_*, then exclude_*, each alphabetically.

    A fixed order is what makes --shard reproducible: the same slice must contain the
    same jobs on a rerun, or a requeue silently trains a different set.

    Under the symmetric design there are no exclude_* runs and both arms are include_*,
    so the outlier and control targets interleave alphabetically. That is wanted: a
    shard then holds a mix of both arms rather than one arm's worth, and a shard that
    dies takes a slice out of both.
    """
    files = {p.stem: p for p in Path(jobs_dir).glob("*.json")}
    names = ([n for n in files if n == "base"] +
             sorted(n for n in files if n.startswith("include_")) +
             sorted(n for n in files if n.startswith("exclude_")))
    return [files[n] for n in names]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", default="results/design",
                    help="directory holding design.json and jobs/")
    ap.add_argument("--cohort", default="data/cohort/metabonet875")
    ap.add_argument("--out", default=None,
                    help="default results/runs/<generator>")
    ap.add_argument("--generator", default=None,
                    help="default: models.generator from --config")
    ap.add_argument("--config", default="configs/experiment.yaml",
                    help="read models.generator and models.params from here; "
                         "nothing about the model is hard-coded in this driver")
    ap.add_argument("--params", default=None,
                    help="JSON dict merged OVER the config's params, e.g. "
                         "'{\"steps\": 2000}' for a validation run")
    ap.add_argument("--K", type=int, default=None,
                    help="released samples; default = the job's training-set size")
    ap.add_argument("--job", default=None, help="a single job name, e.g. include_569")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--list", action="store_true", help="print the slice and exit")
    a = ap.parse_args()

    cfg = {}
    if Path(a.config).exists():
        import yaml
        cfg = (yaml.safe_load(Path(a.config).read_text()) or {}).get("models", {}) or {}
    generator = a.generator or cfg.get("generator") or "copy_paste"
    # The config's params belong to the config's generator. hidden_size, save_cycle and
    # sample_batch are DiM-TS's; handing them to timevae or diffwave is meaningless at
    # best and silently reinterpreted at worst, and nothing downstream would show it --
    # meta.json would record params the model never used. Asking for a DIFFERENT
    # generator therefore starts from that generator's own defaults.
    same = generator == (cfg.get("generator") or generator)
    params = dict(cfg.get("params") or {}) if same else {}
    if not same:
        print(f"[loo] --generator {generator} differs from the config's "
              f"{cfg.get('generator')!r}; NOT inheriting its params "
              f"({sorted(cfg.get('params') or {})})")
    if a.params:
        params.update(json.loads(a.params))     # CLI wins, so a probe can shrink a run

    jobs_dir = Path(a.design) / "jobs"
    if not jobs_dir.is_dir():
        sys.exit(f"no jobs under {jobs_dir}; run scripts/build_design.py first")

    if a.job:
        todo = [jobs_dir / f"{a.job}.json"]
        if not todo[0].exists():
            sys.exit(f"no such job: {todo[0]}")
    else:
        todo = job_order(jobs_dir)[a.shard::a.n_shards]

    out = Path(a.out) if a.out else Path("results/runs") / generator

    print(f"[loo] {len(todo)} job(s), shard {a.shard}/{a.n_shards}, "
          f"generator={generator}, out={out}")
    print(f"      params={params}")
    for p in todo:
        print(f"      {p.stem}")
    if a.list:
        return 0

    for p in todo:
        run(p, a.cohort, out, generator=generator, params=params, K=a.K,
            seed=a.seed, device=a.device, overwrite=a.overwrite)
    return 0


if __name__ == "__main__":
    sys.exit(main())
