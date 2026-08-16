#!/bin/bash
# Launch the stage-4 campaign: 3 replicates x 41 DiM-TS models at hidden_size=128.
#
# Run from the repo root ON THE LOGIN NODE -- this only submits, it computes nothing.
#
#   bash scripts/pbs/launch_campaign.sh            # submit
#   bash scripts/pbs/launch_campaign.sh --dry-run  # print the qsubs and stop
#
# FOUR LANES, TWELVE CARDS. Above 2 h each g* queue admits one running job per user, so
# throughput comes from asking for a queue's GPU band and running one model per card
# (guide section 3). glong allows two concurrent jobs at 4 cards each, g2 takes 3 and g1
# takes 1: 4+4+3+1 = 12. g3 also takes 4 and is the natural fifth lane, but it was 97
# jobs deep when this was written -- check `qstat -Q` and add it if it has cleared.
#
# ONE WAVE PER REPLICATE, chained per lane with afterany. 123 models do not fit one
# 12-shard fan-out: 10-11 models a shard is ~55 h, past g2's and g1's 24 h ceiling. Each
# wave is the validated 41-model shape and finishes as a COMPLETE replicate, so stopping
# after two leaves two usable replicates rather than three partial ones.
#
# afterany, not afterok: one shard of four failing sets the job's exit status, and that
# should not strand the other two replicates. Watch the first wave and qdel the
# dependents if it goes wrong -- they are cheap to resubmit and the design directories
# are already built.
set -euo pipefail

cd "$(dirname "$0")/../.."
DRY=${1:-}
REPLICATES=${REPLICATES:-1 2 3}
WIDTH=${WIDTH:-h128}

# The three-channel study is the same campaign on a different cohort and design:
#
#   DESIGN_ROOT=results/design_sid_c3 RUN_TAG=dimts_c3_${WIDTH} \
#   COHORT=data/cohort/metabonet_sid_c3 bash scripts/pbs/launch_campaign.sh
#
# RUN_TAG must change with the cohort. loo.train refuses to skip a finished directory
# whose meta.json fingerprints a different subject list, so a collision fails loudly --
# but only for jobs that exist in both designs, and `base` exists in every one.
DESIGN_ROOT=${DESIGN_ROOT:-results/design_sym}
RUN_TAG=${RUN_TAG:-dimts_${WIDTH}}
COHORT=${COHORT:-data/cohort/metabonet875}

# lane: name  ncpus  ngpus  walltime  shard_base
# walltime follows the load: shards 0-4 hold 4 models, the rest 3, at ~5.4 h each
# (results/probe/capacity.json, plus the 5% the h=52 probe underestimated the real run
# by). mem is never named -- PBS pins it to 110 gb per card whatever is asked.
LANES=(
    "a 64 4 30:00:00 0"      # -> glong   shards 0-3   4 models/card  21.6 h
    "b 64 4 30:00:00 4"      # -> glong   shards 4-7   4,3,3,3        21.6 h
    "c 48 3 20:00:00 8"      # -> g2      shards 8-10  3              16.2 h
    "d 16 1 20:00:00 11"     # -> g1      shard  11    3              16.2 h
)

declare -A PREV
for r in $REPLICATES; do
    design="$DESIGN_ROOT/rep$r"
    out="results/runs/${RUN_TAG}_rep$r"
    test -f "$design/design.json" || { echo "no $design -- build the design first"; exit 1; }
    test -d "$COHORT" || { echo "no cohort $COHORT"; exit 1; }

    for lane in "${LANES[@]}"; do
        read -r name ncpus ngpus wall base <<< "$lane"
        args=(-l "select=1:ncpus=$ncpus:ngpus=$ngpus" -l "walltime=$wall"
              -v "DESIGN=$design,OUT=$out,COHORT=$COHORT,SHARD_BASE=$base,N_SHARDS=12"
              # the tag is in the log name too: without it a second campaign silently
              # overwrites the first one's lane logs, which are the only record of how
              # the published models were produced
              -N "cgm_d${r}${name}" -o "logs/13_${RUN_TAG}_r${r}${name}.log")
        [ -n "${PREV[$name]:-}" ] && args+=(-W "depend=afterany:${PREV[$name]}")

        if [ "$DRY" = "--dry-run" ]; then
            echo "qsub ${args[*]} scripts/pbs/13_loo_dimts.pbs"
            PREV[$name]="<rep$r-$name>"
        else
            id=$(qsub "${args[@]}" scripts/pbs/13_loo_dimts.pbs)
            echo "rep$r lane $name  shards $base..$((base + ngpus - 1))  -> $id"
            PREV[$name]=$id
        fi
    done
done

[ "$DRY" = "--dry-run" ] || { echo; qstat -u "$USER"; }
