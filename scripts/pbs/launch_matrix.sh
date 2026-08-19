#!/bin/bash
# Launch all four matrix cells, chained, after the prep job succeeds.
#
# LANE ALLOCATION, and why it is not symmetric.
#   A seven-day model at h256/100k costs ~23 h. g1, g2 and g3 cap at 24 h, so they
#   cannot hold even one -- the seven-day cells are glong-only (2 jobs x 4 cards).
#   A one-day model costs ~12 h, so a g* lane holds one per job and finishes the cell
#   over a chain.
#
#   glong x2 (8 cards)  ->  d7_c1  then  d7_c2      ~69 h each
#   g3+g2+g1 (8 cards)  ->  d1_c1  then  d1_c2      ~37 h each, over chains of 3
#
# Chains use afterany, not afterok: one shard failing must not strand the rest, and a
# rerun skips whatever already has samples.npy.
set -euo pipefail
cd "$(dirname "$0")/../.."
PREP=${PREP:?pass PREP=<prep job id>}
DRY=${1:-}
sub(){ if [ "$DRY" = "--dry-run" ]; then echo "qsub $*" >&2; echo "<job>"; else qsub "$@"; fi }

# ---- glong: the seven-day cells, 8 shards over two 4-card jobs, 2 chained rounds ----
prevA=$PREP; prevB=$PREP
for CELL in d7_c1 d7_c2; do
  for round in 1 2; do
    a=$(sub -l select=1:ncpus=64:ngpus=4 -l walltime=120:00:00 \
        -v "CELL=$CELL,SHARD_BASE=0,N_SHARDS=8" -N "m_${CELL}_a$round" \
        -W "depend=afterany:$prevA" -o "logs/B2_${CELL}_a$round.log" \
        scripts/pbs/B2_matrix_train.pbs)
    b=$(sub -l select=1:ncpus=64:ngpus=4 -l walltime=120:00:00 \
        -v "CELL=$CELL,SHARD_BASE=4,N_SHARDS=8" -N "m_${CELL}_b$round" \
        -W "depend=afterany:$prevB" -o "logs/B2_${CELL}_b$round.log" \
        scripts/pbs/B2_matrix_train.pbs)
    echo "$CELL round $round -> glong $a $b"
    prevA=${a%%.*}; prevB=${b%%.*}
  done
done

# ---- g3 / g2 / g1: the one-day cells, 8 shards, 3 chained rounds each ----
prev3=$PREP; prev2=$PREP; prev1=$PREP
for CELL in d1_c1 d1_c2; do
  for round in 1 2 3; do
    c=$(sub -l select=1:ncpus=64:ngpus=4 -l walltime=24:00:00 \
        -v "CELL=$CELL,SHARD_BASE=0,N_SHARDS=8" -N "m_${CELL}_c$round" \
        -W "depend=afterany:$prev3" -o "logs/B2_${CELL}_c$round.log" \
        scripts/pbs/B2_matrix_train.pbs)
    d=$(sub -l select=1:ncpus=48:ngpus=3 -l walltime=24:00:00 \
        -v "CELL=$CELL,SHARD_BASE=4,N_SHARDS=8" -N "m_${CELL}_d$round" \
        -W "depend=afterany:$prev2" -o "logs/B2_${CELL}_d$round.log" \
        scripts/pbs/B2_matrix_train.pbs)
    e=$(sub -l select=1:ncpus=16:ngpus=1 -l walltime=24:00:00 \
        -v "CELL=$CELL,SHARD_BASE=7,N_SHARDS=8" -N "m_${CELL}_e$round" \
        -W "depend=afterany:$prev1" -o "logs/B2_${CELL}_e$round.log" \
        scripts/pbs/B2_matrix_train.pbs)
    echo "$CELL round $round -> g3 $c  g2 $d  g1 $e"
    prev3=${c%%.*}; prev2=${d%%.*}; prev1=${e%%.*}
  done
done
[ "$DRY" = "--dry-run" ] || { echo; qstat -u "$USER"; }
