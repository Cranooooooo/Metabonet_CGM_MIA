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
# DEPENDENCY KINDS, and why they differ.
#   prep -> round 1 is `afterok`. On 08-19 it was `afterany`, the prep failed, PBS
#     released all 26 dependents, each bailed on the missing design in under a second,
#     and the queue emptied with nobody looking: 48.5 hours gone. A prerequisite that
#     failed must not release anything.
#   round n -> round n+1 stays `afterany`: one shard failing must not strand the rest,
#     and a rerun skips whatever already has samples.npy.
set -euo pipefail
cd "$(dirname "$0")/../.."
# PREP is OPTIONAL. It used to be mandatory, which forced the watchdog to invent a
# throwaway job just to have an id to depend on -- and that qsub silently failed every
# time it ran, so the relaunch never once worked. With no prep in flight there is
# nothing to wait for, so round 1 simply starts.
PREP=${PREP:-}
if [ -n "$PREP" ]; then DEP0="-W depend=afterok:$PREP"; else DEP0=""; fi
DRY=${1:-}
STAMP=$(date '+%m%d_%H%M')     # keeps a relaunch from overwriting the evidence of the
                               # chain it is replacing; both post-mortems needed those logs
# sub() is always called as $(sub ...), i.e. in a subshell, so a shell variable counter
# never propagates back -- the first version of this check reported "submitted 0" after a
# successful 26-job launch and exited 1. Via the watchdog that reads as a failed relaunch,
# which would leave the marker unset and permit a duplicate chain on the next tick. Count
# through a file, which a subshell can write.
TALLY=$(mktemp)
trap 'rm -f "$TALLY"' EXIT
sub(){ if [ "$DRY" = "--dry-run" ]; then echo "qsub $*" >&2; echo "<job>"
       else qsub "$@"; fi; echo x >> "$TALLY"; }
# round 1 hangs off the prep with afterok (or off nothing); later rounds chain within
# their own lane with afterany, so one shard failing does not strand the rest
dep(){ if [ -n "$1" ]; then echo "-W depend=afterany:$1"; else echo "$DEP0"; fi; }

# ---- glong: the seven-day cells, 8 shards over two 4-card jobs, 2 chained rounds ----
prevA=""; prevB=""
for CELL in d7_c1 d7_c2; do
  for round in 1 2; do
    a=$(sub -l select=1:ncpus=64:ngpus=4 -l walltime=120:00:00 \
        -v "CELL=$CELL,SHARD_BASE=0,N_SHARDS=8" -N "m_${CELL}_a$round" \
        $(dep "$prevA") -o "logs/B2_${CELL}_a${round}_${STAMP}.log" \
        scripts/pbs/B2_matrix_train.pbs)
    b=$(sub -l select=1:ncpus=64:ngpus=4 -l walltime=120:00:00 \
        -v "CELL=$CELL,SHARD_BASE=4,N_SHARDS=8" -N "m_${CELL}_b$round" \
        $(dep "$prevB") -o "logs/B2_${CELL}_b${round}_${STAMP}.log" \
        scripts/pbs/B2_matrix_train.pbs)
    echo "$CELL round $round -> glong $a $b"
    prevA=${a%%.*}; prevB=${b%%.*}
  done
done

# ---- g3 / g2 / g1: the one-day cells, 8 shards, 3 chained rounds each ----
prev3=""; prev2=""; prev1=""
for CELL in d1_c1 d1_c2; do
  for round in 1 2 3; do
    c=$(sub -l select=1:ncpus=64:ngpus=4 -l walltime=24:00:00 \
        -v "CELL=$CELL,SHARD_BASE=0,N_SHARDS=8" -N "m_${CELL}_c$round" \
        $(dep "$prev3") -o "logs/B2_${CELL}_c${round}_${STAMP}.log" \
        scripts/pbs/B2_matrix_train.pbs)
    d=$(sub -l select=1:ncpus=48:ngpus=3 -l walltime=24:00:00 \
        -v "CELL=$CELL,SHARD_BASE=4,N_SHARDS=8" -N "m_${CELL}_d$round" \
        $(dep "$prev2") -o "logs/B2_${CELL}_d${round}_${STAMP}.log" \
        scripts/pbs/B2_matrix_train.pbs)
    e=$(sub -l select=1:ncpus=16:ngpus=1 -l walltime=24:00:00 \
        -v "CELL=$CELL,SHARD_BASE=7,N_SHARDS=8" -N "m_${CELL}_e$round" \
        $(dep "$prev1") -o "logs/B2_${CELL}_e${round}_${STAMP}.log" \
        scripts/pbs/B2_matrix_train.pbs)
    echo "$CELL round $round -> g3 $c  g2 $d  g1 $e"
    prev3=${c%%.*}; prev2=${d%%.*}; prev1=${e%%.*}
  done
done
# A partial chain is the quiet failure here: `set -e` aborts on one rejected qsub and
# the queue still looks busy, so the missing cells are not noticed for days.
NSUB=$(wc -l < "$TALLY")
if [ "$NSUB" -ne 26 ]; then
  echo "WARNING: submitted $NSUB jobs, expected 26 -- the matrix is INCOMPLETE" >&2
  exit 1
fi
echo "submitted $NSUB/26"
[ "$DRY" = "--dry-run" ] || { echo; qstat -u "$(id -un)"; }   # $USER is unset under cron
