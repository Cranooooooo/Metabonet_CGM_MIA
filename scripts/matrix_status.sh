#!/bin/bash
# One-shot status of the 2x2 matrix. Safe to run any time, from anywhere.
#
# It exists because the campaign sat dead for 48 hours: a multi-stage prep failed at
# step 1, every dependent job was released by `afterany`, each bailed on the missing
# design in under a second, and the queue went empty with nobody looking.
#
# THE ONE RULE THIS SCRIPT OBEYS. Three separate incidents (08-20, 08-22 10:00, and the
# stale-count read found on 08-22) were the same mistake in different clothes: treating
# "I could not observe" as "I observed nothing". So every read below is checked, and
# anything that cannot be read produces UNKNOWN -- never STALLED, never COMPLETE.
# UNKNOWN means the watchdog does nothing, which is the correct action when blind.
say_unknown(){ echo "queue: UNREADABLE -- $1"; echo "VERDICT: UNKNOWN -- cannot see, so no conclusion and no action"; exit 0; }

REPO=${REPO:-$HOME/workspace/Project_CGM/CGM-OutlierMIA-master}
# without `|| exit` a failed cd leaves us in cron's $HOME, where every artefact reads as
# missing and a finished campaign reports STALLED 0/108
cd "$REPO" || { echo "=== $(date '+%Y-%m-%d %H:%M') ==="; say_unknown "cannot cd to $REPO"; }

# cron runs with PATH=/usr/bin:/bin and qstat lives in /opt/pbs/bin
export PATH="/opt/pbs/bin:$PATH"
# `env -i` and some cron daemons leave USER unset, and `qstat -u ""` returns nothing.
# Ask the system -- but `id` can fail too (LDAP timeout), so check the answer.
WHO=$(id -un 2>/dev/null)

echo "=== $(date '+%Y-%m-%d %H:%M') ==="
command -v qstat >/dev/null || say_unknown "qstat not on PATH"
[ -n "$WHO" ] || say_unknown "id -un returned nothing"

# ONE qstat call, and its exit status is kept. The old code ran `qstat 2>/dev/null | ...`
# twice with `|| true`, which threw away stderr and the exit code both -- so a PBS server
# blip, a failover, or a missing shared library all arrived here as "0 running, 0 queued"
# and came out the far end as STALLED.
QRAW=$(qstat -u "$WHO" 2>&1); QRC=$?
[ $QRC -eq 0 ] || say_unknown "qstat exited $QRC: $(printf '%s' "$QRAW" | head -1)"

# state is column 10; count every state, not just R/Q/H -- a job in E (exiting) or S
# (suspended) is still a live chain, and counting it as absent is a false stall
QROWS=$(printf '%s\n' "$QRAW" | awk 'NR>5 && NF>=10')
RUNNING=$(printf '%s\n' "$QROWS" | awk '$10=="R"' | grep -c . )
QUEUED=$(printf '%s\n'  "$QROWS" | awk '$10=="Q"||$10=="H"' | grep -c . )
ANY=$(printf '%s\n'     "$QROWS" | grep -c . )
echo "queue: $RUNNING running, $QUEUED queued/held, $ANY total"
printf '%s\n' "$QROWS" | awk '{printf "  %-12s %-9s %-11s %s %s\n",$1,$3,$4,$10,$11}' | head -12

echo "prep:"
if [ -f results/matrix/design/rep1/design.json ]; then
  echo "  design OK"
else
  echo "  design MISSING -- last lines of the prep log:"
  tail -3 logs/B1_matrix_prep.live.log 2>/dev/null | sed 's/^/    /'
fi
for c in matrix_d1_c1 matrix_d1_c2 matrix_d7_c1 matrix_d7_c2; do
  if [ -f "data/cohort/$c/manifest.json" ]; then echo "  cohort $c  built"
  else echo "  cohort $c  MISSING"; fi
done

# The per-cell count MUST come from the design. It was a literal 21 after the design
# became 27, and the "fix" then piped design.json through `python`, which does not exist
# on this login node -- so the fallback literal ran every time and the comment claiming
# otherwise was false. grep needs no interpreter and cannot silently not-exist.
N=$(grep -o '"n_jobs"[[:space:]]*:[[:space:]]*[0-9]\+' results/matrix/design/rep1/design.json 2>/dev/null \
    | grep -o '[0-9]\+$')
case "$N" in
  ''|*[!0-9]*) say_unknown "cannot read n_jobs from the design -- refusing to score against a guessed total" ;;
esac
echo "cells (models with samples.npy, of $N):"
DONE_TOTAL=0
for c in d1_c1 d1_c2 d7_c1 d7_c2; do
  n=$(find "results/runs/matrix_$c" -name samples.npy 2>/dev/null | grep -c . )
  DONE_TOTAL=$((DONE_TOTAL+n))
  printf "  %-8s %2d/%d\n" "$c" "$n" "$N"
done
TOT=$((N*4)); echo "  total $DONE_TOTAL/$TOT"

# Only FINISHED jobs carry an Exit_status. The old version took the last 25 rows of the
# history including live ones, so with a 26-job chain queued the window held nothing but
# jobs that had not run yet -- the one detector aimed at "26 jobs exited 1 in a second"
# was blind in exactly the situation it was written for.
echo "recent non-zero exits:"
printf '%s\n' "$(qstat -xu "$WHO" 2>/dev/null | awk 'NR>5 && $10=="F"{print $1}' | tail -12)" \
| while read -r j; do
  [ -n "$j" ] || continue
  line=$(qstat -xf "${j%%.*}" 2>/dev/null | grep -oE 'Job_Name = \S+|Exit_status = [0-9-]+' | tr '\n' ' ')
  case "$line" in
    "")                 echo "  ${j%%.*}: record unreadable" ;;
    *"Exit_status = 0"*) ;;
    *Exit_status*)      echo "  $line" ;;
  esac
done | head -8

if [ "$DONE_TOTAL" -ge "$TOT" ]; then
  echo "VERDICT: COMPLETE -- run the analysis"
elif [ "$ANY" -eq 0 ]; then
  echo "VERDICT: STALLED -- nothing in the queue and $DONE_TOTAL/$TOT done. Diagnose and relaunch."
else
  echo "VERDICT: RUNNING"
fi
