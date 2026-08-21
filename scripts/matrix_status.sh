#!/bin/bash
# One-shot status of the 2x2 matrix. Safe to run any time, from anywhere.
#
# It exists because the campaign sat dead for 48 hours: a multi-stage prep failed at
# step 1, every dependent job was released by `afterany`, each bailed on the missing
# design in under a second, and the queue went empty with nobody looking. The line that
# matters most is the VERDICT at the bottom -- STALLED means jobs are gone and work is
# not finished, which is the exact shape of that failure.
cd "${REPO:-$HOME/workspace/Project_CGM/CGM-OutlierMIA-master}"
echo "=== $(date '+%Y-%m-%d %H:%M') ==="

RUNNING=$(qstat -u "$USER" 2>/dev/null | awk 'NR>5' | grep -c ' R ' || true)
QUEUED=$(qstat -u "$USER" 2>/dev/null | awk 'NR>5' | grep -cE ' [QH] ' || true)
echo "queue: $RUNNING running, $QUEUED queued/held"
qstat -u "$USER" 2>/dev/null | awk 'NR>5{printf "  %-12s %-9s %-11s %s %s\n",$1,$3,$4,$10,$11}' | head -12

echo "prep:"
if [ -f results/matrix/design/rep1/design.json ]; then
  echo "  design OK"
else
  echo "  design MISSING -- last lines of the prep log:"
  tail -3 logs/B1_matrix_prep.live.log 2>/dev/null | sed 's/^/    /'
fi
for c in matrix_d1_c1 matrix_d1_c2 matrix_d7_c1 matrix_d7_c2; do
  if [ -f "data/cohort/$c/manifest.json" ]; then
    python -c "import json;m=json.load(open('data/cohort/$c/manifest.json'));print(f'  cohort $c  {m[\"n_subjects\"]} subj  {m[\"n_windows\"]:,} win  T={m[\"T\"]}')" 2>/dev/null \
      || echo "  cohort $c  built"
  else
    echo "  cohort $c  MISSING"
  fi
done

echo "cells (models with samples.npy, of 21):"
DONE_TOTAL=0
for c in d1_c1 d1_c2 d7_c1 d7_c2; do
  n=$(ls results/runs/matrix_$c/*/samples.npy 2>/dev/null | wc -l)
  DONE_TOTAL=$((DONE_TOTAL+n))
  printf "  %-8s %2d/21\n" "$c" "$n"
done
echo "  total $DONE_TOTAL/84"

echo "recent non-zero exits:"
qstat -xu "$USER" 2>/dev/null | awk 'NR>5{print $1}' | tail -25 | while read j; do
  line=$(qstat -xf "${j%%.*}" 2>/dev/null | grep -oE 'Job_Name = \S+|Exit_status = [0-9-]+' | tr '\n' ' ')
  case "$line" in *"Exit_status = 0"*|"") ;; *Exit_status*) echo "  $line" ;; esac
done | head -8

if [ "$DONE_TOTAL" -ge 84 ]; then
  echo "VERDICT: COMPLETE -- run the analysis"
elif [ "$RUNNING" -eq 0 ] && [ "$QUEUED" -eq 0 ]; then
  echo "VERDICT: STALLED -- nothing queued and $DONE_TOTAL/84 done. Diagnose and relaunch."
else
  echo "VERDICT: RUNNING"
fi
