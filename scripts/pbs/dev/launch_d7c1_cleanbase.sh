#!/bin/bash
# d7_c1 干净 base:55.3 卡时 / 24 小时墙钟 -> 4 轮链(留余量)。
# afterany:被墙钟切掉不算失败,下一轮从存档续。
set -euo pipefail
cd "$(dirname "$0")/../../.."
STAMP=$(date '+%m%d_%H%M'); prev=""
for r in 1 2 3 4; do
  dep=""; [ -n "$prev" ] && dep="-W depend=afterany:$prev"
  id=$(qsub -N "d7c1base$r" $dep -o "logs/d7c1_cleanbase_${r}_${STAMP}.log" \
       scripts/pbs/dev/d7c1_cleanbase.pbs)
  echo "  第 $r 轮 -> $id"; prev=${id%%.*}
done
