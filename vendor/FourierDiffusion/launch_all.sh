#!/bin/bash
# Launch FourierDiffusion for one (gpu, seed, window) triple, 4 datasets sequentially.
# Usage: ./launch_all.sh <gpu> <seed> [window=64]
set -u
GPU=$1
SEED=$2
WINDOW=${3:-64}

HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
PY=${PYTHON:-python}

mkdir -p logs

for DS in stocks etth energy kddcup; do
    LOG="logs/fdiff_${DS}_seed${SEED}_T${WINDOW}.log"
    echo "=== [GPU $GPU] $(date -Iseconds) START $DS seed=$SEED T=$WINDOW ===" | tee -a "$LOG"
    "$PY" -u run_fdiff_ts.py \
        --dataset "$DS" --seed "$SEED" --gpu "$GPU" --window "$WINDOW" \
        >> "$LOG" 2>&1
    RC=$?
    echo "=== [GPU $GPU] $(date -Iseconds) END $DS seed=$SEED T=$WINDOW rc=$RC ===" | tee -a "$LOG"
done
echo "[GPU $GPU] all 4 datasets done for seed=$SEED T=$WINDOW"
