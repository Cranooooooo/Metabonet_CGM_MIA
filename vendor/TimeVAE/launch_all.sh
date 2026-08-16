#!/bin/bash
# Launch TimeVAE for one (gpu, seed, window) triple, 4 datasets sequentially.
# Usage: ./launch_all.sh <gpu> <seed> [window=64] [max_epochs=1000]
set -u
GPU=$1
SEED=$2
WINDOW=${3:-64}
MAX_EPOCHS=${4:-1000}

HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
PY=${PYTHON:-python}

mkdir -p logs

for DS in stocks etth energy kddcup; do
    LOG="logs/timevae_${DS}_seed${SEED}_T${WINDOW}.log"
    echo "=== [GPU $GPU] $(date -Iseconds) START $DS seed=$SEED T=$WINDOW epochs=$MAX_EPOCHS ===" | tee -a "$LOG"
    "$PY" -u run_timevae_ts.py \
        --dataset "$DS" --seed "$SEED" --gpu "$GPU" --window "$WINDOW" \
        --max_epochs "$MAX_EPOCHS" \
        --verbose 0 \
        >> "$LOG" 2>&1
    RC=$?
    echo "=== [GPU $GPU] $(date -Iseconds) END $DS seed=$SEED T=$WINDOW rc=$RC ===" | tee -a "$LOG"
done
echo "[GPU $GPU] all 4 datasets done for seed=$SEED T=$WINDOW"
