#!/bin/bash
# Wait for FourierDiffusion T=256 ETTh (all 3 seeds) to finish, then re-launch
# the remaining work (energy + kddcup) with seed=2023 moved onto the now-free
# GPU 0, so it no longer contends with DiffTS/TimeVAE on GPU 1/2/3.
#
# The run_fdiff_ts.py skip-check makes re-launch idempotent (already-done
# datasets are skipped instantly).
#
# Run: nohup ./reallocate_gpu0.sh > /tmp/fdiff_realloc.log 2>&1 & disown
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
RES=${RES:-./results/FourierDiffusion/window_256}

log() { echo "[$(date -Iseconds)] $*"; }

log "waiting for FourierDiff T=256 ETTh (3 seeds) to finish..."
while true; do
    done=0
    for S in 2023 1 2; do
        [ -f "$RES/seed$S/etth_fake.npy" ] && done=$((done+1))
    done
    [ "$done" -eq 3 ] && break
    sleep 60
done
log "ETTh done; killing old FourierDiff launchers + python"

# Kill old fdiff launchers and python
pkill -9 -f "FourierDiffusion/launch_all" 2>/dev/null
pkill -9 -f "run_fdiff_ts" 2>/dev/null
sleep 3

log "relaunching: seed2023->GPU0, seed1->GPU1, seed2->GPU2 (T=256)"
nohup bash -c "cd $HERE && ./launch_all.sh 0 2023 256" > logs/launch_gpu0_seed2023_T256_realloc.log 2>&1 & disown
nohup bash -c "cd $HERE && ./launch_all.sh 1 1 256"    > logs/launch_gpu1_seed1_T256_realloc.log 2>&1 & disown
nohup bash -c "cd $HERE && ./launch_all.sh 2 2 256"    > logs/launch_gpu2_seed2_T256_realloc.log 2>&1 & disown
log "relaunched. stocks/etth will be skipped (fake exists); energy+kddcup will run."
