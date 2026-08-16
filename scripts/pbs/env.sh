# Sourced by every job script here. Not executable on its own, and never on the
# login node -- NSCC auto-flags login-node compute and raises a ticket against the
# project (NSCC_operation_guide_book.md §1).
#
# Everything site-specific lives here so the job scripts stay portable.

set -euo pipefail

PROJECT=${PROJECT:-11704243}
WORK=${WORK:-/data/projects/$PROJECT/$USER}
ENV_PREFIX=${ENV_PREFIX:-$WORK/envs/cgmoutlier}
REPO=${REPO:-$HOME/workspace/Project_CGM/CGM-OutlierMIA-master}

# $HOME quota is enforced and unreadable (§5); pip's cache alone has reached 13 GB.
export PIP_CACHE_DIR=${PIP_CACHE_DIR:-$WORK/.cache/pip}

# torch is un-upgradable by accident. Installing the generators extra once pulled
# torchvision, and pip satisfied it by replacing torch 2.7.0+cu126 with 2.13.0+cu130 --
# a build whose CUDA runtime is newer than this machine's 12.8 driver, so CUDA silently
# became unavailable. See constraints.txt.
export PIP_CONSTRAINT=${PIP_CONSTRAINT:-$REPO/constraints.txt}

# conda is a lazy-init shell function in the interactive profile and simply does not
# exist in a batch shell (§6). The module only sets NSCC_MINIFORGE3_DIR -- it does not
# put conda on PATH -- so the hook has to be sourced by hand.
module load miniforge3/25.3.1
set +u
source "$NSCC_MINIFORGE3_DIR/etc/profile.d/conda.sh"
set -u

# docs/PITFALLS.md §1: a numpy in ~/.local shadows the environment whatever conda does.
export PYTHONNOUSERSITE=1

# docs/PITFALLS.md §2: numba sizes its pool from its own variable and ignores
# OMP_NUM_THREADS. PBS exports NCPUS for the job's allocation.
THREADS=${THREADS:-${NCPUS:-8}}
export OMP_NUM_THREADS=$THREADS
export NUMBA_NUM_THREADS=$THREADS
export MKL_NUM_THREADS=$THREADS

cd "$REPO"

echo "host=$(hostname) job=${PBS_JOBID:-none} threads=$THREADS"
echo "repo=$REPO"
echo "env=$ENV_PREFIX"
