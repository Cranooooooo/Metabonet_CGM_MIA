#!/bin/bash
# Filesystem plumbing only -- no compute, so this one is safe on the login node.
#
# $HOME's quota is enforced and cannot be read (guide §5); you find it by hitting it
# mid-install, which leaves a corrupted environment behind. The repository itself is
# 76 MB and stays in $HOME. Everything that grows goes to the project space:
#
#   results/_encoder_cache   225 MB   TS2Vec weights + encoded cohort
#   results/runs             GB-scale generator checkpoints and samples (stage 4)
#
set -euo pipefail
PROJECT=${PROJECT:-11704243}
WORK=/data/projects/$PROJECT/$USER
REPO=${REPO:-$HOME/workspace/Project_CGM/CGM-OutlierMIA-master}

mkdir -p "$WORK"/{envs,runs,.cache/pip} "$WORK"/encoder_cache
mkdir -p "$REPO"/logs "$REPO"/results

link() {   # link <repo-relative path> <target>
    local src="$REPO/$1" dst="$2"
    if [ -L "$src" ]; then echo "already linked: $1 -> $(readlink "$src")"; return; fi
    if [ -e "$src" ]; then echo "SKIP $1: exists and is not a symlink"; return; fi
    ln -s "$dst" "$src"; echo "linked $1 -> $dst"
}

link results/_encoder_cache "$WORK/encoder_cache"
link results/runs           "$WORK/runs"

echo
echo "project space: $WORK"
du -sh "$WORK" 2>/dev/null || true
