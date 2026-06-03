#!/usr/bin/env bash
# Submit env setup to slurm. Allocates a GPU node so flash-attn can compile against CUDA.
set -euo pipefail
WORKDIR="/mnt/lustre/nfs/banyuanhao/One-Forcing"
export MAX_JOBS=128

mkdir -p "$WORKDIR/logs"

# Use srun (foreground) so we capture build errors live; --gres=gpu:1 gives us nvcc + CUDA libs.
srun \
  --partition=a3u \
  --job-name=oneforcing_setup \
  --gres=gpu:1 \
  --cpus-per-task=128 \
  --mem=0 \
  --time=02:00:00 \
  --chdir="$WORKDIR" \
  --export=ALL,MAX_JOBS=${MAX_JOBS} \
  --output="$WORKDIR/logs/setup_%j.out" \
  --error="$WORKDIR/logs/setup_%j.err" \
  bash "$WORKDIR/scripts/setup_env.sh"
