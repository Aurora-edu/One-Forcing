#!/usr/bin/env bash
# Submit Self-Forcing Stage-2 framewise One-Forcing training to slurm (1 node, 8 GPUs).
set -euo pipefail
WORKDIR="/mnt/lustre/nfs/banyuanhao/One-Forcing"

mkdir -p "$WORKDIR/logs"

srun \
  --partition=a3u \
  --job-name=of_self_forcing_fw \
  --nodes=1 \
  --ntasks=1 \
  --gres=gpu:8 \
  --cpus-per-task=128 \
  --mem=0 \
  --time=12:00:00 \
  --chdir="$WORKDIR" \
  --export=ALL \
  --output="$WORKDIR/logs/self_forcing_fw_%j.out" \
  --error="$WORKDIR/logs/self_forcing_fw_%j.err" \
  bash "$WORKDIR/scripts/run_self_forcing_framewise.sh"
