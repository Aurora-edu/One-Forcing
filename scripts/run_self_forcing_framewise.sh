#!/usr/bin/env bash
# Self-Forcing Stage-2 one-step One-Forcing training (framewise) on 1 H200 node (8 GPUs).
# Launched inside an srun by scripts/run_self_forcing_framewise_slurm.sh.
set -euo pipefail
REPO="/mnt/lustre/nfs/banyuanhao/One-Forcing"
cd "$REPO"

export PATH="$HOME/.local/bin:$HOME/.cargo/bin:/usr/local/bin:$PATH"
export CUDA_HOME="/usr/local/cuda"
export PATH="${CUDA_HOME}/bin:${PATH}"
source "$REPO/.venv/bin/activate"

export TORCH_CUDA_ARCH_LIST="9.0"
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=${MASTER_PORT:-29505}

GENERATOR_CKPT="${GENERATOR_CKPT:-/mnt/lustre/nfs/banyuanhao/CausVid/experiments/wan_causal_ode_framewise/2026-06-02-02-52-47.377891_seed6237543/checkpoint_model_003000/model.pt}"

echo "[self_forcing_fw] launching torchrun (8 GPUs)"
torchrun --standalone --nproc_per_node=8 train.py \
  --config_path self_forcing_config_framewise.yaml \
  --generator_ckpt "$GENERATOR_CKPT" \
  --teacher_model_path wan_models/Wan2.1-T2V-14B \
  --data_path ./mixkit_latents_lmdb \
  --dataset_type clean_latent_lmdb \
  --logdir runs_self_forcing \
  --disable-wandb \
  --no_visualize
