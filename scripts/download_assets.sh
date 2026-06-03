#!/usr/bin/env bash
# Download assets for One-Forcing. Network-bound only — no GPU needed; runs on the login node.
# Toggle each group with the env vars below (all default on except the 14B teacher).
#
#   GET_MIXKIT_LMDB   mixkit_latents_lmdb clean-latent LMDB (from tianweiy/CausVid)   ~24 GB
#   GET_WAN_1_3B      Wan2.1-T2V-1.3B base model (fake_score / 1-step generator)      ~18 GB
#   GET_WAN_14B       Wan2.1-T2V-14B teacher model (config default teacher)           ~57 GB
#   GET_ODE_CKPT      framewise/causal_ode.pt ODE-initialized generator               ~5  GB
#   GET_ONEFORCING    one_forcing.pt trained checkpoint (inference)                   ~5  GB
#   GET_CLEAN_DATA    clean_data/* training dataset                                  large
set -euo pipefail

REPO="/mnt/lustre/nfs/banyuanhao/One-Forcing"
cd "$REPO"

export PATH="$HOME/.local/bin:$HOME/.cargo/bin:/usr/local/bin:$PATH"
export HF_TOKEN="$(< /mnt/lustre/nfs/banyuanhao/credentials/hf.key)"
# hf_transfer is faster but can hang on flaky networks; set =0 to use the
# standard (more resume-robust) downloader.
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"

source "$REPO/.venv/bin/activate"

GET_MIXKIT_LMDB="${GET_MIXKIT_LMDB:-1}"
GET_WAN_1_3B="${GET_WAN_1_3B:-1}"
GET_WAN_14B="${GET_WAN_14B:-0}"
GET_ODE_CKPT="${GET_ODE_CKPT:-1}"
GET_ONEFORCING="${GET_ONEFORCING:-1}"
GET_CLEAN_DATA="${GET_CLEAN_DATA:-0}"

mkdir -p wan_models

if [[ "$GET_MIXKIT_LMDB" == "1" ]]; then
  echo "[mixkit] mixkit_latents_lmdb dataset (~24 GB) from tianweiy/CausVid"
  hf download tianweiy/CausVid \
    mixkit_latents_lmdb/data.mdb mixkit_latents_lmdb/lock.mdb --local-dir .
fi

if [[ "$GET_WAN_1_3B" == "1" ]]; then
  echo "[wan] Wan2.1-T2V-1.3B base model (~18 GB)"
  hf download Wan-AI/Wan2.1-T2V-1.3B --local-dir wan_models/Wan2.1-T2V-1.3B
fi

if [[ "$GET_WAN_14B" == "1" ]]; then
  echo "[wan] Wan2.1-T2V-14B teacher model (~57 GB)"
  hf download Wan-AI/Wan2.1-T2V-14B --local-dir wan_models/Wan2.1-T2V-14B
fi

if [[ "$GET_ODE_CKPT" == "1" ]]; then
  echo "[ckpt] framewise/causal_ode.pt ODE init"
  hf download JiaqiFeng/OneForcing checkpoints/framewise/causal_ode.pt --local-dir .
fi

if [[ "$GET_ONEFORCING" == "1" ]]; then
  echo "[ckpt] one_forcing.pt trained checkpoint"
  hf download JiaqiFeng/OneForcing checkpoints/one_forcing.pt --local-dir .
fi

if [[ "$GET_CLEAN_DATA" == "1" ]]; then
  echo "[data] clean_data/* training dataset"
  hf download JiaqiFeng/OneForcing --include "clean_data/*" --local-dir .
fi

echo "[OK] Downloads complete."
du -sh mixkit_latents_lmdb wan_models/* checkpoints 2>/dev/null || true
