#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/infer.sh [options]

Options:
  --method framewise|chunkwise          Inference method. Default: framewise.
  --schedule all1|ffe|all4              Denoising schedule. Default: ffe.
  --config_path PATH                    Config path. Defaults to config.yaml or chunkwise_config.yaml.
  --checkpoint_path PATH                Trained checkpoint, usually checkpoint_model_xxxxxx/model.pt.
  --prompt_path PATH                    Prompt txt file. Default: prompts/demos.txt.
  --extended_prompt_path PATH           Optional rewritten/extended prompt txt file.
  --output_folder PATH                  Output folder. Default: outputs/<method>.
  --num_output_frames N                 Latent frames. Default: 21.
  --seed N                              Base seed. Default: 0.
  --limit N                             Number of prompts, -1 means all. Default: -1.
  --num_samples_per_prompt N            Samples per prompt. Default: 1.
  --shard_index N                       Zero-based manifest shard. Default: 0.
  --num_shards N                        Number of strided manifest shards. Default: 1.
  --manifest_path PATH                  Optional paired JSONL evaluation manifest.
  --fps N                               Output video FPS. Default: 16.
  --use_ema                             Load generator_ema from checkpoint.
  --naming MODE                         export_videos.py naming mode. Default: prompt_index.
  --prompt_embedding_cache_path PATH    Optional prompt embedding LMDB cache.
  --offload_generator_before_decode     Offload generator before VAE decode.
  --streaming_decode                    Decode one latent at a time (requires offload).
  --gpu_id ID                           Set CUDA_VISIBLE_DEVICES for this run.
  -h, --help                            Show this help.

EOF
}

METHOD="${METHOD:-framewise}"
SCHEDULE="${SCHEDULE:-ffe}"
CONFIG_PATH="${CONFIG_PATH:-}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-}"
PROMPT_PATH="${PROMPT_PATH:-${REPO_ROOT}/prompts/demos.txt}"
EXTENDED_PROMPT_PATH="${EXTENDED_PROMPT_PATH:-}"
OUTPUT_FOLDER="${OUTPUT_FOLDER:-}"
NUM_OUTPUT_FRAMES="${NUM_OUTPUT_FRAMES:-21}"
SEED="${SEED:-0}"
LIMIT="${LIMIT:--1}"
NUM_SAMPLES_PER_PROMPT="${NUM_SAMPLES_PER_PROMPT:-1}"
SHARD_INDEX="${SHARD_INDEX:-0}"
NUM_SHARDS="${NUM_SHARDS:-1}"
MANIFEST_PATH="${MANIFEST_PATH:-}"
FPS="${FPS:-16}"
USE_EMA="${USE_EMA:-0}"
NAMING="${NAMING:-prompt_index}"
PROMPT_EMBEDDING_CACHE_PATH="${PROMPT_EMBEDDING_CACHE_PATH:-}"
OFFLOAD_GENERATOR_BEFORE_DECODE="${OFFLOAD_GENERATOR_BEFORE_DECODE:-0}"
STREAMING_DECODE="${STREAMING_DECODE:-0}"
GPU_ID="${GPU_ID:-}"
PYTHON_BIN="${PYTHON_BIN:-python}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --method)
      METHOD="$2"
      shift 2
      ;;
    --schedule)
      SCHEDULE="$2"
      shift 2
      ;;
    --config_path)
      CONFIG_PATH="$2"
      shift 2
      ;;
    --checkpoint_path)
      CHECKPOINT_PATH="$2"
      shift 2
      ;;
    --prompt_path)
      PROMPT_PATH="$2"
      shift 2
      ;;
    --extended_prompt_path)
      EXTENDED_PROMPT_PATH="$2"
      shift 2
      ;;
    --output_folder)
      OUTPUT_FOLDER="$2"
      shift 2
      ;;
    --num_output_frames)
      NUM_OUTPUT_FRAMES="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    --limit)
      LIMIT="$2"
      shift 2
      ;;
    --num_samples_per_prompt)
      NUM_SAMPLES_PER_PROMPT="$2"
      shift 2
      ;;
    --shard_index)
      SHARD_INDEX="$2"
      shift 2
      ;;
    --num_shards)
      NUM_SHARDS="$2"
      shift 2
      ;;
    --manifest_path)
      MANIFEST_PATH="$2"
      shift 2
      ;;
    --fps)
      FPS="$2"
      shift 2
      ;;
    --use_ema)
      USE_EMA="1"
      shift
      ;;
    --naming)
      NAMING="$2"
      shift 2
      ;;
    --prompt_embedding_cache_path)
      PROMPT_EMBEDDING_CACHE_PATH="$2"
      shift 2
      ;;
    --offload_generator_before_decode)
      OFFLOAD_GENERATOR_BEFORE_DECODE="1"
      shift
      ;;
    --streaming_decode)
      STREAMING_DECODE="1"
      shift
      ;;
    --gpu_id)
      GPU_ID="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

case "${METHOD}" in
  framewise|chunkwise)
    ;;
  *)
    echo "--method must be framewise or chunkwise, got: ${METHOD}" >&2
    exit 1
    ;;
esac

case "${SCHEDULE}" in
  all1|ffe|all4)
    ;;
  *)
    echo "--schedule must be all1, ffe, or all4; got: ${SCHEDULE}" >&2
    exit 1
    ;;
esac
if [[ "${METHOD}" == "chunkwise" && "${SCHEDULE}" == "ffe" ]]; then
  echo "FFE is the first-4-then-framewise schedule and requires --method framewise" >&2
  exit 1
fi

if [[ -z "${CONFIG_PATH}" ]]; then
  if [[ "${METHOD}" == "chunkwise" ]]; then
    CONFIG_PATH="${REPO_ROOT}/chunkwise_config.yaml"
  else
    CONFIG_PATH="${REPO_ROOT}/config.yaml"
  fi
fi

CONFIG_PATH="$(realpath -m "${CONFIG_PATH}")"
if [[ -n "${CHECKPOINT_PATH}" ]]; then
  CHECKPOINT_PATH="$(realpath -m "${CHECKPOINT_PATH}")"
fi
PROMPT_PATH="$(realpath -m "${PROMPT_PATH}")"
if [[ -n "${EXTENDED_PROMPT_PATH}" ]]; then
  EXTENDED_PROMPT_PATH="$(realpath -m "${EXTENDED_PROMPT_PATH}")"
fi
if [[ -n "${PROMPT_EMBEDDING_CACHE_PATH}" ]]; then
  PROMPT_EMBEDDING_CACHE_PATH="$(realpath -m "${PROMPT_EMBEDDING_CACHE_PATH}")"
fi
if [[ -n "${MANIFEST_PATH}" ]]; then
  MANIFEST_PATH="$(realpath -m "${MANIFEST_PATH}")"
fi

if [[ ! "${NUM_OUTPUT_FRAMES}" =~ ^[0-9]+$ || "${NUM_OUTPUT_FRAMES}" -lt 1 ]]; then
  echo "--num_output_frames must be a positive integer" >&2
  exit 1
fi
if [[ ! "${FPS}" =~ ^[0-9]+$ || "${FPS}" -lt 1 ]]; then
  echo "--fps must be a positive integer" >&2
  exit 1
fi
if [[ ! "${NUM_SHARDS}" =~ ^[0-9]+$ || "${NUM_SHARDS}" -lt 1 ]]; then
  echo "--num_shards must be a positive integer" >&2
  exit 1
fi
if [[ ! "${SHARD_INDEX}" =~ ^[0-9]+$ || "${SHARD_INDEX}" -ge "${NUM_SHARDS}" ]]; then
  echo "--shard_index must be in [0, num_shards-1]" >&2
  exit 1
fi
if [[ "${SCHEDULE}" == "ffe" && "${NUM_OUTPUT_FRAMES}" -lt 4 ]]; then
  echo "FFE requires at least 4 latent frames" >&2
  exit 1
fi
if [[ "${METHOD}" == "chunkwise" && $((NUM_OUTPUT_FRAMES % 3)) -ne 0 ]]; then
  echo "chunkwise inference requires --num_output_frames divisible by 3; got ${NUM_OUTPUT_FRAMES}" >&2
  exit 1
fi
if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Config not found: ${CONFIG_PATH}" >&2
  exit 1
fi
if [[ -z "${CHECKPOINT_PATH}" || ! -f "${CHECKPOINT_PATH}" ]]; then
  echo "--checkpoint_path is required and must point to a file" >&2
  exit 1
fi
if [[ ! -f "${PROMPT_PATH}" ]]; then
  echo "Prompt file not found: ${PROMPT_PATH}" >&2
  exit 1
fi
if [[ -n "${EXTENDED_PROMPT_PATH}" && ! -f "${EXTENDED_PROMPT_PATH}" ]]; then
  echo "Extended prompt file not found: ${EXTENDED_PROMPT_PATH}" >&2
  exit 1
fi
if [[ -n "${MANIFEST_PATH}" && ! -f "${MANIFEST_PATH}" ]]; then
  echo "Manifest not found: ${MANIFEST_PATH}" >&2
  exit 1
fi
if [[ "${STREAMING_DECODE}" == "1" && "${OFFLOAD_GENERATOR_BEFORE_DECODE}" != "1" ]]; then
  echo "--streaming_decode requires --offload_generator_before_decode" >&2
  exit 1
fi

if [[ -z "${OUTPUT_FOLDER}" ]]; then
  OUTPUT_FOLDER="${REPO_ROOT}/outputs/${METHOD}"
fi
OUTPUT_FOLDER="$(realpath -m "${OUTPUT_FOLDER}")"
mkdir -p "${OUTPUT_FOLDER}"

RUN_CONFIG="${OUTPUT_FOLDER}/infer_config_${METHOD}_${SCHEDULE}_shard${SHARD_INDEX}.yaml"
"${PYTHON_BIN}" - "${CONFIG_PATH}" "${RUN_CONFIG}" "${METHOD}" "${SCHEDULE}" "${NUM_OUTPUT_FRAMES}" "${REPO_ROOT}" <<'PY'
import sys
from omegaconf import OmegaConf

src, dst, method, schedule, num_output_frames, repo_root = sys.argv[1:7]
sys.path.insert(0, repo_root)
from utils.config import load_config

cfg = load_config(src)
cfg.num_frame_per_block = 1 if method == "framewise" else 3
cfg.warp_denoising_step = True
if "model_kwargs" not in cfg:
    cfg.model_kwargs = {}
cfg.model_kwargs.local_attn_size = 21
cfg.model_kwargs.sink_size = 0

if "first_frame_denoising_step_list" in cfg:
    del cfg["first_frame_denoising_step_list"]
cfg.rollout_schedule = "fixed"

if schedule == "all1":
    cfg.denoising_step_list = [1000]
elif schedule == "ffe":
    cfg.num_frame_per_block = 1
    cfg.rollout_schedule = "first4then1"
    cfg.first_rollout_num_frames = 4
    cfg.denoising_step_list = [1000]
    cfg.first_frame_denoising_step_list = [1000, 750, 500, 250]
elif schedule == "all4":
    cfg.denoising_step_list = [1000, 750, 500, 250]
else:
    raise ValueError(schedule)
if "image_or_video_shape" in cfg and len(cfg.image_or_video_shape) >= 2:
    cfg.image_or_video_shape[1] = int(num_output_frames)
OmegaConf.save(cfg, dst)
print(dst)
PY

if [[ -n "${GPU_ID}" ]]; then
  export CUDA_VISIBLE_DEVICES="${GPU_ID}"
fi

CMD=(
  "${PYTHON_BIN}" scripts/export_videos.py
  --config_path "${RUN_CONFIG}"
  --checkpoint_path "${CHECKPOINT_PATH}"
  --prompt_path "${PROMPT_PATH}"
  --output_folder "${OUTPUT_FOLDER}"
  --num_output_frames "${NUM_OUTPUT_FRAMES}"
  --seed "${SEED}"
  --limit "${LIMIT}"
  --num_samples_per_prompt "${NUM_SAMPLES_PER_PROMPT}"
  --shard_index "${SHARD_INDEX}"
  --num_shards "${NUM_SHARDS}"
  --naming "${NAMING}"
  --fps "${FPS}"
)

if [[ -n "${EXTENDED_PROMPT_PATH}" ]]; then
  CMD+=(--extended_prompt_path "${EXTENDED_PROMPT_PATH}")
fi
if [[ "${USE_EMA}" == "1" ]]; then
  CMD+=(--use_ema)
fi
if [[ -n "${PROMPT_EMBEDDING_CACHE_PATH}" ]]; then
  CMD+=(--prompt_embedding_cache_path "${PROMPT_EMBEDDING_CACHE_PATH}")
fi
if [[ -n "${MANIFEST_PATH}" ]]; then
  CMD+=(--manifest_path "${MANIFEST_PATH}")
fi
if [[ "${OFFLOAD_GENERATOR_BEFORE_DECODE}" == "1" ]]; then
  CMD+=(--offload_generator_before_decode)
fi
if [[ "${STREAMING_DECODE}" == "1" ]]; then
  CMD+=(--streaming_decode)
fi

cd "${REPO_ROOT}"
echo "Running ${METHOD} inference with ${SCHEDULE} schedule" >&2
echo "Config: ${RUN_CONFIG}" >&2
echo "Output: ${OUTPUT_FOLDER}" >&2
exec "${CMD[@]}"
