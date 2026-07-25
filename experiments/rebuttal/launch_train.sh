#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

usage() {
  cat <<'EOF'
Usage:
  bash experiments/rebuttal/launch_train.sh \
    --config_path experiments/rebuttal/configs/train_1step_one_forcing.yaml \
    --run_name train_1step_one_forcing \
    --seed 0 \
    --gpus 0,1,2,3,4,5,6,7 \
    --generator_ckpt checkpoints/framewise/causal_ode.pt \
    --teacher_model_path wan_models/Wan2.1-T2V-14B \
    --data_path clean_data

Options:
  --config_path PATH          Rebuttal training config.
  --run_name NAME             Filesystem/tmux-safe experiment name.
  --seed N                    Fixed non-negative base seed.
  --gpus IDS                  Comma-separated CUDA device IDs.
  --generator_ckpt PATH       Shared ODE initialization.
  --teacher_model_path PATH   Wan 14B teacher directory.
  --data_path PATH            Clean-latent LMDB directory.
  --prompt_embedding_cache_path PATH
                              Optional prompt-embedding LMDB. When set, T5 is
                              not loaded by the distributed training ranks.
  --output_root PATH          Default: runs/rebuttal.
  --session NAME              Default: of_<run_name>_s<seed>.
  --resume_ckpt PATH          Optional weights/step resume (optimizer state is not restored).
  --max_steps N               Optional smoke/debug override.
  --text_encoder_cpu_offload  Hardware-only T5 FSDP CPU offload.
  --real_score_cpu_offload    Hardware-only frozen-teacher FSDP CPU offload.
  --fake_score_cpu_offload    Hardware-only fake-score FSDP CPU offload.
  --manual_generator_backward Low-peak equivalent generator backward.
  --generator_optimizer_state_cpu_offload
                              Keep AdamW state on CPU between generator steps.
  --rank0_preload_generator_ckpt  Avoid every rank loading the ODE checkpoint.
  --no_save                   Do not save checkpoints.
  --python PATH               Python executable. Default: python.
  -h, --help                  Show this help.
EOF
}

CONFIG_PATH=""
RUN_NAME=""
SEED=""
GPUS=""
GENERATOR_CKPT=""
TEACHER_MODEL_PATH=""
DATA_PATH=""
PROMPT_EMBEDDING_CACHE_PATH=""
OUTPUT_ROOT="${REPO_ROOT}/runs/rebuttal"
SESSION=""
RESUME_CKPT=""
MAX_STEPS=""
NO_SAVE="0"
TEXT_ENCODER_CPU_OFFLOAD="0"
REAL_SCORE_CPU_OFFLOAD="0"
FAKE_SCORE_CPU_OFFLOAD="0"
MANUAL_GENERATOR_BACKWARD="0"
GENERATOR_OPTIMIZER_STATE_CPU_OFFLOAD="0"
RANK0_PRELOAD_GENERATOR_CKPT="0"
PYTHON_BIN="${PYTHON_BIN:-python}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config_path) CONFIG_PATH="$2"; shift 2 ;;
    --run_name) RUN_NAME="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --gpus) GPUS="$2"; shift 2 ;;
    --generator_ckpt) GENERATOR_CKPT="$2"; shift 2 ;;
    --teacher_model_path) TEACHER_MODEL_PATH="$2"; shift 2 ;;
    --data_path) DATA_PATH="$2"; shift 2 ;;
    --prompt_embedding_cache_path) PROMPT_EMBEDDING_CACHE_PATH="$2"; shift 2 ;;
    --output_root) OUTPUT_ROOT="$2"; shift 2 ;;
    --session) SESSION="$2"; shift 2 ;;
    --resume_ckpt) RESUME_CKPT="$2"; shift 2 ;;
    --max_steps) MAX_STEPS="$2"; shift 2 ;;
    --no_save) NO_SAVE="1"; shift ;;
    --text_encoder_cpu_offload) TEXT_ENCODER_CPU_OFFLOAD="1"; shift ;;
    --real_score_cpu_offload) REAL_SCORE_CPU_OFFLOAD="1"; shift ;;
    --fake_score_cpu_offload) FAKE_SCORE_CPU_OFFLOAD="1"; shift ;;
    --manual_generator_backward) MANUAL_GENERATOR_BACKWARD="1"; shift ;;
    --generator_optimizer_state_cpu_offload) GENERATOR_OPTIMIZER_STATE_CPU_OFFLOAD="1"; shift ;;
    --rank0_preload_generator_ckpt) RANK0_PRELOAD_GENERATOR_CKPT="1"; shift ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 1 ;;
  esac
done

for value_name in CONFIG_PATH RUN_NAME SEED GPUS GENERATOR_CKPT TEACHER_MODEL_PATH DATA_PATH; do
  if [[ -z "${!value_name}" ]]; then
    echo "--${value_name,,} is required" >&2
    exit 1
  fi
done
if [[ ! "${RUN_NAME}" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  echo "--run_name must contain only letters, digits, dot, underscore, or dash" >&2
  exit 1
fi
if [[ ! "${SEED}" =~ ^[0-9]+$ ]]; then
  echo "--seed must be a non-negative integer" >&2
  exit 1
fi
if [[ -n "${MAX_STEPS}" && ( ! "${MAX_STEPS}" =~ ^[0-9]+$ || "${MAX_STEPS}" -lt 1 ) ]]; then
  echo "--max_steps must be a positive integer" >&2
  exit 1
fi

CONFIG_PATH="$(realpath -m "${CONFIG_PATH}")"
GENERATOR_CKPT="$(realpath -m "${GENERATOR_CKPT}")"
TEACHER_MODEL_PATH="$(realpath -m "${TEACHER_MODEL_PATH}")"
DATA_PATH="$(realpath -m "${DATA_PATH}")"
if [[ -n "${PROMPT_EMBEDDING_CACHE_PATH}" ]]; then
  PROMPT_EMBEDDING_CACHE_PATH="$(realpath -m "${PROMPT_EMBEDDING_CACHE_PATH}")"
fi
OUTPUT_ROOT="$(realpath -m "${OUTPUT_ROOT}")"
if [[ -n "${RESUME_CKPT}" ]]; then
  RESUME_CKPT="$(realpath -m "${RESUME_CKPT}")"
fi

[[ -f "${CONFIG_PATH}" ]] || { echo "Config not found: ${CONFIG_PATH}" >&2; exit 1; }
[[ -f "${GENERATOR_CKPT}" ]] || { echo "Generator checkpoint not found: ${GENERATOR_CKPT}" >&2; exit 1; }
[[ -d "${TEACHER_MODEL_PATH}" ]] || { echo "Teacher model not found: ${TEACHER_MODEL_PATH}" >&2; exit 1; }
[[ -f "${DATA_PATH}/data.mdb" ]] || { echo "LMDB data.mdb not found in: ${DATA_PATH}" >&2; exit 1; }
if [[ -n "${PROMPT_EMBEDDING_CACHE_PATH}" && ! -f "${PROMPT_EMBEDDING_CACHE_PATH}/data.mdb" ]]; then
  echo "Prompt cache data.mdb not found in: ${PROMPT_EMBEDDING_CACHE_PATH}" >&2
  exit 1
fi
if [[ -n "${RESUME_CKPT}" && ! -f "${RESUME_CKPT}" ]]; then
  echo "Resume checkpoint not found: ${RESUME_CKPT}" >&2
  exit 1
fi
command -v tmux >/dev/null || { echo "tmux is required" >&2; exit 1; }
command -v "${PYTHON_BIN}" >/dev/null || { echo "Python not found: ${PYTHON_BIN}" >&2; exit 1; }

IFS=',' read -r -a GPU_ARRAY <<< "${GPUS}"
NPROC="${#GPU_ARRAY[@]}"
if [[ "${NPROC}" -lt 1 ]]; then
  echo "--gpus did not contain a device ID" >&2
  exit 1
fi
for gpu_id in "${GPU_ARRAY[@]}"; do
  if [[ ! "${gpu_id}" =~ ^[0-9]+$ ]]; then
    echo "--gpus must contain only non-negative integer IDs: ${GPUS}" >&2
    exit 1
  fi
  busy_pids="$(nvidia-smi -i "${gpu_id}" --query-compute-apps=pid \
    --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d' || true)"
  if [[ -n "${busy_pids}" ]]; then
    echo "GPU ${gpu_id} already has compute process(es): ${busy_pids}" >&2
    echo "Refusing to overlap another session's training process." >&2
    exit 1
  fi
done

SESSION="${SESSION:-of_${RUN_NAME}_s${SEED}}"
if [[ ! "${SESSION}" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  echo "--session contains unsupported characters" >&2
  exit 1
fi
if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  exit 1
fi

RUN_DIR="${OUTPUT_ROOT}/${RUN_NAME}/seed_${SEED}"
if [[ -e "${RUN_DIR}" && -z "${RESUME_CKPT}" ]]; then
  echo "Run directory already exists; choose a new run name or pass --resume_ckpt: ${RUN_DIR}" >&2
  exit 1
fi

"${PYTHON_BIN}" "${SCRIPT_DIR}/validate_configs.py"
PREFLIGHT_CMD=(
  "${PYTHON_BIN}"
  "${SCRIPT_DIR}/preflight.py"
  --config_path "${CONFIG_PATH}"
  --generator_ckpt "${GENERATOR_CKPT}"
  --teacher_model_path "${TEACHER_MODEL_PATH}"
  --data_path "${DATA_PATH}"
  --gpus "${GPUS}"
  --seed "${SEED}"
)
if [[ -n "${PROMPT_EMBEDDING_CACHE_PATH}" ]]; then
  PREFLIGHT_CMD+=(--prompt_embedding_cache_path "${PROMPT_EMBEDDING_CACHE_PATH}")
fi
if [[ -n "${MAX_STEPS}" ]]; then
  PREFLIGHT_CMD+=(--max_steps "${MAX_STEPS}")
fi
"${PREFLIGHT_CMD[@]}"
mkdir -p "${RUN_DIR}"

CMD=(
  "${PYTHON_BIN}"
  -m
  torch.distributed.run
  --standalone
  --nproc_per_node="${NPROC}"
  train.py
  --config_path "${CONFIG_PATH}"
  --seed "${SEED}"
  --generator_ckpt "${GENERATOR_CKPT}"
  --teacher_model_path "${TEACHER_MODEL_PATH}"
  --data_path "${DATA_PATH}"
  --dataset_type clean_latent_lmdb
  --logdir "${RUN_DIR}"
  --disable-wandb
  --no_visualize
)
if [[ -n "${PROMPT_EMBEDDING_CACHE_PATH}" ]]; then
  CMD+=(--prompt_embedding_cache_path "${PROMPT_EMBEDDING_CACHE_PATH}")
fi
if [[ -n "${RESUME_CKPT}" ]]; then
  CMD+=(--resume_ckpt "${RESUME_CKPT}")
fi
if [[ -n "${MAX_STEPS}" ]]; then
  CMD+=(--max_steps "${MAX_STEPS}")
fi
if [[ "${NO_SAVE}" == "1" ]]; then
  CMD+=(--no_save)
fi
if [[ "${TEXT_ENCODER_CPU_OFFLOAD}" == "1" ]]; then
  CMD+=(--text_encoder_cpu_offload)
fi
if [[ "${REAL_SCORE_CPU_OFFLOAD}" == "1" ]]; then
  CMD+=(--real_score_cpu_offload)
fi
if [[ "${FAKE_SCORE_CPU_OFFLOAD}" == "1" ]]; then
  CMD+=(--fake_score_cpu_offload)
fi
if [[ "${MANUAL_GENERATOR_BACKWARD}" == "1" ]]; then
  CMD+=(--manual_generator_backward)
fi
if [[ "${GENERATOR_OPTIMIZER_STATE_CPU_OFFLOAD}" == "1" ]]; then
  CMD+=(--generator_optimizer_state_cpu_offload)
fi
if [[ "${RANK0_PRELOAD_GENERATOR_CKPT}" == "1" ]]; then
  CMD+=(--rank0_preload_generator_ckpt)
fi

printf -v QUOTED_CMD '%q ' "${CMD[@]}"
printf -v QUOTED_REPO '%q' "${REPO_ROOT}"
printf -v QUOTED_GPUS '%q' "${GPUS}"
printf -v QUOTED_LOG '%q' "${RUN_DIR}/train.log"
ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"
printf -v QUOTED_ALLOC_CONF '%q' "${ALLOC_CONF}"
FULL_COMMAND="set -o pipefail; cd ${QUOTED_REPO}; export CUDA_VISIBLE_DEVICES=${QUOTED_GPUS}; export PYTORCH_CUDA_ALLOC_CONF=${QUOTED_ALLOC_CONF}; ${QUOTED_CMD}2>&1 | tee ${QUOTED_LOG}"
printf -v SHELL_COMMAND '%q' "${FULL_COMMAND}"
tmux new-session -d -s "${SESSION}" "bash -lc ${SHELL_COMMAND}"

echo "Started tmux session: ${SESSION}"
echo "Run directory: ${RUN_DIR}"
echo "Attach with: tmux attach -t ${SESSION}"
echo "Monitor with: tail -f ${RUN_DIR}/train.log"
