#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

usage() {
  cat <<'EOF'
Usage:
  bash experiments/rebuttal/run_vbench_condition.sh \
    --name main_step600_ffe \
    --config_path experiments/rebuttal/configs/eval_ffe.yaml \
    --checkpoint_path PATH/model.pt \
    --schedule ffe \
    --prompt_path eval/manifests/vbench_official_prompts.txt \
    --manifest_path eval/manifests/vbench_official_seed0.jsonl \
    --full_info_path PATH/VBench_full_info.json \
    --output_root eval/final/main_step600_ffe \
    --gpus 0,1,2,3,4,5,6,7 \
    --vbench_python PATH/TO/VBENCH/PYTHON \
    --use_ema

Options:
  --dimensions CSV   Optional selected VBench dimensions. Omit for all 16.
  --python PATH      Training/inference Python. Default: python.
EOF
}

NAME=""
CONFIG_PATH=""
CHECKPOINT_PATH=""
SCHEDULE=""
PROMPT_PATH=""
MANIFEST_PATH=""
FULL_INFO_PATH=""
OUTPUT_ROOT=""
GPUS=""
VBENCH_PYTHON=""
DIMENSIONS=""
USE_EMA="0"
PYTHON_BIN="${PYTHON_BIN:-python}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --name) NAME="$2"; shift 2 ;;
    --config_path) CONFIG_PATH="$2"; shift 2 ;;
    --checkpoint_path) CHECKPOINT_PATH="$2"; shift 2 ;;
    --schedule) SCHEDULE="$2"; shift 2 ;;
    --prompt_path) PROMPT_PATH="$2"; shift 2 ;;
    --manifest_path) MANIFEST_PATH="$2"; shift 2 ;;
    --full_info_path) FULL_INFO_PATH="$2"; shift 2 ;;
    --output_root) OUTPUT_ROOT="$2"; shift 2 ;;
    --gpus) GPUS="$2"; shift 2 ;;
    --vbench_python) VBENCH_PYTHON="$2"; shift 2 ;;
    --dimensions) DIMENSIONS="$2"; shift 2 ;;
    --use_ema) USE_EMA="1"; shift ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 1 ;;
  esac
done

for value_name in NAME CONFIG_PATH CHECKPOINT_PATH SCHEDULE PROMPT_PATH \
  MANIFEST_PATH FULL_INFO_PATH OUTPUT_ROOT GPUS VBENCH_PYTHON; do
  if [[ -z "${!value_name}" ]]; then
    echo "--${value_name,,} is required" >&2
    exit 1
  fi
done
case "${SCHEDULE}" in
  all1|ffe|all4) ;;
  *) echo "--schedule must be all1, ffe, or all4" >&2; exit 1 ;;
esac
if [[ ! "${NAME}" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  echo "--name contains unsupported characters" >&2
  exit 1
fi

CONFIG_PATH="$(realpath -m "${CONFIG_PATH}")"
CHECKPOINT_PATH="$(realpath -m "${CHECKPOINT_PATH}")"
PROMPT_PATH="$(realpath -m "${PROMPT_PATH}")"
MANIFEST_PATH="$(realpath -m "${MANIFEST_PATH}")"
FULL_INFO_PATH="$(realpath -m "${FULL_INFO_PATH}")"
OUTPUT_ROOT="$(realpath -m "${OUTPUT_ROOT}")"

for path in "${CONFIG_PATH}" "${CHECKPOINT_PATH}" "${PROMPT_PATH}" \
  "${MANIFEST_PATH}" "${FULL_INFO_PATH}" "${VBENCH_PYTHON}"; do
  if [[ ! -f "${path}" ]]; then
    echo "Required file not found: ${path}" >&2
    exit 1
  fi
done

VIDEOS_DIR="${OUTPUT_ROOT}/videos"
VBENCH_DIR="${OUTPUT_ROOT}/vbench"
INFER_CMD=(
  "${PYTHON_BIN}"
  "${SCRIPT_DIR}/run_sharded_inference.py"
  --config_path "${CONFIG_PATH}"
  --checkpoint_path "${CHECKPOINT_PATH}"
  --prompt_path "${PROMPT_PATH}"
  --manifest_path "${MANIFEST_PATH}"
  --output_folder "${VIDEOS_DIR}"
  --gpus "${GPUS}"
  --schedule "${SCHEDULE}"
  --num_output_frames 21
  --python "${PYTHON_BIN}"
)
if [[ "${USE_EMA}" == "1" ]]; then
  INFER_CMD+=(--use_ema)
fi
"${INFER_CMD[@]}"

IFS=',' read -r -a GPU_ARRAY <<< "${GPUS}"
NPROC="${#GPU_ARRAY[@]}"
VBENCH_CMD=(
  "${VBENCH_PYTHON}"
  -m torch.distributed.run
  --standalone
  --nproc_per_node="${NPROC}"
  "${REPO_ROOT}/scripts/run_vbench.py"
  --videos_path "${VIDEOS_DIR}"
  --full_info_path "${FULL_INFO_PATH}"
  --output_dir "${VBENCH_DIR}"
  --name "${NAME}"
  --device cuda
)
if [[ -n "${DIMENSIONS}" ]]; then
  IFS=',' read -r -a DIMENSION_ARRAY <<< "${DIMENSIONS}"
  VBENCH_CMD+=(--dimensions "${DIMENSION_ARRAY[@]}")
fi

mkdir -p "${VBENCH_DIR}"
export CUDA_VISIBLE_DEVICES="${GPUS}"
cd "${REPO_ROOT}"
"${VBENCH_CMD[@]}"

RESULT_PATH="${VBENCH_DIR}/${NAME}_eval_results.json"
if [[ ! -s "${RESULT_PATH}" ]]; then
  echo "VBench result missing or empty: ${RESULT_PATH}" >&2
  exit 1
fi
echo "PASS: ${RESULT_PATH}"
