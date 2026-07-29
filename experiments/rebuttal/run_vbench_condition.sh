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
    --samples_per_prompt 1 \
    --require_no_ema

Options:
  --dimensions CSV   Optional selected VBench dimensions. Omit for all 16.
  --samples_per_prompt N  Generated samples per prompt, from 1 to 5. Default: 5.
  --extended_prompt_path PATH  Optional one-line-per-prompt conditioning rewrites.
  --require_no_ema   Reject --use_ema and audit raw generator provenance.
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
EXTENDED_PROMPT_PATH=""
USE_EMA="0"
REQUIRE_NO_EMA="0"
SAMPLES_PER_PROMPT="5"
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
    --extended_prompt_path) EXTENDED_PROMPT_PATH="$2"; shift 2 ;;
    --samples_per_prompt) SAMPLES_PER_PROMPT="$2"; shift 2 ;;
    --require_no_ema) REQUIRE_NO_EMA="1"; shift ;;
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
if [[ ! "${SAMPLES_PER_PROMPT}" =~ ^[1-5]$ ]]; then
  echo "--samples_per_prompt must be an integer in [1, 5]" >&2
  exit 1
fi
if [[ "${REQUIRE_NO_EMA}" == "1" && "${USE_EMA}" == "1" ]]; then
  echo "--require_no_ema cannot be combined with --use_ema" >&2
  exit 1
fi

CONFIG_PATH="$(realpath -m "${CONFIG_PATH}")"
CHECKPOINT_PATH="$(realpath -m "${CHECKPOINT_PATH}")"
PROMPT_PATH="$(realpath -m "${PROMPT_PATH}")"
MANIFEST_PATH="$(realpath -m "${MANIFEST_PATH}")"
FULL_INFO_PATH="$(realpath -m "${FULL_INFO_PATH}")"
OUTPUT_ROOT="$(realpath -m "${OUTPUT_ROOT}")"
if [[ -n "${EXTENDED_PROMPT_PATH}" ]]; then
  EXTENDED_PROMPT_PATH="$(realpath -m "${EXTENDED_PROMPT_PATH}")"
fi

for path in "${CONFIG_PATH}" "${CHECKPOINT_PATH}" "${PROMPT_PATH}" \
  "${MANIFEST_PATH}" "${FULL_INFO_PATH}" "${VBENCH_PYTHON}"; do
  if [[ ! -f "${path}" ]]; then
    echo "Required file not found: ${path}" >&2
    exit 1
  fi
done
if [[ -n "${EXTENDED_PROMPT_PATH}" && ! -f "${EXTENDED_PROMPT_PATH}" ]]; then
  echo "Extended prompt file not found: ${EXTENDED_PROMPT_PATH}" >&2
  exit 1
fi

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
if [[ -n "${EXTENDED_PROMPT_PATH}" ]]; then
  INFER_CMD+=(--extended_prompt_path "${EXTENDED_PROMPT_PATH}")
fi
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
  --samples_per_prompt "${SAMPLES_PER_PROMPT}"
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
if [[ "${REQUIRE_NO_EMA}" == "1" ]]; then
  "${PYTHON_BIN}" - "${VIDEOS_DIR}/export.done" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
if payload.get("use_ema") is not False or payload.get("weight_source") != "generator":
    raise SystemExit(f"Raw/no-EMA provenance audit failed: {path}: {payload}")
print(f"PASS: raw/no-EMA provenance audited in {path}")
PY
fi
echo "PASS: ${RESULT_PATH}"
