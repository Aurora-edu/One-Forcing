#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  cat <<'EOF'
Usage:
  bash experiments/rebuttal/run_paper_aligned_vbench.sh \
    --name published_ffe \
    --checkpoint_path checkpoints/one_forcing.pt \
    --schedule ffe \
    --output_root eval/paper_aligned/published_ffe \
    --gpus all \
    --vbench_python /path/to/vbench/bin/python \
    [--python /path/to/one_forcing/bin/python] \
    [--model one_forcing|self_forcing] \
    [--samples_per_prompt 5] [--prepare_only]

Uses the pinned 944 original VBench prompts for filenames/scoring and their
exact paired Qwen rewrites for text conditioning. One-Forcing uses its raw
generator; --model self_forcing explicitly selects its released EMA weights
and all4 schedule. Both use 21 latent frames, 16 fps, no attention sink,
and all 16 dimensions. The same seed rule is
used across every invocation, so conditions are paired. Five samples are the
official VBench protocol; one sample is a labeled follow-up protocol.
EOF
}

NAME=""
CHECKPOINT_PATH=""
SCHEDULE="ffe"
MODEL="one_forcing"
OUTPUT_ROOT=""
GPUS="all"
VBENCH_PYTHON=""
PYTHON_BIN="${PYTHON_BIN:-python}"
SAMPLES_PER_PROMPT="5"
PREPARE_ONLY="0"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --name) NAME="$2"; shift 2 ;;
    --checkpoint_path) CHECKPOINT_PATH="$2"; shift 2 ;;
    --schedule) SCHEDULE="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --output_root) OUTPUT_ROOT="$2"; shift 2 ;;
    --gpus) GPUS="$2"; shift 2 ;;
    --vbench_python) VBENCH_PYTHON="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    --samples_per_prompt) SAMPLES_PER_PROMPT="$2"; shift 2 ;;
    --prepare_only) PREPARE_ONLY="1"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 1 ;;
  esac
done
if [[ -z "${NAME}" || -z "${OUTPUT_ROOT}" ]]; then
  usage >&2
  exit 1
fi
if [[ ! "${NAME}" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  echo "Invalid --name" >&2
  exit 1
fi
if [[ "${SAMPLES_PER_PROMPT}" != "1" && "${SAMPLES_PER_PROMPT}" != "5" ]]; then
  echo "--samples_per_prompt must be 1 or 5" >&2
  exit 1
fi
case "${SCHEDULE}" in
  ffe) CONFIG_PATH="${SCRIPT_DIR}/configs/eval_ffe.yaml" ;;
  all1) CONFIG_PATH="${SCRIPT_DIR}/configs/eval_all1.yaml" ;;
  all4) CONFIG_PATH="${SCRIPT_DIR}/configs/eval_all4.yaml" ;;
  *) echo "Invalid --schedule: ${SCHEDULE}" >&2; exit 1 ;;
esac
case "${MODEL}" in
  one_forcing) WEIGHT_SOURCE="generator" ;;
  self_forcing)
    if [[ "${SCHEDULE}" != "all4" ]]; then
      echo "Released Self-Forcing baseline must use --schedule all4" >&2
      exit 1
    fi
    WEIGHT_SOURCE="generator_ema"
    ;;
  *) echo "Invalid --model: ${MODEL}" >&2; exit 1 ;;
esac
if [[ "${MODEL}" == "self_forcing" ]]; then
  CONFIG_PATH="${SCRIPT_DIR}/../../self_forcing_config.yaml"
  METHOD="chunkwise"
else
  METHOD="framewise"
fi

OUTPUT_ROOT="$(realpath -m "${OUTPUT_ROOT}")"
INPUT_DIR="${OUTPUT_ROOT}/manifests"
"${PYTHON_BIN}" "${SCRIPT_DIR}/paper_vbench_protocol.py" prepare \
  --output_dir "${INPUT_DIR}" --samples_per_prompt "${SAMPLES_PER_PROMPT}"
if [[ "${PREPARE_ONLY}" == "1" ]]; then
  exit 0
fi
if [[ -z "${CHECKPOINT_PATH}" || -z "${VBENCH_PYTHON}" ]]; then
  echo "--checkpoint_path and --vbench_python are required for a run" >&2
  exit 1
fi
CHECKPOINT_PATH="$(realpath -m "${CHECKPOINT_PATH}")"
if [[ ! -f "${CHECKPOINT_PATH}" || ! -f "${VBENCH_PYTHON}" ]]; then
  echo "Checkpoint or VBench Python not found" >&2
  exit 1
fi
if [[ "${GPUS}" == "all" ]]; then
  mkdir -p "${OUTPUT_ROOT}/audit"
  GPUS="$("${PYTHON_BIN}" "${SCRIPT_DIR}/resolve_all_gpus.py" \
    --requested all --require_idle \
    --output_path "${OUTPUT_ROOT}/audit/gpu_inventory.json")"
fi

RUN_CMD=(
  bash "${SCRIPT_DIR}/run_vbench_condition.sh"
  --name "${NAME}"
  --config_path "${CONFIG_PATH}"
  --checkpoint_path "${CHECKPOINT_PATH}"
  --method "${METHOD}"
  --schedule "${SCHEDULE}"
  --prompt_path "${INPUT_DIR}/vbench_prompts.txt"
  --extended_prompt_path "${INPUT_DIR}/vbench_qwen_rewrites.txt"
  --manifest_path "${INPUT_DIR}/vbench_qwen_${SAMPLES_PER_PROMPT}sample_seed0.jsonl"
  --full_info_path "${SCRIPT_DIR}/assets/qwen_vbench/VBench_full_info.json"
  --output_root "${OUTPUT_ROOT}"
  --gpus "${GPUS}"
  --vbench_python "${VBENCH_PYTHON}"
  --samples_per_prompt "${SAMPLES_PER_PROMPT}"
  --python "${PYTHON_BIN}"
)
if [[ "${WEIGHT_SOURCE}" == "generator" ]]; then
  RUN_CMD+=(--require_no_ema)
else
  RUN_CMD+=(--use_ema)
fi
"${RUN_CMD[@]}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/paper_vbench_protocol.py" audit_run \
  --output_root "${OUTPUT_ROOT}" --name "${NAME}" \
  --schedule "${SCHEDULE}" --samples_per_prompt "${SAMPLES_PER_PROMPT}" \
  --weight_source "${WEIGHT_SOURCE}"
