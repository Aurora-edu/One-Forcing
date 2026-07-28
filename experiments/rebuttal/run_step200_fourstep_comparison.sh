#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

usage() {
  cat <<'EOF'
Usage:
  bash experiments/rebuttal/run_step200_fourstep_comparison.sh \
    --checkpoint PATH/step200/model.pt \
    --full_info_path PATH/VBench_full_info.json \
    --output_root PATH/step200_1v4 \
    --gpus 0,1,2,3,4,5,6,7 \
    --vbench_python PATH/vbench_python [--python PATH/python]

Runs the same raw/no-EMA One-Forcing step-200 checkpoint with framewise all1
and framewise all4 schedules, using exactly one generated sample per prompt.
EOF
}

CHECKPOINT=""
FULL_INFO_PATH=""
OUTPUT_ROOT=""
GPUS=""
VBENCH_PYTHON=""
PYTHON_BIN="${PYTHON_BIN:-python}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoint) CHECKPOINT="$2"; shift 2 ;;
    --full_info_path) FULL_INFO_PATH="$2"; shift 2 ;;
    --output_root) OUTPUT_ROOT="$2"; shift 2 ;;
    --gpus) GPUS="$2"; shift 2 ;;
    --vbench_python) VBENCH_PYTHON="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 1 ;;
  esac
done
for value_name in CHECKPOINT FULL_INFO_PATH OUTPUT_ROOT GPUS VBENCH_PYTHON; do
  if [[ -z "${!value_name}" ]]; then
    echo "--${value_name,,} is required" >&2
    exit 1
  fi
done
CHECKPOINT="$(realpath -m "${CHECKPOINT}")"
FULL_INFO_PATH="$(realpath -m "${FULL_INFO_PATH}")"
OUTPUT_ROOT="$(realpath -m "${OUTPUT_ROOT}")"
VBENCH_PYTHON="$(realpath -m "${VBENCH_PYTHON}")"
for path in "${CHECKPOINT}" "${FULL_INFO_PATH}" "${VBENCH_PYTHON}"; do
  if [[ ! -f "${path}" ]]; then echo "Input not found: ${path}" >&2; exit 1; fi
done

mkdir -p "${OUTPUT_ROOT}"
"${PYTHON_BIN}" "${SCRIPT_DIR}/audit_noema_checkpoint.py" \
  --checkpoint "${CHECKPOINT}" --expected_step 200 \
  --output_path "${OUTPUT_ROOT}/step200_checkpoint_audit.json"

PROMPT_PATH="${OUTPUT_ROOT}/manifests/vbench_prompts.txt"
MANIFEST_PATH="${OUTPUT_ROOT}/manifests/vbench_single_sample_seed0.jsonl"
mkdir -p "${OUTPUT_ROOT}/manifests"
"${PYTHON_BIN}" "${SCRIPT_DIR}/prepare_vbench_prompts.py" \
  --full_info_path "${FULL_INFO_PATH}" --output_path "${PROMPT_PATH}"
"${PYTHON_BIN}" "${SCRIPT_DIR}/make_eval_manifest.py" \
  --prompt_path "${PROMPT_PATH}" --output_path "${MANIFEST_PATH}" \
  --base_seed 0 --num_samples_per_prompt 1 --naming vbench

run_condition() {
  local name="$1"
  local config="$2"
  local schedule="$3"
  bash "${SCRIPT_DIR}/run_vbench_condition.sh" \
    --name "${name}" --config_path "${config}" \
    --checkpoint_path "${CHECKPOINT}" --schedule "${schedule}" \
    --prompt_path "${PROMPT_PATH}" --manifest_path "${MANIFEST_PATH}" \
    --full_info_path "${FULL_INFO_PATH}" \
    --output_root "${OUTPUT_ROOT}/${name}" --gpus "${GPUS}" \
    --vbench_python "${VBENCH_PYTHON}" --samples_per_prompt 1 \
    --require_no_ema --python "${PYTHON_BIN}"
}

cd "${REPO_ROOT}"
run_condition step200_all1_noema_single_sample \
  "${SCRIPT_DIR}/configs/eval_all1.yaml" all1
run_condition step200_all4_noema_single_sample \
  "${SCRIPT_DIR}/configs/eval_all4.yaml" all4
"${PYTHON_BIN}" "${SCRIPT_DIR}/summarize_single_seed_vbench.py" \
  --result "all1=${OUTPUT_ROOT}/step200_all1_noema_single_sample/vbench/step200_all1_noema_single_sample_eval_results.json" \
  --result "all4=${OUTPUT_ROOT}/step200_all4_noema_single_sample/vbench/step200_all4_noema_single_sample_eval_results.json" \
  --comparison 'four_step_minus_one_step=all4,all1' \
  --output_path "${OUTPUT_ROOT}/step200_fourstep_comparison.json"
