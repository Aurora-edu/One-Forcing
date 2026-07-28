#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

usage() {
  cat <<'EOF'
Usage:
  bash experiments/rebuttal/run_full_vbench_noema_single_seed.sh \
    --step200_checkpoint PATH/step200/model.pt \
    --step400_checkpoint PATH/step400/model.pt \
    --full_info_path PATH/VBench_full_info.json \
    --output_root PATH/full_vbench_raw \
    --gpus 0,1,2,3,4,5,6,7 \
    --vbench_python PATH/vbench_python [--python PATH/python]

Runs complete 16-dimension VBench with the paper's FFE schedule for the raw
step-200 and step-400 checkpoints. Each prompt has one generated sample.
EOF
}

STEP200=""
STEP400=""
FULL_INFO_PATH=""
OUTPUT_ROOT=""
GPUS=""
VBENCH_PYTHON=""
PYTHON_BIN="${PYTHON_BIN:-python}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --step200_checkpoint) STEP200="$2"; shift 2 ;;
    --step400_checkpoint) STEP400="$2"; shift 2 ;;
    --full_info_path) FULL_INFO_PATH="$2"; shift 2 ;;
    --output_root) OUTPUT_ROOT="$2"; shift 2 ;;
    --gpus) GPUS="$2"; shift 2 ;;
    --vbench_python) VBENCH_PYTHON="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 1 ;;
  esac
done
for value_name in STEP200 STEP400 FULL_INFO_PATH OUTPUT_ROOT GPUS VBENCH_PYTHON; do
  if [[ -z "${!value_name}" ]]; then
    echo "--${value_name,,} is required" >&2
    exit 1
  fi
done
STEP200="$(realpath -m "${STEP200}")"
STEP400="$(realpath -m "${STEP400}")"
FULL_INFO_PATH="$(realpath -m "${FULL_INFO_PATH}")"
OUTPUT_ROOT="$(realpath -m "${OUTPUT_ROOT}")"
VBENCH_PYTHON="$(realpath -m "${VBENCH_PYTHON}")"
for path in "${STEP200}" "${STEP400}" "${FULL_INFO_PATH}" "${VBENCH_PYTHON}"; do
  if [[ ! -f "${path}" ]]; then echo "Input not found: ${path}" >&2; exit 1; fi
done

mkdir -p "${OUTPUT_ROOT}"
"${PYTHON_BIN}" "${SCRIPT_DIR}/audit_noema_checkpoint.py" \
  --checkpoint "${STEP200}" --expected_step 200 \
  --output_path "${OUTPUT_ROOT}/step200_checkpoint_audit.json"
"${PYTHON_BIN}" "${SCRIPT_DIR}/audit_noema_checkpoint.py" \
  --checkpoint "${STEP400}" --expected_step 400 \
  --output_path "${OUTPUT_ROOT}/step400_checkpoint_audit.json"

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
  local checkpoint="$2"
  bash "${SCRIPT_DIR}/run_vbench_condition.sh" \
    --name "${name}" --config_path "${SCRIPT_DIR}/configs/eval_ffe.yaml" \
    --checkpoint_path "${checkpoint}" --schedule ffe \
    --prompt_path "${PROMPT_PATH}" --manifest_path "${MANIFEST_PATH}" \
    --full_info_path "${FULL_INFO_PATH}" \
    --output_root "${OUTPUT_ROOT}/${name}" --gpus "${GPUS}" \
    --vbench_python "${VBENCH_PYTHON}" --samples_per_prompt 1 \
    --require_no_ema --python "${PYTHON_BIN}"
}

cd "${REPO_ROOT}"
run_condition step200_ffe_noema_single_sample "${STEP200}"
run_condition step400_ffe_noema_single_sample "${STEP400}"
"${PYTHON_BIN}" "${SCRIPT_DIR}/summarize_single_seed_vbench.py" \
  --result "step200=${OUTPUT_ROOT}/step200_ffe_noema_single_sample/vbench/step200_ffe_noema_single_sample_eval_results.json" \
  --result "step400=${OUTPUT_ROOT}/step400_ffe_noema_single_sample/vbench/step400_ffe_noema_single_sample_eval_results.json" \
  --comparison 'step400_minus_step200=step400,step200' \
  --output_path "${OUTPUT_ROOT}/raw_step200_step400_full_vbench.json"
