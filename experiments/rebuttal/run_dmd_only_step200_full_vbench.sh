#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

usage() {
  cat <<'EOF'
Usage:
  bash experiments/rebuttal/run_dmd_only_step200_full_vbench.sh \
    --dmd_checkpoint PATH/checkpoint_model_000200/model.pt \
    --reference_full_result PATH/full_step200_eval_results.json \
    --prompt_path PATH/vbench_prompts.txt \
    --manifest_path PATH/vbench_single_sample_seed0.jsonl \
    --full_info_path PATH/VBench_full_info.json \
    --output_root PATH/dmd_step200_full16 \
    --gpus 0,1,2,3,4,5,6,7 \
    --vbench_python PATH/vbench_python [--python PATH/python]

Runs only the missing DMD-only condition. The reference Full step-200 result
must be raw/no-EMA, complete 16-dimension, one-sample VBench. The final audit
rejects any manifest or resolved FFE-config mismatch.
EOF
}

if [[ $# -eq 1 && ( "$1" == "-h" || "$1" == "--help" ) ]]; then usage; exit 0; fi

DMD_CHECKPOINT=""
REFERENCE_FULL_RESULT=""
PROMPT_PATH=""
MANIFEST_PATH=""
FULL_INFO_PATH=""
OUTPUT_ROOT=""
GPUS=""
VBENCH_PYTHON=""
PYTHON_BIN="${PYTHON_BIN:-python}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dmd_checkpoint) DMD_CHECKPOINT="$2"; shift 2 ;;
    --reference_full_result) REFERENCE_FULL_RESULT="$2"; shift 2 ;;
    --prompt_path) PROMPT_PATH="$2"; shift 2 ;;
    --manifest_path) MANIFEST_PATH="$2"; shift 2 ;;
    --full_info_path) FULL_INFO_PATH="$2"; shift 2 ;;
    --output_root) OUTPUT_ROOT="$2"; shift 2 ;;
    --gpus) GPUS="$2"; shift 2 ;;
    --vbench_python) VBENCH_PYTHON="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 1 ;;
  esac
done
for value_name in DMD_CHECKPOINT REFERENCE_FULL_RESULT PROMPT_PATH MANIFEST_PATH \
  FULL_INFO_PATH OUTPUT_ROOT GPUS VBENCH_PYTHON; do
  if [[ -z "${!value_name}" ]]; then echo "--${value_name,,} is required" >&2; exit 1; fi
done
if [[ ! "${GPUS}" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
  echo "--gpus must be comma-separated non-negative integers" >&2
  exit 1
fi

DMD_CHECKPOINT="$(realpath -m "${DMD_CHECKPOINT}")"
REFERENCE_FULL_RESULT="$(realpath -m "${REFERENCE_FULL_RESULT}")"
PROMPT_PATH="$(realpath -m "${PROMPT_PATH}")"
MANIFEST_PATH="$(realpath -m "${MANIFEST_PATH}")"
FULL_INFO_PATH="$(realpath -m "${FULL_INFO_PATH}")"
OUTPUT_ROOT="$(realpath -m "${OUTPUT_ROOT}")"
VBENCH_PYTHON="$(realpath -m "${VBENCH_PYTHON}")"
for path in "${DMD_CHECKPOINT}" "${REFERENCE_FULL_RESULT}" "${PROMPT_PATH}" \
  "${MANIFEST_PATH}" "${FULL_INFO_PATH}" "${VBENCH_PYTHON}"; do
  if [[ ! -f "${path}" ]]; then echo "Input not found: ${path}" >&2; exit 1; fi
done

mkdir -p "${OUTPUT_ROOT}/audit"
"${PYTHON_BIN}" "${SCRIPT_DIR}/audit_noema_checkpoint.py" \
  --checkpoint "${DMD_CHECKPOINT}" --expected_step 200 \
  --output_path "${OUTPUT_ROOT}/audit/dmd_step200_checkpoint.json"

NAME="dmd_only_step200_ffe_noema_single_sample"
cd "${REPO_ROOT}"
bash "${SCRIPT_DIR}/run_vbench_condition.sh" \
  --name "${NAME}" \
  --config_path "${SCRIPT_DIR}/configs/eval_ffe.yaml" \
  --checkpoint_path "${DMD_CHECKPOINT}" \
  --schedule ffe \
  --prompt_path "${PROMPT_PATH}" \
  --manifest_path "${MANIFEST_PATH}" \
  --full_info_path "${FULL_INFO_PATH}" \
  --output_root "${OUTPUT_ROOT}/${NAME}" \
  --gpus "${GPUS}" \
  --vbench_python "${VBENCH_PYTHON}" \
  --samples_per_prompt 1 \
  --require_no_ema \
  --python "${PYTHON_BIN}"

RESULT_PATH="${OUTPUT_ROOT}/${NAME}/vbench/${NAME}_eval_results.json"
"${PYTHON_BIN}" "${SCRIPT_DIR}/audit_paired_noema_vbench.py" \
  --reference_result "${REFERENCE_FULL_RESULT}" \
  --candidate_result "${RESULT_PATH}" \
  --reference_label full_step200 \
  --candidate_label dmd_only_step200 \
  --comparison_name full_minus_dmd \
  --comparison_direction reference_minus_candidate \
  --output_path "${OUTPUT_ROOT}/dmd_only_step200_full16_summary.json"
