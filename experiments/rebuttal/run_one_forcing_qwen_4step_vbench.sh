#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

usage() {
  cat <<'EOF'
Usage:
  bash experiments/rebuttal/run_one_forcing_qwen_4step_vbench.sh \
    --one_forcing_checkpoint PATH/of4_step300/model.pt \
    --self_forcing_checkpoint PATH/self_forcing_dmd.pt \
    --qwen_pair_shard0 PATH/shard00_pairs.jsonl \
    --qwen_pair_shard1 PATH/shard01_pairs.jsonl \
    --full_info_path PATH/VBench_full_info.json \
    --output_root PATH/qwen_matched_4step_all_gpu \
    --gpus all \
    --vbench_python PATH/vbench_python [--python PATH/python]

Regenerates both methods with one shared per-record seed manifest. One-Forcing
uses raw generator weights; released Self-Forcing uses generator_ema. The
runner detects and requires every physical GPU on the host, then uses all of
them for generation and VBench scoring. No historical videos are reused.
EOF
}

if [[ $# -eq 1 && ( "$1" == "-h" || "$1" == "--help" ) ]]; then usage; exit 0; fi

ONE_FORCING_CHECKPOINT=""
SELF_FORCING_CHECKPOINT=""
QWEN_PAIR_SHARD0=""
QWEN_PAIR_SHARD1=""
FULL_INFO_PATH=""
OUTPUT_ROOT=""
GPUS="all"
VBENCH_PYTHON=""
PYTHON_BIN="${PYTHON_BIN:-python}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --one_forcing_checkpoint) ONE_FORCING_CHECKPOINT="$2"; shift 2 ;;
    --self_forcing_checkpoint) SELF_FORCING_CHECKPOINT="$2"; shift 2 ;;
    --qwen_pair_shard0) QWEN_PAIR_SHARD0="$2"; shift 2 ;;
    --qwen_pair_shard1) QWEN_PAIR_SHARD1="$2"; shift 2 ;;
    --full_info_path) FULL_INFO_PATH="$2"; shift 2 ;;
    --output_root) OUTPUT_ROOT="$2"; shift 2 ;;
    --gpus) GPUS="$2"; shift 2 ;;
    --vbench_python) VBENCH_PYTHON="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 1 ;;
  esac
done
for value_name in ONE_FORCING_CHECKPOINT SELF_FORCING_CHECKPOINT \
  QWEN_PAIR_SHARD0 QWEN_PAIR_SHARD1 FULL_INFO_PATH OUTPUT_ROOT VBENCH_PYTHON; do
  if [[ -z "${!value_name}" ]]; then echo "--${value_name,,} is required" >&2; exit 1; fi
done

ONE_FORCING_CHECKPOINT="$(realpath -m "${ONE_FORCING_CHECKPOINT}")"
SELF_FORCING_CHECKPOINT="$(realpath -m "${SELF_FORCING_CHECKPOINT}")"
QWEN_PAIR_SHARD0="$(realpath -m "${QWEN_PAIR_SHARD0}")"
QWEN_PAIR_SHARD1="$(realpath -m "${QWEN_PAIR_SHARD1}")"
FULL_INFO_PATH="$(realpath -m "${FULL_INFO_PATH}")"
OUTPUT_ROOT="$(realpath -m "${OUTPUT_ROOT}")"
VBENCH_PYTHON="$(realpath -m "${VBENCH_PYTHON}")"
for path in "${ONE_FORCING_CHECKPOINT}" "${SELF_FORCING_CHECKPOINT}" \
  "${QWEN_PAIR_SHARD0}" "${QWEN_PAIR_SHARD1}" "${FULL_INFO_PATH}" \
  "${VBENCH_PYTHON}"; do
  if [[ ! -f "${path}" ]]; then echo "Input not found: ${path}" >&2; exit 1; fi
done

mkdir -p "${OUTPUT_ROOT}/audit" "${OUTPUT_ROOT}/manifests"
GPU_AUDIT="${OUTPUT_ROOT}/audit/gpu_inventory.json"
GPUS="$("${PYTHON_BIN}" "${SCRIPT_DIR}/resolve_all_gpus.py" \
  --requested "${GPUS}" --require_idle --output_path "${GPU_AUDIT}")"

"${PYTHON_BIN}" "${SCRIPT_DIR}/audit_noema_checkpoint.py" \
  --checkpoint "${ONE_FORCING_CHECKPOINT}" --expected_step 300 \
  --output_path "${OUTPUT_ROOT}/audit/one_forcing_step300_noema.json"
"${PYTHON_BIN}" "${SCRIPT_DIR}/audit_ema_checkpoint.py" \
  --checkpoint "${SELF_FORCING_CHECKPOINT}" \
  --output_path "${OUTPUT_ROOT}/audit/self_forcing_ema_checkpoint.json"

PROMPT_PATH="${OUTPUT_ROOT}/manifests/vbench_prompts.txt"
QWEN_REWRITE_PATH="${OUTPUT_ROOT}/manifests/qwen_rewrites_official_order.txt"
MANIFEST_PATH="${OUTPUT_ROOT}/manifests/qwen_single_sample_seed0.jsonl"
"${PYTHON_BIN}" "${SCRIPT_DIR}/prepare_vbench_prompts.py" \
  --full_info_path "${FULL_INFO_PATH}" --output_path "${PROMPT_PATH}"
"${PYTHON_BIN}" "${SCRIPT_DIR}/merge_qwen_rewrite_shards.py" \
  --prompt_path "${PROMPT_PATH}" \
  --pair_shard "${QWEN_PAIR_SHARD0}" \
  --pair_shard "${QWEN_PAIR_SHARD1}" \
  --output_path "${QWEN_REWRITE_PATH}"
"${PYTHON_BIN}" "${SCRIPT_DIR}/make_eval_manifest.py" \
  --prompt_path "${PROMPT_PATH}" \
  --extended_prompt_path "${QWEN_REWRITE_PATH}" \
  --output_path "${MANIFEST_PATH}" \
  --base_seed 0 --num_samples_per_prompt 1 --naming vbench
"${PYTHON_BIN}" "${SCRIPT_DIR}/audit_qwen_vbench_inputs.py" \
  --prompt_path "${PROMPT_PATH}" \
  --qwen_rewrite_path "${QWEN_REWRITE_PATH}" \
  --manifest_path "${MANIFEST_PATH}" \
  --output_path "${OUTPUT_ROOT}/audit/qwen_prompt_seed_manifest.json"

run_condition() {
  local name="$1"
  local checkpoint="$2"
  local weight_mode="$3"
  local command=(
    bash "${SCRIPT_DIR}/run_vbench_condition.sh"
    --name "${name}"
    --config_path "${SCRIPT_DIR}/configs/eval_all4.yaml"
    --checkpoint_path "${checkpoint}"
    --schedule all4
    --prompt_path "${PROMPT_PATH}"
    --extended_prompt_path "${QWEN_REWRITE_PATH}"
    --manifest_path "${MANIFEST_PATH}"
    --full_info_path "${FULL_INFO_PATH}"
    --output_root "${OUTPUT_ROOT}/${name}"
    --gpus "${GPUS}"
    --vbench_python "${VBENCH_PYTHON}"
    --samples_per_prompt 1
    --python "${PYTHON_BIN}"
  )
  if [[ "${weight_mode}" == "raw" ]]; then
    command+=(--require_no_ema)
  elif [[ "${weight_mode}" == "ema" ]]; then
    command+=(--use_ema)
  else
    echo "Internal error: invalid weight mode ${weight_mode}" >&2
    exit 1
  fi
  "${command[@]}"
}

cd "${REPO_ROOT}"
OF_NAME="one_forcing_raw_noema_all4_qwen_paired"
SF_NAME="self_forcing_ema_all4_qwen_paired"
run_condition "${OF_NAME}" "${ONE_FORCING_CHECKPOINT}" raw
run_condition "${SF_NAME}" "${SELF_FORCING_CHECKPOINT}" ema

OF_RESULT="${OUTPUT_ROOT}/${OF_NAME}/vbench/${OF_NAME}_eval_results.json"
SF_RESULT="${OUTPUT_ROOT}/${SF_NAME}/vbench/${SF_NAME}_eval_results.json"
"${PYTHON_BIN}" "${SCRIPT_DIR}/summarize_qwen_4step_comparison.py" \
  --one_forcing_result "${OF_RESULT}" \
  --self_forcing_result "${SF_RESULT}" \
  --prompt_path "${PROMPT_PATH}" \
  --qwen_rewrite_path "${QWEN_REWRITE_PATH}" \
  --manifest_path "${MANIFEST_PATH}" \
  --gpu_audit "${GPU_AUDIT}" \
  --output_path "${OUTPUT_ROOT}/qwen_matched_4step_summary.json"
