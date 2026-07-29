#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

usage() {
  cat <<'EOF'
Usage:
  bash experiments/rebuttal/run_curvature_cd_small.sh PHASE \
    --data_path PATH/shared_curvature_lmdb \
    --raw_checkpoint PATH/raw_ar_model.pt \
    --output_root PATH/curvature_cd_small \
    --gpus 0,1,2,3,4,5,6,7 \
    [--full_info_path PATH/VBench_full_info.json] \
    [--vbench_python PATH/vbench_python] \
    [--prompt_embedding_cache_path PATH/cache] \
    [--python PATH/python]

PHASE (run in this order):
  preflight        Audit the paired intervention and exact timestep alignment.
  train_curved     Train the curved arm for 300 steps (must run inside tmux).
  train_rectified  Train the rectified arm for 300 steps (must run inside tmux).
  evaluate         One-seed full VBench: all1 first, then aligned all4.
  summarize        Audit no-EMA results and compute gain plus DiD.
  all              Run every phase sequentially; must run inside tmux.
EOF
}

if [[ $# -eq 1 && ( "$1" == "-h" || "$1" == "--help" ) ]]; then
  usage
  exit 0
fi
if [[ $# -lt 1 ]]; then
  usage >&2
  exit 1
fi
PHASE="$1"
shift

DATA_PATH=""
RAW_CHECKPOINT=""
OUTPUT_ROOT=""
GPUS=""
FULL_INFO_PATH=""
VBENCH_PYTHON=""
PROMPT_EMBEDDING_CACHE_PATH=""
PYTHON_BIN="${PYTHON_BIN:-python}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data_path) DATA_PATH="$2"; shift 2 ;;
    --raw_checkpoint) RAW_CHECKPOINT="$2"; shift 2 ;;
    --output_root) OUTPUT_ROOT="$2"; shift 2 ;;
    --gpus) GPUS="$2"; shift 2 ;;
    --full_info_path) FULL_INFO_PATH="$2"; shift 2 ;;
    --vbench_python) VBENCH_PYTHON="$2"; shift 2 ;;
    --prompt_embedding_cache_path) PROMPT_EMBEDDING_CACHE_PATH="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 1 ;;
  esac
done

case "${PHASE}" in
  preflight|train_curved|train_rectified|evaluate|summarize|all) ;;
  *) echo "Unknown phase: ${PHASE}" >&2; usage >&2; exit 1 ;;
esac
for value_name in DATA_PATH RAW_CHECKPOINT OUTPUT_ROOT GPUS; do
  if [[ -z "${!value_name}" ]]; then
    echo "--${value_name,,} is required" >&2
    exit 1
  fi
done
if [[ ! "${GPUS}" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
  echo "--gpus must be comma-separated non-negative integers" >&2
  exit 1
fi

DATA_PATH="$(realpath -m "${DATA_PATH}")"
RAW_CHECKPOINT="$(realpath -m "${RAW_CHECKPOINT}")"
OUTPUT_ROOT="$(realpath -m "${OUTPUT_ROOT}")"
if [[ ! -f "${DATA_PATH}/data.mdb" ]]; then
  echo "Paired curvature LMDB not found: ${DATA_PATH}/data.mdb" >&2
  exit 1
fi
if [[ ! -f "${DATA_PATH}/curvature_dataset_manifest.json" ]]; then
  echo "Curvature manifest not found in ${DATA_PATH}" >&2
  exit 1
fi
if [[ ! -f "${RAW_CHECKPOINT}" ]]; then
  echo "Raw initialization checkpoint not found: ${RAW_CHECKPOINT}" >&2
  exit 1
fi
if [[ -n "${PROMPT_EMBEDDING_CACHE_PATH}" ]]; then
  PROMPT_EMBEDDING_CACHE_PATH="$(realpath -m "${PROMPT_EMBEDDING_CACHE_PATH}")"
  if [[ ! -f "${PROMPT_EMBEDDING_CACHE_PATH}/data.mdb" ]]; then
    echo "Prompt embedding cache not found: ${PROMPT_EMBEDDING_CACHE_PATH}/data.mdb" >&2
    exit 1
  fi
fi
IFS=',' read -r -a GPU_ARRAY <<< "${GPUS}"
NPROC="${#GPU_ARRAY[@]}"

TRAIN_CONFIG="${SCRIPT_DIR}/configs/train_curvature_cd_small.yaml"
EVAL_ALL1_CONFIG="${SCRIPT_DIR}/configs/eval_all1.yaml"
EVAL_ALL4_CONFIG="${SCRIPT_DIR}/configs/eval_all4.yaml"
CURVED_DIR="${OUTPUT_ROOT}/training/curved"
RECTIFIED_DIR="${OUTPUT_ROOT}/training/rectified"
CURVED_CKPT="${CURVED_DIR}/checkpoint_model_000300/model.pt"
RECTIFIED_CKPT="${RECTIFIED_DIR}/checkpoint_model_000300/model.pt"
PROMPT_PATH="${OUTPUT_ROOT}/manifests/vbench_prompts.txt"
MANIFEST_PATH="${OUTPUT_ROOT}/manifests/vbench_single_sample_seed0.jsonl"

require_tmux() {
  if [[ -z "${TMUX:-}" ]]; then
    echo "Curvature training must run inside tmux." >&2
    exit 1
  fi
}

preflight() {
  mkdir -p "${OUTPUT_ROOT}/audit"
  "${PYTHON_BIN}" "${SCRIPT_DIR}/analyze_curvature_intervention.py" \
    --data_path "${DATA_PATH}" \
    --output_path "${OUTPUT_ROOT}/audit/paired_intervention.json" \
    --limit 64
  "${PYTHON_BIN}" "${SCRIPT_DIR}/audit_curvature_cd_design.py" \
    --data_path "${DATA_PATH}" \
    --train_config "${TRAIN_CONFIG}" \
    --eval_all1_config "${EVAL_ALL1_CONFIG}" \
    --eval_all4_config "${EVAL_ALL4_CONFIG}" \
    --output_path "${OUTPUT_ROOT}/audit/adjacent_cd_design.json" \
    --limit 64
}

train_arm() {
  local intervention="$1"
  local output_dir="$2"
  require_tmux
  local train_command=(
    "${PYTHON_BIN}" -m torch.distributed.run
    --standalone
    --nproc_per_node="${NPROC}"
    "${SCRIPT_DIR}/train_curvature_cd.py"
    --config_path "${TRAIN_CONFIG}"
    --data_path "${DATA_PATH}"
    --generator_ckpt "${RAW_CHECKPOINT}"
    --output_dir "${output_dir}"
    --intervention "${intervention}"
  )
  if [[ -n "${PROMPT_EMBEDDING_CACHE_PATH}" ]]; then
    train_command+=(--prompt_embedding_cache_path "${PROMPT_EMBEDDING_CACHE_PATH}")
  fi
  CUDA_VISIBLE_DEVICES="${GPUS}" "${train_command[@]}"
}

require_evaluation_inputs() {
  if [[ -z "${FULL_INFO_PATH}" || -z "${VBENCH_PYTHON}" ]]; then
    echo "evaluate/all requires --full_info_path and --vbench_python" >&2
    exit 1
  fi
  FULL_INFO_PATH="$(realpath -m "${FULL_INFO_PATH}")"
  VBENCH_PYTHON="$(realpath -m "${VBENCH_PYTHON}")"
  for path in "${FULL_INFO_PATH}" "${VBENCH_PYTHON}" "${CURVED_CKPT}" "${RECTIFIED_CKPT}"; do
    if [[ ! -f "${path}" ]]; then
      echo "Evaluation input not found: ${path}" >&2
      exit 1
    fi
  done
  "${PYTHON_BIN}" "${SCRIPT_DIR}/audit_curvature_cd_arms.py" \
    --curved_done "${CURVED_DIR}/training.done" \
    --rectified_done "${RECTIFIED_DIR}/training.done" \
    --output_path "${OUTPUT_ROOT}/audit/completed_arm_pairing.json"
  "${PYTHON_BIN}" "${SCRIPT_DIR}/audit_noema_checkpoint.py" \
    --checkpoint "${CURVED_CKPT}" --expected_step 300 \
    --output_path "${OUTPUT_ROOT}/audit/curved_checkpoint_noema.json"
  "${PYTHON_BIN}" "${SCRIPT_DIR}/audit_noema_checkpoint.py" \
    --checkpoint "${RECTIFIED_CKPT}" --expected_step 300 \
    --output_path "${OUTPUT_ROOT}/audit/rectified_checkpoint_noema.json"
}

prepare_manifest() {
  mkdir -p "${OUTPUT_ROOT}/manifests"
  "${PYTHON_BIN}" "${SCRIPT_DIR}/prepare_vbench_prompts.py" \
    --full_info_path "${FULL_INFO_PATH}" \
    --output_path "${PROMPT_PATH}"
  "${PYTHON_BIN}" "${SCRIPT_DIR}/make_eval_manifest.py" \
    --prompt_path "${PROMPT_PATH}" \
    --output_path "${MANIFEST_PATH}" \
    --base_seed 0 \
    --num_samples_per_prompt 1 \
    --naming vbench
}

run_condition() {
  local name="$1"
  local config="$2"
  local checkpoint="$3"
  local schedule="$4"
  local command=(
    bash "${SCRIPT_DIR}/run_vbench_condition.sh"
    --name "${name}"
    --config_path "${config}"
    --checkpoint_path "${checkpoint}"
    --schedule "${schedule}"
    --prompt_path "${PROMPT_PATH}"
    --manifest_path "${MANIFEST_PATH}"
    --full_info_path "${FULL_INFO_PATH}"
    --output_root "${OUTPUT_ROOT}/vbench/${name}"
    --gpus "${GPUS}"
    --vbench_python "${VBENCH_PYTHON}"
    --samples_per_prompt 1
    --require_no_ema
    --python "${PYTHON_BIN}"
  )
  if [[ -n "${PROMPT_EMBEDDING_CACHE_PATH}" ]]; then
    # run_vbench_condition currently has no cache flag; inference uses local T5.
    echo "Training prompt cache is not reused by the standard VBench wrapper." >&2
  fi
  "${command[@]}"
}

evaluate() {
  require_evaluation_inputs
  prepare_manifest
  # Primary comparison first, so the causal answer is available before all4 finishes.
  run_condition curved_cd_all1_noema_seed0 \
    "${EVAL_ALL1_CONFIG}" "${CURVED_CKPT}" all1
  run_condition rectified_cd_all1_noema_seed0 \
    "${EVAL_ALL1_CONFIG}" "${RECTIFIED_CKPT}" all1
  run_condition curved_cd_all4_noema_seed0 \
    "${EVAL_ALL4_CONFIG}" "${CURVED_CKPT}" all4
  run_condition rectified_cd_all4_noema_seed0 \
    "${EVAL_ALL4_CONFIG}" "${RECTIFIED_CKPT}" all4
}

result_path() {
  local name="$1"
  echo "${OUTPUT_ROOT}/vbench/${name}/vbench/${name}_eval_results.json"
}

summarize() {
  "${PYTHON_BIN}" "${SCRIPT_DIR}/summarize_single_seed_vbench.py" \
    --result "curved_all1=$(result_path curved_cd_all1_noema_seed0)" \
    --result "rectified_all1=$(result_path rectified_cd_all1_noema_seed0)" \
    --result "curved_all4=$(result_path curved_cd_all4_noema_seed0)" \
    --result "rectified_all4=$(result_path rectified_cd_all4_noema_seed0)" \
    --comparison 'rectification_gain_all1=rectified_all1,curved_all1' \
    --comparison 'rectification_gain_all4=rectified_all4,curved_all4' \
    --difference_in_differences \
      'curvature_causal_effect=rectified_all1,curved_all1,rectified_all4,curved_all4' \
    --output_path "${OUTPUT_ROOT}/curvature_cd_small_summary.json"
}

cd "${REPO_ROOT}"
case "${PHASE}" in
  preflight) preflight ;;
  train_curved) train_arm curved "${CURVED_DIR}" ;;
  train_rectified) train_arm rectified "${RECTIFIED_DIR}" ;;
  evaluate) evaluate ;;
  summarize) summarize ;;
  all)
    require_tmux
    preflight
    train_arm curved "${CURVED_DIR}"
    train_arm rectified "${RECTIFIED_DIR}"
    evaluate
    summarize
    ;;
esac
