#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

usage() {
  cat <<'EOF'
Usage:
  bash experiments/rebuttal/run_curvature_causal_experiment.sh PHASE \
    --ar_checkpoint PATH/raw_ar_model.pt \
    --clean_lmdb PATH/clean_latents_lmdb \
    --output_root PATH/curvature_control \
    --gpus 0,1,2,3,4,5,6,7 \
    [--full_info_path PATH/VBench_full_info.json] \
    [--vbench_python PATH/vbench_python] \
    [--python PATH/python] [--trajectory_limit N]

PHASE (run in this order):
  prepare          Generate raw/no-EMA paired trajectories and verify rectification.
  train_curved     Train the unmodified-trajectory arm (must run inside tmux).
  train_rectified  Train the rectified-trajectory arm (must run inside tmux).
  evaluate         Full 16-dimension, one-sample VBench for both arms at all1/all4.
  summarize        Audit and compute rectification gains and difference-in-differences.
  all              Run every phase sequentially; must run inside tmux.
EOF
}

if [[ $# -lt 1 ]]; then
  usage >&2
  exit 1
fi
PHASE="$1"
shift

AR_CHECKPOINT=""
CLEAN_LMDB=""
OUTPUT_ROOT=""
GPUS=""
FULL_INFO_PATH=""
VBENCH_PYTHON=""
PYTHON_BIN="${PYTHON_BIN:-python}"
TRAJECTORY_LIMIT="-1"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ar_checkpoint) AR_CHECKPOINT="$2"; shift 2 ;;
    --clean_lmdb) CLEAN_LMDB="$2"; shift 2 ;;
    --output_root) OUTPUT_ROOT="$2"; shift 2 ;;
    --gpus) GPUS="$2"; shift 2 ;;
    --full_info_path) FULL_INFO_PATH="$2"; shift 2 ;;
    --vbench_python) VBENCH_PYTHON="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    --trajectory_limit) TRAJECTORY_LIMIT="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 1 ;;
  esac
done

case "${PHASE}" in
  prepare|train_curved|train_rectified|evaluate|summarize|all) ;;
  *) echo "Unknown phase: ${PHASE}" >&2; usage >&2; exit 1 ;;
esac
for value_name in AR_CHECKPOINT CLEAN_LMDB OUTPUT_ROOT GPUS; do
  if [[ -z "${!value_name}" ]]; then
    echo "--${value_name,,} is required" >&2
    exit 1
  fi
done
if [[ ! "${TRAJECTORY_LIMIT}" =~ ^-1$|^[1-9][0-9]*$ ]]; then
  echo "--trajectory_limit must be -1 or a positive integer" >&2
  exit 1
fi

AR_CHECKPOINT="$(realpath -m "${AR_CHECKPOINT}")"
CLEAN_LMDB="$(realpath -m "${CLEAN_LMDB}")"
OUTPUT_ROOT="$(realpath -m "${OUTPUT_ROOT}")"
if [[ ! -f "${AR_CHECKPOINT}" ]]; then
  echo "Raw AR checkpoint not found: ${AR_CHECKPOINT}" >&2
  exit 1
fi
if [[ ! -f "${CLEAN_LMDB}/data.mdb" ]]; then
  echo "Clean latent LMDB not found: ${CLEAN_LMDB}/data.mdb" >&2
  exit 1
fi
if [[ ! "${GPUS}" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
  echo "--gpus must be comma-separated non-negative integers" >&2
  exit 1
fi
IFS=',' read -r -a GPU_ARRAY <<< "${GPUS}"
NPROC="${#GPU_ARRAY[@]}"

TRAJECTORY_DIR="${OUTPUT_ROOT}/data/curved_trajectories_raw_noema"
DATA_LMDB="${OUTPUT_ROOT}/data/shared_curvature_lmdb"
ANALYSIS_PATH="${OUTPUT_ROOT}/data/paired_intervention_audit.json"
TRAIN_CONFIG="${SCRIPT_DIR}/configs/train_curvature_ode.yaml"
CURVED_DIR="${OUTPUT_ROOT}/training/curved"
RECTIFIED_DIR="${OUTPUT_ROOT}/training/rectified"
CURVED_CKPT="${CURVED_DIR}/checkpoint_model_001000/model.pt"
RECTIFIED_CKPT="${RECTIFIED_DIR}/checkpoint_model_001000/model.pt"
PROMPT_PATH="${OUTPUT_ROOT}/manifests/vbench_prompts.txt"
MANIFEST_PATH="${OUTPUT_ROOT}/manifests/vbench_single_sample_seed0.jsonl"

require_tmux() {
  if [[ -z "${TMUX:-}" ]]; then
    echo "Long curvature training must run inside tmux." >&2
    exit 1
  fi
}

prepare() {
  mkdir -p "${OUTPUT_ROOT}/data"
  CUDA_VISIBLE_DEVICES="${GPUS}" "${PYTHON_BIN}" -m torch.distributed.run \
    --standalone --nproc_per_node="${NPROC}" \
    "${REPO_ROOT}/get_causal_ode_data_framewise.py" \
    --output_folder "${TRAJECTORY_DIR}" \
    --rawdata_path "${CLEAN_LMDB}" \
    --generator_ckpt "${AR_CHECKPOINT}" \
    --guidance_scale 6.0 \
    --seed 0 \
    --limit "${TRAJECTORY_LIMIT}" \
    --require_no_ema
  "${PYTHON_BIN}" "${SCRIPT_DIR}/build_curvature_dataset.py" \
    --trajectory_dir "${TRAJECTORY_DIR}" \
    --output_lmdb "${DATA_LMDB}"
  "${PYTHON_BIN}" "${SCRIPT_DIR}/analyze_curvature_intervention.py" \
    --data_path "${DATA_LMDB}" \
    --output_path "${ANALYSIS_PATH}" \
    --limit 64
}

train_arm() {
  local intervention="$1"
  local output_dir="$2"
  require_tmux
  CUDA_VISIBLE_DEVICES="${GPUS}" "${PYTHON_BIN}" -m torch.distributed.run \
    --standalone --nproc_per_node="${NPROC}" \
    "${SCRIPT_DIR}/train_curvature_ode.py" \
    --config_path "${TRAIN_CONFIG}" \
    --data_path "${DATA_LMDB}" \
    --generator_ckpt "${AR_CHECKPOINT}" \
    --output_dir "${output_dir}" \
    --intervention "${intervention}"
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
  "${PYTHON_BIN}" "${SCRIPT_DIR}/audit_noema_checkpoint.py" \
    --checkpoint "${CURVED_CKPT}" --expected_step 1000 \
    --output_path "${OUTPUT_ROOT}/training/curved_checkpoint_audit.json"
  "${PYTHON_BIN}" "${SCRIPT_DIR}/audit_noema_checkpoint.py" \
    --checkpoint "${RECTIFIED_CKPT}" --expected_step 1000 \
    --output_path "${OUTPUT_ROOT}/training/rectified_checkpoint_audit.json"
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
  bash "${SCRIPT_DIR}/run_vbench_condition.sh" \
    --name "${name}" \
    --config_path "${config}" \
    --checkpoint_path "${checkpoint}" \
    --schedule "${schedule}" \
    --prompt_path "${PROMPT_PATH}" \
    --manifest_path "${MANIFEST_PATH}" \
    --full_info_path "${FULL_INFO_PATH}" \
    --output_root "${OUTPUT_ROOT}/vbench/${name}" \
    --gpus "${GPUS}" \
    --vbench_python "${VBENCH_PYTHON}" \
    --samples_per_prompt 1 \
    --require_no_ema \
    --python "${PYTHON_BIN}"
}

evaluate() {
  require_evaluation_inputs
  prepare_manifest
  run_condition curved_all1_noema_single_sample \
    "${SCRIPT_DIR}/configs/eval_all1.yaml" "${CURVED_CKPT}" all1
  run_condition curved_all4_noema_single_sample \
    "${SCRIPT_DIR}/configs/eval_all4.yaml" "${CURVED_CKPT}" all4
  run_condition rectified_all1_noema_single_sample \
    "${SCRIPT_DIR}/configs/eval_all1.yaml" "${RECTIFIED_CKPT}" all1
  run_condition rectified_all4_noema_single_sample \
    "${SCRIPT_DIR}/configs/eval_all4.yaml" "${RECTIFIED_CKPT}" all4
}

result_path() {
  local name="$1"
  echo "${OUTPUT_ROOT}/vbench/${name}/vbench/${name}_eval_results.json"
}

summarize() {
  "${PYTHON_BIN}" "${SCRIPT_DIR}/summarize_single_seed_vbench.py" \
    --result "curved_all1=$(result_path curved_all1_noema_single_sample)" \
    --result "curved_all4=$(result_path curved_all4_noema_single_sample)" \
    --result "rectified_all1=$(result_path rectified_all1_noema_single_sample)" \
    --result "rectified_all4=$(result_path rectified_all4_noema_single_sample)" \
    --comparison 'rectification_gain_all1=rectified_all1,curved_all1' \
    --comparison 'rectification_gain_all4=rectified_all4,curved_all4' \
    --difference_in_differences \
      'curvature_causal_effect=rectified_all1,curved_all1,rectified_all4,curved_all4' \
    --output_path "${OUTPUT_ROOT}/curvature_causal_summary.json"
}

cd "${REPO_ROOT}"
case "${PHASE}" in
  prepare) prepare ;;
  train_curved) train_arm curved "${CURVED_DIR}" ;;
  train_rectified) train_arm rectified "${RECTIFIED_DIR}" ;;
  evaluate) evaluate ;;
  summarize) summarize ;;
  all)
    require_tmux
    prepare
    train_arm curved "${CURVED_DIR}"
    train_arm rectified "${RECTIFIED_DIR}"
    evaluate
    summarize
    ;;
esac
