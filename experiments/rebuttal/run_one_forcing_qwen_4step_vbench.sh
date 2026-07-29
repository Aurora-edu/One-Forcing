#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

usage() {
  cat <<'EOF'
Usage:
  bash experiments/rebuttal/run_one_forcing_qwen_4step_vbench.sh \
    --one_forcing_checkpoint PATH/of4_step300/model.pt \
    --qwen_pair_shard0 PATH/shard00_pairs.jsonl \
    --qwen_pair_shard1 PATH/shard01_pairs.jsonl \
    --self_forcing_videos_path PATH/self_forcing_ema_all4_qwen_videos \
    --full_info_path PATH/VBench_full_info.json \
    --output_root PATH/qwen_matched_4step \
    --gpus 0,1 \
    --vbench_python PATH/vbench_python [--python PATH/python]

Generates One-Forcing raw/no-EMA all4 videos and re-scores the existing
Self-Forcing EMA all4 videos in the same pinned VBench environment. Generation
uses exactly the historical two-process even/odd prompt sharding, with each
process seeded once with 0. Partial generation directories are not resumable.
EOF
}

if [[ $# -eq 1 && ( "$1" == "-h" || "$1" == "--help" ) ]]; then usage; exit 0; fi

ONE_FORCING_CHECKPOINT=""
QWEN_PAIR_SHARD0=""
QWEN_PAIR_SHARD1=""
SELF_FORCING_VIDEOS_PATH=""
FULL_INFO_PATH=""
OUTPUT_ROOT=""
GPUS=""
VBENCH_PYTHON=""
PYTHON_BIN="${PYTHON_BIN:-python}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --one_forcing_checkpoint) ONE_FORCING_CHECKPOINT="$2"; shift 2 ;;
    --qwen_pair_shard0) QWEN_PAIR_SHARD0="$2"; shift 2 ;;
    --qwen_pair_shard1) QWEN_PAIR_SHARD1="$2"; shift 2 ;;
    --self_forcing_videos_path) SELF_FORCING_VIDEOS_PATH="$2"; shift 2 ;;
    --full_info_path) FULL_INFO_PATH="$2"; shift 2 ;;
    --output_root) OUTPUT_ROOT="$2"; shift 2 ;;
    --gpus) GPUS="$2"; shift 2 ;;
    --vbench_python) VBENCH_PYTHON="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 1 ;;
  esac
done
for value_name in ONE_FORCING_CHECKPOINT QWEN_PAIR_SHARD0 QWEN_PAIR_SHARD1 SELF_FORCING_VIDEOS_PATH \
  FULL_INFO_PATH OUTPUT_ROOT GPUS VBENCH_PYTHON; do
  if [[ -z "${!value_name}" ]]; then echo "--${value_name,,} is required" >&2; exit 1; fi
done
if [[ ! "${GPUS}" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
  echo "--gpus must be comma-separated non-negative integers" >&2
  exit 1
fi

ONE_FORCING_CHECKPOINT="$(realpath -m "${ONE_FORCING_CHECKPOINT}")"
QWEN_PAIR_SHARD0="$(realpath -m "${QWEN_PAIR_SHARD0}")"
QWEN_PAIR_SHARD1="$(realpath -m "${QWEN_PAIR_SHARD1}")"
SELF_FORCING_VIDEOS_PATH="$(realpath -m "${SELF_FORCING_VIDEOS_PATH}")"
FULL_INFO_PATH="$(realpath -m "${FULL_INFO_PATH}")"
OUTPUT_ROOT="$(realpath -m "${OUTPUT_ROOT}")"
VBENCH_PYTHON="$(realpath -m "${VBENCH_PYTHON}")"
for path in "${ONE_FORCING_CHECKPOINT}" "${QWEN_PAIR_SHARD0}" "${QWEN_PAIR_SHARD1}" \
  "${FULL_INFO_PATH}" "${VBENCH_PYTHON}"; do
  if [[ ! -f "${path}" ]]; then echo "Input not found: ${path}" >&2; exit 1; fi
done
if [[ ! -d "${SELF_FORCING_VIDEOS_PATH}" ]]; then
  echo "Self-Forcing video directory not found: ${SELF_FORCING_VIDEOS_PATH}" >&2
  exit 1
fi
SF_VIDEO_COUNT="$(find -L "${SELF_FORCING_VIDEOS_PATH}" -maxdepth 1 -type f -name '*.mp4' | wc -l)"
if [[ "${SF_VIDEO_COUNT}" -ne 944 ]]; then
  echo "Expected exactly 944 Self-Forcing videos, found ${SF_VIDEO_COUNT}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}/audit" "${OUTPUT_ROOT}/manifests"
"${PYTHON_BIN}" "${SCRIPT_DIR}/audit_noema_checkpoint.py" \
  --checkpoint "${ONE_FORCING_CHECKPOINT}" --expected_step 300 \
  --output_path "${OUTPUT_ROOT}/audit/one_forcing_step300_noema.json"

PROMPT_PATH="${OUTPUT_ROOT}/manifests/vbench_prompts.txt"
QWEN_REWRITE_PATH="${OUTPUT_ROOT}/manifests/qwen_rewrites_official_order.txt"
MANIFEST_PATH="${OUTPUT_ROOT}/manifests/qwen_self_forcing_two_shard_seed0.jsonl"
"${PYTHON_BIN}" "${SCRIPT_DIR}/prepare_vbench_prompts.py" \
  --full_info_path "${FULL_INFO_PATH}" --output_path "${PROMPT_PATH}"
"${PYTHON_BIN}" "${SCRIPT_DIR}/merge_qwen_rewrite_shards.py" \
  --prompt_path "${PROMPT_PATH}" \
  --pair_shard "${QWEN_PAIR_SHARD0}" \
  --pair_shard "${QWEN_PAIR_SHARD1}" \
  --output_path "${QWEN_REWRITE_PATH}"
"${PYTHON_BIN}" "${SCRIPT_DIR}/make_self_forcing_seed0_manifest.py" \
  --prompt_path "${PROMPT_PATH}" \
  --qwen_rewrite_path "${QWEN_REWRITE_PATH}" \
  --output_path "${MANIFEST_PATH}"
"${PYTHON_BIN}" "${SCRIPT_DIR}/audit_qwen_vbench_inputs.py" \
  --prompt_path "${PROMPT_PATH}" \
  --qwen_rewrite_path "${QWEN_REWRITE_PATH}" \
  --manifest_path "${MANIFEST_PATH}" \
  --output_path "${OUTPUT_ROOT}/audit/qwen_prompt_seed_manifest.json"
"${PYTHON_BIN}" "${SCRIPT_DIR}/audit_self_forcing_qwen_videos.py" \
  --videos_path "${SELF_FORCING_VIDEOS_PATH}" \
  --manifest_path "${MANIFEST_PATH}" \
  --output_path "${OUTPUT_ROOT}/audit/self_forcing_video_set.json"

OF_NAME="one_forcing_raw_noema_all4_qwen_seed0"
cd "${REPO_ROOT}"
OF_ROOT="${OUTPUT_ROOT}/${OF_NAME}"
OF_VIDEOS_DIR="${OF_ROOT}/videos"
OF_VBENCH_DIR="${OF_ROOT}/vbench"
"${PYTHON_BIN}" "${SCRIPT_DIR}/run_self_forcing_seed_protocol_inference.py" \
  --config_path "${SCRIPT_DIR}/configs/eval_all4.yaml" \
  --checkpoint_path "${ONE_FORCING_CHECKPOINT}" \
  --prompt_path "${PROMPT_PATH}" \
  --qwen_rewrite_path "${QWEN_REWRITE_PATH}" \
  --manifest_path "${MANIFEST_PATH}" \
  --output_folder "${OF_VIDEOS_DIR}" \
  --gpus "${GPUS}" \
  --python "${PYTHON_BIN}"

IFS=',' read -r -a GPU_ARRAY <<< "${GPUS}"
NPROC="${#GPU_ARRAY[@]}"
mkdir -p "${OF_VBENCH_DIR}"
CUDA_VISIBLE_DEVICES="${GPUS}" "${VBENCH_PYTHON}" -m torch.distributed.run \
  --standalone --nproc_per_node="${NPROC}" \
  "${REPO_ROOT}/scripts/run_vbench.py" \
  --videos_path "${OF_VIDEOS_DIR}" \
  --full_info_path "${FULL_INFO_PATH}" \
  --output_dir "${OF_VBENCH_DIR}" \
  --name "${OF_NAME}" \
  --device cuda \
  --samples_per_prompt 1

SF_NAME="self_forcing_ema_all4_qwen_seed0"
SF_VBENCH_DIR="${OUTPUT_ROOT}/${SF_NAME}/vbench"
mkdir -p "${SF_VBENCH_DIR}"
CUDA_VISIBLE_DEVICES="${GPUS}" "${VBENCH_PYTHON}" -m torch.distributed.run \
  --standalone --nproc_per_node="${NPROC}" \
  "${REPO_ROOT}/scripts/run_vbench.py" \
  --videos_path "${SELF_FORCING_VIDEOS_PATH}" \
  --full_info_path "${FULL_INFO_PATH}" \
  --output_dir "${SF_VBENCH_DIR}" \
  --name "${SF_NAME}" \
  --device cuda \
  --samples_per_prompt 1

OF_RESULT="${OF_VBENCH_DIR}/${OF_NAME}_eval_results.json"
SF_RESULT="${SF_VBENCH_DIR}/${SF_NAME}_eval_results.json"
"${PYTHON_BIN}" "${SCRIPT_DIR}/summarize_qwen_4step_comparison.py" \
  --one_forcing_result "${OF_RESULT}" \
  --self_forcing_result "${SF_RESULT}" \
  --self_forcing_video_audit "${OUTPUT_ROOT}/audit/self_forcing_video_set.json" \
  --prompt_path "${PROMPT_PATH}" \
  --qwen_rewrite_path "${QWEN_REWRITE_PATH}" \
  --manifest_path "${MANIFEST_PATH}" \
  --output_path "${OUTPUT_ROOT}/qwen_matched_4step_summary.json"
