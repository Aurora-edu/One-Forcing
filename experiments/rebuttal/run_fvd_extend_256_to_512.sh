#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Extend the existing matched 256-video FVD evaluation to 512 videos.

Required arguments:
  --lmdb_path PATH
  --existing_real_dir PATH
  --existing_main_fake_dir PATH
  --existing_dmd_fake_dir PATH
  --main_checkpoint PATH
  --dmd_checkpoint PATH
  --i3d_path PATH
  --output_root PATH
  --gpus CSV                    Example: 0,1,2,3,4,5,6,7

Optional arguments:
  --config_path PATH            Default: experiments/rebuttal/configs/eval_ffe.yaml
  --python PATH                 Default: python
  --batch_size N                Default: 4
  --bootstrap_samples N         Default: 0
  --launch_delay_seconds SEC    Default: 0

The existing real directory must contain the original reference_manifest.jsonl
and generation_manifest.jsonl. The script creates only the additional 256 real
and fake videos, then scores the union of the old and new directories.
EOF
}

require_value() {
  if [[ $# -lt 2 || -z "$2" ]]; then
    echo "Missing value for $1" >&2
    exit 2
  fi
}

LMDB_PATH=""
EXISTING_REAL_DIR=""
EXISTING_MAIN_FAKE_DIR=""
EXISTING_DMD_FAKE_DIR=""
MAIN_CHECKPOINT=""
DMD_CHECKPOINT=""
I3D_PATH=""
OUTPUT_ROOT=""
GPUS=""
CONFIG_PATH="experiments/rebuttal/configs/eval_ffe.yaml"
PYTHON_BIN="python"
BATCH_SIZE=4
BOOTSTRAP_SAMPLES=0
LAUNCH_DELAY_SECONDS=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --lmdb_path)
      require_value "$@"; LMDB_PATH="$2"; shift 2 ;;
    --existing_real_dir)
      require_value "$@"; EXISTING_REAL_DIR="$2"; shift 2 ;;
    --existing_main_fake_dir)
      require_value "$@"; EXISTING_MAIN_FAKE_DIR="$2"; shift 2 ;;
    --existing_dmd_fake_dir)
      require_value "$@"; EXISTING_DMD_FAKE_DIR="$2"; shift 2 ;;
    --main_checkpoint)
      require_value "$@"; MAIN_CHECKPOINT="$2"; shift 2 ;;
    --dmd_checkpoint)
      require_value "$@"; DMD_CHECKPOINT="$2"; shift 2 ;;
    --i3d_path)
      require_value "$@"; I3D_PATH="$2"; shift 2 ;;
    --output_root)
      require_value "$@"; OUTPUT_ROOT="$2"; shift 2 ;;
    --gpus)
      require_value "$@"; GPUS="$2"; shift 2 ;;
    --config_path)
      require_value "$@"; CONFIG_PATH="$2"; shift 2 ;;
    --python)
      require_value "$@"; PYTHON_BIN="$2"; shift 2 ;;
    --batch_size)
      require_value "$@"; BATCH_SIZE="$2"; shift 2 ;;
    --bootstrap_samples)
      require_value "$@"; BOOTSTRAP_SAMPLES="$2"; shift 2 ;;
    --launch_delay_seconds)
      require_value "$@"; LAUNCH_DELAY_SECONDS="$2"; shift 2 ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2 ;;
  esac
done

for variable_name in \
  LMDB_PATH EXISTING_REAL_DIR EXISTING_MAIN_FAKE_DIR EXISTING_DMD_FAKE_DIR \
  MAIN_CHECKPOINT DMD_CHECKPOINT I3D_PATH OUTPUT_ROOT GPUS; do
  if [[ -z "${!variable_name}" ]]; then
    echo "Missing required argument for ${variable_name}" >&2
    usage >&2
    exit 2
  fi
done

for directory in "$EXISTING_REAL_DIR" "$EXISTING_MAIN_FAKE_DIR" "$EXISTING_DMD_FAKE_DIR"; do
  if [[ ! -d "$directory" ]]; then
    echo "Missing directory: $directory" >&2
    exit 1
  fi
done
for file in \
  "$EXISTING_REAL_DIR/reference_manifest.jsonl" \
  "$EXISTING_REAL_DIR/generation_manifest.jsonl" \
  "$MAIN_CHECKPOINT" "$DMD_CHECKPOINT" "$I3D_PATH" "$CONFIG_PATH"; do
  if [[ ! -f "$file" ]]; then
    echo "Missing file: $file" >&2
    exit 1
  fi
done
if [[ ! -e "$LMDB_PATH" ]]; then
  echo "Missing LMDB path: $LMDB_PATH" >&2
  exit 1
fi
if ! [[ "$BATCH_SIZE" =~ ^[1-9][0-9]*$ ]]; then
  echo "--batch_size must be a positive integer" >&2
  exit 2
fi
if ! [[ "$BOOTSTRAP_SAMPLES" =~ ^[0-9]+$ ]]; then
  echo "--bootstrap_samples must be a non-negative integer" >&2
  exit 2
fi

count_videos() {
  find "$1" -maxdepth 1 -type f \
    \( -iname '*.mp4' -o -iname '*.mov' -o -iname '*.avi' -o -iname '*.mkv' -o -iname '*.webm' \) \
    -printf '%f\n' | wc -l
}

for directory in "$EXISTING_REAL_DIR" "$EXISTING_MAIN_FAKE_DIR" "$EXISTING_DMD_FAKE_DIR"; do
  count="$(count_videos "$directory")"
  if [[ "$count" -ne 256 ]]; then
    echo "Expected exactly 256 existing videos in $directory, found $count" >&2
    exit 1
  fi
done

mkdir -p "$OUTPUT_ROOT"
EXTRA_REAL_DIR="$OUTPUT_ROOT/real_extra256"
EXTRA_MAIN_DIR="$OUTPUT_ROOT/main_extra256"
EXTRA_DMD_DIR="$OUTPUT_ROOT/dmd_extra256"
FIRST_GPU="${GPUS%%,*}"

echo "[1/5] Decoding 256 unseen real references"
CUDA_VISIBLE_DEVICES="$FIRST_GPU" "$PYTHON_BIN" \
  experiments/rebuttal/decode_lmdb_references.py \
  --lmdb_path "$LMDB_PATH" \
  --output_dir "$EXTRA_REAL_DIR" \
  --num_videos 256 \
  --seed 0 \
  --streaming_decode \
  --existing_reference_manifest_path "$EXISTING_REAL_DIR/reference_manifest.jsonl" \
  --existing_generation_manifest_path "$EXISTING_REAL_DIR/generation_manifest.jsonl"

echo "[2/5] Generating the additional 256 main-method videos"
"$PYTHON_BIN" experiments/rebuttal/run_sharded_inference.py \
  --config_path "$CONFIG_PATH" \
  --checkpoint_path "$MAIN_CHECKPOINT" \
  --prompt_path "$EXTRA_REAL_DIR/reference_prompts.txt" \
  --manifest_path "$EXTRA_REAL_DIR/generation_manifest.jsonl" \
  --output_folder "$EXTRA_MAIN_DIR" \
  --gpus "$GPUS" \
  --schedule ffe \
  --launch_delay_seconds "$LAUNCH_DELAY_SECONDS" \
  --python "$PYTHON_BIN"

echo "[3/5] Generating the additional 256 DMD-only videos"
"$PYTHON_BIN" experiments/rebuttal/run_sharded_inference.py \
  --config_path "$CONFIG_PATH" \
  --checkpoint_path "$DMD_CHECKPOINT" \
  --prompt_path "$EXTRA_REAL_DIR/reference_prompts.txt" \
  --manifest_path "$EXTRA_REAL_DIR/generation_manifest.jsonl" \
  --output_folder "$EXTRA_DMD_DIR" \
  --gpus "$GPUS" \
  --schedule ffe \
  --launch_delay_seconds "$LAUNCH_DELAY_SECONDS" \
  --python "$PYTHON_BIN"

COMMON_SCORE_ARGS=(
  --real_videos_dir "$EXISTING_REAL_DIR"
  --real_videos_dir "$EXTRA_REAL_DIR"
  --real_manifest_path "$EXTRA_REAL_DIR/reference_manifest_combined.jsonl"
  --fake_manifest_path "$EXTRA_REAL_DIR/generation_manifest_combined.jsonl"
  --i3d_path "$I3D_PATH"
  --num_frames 16
  --batch_size "$BATCH_SIZE"
  --min_videos 512
  --nearest_k 5
  --bootstrap_samples "$BOOTSTRAP_SAMPLES"
)

echo "[4/5] Computing 512-sample main-method FVD and manifold metrics"
CUDA_VISIBLE_DEVICES="$FIRST_GPU" "$PYTHON_BIN" \
  experiments/rebuttal/evaluate_fvd.py \
  "${COMMON_SCORE_ARGS[@]}" \
  --fake_videos_dir "$EXISTING_MAIN_FAKE_DIR" \
  --fake_videos_dir "$EXTRA_MAIN_DIR" \
  --output_json "$OUTPUT_ROOT/main_512.json"

echo "[5/5] Computing 512-sample DMD-only FVD and manifold metrics"
CUDA_VISIBLE_DEVICES="$FIRST_GPU" "$PYTHON_BIN" \
  experiments/rebuttal/evaluate_fvd.py \
  "${COMMON_SCORE_ARGS[@]}" \
  --fake_videos_dir "$EXISTING_DMD_FAKE_DIR" \
  --fake_videos_dir "$EXTRA_DMD_DIR" \
  --output_json "$OUTPUT_ROOT/dmd_512.json"

echo "Complete. Metrics:"
echo "  $OUTPUT_ROOT/main_512.json"
echo "  $OUTPUT_ROOT/dmd_512.json"
