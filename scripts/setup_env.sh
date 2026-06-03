#!/usr/bin/env bash
# Set up uv venv for One-Forcing on H200 (Hopper, sm_90).
# Run inside a slurm allocation (login node lacks CUDA driver + nproc may be low).
# flash-attn is built from source against the node's CUDA toolkit.
set -euo pipefail

REPO="/mnt/lustre/nfs/banyuanhao/One-Forcing"
cd "$REPO"

VENV_DIR=".venv"
PYTHON_VERSION="3.10"
TORCH_INDEX_URL="https://download.pytorch.org/whl/cu128"

export UV_PYTHON_INSTALL_DIR="/mnt/lustre/nfs/banyuanhao/uv/python"
export UV_CACHE_DIR="/mnt/lustre/nfs/banyuanhao/uv/cache"
export UV_LINK_MODE=copy
export UV_PYTHON_PREFERENCE=managed

export PATH="$HOME/.local/bin:$HOME/.cargo/bin:/usr/local/bin:$PATH"
export CUDA_HOME="/usr/local/cuda"
export PATH="${CUDA_HOME}/bin:${PATH}"
export MAX_JOBS="${MAX_JOBS:-$(nproc)}"

# H200 / Hopper
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"
export CUTLASS_NVCC_ARCHS="${CUTLASS_NVCC_ARCHS:-90}"

export CCACHE_DIR="${CCACHE_DIR:-/mnt/lustre/nfs/banyuanhao/ccache}"
mkdir -p "${CCACHE_DIR}"

if ! command -v uv >/dev/null 2>&1; then
  echo "[INFO] uv not found; installing..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:/usr/local/bin:$PATH"
fi
command -v uv >/dev/null 2>&1 || { echo "[ERROR] uv still not found"; exit 1; }

uv venv --python "${PYTHON_VERSION}" "${VENV_DIR}"
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

uv pip install -U pip setuptools wheel packaging ninja cmake

# Torch first so requirements.txt's torch>=2.4 is satisfied without re-resolving CPU wheels.
uv pip install --index-url "${TORCH_INDEX_URL}" torch==2.8.0 torchvision

# Project requirements (One-Forcing keeps flash-attn out of requirements.txt and
# builds it separately from source below). numpy is pinned to 1.24.4.
uv pip install -r requirements.txt

# Repo as editable (installs package "causal_forcing").
uv pip install -e .

# flash-attn — source build pinned for sm_90 (H200).
echo "[INFO] Setting MAX_JOBS=${MAX_JOBS} for flash-attn build"
export FLASH_ATTENTION_FORCE_BUILD=TRUE
export TORCH_CUDA_ARCH_LIST="9.0"
export FLASH_ATTN_CUDA_ARCHS="90"
uv pip install --no-build-isolation --no-binary :all: \
  "flash-attn @ git+https://github.com/Dao-AILab/flash-attention.git@v2.8.0"

echo "[VERIFY] importing torch + flash_attn"
python - <<'PY'
import torch, flash_attn
print("torch", torch.__version__, "cuda", torch.version.cuda, "avail", torch.cuda.is_available())
print("flash_attn", flash_attn.__version__)
PY

echo "[OK] Env setup complete: $REPO/$VENV_DIR"
