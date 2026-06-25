#!/usr/bin/env bash
# =============================================================
# OAT Environment Setup Script
# Reproduces the oat dev environment on a fresh Ubuntu machine.
#
# Prerequisites:
#   - Ubuntu 22.04+ (x86_64)
#   - NVIDIA GPU with driver >= 570 (for CUDA 12.9)
#   - Git
#   - miniforge3 / conda
#
# Usage:
#   chmod +x setup_env.sh
#   ./setup_env.sh [--env-name oat]
# =============================================================

set -euo pipefail

# ---- Config ----
PYTHON_VERSION="3.10"
ENV_NAME="oat"
CONDA_BIN="$HOME/miniforge3/bin/conda"

# Parse args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --env-name) ENV_NAME="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

echo "===== OAT Environment Setup ====="
echo "Python version : ${PYTHON_VERSION}"
echo "Conda env      : ${ENV_NAME}"
echo ""

# ---- 1. System dependencies ----
echo "[1/6] Installing system dependencies..."
sudo apt-get update -qq
sudo apt-get install -y -qq \
    build-essential cmake git wget curl \
    libgl1 libegl1 libglib2.0-0t64 libsm6 libxrender1 libxext6 \
    libglfw3 libglew-dev libosmesa6-dev patchelf \
    > /dev/null 2>&1
echo "  Done."

# ---- 2. Create conda environment ----
echo "[2/6] Creating conda environment '${ENV_NAME}' with Python ${PYTHON_VERSION}..."
"${CONDA_BIN}" create -n "${ENV_NAME}" python="${PYTHON_VERSION}" -y -q
eval "$("${CONDA_BIN}" shell.bash hook)"
conda activate "${ENV_NAME}"
pip install --upgrade pip setuptools wheel > /dev/null 2>&1
echo "  Done. Python: $(python --version)"

# ---- 3. Install PyTorch (CUDA 12.9) ----
echo "[3/6] Installing PyTorch 2.10.0 + CUDA 12.9..."
pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu129
echo "  Done."

# ---- 4. Install pip packages (pinned versions) ----
echo "[4/6] Installing pinned pip packages..."
pip install \
    absl-py==2.4.0 \
    accelerate==1.12.0 \
    av==16.1.0 \
    bddl==3.6.0 \
    beautifulsoup4==4.14.3 \
    cloudpickle==3.1.2 \
    diffusers==0.37.0 \
    dill==0.4.1 \
    easydict==1.13 \
    einops==0.8.2 \
    einx==0.3.0 \
    gdown==5.2.1 \
    glfw==2.10.0 \
    gymnasium==1.2.3 \
    gym==0.26.2 \
    h5py==3.15.1 \
    huggingface_hub==1.5.0 \
    hydra-core==1.3.2 \
    imageio==2.37.2 \
    imageio-ffmpeg==0.6.0 \
    joblib==1.5.3 \
    matplotlib==3.10.8 \
    mujoco==3.2.6 \
    nltk==3.9.3 \
    numba==0.64.0 \
    numcodecs==0.13.1 \
    numpy==2.2.6 \
    omegaconf==2.3.0 \
    opencv-python==4.13.0.92 \
    pandas==2.3.3 \
    peft==0.18.1 \
    protobuf==5.29.6 \
    PyOpenGL==3.1.10 \
    pybind11==3.0.1 \
    robomimic==0.2.0 \
    robosuite==1.4.0 \
    safetensors==0.7.0 \
    scipy==1.15.3 \
    tensorboard==2.20.0 \
    tensorboardX==2.6.4 \
    tokenizers==0.22.2 \
    transformers==5.2.0 \
    tqdm==4.67.3 \
    vector-quantize-pytorch==1.27.21 \
    wandb==0.18.7 \
    zarr==2.18.3 \
    pydantic==2.12.5 \
    pytest==9.0.2 \
    pre_commit==4.5.1 \
    jupytext==1.19.1 \
    lxml==6.0.2
echo "  Done."

# ---- 5. Install oat project (editable) ----
echo "[5/6] Installing oat project and LIBERO (editable)..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Install LIBERO first (oat depends on it)
pip install -e "${SCRIPT_DIR}/third_party/LIBERO"

# Install oat
pip install -e "${SCRIPT_DIR}"
echo "  Done."

# ---- 6. Verify ----
echo "[6/6] Verifying installation..."
python -c "
import torch
import torchvision
import oat
import libero

print(f'  PyTorch       : {torch.__version__}')
print(f'  TorchVision   : {torchvision.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  CUDA version  : {torch.version.cuda}')
    print(f'  GPU           : {torch.cuda.get_device_name(0)}')
print(f'  oat           : imported OK')
print(f'  libero        : imported OK')
"

echo ""
echo "===== Setup complete! ====="
echo "Activate with:  conda activate ${ENV_NAME}"
