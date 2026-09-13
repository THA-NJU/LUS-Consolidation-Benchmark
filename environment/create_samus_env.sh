#!/usr/bin/env bash
set -euo pipefail

# Create an isolated modern environment for the official SAMUS source.
# Run this script from the APRIL project root:
#   bash create_samus_bench_env.sh samus_bench auto
#
# Arguments:
#   1. Conda environment name (default: samus_bench)
#   2. PyTorch CUDA wheel: auto, cu121, or cu124 (default: auto)
#
# Optional environment variables:
#   APRIL_ROOT=/absolute/path/to/APRIL
#   SAMUS_SOURCE_DIR=/absolute/path/to/SAMUS
#   SAMUS_CHECKPOINT_DIR=/absolute/path/to/checkpoints

ENV_NAME="${1:-samus_bench}"
CUDA_WHEEL="${2:-auto}"
APRIL_ROOT="${APRIL_ROOT:-$(pwd)}"
SAMUS_SOURCE_DIR="${SAMUS_SOURCE_DIR:-${HOME}/third_party/SAMUS}"
SAMUS_CHECKPOINT_DIR="${SAMUS_CHECKPOINT_DIR:-${APRIL_ROOT}/pretrained/samus}"
CHECKPOINT_PATH="${SAMUS_CHECKPOINT_DIR}/sam_vit_b_01ec64.pth"
CHECKPOINT_URL="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
EXPECTED_SHA256_PREFIX="01ec64"

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda was not found in PATH. Initialize Conda first." >&2
  exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi was not found. This setup expects an NVIDIA GPU." >&2
  exit 1
fi

if [[ "${CUDA_WHEEL}" == "auto" ]]; then
  DRIVER_MAJOR="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1 | cut -d. -f1)"
  if [[ ! "${DRIVER_MAJOR}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: Could not determine the NVIDIA driver version." >&2
    exit 1
  fi
  if (( DRIVER_MAJOR >= 550 )); then
    CUDA_WHEEL="cu124"
  elif (( DRIVER_MAJOR >= 525 )); then
    CUDA_WHEEL="cu121"
  else
    echo "ERROR: NVIDIA driver ${DRIVER_MAJOR}.x is too old for cu121/cu124." >&2
    exit 1
  fi
fi

if [[ "${CUDA_WHEEL}" != "cu121" && "${CUDA_WHEEL}" != "cu124" ]]; then
  echo "ERROR: CUDA wheel must be auto, cu121, or cu124; got '${CUDA_WHEEL}'." >&2
  exit 1
fi

echo "APRIL root:       ${APRIL_ROOT}"
echo "Conda env:        ${ENV_NAME}"
echo "PyTorch wheel:    ${CUDA_WHEEL}"
echo "SAMUS source:     ${SAMUS_SOURCE_DIR}"
echo "SAM checkpoint:   ${CHECKPOINT_PATH}"

if conda env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"; then
  echo "Conda environment '${ENV_NAME}' already exists; reusing it."
else
  conda create -n "${ENV_NAME}" python=3.10 pip -y
fi

conda run -n "${ENV_NAME}" python -m pip install --upgrade pip setuptools wheel
conda run -n "${ENV_NAME}" python -m pip install \
  torch==2.5.1 torchvision==0.20.1 \
  --index-url "https://download.pytorch.org/whl/${CUDA_WHEEL}"

# Do not install SAMUS requirements.txt: it pins PyTorch 1.8/CUDA 11.1 and a
# large set of unrelated 2023 packages. The APRIL benchmark imports only the
# official model code and installs its actual runtime dependencies explicitly.
conda run -n "${ENV_NAME}" python -m pip install \
  numpy==1.26.4 \
  pillow==10.4.0 \
  tqdm==4.67.1 \
  einops==0.8.1
conda run -n "${ENV_NAME}" python -m pip install \
  --no-deps opencv-python-headless==4.10.0.84

if [[ -d "${SAMUS_SOURCE_DIR}/.git" ]]; then
  echo "Existing SAMUS checkout found; leaving its revision unchanged."
else
  if [[ -e "${SAMUS_SOURCE_DIR}" ]]; then
    echo "ERROR: ${SAMUS_SOURCE_DIR} exists but is not a Git checkout." >&2
    exit 1
  fi
  if ! command -v git >/dev/null 2>&1; then
    echo "ERROR: git was not found; install git or clone SAMUS manually." >&2
    exit 1
  fi
  mkdir -p "$(dirname "${SAMUS_SOURCE_DIR}")"
  git clone https://github.com/xianlin7/SAMUS.git "${SAMUS_SOURCE_DIR}"
fi

BUILD_FILE="${SAMUS_SOURCE_DIR}/models/segment_anything_samus/build_sam_us.py"
if [[ ! -f "${BUILD_FILE}" ]]; then
  echo "ERROR: official SAMUS model builder is missing: ${BUILD_FILE}" >&2
  exit 1
fi

mkdir -p "${SAMUS_CHECKPOINT_DIR}"
if [[ -s "${CHECKPOINT_PATH}" ]]; then
  echo "Checkpoint already exists; verifying it."
else
  PART_PATH="${CHECKPOINT_PATH}.part"
  if command -v curl >/dev/null 2>&1; then
    if [[ -s "${PART_PATH}" ]]; then
      curl --fail --location --retry 3 --continue-at - \
        --output "${PART_PATH}" "${CHECKPOINT_URL}"
    else
      curl --fail --location --retry 3 \
        --output "${PART_PATH}" "${CHECKPOINT_URL}"
    fi
  elif command -v wget >/dev/null 2>&1; then
    if [[ -s "${PART_PATH}" ]]; then
      wget --tries=3 --continue --output-document="${PART_PATH}" "${CHECKPOINT_URL}"
    else
      wget --tries=3 --output-document="${PART_PATH}" "${CHECKPOINT_URL}"
    fi
  else
    echo "ERROR: neither curl nor wget is available." >&2
    exit 1
  fi
  mv "${PART_PATH}" "${CHECKPOINT_PATH}"
fi

if command -v sha256sum >/dev/null 2>&1; then
  ACTUAL_SHA256="$(sha256sum "${CHECKPOINT_PATH}" | awk '{print $1}')"
else
  ACTUAL_SHA256="$(conda run -n "${ENV_NAME}" python -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' "${CHECKPOINT_PATH}")"
fi
if [[ "${ACTUAL_SHA256}" != "${EXPECTED_SHA256_PREFIX}"* ]]; then
  echo "ERROR: SAM ViT-B checkpoint SHA-256 does not begin with ${EXPECTED_SHA256_PREFIX}." >&2
  echo "Actual: ${ACTUAL_SHA256}" >&2
  echo "Move the bad checkpoint aside and rerun this script." >&2
  exit 1
fi

SAMUS_SOURCE_DIR="${SAMUS_SOURCE_DIR}" conda run -n "${ENV_NAME}" python - <<'PY'
import os
import sys
from pathlib import Path

import einops
import numpy
import cv2
import torch
import torchvision

source = Path(os.environ["SAMUS_SOURCE_DIR"]).expanduser().resolve()
sys.path.insert(0, str(source / "models"))
from segment_anything_samus import samus_model_registry

print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("numpy:", numpy.__version__)
print("einops:", einops.__version__)
print("opencv:", cv2.__version__)
print("SAMUS registry:", sorted(samus_model_registry))
print("cuda available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("ERROR: PyTorch cannot access the NVIDIA GPU.")
print("gpu:", torch.cuda.get_device_name(0))
print("bf16 supported:", torch.cuda.is_bf16_supported())
PY

echo "SAMUS revision:   $(git -C "${SAMUS_SOURCE_DIR}" rev-parse --short HEAD)"
echo "Checkpoint SHA256:${ACTUAL_SHA256}"
echo
echo "Environment setup completed."
echo "Activate it with: conda activate ${ENV_NAME}"