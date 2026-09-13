#!/usr/bin/env bash
set -euo pipefail

# Run this script from the APRIL project root:
#   bash create_sam_bench_env.sh sam_bench auto
#
# Arguments:
#   1. Conda environment name (default: sam_bench)
#   2. PyTorch CUDA wheel: auto, cu121, or cu124 (default: auto)
#
# Optional environment variables:
#   APRIL_ROOT=/absolute/path/to/APRIL
#   SAM2_SOURCE_DIR=/absolute/path/to/sam2
#   SAM2_CHECKPOINT_DIR=/absolute/path/to/checkpoints

ENV_NAME="${1:-sam_bench}"
CUDA_WHEEL="${2:-auto}"
APRIL_ROOT="${APRIL_ROOT:-$(pwd)}"
SAM2_SOURCE_DIR="${SAM2_SOURCE_DIR:-${HOME}/third_party/sam2}"
SAM2_CHECKPOINT_DIR="${SAM2_CHECKPOINT_DIR:-${APRIL_ROOT}/pretrained/sam2}"
CHECKPOINT_PATH="${SAM2_CHECKPOINT_DIR}/sam2.1_hiera_base_plus.pt"
CHECKPOINT_URL="https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt"

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
    echo "ERROR: NVIDIA driver ${DRIVER_MAJOR}.x is too old for the supported wheels." >&2
    echo "Update the driver, or install a compatible PyTorch build manually." >&2
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
echo "SAM2 source:      ${SAM2_SOURCE_DIR}"
echo "SAM2 checkpoint:  ${CHECKPOINT_PATH}"

if conda env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"; then
  echo "Conda environment '${ENV_NAME}' already exists; reusing it."
else
  conda create -n "${ENV_NAME}" python=3.10 pip -y
fi

conda run -n "${ENV_NAME}" python -m pip install --upgrade pip setuptools wheel
conda run -n "${ENV_NAME}" python -m pip install \
  torch==2.5.1 torchvision==0.20.1 \
  --index-url "https://download.pytorch.org/whl/${CUDA_WHEEL}"

# Pin NumPy below 2 for compatibility with the later shared MedSAM setup.
conda run -n "${ENV_NAME}" python -m pip install \
  numpy==1.26.4 \
  pillow==10.4.0 \
  pandas==2.2.3 \
  tqdm==4.67.1 \
  hydra-core==1.3.2 \
  iopath==0.1.10 \
  pyyaml==6.0.2 \
  tensorboard==2.18.0

if [[ -d "${SAM2_SOURCE_DIR}/.git" ]]; then
  echo "Existing SAM2 checkout found; leaving its revision unchanged."
else
  if ! command -v git >/dev/null 2>&1; then
    echo "ERROR: git was not found; install git or clone SAM2 manually." >&2
    exit 1
  fi
  mkdir -p "$(dirname "${SAM2_SOURCE_DIR}")"
  git clone https://github.com/facebookresearch/sam2.git "${SAM2_SOURCE_DIR}"
fi

# The optional connected-components CUDA extension is not used by the benchmark.
# Skipping it avoids dependence on a locally installed nvcc toolkit.
SAM2_BUILD_CUDA=0 conda run -n "${ENV_NAME}" \
  python -m pip install --no-deps -e "${SAM2_SOURCE_DIR}"

mkdir -p "${SAM2_CHECKPOINT_DIR}"
if [[ -s "${CHECKPOINT_PATH}" ]]; then
  echo "Checkpoint already exists; skipping download."
elif command -v curl >/dev/null 2>&1; then
  curl --fail --location --retry 3 \
    --output "${CHECKPOINT_PATH}.part" "${CHECKPOINT_URL}"
  mv "${CHECKPOINT_PATH}.part" "${CHECKPOINT_PATH}"
elif command -v wget >/dev/null 2>&1; then
  wget --tries=3 --output-document="${CHECKPOINT_PATH}.part" "${CHECKPOINT_URL}"
  mv "${CHECKPOINT_PATH}.part" "${CHECKPOINT_PATH}"
else
  echo "ERROR: neither curl nor wget is available to download the checkpoint." >&2
  exit 1
fi

conda run -n "${ENV_NAME}" python - <<'PY'
import torch
import torchvision
import sam2

print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("sam2:", sam2.__file__)
print("cuda available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("ERROR: PyTorch cannot access the NVIDIA GPU.")
print("gpu:", torch.cuda.get_device_name(0))
print("bf16 supported:", torch.cuda.is_bf16_supported())
PY

echo
echo "Environment setup completed."
echo "Activate it with: conda activate ${ENV_NAME}"
echo "Checkpoint: ${CHECKPOINT_PATH}"