#!/usr/bin/env bash
set -euo pipefail

# Add MedSAM to the existing SAM2 benchmark environment without changing
# PyTorch, CUDA, NumPy, or the installed SAM2 package.
#
# Run from the APRIL root:
#   bash setup_medsam_in_sam_bench.sh sam_bench
#
# Optional environment variables:
#   APRIL_ROOT=/absolute/path/to/APRIL
#   MEDSAM_SOURCE_DIR=/absolute/path/to/MedSAM
#   MEDSAM_CHECKPOINT_DIR=/absolute/path/to/checkpoints

ENV_NAME="${1:-sam_bench}"
APRIL_ROOT="${APRIL_ROOT:-$(pwd)}"
MEDSAM_SOURCE_DIR="${MEDSAM_SOURCE_DIR:-${HOME}/third_party/MedSAM}"
MEDSAM_CHECKPOINT_DIR="${MEDSAM_CHECKPOINT_DIR:-${APRIL_ROOT}/pretrained/medsam}"
CHECKPOINT_PATH="${MEDSAM_CHECKPOINT_DIR}/medsam_vit_b.pth"
CHECKPOINT_URL="https://zenodo.org/records/10689643/files/medsam_vit_b.pth?download=1"
EXPECTED_MD5="3bb6db55bd0c9ca30b61248bca72f8d6"

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda was not found in PATH." >&2
  exit 1
fi

if ! conda env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"; then
  echo "ERROR: Conda environment '${ENV_NAME}' does not exist." >&2
  echo "Create and verify the SAM2 environment first." >&2
  exit 1
fi

if ! command -v git >/dev/null 2>&1; then
  echo "ERROR: git was not found in PATH." >&2
  exit 1
fi

echo "APRIL root:       ${APRIL_ROOT}"
echo "Conda env:        ${ENV_NAME}"
echo "MedSAM source:    ${MEDSAM_SOURCE_DIR}"
echo "MedSAM checkpoint:${CHECKPOINT_PATH}"

# Deliberately do not run MedSAM's setup.py. Its current custom installer can
# modify NumPy and NVIDIA runtime packages. The benchmark imports the cloned
# source tree explicitly. Only headless OpenCV is added if cv2 is unavailable,
# because segment_anything imports it at package initialization.
if [[ -d "${MEDSAM_SOURCE_DIR}/.git" ]]; then
  echo "Existing MedSAM checkout found; leaving its revision unchanged."
else
  if [[ -e "${MEDSAM_SOURCE_DIR}" ]]; then
    echo "ERROR: ${MEDSAM_SOURCE_DIR} exists but is not a Git checkout." >&2
    exit 1
  fi
  mkdir -p "$(dirname "${MEDSAM_SOURCE_DIR}")"
  git clone https://github.com/bowang-lab/MedSAM.git "${MEDSAM_SOURCE_DIR}"
fi

if [[ ! -f "${MEDSAM_SOURCE_DIR}/segment_anything/build_sam.py" ]]; then
  echo "ERROR: segment_anything/build_sam.py is missing from ${MEDSAM_SOURCE_DIR}." >&2
  exit 1
fi

if conda run -n "${ENV_NAME}" python -c 'import cv2' >/dev/null 2>&1; then
  echo "OpenCV is already available; leaving it unchanged."
else
  conda run -n "${ENV_NAME}" python -m pip install \
    --no-deps opencv-python-headless==4.10.0.84
fi

mkdir -p "${MEDSAM_CHECKPOINT_DIR}"
if [[ -s "${CHECKPOINT_PATH}" ]]; then
  echo "Checkpoint already exists; verifying MD5."
else
  PART_PATH="${CHECKPOINT_PATH}.part"
  if command -v curl >/dev/null 2>&1; then
    curl --fail --location --retry 3 --output "${PART_PATH}" "${CHECKPOINT_URL}"
  elif command -v wget >/dev/null 2>&1; then
    wget --tries=3 --output-document="${PART_PATH}" "${CHECKPOINT_URL}"
  else
    echo "ERROR: neither curl nor wget is available." >&2
    exit 1
  fi
  mv "${PART_PATH}" "${CHECKPOINT_PATH}"
fi

ACTUAL_MD5="$(md5sum "${CHECKPOINT_PATH}" | awk '{print $1}')"
if [[ "${ACTUAL_MD5}" != "${EXPECTED_MD5}" ]]; then
  echo "ERROR: MedSAM checkpoint MD5 mismatch." >&2
  echo "Expected: ${EXPECTED_MD5}" >&2
  echo "Actual:   ${ACTUAL_MD5}" >&2
  echo "Move the bad checkpoint aside and rerun this script." >&2
  exit 1
fi

conda run -n "${ENV_NAME}" python -c \
  'import cv2, numpy, torch, torchvision; print("torch:", torch.__version__); print("torchvision:", torchvision.__version__); print("numpy:", numpy.__version__); print("opencv:", cv2.__version__); print("cuda available:", torch.cuda.is_available()); assert torch.cuda.is_available()'

echo "MedSAM revision: $(git -C "${MEDSAM_SOURCE_DIR}" rev-parse --short HEAD)"
echo "Checkpoint MD5:  ${ACTUAL_MD5}"
echo
echo "MedSAM setup completed without changing PyTorch, CUDA, NumPy, or SAM2."
echo "Activate with: conda activate ${ENV_NAME}"