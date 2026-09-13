#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-${SCRIPT_DIR}/train.py}"
REPO_ROOT="${REPO_ROOT:-${PROJECT_ROOT}/third_party/S2DENet}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/Evaluation/S2DENet}"
DATA_512="${DATA_512:-../Myidea/Consolidation_Multiscale_Patches/Size_512}"
DATA_224="${DATA_224:-../Myidea/Consolidation_Multiscale_Patches/Size_224_filtered}"
LOG_ROOT="${LOG_ROOT:-${PROJECT_ROOT}/logs/s2denet}"

mkdir -p "${LOG_ROOT}"
cd "${PROJECT_ROOT}"

for SIZE in 512 224; do
  if [[ "${SIZE}" == "512" ]]; then
    DATA_ROOT="${DATA_512}"
  else
    DATA_ROOT="${DATA_224}"
  fi

  RESULT_DIR="${OUTPUT_ROOT}/${SIZE}/s2denet"
  if [[ -f "${RESULT_DIR}/evaluation_settings.json" ]]; then
    echo "[SKIP] S2DENet ${SIZE} is already fully trained and evaluated."
    continue
  fi
  RESUME_ARGS=()
  if [[ -f "${RESULT_DIR}/last.pt" ]]; then
    echo "[RESUME] Found ${RESULT_DIR}/last.pt"
    RESUME_ARGS+=(--resume)
  fi

  "${PYTHON_BIN}" -u "${TRAIN_SCRIPT}" \
    --size "${SIZE}" \
    --data-root "${DATA_ROOT}" \
    --repo-root "${REPO_ROOT}" \
    --output-root "${OUTPUT_ROOT}" \
    --device 0 \
    --workers 4 \
    "${RESUME_ARGS[@]}" \
    2>&1 | tee "${LOG_ROOT}/s2denet_size${SIZE}.log"
done

echo "S2DENet 512 and 224 training/evaluation completed."
