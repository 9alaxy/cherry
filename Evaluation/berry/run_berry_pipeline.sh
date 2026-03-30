#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOG_DIR}"

CALIB_LOG="${LOG_DIR}/pipeline_calibration_latest.log"
TRAIN_LOG="${LOG_DIR}/pipeline_train_latest.log"

# Phase 1: calibration and beta fitting
"${SCRIPT_DIR}/run_berry_calibration.sh" 2>&1 | tee "${CALIB_LOG}"

# Discover newest beta file from calibration logs
LATEST_BETA="$(ls -t "${LOG_DIR}"/calib_*_beta.json 2>/dev/null | head -n 1 || true)"

if [[ -z "${LATEST_BETA}" ]]; then
  echo "[ERROR] No beta file produced by calibration. Abort pipeline."
  exit 1
fi

echo "[INFO] Pipeline selected beta: ${LATEST_BETA}"

# Phase 2: training with regression enabled
BETA_PATH="${LATEST_BETA}" "${SCRIPT_DIR}/run_berry_train.sh" 2>&1 | tee "${TRAIN_LOG}"

echo "[INFO] Pipeline completed."
echo "[INFO] Calibration log: ${CALIB_LOG}"
echo "[INFO] Training log: ${TRAIN_LOG}"
