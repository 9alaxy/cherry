#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ROOT_DIR="$(cd "${EVAL_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/cherry/bin/python}"

DATASET="${DATASET:-ogbn-arxiv}"
MODEL="${MODEL:-GCN}"
AGGRE="${AGGRE:-mean}"
NUM_BATCH="${NUM_BATCH:-2}"
NUM_EPOCHS="${NUM_EPOCHS:-1}"
NUM_HIDDEN="${NUM_HIDDEN:-64}"
NUM_LAYERS="${NUM_LAYERS:-2}"
FAN_OUT="${FAN_OUT:-10,25}"
DEVICE_NUMBER="${DEVICE_NUMBER:-1}"
MEM_BUDGET_GB="${MEM_BUDGET_GB:-10}"
SAFETY_FACTOR="${SAFETY_FACTOR:-1.0}"
MAX_PARTITION_STEPS="${MAX_PARTITION_STEPS:-8}"
PROFILE_MAX_SAMPLES="${PROFILE_MAX_SAMPLES:-32}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOG_DIR}"

RUN_TAG="calib_${DATASET}_${MODEL}_${AGGRE}_b${NUM_BATCH}_l${NUM_LAYERS}_h${NUM_HIDDEN}_${TIMESTAMP}"
MAIN_LOG="${LOG_DIR}/${RUN_TAG}.log"
GPU_LOG="${LOG_DIR}/${RUN_TAG}_gpu_mem.log"
PROFILE_PATH="${LOG_DIR}/${RUN_TAG}_samples.jsonl"
BETA_PATH="${LOG_DIR}/${RUN_TAG}_beta.json"

cd "${EVAL_DIR}"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=timestamp,index,name,memory.used,memory.total,utilization.gpu --format=csv -l 1 >"${GPU_LOG}" 2>&1 &
  GPU_WATCH_PID=$!
else
  GPU_WATCH_PID=""
  echo "nvidia-smi not found, skip GPU memory sampling." >"${GPU_LOG}"
fi

cleanup() {
  if [[ -n "${GPU_WATCH_PID}" ]] && kill -0 "${GPU_WATCH_PID}" >/dev/null 2>&1; then
    kill "${GPU_WATCH_PID}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

{
  echo "[INFO] Calibration run started: ${RUN_TAG}"
  echo "[INFO] Main log: ${MAIN_LOG}"
  echo "[INFO] GPU log: ${GPU_LOG}"
  echo "[INFO] Profile samples: ${PROFILE_PATH}"
  echo "[INFO] Beta output: ${BETA_PATH}"
  echo "[INFO] Working dir: ${EVAL_DIR}"
  echo "[INFO] Command: ${PYTHON_BIN} micro_batch_train_berry.py ..."

  "${PYTHON_BIN}" micro_batch_train_berry.py \
    --dataset "${DATASET}" \
    --model "${MODEL}" \
    --aggre "${AGGRE}" \
    --selection-method Berry \
    --num-batch "${NUM_BATCH}" \
    --num-epochs "${NUM_EPOCHS}" \
    --num-hidden "${NUM_HIDDEN}" \
    --num-layers "${NUM_LAYERS}" \
    --fan-out "${FAN_OUT}" \
    --device-number "${DEVICE_NUMBER}" \
    --memory-aware-partition \
    --memory-budget-gb "${MEM_BUDGET_GB}" \
    --memory-safety-factor "${SAFETY_FACTOR}" \
    --memory-max-partition-steps "${MAX_PARTITION_STEPS}" \
    --memory-profile-collect \
    --memory-profile-max-samples "${PROFILE_MAX_SAMPLES}" \
    --memory-profile-path "${PROFILE_PATH}" \
    --memory-fit-beta \
    --memory-beta-path "${BETA_PATH}" \
    --memory-calibrate-only

  echo "[INFO] Calibration run finished: ${RUN_TAG}"
} 2>&1 | tee "${MAIN_LOG}"
