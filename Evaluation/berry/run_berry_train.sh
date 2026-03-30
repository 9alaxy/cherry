#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/cherry/bin/python}"

DATASET="${DATASET:-ogbn-arxiv}"
MODEL="${MODEL:-GCN}"
AGGRE="${AGGRE:-mean}"
SELECTION_METHOD="${SELECTION_METHOD:-Berry}"
NUM_BATCH="${NUM_BATCH:-2}"
NUM_EPOCHS="${NUM_EPOCHS:-2}"
NUM_HIDDEN="${NUM_HIDDEN:-64}"
NUM_LAYERS="${NUM_LAYERS:-2}"
FAN_OUT="${FAN_OUT:-10,25}"
DEVICE_NUMBER="${DEVICE_NUMBER:-1}"
ENABLE_MEMORY_AWARE="${ENABLE_MEMORY_AWARE:-1}"
ENABLE_EVAL="${ENABLE_EVAL:-1}"
ENABLE_ASYNC_PREFETCH="${ENABLE_ASYNC_PREFETCH:-0}"
PREFETCH_WORKERS="${PREFETCH_WORKERS:-2}"
PREFETCH_NON_BLOCKING="${PREFETCH_NON_BLOCKING:-1}"
MEM_BUDGET_GB="${MEM_BUDGET_GB:-10}"
SAFETY_FACTOR="${SAFETY_FACTOR:-1.0}"
MAX_PARTITION_STEPS="${MAX_PARTITION_STEPS:-8}"
BETA_PATH="${BETA_PATH:-}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOG_DIR}"

METHOD_TAG="$(echo "${SELECTION_METHOD}" | tr '[:upper:]' '[:lower:]')"
MA_TAG="ma${ENABLE_MEMORY_AWARE}"
AP_TAG="ap${ENABLE_ASYNC_PREFETCH}"
RUN_TAG="train_${METHOD_TAG}_${DATASET}_${MODEL}_${AGGRE}_b${NUM_BATCH}_l${NUM_LAYERS}_h${NUM_HIDDEN}_${MA_TAG}_${AP_TAG}_${TIMESTAMP}"
MAIN_LOG="${LOG_DIR}/${RUN_TAG}.log"
GPU_LOG="${LOG_DIR}/${RUN_TAG}_gpu_mem.log"

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

REGRESSION_FLAGS=()
if [[ -n "${BETA_PATH}" ]]; then
  REGRESSION_FLAGS+=(--memory-use-regression --memory-beta-path "${BETA_PATH}")
fi

EVAL_FLAGS=()
if [[ "${ENABLE_EVAL}" == "1" ]]; then
  EVAL_FLAGS+=(--eval)
fi

MEMORY_AWARE_FLAGS=()
if [[ "${ENABLE_MEMORY_AWARE}" == "1" ]]; then
  MEMORY_AWARE_FLAGS+=(
    --memory-aware-partition
    --memory-budget-gb "${MEM_BUDGET_GB}"
    --memory-safety-factor "${SAFETY_FACTOR}"
    --memory-max-partition-steps "${MAX_PARTITION_STEPS}"
  )
fi

ASYNC_PREFETCH_FLAGS=()
if [[ "${ENABLE_ASYNC_PREFETCH}" == "1" ]]; then
  ASYNC_PREFETCH_FLAGS+=(--async-prefetch --prefetch-workers "${PREFETCH_WORKERS}")
  if [[ "${PREFETCH_NON_BLOCKING}" == "1" ]]; then
    ASYNC_PREFETCH_FLAGS+=(--prefetch-non-blocking)
  fi
fi

{
  echo "[INFO] Train run started: ${RUN_TAG}"
  echo "[INFO] Main log: ${MAIN_LOG}"
  echo "[INFO] GPU log: ${GPU_LOG}"
  echo "[INFO] Selection method: ${SELECTION_METHOD}"
  echo "[INFO] Memory-aware enabled: ${ENABLE_MEMORY_AWARE}"
  echo "[INFO] Async prefetch enabled: ${ENABLE_ASYNC_PREFETCH}"
  echo "[INFO] Eval enabled: ${ENABLE_EVAL}"
  if [[ "${ENABLE_MEMORY_AWARE}" == "1" ]]; then
    echo "[INFO] Memory budget(GB): ${MEM_BUDGET_GB}"
    echo "[INFO] Safety factor: ${SAFETY_FACTOR}"
    echo "[INFO] Max partition steps: ${MAX_PARTITION_STEPS}"
  fi
  if [[ "${ENABLE_ASYNC_PREFETCH}" == "1" ]]; then
    echo "[INFO] Prefetch workers: ${PREFETCH_WORKERS}"
    echo "[INFO] Prefetch non_blocking: ${PREFETCH_NON_BLOCKING}"
  fi
  if [[ -n "${BETA_PATH}" ]]; then
    echo "[INFO] Using beta: ${BETA_PATH}"
  else
    echo "[INFO] No beta configured, fallback to analytical estimate."
  fi

  "${PYTHON_BIN}" micro_batch_train_berry.py \
    --dataset "${DATASET}" \
    --model "${MODEL}" \
    --aggre "${AGGRE}" \
    --selection-method "${SELECTION_METHOD}" \
    --num-batch "${NUM_BATCH}" \
    --num-epochs "${NUM_EPOCHS}" \
    --num-hidden "${NUM_HIDDEN}" \
    --num-layers "${NUM_LAYERS}" \
    --fan-out "${FAN_OUT}" \
    --device-number "${DEVICE_NUMBER}" \
    "${MEMORY_AWARE_FLAGS[@]}" \
    "${ASYNC_PREFETCH_FLAGS[@]}" \
    "${EVAL_FLAGS[@]}" \
    "${REGRESSION_FLAGS[@]}"

  echo "[INFO] Train run finished: ${RUN_TAG}"
  echo "[RESULT] MAIN_LOG=${MAIN_LOG}"
  echo "[RESULT] GPU_LOG=${GPU_LOG}"
} 2>&1 | tee "${MAIN_LOG}"
