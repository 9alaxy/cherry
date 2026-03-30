#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/cherry/bin/python}"

DATASET="${DATASET:-ogbn-products}"
MODEL="${MODEL:-GCN}"
AGGRE="${AGGRE:-mean}"
NUM_BATCH="${NUM_BATCH:-8}"
NUM_EPOCHS="${NUM_EPOCHS:-10}"
NUM_HIDDEN="${NUM_HIDDEN:-128}"
NUM_LAYERS="${NUM_LAYERS:-3}"
FAN_OUT="${FAN_OUT:-10,25,30}"
DEVICE_NUMBER="${DEVICE_NUMBER:-1}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOG_DIR}"

RUN_TAG="cherry_conv_${DATASET}_${MODEL}_${AGGRE}_b${NUM_BATCH}_l${NUM_LAYERS}_h${NUM_HIDDEN}_e${NUM_EPOCHS}_${TIMESTAMP}"
MAIN_LOG="${LOG_DIR}/${RUN_TAG}.log"
GPU_LOG="${LOG_DIR}/${RUN_TAG}_gpu_mem.log"
PLOT_FILE="${LOG_DIR}/${RUN_TAG}_curve.png"
LAUNCHER_LOG="${LOG_DIR}/${RUN_TAG}_launcher.log"

cd "${EVAL_DIR}"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --id="${DEVICE_NUMBER}" --query-gpu=timestamp,index,name,memory.used,memory.total,utilization.gpu --format=csv -l 1 >"${GPU_LOG}" 2>&1 &
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
  echo "[INFO] Convergence run started: ${RUN_TAG}"
  echo "[INFO] Main log: ${MAIN_LOG}"
  echo "[INFO] Launcher log: ${LAUNCHER_LOG}"
  echo "[INFO] GPU log: ${GPU_LOG}"
  echo "[INFO] Plot file: ${PLOT_FILE}"

  "${PYTHON_BIN}" micro_batch_train.py \
    --dataset "${DATASET}" \
    --aggre "${AGGRE}" \
    --seed 1236 \
    --setseed True \
    --GPUmem True \
    --selection-method Cherry \
    --num-batch "${NUM_BATCH}" \
    --lr 0.01 \
    --num-runs 1 \
    --num-epochs "${NUM_EPOCHS}" \
    --num-layers "${NUM_LAYERS}" \
    --num-hidden "${NUM_HIDDEN}" \
    --dropout 0.5 \
    --fan-out "${FAN_OUT}" \
    --device-number "${DEVICE_NUMBER}" \
    --num-heads 4 \
    --model "${MODEL}" \
    --eval \
    2>&1 | tee "${MAIN_LOG}"

  "${PYTHON_BIN}" data_collection.py \
    --path "${LOG_DIR}" \
    --files "$(basename "${MAIN_LOG}")" \
    --labels "Cherry-${MODEL}-${DATASET}" \
    --output "${PLOT_FILE}"

  echo "[RESULT] MAIN_LOG=${MAIN_LOG}"
  echo "[RESULT] LAUNCHER_LOG=${LAUNCHER_LOG}"
  echo "[RESULT] GPU_LOG=${GPU_LOG}"
  echo "[RESULT] CURVE_PNG=${PLOT_FILE}"
  echo "[INFO] Convergence run finished: ${RUN_TAG}"
} 2>&1 | tee "${LAUNCHER_LOG}"
