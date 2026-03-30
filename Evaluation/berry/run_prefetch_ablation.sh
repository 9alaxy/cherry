#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DATASET="${DATASET:-ogbn-arxiv}"
MODEL="${MODEL:-GCN}"
AGGRE="${AGGRE:-mean}"
NUM_BATCH="${NUM_BATCH:-4}"
NUM_EPOCHS="${NUM_EPOCHS:-1}"
NUM_HIDDEN="${NUM_HIDDEN:-64}"
NUM_LAYERS="${NUM_LAYERS:-2}"
FAN_OUT="${FAN_OUT:-10,25}"
DEVICE_NUMBER="${DEVICE_NUMBER:-1}"
ENABLE_MEMORY_AWARE="${ENABLE_MEMORY_AWARE:-0}"
ENABLE_EVAL="${ENABLE_EVAL:-0}"
PREFETCH_WORKERS="${PREFETCH_WORKERS:-2}"
PREFETCH_NON_BLOCKING="${PREFETCH_NON_BLOCKING:-1}"

run_once() {
  local async_flag="$1"
  local workers="$2"

  ENABLE_ASYNC_PREFETCH="${async_flag}" \
  PREFETCH_WORKERS="${workers}" \
  PREFETCH_NON_BLOCKING="${PREFETCH_NON_BLOCKING}" \
  ENABLE_MEMORY_AWARE="${ENABLE_MEMORY_AWARE}" \
  ENABLE_EVAL="${ENABLE_EVAL}" \
  DATASET="${DATASET}" \
  MODEL="${MODEL}" \
  AGGRE="${AGGRE}" \
  NUM_BATCH="${NUM_BATCH}" \
  NUM_EPOCHS="${NUM_EPOCHS}" \
  NUM_HIDDEN="${NUM_HIDDEN}" \
  NUM_LAYERS="${NUM_LAYERS}" \
  FAN_OUT="${FAN_OUT}" \
  DEVICE_NUMBER="${DEVICE_NUMBER}" \
  bash "${SCRIPT_DIR}/run_berry_train.sh"
}

echo "[INFO] Prefetch ablation started"
echo "[INFO] dataset=${DATASET} model=${MODEL} batch=${NUM_BATCH} layers=${NUM_LAYERS} hidden=${NUM_HIDDEN} fanout=${FAN_OUT}"
echo "[INFO] device_number=${DEVICE_NUMBER} (should be RTX 3090 in this env)"

run_once 0 1
run_once 1 "${PREFETCH_WORKERS}"

python3 "${SCRIPT_DIR}/summarize_prefetch_results.py" \
  --dataset "${DATASET}" \
  --model "${MODEL}" \
  --aggre "${AGGRE}" \
  --num-batch "${NUM_BATCH}" \
  --num-layers "${NUM_LAYERS}" \
  --num-hidden "${NUM_HIDDEN}" \
  --memory-aware "${ENABLE_MEMORY_AWARE}"

echo "[INFO] Prefetch ablation finished"