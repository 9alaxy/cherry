#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

DATASET="${DATASET:-ogbn-arxiv}"
MODELS="${MODELS:-GCN}"
AGGRE="${AGGRE:-mean}"
DEVICE_NUMBER="${DEVICE_NUMBER:-1}"
NUM_EPOCHS="${NUM_EPOCHS:-10}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-1}"
BASELINE_NUM_BATCH="${BASELINE_NUM_BATCH:-4}"
BERRY_INIT_BATCH="${BERRY_INIT_BATCH:-2}"
NUM_HIDDEN="${NUM_HIDDEN:-64}"
NUM_LAYERS="${NUM_LAYERS:-2}"
FAN_OUT="${FAN_OUT:-10,25}"
SAFETY_FACTOR="${SAFETY_FACTOR:-1.0}"
MAX_PARTITION_STEPS="${MAX_PARTITION_STEPS:-8}"
RELAXED_RATIO="${RELAXED_RATIO:-0.92}"
TIGHT_RATIO="${TIGHT_RATIO:-0.80}"
BETA_PATH="${BETA_PATH:-}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
EXP_DIR="${SCRIPT_DIR}/logs/effectiveness_${DATASET}_${TIMESTAMP}"
MANIFEST_CSV="${EXP_DIR}/manifest.csv"
SUMMARY_CSV="${EXP_DIR}/summary.csv"
SUMMARY_MD="${EXP_DIR}/summary.md"

mkdir -p "${EXP_DIR}"

cat >"${MANIFEST_CSV}" <<'EOF'
run_id,model,method,memory_aware,budget_gb,num_batch_init,num_epochs,role,main_log,gpu_log
EOF

extract_result_path() {
  local launcher_log="$1"
  local key="$2"
  grep -E "^\[RESULT\] ${key}=" "${launcher_log}" | tail -n 1 | cut -d'=' -f2-
}

extract_peak_gb_from_main_log() {
  local main_log="$1"
  grep -E "max memory allocated:" "${main_log}" | awk '{print $(NF-1)}' | awk 'BEGIN {max=0} {if ($1+0>max) max=$1+0} END {printf "%.6f", max}'
}

append_manifest() {
  local run_id="$1"
  local model="$2"
  local method="$3"
  local memory_aware="$4"
  local budget_gb="$5"
  local num_batch_init="$6"
  local num_epochs="$7"
  local role="$8"
  local main_log="$9"
  local gpu_log="${10}"

  echo "${run_id},${model},${method},${memory_aware},${budget_gb},${num_batch_init},${num_epochs},${role},${main_log},${gpu_log}" >>"${MANIFEST_CSV}"
}

run_case() {
  local run_id="$1"
  local model="$2"
  local method="$3"
  local memory_aware="$4"
  local budget_gb="$5"
  local num_batch_init="$6"
  local num_epochs="$7"
  local role="$8"

  local launcher_log="${EXP_DIR}/${run_id}_launcher.log"
  echo "[INFO] Running ${run_id}: model=${model}, method=${method}, memory_aware=${memory_aware}, budget=${budget_gb}, init_batch=${num_batch_init}, epochs=${num_epochs}"

  (
    DATASET="${DATASET}" \
    MODEL="${model}" \
    AGGRE="${AGGRE}" \
    SELECTION_METHOD="${method}" \
    ENABLE_MEMORY_AWARE="${memory_aware}" \
    MEM_BUDGET_GB="${budget_gb}" \
    NUM_BATCH="${num_batch_init}" \
    NUM_EPOCHS="${num_epochs}" \
    NUM_HIDDEN="${NUM_HIDDEN}" \
    NUM_LAYERS="${NUM_LAYERS}" \
    FAN_OUT="${FAN_OUT}" \
    DEVICE_NUMBER="${DEVICE_NUMBER}" \
    SAFETY_FACTOR="${SAFETY_FACTOR}" \
    MAX_PARTITION_STEPS="${MAX_PARTITION_STEPS}" \
    BETA_PATH="${BETA_PATH}" \
    "${SCRIPT_DIR}/run_berry_train.sh"
  ) 2>&1 | tee "${launcher_log}"

  local main_log
  local gpu_log
  main_log="$(extract_result_path "${launcher_log}" MAIN_LOG)"
  gpu_log="$(extract_result_path "${launcher_log}" GPU_LOG)"

  if [[ -z "${main_log}" || -z "${gpu_log}" ]]; then
    echo "[ERROR] Failed to parse produced logs for ${run_id}. Check ${launcher_log}" >&2
    exit 1
  fi

  append_manifest "${run_id}" "${model}" "${method}" "${memory_aware}" "${budget_gb}" "${num_batch_init}" "${num_epochs}" "${role}" "${main_log}" "${gpu_log}"
}

for model in ${MODELS}; do
  warmup_id="${model}_cherry_warmup"
  run_case "${warmup_id}" "${model}" "Cherry" "0" "0" "${BASELINE_NUM_BATCH}" "${WARMUP_EPOCHS}" "warmup"

  warmup_main_log="$(tail -n 1 "${MANIFEST_CSV}" | cut -d',' -f9)"
  warmup_peak="$(extract_peak_gb_from_main_log "${warmup_main_log}")"
  if [[ "${warmup_peak}" == "0.000000" ]]; then
    echo "[ERROR] Warmup peak memory not found for ${model}. Log: ${warmup_main_log}" >&2
    exit 1
  fi

  relaxed_budget="$(awk -v p="${warmup_peak}" -v r="${RELAXED_RATIO}" 'BEGIN {printf "%.4f", p*r}')"
  tight_budget="$(awk -v p="${warmup_peak}" -v r="${TIGHT_RATIO}" 'BEGIN {printf "%.4f", p*r}')"

  echo "[INFO] ${model} warmup peak=${warmup_peak} GB, relaxed_budget=${relaxed_budget} GB, tight_budget=${tight_budget} GB"

  run_case "${model}_cherry_baseline" "${model}" "Cherry" "0" "0" "${BASELINE_NUM_BATCH}" "${NUM_EPOCHS}" "baseline"
  run_case "${model}_metis_baseline" "${model}" "Metis" "0" "0" "${BASELINE_NUM_BATCH}" "${NUM_EPOCHS}" "baseline"
  run_case "${model}_berry_relaxed" "${model}" "Berry" "1" "${relaxed_budget}" "${BERRY_INIT_BATCH}" "${NUM_EPOCHS}" "memory-aware"
  run_case "${model}_berry_tight" "${model}" "Berry" "1" "${tight_budget}" "${BERRY_INIT_BATCH}" "${NUM_EPOCHS}" "memory-aware"
done

"${SCRIPT_DIR}/summarize_memory_aware_results.py" \
  --manifest "${MANIFEST_CSV}" \
  --output-csv "${SUMMARY_CSV}" \
  --output-md "${SUMMARY_MD}"

echo "[INFO] Experiment directory: ${EXP_DIR}"
echo "[INFO] Manifest: ${MANIFEST_CSV}"
echo "[INFO] Summary CSV: ${SUMMARY_CSV}"
echo "[INFO] Summary MD: ${SUMMARY_MD}"
