#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BERRY_OUT="${SCRIPT_DIR}/Berry-replication-factor-ogbn-arxiv.csv"
REG_OUT="${SCRIPT_DIR}/REG-replication-factor-ogbn-arxiv.csv"

berry_mbs=(2 4 8 16 32 64)
reg_mbs=(2 4 8 32 64)

# Optional: limit REG statistics to first N epochs.
# Default is "all epochs found in log".
REG_EPOCHS_LIMIT=""
if [[ "${1:-}" == "--reg-epochs" ]]; then
  if [[ -z "${2:-}" || ! "${2}" =~ ^[0-9]+$ || "${2}" -le 0 ]]; then
    echo "Usage: $0 [--reg-epochs N]" >&2
    exit 1
  fi
  REG_EPOCHS_LIMIT="${2}"
fi

extract_last_rf() {
  local file="$1"
  rg -o 'Replication Factor:\s*[0-9]+\.[0-9]+' "$file" | tail -n 1 | awk '{print $3}'
}

extract_all_rfs() {
  local file="$1"
  rg -o 'Replication Factor:\s*[0-9]+\.[0-9]+' "$file" | awk '{print $3}'
}

echo 'micro_batch,replication_factor' > "$BERRY_OUT"
for mb in "${berry_mbs[@]}"; do
  file="${SCRIPT_DIR}/Berry-${mb}-batch-3-layer-256-hid-GCN-ogbn-arxiv.log"
  if [[ ! -f "$file" ]]; then
    echo "Missing file: $file" >&2
    exit 1
  fi
  rf="$(extract_last_rf "$file")"
  echo "${mb},${rf}" >> "$BERRY_OUT"
done

echo 'micro_batch,epochs_used,avg_replication_factor,max_replication_factor,min_replication_factor' > "$REG_OUT"
for mb in "${reg_mbs[@]}"; do
  file="${SCRIPT_DIR}/REG-${mb}-batch-3-layer-256-hid-GCN-ogbn-arxiv.log"
  if [[ ! -f "$file" ]]; then
    echo "Missing file: $file" >&2
    exit 1
  fi

  if [[ -n "$REG_EPOCHS_LIMIT" ]]; then
    values="$(extract_all_rfs "$file" | sed -n "1,${REG_EPOCHS_LIMIT}p")"
  else
    values="$(extract_all_rfs "$file")"
  fi

  count="$(printf '%s\n' "$values" | awk 'NF{n+=1} END{print n+0}')"
  avg="$(printf '%s\n' "$values" | awk '{s+=$1;n+=1} END{if(n>0) printf "%.4f", s/n; else printf "NaN"}')"
  maxv="$(printf '%s\n' "$values" | awk 'NR==1{m=$1} $1>m{m=$1} END{if(NR>0) printf "%.4f", m; else printf "NaN"}')"
  minv="$(printf '%s\n' "$values" | awk 'NR==1{m=$1} $1<m{m=$1} END{if(NR>0) printf "%.4f", m; else printf "NaN"}')"
  echo "${mb},${count},${avg},${maxv},${minv}" >> "$REG_OUT"
done

echo "Generated:"
echo "  $BERRY_OUT"
echo "  $REG_OUT"
if [[ -n "$REG_EPOCHS_LIMIT" ]]; then
  echo "REG mode: first ${REG_EPOCHS_LIMIT} epochs"
else
  echo "REG mode: all epochs in each log"
fi
