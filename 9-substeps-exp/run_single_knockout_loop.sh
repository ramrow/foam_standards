#!/bin/bash
set -euo pipefail

# Activate env + repo root (adjust if needed)
source /mnt/lustre/rpi/pxu10/agent/bin/activate
cd "/mnt/lustre/rpi/pxu10/abalation"

export OPENAI_API_KEY="EMPTY"
export OPENAI_BASE_URL="http://127.0.0.1:8000/v1"

CFG_DIR="./9-substeps-exp/single_knockout_configs"
OUT_ROOT="./experiment-finetuned/single_knockout"

mkdir -p "$OUT_ROOT"

for cfg in "$CFG_DIR"/knockout_*.json; do
  name="$(basename "$cfg" .json)"
  out_dir="$OUT_ROOT/$name"

  echo "========================================"
  echo "Running knockout: $name"
  echo "Config: $cfg"
  echo "Output: $out_dir"
  echo "========================================"

  # Ensure benchmark_finetuned reads the intended config
  python benchmark_finetuned.py \
    --all_finetuned \
    --finetuned_config "$cfg" \
    --workers 1

  # Preserve each run separately
  if [ -d "./experiment-finetuned/all_finetuned" ]; then
    rm -rf "$out_dir"
    mv "./experiment-finetuned/all_finetuned" "$out_dir"
  fi

done

echo "All single-knockout runs finished."
