#!/bin/bash
set -euo pipefail

# Usage:
#   bash run_no_similar_allrun_commands.sh [RUN_NAME]
# Default RUN_NAME targets allrun_commands knockout config.

RUN_NAME="${1:-knockout_allrun_commands}"

source /pscratch/sd/p/peijingx/ablation/.venv/bin/activate
cd /pscratch/sd/p/peijingx/ablation

export OPENAI_API_KEY="EMPTY"
export OPENAI_BASE_URL="http://127.0.0.1:8000/v1"
export OPENAI_API_BASE="http://127.0.0.1:8000/v1"

# Key ablation switch: disable similar-case references only in allrun generation.

CFG_DIR="/pscratch/sd/p/peijingx/ablation/9-substeps-exp/single_knockout_configs"
CFG_PATH="$CFG_DIR/${RUN_NAME}.json"

if [[ ! -f "$CFG_PATH" ]]; then
  echo "Config not found: $CFG_PATH"
  exit 1
fi

TS="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="/pscratch/sd/p/peijingx/ablation/experiment-no-similar-test/single_knockout/${RUN_NAME}_noSimilarAllrun_${TS}"
RUNS_DIR="$OUT_ROOT/runs"
RESULTS_DIR="$OUT_ROOT/results"
mkdir -p "$RUNS_DIR" "$RESULTS_DIR"

echo "[1/2] Running benchmark with DISABLE_SIMILAR_FOR_ALLRUN=1"
python benchmark_finetuned.py \
  --single_knockout \
  --finetuned_config "$CFG_PATH" \
  --run_name "${RUN_NAME}_noSimilarAllrun_${TS}" \
  --output_root "$OUT_ROOT" \
  --workers 1

echo "[2/2] Done. Outputs preserved at: $OUT_ROOT"

