#!/bin/bash
set -euo pipefail

# Usage:
#   bash run_single_knockout_one.sh knockout_parse_case_info
#   bash run_single_knockout_one.sh baseline

source /pscratch/sd/p/peijingx/ablation/.venv/bin/activate
cd "/pscratch/sd/p/peijingx/ablation"

BASE_MODEL="unsloth/Nemotron-3-Nano-30B-A3B"
PORT=8000

# Source secrets (can define OPENAI_EMBED_API_KEY / OPENAI_EMBED_BASE_URL)
source /pscratch/sd/p/peijingx/ablation/secrets/openai_env.sh

# Chat/completions must go to local vLLM
export OPENAI_API_KEY="EMPTY"
export OPENAI_BASE_URL="http://127.0.0.1:${PORT}/v1"
export OPENAI_API_BASE="http://127.0.0.1:${PORT}/v1"

# Embeddings must go to real OpenAI
export OPENAI_EMBED_BASE_URL="${OPENAI_EMBED_BASE_URL:-https://api.openai.com/v1}"
if [[ -z "${OPENAI_EMBED_API_KEY:-}" ]]; then
  echo "ERROR: OPENAI_EMBED_API_KEY is not set. Put it in /pscratch/sd/p/peijingx/ablation/secrets/openai_env.sh"
  exit 1
fi

unset OPENAI_ORG_ID
unset OPENAI_ORGANIZATION

CFG_DIR="./9-substeps-exp/single_knockout_configs"
OUT_ROOT="./experiment-finetuned/single_knockout"
mkdir -p "$OUT_ROOT"

RUN_NAME="${1:-}"
if [[ -z "$RUN_NAME" ]]; then
  echo "ERROR: Missing run name."
  echo "Use: baseline or knockout_<substep>"
  ls "$CFG_DIR"/knockout_*.json 2>/dev/null | sed 's#.*/##; s#\.json$##' || true
  exit 1
fi

if [[ "$RUN_NAME" == "baseline" ]]; then
  CFG_PATH="./9-substeps-exp/finetuned_models.json"
  OUT_DIR="$OUT_ROOT/baseline_all_finetuned"
  TEMP_ROOT="$OUT_DIR"
else
  CFG_PATH="$CFG_DIR/${RUN_NAME}.json"
  OUT_DIR="$OUT_ROOT/${RUN_NAME}"
  TEMP_ROOT="$OUT_DIR"
fi

mkdir -p "$TEMP_ROOT/runs" "$TEMP_ROOT/results"

if [[ ! -f "$CFG_PATH" ]]; then
  echo "ERROR: Config not found: $CFG_PATH"
  exit 1
fi

echo "OPENAI_BASE_URL=$OPENAI_BASE_URL"
echo "OPENAI_API_KEY=$OPENAI_API_KEY"
echo "OPENAI_EMBED_BASE_URL=$OPENAI_EMBED_BASE_URL"
echo "OPENAI_EMBED_API_KEY_SET=${OPENAI_EMBED_API_KEY:+yes}"

echo "[1/4] Starting vLLM..."
vllm serve "$BASE_MODEL" \
  --host 127.0.0.1 \
  --port ${PORT} \
  --tensor-parallel-size 4 \
  --dtype bfloat16 \
  --api-key EMPTY \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --max-model-len 16384 \
  --trust-remote-code \
  --enable-lora \
  --max-lora-rank 32 \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --lora-modules \
    parse_case_info=/pscratch/sd/p/peijingx/ablation/nemo/parse_case_info/final_adapter \
    build_advice=/pscratch/sd/p/peijingx/ablation/nemo/build_advice/final_adapter \
    decompose_subtasks=/pscratch/sd/p/peijingx/ablation/nemo/decompose_subtasks/final_adapter \
    generate_file=/pscratch/sd/p/peijingx/ablation/nemo/generate_file/final_adapter \
    allrun_commands=/pscratch/sd/p/peijingx/ablation/nemo/allrun_commands/final_adapter \
    allrun_script=/pscratch/sd/p/peijingx/ablation/nemo/allrun_script/final_adapter \
    error_analysis=/pscratch/sd/p/peijingx/ablation/nemo/error_analysis/final_adapter \
    rewrite_plan=/pscratch/sd/p/peijingx/ablation/nemo/rewrite_plan/final_adapter \
    rewrite_files=/pscratch/sd/p/peijingx/ablation/nemo/rewrite_files/final_adapter \
  > ./9-substeps-exp/vllm_one_${RUN_NAME}.log 2>&1 &
VLLM_PID=$!
trap 'kill "$VLLM_PID" 2>/dev/null || true' EXIT

echo "[2/4] Waiting for vLLM ready..."
python - <<'PY'
import time, requests
url = "http://127.0.0.1:8000/v1/models"
for _ in range(900):
    try:
        r = requests.get(url, headers={"Authorization": "Bearer EMPTY"}, timeout=5)
        if r.status_code == 200:
            print("vLLM ready")
            break
    except Exception:
        pass
    time.sleep(2)
else:
    raise SystemExit("vLLM did not become ready in time")
PY

echo "[3/4] Running benchmark for: $RUN_NAME"
python benchmark_finetuned.py \
  --all_finetuned \
  --finetuned_config "$CFG_PATH" \
  --output_root "$TEMP_ROOT" \
  --workers 1


echo "[4/4] Done: $RUN_NAME"
echo "Output: $OUT_DIR"

