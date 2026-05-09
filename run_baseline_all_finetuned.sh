#!/bin/bash
set -euo pipefail

# Usage:
#   bash run_baseline_all_finetuned.sh baseline
#   bash run_baseline_all_finetuned.sh knockout_parse_case_info
#   bash run_baseline_all_finetuned.sh knockout_allrun_commands

source /pscratch/sd/p/peijingx/ablation/.venv/bin/activate
cd /pscratch/sd/p/peijingx/debug

BASE_MODEL="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
PORT=8000

source /pscratch/sd/p/peijingx/ablation/secrets/openai_env.sh

export OPENAI_API_KEY="EMPTY"
export OPENAI_BASE_URL="http://127.0.0.1:${PORT}/v1"
export OPENAI_API_BASE="http://127.0.0.1:${PORT}/v1"
export OPENAI_EMBED_BASE_URL="${OPENAI_EMBED_BASE_URL:-https://api.openai.com/v1}"

if [[ -z "${OPENAI_EMBED_API_KEY:-}" ]]; then
  echo "ERROR: OPENAI_EMBED_API_KEY missing in /pscratch/sd/p/peijingx/ablation/secrets/openai_env.sh"
  exit 1
fi

unset OPENAI_ORG_ID
unset OPENAI_ORGANIZATION
export FOAMAGENT_FORCE_LOCAL_RUN=1
unset FOAMAGENT_FORCE_HPC_RUN

RUN_NAME="${1:-}"
if [[ -z "$RUN_NAME" ]]; then
  echo "ERROR: Missing run name. Use baseline or knockout_<substep>"
  exit 1
fi

CFG_DIR="./9-substeps-exp/single_knockout_configs"
if [[ "$RUN_NAME" == "baseline" ]]; then
  CFG_PATH="./9-substeps-exp/finetuned_models.json"
  OUT_ROOT="/pscratch/sd/p/peijingx/debug/experiment-finetuned/single_knockout/baseline_all_finetuned"
else
  CFG_PATH="$CFG_DIR/${RUN_NAME}.json"
  OUT_ROOT="/pscratch/sd/p/peijingx/debug/experiment-finetuned/single_knockout/${RUN_NAME}"
fi

if [[ ! -f "$CFG_PATH" ]]; then
  echo "ERROR: Config not found: $CFG_PATH"
  exit 1
fi
mkdir -p "$OUT_ROOT"

echo "[1/4] Starting vLLM..."
vllm serve "$BASE_MODEL" \
  --host 127.0.0.1 \
  --port ${PORT} \
  --tensor-parallel-size 4 \
  --dtype bfloat16 \
  --api-key EMPTY \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --max-model-len 131072 \
  --trust-remote-code \
  --enable-lora \
  --max-lora-rank 32 \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --override-generation-config '{"temperature": 0.1, "repetition_penalty": 1.2, "top_p": 1.0}' \
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
  > /pscratch/sd/p/peijingx/debug/vllm_one_${RUN_NAME}.log 2>&1 &
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
  --output_root "$OUT_ROOT" \
  --workers 1

echo "[4/4] Done: $RUN_NAME"
echo "Output root: $OUT_ROOT"
