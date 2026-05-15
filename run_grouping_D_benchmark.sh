#!/bin/bash
set -euo pipefail

ROOT="/pscratch/sd/p/peijingx/troubleshoot"
VENV_ROOT="/pscratch/sd/p/peijingx/ablation"
source "${VENV_ROOT}/.venv/bin/activate"
cd "$ROOT"

PORT="${PORT:-8000}"
RUN_TAG="${1:-default}"
BASE_MODEL="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
CFG_PATH="$ROOT/grouping_D_finetuned_models.json"
OUT_ROOT="$ROOT/benchmark_grouping_D_${RUN_TAG}"
LOG_PATH="$ROOT/vllm_grouping_D_${RUN_TAG}.log"

source "${VENV_ROOT}/secrets/openai_env.sh"
export OPENAI_API_KEY="EMPTY"
export OPENAI_BASE_URL="http://127.0.0.1:${PORT}/v1"
export OPENAI_API_BASE="http://127.0.0.1:${PORT}/v1"
export OPENAI_EMBED_BASE_URL="${OPENAI_EMBED_BASE_URL:-https://api.openai.com/v1}"
[[ -z "${OPENAI_EMBED_API_KEY:-}" ]] && { echo "ERROR: OPENAI_EMBED_API_KEY missing"; exit 1; }
unset OPENAI_ORG_ID OPENAI_ORGANIZATION
export FOAMAGENT_FORCE_LOCAL_RUN=1
unset FOAMAGENT_FORCE_HPC_RUN

mkdir -p "$OUT_ROOT"

vllm serve "$BASE_MODEL" \
  --host 127.0.0.1 --port ${PORT} --tensor-parallel-size 4 --dtype bfloat16 --api-key EMPTY \
  --enable-auto-tool-choice --tool-call-parser hermes --max-model-len 131072 --trust-remote-code \
  --enable-lora --max-lora-rank 32 \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --override-generation-config '{"repetition_penalty": 1.1}' \
  --lora-modules \
    d_m1="$ROOT/grouping_D_models/model_1/adapter" \
    d_m2="$ROOT/grouping_D_models/model_2/adapter" \
    d_m3="$ROOT/grouping_D_models/model_3/adapter" \
  > "$LOG_PATH" 2>&1 &
VLLM_PID=$!
trap 'kill "$VLLM_PID" 2>/dev/null || true' EXIT

python - <<'PY'
import os,time,requests
port=os.environ.get('PORT','8000')
url=f'http://127.0.0.1:{port}/v1/models'
for _ in range(900):
    try:
        r=requests.get(url,headers={'Authorization':'Bearer EMPTY'},timeout=5)
        if r.status_code==200: break
    except Exception: pass
    time.sleep(2)
else: raise SystemExit('vLLM did not become ready in time')
PY

python benchmark_finetuned.py --all_finetuned --finetuned_config "$CFG_PATH" --output_root "$OUT_ROOT" --workers 1
