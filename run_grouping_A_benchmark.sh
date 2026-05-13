#!/bin/bash
set -euo pipefail

ROOT="/pscratch/sd/p/peijingx/troubleshoot"
VENV_ROOT="/pscratch/sd/p/peijingx/ablation"

source "${VENV_ROOT}/.venv/bin/activate"
cd "$ROOT"

BASE_MODEL="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
PORT="${PORT:-8000}"
RUN_TAG="${1:-default}"

CFG_PATH="$ROOT/grouping_A_finetuned_models.json"
OUT_ROOT="$ROOT/benchmark_grouping_A_${RUN_TAG}"
LOG_PATH="$ROOT/vllm_grouping_A_${RUN_TAG}.log"

source "${VENV_ROOT}/secrets/openai_env.sh"

export OPENAI_API_KEY="EMPTY"
export OPENAI_BASE_URL="http://127.0.0.1:${PORT}/v1"
export OPENAI_API_BASE="http://127.0.0.1:${PORT}/v1"
export OPENAI_EMBED_BASE_URL="${OPENAI_EMBED_BASE_URL:-https://api.openai.com/v1}"

if [[ -z "${OPENAI_EMBED_API_KEY:-}" ]]; then
  echo "ERROR: OPENAI_EMBED_API_KEY missing in ${VENV_ROOT}/secrets/openai_env.sh"
  exit 1
fi

unset OPENAI_ORG_ID
unset OPENAI_ORGANIZATION
export FOAMAGENT_FORCE_LOCAL_RUN=1
unset FOAMAGENT_FORCE_HPC_RUN

if [[ ! -f "$CFG_PATH" ]]; then
  echo "ERROR: Config not found: $CFG_PATH"
  exit 1
fi
mkdir -p "$OUT_ROOT"

echo "[1/5] Starting vLLM (Grouping A)..."
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
  --override-generation-config '{"repetition_penalty": 1.1}' \
  --lora-modules \
    a_plan="$ROOT/grouping_A_models/plan/adapter" \
    a_write_exec="$ROOT/grouping_A_models/write_exec/adapter" \
    a_review="$ROOT/grouping_A_models/review/adapter" \
  > "$LOG_PATH" 2>&1 &
VLLM_PID=$!
trap 'kill "$VLLM_PID" 2>/dev/null || true' EXIT

echo "[2/5] Waiting for vLLM ready..."
python - <<'PY'
import os, time, requests
port=os.environ.get('PORT','8000')
url=f"http://127.0.0.1:{port}/v1/models"
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

echo "[3/5] Verifying LoRA aliases..."
python - <<'PY'
import os, requests, sys
port=os.environ.get('PORT','8000')
expected={"a_plan","a_write_exec","a_review"}
r=requests.get(f"http://127.0.0.1:{port}/v1/models", headers={"Authorization":"Bearer EMPTY"}, timeout=10)
r.raise_for_status()
ids={m.get("id") for m in r.json().get("data",[])}
missing=sorted(expected-ids)
print("Found aliases:", sorted(expected & ids))
if missing:
    print("Missing aliases:", missing)
    sys.exit(2)
PY

echo "[4/5] Running benchmark (Grouping A)"
python benchmark_finetuned.py --all_finetuned --finetuned_config "$CFG_PATH" --output_root "$OUT_ROOT" --workers 1

echo "[5/5] Done"
echo "Output root: $OUT_ROOT"
