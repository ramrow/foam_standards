#!/bin/bash
set -euo pipefail

# Usage:
#   bash run_baseline_all_finetuned.sh baseline
#   bash run_baseline_all_finetuned.sh knockout_parse_case_info

ROOT="/pscratch/sd/p/peijingx/troubleshoot"
VENV_ROOT="/pscratch/sd/p/peijingx/ablation"

source "${VENV_ROOT}/.venv/bin/activate"
cd "$ROOT"

BASE_MODEL="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
PORT=8000

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

RUN_NAME="${1:-}"
if [[ -z "$RUN_NAME" ]]; then
  echo "ERROR: Missing run name. Use baseline or knockout_<substep>"
  exit 1
fi

CFG_DIR="./9-substeps-exp/single_knockout_configs"
OUT_ROOT_BASE="$ROOT/experiment-finetuned/single_knockout"

if [[ "$RUN_NAME" == "baseline" ]]; then
  CFG_PATH="./9-substeps-exp/finetuned_models.json"
  OUT_ROOT="$OUT_ROOT_BASE/baseline_all_finetuned"
else
  CFG_PATH="$CFG_DIR/${RUN_NAME}.json"
  OUT_ROOT="$OUT_ROOT_BASE/${RUN_NAME}"
fi

if [[ ! -f "$CFG_PATH" ]]; then
  echo "ERROR: Config not found: $CFG_PATH"
  exit 1
fi
mkdir -p "$OUT_ROOT"

echo "[1/5] Starting vLLM..."
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
    parse_case_info=${VENV_ROOT}/nemo/parse_case_info/final_adapter \
    build_advice=${VENV_ROOT}/nemo/build_advice/final_adapter \
    decompose_subtasks=${VENV_ROOT}/nemo/decompose_subtasks/final_adapter \
    generate_file=${VENV_ROOT}/nemo/generate_file/final_adapter \
    allrun_commands=${VENV_ROOT}/nemo/allrun_commands/final_adapter \
    allrun_script=${VENV_ROOT}/nemo/allrun_script/final_adapter \
    error_analysis=${VENV_ROOT}/nemo/error_analysis/final_adapter \
    rewrite_plan=${VENV_ROOT}/nemo/rewrite_plan/final_adapter \
    rewrite_files=${VENV_ROOT}/nemo/rewrite_files/final_adapter \
  > "$ROOT/9-substeps-exp/vllm_one_${RUN_NAME}.log" 2>&1 &
VLLM_PID=$!
trap 'kill "$VLLM_PID" 2>/dev/null || true' EXIT

echo "[2/5] Waiting for vLLM ready..."
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

echo "[3/5] Verifying LoRA model aliases are exposed..."
python - <<'PY'
import requests, sys
expected = {
    "parse_case_info","build_advice","decompose_subtasks","generate_file",
    "allrun_commands","allrun_script","error_analysis","rewrite_plan","rewrite_files"
}
r = requests.get("http://127.0.0.1:8000/v1/models", headers={"Authorization":"Bearer EMPTY"}, timeout=10)
r.raise_for_status()
ids = {m.get("id") for m in r.json().get("data", [])}
missing = sorted(expected - ids)
print("Found aliases:", sorted(expected & ids))
if missing:
    print("Missing aliases:", missing)
    sys.exit(2)
PY

echo "[4/5] Running benchmark for: $RUN_NAME"
python benchmark_finetuned.py \
  --all_finetuned \
  --finetuned_config "$CFG_PATH" \
  --output_root "$OUT_ROOT" \
  --workers 1

echo "[5/5] Done: $RUN_NAME"
echo "Output root: $OUT_ROOT"
