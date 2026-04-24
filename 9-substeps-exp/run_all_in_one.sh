#!/bin/bash
set -euo pipefail

source /mnt/lustre/rpi/pxu10/agent/bin/activate
cd /mnt/lustre/rpi/pxu10/official

BASE_MODEL="Qwen/Qwen3-Coder-30B-A3B-Instruct"
PORT=8000

export OPENAI_API_KEY="EMPTY"
export OPENAI_BASE_URL="http://127.0.0.1:${PORT}/v1"
export OPENAI_API_BASE="http://127.0.0.1:${PORT}/v1"
unset OPENAI_ORG_ID
unset OPENAI_ORGANIZATION

echo "[1/4] Starting vLLM with Qwen3 base + 9 LoRA adapters..."
vllm serve "$BASE_MODEL" \
  --host 127.0.0.1 \
  --port ${PORT} \
  --tensor-parallel-size 4 \
  --dtype bfloat16 \
  --api-key EMPTY \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --max-model-len 65536 \
  --trust-remote-code \
  --enable-lora \
  --max-lora-rank 32 \
  --lora-modules \
    parse_case_info=/mnt/lustre/rpi/pxu10/9tune/parse_case_info/parse_case_info_results \
    build_advice=/mnt/lustre/rpi/pxu10/9tune/build_advice/build_advice_results \
    decompose_subtasks=/mnt/lustre/rpi/pxu10/9tune/decompose_subtasks/decompose_subtasks_results \
    generate_file=/mnt/lustre/rpi/pxu10/9tune/generate_file/generate_file_results \
    allrun_commands=/mnt/lustre/rpi/pxu10/9tune/allrun_commands/allrun_commands_results \
    allrun_script=/mnt/lustre/rpi/pxu10/9tune/allrun_script/allrun_script_results \
    error_analysis=/mnt/lustre/rpi/pxu10/9tune/error_analysis/error_analysis_results \
    rewrite_plan=/mnt/lustre/rpi/pxu10/9tune/rewrite_plan/rewrite_plan_results \
    rewrite_files=/mnt/lustre/rpi/pxu10/5tune/rewrite/rewrite_results \
  > ./9-substeps-exp/vllm_lora.log 2>&1 &

VLLM_PID=$!
trap 'kill "$VLLM_PID" 2>/dev/null || true' EXIT

echo "[2/4] Waiting for vLLM to become ready..."
python - <<'PY'
import time, requests
url = "http://127.0.0.1:8000/v1/models"
for i in range(240):
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

echo "[3/4] Running all-finetuned 9-substep benchmark..."
python benchmark_finetuned.py \
  --all_finetuned \
  --finetuned_config "./9-substeps-exp/finetuned_models.json" \
  --workers 1

echo "[4/4] Summarizing results..."
python - <<'PY'
import os
from benchmark_utils import DATASETS_BASIC, CASES_BASIC

base = './experiment-finetuned/all_finetuned/runs'

def case_success(case_path: str) -> bool:
    if not os.path.isdir(case_path):
        return False
    logs = [f for f in os.listdir(case_path) if f.startswith('log.') and f.endswith('Foam')]
    if not logs:
        return False
    for lf in logs:
        p = os.path.join(case_path, lf)
        try:
            with open(p, 'r', encoding='utf-8', errors='ignore') as fh:
                lines = fh.readlines()
            if len(lines) > 1 and lines[-2].strip() == 'End':
                return True
        except Exception:
            pass
    return False

rows=[]; ok=0
for d in DATASETS_BASIC:
    for c in CASES_BASIC:
        s = case_success(os.path.join(base,d,str(c)))
        rows.append((d,c,s)); ok += int(s)

total=len(rows)
out='./experiment-finetuned/all_finetuned/summary.txt'
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out,'w',encoding='utf-8') as f:
    f.write(f'Success: {ok}/{total} ({(100*ok/total if total else 0):.2f}%)\n')
    f.write('Failed cases:\n')
    for d,c,s in rows:
        if not s:
            f.write(f'- {d}/{c}\n')

print(f'Success: {ok}/{total} ({(100*ok/total if total else 0):.2f}%)')
print(f'Wrote: {out}')
PY

echo "Done. vLLM log: ./9-substeps-exp/vllm_lora.log"
