#!/bin/bash
set -euo pipefail

source /mnt/lustre/rpi/pxu10/agent/bin/activate
cd /mnt/lustre/rpi/pxu10/official

BASE_MODEL="Qwen/Qwen3-Coder-30B-A3B-Instruct"

# Upstream split vLLM servers
PORT_A=8100
PORT_B=8101
# Router public endpoint for benchmark/src
ROUTER_PORT=8000

export OPENAI_API_KEY="EMPTY"
export OPENAI_BASE_URL="http://127.0.0.1:${ROUTER_PORT}/v1"
export OPENAI_API_BASE="http://127.0.0.1:${ROUTER_PORT}/v1"
unset OPENAI_ORG_ID
unset OPENAI_ORGANIZATION

mkdir -p ./9-substeps-exp/logs

echo "[1/5] Starting split vLLM servers: (GPU0,1)=4 models, (GPU2,3)=5 models"

CUDA_VISIBLE_DEVICES=0,1 vllm serve "$BASE_MODEL" \
  --host 127.0.0.1 \
  --port ${PORT_A} \
  --tensor-parallel-size 2 \
  --dtype bfloat16 \
  --api-key EMPTY \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --max-model-len 65536 \
  --trust-remote-code \
  --enable-lora \
  --max-lora-rank 32 \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --lora-modules \
    parse_case_info=/mnt/lustre/rpi/pxu10/9tune/parse_case_info/parse_case_info_results \
    build_advice=/mnt/lustre/rpi/pxu10/9tune/build_advice/build_advice_results \
    decompose_subtasks=/mnt/lustre/rpi/pxu10/9tune/decompose_subtasks/decompose_subtasks_results \
    generate_file=/mnt/lustre/rpi/pxu10/9tune/generate_file/generate_file_results \
  > ./9-substeps-exp/logs/vllm_gpu01.log 2>&1 &
PID_A=$!

CUDA_VISIBLE_DEVICES=2,3 vllm serve "$BASE_MODEL" \
  --host 127.0.0.1 \
  --port ${PORT_B} \
  --tensor-parallel-size 2 \
  --dtype bfloat16 \
  --api-key EMPTY \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --max-model-len 65536 \
  --trust-remote-code \
  --enable-lora \
  --max-lora-rank 32 \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --lora-modules \
    allrun_commands=/mnt/lustre/rpi/pxu10/9tune/allrun_commands/allrun_commands_results \
    allrun_script=/mnt/lustre/rpi/pxu10/9tune/allrun_script/allrun_script_results \
    error_analysis=/mnt/lustre/rpi/pxu10/9tune/error_analysis/error_analysis_results \
    rewrite_plan=/mnt/lustre/rpi/pxu10/9tune/rewrite_plan/rewrite_plan_results \
    rewrite_files=/mnt/lustre/rpi/pxu10/9tune/rewrite_files/rewrite_files_results \
  > ./9-substeps-exp/logs/vllm_gpu23.log 2>&1 &
PID_B=$!

cleanup() {
  kill "$PID_R" "$PID_A" "$PID_B" 2>/dev/null || true
}
trap cleanup EXIT

echo "[2/5] Waiting for split servers..."
python - <<PY
import time, requests
for p in (${PORT_A}, ${PORT_B}):
    url=f"http://127.0.0.1:{p}/v1/models"
    for _ in range(900):
        try:
            r=requests.get(url, headers={"Authorization":"Bearer EMPTY"}, timeout=5)
            if r.status_code==200:
                print(f"vLLM ready on :{p}")
                break
        except Exception:
            pass
        time.sleep(2)
    else:
        raise SystemExit(f"vLLM on :{p} did not become ready")
PY

echo "[3/5] Starting alias router on :${ROUTER_PORT} -> :${PORT_A}, :${PORT_B}"
python - <<'PY' > ./9-substeps-exp/logs/router.log 2>&1 &
from fastapi import FastAPI, Request, Response
import requests, uvicorn

app = FastAPI()

A = "http://127.0.0.1:8100"
B = "http://127.0.0.1:8101"

A_MODELS = {"parse_case_info", "build_advice", "decompose_subtasks", "generate_file"}
B_MODELS = {"allrun_commands", "allrun_script", "error_analysis", "rewrite_plan", "rewrite_files"}

@app.get('/v1/models')
def models():
    ma = requests.get(f"{A}/v1/models", headers={"Authorization":"Bearer EMPTY"}, timeout=30).json()
    mb = requests.get(f"{B}/v1/models", headers={"Authorization":"Bearer EMPTY"}, timeout=30).json()
    da = ma.get('data', []) if isinstance(ma, dict) else []
    db = mb.get('data', []) if isinstance(mb, dict) else []
    return {"object":"list","data": da + db}

@app.post('/v1/chat/completions')
async def chat(request: Request):
    body = await request.json()
    model = body.get("model", "")
    target = A if model in A_MODELS else B if model in B_MODELS else A
    r = requests.post(f"{target}/v1/chat/completions", json=body, headers={"Authorization":"Bearer EMPTY"}, timeout=600)
    return Response(content=r.content, status_code=r.status_code, media_type=r.headers.get('content-type','application/json'))

@app.post('/v1/completions')
async def completions(request: Request):
    body = await request.json()
    model = body.get("model", "")
    target = A if model in A_MODELS else B if model in B_MODELS else A
    r = requests.post(f"{target}/v1/completions", json=body, headers={"Authorization":"Bearer EMPTY"}, timeout=600)
    return Response(content=r.content, status_code=r.status_code, media_type=r.headers.get('content-type','application/json'))

uvicorn.run(app, host='127.0.0.1', port=8000, log_level='info')
PY
PID_R=$!

echo "[4/5] Waiting for router ready..."
python - <<'PY'
import time, requests
url = "http://127.0.0.1:8000/v1/models"
for _ in range(300):
    try:
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            print("Router ready")
            break
    except Exception:
        pass
    time.sleep(1)
else:
    raise SystemExit("Router did not become ready")
PY

echo "[5/5] Running benchmark + summary..."
python benchmark_finetuned.py \
  --all_finetuned \
  --finetuned_config "./9-substeps-exp/finetuned_models.json" \
  --workers 1

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

echo "Done. Logs:"
echo "  - ./9-substeps-exp/logs/vllm_gpu01.log"
echo "  - ./9-substeps-exp/logs/vllm_gpu23.log"
echo "  - ./9-substeps-exp/logs/router.log"

wait

