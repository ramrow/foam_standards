#!/bin/bash
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -p rpi
#SBATCH -A rpi
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:4
#SBATCH --time=25:00:00
#SBATCH --output=%x_out
#SBATCH --error=%x_err

set -euo pipefail

cd /mnt/lustre/rpi/pxu10/standard
source /mnt/lustre/rpi/pxu10/agent/bin/activate

PROJECT_TAG="qwen-standard"
export VLLM_BASE_URL="http://127.0.0.1:8000/v1"

vllm serve Qwen/Qwen3-Coder-30B-A3B-Instruct \
  --host 127.0.0.1 \
  --port 8000 \
  --tensor-parallel-size 4 \
  --dtype bfloat16 \
  --api-key EMPTY \
  --enable-lora \
  --max-lora-rank 32 \
  --lora-modules foamqwen=finalform/foamQwen-30B \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --max-model-len 65536 \
  --trust-remote-code \
  > vllm_server.log 2>&1 &

VLLM_PID=$!
trap 'kill "$VLLM_PID" 2>/dev/null || true' EXIT

python - <<'PY'
import time
import requests
for _ in range(240):
    try:
        r = requests.get("http://127.0.0.1:8000/v1/models", headers={"Authorization": "Bearer EMPTY"}, timeout=5)
        if r.status_code == 200:
            break
    except Exception:
        pass
    time.sleep(5)
else:
    raise SystemExit(1)
PY

python benchmark.py --model_tag "$PROJECT_TAG" --cases 1 2 3
python summarize_benchmark.py --model_tag "$PROJECT_TAG" --save_json "$PROJECT_TAG/summary.json"