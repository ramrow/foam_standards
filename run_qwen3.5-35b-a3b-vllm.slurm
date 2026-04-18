#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p rpi
#SBATCH -A rpi
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:1
#SBATCH --time=06:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

cd /mnt/lustre/rpi/pxu10/criteria
source /mnt/lustre/rpi/pxu10/agent/bin/activate

PROJECT_TAG="qwen3.5-35b-a3b-vllm"
export VLLM_BASE_URL="http://127.0.0.1:8000/v1"

vllm serve Qwen/Qwen3.5-35B-A3B \
  --host 127.0.0.1 \
  --port 8000 \
  --tensor-parallel-size 1 \
  --reasoning-parser qwen3 \
  --dtype bfloat16 \
  --api-key EMPTY \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --max-model-len 32768 \
  > vllm_server_${PROJECT_TAG}.log 2>&1 &

VLLM_PID=$!
trap 'kill "$VLLM_PID" 2>/dev/null || true' EXIT

python - <<'PY'
import time, requests
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

