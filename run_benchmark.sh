#!/bin/bash

set -o pipefail

# 1. Activate environment
source /mnt/lustre/rpi/pxu10/agent/bin/activate
cd /mnt/lustre/rpi/pxu10/criteria

# 2. Set variables
MODEL_NAME="Qwen/Qwen3-Coder-Next"
PROJECT_TAG="qwen3-coder-next-benchmark"
export OPENAI_BASE_URL="http://127.0.0.1:8000/v1"
export OPENAI_API_KEY="EMPTY"
export VLLM_BASE_URL="http://127.0.0.1:8000/v1"
export FOAMAGENT_FORCE_LOCAL_RUN="1"
export FOAMAGENT_DISABLE_VISUALIZATION="1"
export FOAMAGENT_TEMPERATURE="0.1"

# 3. Start vLLM Server in the background
echo "Starting vLLM server..."
vllm serve "$MODEL_NAME" \
  --host 127.0.0.1 \
  --port 8000 \
  --tensor-parallel-size 4 \
  --dtype bfloat16 \
  --api-key EMPTY \
  --tool-call-parser hermes \
  --enable-auto-tool-choice \
  --max-model-len 65536 \
  --trust-remote-code \
  > vllm_server.log 2>&1 &

VLLM_PID=$!
trap 'kill "$VLLM_PID" 2>/dev/null || true' EXIT

# 4. Wait for the server to be ready
echo "Waiting for vLLM to spin up (this may take a few minutes)..."
python - <<'PY'
import time
import requests
for _ in range(180):
    try:
        r = requests.get("http://127.0.0.1:8000/v1/models", headers={"Authorization": "Bearer EMPTY"}, timeout=5)
        if r.status_code == 200:
            print("Server is ready!")
            break
    except Exception:
        pass
    time.sleep(5)
else:
    print("Server failed to start in time. Check vllm_server.log")
    raise SystemExit(1)
PY

# 5. Run the benchmarks
echo "Running benchmarks..."
python benchmark.py --model_tag "$PROJECT_TAG" --cases 1 2 3
python summarize_benchmark.py --model_tag "$PROJECT_TAG" --save_json "$PROJECT_TAG/summary.json"

echo "Done!"
