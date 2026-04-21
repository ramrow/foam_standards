#!/bin/bash
set -o pipefail

# 1. Activate environment
source /mnt/lustre/rpi/pxu10/agent/bin/activate

# 2. Set variables
MODEL_NAME="Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled"
PROJECT_TAG="qwen3.5-27b-claude-4.6-opus-benchmark"
export VLLM_BASE_URL="http://127.0.0.1:8000/v1"

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
  --max-model-len 262144 \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  > vllm_server.log 2>&1 &

# Capture the process ID so it cleanly exits when the script finishes
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
