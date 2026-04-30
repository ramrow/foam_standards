#!/bin/bash
set -euo pipefail

cd /pscratch/sd/p/peijingx/ablation

echo "Stopping local vLLM / router processes if running..."
pkill -f "vllm serve" || true
pkill -f "uvicorn.run\(app, host='127.0.0.1', port=8000" || true
pkill -f "run_single_knockout_one.sh" || true

echo "Removing transient benchmark output folder (if present)..."
rm -rf ./experiment-finetuned/all_finetuned

echo "Keeping preserved outputs under ./experiment-finetuned/single_knockout"
echo "Cleanup complete."
