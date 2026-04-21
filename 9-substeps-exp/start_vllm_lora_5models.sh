#!/bin/bash
set -euo pipefail

BASE_MODEL="Qwen/Qwen3-Coder-30B-A3B-Instruct"

vllm serve "$BASE_MODEL" \
  --host 127.0.0.1 \
  --port 8000 \
  --tensor-parallel-size 4 \
  --dtype bfloat16 \
  --api-key EMPTY \
  --enable-lora \
  --lora-modules \
    plan=/mnt/lustre/rpi/pxu10/5tune/plan/plan_results \
    initial_write=/mnt/lustre/rpi/pxu10/5tune/initial_write/initial_write_results \
    review_analysis=/mnt/lustre/rpi/pxu10/5tune/review_analysis/review_analysis_results \
    review_plan=/mnt/lustre/rpi/pxu10/5tune/review_plan/review_plan_results \
    rewrite=/mnt/lustre/rpi/pxu10/5tune/rewrite/rewrite_results
