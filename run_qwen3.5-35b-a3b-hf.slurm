#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p rpi
#SBATCH -A rpi
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:4
#SBATCH --cpus-per-task=32
#SBATCH --time=08:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

cd /mnt/lustre/rpi/pxu10/criteria
source /mnt/lustre/rpi/pxu10/agent/bin/activate

PROJECT_TAG="qwen3.5-35b-a3b-hf"

# Foam-Agent HF runtime
export FOAMAGENT_MODEL_PROVIDER="huggingface"
export FOAMAGENT_MODEL_VERSION="Qwen/Qwen3.5-35B-A3B"


# Reduce tokenizer parallel thread contention on HPC
export TOKENIZERS_PARALLELISM=false

echo "Starting benchmark with provider=$FOAMAGENT_MODEL_PROVIDER model=$FOAMAGENT_MODEL_VERSION"
python -u benchmark.py --model_tag "$PROJECT_TAG" --cases 1 2 3
python -u summarize_benchmark.py --model_tag "$PROJECT_TAG" --save_json "$PROJECT_TAG/summary.json"

echo "Done: $PROJECT_TAG"
