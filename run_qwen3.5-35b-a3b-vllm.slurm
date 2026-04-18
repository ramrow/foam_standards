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

PROJECT_TAG="qwen3.5-35b-a3b-hf"
export FOAMAGENT_MODEL_PROVIDER="huggingface"
export FOAMAGENT_MODEL_VERSION="Qwen/Qwen3.5-35B-A3B"

# Optional cache locations on cluster (adjust if needed)
export HF_HOME="${HF_HOME:-/mnt/lustre/rpi/pxu10/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/mnt/lustre/rpi/pxu10/.cache/huggingface/transformers}"

python benchmark.py --model_tag "$PROJECT_TAG" --cases 1 2 3
python summarize_benchmark.py --model_tag "$PROJECT_TAG" --save_json "$PROJECT_TAG/summary.json"
