#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p rpi
#SBATCH -A rpi
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:4
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --job-name=qwen3-coder-next-benchmark

set -euo pipefail

cd /mnt/lustre/rpi/pxu10/criteria

bash run_benchmark.sh
