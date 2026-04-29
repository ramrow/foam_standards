#!/bin/bash
#SBATCH -N 1
#SBATCH -G 4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH -q regular
#SBATCH -t 12:00:00
#SBATCH -C gpu&hbm80g
#SBATCH -A m4789
#SBATCH -J gpu_job
#SBATCH -o slurm/%j.out
#SBATCH -e slurm/%j.err
set -euo pipefail

source /pscratch/sd/p/peijingx/ablation/.venv/bin/activate
cd /pscratch/sd/p/peijingx/ablation

BASE_MODEL="unsloth/Nemotron-3-Nano-30B-A3B"
PORT=8000

source /pscratch/sd/p/peijingx/ablation/secrets/openai_env.sh`nexport OPENAI_BASE_URL="http://127.0.0.1:${PORT}/v1"
export OPENAI_API_BASE="http://127.0.0.1:${PORT}/v1"
unset OPENAI_ORG_ID
unset OPENAI_ORGANIZATION

CFG_DIR="./9-substeps-exp/single_knockout_configs"
OUT_ROOT="./experiment-finetuned/single_knockout"
mkdir -p "$OUT_ROOT"

echo "[0/3] Starting vLLM with Nemotron base + 9 LoRA adapters..."
vllm serve "$BASE_MODEL" \
  --host 127.0.0.1 \
  --port ${PORT} \
  --tensor-parallel-size 4 \
  --dtype bfloat16 \
  --api-key EMPTY \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --max-model-len 65536 \
  --trust-remote-code \
  --enable-lora \
  --max-lora-rank 32 \
  --generation-config vllm \
  --override-generation-config '{"temperature": 0.5, "repetition_penalty": 1.1, "top_k": 30, "top_p": 0.7}' \
  --lora-modules \
    parse_case_info=/pscratch/sd/p/peijingx/ablation/nemo/parse_case_info/final_adapter \
    build_advice=/pscratch/sd/p/peijingx/ablation/nemo/build_advice/final_adapter \
    decompose_subtasks=/pscratch/sd/p/peijingx/ablation/nemo/decompose_subtasks/final_adapter \
    generate_file=/pscratch/sd/p/peijingx/ablation/nemo/generate_file/final_adapter \
    allrun_commands=/pscratch/sd/p/peijingx/ablation/nemo/allrun_commands/final_adapter \
    allrun_script=/pscratch/sd/p/peijingx/ablation/nemo/allrun_script/final_adapter \
    error_analysis=/pscratch/sd/p/peijingx/ablation/nemo/error_analysis/final_adapter \
    rewrite_plan=/pscratch/sd/p/peijingx/ablation/nemo/rewrite_plan/final_adapter \
    rewrite_files=/pscratch/sd/p/peijingx/ablation/nemo/rewrite_files/final_adapter \
  > ./9-substeps-exp/vllm_single_knockout.log 2>&1 &

VLLM_PID=$!
trap 'kill "$VLLM_PID" 2>/dev/null || true' EXIT

echo "[1/3] Waiting for vLLM ready..."
python - <<'PY'
import time, requests
url = "http://127.0.0.1:8000/v1/models"
for _ in range(900):
    try:
        r = requests.get(url, headers={"Authorization": "Bearer EMPTY"}, timeout=5)
        if r.status_code == 200:
            print("vLLM ready")
            break
    except Exception:
        pass
    time.sleep(2)
else:
    raise SystemExit("vLLM did not become ready in time")
PY


echo "[2/4] Running baseline (all 9 finetuned)..."
python benchmark_finetuned.py \
  --all_finetuned \
  --finetuned_config "./9-substeps-exp/finetuned_models.json" \
  --workers 1

BASELINE_OUT="$OUT_ROOT/baseline_all_finetuned"
if [ -d "./experiment-finetuned/all_finetuned" ]; then
  rm -rf "$BASELINE_OUT"
  mv "./experiment-finetuned/all_finetuned" "$BASELINE_OUT"
fi

echo "[3/4] Running single-knockout benchmark loop..."
for cfg in "$CFG_DIR"/knockout_*.json; do
  name="$(basename "$cfg" .json)"
  out_dir="$OUT_ROOT/$name"

  echo "========================================"
  echo "Running knockout: $name"
  echo "Config: $cfg"
  echo "Output: $out_dir"
  echo "========================================"

  python benchmark_finetuned.py \
    --all_finetuned \
    --finetuned_config "$cfg" \
    --workers 1

  if [ -d "./experiment-finetuned/all_finetuned" ]; then
    rm -rf "$out_dir"
    mv "./experiment-finetuned/all_finetuned" "$out_dir"
  fi
done

echo "[4/5] All single-knockout runs finished."`necho "vLLM log: ./9-substeps-exp/vllm_single_knockout.log"

echo "[5/5] Summarizing single-knockout ablation results..."
python - <<'PY'
import os
from benchmark_utils import DATASETS_BASIC, CASES_BASIC

out_root = './experiment-finetuned/single_knockout'
summary_path = os.path.join(out_root, 'ablation_summary.txt')

knockouts = sorted([d for d in os.listdir(out_root) if d.startswith('knockout_') and os.path.isdir(os.path.join(out_root, d))])

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

rows = []
for ko in knockouts:
    base = os.path.join(out_root, ko, 'runs')
    total = 0
    ok = 0
    failed = []
    for d in DATASETS_BASIC:
        for c in CASES_BASIC:
            total += 1
            cp = os.path.join(base, d, str(c))
            s = case_success(cp)
            ok += int(s)
            if not s:
                failed.append(f'{d}/{c}')
    rows.append((ko, ok, total, failed))

with open(summary_path, 'w', encoding='utf-8') as f:
    f.write('Single-knockout ablation summary\n')
    f.write('================================\n')
    for ko, ok, total, failed in rows:
        pct = (100.0 * ok / total) if total else 0.0
        f.write(f'{ko}: {ok}/{total} ({pct:.2f}%)\n')
        if failed:
            f.write('  Failed: ' + ', '.join(failed) + '\n')
        else:
            f.write('  Failed: none\n')

print(f'Wrote: {summary_path}')
for ko, ok, total, _ in rows:
    pct = (100.0 * ok / total) if total else 0.0
    print(f'{ko}: {ok}/{total} ({pct:.2f}%)')
PY




