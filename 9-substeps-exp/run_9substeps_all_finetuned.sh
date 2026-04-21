#!/bin/bash
set -euo pipefail

# Run from repo root
source /mnt/lustre/rpi/pxu10/agent/bin/activate
cd "/mnt/lustre/rpi/pxu10/official"

# OpenAI-compatible endpoint from local vLLM server
export OPENAI_API_KEY="EMPTY"
export OPENAI_BASE_URL="http://127.0.0.1:8000/v1"

echo "[1/2] Running all-finetuned 9-substep benchmark..."
python benchmark_finetuned.py \
  --all_finetuned \
  --finetuned_config "./9-substeps-exp/finetuned_models.json" \
  --workers 1

echo "[2/2] Summarizing results..."
python - <<'PY'
import os
from benchmark_utils import DATASETS_BASIC, CASES_BASIC

base = './experiment-finetuned/all_finetuned/runs'

def case_success(case_path: str) -> bool:
    if not os.path.isdir(case_path):
        return False
    log_files = [f for f in os.listdir(case_path) if f.startswith('log.') and f.endswith('Foam')]
    if not log_files:
        return False
    for lf in log_files:
        p = os.path.join(case_path, lf)
        try:
            with open(p, 'r', encoding='utf-8', errors='ignore') as fh:
                lines = fh.readlines()
            last = lines[-2].strip() if len(lines) > 1 else ''
            if last == 'End':
                return True
        except Exception:
            pass
    return False

rows = []
ok = 0
for d in DATASETS_BASIC:
    for c in CASES_BASIC:
        cp = os.path.join(base, d, str(c))
        s = case_success(cp)
        rows.append((d, c, s))
        ok += int(s)

total = len(rows)
print('\n===== FINETUNED BENCHMARK SUMMARY =====')
print(f'Success: {ok}/{total} ({(100*ok/total if total else 0):.2f}%)')
print('Failed cases:')
for d,c,s in rows:
    if not s:
        print(f'  - {d}/{c}')

out = './experiment-finetuned/all_finetuned/summary.txt'
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, 'w', encoding='utf-8') as f:
    f.write(f'Success: {ok}/{total} ({(100*ok/total if total else 0):.2f}%)\n')
    f.write('Failed cases:\n')
    for d,c,s in rows:
        if not s:
            f.write(f'- {d}/{c}\n')
print(f'Wrote: {out}')
PY
