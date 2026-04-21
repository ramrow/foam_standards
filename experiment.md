## Running fine-tuned models on individual substeps

This branch tests fine-tuned HuggingFace/vLLM models on the 9 LLM substeps. Each substep
can have its own fine-tuned model; the other substeps use the strong baseline model.

### The 9 substeps

| substep | step | service |
|---|---|---|
| `parse_case_info` | plan | plan |
| `build_advice` | plan | plan |
| `decompose_subtasks` | plan | plan |
| `generate_file` | initial_write | write |
| `allrun_commands` | initial_write | plan |
| `allrun_script` | initial_write | plan |
| `error_analysis` | review_analysis | review |
| `rewrite_plan` | review_plan | review |
| `rewrite_files` | rewrite | write |

---

### Checklist before running

1. API key / credentials for the strong baseline model are set in the environment.
2. `FoamAgent` conda env is activated.
3. Pick GPU(s) for local HuggingFace inference (vLLM):
   ```bash
   nvidia-smi -L                    # list GPUs
   export CUDA_VISIBLE_DEVICES=0    # use GPU 0
   export FOAMAGENT_HF_TP=1         # tensor-parallel size must match visible GPUs
   ```
   For 2 GPUs:
   ```bash
   export CUDA_VISIBLE_DEVICES=0,1
   export FOAMAGENT_HF_TP=2
   ```
4. Optional vLLM tuning:
   ```bash
   export FOAMAGENT_HF_DTYPE=bfloat16
   export FOAMAGENT_HF_MAX_MODEL_LEN=8192
   ```
5. `WM_PROJECT_DIR` points to the OpenFOAM installation:
   ```bash
   source /mnt/sda1/openfoam10/etc/bashrc          # workstation
   source /global/cfs/cdirs/m4756/theseus/OpenFOAM-10/etc/bashrc  # Perlmutter
   ```

---

### Step 1: Fill in model paths

Edit [`finetuned_models.json`](finetuned_models.json). Replace each `/path/to/ft-<substep>` with
the actual HuggingFace repo ID or local path for that substep's fine-tuned model.

You only need to fill in the substeps you want to test. Substeps missing from the file are
skipped automatically (the strong model runs on them instead).

---

### Step 2: Run the benchmark

#### Single-substep experiments (one FT model at a time, strong on all others)

```bash
# Basic cases (11 datasets × 3 cases = 33 total per substep)
nohup python benchmark_finetuned.py > finetuned.log 2>&1 &

# Advanced cases (16 datasets × 1 case per substep)
nohup python benchmark_finetuned_advanced.py > finetuned_advanced.log 2>&1 &

# Run a single substep only
python benchmark_finetuned.py --substep generate_file
```

Results land in `./experiment-finetuned/<substep>/` and `./experiment-finetuned-advanced/<substep>/`.

#### All-fine-tuned experiment (all substeps use their FT model simultaneously)

```bash
python benchmark_finetuned.py --all_finetuned > finetuned_all.log 2>&1 &
python benchmark_finetuned_advanced.py --all_finetuned > finetuned_advanced_all.log 2>&1 &
```

Results land in `./experiment-finetuned/all_finetuned/`.

#### Workers (vLLM memory note)

vLLM loads the model inside each worker process. The default is `--workers 1` to avoid
multiplying VRAM usage. Increase only if you have headroom:

```bash
python benchmark_finetuned.py --workers 2
```

---

### Step 3: Retry failures

```bash
python benchmark_finetuned_retry.py --log finetuned.log
python benchmark_finetuned_retry_advanced.py --log finetuned_advanced.log

# Dry-run first to see what would be retried
python benchmark_finetuned_retry.py --dry-run
```

---

### Step 4: Analyze results

In [`ablation-notebooks/`](ablation-notebooks/), use:

- `utils_finetuned.py` — basic cases (`experiment-finetuned/`)
- `utils_finetuned_advanced.py` — advanced cases (`experiment-finetuned-advanced/`)

Key functions:

```python
from utils_finetuned import collect_substep, collect_ablation_substep, success_rate, avg_loops

# Fine-tuned results for one substep
ft_results = collect_substep("generate_file")

# Ablation baseline for comparison (reads from experiment-ablation/)
baseline = collect_ablation_substep("baseline")
weak = collect_ablation_substep("generate_file")

print(f"Baseline:   {success_rate(baseline):.1%}")
print(f"Weak model: {success_rate(weak):.1%}")
print(f"Fine-tuned: {success_rate(ft_results):.1%}")
```

For the standard model baseline (all-strong, standard benchmark):
- `utils_basic.py` / `utils_advanced.py` + `opus-4.6-basic.ipynb` / `opus-4.6-advanced.ipynb`
