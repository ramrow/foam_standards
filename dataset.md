I want to create a large dataset for fine-tuning an open-source large language model for OpenFOAM.

We need the model to do the end-to-end file-writing.

```
plan, initial write, reviewer analysis, reviewer plan, rewrite.
```

We want to extract the output of a frontier LLM (e.g. opus 4.6) on the benchmark cases as the fine-tuning. We will run the 110 basic cases and the 16 advanced cases. For step in the workflow, we will extract the entire prompt and what opus-4.6 responded. We need more than what the Foam-Agent natively save, as it omits some prompts and only kept output.

For each category, we need to tag them to create a nice dataset. I want to store in the format of jsonl.

What should be the schema of the dataset? How should we modify Foam-Agent to allow storing our data?

One jsonl for every case. 5 main categories (9 call sites). Eventually, store into 5 large jsonl (plan, initial write, reviewer analysis, reviewer plan, rewrite)

Per case:
```
  {                                                                                                                                             
    "case_id": "Basic/Cavity/1",                  
    "step": "initial_write",                                                                                                                    
    "substep": "generate_file",                                                                                                                 
    "loop_iteration": 0,                                                                                                                        
    "file_target": "system/blockMeshDict",                                                                                                      
    "system_prompt": "...",                                                                                                                     
    "user_prompt": "...",                                                                                                                       
    "response": "...",                                                                                                                          
    "model": "claude-opus-4-6",                                                                                                                 
    "prompt_tokens": 1234,                                                                                                                      
    "completion_tokens": 567,                                                                                                                   
    "timestamp": "2026-03-29T12:00:00Z"
  }
```

## How to run the pipeline

### 1. Run benchmarks (generates per-case dataset.jsonl)

```bash
# Basic benchmark (110 cases = 11 datasets x 10 cases, currently 3 active)
python benchmark_basic.py

# Advanced benchmark (16 cases)
python benchmark_advanced.py
```

Both scripts automatically pass `--dataset_log_path` and `--case_id` to each case run. Up to 5 cases run in parallel.

Each case produces a `dataset.jsonl` under its results directory:
```
experiment-basic/<model>/results/<dataset>/<case>/dataset.jsonl
experiment-advanced/<model>/results/<dataset>/dataset.jsonl
```

### 2. Run a single case manually (optional)

```bash
python foambench_main.py \
    --openfoam_path $WM_PROJECT_DIR \
    --output experiment-basic/opus-4.6/runs/Cavity/1 \
    --prompt_path Dataset/Basic/Cavity/1/usr_requirement.txt \
    --dataset_log_path experiment-basic/opus-4.6/results/Cavity/1/dataset.jsonl \
    --case_id "Basic/Cavity/1"
```

Omit `--dataset_log_path` to disable logging (zero overhead).

### 3. Consolidate into 5 category files

```bash
# Basic
python consolidate_dataset.py \
    --input_dir experiment-basic/opus-4.6/results \
    --output_dir dataset_out

# Advanced (append to the same output)
python consolidate_dataset.py \
    --input_dir experiment-advanced/opus-4.6/results \
    --output_dir dataset_out \
    --append
```

## Expected output

### Per-case dataset.jsonl

One file per case containing all LLM interactions for that run, one JSON object per line. A case with no review loops produces ~6 records (3 plan + 3 initial_write). A case with N review loops adds 3N more records (1 review_analysis + 1 review_plan + 1 rewrite per loop).

### Consolidated output (dataset_out/)

5 JSONL files, one per workflow step:

| File | Contents | Records per case |
|---|---|---|
| `plan.jsonl` | parse_case_info, build_advice, decompose_subtasks | 3 (always) |
| `initial_write.jsonl` | generate_file (per file), allrun_commands, allrun_script | varies (typically 10-20+) |
| `review_analysis.jsonl` | error_analysis from reviewer | 0 to max_loop |
| `review_plan.jsonl` | rewrite_plan from reviewer | 0 to max_loop |
| `rewrite.jsonl` | rewrite_files (file corrections) | 0 to max_loop |

### 9 LLM call sites captured

| step | substep | service | source file |
|---|---|---|---|
| `plan` | `parse_case_info` | plan | `services/plan.py` |
| `plan` | `build_advice` | plan | `services/plan.py` |
| `plan` | `decompose_subtasks` | plan | `services/plan.py` |
| `initial_write` | `generate_file` | write | `services/input_writer.py` |
| `initial_write` | `allrun_commands` | plan | `services/input_writer.py` |
| `initial_write` | `allrun_script` | plan | `services/input_writer.py` |
| `review_analysis` | `error_analysis` | review | `services/review.py` |
| `review_plan` | `rewrite_plan` | review | `services/review.py` |
| `rewrite` | `rewrite_files` | write | `services/input_writer.py` |