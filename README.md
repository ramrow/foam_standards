# Foam-Agent Benchmark (foamstandard)

This folder contains a Python/OpenFOAM benchmark setup for Foam-Agent v2-style workflow.

## Location

`C:\Users\Peijing Xu\projects\yue_research\foamstandard\Foam-Agent`

---

## 1) Prerequisites

- Python environment with required packages (from `environment.yml` or equivalent)
- OpenFOAM already installed
- `WM_PROJECT_DIR` set in your shell
- vLLM server running (for `model_provider = vllm`)

### Quick checks

```bash
python --version
echo $WM_PROJECT_DIR
```

For Windows PowerShell:
```powershell
python --version
$env:WM_PROJECT_DIR
```

---

## 2) Model/runtime config

Config file:

`src/config.py`

Current setup is single-service (`general`) and uses:
- provider: `vllm`
- model: `Qwen/Qwen3-Coder-30B-A3B-Instruct`
- temperature: `0.5`

Optional runtime overrides:
- `FOAMAGENT_MODEL_SERVICE`
- `FOAMAGENT_MODEL_PROVIDER`
- `FOAMAGENT_MODEL_VERSION`
- `FOAMAGENT_TEMPERATURE`
- `VLLM_BASE_URL` (default: `http://localhost:8000/v1`)

---

## 3) Run benchmark (Basic split)

Benchmark script:

`benchmark.py`

Defaults:
- 11 Basic datasets
- cases `1 2 3`
- output bucket name (`model_tag`) default: `qwen-benchmark`

### Default run

```bash
python benchmark.py
```

### Explicit run

```bash
python benchmark.py --model_tag qwen-benchmark --cases 1 2 3
```

### Run only selected datasets

```bash
python benchmark.py --model_tag qwen-benchmark --datasets Cavity Cylinder --cases 1 2 3
```

Output layout:
- `qwen-benchmark/runs/<dataset>/<case>/...`
- `qwen-benchmark/results/<dataset>/<case>/output.txt`

---

## 4) Summarize results

Summary script:

`summarize_benchmark.py`

It reports:
- total tokens
- average tokens
- number of successful cases

A case is counted as **success** when:
1. case files were generated in `runs/...`, and
2. no error pattern is found in `results/.../output.txt`.

### Run summary

```bash
python summarize_benchmark.py --model_tag qwen-benchmark
```

### Save JSON report

```bash
python summarize_benchmark.py --model_tag qwen-benchmark --save_json ./qwen-benchmark/summary.json
```

---

## 5) Typical workflow

1. Start/verify vLLM server.
2. Confirm OpenFOAM env (`WM_PROJECT_DIR`).
3. Run benchmark:
   ```bash
   python benchmark.py --model_tag qwen-benchmark --cases 1 2 3
   ```
4. Summarize:
   ```bash
   python summarize_benchmark.py --model_tag qwen-benchmark
   ```

---

## 6) Notes

- `model_tag` is only a folder label for outputs.
- If benchmark output path changes, pass the same `model_tag` to the summary script.
- Clear `runs/` or use a new `model_tag` for clean experiment separation.
