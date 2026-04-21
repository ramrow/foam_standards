# 9-substeps-exp (Qwen3 LoRA benchmark)

This benchmark uses **one Qwen3 base model** served by vLLM, with **5 LoRA adapters**:
- plan
- initial_write
- review_analysis
- review_plan
- rewrite

Substeps are mapped in `finetuned_models.json` using LoRA aliases (`plan`, `initial_write`, etc.).

## One-command run

```bash
bash 9-substeps-exp/run_all_one_command.sh
```

Outputs:
- Per-case run logs/results under `experiment-finetuned/all_finetuned/`
- Summary file: `experiment-finetuned/all_finetuned/summary.txt`
- vLLM server log: `9-substeps-exp/vllm_lora.log`
