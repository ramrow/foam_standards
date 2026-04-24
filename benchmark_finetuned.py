import os
import json
import subprocess
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

from benchmark_utils import DATASETS_BASIC, CASES_BASIC, DIR_BASIC

WM_PROJECT_DIR = os.environ.get('WM_PROJECT_DIR')
if not WM_PROJECT_DIR:
    print("Error: WM_PROJECT_DIR is not set in the environment.")
    exit(1)

EXP_FINETUNED = "./experiment-finetuned"

SUBSTEPS = [
    "parse_case_info",
    "build_advice",
    "decompose_subtasks",
    "generate_file",
    "allrun_commands",
    "allrun_script",
    "error_analysis",
    "rewrite_plan",
    "rewrite_files",
]

STRONG_MODEL_CFG = {
    "model_provider": "openai",
    "model_version": "plan",
    "temperature": 0.01,
}


def load_finetuned_config(path: str) -> dict:
    """Load per-substep fine-tuned model configs from a JSON file."""
    with open(path, encoding="utf-8-sig") as f:
        return json.load(f)


def build_finetuned_config(substep: str, ft_cfg: dict) -> dict:
    """All 3 services use STRONG; one substep is overridden with its fine-tuned model."""
    return {
        "models": {
            "plan":   STRONG_MODEL_CFG,
            "write":  STRONG_MODEL_CFG,
            "review": STRONG_MODEL_CFG,
        },
        "substep_model_overrides": {
            substep: ft_cfg[substep],
        },
    }


def build_all_finetuned_config(ft_cfg: dict) -> dict:
    """All 3 services use STRONG; all substeps present in ft_cfg use their fine-tuned model."""
    return {
        "models": {
            "plan":   STRONG_MODEL_CFG,
            "write":  STRONG_MODEL_CFG,
            "review": STRONG_MODEL_CFG,
        },
        "substep_model_overrides": dict(ft_cfg),
    }


def run_benchmark(dataset, case, substep_label, model_config_path):
    """Run foambench_main.py for one (dataset, case) with a pre-written config JSON."""
    folder_path = os.path.abspath(os.path.join(DIR_BASIC, dataset, str(case)))
    requirement_txt_path = os.path.abspath(os.path.join(folder_path, "usr_requirement.txt"))

    output_folder = os.path.abspath(os.path.join(
        EXP_FINETUNED, substep_label, "runs", dataset, str(case)
    ))
    os.makedirs(output_folder, exist_ok=True)

    case_id = f"Basic/{dataset}/{case}"
    dataset_log_path = os.path.abspath(os.path.join(
        EXP_FINETUNED, substep_label, "results", dataset, str(case), "dataset.jsonl"
    ))

    output_text_path = os.path.abspath(os.path.join(
        EXP_FINETUNED, substep_label, "results", dataset, str(case), "output.txt"
    ))
    os.makedirs(os.path.dirname(output_text_path), exist_ok=True)

    command = (
        f"python foambench_main.py"
        f" --openfoam_path {WM_PROJECT_DIR}"
        f" --output {output_folder}"
        f" --prompt_path {requirement_txt_path}"
        f" --dataset_log_path {dataset_log_path}"
        f" --case_id '{case_id}'"
        f" --model_config_path {model_config_path}"
    )
    print(f"Running: {command}")

    with open(output_text_path, 'w') as f:
        try:
            subprocess.run(command, shell=True, check=True, stdout=f, stderr=f)
        except subprocess.CalledProcessError as e:
            print(f"Error running benchmark for {dataset}/{case}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run fine-tuned model benchmark: one substep at a time uses its fine-tuned model."
    )
    parser.add_argument(
        "--finetuned_config", type=str, default="./9-substeps-exp/finetuned_models.json",
        help="Path to JSON file mapping substep names to fine-tuned model configs (default: ./finetuned_models.json)."
    )
    parser.add_argument(
        "--substep", type=str, default=None,
        help=f"Run only a single named substep. Choices: {SUBSTEPS}"
    )
    parser.add_argument(
        "--all_finetuned", action="store_true",
        help="Run one experiment with ALL substeps using their fine-tuned models simultaneously."
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Number of parallel workers (default: 1; vLLM loads model per process â€” increase only if VRAM allows)."
    )
    args = parser.parse_args()

    ft_cfg = load_finetuned_config(args.finetuned_config)

    # Validate config shape early
    missing = [s for s in SUBSTEPS if s not in ft_cfg]
    if args.all_finetuned and missing:
        raise ValueError(f"Missing substep configs in {args.finetuned_config}: {missing}")

    if args.all_finetuned:
        runs = [("all_finetuned", build_all_finetuned_config(ft_cfg))]
    elif args.substep:
        if args.substep not in SUBSTEPS:
            raise ValueError(f"Unknown substep '{args.substep}'. Choices: {SUBSTEPS}")
        if args.substep not in ft_cfg:
            raise ValueError(f"Substep '{args.substep}' not found in {args.finetuned_config}.")
        runs = [(args.substep, build_finetuned_config(args.substep, ft_cfg))]
    else:
        runs = []
        for s in SUBSTEPS:
            if s not in ft_cfg:
                print(f"Warning: substep '{s}' not in {args.finetuned_config} â€” skipping.")
                continue
            runs.append((s, build_finetuned_config(s, ft_cfg)))

    for substep_label, model_config in runs:
        config_dir = os.path.abspath(os.path.join(EXP_FINETUNED, substep_label))
        os.makedirs(config_dir, exist_ok=True)
        model_config_path = os.path.join(config_dir, "model_config.json")
        with open(model_config_path, 'w') as f:
            json.dump(model_config, f, indent=2)

        print(f"\n=== Fine-tuned benchmark: substep = '{substep_label}' ===")
        print(f"Config written to: {model_config_path}")

        tasks = [
            (dataset, case, substep_label, model_config_path)
            for dataset in DATASETS_BASIC
            for case in CASES_BASIC
        ]
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(run_benchmark, *t): t for t in tasks}
            for future in as_completed(futures):
                task = futures[future]
                try:
                    future.result()
                except Exception as e:
                    print(f"Benchmark {task} failed: {e}")

# nohup python benchmark_finetuned.py > finetuned.log 2>&1 &
# nohup python benchmark_finetuned.py --all_finetuned > finetuned_all.log 2>&1 &
# python benchmark_finetuned.py --substep generate_file







