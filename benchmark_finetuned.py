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

EXP_FINETUNED_DEFAULT = "./experiment-finetuned"

CASE_TIMEOUT_SEC = int(os.environ.get("FOAMAGENT_CASE_TIMEOUT_SEC", "900"))
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



def has_non_openfoam_garbage(case_dir: str) -> tuple[bool, list[str]]:
    """Minimal artifact guard: delete only *.py and *.slurm in case root."""
    bad = []
    try:
        for name in os.listdir(case_dir):
            if name.endswith(".py") or name.endswith(".slurm"):
                p = os.path.join(case_dir, name)
                try:
                    if os.path.isdir(p):
                        import shutil
                        shutil.rmtree(p, ignore_errors=True)
                    else:
                        os.remove(p)
                    bad.append(name)
                except Exception as e:
                    bad.append(f"<delete_failed:{name}:{e}>")
    except Exception as e:
        bad.append(f"<scan_error:{e}>")
    return (len(bad) > 0, bad)


def load_finetuned_config(path: str) -> dict:
    """Load per-substep fine-tuned model configs from a JSON file."""
    with open(path, encoding="utf-8-sig") as f:
        return json.load(f)


def build_finetuned_config(substep: str, ft_cfg: dict) -> dict:
    """Service-level defaults use valid loaded aliases; one substep is overridden."""
    return {
        "models": {
            "plan":   {"model_provider": "openai", "model_version": "parse_case_info", "temperature": 0.1},
            "write":  {"model_provider": "openai", "model_version": "generate_file", "temperature": 0.1},
            "review": {"model_provider": "openai", "model_version": "error_analysis", "temperature": 0.1},
        },
        "substep_model_overrides": {
            substep: ft_cfg[substep],
        },
    }


def build_all_finetuned_config(ft_cfg: dict) -> dict:
    """Service-level defaults use valid loaded aliases; all substeps override appropriately."""
    return {
        "models": {
            "plan":   {"model_provider": "openai", "model_version": "parse_case_info", "temperature": 0.1},
            "write":  {"model_provider": "openai", "model_version": "generate_file", "temperature": 0.1},
            "review": {"model_provider": "openai", "model_version": "error_analysis", "temperature": 0.1},
        },
        "substep_model_overrides": dict(ft_cfg),
    }

def run_benchmark(dataset, case, substep_label, model_config_path, exp_root):
    """Run foambench_main.py for one (dataset, case) with a pre-written config JSON."""
    folder_path = os.path.abspath(os.path.join(DIR_BASIC, dataset, str(case)))
    requirement_txt_path = os.path.abspath(os.path.join(folder_path, "usr_requirement.txt"))

    output_folder = os.path.abspath(os.path.join(exp_root, "runs", dataset, str(case)))
    os.makedirs(output_folder, exist_ok=True)

    case_id = f"Basic/{dataset}/{case}"
    dataset_log_path = os.path.abspath(os.path.join(exp_root, "results", dataset, str(case), "dataset.jsonl"))

    output_text_path = os.path.abspath(os.path.join(exp_root, "results", dataset, str(case), "output.txt"))
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
            subprocess.run(command, shell=True, check=True, stdout=f, stderr=f, timeout=CASE_TIMEOUT_SEC)
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
    parser.add_argument("--output_root", type=str, default=EXP_FINETUNED_DEFAULT, help="Root output dir (default: ./experiment-finetuned)")
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

    exp_root = os.path.abspath(args.output_root)
    for substep_label, model_config in runs:
        config_dir = os.path.abspath(exp_root)
        os.makedirs(config_dir, exist_ok=True)
        model_config_path = os.path.join(config_dir, "model_config.json")
        with open(model_config_path, 'w') as f:
            json.dump(model_config, f, indent=2)

        print(f"\n=== Fine-tuned benchmark: substep = '{substep_label}' ===")
        print(f"Config written to: {model_config_path}")

        tasks = [
            (dataset, case, substep_label, model_config_path, exp_root)
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



