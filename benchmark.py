import os
import subprocess
import argparse
import torch
import gc
from pathlib import Path

# Basic split: 11 datasets, run cases 1/2/3 by default
DATASETS = [
    "BernardCells",
    "Cavity",
    "counterFlowFlame2D",
    "Cylinder",
    "forwardStep",
    "obliqueShock",
    "pitzDaily",
    "squareBend",
    "wedge",
    "shallowWaterWithSquareBump",
    "damBreakWithObstacle",
]
CASES = [1, 2, 3]
BASE_DIR = "Dataset/Basic/"
WM_PROJECT_DIR = os.environ.get("WM_PROJECT_DIR")

if not WM_PROJECT_DIR:
    print("Error: WM_PROJECT_DIR is not set in the environment.")
    raise SystemExit(1)


def run_benchmark(dataset: str, case: int, model_tag: str, repo_root: Path):
    folder_path = (repo_root / BASE_DIR / dataset / str(case)).resolve()
    requirement_txt_path = (folder_path / "usr_requirement.txt").resolve()

    output_folder = (repo_root / model_tag / "runs" / dataset / str(case)).resolve()
    output_folder.mkdir(parents=True, exist_ok=True)

    command = (
        f'python foambench_main.py '
        f'--openfoam_path "{WM_PROJECT_DIR}" '
        f'--output "{output_folder}" '
        f'--prompt_path "{requirement_txt_path}"'
    )
    print(f"Running: {command}")

    output_text_path = (repo_root / model_tag / "results" / dataset / str(case) / "output.txt").resolve()
    output_text_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_text_path, "w", encoding="utf-8") as file:
        try:
            subprocess.run(command, shell=True, check=True, cwd=str(repo_root), stdout=file, stderr=file)
        except subprocess.CalledProcessError as e:
            print(f"Error running benchmark for {dataset}/{case}: {e}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Foam-Agent v2 benchmark (basic split)")
    parser.add_argument("--model_tag", type=str, default="qwen-benchmark", help="Output root bucket name")
    parser.add_argument("--datasets", nargs="*", default=DATASETS, help="Dataset names")
    parser.add_argument("--cases", nargs="*", type=int, default=CASES, help="Case numbers")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    print(f"Repo root: {root}")
    print(f"Datasets: {args.datasets}")
    print(f"Cases: {args.cases}")

    for dataset in args.datasets:
        for case in args.cases:
            run_benchmark(dataset, case, args.model_tag, root)
