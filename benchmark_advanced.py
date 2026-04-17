import os
import subprocess
import argparse
import torch
import gc
from pathlib import Path

# Advanced split: each dataset has one case (no numbered subfolders)
DATASETS = [
    "Cavity_LES",
    "Cavity_SA",
    "Cavity_geometry_1",
    "Cylinder_LES",
    "Cylinder_SA",
    "Diamond_Obstacle_KOMEGASST",
    "Diamond_Obstacle_SA",
    "Double_Square_SA",
    "Rectangular_Obstacle_KOMEGASST",
    "Rectangular_Obstacle_SA",
    "counterFlowFlame2D_KE",
    "counterFlowFlame2D_SA",
    "nozzleFlow2D_SA",
    "obliqueShock_KE",
    "obliqueShock_LES",
    "wedge_SA"
]

BASE_DIR = "Dataset/Advanced/"
WM_PROJECT_DIR = os.environ.get("WM_PROJECT_DIR")

if not WM_PROJECT_DIR:
    print("Error: WM_PROJECT_DIR is not set in the environment.")
    raise SystemExit(1)


def find_requirement_file(dataset_dir: Path) -> Path:
    """Resolve advanced dataset prompt path.

    Expected primary layout: Dataset/Advanced/<dataset>/usr_requirement.txt
    Fallback: Dataset/Advanced/<dataset>/1/usr_requirement.txt
    """
    p1 = dataset_dir / "usr_requirement.txt"
    if p1.exists():
        return p1

    p2 = dataset_dir / "1" / "usr_requirement.txt"
    if p2.exists():
        return p2

    raise FileNotFoundError(f"No usr_requirement.txt found for dataset: {dataset_dir}")


def run_benchmark(dataset: str, model_tag: str, repo_root: Path):
    dataset_dir = (repo_root / BASE_DIR / dataset).resolve()
    requirement_txt_path = find_requirement_file(dataset_dir).resolve()

    # use '__single__' as synthetic case id so downstream summary format stays consistent
    case_id = "__single__"
    output_folder = (repo_root / model_tag / "runs" / dataset / case_id).resolve()
    output_folder.mkdir(parents=True, exist_ok=True)

    command = (
        f'python foambench_main.py '
        f'--openfoam_path "{WM_PROJECT_DIR}" '
        f'--output "{output_folder}" '
        f'--prompt_path "{requirement_txt_path}"'
    )
    print(f"Running: {command}")

    output_text_path = (repo_root / model_tag / "results" / dataset / case_id / "output.txt").resolve()
    output_text_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_text_path, "w", encoding="utf-8") as file:
        try:
            subprocess.run(command, shell=True, check=True, cwd=str(repo_root), stdout=file, stderr=file)
        except subprocess.CalledProcessError as e:
            print(f"Error running benchmark for {dataset}: {e}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Foam-Agent v2 benchmark (advanced split)")
    parser.add_argument("--model_tag", type=str, default="qwen-advanced", help="Output root bucket name")
    parser.add_argument("--datasets", nargs="*", default=DATASETS, help="Advanced dataset names")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    print(f"Repo root: {root}")
    print(f"Datasets: {args.datasets}")

    for dataset in args.datasets:
        run_benchmark(dataset, args.model_tag, root)
