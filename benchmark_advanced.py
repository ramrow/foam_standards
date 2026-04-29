import os
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed

from benchmark_utils import DIR_ADVANCED, DATASETS_ADVANCED, EXP_ADVANCED, MODEL

WM_PROJECT_DIR = os.environ.get('WM_PROJECT_DIR')

if not WM_PROJECT_DIR:
    print("Error: WM_PROJECT_DIR is not set in the environment.")
    exit(1)

def read_user_requirement(file_path):
    """Reads and returns the content of user_requirement.txt."""
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            content = file.read()
            return content
    else:
        print(f"File not found: {file_path}")
        return ""
    

def run_benchmark(dataset, model):
    """Creates user_requirement.txt and runs foambench_main.py."""
    folder_path = os.path.abspath(os.path.join(DIR_ADVANCED, dataset))
    requirement_txt_path = os.path.abspath(os.path.join(folder_path, "usr_requirement.txt"))

    output_folder = os.path.abspath(os.path.join(
        EXP_ADVANCED,
        f"{model}/runs",
        dataset,
    ))
    os.makedirs(output_folder, exist_ok=True)

    case_id = f"Advanced/{dataset}"
    dataset_log_path = os.path.abspath(os.path.join(
        EXP_ADVANCED,
        f"{model}/results",
        dataset,
        "dataset.jsonl"
    ))

    command = (
        f"python foambench_main.py --openfoam_path {WM_PROJECT_DIR} --output {output_folder}"
        f" --prompt_path {requirement_txt_path}"
        f" --dataset_log_path {dataset_log_path} --case_id '{case_id}'"
    )
    print(f"Running: {command}")

    output_text_path = os.path.abspath(os.path.join(
        EXP_ADVANCED,
        f"{model}/results",
        dataset,
        "output.txt"
    ))
    print(output_text_path)

    os.makedirs(os.path.dirname(output_text_path), exist_ok=True)
    with open(output_text_path, 'w') as file:
        try:
            subprocess.run(command, shell=True, check=True, stdout=file, stderr=file)
        except subprocess.CalledProcessError as e:
            print(f"Error running benchmark for {dataset}: {e}")


if __name__ == "__main__":
    """Loops through all datasets and runs benchmarks for cases (up to 5 in parallel)"""
    tasks = [(dataset, MODEL) for dataset in DATASETS_ADVANCED]
    with ProcessPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(run_benchmark, *t): t for t in tasks}
        for future in as_completed(futures):
            task = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"Benchmark {task} failed: {e}")

