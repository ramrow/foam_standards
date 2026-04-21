"""Consolidate per-case dataset.jsonl files into 5 category JSONL files.

Usage:
    python consolidate_dataset.py --input_dir experiment-basic/opus-4.6/results --output_dir dataset_out
    python consolidate_dataset.py --input_dir experiment-advanced/opus-4.6/results --output_dir dataset_out --append

The 5 output files (one per step category):
    plan.jsonl, initial_write.jsonl, review_analysis.jsonl, review_plan.jsonl, rewrite.jsonl
"""
import argparse
import json
import os
import glob


CATEGORIES = ["plan", "initial_write", "review_analysis", "review_plan", "rewrite"]


def consolidate(input_dir: str, output_dir: str, append: bool = False):
    os.makedirs(output_dir, exist_ok=True)

    # Collect all per-case dataset.jsonl files
    pattern = os.path.join(input_dir, "**", "dataset.jsonl")
    files = sorted(glob.glob(pattern, recursive=True))
    if not files:
        print(f"No dataset.jsonl files found under {input_dir}")
        return

    print(f"Found {len(files)} dataset.jsonl files")

    # Bucket records by step
    buckets = {cat: [] for cat in CATEGORIES}
    unknown = []

    for fpath in files:
        with open(fpath, "r") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"Warning: bad JSON in {fpath}:{line_no}: {e}")
                    continue
                step = record.get("step", "")
                if step in buckets:
                    buckets[step].append(record)
                else:
                    unknown.append(record)

    # Write output files
    mode = "a" if append else "w"
    total = 0
    for cat in CATEGORIES:
        records = buckets[cat]
        if not records:
            continue
        out_path = os.path.join(output_dir, f"{cat}.jsonl")
        with open(out_path, mode) as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  {cat}.jsonl: {len(records)} records")
        total += len(records)

    if unknown:
        out_path = os.path.join(output_dir, "unknown.jsonl")
        with open(out_path, mode) as f:
            for r in unknown:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  unknown.jsonl: {len(unknown)} records")
        total += len(unknown)

    print(f"Total: {total} records consolidated")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Consolidate per-case dataset.jsonl into category files")
    parser.add_argument("--input_dir", required=True, help="Root results directory (e.g. experiment-basic/opus-4.6/results)")
    parser.add_argument("--output_dir", required=True, help="Output directory for consolidated JSONL files")
    parser.add_argument("--append", action="store_true", help="Append to existing output files instead of overwriting")
    args = parser.parse_args()
    consolidate(args.input_dir, args.output_dir, args.append)
