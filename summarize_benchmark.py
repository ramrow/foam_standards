import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

ERROR_PATTERNS = [
    re.compile(r"\bERROR:\b", re.IGNORECASE),
    re.compile(r"\bFoam::error\b", re.IGNORECASE),
    re.compile(r"\bTraceback \(most recent call last\):", re.IGNORECASE),
    re.compile(r"\bRuntimeError\b", re.IGNORECASE),
    re.compile(r"\bException\b", re.IGNORECASE),
    re.compile(r"\bWorkflow failed\b", re.IGNORECASE),
]

TOKEN_PATTERNS = {
    "prompt": re.compile(r"Total prompt tokens:\s*(\d+)", re.IGNORECASE),
    "completion": re.compile(r"Total completion tokens:\s*(\d+)", re.IGNORECASE),
    "total": re.compile(r"Total tokens:\s*(\d+)", re.IGNORECASE),
}


def parse_tokens(text: str) -> Dict[str, int]:
    out = {"prompt": 0, "completion": 0, "total": 0}
    for k, pat in TOKEN_PATTERNS.items():
        m = pat.findall(text)
        if m:
            # take the last stats block in file
            out[k] = int(m[-1])
    # fallback if only prompt+completion available
    if out["total"] == 0 and (out["prompt"] or out["completion"]):
        out["total"] = out["prompt"] + out["completion"]
    return out


def has_log_error(text: str) -> bool:
    return any(p.search(text) for p in ERROR_PATTERNS)


def case_generated(run_case_dir: Path) -> bool:
    """Success requires OpenFOAM execution output, not just input-file generation.

    We count a case as generated/executed only if there is at least one numeric
    time directory other than "0" (e.g., "1", "2", "3", "0.5").
    """
    if not run_case_dir.exists() or not run_case_dir.is_dir():
        return False

    for p in run_case_dir.iterdir():
        if not p.is_dir():
            continue
        name = p.name.strip()
        if name == "0":
            continue
        try:
            float(name)
            return True
        except ValueError:
            continue

    return False


def summarize(model_root: Path) -> Tuple[Dict, List[Dict]]:
    runs_root = model_root / "runs"
    results_root = model_root / "results"

    if not runs_root.exists() or not results_root.exists():
        raise FileNotFoundError(f"Expected both runs/ and results/ under {model_root}")

    case_records = []

    for output_txt in results_root.rglob("output.txt"):
        rel = output_txt.relative_to(results_root)
        # expected: <dataset>/<case>/output.txt
        if len(rel.parts) < 3:
            continue
        dataset = rel.parts[0]
        case = rel.parts[1]

        run_case_dir = runs_root / dataset / case

        text = output_txt.read_text(encoding="utf-8", errors="ignore")
        generated = case_generated(run_case_dir)
        error_free = not has_log_error(text)
        success = generated and error_free

        tokens = parse_tokens(text)

        case_records.append(
            {
                "dataset": dataset,
                "case": case,
                "success": success,
                "generated": generated,
                "error_free": error_free,
                "prompt_tokens": tokens["prompt"],
                "completion_tokens": tokens["completion"],
                "total_tokens": tokens["total"],
                "output_path": str(output_txt),
                "run_case_dir": str(run_case_dir),
            }
        )

    total_cases = len(case_records)
    successes = sum(1 for r in case_records if r["success"])
    total_tokens = sum(r["total_tokens"] for r in case_records)
    avg_tokens = (total_tokens / total_cases) if total_cases else 0.0

    summary = {
        "total_cases": total_cases,
        "successes": successes,
        "success_rate": (successes / total_cases) if total_cases else 0.0,
        "total_tokens": total_tokens,
        "average_tokens": avg_tokens,
    }

    return summary, case_records


def main():
    parser = argparse.ArgumentParser(description="Summarize benchmark success/tokens from Foam-Agent outputs")
    parser.add_argument(
        "--repo_root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Foam-Agent repo root (contains <model_tag>/runs and <model_tag>/results)",
    )
    parser.add_argument(
        "--model_tag",
        type=str,
        required=True,
        help="Model tag folder name used by benchmark.py (e.g., qwen-benchmark)",
    )
    parser.add_argument(
        "--save_json",
        type=Path,
        default=None,
        help="Optional path to write detailed JSON report",
    )

    args = parser.parse_args()

    model_root = args.repo_root / args.model_tag
    summary, cases = summarize(model_root)

    print("=== Benchmark Summary ===")
    print(f"Model tag: {args.model_tag}")
    print(f"Total cases: {summary['total_cases']}")
    print(f"Successes: {summary['successes']}")
    print(f"Success rate: {summary['success_rate']:.2%}")
    print(f"Total tokens: {summary['total_tokens']}")
    print(f"Average tokens: {summary['average_tokens']:.2f}")

    failed = [c for c in cases if not c["success"]]
    if failed:
        print("\nFailed cases:")
        for c in failed:
            print(
                f"- {c['dataset']}/{c['case']} | generated={c['generated']} | "
                f"error_free={c['error_free']} | output={c['output_path']}"
            )

    if args.save_json:
        payload = {"summary": summary, "cases": cases}
        args.save_json.parent.mkdir(parents=True, exist_ok=True)
        args.save_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nWrote report: {args.save_json}")


if __name__ == "__main__":
    main()

