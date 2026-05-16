#!/usr/bin/env python3
import argparse, json, subprocess, sys, time
from pathlib import Path

SUCCESS_MARKER = "Allrun executed successfully without errors."

SCRIPT_MAP = {
    "B": "run_grouping_B_benchmark.sh",
    "C": "run_grouping_C_benchmark.sh",
    "E2": "run_grouping_E2_benchmark.sh",
}

OUT_PREFIX = {
    "B": "benchmark_grouping_B",
    "C": "benchmark_grouping_C",
    "E2": "benchmark_grouping_E2",
}

def run_shell_once(root: Path, group: str, trial_tag: str):
    sh = root / SCRIPT_MAP[group]
    if not sh.exists():
        raise FileNotFoundError(f"Missing script: {sh}")
    cmd = ["bash", str(sh), trial_tag]
    print("RUN:", " ".join(cmd))
    p = subprocess.run(cmd, cwd=str(root))
    if p.returncode != 0:
        raise RuntimeError(f"{sh.name} failed with code {p.returncode}")


def collect_successes(results_root: Path):
    case_success = {}
    for op in results_root.rglob("output.txt"):
        txt = op.read_text(encoding="utf-8", errors="ignore")
        cid = str(op.parent.relative_to(results_root)).replace("\\", "/")
        case_success[cid] = SUCCESS_MARKER in txt
    return case_success


def evaluate_group(root: Path, group: str, trials: int, stamp: str):
    run_rates = []
    union_pass = {}

    for i in range(1, trials + 1):
        trial_tag = f"{stamp}_run{i}"
        run_shell_once(root, group, trial_tag)

        out_root = root / f"{OUT_PREFIX[group]}_{trial_tag}"
        results_root = out_root / "results"
        if not results_root.exists():
            raise FileNotFoundError(f"Results folder missing: {results_root}")

        res = collect_successes(results_root)
        total = len(res)
        succ = sum(1 for v in res.values() if v)
        rate = (succ / total) if total else 0.0
        run_rates.append(rate)

        for c, ok in res.items():
            union_pass[c] = union_pass.get(c, False) or ok

        print(f"[grouping_{group}] run{i}: {succ}/{total} ({rate*100:.2f}%)")

    total_cases = len(union_pass)
    pass3_success = sum(1 for v in union_pass.values() if v)

    return {
        "group": f"grouping_{group}",
        "trials": trials,
        "run_rates": run_rates,
        "pass1_avg_rate": sum(run_rates) / len(run_rates) if run_rates else 0.0,
        "pass3_success": pass3_success,
        "total_cases": total_cases,
        "pass3_rate": (pass3_success / total_cases) if total_cases else 0.0,
        "pass3_success_cases": sorted([c for c, ok in union_pass.items() if ok]),
        "output_prefix": OUT_PREFIX[group],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", help="debug/troubleshoot directory containing run_grouping_*.sh")
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--groups", nargs="*", default=["B", "C", "E2"], choices=["B", "C", "E2"])
    args = ap.parse_args()

    root = Path(args.root).resolve()
    stamp = time.strftime("%Y%m%d_%H%M%S")

    summary = {"timestamp": stamp, "root": str(root), "groups": []}

    for g in args.groups:
        summary["groups"].append(evaluate_group(root, g, args.trials, stamp))

    out_json = root / f"pass_eval_summary_{stamp}.json"
    out_txt = root / f"pass_eval_summary_{stamp}.txt"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [f"Pass evaluation summary ({stamp})", "Uses grouping shell scripts per trial."]
    for g in summary["groups"]:
        lines += [
            "",
            g["group"],
            f"  pass1 avg success rate: {g['pass1_avg_rate']*100:.2f}%",
            f"  run rates: {', '.join(f'{r*100:.2f}%' for r in g['run_rates'])}",
            f"  pass3 success: {g['pass3_success']}/{g['total_cases']} ({g['pass3_rate']*100:.2f}%)",
            f"  output prefix: {g['output_prefix']}",
        ]
    out_txt.write_text("\n".join(lines), encoding="utf-8")

    print(f"Wrote: {out_json}")
    print(f"Wrote: {out_txt}")


if __name__ == "__main__":
    main()
