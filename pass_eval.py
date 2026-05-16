#!/usr/bin/env python3
import argparse
import json
import subprocess
import sys
import time
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


def run_shell_once(root, group, trial_tag):
    sh = root / SCRIPT_MAP[group]
    if not sh.exists():
        raise FileNotFoundError("Missing script: {}".format(sh))
    cmd = ["bash", str(sh), trial_tag]
    print("RUN:", " ".join(cmd))
    p = subprocess.run(cmd, cwd=str(root))
    if p.returncode != 0:
        raise RuntimeError("{} failed with code {}".format(sh.name, p.returncode))


def collect_successes(results_root):
    case_success = {}
    for op in results_root.rglob("output.txt"):
        txt = op.read_text(encoding="utf-8", errors="ignore")
        cid = str(op.parent.relative_to(results_root)).replace("\\", "/")
        case_success[cid] = (SUCCESS_MARKER in txt)
    return case_success


def evaluate_group(root, group, trials, stamp):
    run_rates = []
    union_pass = {}

    for i in range(1, trials + 1):
        trial_tag = "{}_run{}".format(stamp, i)
        run_shell_once(root, group, trial_tag)

        out_root = root / "{}_{}".format(OUT_PREFIX[group], trial_tag)
        results_root = out_root / "results"
        if not results_root.exists():
            raise FileNotFoundError("Results folder missing: {}".format(results_root))

        res = collect_successes(results_root)
        total = len(res)
        succ = sum(1 for v in res.values() if v)
        rate = (float(succ) / float(total)) if total else 0.0
        run_rates.append(rate)

        for c, ok in res.items():
            union_pass[c] = union_pass.get(c, False) or ok

        print("[grouping_{}] run{}: {}/{} ({:.2f}%)".format(group, i, succ, total, rate * 100.0))

    total_cases = len(union_pass)
    pass3_success = sum(1 for v in union_pass.values() if v)

    return {
        "group": "grouping_{}".format(group),
        "trials": trials,
        "run_rates": run_rates,
        "pass1_avg_rate": (sum(run_rates) / len(run_rates)) if run_rates else 0.0,
        "pass3_success": pass3_success,
        "total_cases": total_cases,
        "pass3_rate": (float(pass3_success) / float(total_cases)) if total_cases else 0.0,
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

    out_json = root / "pass_eval_summary_{}.json".format(stamp)
    out_txt = root / "pass_eval_summary_{}.txt".format(stamp)
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = ["Pass evaluation summary ({})".format(stamp), "Uses grouping shell scripts per trial."]
    for g in summary["groups"]:
        run_rates = ", ".join(["{:.2f}%".format(r * 100.0) for r in g["run_rates"]])
        lines += [
            "",
            g["group"],
            "  pass1 avg success rate: {:.2f}%".format(g["pass1_avg_rate"] * 100.0),
            "  run rates: {}".format(run_rates),
            "  pass3 success: {}/{} ({:.2f}%)".format(g["pass3_success"], g["total_cases"], g["pass3_rate"] * 100.0),
            "  output prefix: {}".format(g["output_prefix"]),
        ]
    out_txt.write_text("\n".join(lines), encoding="utf-8")

    print("Wrote: {}".format(out_json))
    print("Wrote: {}".format(out_txt))


if __name__ == "__main__":
    main()
