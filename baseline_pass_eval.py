#!/usr/bin/env python3
import argparse
import json
import subprocess
import time
from pathlib import Path

SUCCESS_MARKER = "Allrun executed successfully without errors."


def run_shell_once(root, trial_tag):
    sh = root / "run_baseline_pass3_benchmark.sh"
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", help="troubleshoot/debug directory")
    ap.add_argument("--trials", type=int, default=3)
    args = ap.parse_args()

    root = Path(args.root).resolve()
    stamp = time.strftime("%Y%m%d_%H%M%S")

    run_rates = []
    union_pass = {}
    run_roots = []

    for i in range(1, args.trials + 1):
        trial_tag = "{}_run{}".format(stamp, i)
        run_shell_once(root, trial_tag)

        out_root = root / "benchmark_baseline_pass3_{}".format(trial_tag)
        results_root = out_root / "results"
        if not results_root.exists():
            raise FileNotFoundError("Results folder missing: {}".format(results_root))

        run_roots.append(str(results_root))
        res = collect_successes(results_root)
        total = len(res)
        succ = sum(1 for v in res.values() if v)
        rate = (float(succ) / float(total)) if total else 0.0
        run_rates.append(rate)

        for c, ok in res.items():
            union_pass[c] = union_pass.get(c, False) or ok

        print("[baseline] run{}: {}/{} ({:.2f}%)".format(i, succ, total, rate * 100.0))

    total_cases = len(union_pass)
    pass3_success = sum(1 for v in union_pass.values() if v)
    pass3_rate = (float(pass3_success) / float(total_cases)) if total_cases else 0.0
    pass1_avg = sum(run_rates) / len(run_rates) if run_rates else 0.0

    summary = {
        "timestamp": stamp,
        "group": "baseline_all_finetuned_base_only",
        "trials": args.trials,
        "run_rates": run_rates,
        "pass1_avg_rate": pass1_avg,
        "pass3_success": pass3_success,
        "total_cases": total_cases,
        "pass3_rate": pass3_rate,
        "result_roots": run_roots,
        "output_prefix": "benchmark_baseline_pass3",
    }

    out_json = root / "baseline_pass_eval_summary_{}.json".format(stamp)
    out_txt = root / "baseline_pass_eval_summary_{}.txt".format(stamp)
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "baseline pass evaluation summary ({})".format(stamp),
        "pass1 avg success rate: {:.2f}%".format(pass1_avg * 100.0),
        "run rates: {}".format(", ".join(["{:.2f}%".format(r * 100.0) for r in run_rates])),
        "pass3 success: {}/{} ({:.2f}%)".format(pass3_success, total_cases, pass3_rate * 100.0),
        "output prefix: benchmark_baseline_pass3",
        "result roots:",
    ]
    lines.extend(["  - {}".format(p) for p in run_roots])
    out_txt.write_text("\n".join(lines), encoding="utf-8")

    print("Wrote: {}".format(out_json))
    print("Wrote: {}".format(out_txt))


if __name__ == "__main__":
    main()
