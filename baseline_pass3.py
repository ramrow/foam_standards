#!/usr/bin/env python3
import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CFG = json.loads((ROOT / "baseline_pass3_config.json").read_text(encoding="utf-8"))
SUCCESS_MARKER = CFG["success_marker"]
TRIALS = int(CFG.get("trials", 3))


def collect_successes(results_root: Path):
    case_success = {}
    for op in results_root.rglob("output.txt"):
        txt = op.read_text(encoding="utf-8", errors="ignore")
        cid = str(op.parent.relative_to(results_root)).replace("\\", "/")
        case_success[cid] = (SUCCESS_MARKER in txt)
    return case_success


def run_one(trial_tag: str):
    script = ROOT / CFG["baseline_script"]
    cmd = ["bash", str(script), CFG["baseline_run_name"], trial_tag]
    print("RUN:", " ".join(cmd))
    p = subprocess.run(cmd, cwd=str(ROOT))
    if p.returncode != 0:
        raise RuntimeError("baseline script failed with code {}".format(p.returncode))

    rel = CFG["results_pattern"].format(trial_tag=trial_tag)
    results_root = ROOT / rel
    if not results_root.exists():
        raise FileNotFoundError("Results folder missing: {}".format(results_root))
    return collect_successes(results_root), results_root


def main():
    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_rates = []
    union_pass = {}
    run_roots = []

    for i in range(1, TRIALS + 1):
        tag = "baseline_pass3_{}_run{}".format(stamp, i)
        res, rr = run_one(tag)
        run_roots.append(str(rr))

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
        "group": "baseline_all_finetuned",
        "run_rates": run_rates,
        "pass1_avg_rate": pass1_avg,
        "pass3_success": pass3_success,
        "total_cases": total_cases,
        "pass3_rate": pass3_rate,
        "result_roots": run_roots,
    }

    out_json = ROOT / "baseline_pass3_summary_{}.json".format(stamp)
    out_txt = ROOT / "baseline_pass3_summary_{}.txt".format(stamp)
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "Baseline pass evaluation summary ({})".format(stamp),
        "pass1 avg success rate: {:.2f}%".format(pass1_avg * 100.0),
        "run rates: {}".format(", ".join(["{:.2f}%".format(r * 100.0) for r in run_rates])),
        "pass3 success: {}/{} ({:.2f}%)".format(pass3_success, total_cases, pass3_rate * 100.0),
        "result roots:",
    ]
    lines.extend(["  - {}".format(p) for p in run_roots])
    out_txt.write_text("\n".join(lines), encoding="utf-8")

    print("Wrote:", out_json)
    print("Wrote:", out_txt)


if __name__ == "__main__":
    main()
