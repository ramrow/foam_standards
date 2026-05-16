#!/usr/bin/env python3
import argparse, json, os, subprocess, sys, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
SUCCESS_MARKER = "Allrun executed successfully without errors."

def run_once(root: Path, cfg: Path, out_root: Path, workers: int, port: int | None):
    out_root.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(root / "benchmark_finetuned.py"), "--all_finetuned", "--finetuned_config", str(cfg), "--output_root", str(out_root), "--workers", str(workers)]
    env = os.environ.copy()
    if port is not None:
        env["OPENAI_BASE_URL"] = f"http://127.0.0.1:{port}/v1"
        env["OPENAI_API_BASE"] = f"http://127.0.0.1:{port}/v1"
    p = subprocess.run(cmd, cwd=str(root), env=env)
    if p.returncode != 0:
        raise RuntimeError(f"benchmark failed: {p.returncode}")

def collect_successes(results_root: Path):
    case_success = {}
    for op in results_root.rglob("output.txt"):
        txt = op.read_text(encoding="utf-8", errors="ignore")
        cid = str(op.parent.relative_to(results_root)).replace("\\", "/")
        case_success[cid] = SUCCESS_MARKER in txt
    return case_success

def eval_group(root: Path, cfg: Path, group_name: str, trials: int, workers: int, stamp: str, port: int | None):
    base = root / f"pass_eval_{group_name}_{stamp}"
    run_rates, union_pass = [], {}
    for i in range(1, trials + 1):
        run_out = base / f"run{i}"
        run_once(root, cfg, run_out, workers, port)
        res = collect_successes(run_out / "results")
        total = len(res)
        succ = sum(1 for v in res.values() if v)
        run_rates.append((succ / total) if total else 0.0)
        for c, ok in res.items():
            union_pass[c] = union_pass.get(c, False) or ok
    total_cases = len(union_pass)
    pass3_success = sum(1 for v in union_pass.values() if v)
    return {"group": group_name, "run_rates": run_rates, "pass1_avg_rate": sum(run_rates)/len(run_rates) if run_rates else 0.0, "pass3_success": pass3_success, "total_cases": total_cases, "pass3_rate": (pass3_success/total_cases) if total_cases else 0.0, "output_base": str(base)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--groups", nargs="*", default=["B","C","E2"], choices=["B","C","E2"])
    ap.add_argument("--parallel", action="store_true")
    ap.add_argument("--group-ports", nargs="*", default=[], help="e.g. B=8001 C=8002 E2=8003")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    cfg_map = {"B": root / "grouping_B_finetuned_models.json", "C": root / "grouping_C_finetuned_models.json", "E2": root / "grouping_E2_finetuned_models.json"}
    for g in args.groups:
        if not cfg_map[g].exists():
            raise FileNotFoundError(f"Missing config: {cfg_map[g]}")
    port_map = {}
    for kv in args.group_ports:
        if "=" in kv:
            k,v = kv.split("=",1)
            try: port_map[k.strip()] = int(v.strip())
            except: pass

    stamp = time.strftime("%Y%m%d_%H%M%S")
    summary = {"timestamp": stamp, "groups": []}

    if args.parallel:
        with ThreadPoolExecutor(max_workers=len(args.groups)) as ex:
            futs = [ex.submit(eval_group, root, cfg_map[g], f"grouping_{g}", args.trials, args.workers, stamp, port_map.get(g)) for g in args.groups]
            for fut in as_completed(futs):
                summary["groups"].append(fut.result())
    else:
        for g in args.groups:
            summary["groups"].append(eval_group(root, cfg_map[g], f"grouping_{g}", args.trials, args.workers, stamp, port_map.get(g)))

    out_json = root / f"pass_eval_summary_{stamp}.json"
    out_txt = root / f"pass_eval_summary_{stamp}.txt"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = [f"Pass evaluation summary ({stamp})"]
    for g in summary["groups"]:
        lines += ["", g["group"], f"  pass1 avg success rate: {g['pass1_avg_rate']*100:.2f}%", f"  run rates: {', '.join(f'{r*100:.2f}%' for r in g['run_rates'])}", f"  pass3 success: {g['pass3_success']}/{g['total_cases']} ({g['pass3_rate']*100:.2f}%)", f"  outputs: {g['output_base']}"]
    out_txt.write_text("\n".join(lines), encoding="utf-8")
    print(out_json)
    print(out_txt)

if __name__ == "__main__":
    main()
