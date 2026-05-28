#!/usr/bin/env python3
import argparse, json, subprocess, time
from pathlib import Path

SUCCESS_MARKER = "Allrun executed successfully without errors."


def run_once(root, trial_tag):
    cmd=["bash", str(root/"run_unified_nemo_benchmark.sh"), trial_tag]
    print("RUN:", " ".join(cmd))
    p=subprocess.run(cmd, cwd=str(root))
    if p.returncode!=0:
        raise RuntimeError("run_unified_nemo_benchmark.sh failed")


def collect(results_root):
    d={}
    for op in results_root.rglob("output.txt"):
        txt=op.read_text(encoding="utf-8", errors="ignore")
        cid=str(op.parent.relative_to(results_root)).replace("\\","/")
        d[cid]=(SUCCESS_MARKER in txt)
    return d


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--trials", type=int, default=3)
    args=ap.parse_args()

    root=Path(args.root).resolve()
    stamp=time.strftime("%Y%m%d_%H%M%S")
    run_rates=[]
    union={}
    roots=[]

    for i in range(1,args.trials+1):
      tag=f"{stamp}_run{i}"
      run_once(root, tag)
      rr=root/f"benchmark_unified_nemo_{tag}"/"results"
      if not rr.exists():
        raise FileNotFoundError(rr)
      roots.append(str(rr))
      res=collect(rr)
      total=len(res); succ=sum(1 for v in res.values() if v)
      rate=(succ/total) if total else 0.0
      run_rates.append(rate)
      for c,ok in res.items():
        union[c]=union.get(c,False) or ok
      print(f"[unified_nemo] run{i}: {succ}/{total} ({rate*100:.2f}%)")

    total_cases=len(union)
    pass3_success=sum(1 for v in union.values() if v)
    pass3_rate=(pass3_success/total_cases) if total_cases else 0.0
    pass1_avg=sum(run_rates)/len(run_rates) if run_rates else 0.0

    summary={
      "timestamp":stamp,
      "group":"unified_nemo",
      "run_rates":run_rates,
      "pass1_avg_rate":pass1_avg,
      "pass3_success":pass3_success,
      "total_cases":total_cases,
      "pass3_rate":pass3_rate,
      "output_prefix":"benchmark_unified_nemo",
      "result_roots":roots,
    }

    out_json=root/f"unified_nemo_pass_eval_{stamp}.json"
    out_txt=root/f"unified_nemo_pass_eval_{stamp}.txt"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    out_txt.write_text(
      "\n".join([
        f"unified_nemo pass1 avg success rate: {pass1_avg*100:.2f}%",
        "run rates: "+", ".join([f"{r*100:.2f}%" for r in run_rates]),
        f"pass3 success: {pass3_success}/{total_cases} ({pass3_rate*100:.2f}%)",
        "output prefix: benchmark_unified_nemo",
      ]), encoding="utf-8"
    )
    print(out_json)
    print(out_txt)

if __name__=="__main__":
    main()
