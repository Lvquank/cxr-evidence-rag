"""Run fresh CUDA/high-effort smoke tests, four full latency runs, then metrics."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.profile_learned_gate_latency import write_json
from scripts.validate_common_studies import sha256


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def audit_branch(path, count, seen):
    config=read_json(path/"run_config.json")
    summary=read_json(path/"latency_summary.json")
    assert summary["complete"] and summary["succeeded"]==count and summary["failed"]==0
    assert config["reasoning_effort"]=="high" and config["max_output_tokens"]==8192 and config["device"]=="cuda"
    expected=read_json(path/"manifest.json")["requests"][:count]
    assert [r["study_id"] for r in summary["results"]]==[str(r["study_id"]) for r in expected]
    routes, tokens = Counter(), []
    report_paths=[]
    for row in summary["results"]:
        report_path=ROOT/row["summary"]
        report=read_json(report_path)
        assert report["reasoning_effort"]=="high" and report["api"]["model"]=="gpt-5.6-luna"
        assert not report["api"]["cached"] and report["api"]["response_id"] not in seen
        assert report["prediction"] and report["prediction"]["problems"]
        seen.add(report["api"]["response_id"])
        tokens.append(report["api"]["usage"]["output_tokens_details"]["reasoning_tokens"])
        routes[row["route"]]+=1
        report_paths.append({"study_id":row["study_id"],"path":str(report_path),"sha256":sha256(report_path)})
    result={"passed":True,"study_count":count,"valid_reports":count,"device":"cuda","reasoning_effort":"high","max_output_tokens":8192,"routes":dict(routes),"reasoning_token_min":min(tokens),"reasoning_token_max":max(tokens),"reasoning_token_total":sum(tokens),"fresh_unique_response_ids":True,"report_hashes":report_paths}
    write_json(path/"final_audit.json",result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root",type=Path,default=ROOT/"outputs/high_effort_4branches_20261031")
    parser.add_argument("--smoke-only",action="store_true")
    parser.add_argument("--preview",action="store_true",help="Print the exact plan without writing files or making API calls.")
    args=parser.parse_args()
    root=args.output_root.resolve()
    if args.preview:
        print(json.dumps({"output_root":str(root),"model":"gpt-5.6-luna","reasoning_effort":"high","max_output_tokens":8192,"device":"cuda","api_destination":"https://api.openai.com/v1/responses","api_payload":"IU-Xray query image when routed to vision; report/evidence prompt, classifier scores, retrieval summaries, and selected-label JSON schema; no new training","smoke_calls":4,"full_studies_per_branch":206,"full_branches":["structured_rule","structured_learned_gate","baseline_a","baseline_b"],"full_calls":824,"postprocessing":"CUDA CheXbert/RadGraph, Table5/6, 2000 paired bootstrap samples"},ensure_ascii=False,indent=2))
        return
    root.mkdir(parents=True,exist_ok=True)
    logs=root/"driver_logs"
    logs.mkdir(exist_ok=True)
    state={"status":"running","started_at":datetime.now(timezone.utc).isoformat(),"reasoning_effort":"high","max_output_tokens":8192,"device":"cuda","source_manifest_sha256":sha256(ROOT/"data/evaluations/leave_one_out_206_manifest.json"),"completed_stages":[]}
    write_json(root/"execution_status.json",state)
    snapshot=root/"generation_code_snapshot"
    for path in [ROOT/"AGENTS.md",ROOT/"src/llm_client.py",*(ROOT/"scripts").glob("profile_*latency.py"),*(ROOT/"scripts").glob("run_baseline*.py"),ROOT/"scripts/run_full_pipeline.py",Path(__file__)]:
        dest=snapshot/path.relative_to(ROOT)
        if not dest.exists():
            dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(path,dest)
    seen=set()
    old_paths=[ROOT/"outputs/rule_latency_206_20261029",ROOT/"outputs/learned_gate_latency_206_20261029",ROOT/"outputs/ab_latency_206_20261030/baseline_a",ROOT/"outputs/ab_latency_206_20261030/baseline_b"]
    for path in old_paths:
        seen.update(row["response_id"] for row in read_json(path/"latency_summary.json")["results"])
    env={**os.environ,"PYTHONUNBUFFERED":"1","PYTHONIOENCODING":"utf-8","HF_HUB_OFFLINE":"1","TRANSFORMERS_OFFLINE":"1","TMP":str(ROOT/"tmp"),"TEMP":str(ROOT/"tmp")}
    def run(stage, command):
        state.update(stage=stage,stage_started_at=datetime.now(timezone.utc).isoformat(),command=command)
        write_json(root/"execution_status.json",state)
        print(f"STAGE {stage}",flush=True)
        log=logs/(stage+".log")
        with log.open("a",encoding="utf-8") as f:
            f.write("COMMAND: "+subprocess.list2cmdline(command)+"\n")
            f.flush()
            process=subprocess.Popen(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,env=env)
            state["child_pid"]=process.pid
            write_json(root/"execution_status.json",state)
            code=process.wait()
        if code:
            state.update(status="failed",exit_code=code,failed_log=str(log))
            write_json(root/"execution_status.json",state)
            raise RuntimeError(f"Stage {stage} failed; see {log}")
        state["completed_stages"].append(stage)
        write_json(root/"execution_status.json",state)
    common=["--device","cuda","--reasoning-effort","high","--max-output-tokens","8192"]
    py=sys.executable
    try:
        for phase,count in [("smoke",1),("full",206)]:
            phase_root=root/"smoke" if phase=="smoke" else root
            paths={"structured_rule":phase_root/"structured_rule","structured_learned_gate":phase_root/"structured_learned_gate","baseline_a":phase_root/"baselines/baseline_a","baseline_b":phase_root/"baselines/baseline_b"}
            for name,branch in [("structured_rule","rule"),("structured_learned_gate","learned")]:
                path=paths[name]
                summary_path=path/"latency_summary.json"
                if not (summary_path.exists() and read_json(summary_path)["complete"]):
                    command=[py,"-u",str(ROOT/"scripts/profile_learned_gate_latency.py"),"--branch",branch,"--limit",str(count),"--output-root",str(path),*common]
                    if summary_path.exists(): command.append("--resume")
                    run(f"{phase}_{name}",command)
                audit_branch(path,count,seen)
            baseline_root=phase_root/"baselines"
            complete=all((paths[n]/"latency_summary.json").exists() and read_json(paths[n]/"latency_summary.json")["complete"] for n in ["baseline_a","baseline_b"])
            if not complete:
                command=[py,"-u",str(ROOT/"scripts/profile_baseline_latency.py"),"--baseline","both","--limit",str(count),"--output-root",str(baseline_root),"--reference-run",str(paths["structured_rule"]),*common]
                if baseline_root.exists() and any(baseline_root.iterdir()): command.append("--resume")
                run(f"{phase}_baselines",command)
            for name in ["baseline_a","baseline_b"]: audit_branch(paths[name],count,seen)
            write_json(phase_root/"four_branch_audit.json",{"passed":True,"count_per_branch":count,"reasoning_effort":"high","max_output_tokens":8192,"device":"cuda","cross_run_response_ids_disjoint":True})
            if args.smoke_only: break
        if not args.smoke_only:
            metrics=root/"metrics"
            base=[py,"-u",str(ROOT/"scripts/recompute_latency_tables.py")]
            options=["--runs-root",str(root),"--expected-reasoning-effort","high","--expected-max-output-tokens","8192","--output-dir",str(metrics)]
            for stage in ["self-check","prepare","chexbert","radgraph","summarize","finalize"]:
                artifact={"prepare":"input_manifest.json","chexbert":"chexbert_raw/inference_manifest.json","radgraph":"radgraph_raw/inference_manifest.json","summarize":"table6_results.json","finalize":"final_audit.json"}.get(stage)
                if artifact and (metrics/artifact).exists(): continue
                run("metrics_"+stage,base+[stage,*options])
        state.update(status="complete",finished_at=datetime.now(timezone.utc).isoformat())
        write_json(root/"execution_status.json",state)
        print("COMPLETE: "+str(root),flush=True)
    except BaseException as error:
        state.update(status="failed",error=str(error))
        write_json(root/"execution_status.json",state)
        raise


if __name__=="__main__":
    main()
