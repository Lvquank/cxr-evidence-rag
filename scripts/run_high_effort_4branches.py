"""Run fresh CUDA/high-effort smoke tests, four full latency runs, then metrics."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from math import isclose
import os
from pathlib import Path
import shutil
import subprocess
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.profile_learned_gate_latency import GATE_ROOT, query_image_path, write_json
from scripts.validate_common_studies import sha256

MANIFEST = ROOT / "data/evaluations/leave_one_out_206_manifest.json"
BRANCH_PATHS = {"structured_rule": "structured_rule", "structured_learned_gate": "structured_learned_gate",
                "baseline_a": "baselines/baseline_a", "baseline_b": "baselines/baseline_b"}
HISTORICAL_HIGH_RUN = ROOT / "outputs/high_effort_4branches_20261031"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def audit_branch(path, requests, manifest_sha, lambda_global, seen):
    config=read_json(path/"run_config.json")
    summary=read_json(path/"latency_summary.json")
    count=len(requests)
    assert summary["complete"] and summary["succeeded"]==count and summary["failed"]==0
    assert config["reasoning_effort"]=="high" and config["max_output_tokens"]==8192 and config["device"]=="cuda"
    assert config["manifest_sha256"] == manifest_sha and config.get("lambda_global") == lambda_global
    assert read_json(path/"manifest.json")["requests"][:count] == requests
    assert len(summary["results"]) == count
    routes, tokens = Counter(), []
    consumption = Counter()
    scored_candidates = 0
    report_paths=[]
    for row, item in zip(summary["results"], requests, strict=True):
        assert row["study_id"] == str(item["study_id"]) and row["request_id"] == item["request_id"]
        report_path=(ROOT/row["summary"]).resolve()
        assert report_path.is_relative_to((path/"llm"/row["request_id"]).resolve())
        report=read_json(report_path)
        assert report["reasoning_effort"]=="high" and report["api"]["model"]=="gpt-5.6-luna"
        assert report["source"]["study_id"] == row["study_id"]
        assert report["source"]["image_name"] == query_image_path(item).name
        assert not report["api"]["cached"] and row["response_id"] == report["api"]["response_id"]
        assert row["response_id"] and row["response_id"] not in seen
        assert report["prediction"] and report["prediction"]["problems"]
        seen.add(row["response_id"])
        usage = report["api"]["usage"]
        tokens.append(usage["output_tokens_details"]["reasoning_tokens"])
        consumption.update({key: usage[key] for key in ("input_tokens", "output_tokens", "total_tokens")})
        routes[row["route"]]+=1
        report_paths.append({"study_id":row["study_id"],"path":str(report_path),"sha256":sha256(report_path)})
        if lambda_global is not None:
            raw_path = path/"raw_evidence"/f"{row['request_id']}.json"
            raw = read_json(raw_path)
            retrieval = raw["retriever"]
            assert retrieval["local_rerank_enabled"] == bool(raw["classifier"]["selected_labels"])
            assert retrieval["lambda_global"] == lambda_global
            assert retrieval["excluded_study_ids"] == [row["study_id"]]
            for label in retrieval["label_evidence"]:
                for key in ("supporting_evidence", "matched_non_support_evidence"):
                    for candidate in label[key]:
                        assert str(candidate["study_id"]) != row["study_id"]
                        if (local := candidate.get("densenet_local_similarity")) is not None:
                            expected = lambda_global * candidate["base_rank_score"] + (1 - lambda_global) * local
                            # All three saved scores are rounded to six decimal places.
                            assert isclose(candidate["final_rank_score"], expected, rel_tol=0, abs_tol=1.01e-6)
                            scored_candidates += 1
            report_paths[-1]["raw_evidence_sha256"] = sha256(raw_path)
    result={"passed":True,"study_count":count,"valid_reports":count,"device":"cuda","reasoning_effort":"high","max_output_tokens":8192,"lambda_global":lambda_global,"verified_rerank_candidates":scored_candidates,"routes":dict(routes),"reasoning_token_min":min(tokens),"reasoning_token_max":max(tokens),"reasoning_token_total":sum(tokens),"api_usage":dict(consumption),"fresh_unique_response_ids":True,"report_hashes":report_paths}
    write_json(path/"final_audit.json",result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root",type=Path,required=True)
    parser.add_argument("--lambda-global",type=float,default=0.9,help="Global score weight for both structured branches.")
    parser.add_argument("--smoke-only",action="store_true")
    parser.add_argument("--preview",action="store_true",help="Print the exact plan without writing files or making API calls.")
    args=parser.parse_args()
    if not 0 <= args.lambda_global <= 1:
        parser.error("--lambda-global must be between 0 and 1.")
    root=args.output_root.resolve()
    manifest_sha=sha256(MANIFEST)
    manifest=read_json(MANIFEST)
    requests=manifest["requests"]
    quotas=Counter(item["case_label"] for item in requests)
    assert len({str(item["study_id"]) for item in requests})==len(requests)==206
    assert len({item["request_id"] for item in requests})==206
    assert quotas == manifest["selected_per_class"] and quotas["normal"] == 30
    if args.preview:
        print(json.dumps({"output_root":str(root),"model":"gpt-5.6-luna","reasoning_effort":"high","max_output_tokens":8192,"structured_lambda_global":args.lambda_global,"device":"cuda","manifest_sha256":manifest_sha,"selected_per_class":quotas,"api_destination":"https://api.openai.com/v1/responses","api_payload":"IU-Xray query image when routed to vision; report/evidence prompt, classifier scores, retrieval summaries, and selected-label JSON schema; no new training","smoke_calls":4,"full_studies_per_branch":206,"full_branches":list(BRANCH_PATHS),"full_calls":824,"postprocessing":"CUDA CheXbert/RadGraph, Table5/6, 2000 paired bootstrap samples"},ensure_ascii=False,indent=2))
        return
    if not torch.cuda.is_available():
        parser.error("CUDA is required; CPU fallback is disabled.")
    for item in requests:
        query_image_path(item)
    status_path = root/"execution_status.json"
    previous = {}
    if root.exists() and any(root.iterdir()):
        if not status_path.is_file():
            parser.error("Use a new output directory or resume a run created by this driver.")
        previous = read_json(status_path)
        if previous.get("lambda_global") != args.lambda_global or previous.get("source_manifest_sha256") != manifest_sha:
            parser.error("Existing run configuration differs. Use a new output directory.")
    root.mkdir(parents=True,exist_ok=True)
    logs=root/"driver_logs"
    logs.mkdir(exist_ok=True)
    state={"status":"running","started_at":previous.get("started_at",datetime.now(timezone.utc).isoformat()),"reasoning_effort":"high","max_output_tokens":8192,"lambda_global":args.lambda_global,"device":"cuda","gpu":torch.cuda.get_device_name(0),"torch":torch.__version__,"cuda_runtime":torch.version.cuda,"source_manifest_sha256":manifest_sha,"selected_per_class":dict(quotas),"completed_stages":previous.get("completed_stages",[])}
    snapshot=root/"generation_code_snapshot"
    for path in [ROOT/"AGENTS.md",ROOT/"src/llm_client.py",*(ROOT/"scripts").glob("profile_*latency.py"),*(ROOT/"scripts").glob("run_baseline*.py"),ROOT/"scripts/run_full_pipeline.py",ROOT/"scripts/test_classifier_to_retriever.py",ROOT/"scripts/run_terra_agreement_vision_batch.py",ROOT/"src/retriever/faiss_retriever.py",ROOT/"src/retriever/densenet_local_reranker.py",Path(__file__)]:
        dest=snapshot/path.relative_to(ROOT)
        if dest.exists():
            if sha256(dest) != sha256(path):
                parser.error(f"Generation code has changed since this run started: {path}")
        else:
            dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(path,dest)
    state["gate_artifact_hashes"] = {str(path.relative_to(ROOT)):sha256(path) for path in [GATE_ROOT/"folds.json",*sorted((GATE_ROOT/"models").glob("fold_*.pkl"))]}
    if previous and previous.get("gate_artifact_hashes") != state["gate_artifact_hashes"]:
        parser.error("Frozen gate artifacts have changed since this run started.")
    write_json(status_path,state)
    seen=set()
    old_paths=[ROOT/"outputs/rule_latency_206_20261029",ROOT/"outputs/learned_gate_latency_206_20261029",ROOT/"outputs/ab_latency_206_20261030/baseline_a",ROOT/"outputs/ab_latency_206_20261030/baseline_b"]
    old_paths.extend(HISTORICAL_HIGH_RUN / path for path in BRANCH_PATHS.values())
    for path in old_paths:
        historical = path/"latency_summary.json"
        if historical.is_file():
            seen.update(row["response_id"] for row in read_json(historical)["results"])
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
        if stage not in state["completed_stages"]:
            state["completed_stages"].append(stage)
        write_json(root/"execution_status.json",state)
    common=["--device","cuda","--reasoning-effort","high","--max-output-tokens","8192"]
    py=sys.executable
    try:
        for phase,count in [("smoke",1),("full",206)]:
            phase_root=root/"smoke" if phase=="smoke" else root
            paths={name:phase_root/path for name,path in BRANCH_PATHS.items()}
            audits={}
            for name,branch in [("structured_rule","rule"),("structured_learned_gate","learned")]:
                path=paths[name]
                summary_path=path/"latency_summary.json"
                if not (summary_path.exists() and read_json(summary_path)["complete"]):
                    command=[py,"-u",str(ROOT/"scripts/profile_learned_gate_latency.py"),"--branch",branch,"--limit",str(count),"--output-root",str(path),"--lambda-global",str(args.lambda_global),*common]
                    if summary_path.exists(): command.append("--resume")
                    run(f"{phase}_{name}",command)
                audits[name]=audit_branch(path,requests[:count],manifest_sha,args.lambda_global,seen)
            baseline_root=phase_root/"baselines"
            complete=all((paths[n]/"latency_summary.json").exists() and read_json(paths[n]/"latency_summary.json")["complete"] for n in ["baseline_a","baseline_b"])
            if not complete:
                command=[py,"-u",str(ROOT/"scripts/profile_baseline_latency.py"),"--baseline","both","--limit",str(count),"--output-root",str(baseline_root),"--reference-run",str(paths["structured_rule"]),*common]
                if baseline_root.exists() and any(baseline_root.iterdir()): command.append("--resume")
                run(f"{phase}_baselines",command)
            for name in ["baseline_a","baseline_b"]:
                audits[name]=audit_branch(paths[name],requests[:count],manifest_sha,None,seen)
            write_json(phase_root/"four_branch_audit.json",{"passed":True,"count_per_branch":count,"reasoning_effort":"high","max_output_tokens":8192,"structured_lambda_global":args.lambda_global,"device":"cuda","cross_run_response_ids_disjoint":True})
            branch_usage={name:{**audit["api_usage"],"reasoning_tokens":audit["reasoning_token_total"],"reports":audit["valid_reports"]} for name,audit in audits.items()}
            totals = sum((Counter(usage) for usage in branch_usage.values()), Counter())
            write_json(phase_root/"api_usage_totals.json",{"scope":"Selected valid responses in this phase; excludes invalid and unrecorded attempts.","by_branch":branch_usage,"total":dict(totals)})
            if args.smoke_only: break
        if not args.smoke_only:
            metrics=root/"metrics"
            base=[py,"-u",str(ROOT/"scripts/recompute_latency_tables.py")]
            options=["--runs-root",str(root),"--expected-reasoning-effort","high","--expected-max-output-tokens","8192","--expected-lambda-global",str(args.lambda_global),"--output-dir",str(metrics)]
            for stage in ["self-check","prepare","chexbert","radgraph","summarize","finalize"]:
                artifact={"prepare":"input_manifest.json","chexbert":"chexbert_raw/inference_manifest.json","radgraph":"radgraph_raw/inference_manifest.json","summarize":"table6_results.json","finalize":"final_audit.json"}.get(stage)
                if artifact and (metrics/artifact).exists(): continue
                run("metrics_"+stage,base+[stage,*options])
            comparison = root/"lambda_comparison"
            if args.lambda_global != 0.9 and not (comparison/"LAMBDA_COMPARISON.md").exists():
                run("compare_lambda",[py,"-u",str(ROOT/"scripts/compare_high_effort_retrieval_lambda.py"),
                                      "--reference-metrics",str(HISTORICAL_HIGH_RUN/"metrics"),"--run-metrics",str(metrics),
                                      "--run-lambda",str(args.lambda_global),"--output-dir",str(comparison)])
        state.update(status="complete",finished_at=datetime.now(timezone.utc).isoformat())
        write_json(root/"execution_status.json",state)
        print("COMPLETE: "+str(root),flush=True)
    except BaseException as error:
        state.update(status="failed",error=str(error))
        write_json(root/"execution_status.json",state)
        raise


if __name__=="__main__":
    main()
