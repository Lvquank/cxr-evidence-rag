"""Recompute Tables 5/6 from four completed runs on the frozen cohort, using CUDA.

Run stages separately with .venv-bertscore/Scripts/python.exe. No API calls.
All selected reports come from latency_per_case.csv, never a latest-file guess.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from functools import partial
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import pandas as pd
import torch

from evaluate_llm_outputs import load_ontology, normalize_target_problems, split_problems, target_problem_metrics
from validate_common_studies import build_canonical_pairs, write_canonical, sha256
from src.chexbert_metrics import _decisions, compute_chexbert_aggregate

RUNS = {
    "structured_rule": "outputs/rule_latency_206_20261029",
    "structured_learned_gate": "outputs/learned_gate_latency_206_20261029",
    "baseline_a": "outputs/ab_latency_206_20261030/baseline_a",
    "baseline_b": "outputs/ab_latency_206_20261030/baseline_b",
}
LABELS14 = ["Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion", "Edema", "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices", "No Finding"]
LABELS6 = ["Atelectasis", "Cardiomegaly", "Edema", "Pleural Effusion", "Pneumonia", "Pneumothorax"]
T5 = ["precision", "recall", "micro_f1", "sample_f1", "label_groundedness", "unmatched_rate", "normal_f1"]
T6 = ["chexbert_all_micro_f1", "chexbert_headline_micro_f1", "radgraph_entity_f1", "radgraph_relation_f1", "radgraph_complete_f1"]
PAIRS = [("structured_learned_gate", "structured_rule"), ("baseline_b", "baseline_a"), ("baseline_b", "structured_rule"), ("baseline_a", "structured_rule"), ("structured_learned_gate", "baseline_a"), ("structured_learned_gate", "baseline_b")]
SAMPLES, SEED = 2000, 20260904
EXPECTED_EFFORT, EXPECTED_MAX_TOKENS = "none", 160


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def save_csv(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def require_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required. CPU fallback is prohibited for this run.")
    torch.cuda.set_device(0)
    return {"device": "cuda:0", "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "cuda_runtime": torch.version.cuda}


def tensor(values):
    return torch.as_tensor(values, dtype=torch.float64, device="cuda:0")


def div(a, b, empty=0.):
    return torch.where(b != 0, a / b.clamp_min(1), torch.full_like(b, empty))


def paired_bootstrap(values, metric_names, ids, out, table):
    """values[name] is (study, sufficient-statistic); all arithmetic is CUDA."""
    index_np = np.random.default_rng(SEED).integers(0, len(ids), size=(SAMPLES, len(ids)))
    np.save(out / f"{table}_bootstrap_indices.npy", index_np)
    index = torch.as_tensor(index_np, dtype=torch.long, device="cuda:0")
    results, draws = {}, {}
    for name, raw in values.items():
        v = tensor(raw)
        stats = torch.cat([v.sum(0, keepdim=True), v[index].sum(1)], dim=0)
        n = len(ids)
        if table == "table5":
            tp, fp, fn, sf, ntp, nfp, nfn = stats.unbind(1)
            precision = div(tp, tp + fp)
            scored = torch.stack([precision, div(tp, tp + fn), div(2*tp, 2*tp+fp+fn), sf/n, div(tp, tp+fp, 1.), div(fp, tp+fp), div(2*ntp, 2*ntp+nfp+nfn)], dim=1)
        else:
            atp, afp, afn, htp, hfp, hfn, ent, rel, complete = stats.unbind(1)
            scored = torch.stack([div(2*atp, 2*atp+afp+afn), div(2*htp, 2*htp+hfp+hfn), ent/n, rel/n, complete/n], dim=1)
        draws[name] = scored[1:]
        bounds = torch.quantile(scored[1:], tensor([.025, .975]), dim=0)
        results[name] = {metric: {"estimate": float(scored[0,i]), "ci95_lower": float(bounds[0,i]), "ci95_upper": float(bounds[1,i])} for i, metric in enumerate(metric_names)}
    deltas = []
    for left, right in PAIRS:
        bounds = torch.quantile(draws[left]-draws[right], tensor([.025,.975]), dim=0)
        for i, metric in enumerate(metric_names):
            lo, hi = float(bounds[0,i]), float(bounds[1,i])
            deltas.append({"comparison": f"{left}_minus_{right}", "metric": metric, "delta": results[left][metric]["estimate"]-results[right][metric]["estimate"], "ci95_lower": lo, "ci95_upper": hi, "ci_contains_zero": lo <= 0 <= hi})
    payload = {"study_count": len(ids), "study_ids": ids, "bootstrap": {"samples": SAMPLES, "seed": SEED, "method": "paired study percentile bootstrap", "arithmetic_device": "cuda:0", "index_generator": "numpy.default_rng(20260904), numeric study-ID order", "delta_direction": "explicit left_minus_right", "indices_sha256": sha256(out / f"{table}_bootstrap_indices.npy")}, "absolute": results, "paired_deltas": deltas}
    save_json(out / f"{table}_results.json", payload)
    save_csv(out / f"{table}_paired_deltas.csv", deltas)
    rows = []
    for name, metrics in results.items():
        row = {"pipeline": name, "study_count": len(ids)}
        for metric, scores in metrics.items():
            row.update({metric if key == "estimate" else f"{metric}_{key}": val for key,val in scores.items()})
        rows.append(row)
    save_csv(out / f"{table}.csv", rows)
    lines = [f"# {table.capitalize()} (N={len(ids)})", "", "Point estimates [95% CI]; 2,000 paired study resamples, seed 20260904.", "", "| Pipeline | " + " | ".join(metric_names) + " |", "|---|" + "---:|"*len(metric_names)]
    for name, metrics in results.items():
        lines.append("| " + name + " | " + " | ".join(f"{x['estimate']:.4f} [{x['ci95_lower']:.4f}, {x['ci95_upper']:.4f}]" for x in metrics.values()) + " |")
    lines += ["", "All six explicit paired comparisons, including whether CI contains zero, are in the paired-deltas CSV and results JSON."]
    (out / f"{table}.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    return payload


def prepare(out, expected_lambda_global=None):
    if (out / "input_manifest.json").exists():
        raise ValueError("Preparation already exists; use another output directory.")
    hardware = require_cuda()
    ontology = load_ontology()
    assert ontology["target_labels"] == LABELS6
    reports_path = ROOT / "src/data/indiana_reports.csv"
    raw_refs = read_csv(reports_path)
    ref_ids = [r["uid"] for r in raw_refs]
    assert len(ref_ids) == len(set(ref_ids)), "Duplicate reference IDs"
    refs = {r["uid"]: r for r in raw_refs}
    manifest = read_json(ROOT / "data/evaluations/leave_one_out_206_manifest.json")
    reqs = manifest["requests"]
    ids = sorted((str(r["study_id"]) for r in reqs), key=int)
    assert len(ids) == len(set(ids)) == 206
    reqmap = {r["request_id"]: str(r["study_id"]) for r in reqs}
    configs = {name: read_json(ROOT / relative / "run_config.json") for name, relative in RUNS.items()}
    lambda_global = configs["structured_rule"]["lambda_global"]
    if not 0 <= lambda_global <= 1 or configs["structured_learned_gate"]["lambda_global"] != lambda_global:
        raise ValueError("The structured branches must use the same lambda_global in [0, 1].")
    if expected_lambda_global is not None and lambda_global != expected_lambda_global:
        raise ValueError(f"Structured lambda_global is {lambda_global}, expected {expected_lambda_global}.")
    response_ids, provenance, all_rows, audit = set(), [], [], {}
    target_stats = {}
    source_files = {reports_path, ROOT/"config/evaluation_ontology.json", ROOT/"data/evaluations/leave_one_out_206_manifest.json"}
    for name, relative in RUNS.items():
        run = ROOT / relative
        cfg = configs[name]
        assert cfg["model"] == "gpt-5.6-luna" and cfg["reasoning_effort"] == EXPECTED_EFFORT and cfg["max_output_tokens"] == EXPECTED_MAX_TOKENS
        assert cfg["device"] == "cuda" and cfg["manifest_sha256"] == sha256(ROOT/"data/evaluations/leave_one_out_206_manifest.json")
        assert read_json(run/"manifest.json")["requests"] == reqs
        assert read_json(run/"final_audit.json")["passed"]
        latency = read_csv(run/"latency_per_case.csv")
        assert len(latency) == 206 and {r["study_id"] for r in latency} == set(ids)
        assert len({r["request_id"] for r in latency}) == 206
        selected_paths, rows, per_study, routes, prompts = set(), {}, [], Counter(), Counter()
        for row in latency:
            sid = row["study_id"]
            assert reqmap[row["request_id"]] == sid and row["status"] == "success"
            path = (ROOT / row["summary"]).resolve()
            assert path.is_relative_to((run/"llm").resolve())
            data = read_json(path)
            assert str(data["source"]["study_id"]) == sid
            assert data["api"]["response_id"] == row["response_id"] and row["response_id"] not in response_ids
            response_ids.add(row["response_id"])
            assert data["api"]["model"] == cfg["model"] and not data["api"]["cached"]
            pred = data["prediction"]
            assert isinstance(pred["problems"], list) and pred["problems"]
            assert all(isinstance(pred[k], str) and pred[k].strip() for k in ["findings", "impression"])
            assert json.loads(data["response"]) == pred == json.loads(data["api"]["text"])
            normalized = normalize_target_problems(split_problems(pred["problems"]), ontology)
            gold = normalize_target_problems(split_problems(refs[sid]["Problems"]), ontology)
            assert not normalized["other_findings"]
            assert not (normalized["normal"] and normalized["target_problems"])
            allowed = data.get("allowed_problem_labels", data.get("classifier_selected_labels"))
            assert allowed is not None and set(normalized["target_problems"]) <= set(allowed)
            if name.startswith("structured"):
                assert data["evidence_schema"] == 5 and data["reasoning_effort"] == EXPECTED_EFFORT
                assert data["route"]["name"] == row["route"]
                assert data["route"]["image_attached"] == (row["image_attached"].lower() == "true")
            else:
                assert row["route"] == "vision_always" and row["image_attached"].lower() == "true"
                if EXPECTED_EFFORT != "none":
                    assert data["reasoning_effort"] == EXPECTED_EFFORT and data["max_output_tokens"] == EXPECTED_MAX_TOKENS
            metric = target_problem_metrics(normalized, gold)
            n = metric["normal_decision"]
            rows[sid] = [metric["tp"], metric["fp"], metric["fn"], metric["f1"], n["tp"], n["fp"], n["fn"]]
            per_study.append({"pipeline": name, "study_id": sid, "problem_tp": metric["tp"], "problem_fp": metric["fp"], "problem_fn": metric["fn"], "problem_f1": metric["f1"], "gold_normal": n["gold_normal"], "predicted_normal": n["predicted_normal"], "predicted_target_problems": ";".join(normalized["target_problems"]), "gold_target_problems": ";".join(gold["target_problems"])})
            reference = refs[sid]
            pair = {"pipeline": name, "study_id": sid, "generated_findings": pred["findings"].strip(), "reference_findings": reference["findings"].strip(), "generated_impression": pred["impression"].strip(), "reference_impression": reference["impression"].strip(), "model": cfg["model"], "source_summary": str(path)}
            all_rows.append(pair)
            provenance.append({"pipeline": name, "request_id": row["request_id"], "study_id": sid, "summary_path": str(path), "summary_sha256": sha256(path), "response_id": row["response_id"], "route": row["route"], "image_attached": row["image_attached"], "reference_problems": reference["Problems"], "reference_text_sha256": hashlib.sha256((pair["reference_findings"]+"\n"+pair["reference_impression"]).encode()).hexdigest(), "prediction_text_sha256": hashlib.sha256((pair["generated_findings"]+"\n"+pair["generated_impression"]).encode()).hexdigest()})
            selected_paths.add(path)
            routes[row["route"]] += 1
            prompts[str((data.get("prompt_type"), data.get("evidence_schema"), data.get("evidence_enrichment"), data.get("baseline")))] += 1
        candidates = set(p.resolve() for p in (run/"llm").rglob("summary_*.json"))
        assert selected_paths == candidates, f"Unaccounted summary candidates in {run}: {candidates-selected_paths}"
        target_stats[name] = [rows[sid] for sid in ids]
        save_csv(out/"per_study"/f"{name}_target_labels.csv", sorted(per_study,key=lambda r:int(r["study_id"])))
        save_csv(out/"clinical_pairs"/f"{name}_clinical_text_pairs.csv", [r for r in all_rows if r["pipeline"] == name])
        audit[name] = {"study_count": len(rows), "selected_reports": len(selected_paths), "unselected_reports": 0, "routes": dict(routes), "prompt_variants": dict(prompts), "run_config": cfg}
        source_files.update(run/x for x in ["manifest.json", "run_config.json", "final_audit.json", "latency_per_case.csv"])
    canonical, validation = build_canonical_pairs(out/"clinical_pairs", required_count=206, expected_pipelines=tuple(RUNS), empty_reference_policy="exclude")
    write_canonical(canonical, validation, out/"canonical_pairs_205.csv", out/"clinical_cohort.json")
    clinical_ids = validation["study_ids"]
    save_json(out/"input_manifest.json", {"structured_lambda_global":lambda_global,"source_cohort_count":206,"source_study_ids":ids,"clinical_study_count":len(clinical_ids),"excluded_empty_reference_study_ids":validation["excluded_empty_reference_study_ids"],"hardware":hardware,"branches":audit,"source_files":{str(p):sha256(p) for p in sorted(source_files)},"selection_policy":"exact summary path and response_id recorded in latency_per_case.csv; all candidates accounted; never latest-file selection", "missing_handoff_paths":[str(p) for p in [ROOT/"docs/CXR_RAG_LNCS_FINAL_v3_TABLE6_ROUTE_UPDATE_20261027/analysis/recompute_table6_chexbert.py",ROOT/"docs/CXR_RAG_LNCS_FINAL_v3_TABLE6_ROUTE_UPDATE_20261027/LaTeX/paper.tex"] if not p.exists()],"reference_policy":"Table5 uses all 206 canonical Problems labels. Table6 excludes study with both reference fields blank in all branches; no reference substitution.","metric_policy":{"ontology":"cxr-target-6-v1","target_labels":LABELS6,"normal":"Separate binary decision: gold explicit normal with zero target labels; predicted zero target labels. Normal never seventh pathology.","sample_f1":"Mean per-study six-target F1, including both-empty target sets as 1, even with non-target reference findings.","groundedness":"pooled TP/(TP+FP), not mean per-study precision; complement unmatched rate; empty pooled emission=1","clinical_text":"non-empty Findings then Impression, joined with newline","chexbert":"positive-class F1; reference uncertain excluded; reference blank and prediction blank/uncertain are not positive; common reference-only mask","chexbert_14_labels":LABELS14,"chexbert_six_labels":LABELS6,"radgraph":"radgraph-xl, official simple/partial/complete rewards reported as entity/relation/complete"}})
    save_csv(out/"report_provenance.csv", provenance)
    paired_bootstrap(target_stats, T5, ids, out, "table5")
    print(f"PREPARED: 824 selected reports, Table5 N=206, Table6 N={len(clinical_ids)}; excluded={validation['excluded_empty_reference_study_ids']}",flush=True)


def clinical_batches(out):
    m = read_json(out/"clinical_cohort.json")
    ids = m["study_ids"]
    rows = read_csv(out/"canonical_pairs_205.csv")
    result = {}
    for name in RUNS:
        by_id = {r["study_id"]:r for r in rows if r["pipeline"]==name}
        result[name] = [by_id[sid] for sid in ids]
    return ids, result


def joined(row, prefix):
    text = "\n".join(row[prefix+x].strip() for x in ["findings","impression"] if row[prefix+x].strip())
    assert text
    return text


def chexbert(out, smoke=False):
    hardware = require_cuda()
    ids, batches = clinical_batches(out)
    if smoke:
        ids = ids[:2]
        batches = {k:v[:2] for k,v in batches.items()}
    dest = out/("smoke_chexbert" if smoke else "chexbert_raw")
    if (dest/"labeled_reports.csv").exists():
        raise ValueError("CheXbert output already exists; don't overwrite raw inference.")
    reports = []
    for name in ["reference", *RUNS]:
        rows = batches["structured_rule"] if name=="reference" else batches[name]
        prefix = "reference_" if name=="reference" else "generated_"
        reports.extend({"pipeline":name,"study_id":sid,"Report Impression":joined(row,prefix)} for sid,row in zip(ids,rows,strict=True))
    save_csv(dest/"reports.csv",reports)
    sys.path.insert(0,str(ROOT/"third_party/CheXbert/src"))
    import label as official
    assert official.CONDITIONS == LABELS14
    official.load_unlabeled_data = partial(official.load_unlabeled_data,batch_size=4,num_workers=0)
    official.BATCH_SIZE = 4
    factory = official.bert_labeler
    calls = []
    def make_model():
        model = factory()
        def check_gpu(module, inputs):
            assert next(module.parameters()).is_cuda and inputs[0].is_cuda
            calls.append(str(inputs[0].device))
        model.register_forward_pre_hook(check_gpu)
        return model
    official.bert_labeler = make_model
    checkpoint = ROOT/"models/chexbert/chexbert.pth"
    preds = official.label(str(checkpoint),str(dest/"reports.csv"))
    official.save_preds(preds,str(dest/"reports.csv"),str(dest))
    assert calls and set(calls)=={"cuda:0"}
    labeled = pd.read_csv(dest/"labeled_reports.csv")
    assert labeled["Report Impression"].tolist() == [r["Report Impression"] for r in reports]
    save_json(dest/"inference_manifest.json",{"hardware":hardware,"forward_batches":len(calls),"all_model_forwards_cuda":True,"row_count":len(reports),"checkpoint":str(checkpoint),"checkpoint_sha256":sha256(checkpoint),"batch_size":4,"num_workers":0,"report_input_sha256":sha256(dest/"reports.csv"),"output_sha256":sha256(dest/"labeled_reports.csv"),"class_mapping":{"0":"blank/NaN","1":"positive/1","2":"negative/0","3":"uncertain/-1"}})
    print(f"CHEXBERT DONE: {len(reports)} reports, all {len(calls)} forwards on CUDA",flush=True)


def radgraph(out, smoke=False):
    hardware = require_cuda()
    ids, batches = clinical_batches(out)
    if smoke:
        ids = ids[:2]
        batches = {k:v[:2] for k,v in batches.items()}
    dest = out/("smoke_radgraph" if smoke else "radgraph_raw")
    dest.mkdir(parents=True,exist_ok=True)
    refs = [joined(r,"reference_") for r in batches["structured_rule"]]
    hyps = {name:[joined(r,"generated_") for r in rows] for name,rows in batches.items()}
    unique = list(dict.fromkeys(refs + [text for name in RUNS for text in hyps[name]]))
    save_json(dest/"input.json",{"study_ids":ids,"references":refs,"hypotheses":hyps,"unique_texts":unique})
    from radgraph import F1RadGraph
    from radgraph.rewards import compute_reward
    cache = ROOT/"scratch/radgraph_cache"
    scorer = F1RadGraph(reward_level="all",model_type="radgraph-xl",cuda=0,model_cache_dir=str(cache))
    assert scorer.radgraph.device.type == "cuda" and next(scorer.parameters()).is_cuda
    calls = []
    def check_gpu(module,inputs):
        assert next(module.parameters()).is_cuda
        calls.append("cuda:0")
    scorer.radgraph.model.register_forward_pre_hook(check_gpu)
    partial_path = dest/"annotations.json"
    cache_payload = read_json(partial_path) if partial_path.exists() else {"input_sha256":sha256(dest/"input.json"),"annotations":{}}
    assert cache_payload["input_sha256"] == sha256(dest/"input.json")
    annotations = cache_payload["annotations"]
    remaining = [text for text in unique if text not in annotations]
    with torch.inference_mode():
        for start in range(0,len(remaining),20):
            texts = remaining[start:start+20]
            raw = scorer.radgraph(texts)
            assert len(raw)==len(texts)
            annotations.update({text:raw[str(i)] for i,text in enumerate(texts)})
            save_json(partial_path,cache_payload)
            print(f"RADGRAPH CUDA: {len(annotations)}/{len(unique)} unique current-run texts",flush=True)
        # Verify deduplicated inference against the public pair-scoring API.
        mean, scores, _, _ = scorer(refs=refs[:2],hyps=hyps["structured_rule"][:2])
        public = np.asarray(scores,dtype=float).T
        direct = np.asarray([compute_reward(annotations[h],annotations[r],"all") for h,r in zip(hyps["structured_rule"][:2],refs[:2],strict=True)])
        np.testing.assert_allclose(public,direct,rtol=0,atol=1e-12)
    result, per_study = {}, []
    for name in RUNS:
        scores = [[float(v) for v in compute_reward(annotations[h],annotations[r],"all")] for h,r in zip(hyps[name],refs,strict=True)]
        assert np.asarray(scores).shape==(len(ids),3) and np.isfinite(scores).all()
        result[name] = {"study_ids":ids,"scores":scores}
        per_study.extend({"pipeline":name,"study_id":sid,**dict(zip(T6[2:],score))} for sid,score in zip(ids,scores,strict=True))
    save_json(dest/"scores.json",result)
    save_csv(dest/"per_study.csv",per_study)
    save_json(dest/"inference_manifest.json",{"hardware":hardware,"model_type":"radgraph-xl","forward_calls_this_process":len(calls),"all_model_forwards_cuda":True,"unique_texts":len(unique),"deduplication":"exact current-run text only; reference inferred once, shared across four branches; no historical annotations","public_api_equivalence_check":"two pairs matched at atol=1e-12; official returns metric-by-study, transposed to study-by-metric","model_cache_dir":str(cache),"weights_sha256":sha256(cache/"radgraph-xl/weights.th"),"config_sha256":sha256(cache/"radgraph-xl/config.json"),"input_sha256":sha256(dest/"input.json"),"output_sha256":sha256(dest/"scores.json")})
    print("RADGRAPH DONE: CUDA verified, public API equivalence passed",flush=True)


def summarize(out):
    require_cuda()
    ids,batches = clinical_batches(out)
    n = len(ids)
    report_index = read_csv(out/"chexbert_raw/reports.csv")
    labeled = pd.read_csv(out/"chexbert_raw/labeled_reports.csv")
    assert len(labeled)==len(report_index)==5*n
    assert labeled["Report Impression"].tolist()==[r["Report Impression"] for r in report_index]
    refs = labeled.iloc[:n].reset_index(drop=True)
    rg = read_json(out/"radgraph_raw/scores.json")
    values, per_study, aggregates = {}, [], {}
    for j,name in enumerate(RUNS):
        pred = labeled.iloc[(j+1)*n:(j+2)*n].reset_index(drop=True)
        ix = report_index[(j+1)*n:(j+2)*n]
        assert [r["study_id"] for r in ix]==ids and {r["pipeline"] for r in ix}=={name}
        assert rg[name]["study_ids"]==ids
        counts = []
        aggregates[name] = {}
        for labels,title in [(LABELS14,"all"),(LABELS6,"headline")]:
            ref,prediction,mask = _decisions(refs,pred,labels)
            tp = (ref & prediction & mask).sum(1)
            fp = (~ref & prediction & mask).sum(1)
            fn = (ref & ~prediction & mask).sum(1)
            counts.extend([tp,fp,fn])
            aggregates[name][title] = compute_chexbert_aggregate(refs,pred,labels)
        data = np.column_stack([*counts,np.asarray(rg[name]["scores"])])
        values[name] = data.tolist()
        for i,sid in enumerate(ids):
            per_study.append({"pipeline":name,"study_id":sid,**dict(zip(["chexbert_all_tp","chexbert_all_fp","chexbert_all_fn","chexbert_headline_tp","chexbert_headline_fp","chexbert_headline_fn",*T6[2:]],data[i].tolist()))})
    save_csv(out/"table6_per_study.csv",per_study)
    mask14 = (refs[LABELS14].to_numpy(float)!=-1)
    mask6 = (refs[LABELS6].to_numpy(float)!=-1)
    save_json(out/"chexbert_policy_audit.json",{"study_count":n,"labels14":LABELS14,"labels6":LABELS6,"valid_reference_decisions_14":int(mask14.sum()),"valid_reference_decisions_6":int(mask6.sum()),"reference_uncertain_14":int((~mask14).sum()),"reference_uncertain_6":int((~mask6).sum()),"reference_blank_14":int(refs[LABELS14].isna().to_numpy().sum()),"reference_blank_6":int(refs[LABELS6].isna().to_numpy().sum()),"mask_14_sha256":hashlib.sha256(mask14.tobytes()).hexdigest(),"mask_6_sha256":hashlib.sha256(mask6.tobytes()).hexdigest(),"policy":"Reference-only shared mask; exclude ref=-1; blank ref and blank/uncertain prediction are not positive","aggregates":aggregates})
    result = paired_bootstrap(values,T6,ids,out,"table6")
    for name in RUNS:
        assert abs(result["absolute"][name][T6[0]]["estimate"]-aggregates[name]["all"]["micro_f1"])<1e-12
        assert abs(result["absolute"][name][T6[1]]["estimate"]-aggregates[name]["headline"]["micro_f1"])<1e-12
    packages = {x:importlib.metadata.version(x) for x in ["torch","transformers","radgraph","numpy","pandas"]}
    files = [Path(__file__),ROOT/"scripts/evaluate_llm_outputs.py",ROOT/"src/chexbert_metrics.py",ROOT/"scripts/bootstrap_end_to_end_target_metrics.py",ROOT/"scripts/prepare_clinical_text_pairs.py",ROOT/"scripts/validate_common_studies.py",ROOT/"third_party/CheXbert/src/label.py",ROOT/"third_party/CheXbert/src/bert_tokenizer.py",ROOT/".venv-bertscore/Lib/site-packages/radgraph/radgraph.py",ROOT/".venv-bertscore/Lib/site-packages/radgraph/rewards.py"]
    save_json(out/"evaluator_versions.json",{"packages":packages,"python":sys.version,"executable":sys.executable,"source_sha256":{str(p):sha256(p) for p in files},"offline":True,"gpu":require_cuda()})
    print(json.dumps({"Table6":{k:{m:v["estimate"] for m,v in ms.items()} for k,ms in result["absolute"].items()},"valid_decisions_14":int(mask14.sum()),"valid_decisions_6":int(mask6.sum())},indent=2),flush=True)


def self_check():
    require_cuda()
    ontology = load_ontology()
    norm = lambda x:normalize_target_problems(x,ontology)
    both_empty = target_problem_metrics(norm(["normal"]),norm(["Cicatrix"]))
    assert both_empty["f1"]==1 and not both_empty["normal_decision"]["gold_normal"]
    mixed = target_problem_metrics(norm(["Cardiomegaly"]),norm(["Cardiomegaly","Pleural Effusion"]))
    assert (mixed["tp"],mixed["fp"],mixed["fn"])==(1,0,1)
    r = pd.DataFrame({"x":[1.,1.,-1.,np.nan,0.]})
    p = pd.DataFrame({"x":[np.nan,-1.,1.,1.,0.]})
    a = compute_chexbert_aggregate(r,p,["x"])
    assert (a["tp"],a["fp"],a["fn"],a["evaluated_label_decisions"])==(0,1,2,4)
    np.testing.assert_allclose(torch.quantile(tensor([0,1,2,3]),tensor([.025,.975])).cpu().numpy(),np.quantile([0,1,2,3],[.025,.975]),atol=1e-12)
    print("SELF CHECK PASSED: ontology, both-empty target F1, separate Normal, CheXbert omissions/uncertainty/blank, CUDA quantiles",flush=True)


def finalize(out):
    require_cuda()
    manifest = read_json(out/"input_manifest.json")
    lambda_global = manifest.get("structured_lambda_global", manifest["branches"]["structured_rule"]["run_config"]["lambda_global"])
    t5,t6 = read_json(out/"table5_results.json"),read_json(out/"table6_results.json")
    policy = read_json(out/"chexbert_policy_audit.json")
    ids = t6["study_ids"]
    assert t5["study_count"]==206 and t6["study_count"]==205
    assert manifest["excluded_empty_reference_study_ids"]==["1615"]
    for raw,digest in manifest["source_files"].items():
        assert sha256(Path(raw))==digest, f"Source changed: {raw}"
    provenance = read_csv(out/"report_provenance.csv")
    for row in provenance:
        assert sha256(Path(row["summary_path"]))==row["summary_sha256"]
    for stage in ["chexbert_raw","radgraph_raw"]:
        assert read_json(out/stage/"inference_manifest.json")["all_model_forwards_cuda"]
    counts,label_rows = {},[]
    for name in RUNS:
        rows = read_csv(out/"per_study"/f"{name}_target_labels.csv")
        counts[name] = {key:sum(int(r[key]) for r in rows) for key in ["problem_tp","problem_fp","problem_fn"]}
        counts[name]["gold_normal_count"] = sum(r["gold_normal"]=="True" for r in rows)
        for label in LABELS6:
            tp=fp=fn=0
            for row in rows:
                pred=label in row["predicted_target_problems"].split(";")
                gold=label in row["gold_target_problems"].split(";")
                tp+=pred and gold
                fp+=pred and not gold
                fn+=not pred and gold
            label_rows.append({"pipeline":name,"label":label,"support":tp+fn,"tp":tp,"fp":fp,"fn":fn,"f1":2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.})
    save_json(out/"table5_counts.json",counts)
    save_csv(out/"table5_per_label.csv",label_rows)
    oldroot = ROOT/"outputs/comparison_4pipelines"
    old5 = {r["pipeline"]:r for r in read_csv(ROOT/"outputs/table_iv_target_label_206/table_iv_target_label_full.csv")}
    old6 = {r["pipeline"]:r for r in read_csv(oldroot/"clinical_final_table.csv")}
    old_names = {"baseline_a":"baseline_a_classifier_vision","baseline_b":"baseline_b_classifier_retriever_vision"}
    comparison=[]
    for table,new,old,metrics in [("Table5",t5,old5,T5),("Table6",t6,old6,T6)]:
        for name in RUNS:
            saved = old[old_names.get(name,name)] if table=="Table5" else old[name]
            for metric in metrics:
                before=float(saved[metric]); after=new["absolute"][name][metric]["estimate"]
                comparison.append({"table":table,"pipeline":name,"metric":metric,"old_reported":before,"new":after,"new_minus_old_descriptive_only":after-before,"old_study_count":206,"new_study_count":new["study_count"],"comparability":"same target policy/cohort, different generated reports" if table=="Table5" else "not a controlled performance delta: different reports/cohort; historical CheXbert cache and tokenizer differ"})
    save_csv(out/"old_vs_new.csv",comparison)
    old_manifest=read_json(oldroot/"clinical_metrics_manifest.json")
    old_ids=old_manifest["study_ids"]
    take=[old_ids.index(sid) for sid in ids]
    old_ref=pd.read_csv(oldroot/"chexbert_raw/reference/labeled_reports.csv").iloc[take].reset_index(drop=True)
    new_ref=pd.read_csv(out/"chexbert_raw/labeled_reports.csv").iloc[:len(ids)]
    assert old_ref["Report Impression"].tolist()==new_ref["Report Impression"].tolist()
    a,b=old_ref[LABELS14].to_numpy(),new_ref[LABELS14].to_numpy()
    same=(a==b)|(np.isnan(a)&np.isnan(b))
    old_pairs=read_csv(oldroot/"canonical_pairs.csv")
    override=[r for r in old_pairs if r["study_id"]=="1615"]
    diagnostic={"historical_source":str(oldroot),"common_nonempty_reference_studies":len(ids),"reference_text_identical":True,"historical_vs_current_reference_label_differing_cells":int((~same).sum()),"historical_vs_current_reference_label_differing_studies":int((~same).any(1).sum()),"historical_reference_valid_decisions_14":int((a!=-1).sum()),"current_reference_valid_decisions_14":policy["valid_reference_decisions_14"],"historical_tokenizer_script_sha256":old_manifest["chexbert"]["tokenizer_script_sha256"],"current_tokenizer_script_sha256":sha256(ROOT/"third_party/CheXbert/src/bert_tokenizer.py"),"historical_checkpoint_sha256":old_manifest["chexbert"]["checkpoint_sha256"],"current_checkpoint_sha256":read_json(out/"chexbert_raw/inference_manifest.json")["checkpoint_sha256"],"historical_study_1615_override":[{"pipeline":r["pipeline"],"findings":r["reference_findings"],"impression":r["reference_impression"]} for r in override],"interpretation":"Historical label cache is not interchangeable with the current evaluator. Tokenizer script hashes differ; the exact historical inference stack is unavailable. No causal attribution or common-policy old/new quality gain is inferred."}
    save_json(out/"historical_comparability_audit.json",diagnostic)
    def delta(table,comparison,metric):
        r=next(r for r in table["paired_deltas"] if r["comparison"]==comparison and r["metric"]==metric)
        return f"{r['delta']:+.4f} [{r['ci95_lower']:+.4f}, {r['ci95_upper']:+.4f}]; CI chứa 0: {'có' if r['ci_contains_zero'] else 'không'}"
    lr="structured_learned_gate_minus_structured_rule"
    lines=["# Kết quả tính lại Table 5 và Table 6 — 31/10/2026", "", "Đã tính lại từ đúng 824 report đã lưu của bốn run latency ngày 29–30/10/2026, không gọi LLM. Mọi forward CheXbert/RadGraph và phép tính bootstrap chính chạy trên CUDA:0 (RTX 4050 Laptop GPU). File gốc không đổi.", "", "## Cohort và provenance", "", "- Table 5: 206 study duy nhất cho mỗi nhánh, cùng Problems tham chiếu, 824 summary và 824 response ID duy nhất.", "- Table 6: 205 study có Findings hoặc Impression tham chiếu, chung cho cả bốn nhánh. Study 1615 có cả hai trường rỗng nên bị loại; không sinh hoặc thay reference bằng normal. Không có điểm primary N=206.", "- Chọn đúng đường dẫn summary và response ID trong latency_per_case.csv; mỗi ca chỉ có một summary trong thư mục run. Không lấy report mới nhất từ các run khác.", "- Manifest, report nguồn và reference được hash SHA-256 và kiểm tra lại sau khi chạy. Chi tiết: report_provenance.csv, input_manifest.json, clinical_cohort.json.", "", "## Quy ước độ đo", "", "Table 5 tái sử dụng target_problem_metrics trong scripts/evaluate_llm_outputs.py và ontology cxr-target-6-v1. Sáu nhãn theo thứ tự: "+", ".join(LABELS6)+". Normal là quyết định nhị phân riêng, không phải pathology thứ bảy; gold normal cần nhãn normal tường minh và không có target pathology. Prediction không có target pathology được xem là predicted normal. Sample F1 lấy trung bình 206 F1 từng ca; cả hai tập target rỗng có F1=1, kể cả reference có quan sát ngoài ontology. Groundedness là TP/(TP+FP) gộp, bằng micro precision trên run này, không phải trung bình precision từng ca và không phải độ đo factuality từng claim.", "", f"Table 6 ghép Findings rồi Impression bằng newline. CheXbert positive-class micro F1 trên 14 nhãn và sáu nhãn mục tiêu: mask chỉ phụ thuộc reference, loại uncertain reference (-1), reference trống và prediction trống/uncertain không được xem là positive. Có {policy['valid_reference_decisions_14']} / {205*14} decision hợp lệ trên 14 nhãn và {policy['valid_reference_decisions_6']} / {205*6} trên sáu nhãn; loại {policy['reference_uncertain_14']} / {policy['reference_uncertain_6']} decision uncertain tương ứng. Cả bốn nhánh dùng đúng một tập reference labels/mask.", "", "CheXbert dùng checkpoint local chính thức, batch size 4; toàn bộ 1.025 report (205 reference + 4×205 prediction) được label mới. RadGraph dùng radgraph-xl 0.1.18 với cuda=0; official simple/partial/complete rewards được trình bày theo tên entity/relation/complete của project. Suy luận 882 văn bản duy nhất của run hiện tại; dùng chung annotation reference và chỉ khử trùng văn bản giống hệt trong run này. Hai cặp đã đối chiếu khớp public F1RadGraph API ở atol=1e-12. Không tái dùng cache clinical lịch sử.", "", "Bootstrap ghép cặp theo study: 2.000 resample, seed 20260904, CI percentile 95%. Mọi nhánh dùng cùng ma trận chỉ số trong từng bảng; Table 5 lấy mẫu từ 206 ca, Table 6 từ 205 ca. Sáu cặp so sánh được lưu đầy đủ; chiều delta ghi tường minh left_minus_right. CI chưa điều chỉnh đa so sánh.", "", "## Table 5", ""]
    lines += (out/"table5.md").read_text(encoding="utf-8").splitlines()[4:10]
    lines += ["", "TP/FP/FN: "+"; ".join(f"{name}={r['problem_tp']}/{r['problem_fp']}/{r['problem_fn']}" for name,r in counts.items())+". Mỗi nhánh có 30 gold-normal.", "", "## Table 6", ""]
    lines += (out/"table6.md").read_text(encoding="utf-8").splitlines()[4:10]
    lines += ["", "## So sánh ghép cặp cần cập nhật trong paper", ""]
    for metric in ["precision","recall","micro_f1","sample_f1"]:
        lines.append(f"- Learned − rule, {metric}: "+delta(t5,lr,metric))
    for metric in T6:
        lines.append(f"- Learned − rule, {metric}: "+delta(t6,lr,metric))
    lines += ["", "Learned có micro/sample F1 Table 5 cao nhất theo point estimate, nhưng CI của cả hai chênh lệch learned−rule chứa 0. Precision tăng đi kèm recall giảm. A có CheXbert 14 nhãn và ba điểm RadGraph cao nhất theo point estimate; learned có CheXbert sáu nhãn cao nhất theo point estimate. Chênh lệch CheXbert learned−rule đều có CI chứa 0; cả ba chênh lệch RadGraph learned−rule âm với CI không chứa 0. Không kết luận một nhánh vượt trội đồng đều.", "", "## Đối chiếu số cũ–mới và phần chưa khôi phục", "", "old_vs_new.csv lưu từng số cũ, số mới và chênh lệch mô tả. Các số cũ được lấy từ outputs/table_iv_target_label_206 và outputs/comparison_4pipelines, khớp bản paper Source còn có trong workspace. Đó là report lịch sử khác, không phải kết quả của bốn run latency hiện tại.", "", "Các điểm CheXbert cũ khoảng 0,89–0,91 không được trộn với positive-class F1 hiện tại khoảng 0,59–0,61. Trên 205 reference có cùng văn bản, cache reference lịch sử khác kết quả mới ở 613 decision/199 study; mask lịch sử có 2.823 decision hợp lệ so với 2.784 hiện tại. SHA tokenizer script lịch sử và hiện tại khác nhau, dù SHA checkpoint giống nhau. Chưa khôi phục được toàn bộ stack lịch sử nên không suy ra nguyên nhân chính xác hoặc diễn giải chênh lệch này là cải thiện/suy giảm chất lượng. Cache lịch sử còn dùng câu thay thế ‘No acute cardiopulmonary abnormality.’ cho study 1615; primary mới không dùng quy ước đó. Chi tiết trong historical_comparability_audit.json.", "", "Hai đường dẫn sau trong handoff không tồn tại ở workspace hiện tại:", ""]
    lines += ["- `"+p+"`" for p in manifest["missing_handoff_paths"]]
    lines += ["", "Do đó chưa thể đối chiếu nguyên văn hoặc sửa đúng bản paper TABLE6_ROUTE_UPDATE_20261027. Bản Source hiện có chỉ dùng làm đối chứng số đã lưu. Đề xuất nội dung cập nhật, abstract và kết luận nằm trong PAPER_UPDATE_PROPOSAL.md; các file paper gốc không bị sửa.", "", "## Tái chạy", "", "Dùng .venv-bertscore/Scripts/python.exe scripts/recompute_latency_tables.py với thư mục output mới, lần lượt các stage self-check, prepare, chexbert --smoke, radgraph --smoke, chexbert, radgraph, summarize, finalize. Nếu CUDA không có, script dừng, không fallback CPU. RadGraph có thể resume annotation của chính run này khi input hash khớp. Chạy inference tuần tự để tránh tranh VRAM.", "", "Kiểm tra: self-check ontology/Normal/blank/uncertain; smoke hai evaluator; hook xác nhận mọi model forward trên GPU; 824 provenance hash và toàn bộ source-file hash giữ nguyên; common-reference/mask; thứ tự score RadGraph và đối chiếu public API; 2.000 bootstrap indices được lưu để tái lập. Phiên bản, SHA evaluator/checkpoint/config nằm trong evaluator_versions.json và inference_manifest.json của từng evaluator."]
    (out/"REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    p=t5["absolute"]
    proposal=["# Đề xuất cập nhật paper sau khi tính lại", "", "Đích trong handoff chưa có trên đĩa; đây là nội dung review được, chưa áp dụng vào paper gốc.", "", "1. Thay toàn bộ Table 5 bằng table5.csv (206 study); Table 6 bằng table6.csv (205 study). Giữ 4 chữ số cho clinical metrics, 3 chữ số cho target metrics, groundedness nhân 100 nếu dùng phần trăm. Điền mọi CI từ hai results.json, không dùng CI lịch sử.", "2. Sửa methods: cùng model gpt-5.6-luna, reasoning none, output cap 160, nhưng A/B dùng prompt unstructured riêng và luôn gửi ảnh; rule/learned dùng compact evidence v5 và routing có điều kiện. Đây không phải thí nghiệm chỉ thay một thành phần prompt/gate. Route rule 197 vision/9 text, learned 153/53, A và B 206 vision. Giữ rõ giới hạn attribution.", "3. Sửa cohort: 206 ca cho target Problems; 205 ca cho clinical text sau khi loại study 1615 có cả hai reference rỗng. Ghi mask positive-class, decision counts 2784/1171, thứ tự sáu nhãn và checkpoint/version đã hash.", "4. Xóa/viết lại các nhận định lịch sử ‘B cao nhất micro/sample F1’, ‘learned cao nhất CheXbert 14 nhãn’ và ‘B cao nhất RadGraph’. Không diễn giải mức giảm từ CheXbert lịch sử ~0.9 như giảm chất lượng vì evaluator/cache/cohort không so sánh trực tiếp.", "", "## Abstract — đoạn kết quả đề xuất (English)", "", f"On 206 studies, the learned gate achieved target-label precision/recall/micro F1/sample F1 of {p['structured_learned_gate']['precision']['estimate']:.3f}/{p['structured_learned_gate']['recall']['estimate']:.3f}/{p['structured_learned_gate']['micro_f1']['estimate']:.3f}/{p['structured_learned_gate']['sample_f1']['estimate']:.3f}, compared with {p['structured_rule']['precision']['estimate']:.3f}/{p['structured_rule']['recall']['estimate']:.3f}/{p['structured_rule']['micro_f1']['estimate']:.3f}/{p['structured_rule']['sample_f1']['estimate']:.3f} for the rule gate. Learned-minus-rule precision increased by 0.040 (95% CI 0.011–0.071), while the micro-F1 difference remained unresolved (+0.005, 95% CI −0.024–0.033). On the common 205-study non-empty-reference cohort, Baseline A had the highest point estimates for 14-label CheXbert positive-class micro F1 (0.612) and RadGraph complete score (0.177), whereas the learned gate had the highest six-label CheXbert micro F1 (0.676). These results indicate complementary precision–recall and report-agreement trade-offs.", "", "## Kết quả / kết luận — nội dung cần thay", "", "Learned-minus-rule recall decreased by −0.037 (95% CI −0.075 to −0.004). The sample-F1 difference was +0.014 (95% CI −0.016 to +0.048), including zero. Learned-minus-rule CheXbert differences were +0.010 [−0.019,+0.037] on 14 labels and +0.011 [−0.024,+0.042] on the six target labels; both intervals included zero. RadGraph entity/relation/complete differences were −0.042 [−0.060,−0.025], −0.037 [−0.053,−0.022], and −0.027 [−0.040,−0.014], respectively. Precision gains therefore do not establish uniform improvement in F1 or report-level agreement. These comparisons use different evidence/prompt/routing configurations and cannot isolate a causal gate effect.", "", "Giữ nguyên các phần classifier/retrieval không được đo lại. Kiểm tra mọi số end-to-end trong abstract, results, discussion, conclusion, bảng CI, hình so sánh, Normal F1 và bảng per-label; dữ liệu thay thế có trong table5_per_label.csv và table5_results.json. Không gán các kết quả chất lượng mới cho report lịch sử hoặc ngược lại."]
    (out/"PAPER_UPDATE_PROPOSAL.md").write_text("\n".join(proposal)+"\n",encoding="utf-8")
    if EXPECTED_EFFORT != "none":
        source_branches=manifest["branches"]
        current=["# Kết quả bốn nhánh chạy mới với reasoning effort high", "", f"Model gpt-5.6-luna; reasoning_effort={EXPECTED_EFFORT}; max_output_tokens={EXPECTED_MAX_TOKENS}; CUDA:0. Sinh lại từ ảnh với inference và API response mới, cache local tắt. Giữ nguyên prompt/evidence/routing riêng của từng nhánh. Các run lịch sử không bị ghi đè.", "", "Nguồn report:", ""]
        if lambda_global != 0.9:
            current[0] = f"# Kết quả bốn pipeline với λ={lambda_global:g} ở hai nhánh structured"
            current[4:4] = [f"Rule và Learned kết hợp {lambda_global:g} × điểm global RAD-DINO + {1 - lambda_global:g} × điểm local DenseNet ở bước rerank bằng chứng. Baseline A dùng classifier và ảnh. Baseline B giữ Top-5 theo điểm global và cung cấp điểm local dưới dạng thông tin mô tả.", ""]
        current += [f"- {name}: {RUNS[name]}; routes={source_branches[name]['routes']}" for name in RUNS]
        current += ["", "Table 5 dùng 206 study và Problems gốc với ontology cxr-target-6-v1. Groundedness là pooled TP/(TP+FP), Sample F1 là mean F1 từng study (both-empty target sets=1). Normal được đánh giá riêng, không thêm thành pathology thứ bảy.", "", "Table 6 dùng cùng 205 study có ít nhất một reference Findings/Impression; loại study 1615 có cả hai trường trống, không tạo reference thay thế. CheXbert positive-class micro F1 dùng mask chung chỉ dựa vào reference: bỏ reference uncertain; blank reference và blank/uncertain prediction không phải positive. Decision hợp lệ 14/6 nhãn: "+str(policy["valid_reference_decisions_14"])+"/"+str(policy["valid_reference_decisions_6"])+". Thứ tự sáu nhãn: "+", ".join(LABELS6)+".", "", "CheXbert/RadGraph được suy luận mới trên GPU; bootstrap bằng CUDA gồm 2.000 resample theo study, seed 20260904. Dùng chung indices giữa bốn nhánh trong từng bảng; chiều delta ghi rõ left_minus_right. RadGraph dùng radgraph-xl và official simple/partial/complete rewards (tên project: entity/relation/complete).", "", *(out/"table5.md").read_text(encoding="utf-8").splitlines(), "", *(out/"table6.md").read_text(encoding="utf-8").splitlines(), "", "## Paired differences", ""]
        for title,table,metrics in [("Table5",t5,T5),("Table6",t6,T6)]:
            for comparison in dict.fromkeys(r["comparison"] for r in table["paired_deltas"]):
                for metric in metrics:
                    current.append(f"- {title}, {comparison}, {metric}: "+delta(table,comparison,metric))
        interval_note = "95% CI là percentile nominal, chưa điều chỉnh đa so sánh."
        if lambda_global == 0.9:
            interval_note = "Trần output đã đổi từ 160 ở run none sang 8192 ở run high để chứa reasoning và report text, theo cùng cấu hình cho cả bốn nhánh. Không quy chênh lệch cũ–mới cho riêng effort. " + interval_note
        current += ["", "Đối chiếu lịch sử trong old_vs_new.csv chỉ là chênh lệch mô tả. Cache CheXbert lịch sử khác nhãn reference và SHA tokenizer dù cùng văn bản/checkpoint; không diễn giải chênh lệch từ các điểm cũ ~0.9 như tăng/giảm chất lượng. historical_comparability_audit.json ghi chi tiết.", "", interval_note, "", "Provenance/kiểm tra: input_manifest.json, report_provenance.csv, evaluator_versions.json, *_raw/inference_manifest.json, final_audit.json và code_snapshot/. Các model forward đều được hook kiểm tra CUDA; RadGraph đã đối chiếu public API; report và source hashes được kiểm tra lại sau khi chạy. Các file bootstrap_indices.npy lưu đúng ma trận resample.", "", "Bản paper TABLE6_ROUTE_UPDATE_20261027 và script CheXbert được handoff chỉ định không có trong workspace. Chưa sửa paper gốc; nội dung cập nhật có trong PAPER_UPDATE_PROPOSAL.md."]
        (out/"REPORT.md").write_text("\n".join(current)+"\n",encoding="utf-8")
        proposed=["# Đề xuất cập nhật paper — run high mới", "", "Dùng table5.csv (N=206) và table6.csv (N=205) thay toàn bộ số end-to-end và clinical metrics/CI lịch sử. Giữ cấu hình high/8192/CUDA, phiên bản prompt/routing và cohort đúng theo REPORT.md. Bản paper đích hiện không có trên đĩa; chưa áp dụng sửa đổi.", "", "## Results for abstract/conclusion", ""]
        if lambda_global != 0.9:
            proposed[0] = f"# Kết quả đề xuất cho paper ở λ={lambda_global:g}"
        for name in RUNS:
            values=[f"{metric}={scores['estimate']:.4f} [{scores['ci95_lower']:.4f}, {scores['ci95_upper']:.4f}]" for table in [t5,t6] for metric,scores in table["absolute"][name].items()]
            proposed.append(name+": "+"; ".join(values)+".")
        proposed += ["", "## Required paired claims", ""]
        for title,table,metrics in [("Table5",t5,T5),("Table6",t6,T6)]:
            proposed.extend(f"{title}, learned minus rule, {metric}: "+delta(table,lr,metric) for metric in metrics)
        proposed += ["", "Không dùng thứ hạng/CI từ run none hoặc historical cache trong abstract, discussion hay conclusion. Chỉ mô tả khác biệt được CI ghép cặp hỗ trợ; CI chứa 0 phải nói chưa rõ chiều chênh lệch. Các nhánh khác nhau về evidence, prompt và image routing; không kết luận tác động nhân quả riêng của gate. Sáu cặp đầy đủ có trong các paired_deltas.csv. Kiểm tra thêm Normal F1, hình và per-label table theo dữ liệu mới."]
        (out/"PAPER_UPDATE_PROPOSAL.md").write_text("\n".join(proposed)+"\n",encoding="utf-8")
    import shutil
    version=read_json(out/"evaluator_versions.json")
    version["source_sha256"][str(Path(__file__))]=sha256(Path(__file__))
    version["huggingface_cached_refs"]={name:(Path.home()/".cache/huggingface/hub"/name/"refs/main").read_text().strip() for name in ["models--bert-base-uncased","models--microsoft--BiomedVLP-CXR-BERT-general"]}
    save_json(out/"evaluator_versions.json",version)
    for raw,digest in version["source_sha256"].items():
        path=Path(raw)
        dest=out/"code_snapshot"/path.relative_to(ROOT)
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,dest)
        assert sha256(dest)==digest
    save_json(out/"final_audit.json",{"passed":True,"structured_lambda_global":lambda_global,"source_reports":824,"unique_response_ids":len({r['response_id'] for r in provenance}),"all_source_hashes_unchanged":True,"all_model_forwards_cuda":True,"bootstrap_arithmetic_device":"cuda:0","table5_studies":206,"table6_studies":205,"table6_excluded_study_ids":["1615"],"chexbert_valid_decisions14":2784,"chexbert_valid_decisions6":1171,"bootstrap_samples":2000,"seed":20260904,"paired_comparisons_per_table":len(PAIRS),"unavailable":["Primary Table6 N=206: source reference blank for study 1615","Exact handoff paper revision and its CheXbert script are absent; update proposal saved instead"],"source_runs_overwritten":False})
    save_json(out/"output_checksums.json",{str(p.relative_to(out)):sha256(p) for p in sorted(out.rglob("*")) if p.is_file() and p.name!="output_checksums.json"})
    print(f"FINAL AUDIT PASSED: {out}",flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=["prepare","chexbert","radgraph","summarize","finalize","self-check"])
    parser.add_argument("--output-dir",type=Path,default=ROOT/"outputs/recompute_table5_table6_20261031")
    parser.add_argument("--smoke",action="store_true")
    parser.add_argument("--runs-root",type=Path,help="Root containing structured_rule, structured_learned_gate, and baselines/baseline_a,b.")
    parser.add_argument("--expected-reasoning-effort",default="none")
    parser.add_argument("--expected-max-output-tokens",type=int,default=160)
    parser.add_argument("--expected-lambda-global",type=float,help="Expected fusion weight in both structured runs.")
    args=parser.parse_args()
    EXPECTED_EFFORT, EXPECTED_MAX_TOKENS = args.expected_reasoning_effort, args.expected_max_output_tokens
    if args.runs_root:
        RUNS={name:str(args.runs_root.resolve()/relative) for name,relative in {"structured_rule":"structured_rule","structured_learned_gate":"structured_learned_gate","baseline_a":"baselines/baseline_a","baseline_b":"baselines/baseline_b"}.items()}
    args.output_dir.mkdir(parents=True,exist_ok=True)
    if args.stage=="self-check": self_check()
    elif args.stage=="prepare": prepare(args.output_dir,args.expected_lambda_global)
    elif args.stage in ["chexbert","radgraph"]: globals()[args.stage](args.output_dir,args.smoke)
    else: globals()[args.stage](args.output_dir)
