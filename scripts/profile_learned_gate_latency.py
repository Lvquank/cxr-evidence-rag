"""Measure fresh, sequential image-to-report latency for learned or rule gating.

Each case starts new classifier/retriever and LLM processes, like the original
batch runners. Model loading, process startup, API retries and output writes are
included. Gate training and offline evaluation are not inference latency.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import torch
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_nested_gate_threshold_oof import NUMERIC_FEATURES, PATHOLOGIES, make_feature_matrix
from scripts.run_terra_agreement_vision_batch import classifier_retriever_command, query_image_path
from src.prompts import validate_radiology_prediction

GATE_ROOT = ROOT / "src/learned_gate_oof_206"
TIMINGS = ("classifier_retriever_s", "learned_gate_s", "llm_pipeline_s", "end_to_end_s")


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def feature_rows(evidence: dict) -> pd.DataFrame:
    by_label = {row["label"]: row for row in evidence["retriever"]["label_evidence"]}
    rows = []
    for label in PATHOLOGIES:
        item = by_label.get(label, {})
        support = item.get("supporting_evidence", [])
        contrast = item.get("matched_non_support_evidence", [])
        rows.append({
            "pathology": label,
            "raw_prob": evidence["classifier"]["raw_probabilities"][label],
            "m5": item.get("margins", {}).get("M@5", 0.0),
            "support_mean": float(np.mean([x.get("final_rank_score", x["cosine_similarity"]) for x in support])) if support else 0.0,
            "non_support_mean": float(np.mean([x.get("final_rank_score", x["cosine_similarity"]) for x in contrast])) if contrast else 0.0,
            "support_count": len(support),
            "non_support_count": len(contrast),
        })
    return pd.DataFrame(rows)


def apply_gate(evidence: dict, artifact: dict, study_id: str) -> dict:
    if study_id in artifact["train_study_ids"] or study_id not in artifact["held_out_study_ids"]:
        raise ValueError(f"Study {study_id} is not held out of fold {artifact['fold']}.")
    expected_features = list(NUMERIC_FEATURES) + [f"pathology={label}" for label in PATHOLOGIES]
    if artifact["feature_names"] != expected_features or artifact["threshold"] != 0.5:
        raise ValueError("Gate artifact differs from the frozen feature order/threshold.")
    result = copy.deepcopy(evidence)
    probabilities = artifact["pipeline"].predict_proba(make_feature_matrix(feature_rows(evidence)))[:, 1]
    rule_labels = list(evidence["classifier"]["selected_labels"])
    by_label = {row["label"]: row for row in evidence["retriever"]["label_evidence"]}
    decisions, emitted = [], []
    for label, probability in zip(PATHOLOGIES, probabilities):
        decision = {
            "pathology": label, "method": artifact["method"], "fold": artifact["fold"],
            "probability": float(probability), "threshold": 0.5,
            "action": "EMIT" if probability >= 0.5 else "SUPPRESS",
            "gold_excluded_from_inference": True,
            "rule_action": "EMIT" if label in rule_labels else "SUPPRESS",
        }
        decisions.append(decision)
        if probability < 0.5:
            continue
        item = copy.deepcopy(by_label.get(label, {
            "label": label, "classifier_probability": evidence["classifier"]["raw_probabilities"][label],
            "evidence_status": "INSUFFICIENT", "evidence_available": False,
            "margins": {}, "quality_checks": {}, "supporting_evidence": [], "matched_non_support_evidence": [],
        }))
        item["learned_gate"] = {k: v for k, v in decision.items() if k not in ("pathology", "rule_action")}
        emitted.append(item)
    emitted.sort(key=lambda item: -float(item["classifier_probability"]))
    selected = [item["label"] for item in emitted]
    result["classifier"].update(rule_selected_labels=rule_labels, selected_labels=selected)
    result["retriever"].update(label_evidence=emitted, rule_label_evidence_count=len(by_label))
    result["learned_gate"] = {
        "method": artifact["method"], "fold": artifact["fold"], "threshold": 0.5,
        "rule_selected_labels": rule_labels, "learned_selected_labels": selected, "decisions": decisions,
    }
    result["study_id"] = study_id
    return result


def run_logged(command: list[str], path: Path) -> float:
    started = time.perf_counter()
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                               encoding="utf-8", errors="replace", stdin=subprocess.DEVNULL,
                               env={**os.environ, "PYTHONIOENCODING": "utf-8", "HF_HUB_OFFLINE": "1"})
    elapsed = time.perf_counter() - started
    path.write_text("COMMAND:\n" + subprocess.list2cmdline(command) + "\n\nSTDOUT:\n" + completed.stdout
                    + "\n\nSTDERR:\n" + completed.stderr, encoding="utf-8")
    if completed.returncode:
        raise RuntimeError(f"Process exited {completed.returncode}; see {path}")
    return elapsed


def describe(values: list[float]) -> dict:
    return {"count": len(values), "mean_s": float(np.mean(values)), "median_s": float(np.median(values)),
            "p95_s": float(np.percentile(values, 95)), "min_s": min(values), "max_s": max(values),
            "total_s": sum(values)} if values else {"count": 0}


def save_results(root: Path, rows: list[dict], config: dict, elapsed: float, complete: bool) -> dict:
    good = [row for row in rows if row["status"] == "success"]
    fields = list(dict.fromkeys(key for row in rows for key in row))
    if fields:
        with (root / "latency_per_case.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    summary = {**config, "updated_at": datetime.now(timezone.utc).isoformat(), "complete": complete,
               "succeeded": len(good), "failed": len(rows) - len(good), "batch_wall_s": elapsed,
               "latency": {key: describe([row[key] for row in good]) for key in config["timing_fields"]}, "results": rows}
    write_json(root / "latency_summary.json", summary)
    return summary


def resume_rows(root: Path, config: dict, resume: bool) -> tuple[list[dict], float]:
    if not root.exists() or not any(root.iterdir()):
        return [], 0.0
    if not resume:
        raise ValueError("Use a new empty output root, or --resume for this same experiment.")
    previous = json.loads((root / "latency_summary.json").read_text(encoding="utf-8"))
    for key in ("branch", "requested", "device", "model", "reasoning_effort", "max_output_tokens", "manifest_sha256"):
        if previous[key] != config[key]:
            raise ValueError(f"Resume configuration differs: {key}")
    if previous.get("lambda_global") != config.get("lambda_global"):
        raise ValueError("Resume configuration differs: lambda_global")
    rows = previous["results"]
    if rows and rows[-1]["status"] != "success":
        failed = rows.pop()
        request_id = failed["request_id"]
        archive = root / "failed_attempts" / request_id / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        write_json(archive / "attempt.json", failed)
        for folder in ("llm", "logs", "raw_evidence", "evidence"):
            parent = root / folder
            if not parent.is_dir():
                continue
            for path in parent.glob(request_id + "*"):
                if not path.resolve().is_relative_to(root.resolve()):
                    raise ValueError("Attempt path escaped the experiment root.")
                destination = archive / folder / path.name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(path), str(destination))
    if any(row["status"] != "success" for row in rows):
        raise ValueError("Resume rows are not a successful prefix.")
    return rows, float(previous["batch_wall_s"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/evaluations/leave_one_out_206_manifest.json")
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--reasoning-effort", choices=("none", "low", "medium", "high", "xhigh", "max"), default="high")
    parser.add_argument("--max-output-tokens", type=int, default=8192)
    parser.add_argument("--lambda-global", type=float, default=0.9, help="Global retrieval score weight; local weight is 1 minus this value.")
    parser.add_argument("--branch", choices=("learned", "rule"), default="learned")
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not 0.0 <= args.lambda_global <= 1.0:
        parser.error("--lambda-global must be between 0 and 1.")
    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        parser.error("CUDA is required; CPU fallback is disabled.")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if len(manifest["requests"]) != 206 or not 1 <= args.limit <= 206:
        parser.error("Expected the frozen 206-study manifest and --limit in [1, 206].")
    requests = manifest["requests"][:args.limit]
    folds = json.loads((GATE_ROOT / "folds.json").read_text(encoding="utf-8"))["study_to_fold"]
    for item in requests:
        query_image_path(item)
        if str(item["study_id"]) not in folds:
            raise ValueError(f"Missing OOF fold: {item['study_id']}")
    control_timing = "learned_gate_s" if args.branch == "learned" else "rule_evidence_io_s"
    config = {
        "branch": args.branch,
        "requested": args.limit, "device": args.device, "model": "gpt-5.6-luna",
        "reasoning_effort": args.reasoning_effort, "max_output_tokens": args.max_output_tokens, "lambda_global": args.lambda_global,
        "gate_threshold": 0.5 if args.branch == "learned" else None, "compact_schema": 5,
        "timing_fields": list(TIMINGS) if args.branch == "learned" else ["classifier_retriever_s", control_timing, "llm_pipeline_s", "end_to_end_s"],
        "manifest": str(args.manifest), "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "python": sys.version, "platform": platform.platform(), "sklearn": sklearn.__version__,
        "protocol": "sequential fresh processes; image read through saved validated report; process/model loading and API retries included",
        "cache_policy": "fresh image inference and API response; local LLM cache disabled; pretrained weights and fixed gallery reused; provider prompt caching may occur",
        "comparison_note": "Current v5 prompt/compact code; historical learned-gate reports used v4. Hardware and API load also differ.",
        "control_timing_note": ("OOF gate model loading, feature extraction, predict_proba and evidence writing"
                                if args.branch == "learned" else
                                "Rule threshold selection and retrieval-quality decisions are included in classifier_retriever_s; rule_evidence_io_s only reads/checks/writes evidence for the LLM"),
    }
    if args.preview:
        print(json.dumps(config, indent=2))
        return
    rows, previous_elapsed = resume_rows(args.output_root, config, args.resume)
    if [r["request_id"] for r in rows] != [r["request_id"] for r in requests[:len(rows)]]:
        parser.error("Resume studies do not match the requested manifest prefix.")
    for name in ("raw_evidence", "evidence", "llm", "logs"):
        (args.output_root / name).mkdir(parents=True, exist_ok=True)
    write_json(args.output_root / "manifest.json", manifest)
    write_json(args.output_root / "run_config.json", config)
    batch_started = time.perf_counter()
    for position, item in enumerate(requests[len(rows):], len(rows)+1):
        request_id, study_id = item["request_id"], str(item["study_id"])
        raw_path = args.output_root / "raw_evidence" / f"{request_id}.json"
        gate_path = args.output_root / "evidence" / f"{request_id}.json"
        llm_dir = args.output_root / "llm" / request_id
        row = {"request_id": request_id, "study_id": study_id, "status": "failed"}
        if args.branch == "learned":
            row["fold"] = folds[study_id]
        print(f"[{position}/{len(requests)}] {request_id}: starting fresh image inference", flush=True)
        case_started = time.perf_counter()
        try:
            row["classifier_retriever_s"] = run_logged(
                classifier_retriever_command(item, raw_path, args.device, args.lambda_global),
                args.output_root / "logs" / f"{request_id}_classifier_retriever.log")
            gate_started = time.perf_counter()
            raw = json.loads(raw_path.read_text(encoding="utf-8"))
            retriever = raw["retriever"]
            if retriever["excluded_study_ids"] != [study_id]:
                raise ValueError("Query study exclusion is missing.")
            has_selected_labels = bool(raw["classifier"]["selected_labels"])
            if retriever["lambda_global"] != args.lambda_global or retriever["local_rerank_enabled"] != has_selected_labels:
                raise ValueError("Evidence retrieval settings differ from the requested reranking configuration.")
            if args.branch == "learned":
                artifact = joblib.load(GATE_ROOT / "models" / f"fold_{folds[study_id]}.pkl")
                gated = apply_gate(raw, artifact, study_id)
            else:
                gated = raw
                gated["study_id"] = study_id
            gated["request_id"] = request_id
            write_json(gate_path, gated)
            row[control_timing] = time.perf_counter() - gate_started
            command = [sys.executable, str(ROOT / "scripts/run_full_pipeline.py"),
                       "--input", str(gate_path), "--output", str(llm_dir),
                       "--image-path", str(query_image_path(item)), "--study-id", study_id,
                       "--model", "gpt-5.6-luna", "--escalation-model", "gpt-5.6-luna",
                       "--reasoning-effort", args.reasoning_effort, "--max-output-tokens", str(args.max_output_tokens), "--no-cache"]
            row["llm_pipeline_s"] = run_logged(command, args.output_root / "logs" / f"{request_id}_llm.log")
            row["end_to_end_s"] = time.perf_counter() - case_started
            summaries = list(llm_dir.glob("summary_*.json"))
            if len(summaries) != 1:
                raise ValueError("Expected exactly one newly generated report.")
            report = json.loads(summaries[0].read_text(encoding="utf-8"))
            if report.get("reasoning_effort") != args.reasoning_effort:
                raise ValueError("Report reasoning effort differs from requested setting.")
            if validate_radiology_prediction(report["prediction"], tuple(gated["classifier"]["selected_labels"])) is None:
                raise ValueError("Generated report does not satisfy the selected-label output contract.")
            api = report["api"]
            if api.get("cached") or not api.get("response_id"):
                raise ValueError("Expected a fresh API response, not a cached/simulated report.")
            row.update(status="success", response_id=api["response_id"], route=report["route"]["name"],
                       image_attached=report["route"]["image_attached"], summary=str(summaries[0]))
            print(f"[{position}/{len(requests)}] OK {row['end_to_end_s']:.3f}s "
                  f"(image={row['classifier_retriever_s']:.3f}, control={row[control_timing]:.4f}, llm={row['llm_pipeline_s']:.3f})", flush=True)
        except Exception as error:
            row.update(error=str(error), failed_attempt_s=time.perf_counter() - case_started)
            print(f"[{position}/{len(requests)}] FAILED: {error}", flush=True)
        rows.append(row)
        save_results(args.output_root, rows, config, previous_elapsed + time.perf_counter() - batch_started, False)
        if row["status"] != "success":
            break
    summary = save_results(args.output_root, rows, config, previous_elapsed + time.perf_counter() - batch_started,
                           len(rows) == len(requests) and all(row["status"] == "success" for row in rows))
    print(json.dumps({key: summary[key] for key in ("complete", "succeeded", "failed", "latency")}, indent=2), flush=True)
    if not summary["complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
