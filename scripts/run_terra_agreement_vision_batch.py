#!/usr/bin/env python3
"""Full 206-study CXR-RAG experiment with configurable model and agreement routing.

Policy
------
* Text-only: every pathology selected by DenseNet has corresponding retriever
  evidence whose status is ``SUPPORTED``.
* Vision: every other case, including no classifier-selected pathology, uses
  the original query image plus the compact support/non-support evidence.

The script keeps outputs in a new root, snapshots the 206-study manifest and
writes an audit record for every route.  It evaluates target labels and text
metrics only after a clean batch by default.  It intentionally does not run
RadGraph: that evaluator requires its separately installed model/checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.prompts import validate_radiology_prediction

DEFAULT_MANIFEST = ROOT / "data" / "evaluations" / "leave_one_out_206_manifest.json"
DEFAULT_OUTPUT = ROOT / "outputs" / "luna_high_evidence_v5_206"


def is_classifier_retriever_agreement(evidence: dict[str, Any]) -> tuple[bool, str, list[str], dict[str, str]]:
    """Return true only for positive labels supported by the label retriever."""
    selected = [str(label) for label in evidence.get("classifier", {}).get("selected_labels", [])]
    by_label = {
        str(item.get("label")): str(item.get("evidence_status"))
        for item in evidence.get("retriever", {}).get("label_evidence", [])
        if isinstance(item, dict) and item.get("label")
    }
    if not selected:
        return False, "no_classifier_selected_label_has_no_label_specific_retrieval_confirmation", selected, by_label
    missing = [label for label in selected if label not in by_label]
    non_supported = [label for label in selected if by_label.get(label) != "SUPPORTED"]
    if missing:
        return False, "selected_label_missing_retriever_evidence", selected, by_label
    if non_supported:
        statuses = ",".join(f"{label}:{by_label.get(label)}" for label in non_supported)
        return False, f"retriever_not_supported({statuses})", selected, by_label
    return True, "all_classifier_selected_labels_supported_by_retriever", selected, by_label


def run_logged(command: list[str], log_path: Path) -> None:
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True,
                               stdin=subprocess.DEVNULL,
                               encoding="utf-8", errors="replace",
                               env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        "COMMAND:\n" + subprocess.list2cmdline(command) + "\n\nSTDOUT:\n" + completed.stdout + "\n\nSTDERR:\n" + completed.stderr,
        encoding="utf-8",
    )
    if completed.returncode:
        raise subprocess.CalledProcessError(completed.returncode, command)


def query_image_path(item: dict[str, Any]) -> Path:
    original = Path(item["image_path"])
    if original.is_file():
        return original
    relocated = ROOT / "data" / "reprocessed" / "images" / f"{item['image_id']}{original.suffix}"
    if relocated.is_file():
        return relocated
    raise FileNotFoundError(f"Query image unavailable for {item['request_id']}: {original}")


def classifier_retriever_command(item: dict[str, Any], evidence_path: Path, device: str, lambda_global: float) -> list[str]:
    return [
        sys.executable, str(ROOT / "scripts" / "test_classifier_to_retriever.py"),
        "--image_path", str(query_image_path(item)),
        "--output_json", str(evidence_path),
        "--exclude-study-id", str(item["study_id"]),
        "--device", device,
        "--lambda_global", str(lambda_global),
    ]


def llm_command(
    item: dict[str, Any], evidence_path: Path, llm_dir: Path, max_output_tokens: int, agreement: bool,
    model: str, reasoning_effort: str,
) -> list[str]:
    command = [
        sys.executable, str(ROOT / "scripts" / "run_full_pipeline.py"),
        "--input", str(evidence_path),
        "--output", str(llm_dir),
        "--image-path", str(query_image_path(item)),
        "--study-id", str(item["study_id"]),
        "--model", model,
        "--escalation-model", model,
        "--reasoning-effort", reasoning_effort,
        "--max-output-tokens", str(max_output_tokens),
        "--image-detail", "high",
        "--no-cache",
    ]
    command.append("--disable-vision-escalation" if agreement else "--force-vision")
    return command


def write_route_audit(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = ["request_id", "study_id", "case_label", "agreement", "route_policy", "route_reason", "selected_labels", "retriever_statuses", "status", "error"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                **row,
                "agreement": str(row.get("agreement", "")).lower(),
                "selected_labels": "; ".join(row.get("selected_labels", [])),
                "retriever_statuses": "; ".join(f"{label}:{state}" for label, state in sorted(row.get("retriever_statuses", {}).items())),
            })


def has_completed_llm_summary(llm_dir: Path, model: str, reasoning_effort: str) -> bool:
    for path in llm_dir.glob("summary_*.json"):
        try:
            summary = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if (
            summary.get("requested_model") == model
            and summary.get("model") == model
            and summary.get("reasoning_effort") == reasoning_effort
            and summary.get("evidence_schema") == 5
            and isinstance(summary.get("response"), str)
            and validate_radiology_prediction(summary.get("prediction")) is not None
            and summary.get("source")
        ):
            return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default="gpt-5.6-luna")
    parser.add_argument("--reasoning-effort", choices=["none", "low", "medium", "high", "xhigh", "max"], default="high")
    parser.add_argument("--limit", type=int, default=1, help="Use 206 for the complete frozen manifest; default 1 is a smoke test.")
    parser.add_argument("--device", default="cuda", help="Classifier/retriever CUDA device")
    parser.add_argument("--lambda-global", type=float, default=0.9)
    parser.add_argument("--max-output-tokens", type=int, default=25000)
    parser.add_argument("--preview", action="store_true", help="Show commands without running models or the API.")
    parser.add_argument("--resume", action="store_true", help="Reuse matching evidence and completed summaries in the output root.")
    parser.add_argument("--continue-on-error", action="store_true", help="Continue remaining requests after a failed case.")
    parser.add_argument("--skip-nlp-evaluation", action="store_true")
    parser.add_argument("--skip-target-bootstrap", action="store_true")
    parser.add_argument("--allow-partial-evaluation", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.limit <= 206:
        parser.error("--limit must be between 1 and 206.")
    if not 0.0 <= args.lambda_global <= 1.0:
        parser.error("--lambda-global must be between 0 and 1.")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    requests = list(manifest.get("requests", []))
    if len(requests) != 206:
        raise RuntimeError(f"Expected exactly 206 frozen requests, found {len(requests)}.")
    selected = requests[:args.limit]
    if args.preview:
        for item in selected:
            evidence = args.output_root / "evidence" / f"{item['request_id']}.json"
            print(json.dumps({"request_id": item["request_id"], "classifier_retriever": classifier_retriever_command(item, evidence, args.device, args.lambda_global), "model": args.model, "reasoning_effort": args.reasoning_effort, "llm_policy_after_evidence": "text_only_if_all_selected_labels_are_SUPPORTED_else_force_vision"}, ensure_ascii=False))
        return
    if args.output_root.exists() and any(args.output_root.iterdir()) and not args.resume:
        parser.error("--output-root must be empty for a new batch; use --resume only for this batch.")

    evidence_dir, llm_root, logs_dir = (args.output_root / name for name in ("evidence", "llm", "logs"))
    for directory in (evidence_dir, llm_root, logs_dir):
        directory.mkdir(parents=True, exist_ok=True)
    (args.output_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    route_rows: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    for position, item in enumerate(selected, start=1):
        request_id = str(item["request_id"])
        evidence_path = evidence_dir / f"{request_id}.json"
        llm_dir = llm_root / request_id
        print(f"[{position}/{len(selected)}] {request_id}: study={item['study_id']}", flush=True)
        audit: dict[str, Any] = {"request_id": request_id, "study_id": str(item["study_id"]), "case_label": str(item["case_label"]), "status": "failed"}
        try:
            if not (args.resume and evidence_path.is_file()):
                run_logged(classifier_retriever_command(item, evidence_path, args.device, args.lambda_global), logs_dir / f"{request_id}_classifier_retriever.log")
            evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
            agreement, reason, labels, statuses = is_classifier_retriever_agreement(evidence)
            audit.update({"agreement": agreement, "route_policy": "text_only" if agreement else "vision", "route_reason": reason, "selected_labels": labels, "retriever_statuses": statuses})
            if args.resume and has_completed_llm_summary(llm_dir, args.model, args.reasoning_effort):
                audit["status"] = "success"
                results.append({"request_id": request_id, "study_id": str(item["study_id"]), "case_label": str(item["case_label"]), "status": "success", "resumed": True})
                route_rows.append(audit)
                write_route_audit(args.output_root / "route_audit.csv", route_rows)
                continue
            run_logged(llm_command(item, evidence_path, llm_dir, args.max_output_tokens, agreement, args.model, args.reasoning_effort), logs_dir / f"{request_id}_llm.log")
            audit["status"] = "success"
            results.append({"request_id": request_id, "study_id": str(item["study_id"]), "case_label": str(item["case_label"]), "status": "success"})
        except (subprocess.CalledProcessError, OSError, json.JSONDecodeError) as error:
            audit["error"] = str(error)
            results.append({"request_id": request_id, "study_id": str(item["study_id"]), "case_label": str(item["case_label"]), "status": "failed", "error": str(error)})
            route_rows.append(audit)
            write_route_audit(args.output_root / "route_audit.csv", route_rows)
            if not args.continue_on_error:
                break
            continue
        route_rows.append(audit)
        write_route_audit(args.output_root / "route_audit.csv", route_rows)

    succeeded = sum(row["status"] == "success" for row in results)
    failed = len(results) - succeeded
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "evidence_schema": 5,
        "evidence_enrichment": "polarity-strength-selected-label-checklist-v1",
        "vision_policy": "text-only iff every classifier-selected label is retriever SUPPORTED; otherwise attach original image + evidence",
        "image_detail_for_disagreement": "high",
        "lambda_global": args.lambda_global,
        "requested": len(selected), "succeeded": succeeded, "failed": failed, "results": results,
    }
    (args.output_root / "batch_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.skip_nlp_evaluation:
        return
    if failed and not args.allow_partial_evaluation:
        raise RuntimeError(f"Batch is incomplete: succeeded={succeeded}, failed={failed}. NLP evaluation was not run.")
    evaluation_command = [
        sys.executable, str(ROOT / "scripts" / "evaluate_llm_outputs.py"),
        "--summaries-dir", str(llm_root),
        "--summary-glob", "**/summary_*.json",
        "--output-json", str(args.output_root / "nlp_metrics_target6.json"),
        "--per-sample-csv", str(args.output_root / "nlp_per_sample_target6.csv"),
    ]
    run_logged(evaluation_command, logs_dir / "nlp_evaluation.log")
    summary["nlp_evaluation"] = {
        "metrics_json": str(args.output_root / "nlp_metrics_target6.json"),
        "per_sample_csv": str(args.output_root / "nlp_per_sample_target6.csv"),
    }
    if args.limit == 206 and not args.skip_target_bootstrap:
        bootstrap_command = [
            sys.executable, str(ROOT / "scripts" / "bootstrap_end_to_end_target_metrics.py"),
            "--input", f"{args.model.replace('.', '_')}={args.output_root / 'nlp_per_sample_target6.csv'}",
            "--output-dir", str(args.output_root / "target_label_bootstrap"),
            "--skip-reference-validation",
        ]
        run_logged(bootstrap_command, logs_dir / "target_bootstrap.log")
        summary["target_label_bootstrap"] = str(args.output_root / "target_label_bootstrap" / "table_iv_target_label_bootstrap_ci.json")
    elif args.limit != 206:
        (args.output_root / "evaluation_note.txt").write_text(
            "NLP metrics were calculated for a smoke/partial run. Target-label bootstrap is intentionally skipped unless --limit 206.\n",
            encoding="utf-8",
        )
    (args.output_root / "batch_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
