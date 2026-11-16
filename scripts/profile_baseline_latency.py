"""Fresh, sequential image-to-report latency for baselines A and B."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
import torch
from pathlib import Path

import sklearn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.profile_learned_gate_latency import run_logged, save_results, write_json, resume_rows
from scripts.run_baseline_leave_one_out_batch import command_for_request
from scripts.run_terra_agreement_vision_batch import query_image_path
from src.classifier_thresholds import evidence_thresholds
from src.prompts import validate_radiology_prediction

BASELINES = {"A": "classifier_vision", "B": "classifier_retriever_vision"}
API_SETTINGS = {"model": "gpt-5.6-luna", "reasoning_effort": "high", "max_output_tokens": 8192}


def validate_summary(report: dict, item: dict, baseline: str, settings: dict | None = None) -> None:
    settings = settings or API_SETTINGS
    if report["model"] != API_SETTINGS["model"]:
        raise ValueError("Unexpected LLM model.")
    if report.get("reasoning_effort") != settings["reasoning_effort"]:
        raise ValueError("Unexpected reasoning effort.")
    if report.get("max_output_tokens") != settings["max_output_tokens"]:
        raise ValueError("Unexpected output token budget.")
    if str(report["source"]["study_id"]) != str(item["study_id"]):
        raise ValueError("Report study ID differs from the query.")
    if report["source"]["image_name"] != query_image_path(item).name:
        raise ValueError("Report image differs from the query.")
    api = report["api"]
    if api.get("cached") or not api.get("response_id"):
        raise ValueError("Expected a fresh API response.")
    evidence_thresholds({"classifier": {
        "raw_probabilities": report["classifier_probabilities"],
        "selected_labels": report["classifier_selected_labels"],
        "thresholds": report["classifier_thresholds"],
        "threshold_version": report["classifier_threshold_version"],
    }})
    if validate_radiology_prediction(report["prediction"], tuple(report["classifier_selected_labels"])) is None:
        raise ValueError("Generated report does not satisfy the selected-label output contract.")
    if report["baseline"] != BASELINES[baseline] + "_unstructured_input":
        raise ValueError("Report baseline differs from the requested branch.")
    if baseline == "B":
        nearest = report["five_nearest_cases"]
        studies = [str(case["reference_study_id"]) for case in nearest]
        if len(studies) != 5 or len(set(studies)) != 5 or str(item["study_id"]) in studies:
            raise ValueError("Baseline B requires five distinct references excluding the query study.")
        if any(case["densenet_feature_map_similarity"] is None for case in nearest):
            raise ValueError("Baseline B is missing local feature-map scores.")
    elif "five_nearest_cases" in report:
        raise ValueError("Baseline A must not include retrieval results.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/evaluations/leave_one_out_206_manifest.json")
    parser.add_argument("--reference-run", type=Path, default=ROOT / "outputs/rule_latency_206_20261029")
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--baseline", choices=("A", "B", "both"), default="both")
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--reasoning-effort", choices=("none", "low", "medium", "high", "xhigh", "max"), default="high")
    parser.add_argument("--max-output-tokens", type=int, default=8192)
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        parser.error("CUDA is required; CPU fallback is disabled.")
    settings = {**API_SETTINGS, "reasoning_effort": args.reasoning_effort, "max_output_tokens": args.max_output_tokens}
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    reference = json.loads((args.reference_run / "run_config.json").read_text(encoding="utf-8"))
    digest = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    if any(reference[key] != value for key, value in settings.items()):
        parser.error("Existing baseline API settings do not match the reference run.")
    if digest != reference["manifest_sha256"] or args.device != reference["device"]:
        parser.error("Manifest/device differs from the rule reference run.")
    if len(manifest["requests"]) != 206 or not 1 <= args.limit <= 206:
        parser.error("Expected the frozen 206-study manifest and --limit in [1, 206].")
    requests = manifest["requests"][:args.limit]
    for item in requests:
        query_image_path(item)
    config = {
        **settings, "requested": args.limit, "device": args.device, "image_detail": "high",
        "local_response_cache": False, "timing_fields": ["end_to_end_s"],
        "manifest": str(args.manifest), "manifest_sha256": digest,
        "reference_run": str(args.reference_run),
        "python": sys.version, "platform": platform.platform(), "sklearn": sklearn.__version__,
        "protocol": "sequential fresh baseline process per case; process/model loading, image inference, API retries and saved validated report included",
        "cache_policy": "fresh image inference and API response; local LLM cache disabled; pretrained weights and fixed gallery reused; provider prompt caching may occur",
        "comparison_note": "A/B retain their unstructured prompts and always attach the high-detail query image. Each baseline uses one child process per case; rule/learned runners use separate inference and LLM child processes.",
    }
    baselines = list(BASELINES) if args.baseline == "both" else [args.baseline]
    if args.preview:
        print(json.dumps({**config, "baselines": baselines}, indent=2))
        return
    if args.output_root.exists() and any(args.output_root.iterdir()) and not args.resume:
        parser.error("Use a new empty --output-root to preserve existing results.")
    args.output_root.mkdir(parents=True, exist_ok=True)
    write_json(args.output_root / "manifest.json", manifest)
    write_json(args.output_root / "run_config.json", {**config, "baselines": baselines})
    completed_baselines = {}
    for baseline in baselines:
        root = args.output_root / f"baseline_{baseline.lower()}"
        branch_config = {**config, "branch": f"baseline_{baseline.lower()}", "baseline": BASELINES[baseline]}
        rows, previous_elapsed = resume_rows(root, branch_config, args.resume)
        for folder in ("llm", "logs"):
            (root / folder).mkdir(parents=True, exist_ok=True)
        write_json(root / "manifest.json", manifest)
        write_json(root / "run_config.json", branch_config)
        if [r["request_id"] for r in rows] != [r["request_id"] for r in requests[:len(rows)]]:
            parser.error("Resume studies do not match the manifest prefix.")
        started = time.perf_counter()
        for position, item in enumerate(requests[len(rows):], len(rows)+1):
            request_id = item["request_id"]
            llm_dir = root / "llm" / request_id
            local_item = {**item, "image_path": str(query_image_path(item))}
            command = command_for_request(local_item, BASELINES[baseline], root, args.device, "high", args.max_output_tokens, args.reasoning_effort)
            row = {"request_id": request_id, "study_id": str(item["study_id"]), "status": "failed"}
            print(f"[{baseline} {position}/{len(requests)}] {request_id}: starting fresh image inference", flush=True)
            case_started = time.perf_counter()
            try:
                run_logged(command, root / "logs" / f"{request_id}.log")
                row["end_to_end_s"] = time.perf_counter() - case_started
                summaries = list(llm_dir.glob("summary_*.json"))
                if len(summaries) != 1:
                    raise ValueError("Expected exactly one newly generated report.")
                report = json.loads(summaries[0].read_text(encoding="utf-8"))
                validate_summary(report, item, baseline, settings)
                row.update(status="success", response_id=report["api"]["response_id"],
                           route="vision_always", image_attached=True, summary=str(summaries[0]))
                print(f"[{baseline} {position}/{len(requests)}] OK {row['end_to_end_s']:.3f}s", flush=True)
            except Exception as error:
                row.update(error=str(error), failed_attempt_s=time.perf_counter() - case_started)
                print(f"[{baseline} {position}/{len(requests)}] FAILED: {error}", flush=True)
            rows.append(row)
            save_results(root, rows, branch_config, previous_elapsed + time.perf_counter() - started, False)
            if row["status"] != "success":
                break
        complete = len(rows) == len(requests) and all(row["status"] == "success" for row in rows)
        result = save_results(root, rows, branch_config, previous_elapsed + time.perf_counter() - started, complete)
        completed_baselines[baseline] = {key: result[key] for key in ("complete", "succeeded", "failed", "latency", "batch_wall_s")}
        write_json(args.output_root / "batch_summary.json", completed_baselines)
        print(f"Baseline {baseline}: " + json.dumps(completed_baselines[baseline]), flush=True)
        if not complete:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
