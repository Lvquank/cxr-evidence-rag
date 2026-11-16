#!/usr/bin/env python3
"""Run the fixed leave-one-study-out manifest sequentially.

Each request excludes only its own query study from retrieval.  Evidence and
LLM outputs are isolated by request_id to make the 206 cases auditable.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parents[1]


def command_for_request(
    item: Dict[str, Any], output_root: Path, device: str, max_output_tokens: int, lambda_global: float = 0.9
) -> List[List[str]]:
    request_id = item["request_id"]
    evidence = output_root / "evidence" / f"{request_id}.json"
    llm_output = output_root / "llm" / request_id
    classifier_retriever = [
        sys.executable, str(ROOT / "scripts/test_classifier_to_retriever.py"),
        "--image_path", item["image_path"],
        "--output_json", str(evidence),
        "--exclude-study-id", str(item["study_id"]),
        "--device", device,
        "--lambda_global", str(lambda_global),
    ]
    pipeline = [
        sys.executable, str(ROOT / "scripts/run_full_pipeline.py"),
        "--input", str(evidence),
        "--output", str(llm_output),
        "--image-path", item["image_path"],
        "--study-id", str(item["study_id"]),
        "--model", "gpt-5.6-luna",
        "--max-output-tokens", str(max_output_tokens),
        "--no-cache",
    ]
    return [classifier_retriever, pipeline]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 206 leave-one-study-out classifier/retriever/LLM requests.")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/evaluations/leave_one_out_206_manifest.json")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/leave_one_out_206")
    parser.add_argument("--limit", type=int, default=206, help="Use 1 first as a no-surprises smoke test; max 206.")
    parser.add_argument("--device", default="cuda", help="Classifier/retriever CUDA device")
    parser.add_argument("--lambda_global", type=float, default=0.9, help="Global retrieval score weight; local DenseNet weight is 1 - lambda_global")
    parser.add_argument("--max-output-tokens", type=int, default=8192)
    parser.add_argument("--preview", action="store_true", help="Print planned commands without running classifier, retriever, or LLM.")
    parser.add_argument("--stop-on-error", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.limit <= 206:
        parser.error("--limit must be between 1 and 206.")
    if not 0.0 <= args.lambda_global <= 1.0:
        parser.error("--lambda_global must be between 0 and 1.")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    requests = list(manifest.get("requests", []))
    if len(requests) != 206:
        raise RuntimeError(f"Manifest must contain exactly 206 requests; found {len(requests)}.")
    selected = requests[:args.limit]
    if args.preview:
        for item in selected:
            print(json.dumps({"request_id": item["request_id"], "study_id": item["study_id"], "commands": command_for_request(item, args.output_root, args.device, args.max_output_tokens, args.lambda_global)}, ensure_ascii=False))
        return

    evidence_dir = args.output_root / "evidence"
    llm_dir = args.output_root / "llm"
    logs_dir = args.output_root / "logs"
    for directory in (evidence_dir, llm_dir, logs_dir):
        directory.mkdir(parents=True, exist_ok=True)
    # Snapshot the exact sample list used so results remain reproducible if the
    # source manifest is regenerated with another seed later.
    (args.output_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    results = []
    for position, item in enumerate(selected, start=1):
        request_id = item["request_id"]
        print(f"[{position}/{len(selected)}] {request_id}: study={item['study_id']}, class={item['case_label']}", flush=True)
        try:
            for step, command in zip(("classifier_retriever", "llm"), command_for_request(item, args.output_root, args.device, args.max_output_tokens, args.lambda_global)):
                completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
                (logs_dir / f"{request_id}_{step}.log").write_text(
                    "COMMAND:\n" + subprocess.list2cmdline(command) + "\n\nSTDOUT:\n" + completed.stdout + "\n\nSTDERR:\n" + completed.stderr,
                    encoding="utf-8",
                )
                if completed.returncode:
                    raise subprocess.CalledProcessError(completed.returncode, command)
            results.append({"request_id": request_id, "study_id": item["study_id"], "case_label": item["case_label"], "status": "success"})
        except subprocess.CalledProcessError as error:
            results.append({"request_id": request_id, "study_id": item["study_id"], "case_label": item["case_label"], "status": "failed", "returncode": error.returncode})
            if args.stop_on_error:
                break
    result_path = args.output_root / "batch_summary.json"
    result_path.write_text(json.dumps({
        "manifest": str(args.manifest), "output_root": str(args.output_root), "lambda_global": args.lambda_global, "finished_at": datetime.now(timezone.utc).isoformat(),
        "requested": len(selected), "succeeded": sum(row["status"] == "success" for row in results),
        "failed": sum(row["status"] == "failed" for row in results), "results": results,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Batch summary: {result_path}")


if __name__ == "__main__":
    main()
