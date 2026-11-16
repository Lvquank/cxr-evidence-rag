#!/usr/bin/env python3
"""Run one unstructured-input Luna baseline over the frozen 206-study manifest.

The source image, study ID, and request order are exactly those used by
``run_leave_one_out_batch.py``.  Outputs are deliberately written under a
separate baseline-specific root so that they cannot overwrite the structured
evidence experiment.
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
BASELINE_SCRIPTS = {
    "classifier_vision": ROOT / "scripts" / "run_baseline_classifier_vision.py",
    "classifier_retriever_vision": ROOT / "scripts" / "run_baseline_classifier_retriever_vision.py",
}


def command_for_request(
    item: Dict[str, Any], baseline: str, output_root: Path, device: str,
    image_detail: str, max_output_tokens: int, reasoning_effort: str = "high",
) -> List[str]:
    """Build one no-cache request using the same query image and UID as the original run."""
    return [
        sys.executable,
        str(BASELINE_SCRIPTS[baseline]),
        "--image-path", str(item["image_path"]),
        "--study-id", str(item["study_id"]),
        "--output", str(output_root / "llm" / str(item["request_id"])),
        "--device", device,
        "--image-detail", image_detail,
        "--max-output-tokens", str(max_output_tokens),
        "--reasoning-effort", reasoning_effort,
        "--no-cache",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run an image-attached Luna baseline on the same leave-one-study-out manifest."
    )
    parser.add_argument(
        "--baseline", required=True, choices=sorted(BASELINE_SCRIPTS),
        help="classifier_vision or classifier_retriever_vision",
    )
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/evaluations/leave_one_out_206_manifest.json")
    parser.add_argument(
        "--output-root", type=Path,
        help="Defaults to outputs/baselines/<baseline>_leave_one_out_206.",
    )
    parser.add_argument("--limit", type=int, default=206, help="Number of leading manifest requests; 206 runs all frozen queries.")
    parser.add_argument("--device", default="cuda", help="Classifier/retriever CUDA device")
    parser.add_argument("--image-detail", choices=["low", "high"], default="high")
    parser.add_argument("--max-output-tokens", type=int, default=8192)
    parser.add_argument("--reasoning-effort", choices=["none", "low", "medium", "high", "xhigh", "max"], default="high")
    parser.add_argument("--preview", action="store_true", help="Print commands only; do not run classifier, retriever, or LLM.")
    parser.add_argument("--stop-on-error", action="store_true")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    requests = list(manifest.get("requests", []))
    if len(requests) != 206:
        raise RuntimeError(f"Expected the frozen 206-request manifest; found {len(requests)} requests.")
    if not 1 <= args.limit <= len(requests):
        parser.error(f"--limit must be between 1 and {len(requests)}.")

    output_root = args.output_root or ROOT / "outputs" / "baselines" / f"{args.baseline}_leave_one_out_206"
    selected = requests[:args.limit]
    if args.preview:
        for item in selected:
            print(json.dumps({
                "request_id": item["request_id"],
                "study_id": item["study_id"],
                "image_path": item["image_path"],
                "command": command_for_request(
                    item, args.baseline, output_root, args.device, args.image_detail, args.max_output_tokens, args.reasoning_effort
                ),
            }, ensure_ascii=False))
        return

    llm_dir = output_root / "llm"
    logs_dir = output_root / "logs"
    llm_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    (output_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    results = []
    for position, item in enumerate(selected, start=1):
        request_id = str(item["request_id"])
        print(f"[{position}/{len(selected)}] {request_id}: study={item['study_id']}, class={item['case_label']}", flush=True)
        command = command_for_request(
            item, args.baseline, output_root, args.device, args.image_detail, args.max_output_tokens, args.reasoning_effort
        )
        completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
        (logs_dir / f"{request_id}.log").write_text(
            "COMMAND:\n" + subprocess.list2cmdline(command)
            + "\n\nSTDOUT:\n" + completed.stdout + "\n\nSTDERR:\n" + completed.stderr,
            encoding="utf-8",
        )
        row = {
            "request_id": request_id,
            "study_id": item["study_id"],
            "case_label": item["case_label"],
            "status": "success" if completed.returncode == 0 else "failed",
        }
        if completed.returncode:
            row["returncode"] = completed.returncode
        results.append(row)
        if completed.returncode and args.stop_on_error:
            break

    summary_path = output_root / "batch_summary.json"
    summary_path.write_text(json.dumps({
        "baseline": args.baseline,
        "reasoning_effort": args.reasoning_effort,
        "max_output_tokens": args.max_output_tokens,
        "manifest": str(args.manifest.resolve()),
        "output_root": str(output_root.resolve()),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "requested": len(selected),
        "succeeded": sum(row["status"] == "success" for row in results),
        "failed": sum(row["status"] == "failed" for row in results),
        "results": results,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Batch summary: {summary_path}")


if __name__ == "__main__":
    main()
