"""Text-only arm of the vision ablation on the frozen 206-study manifest.

Reuses the frozen classifier/retriever evidence of the Luna-high v5 fresh run
(the vision arm) and sends every case to the same model and prompt with
--disable-vision-escalation. Only the image input differs between arms.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_terra_agreement_vision_batch import has_completed_llm_summary, query_image_path, run_logged

DEFAULT_MANIFEST = ROOT / "data" / "evaluations" / "leave_one_out_206_manifest.json"
DEFAULT_VISION_ROOT = ROOT / "outputs" / "luna_high_evidence_v5_fresh_20260930"
DEFAULT_OUTPUT = ROOT / "outputs" / "text_only_ablation_luna_high_206"


def llm_command(item: dict, evidence_path: Path, llm_dir: Path, args: argparse.Namespace) -> list[str]:
    return [
        sys.executable, str(ROOT / "scripts" / "run_full_pipeline.py"),
        "--input", str(evidence_path),
        "--output", str(llm_dir),
        "--image-path", str(query_image_path(item)),
        "--study-id", str(item["study_id"]),
        "--model", args.model,
        "--escalation-model", args.model,
        "--reasoning-effort", args.reasoning_effort,
        "--max-output-tokens", str(args.max_output_tokens),
        "--no-cache",
        "--disable-vision-escalation",
        *(["--dry-run"] if args.dry_run else []),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--vision-root", type=Path, default=DEFAULT_VISION_ROOT, help="Vision arm whose evidence/ is reused")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default="gpt-5.6-luna")
    parser.add_argument("--reasoning-effort", default="high")
    parser.add_argument("--max-output-tokens", type=int, default=1024)
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--only", nargs="*", help="Run only these request IDs (e.g. token-cap retries)")
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Build prompts without sending API requests")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    requests = json.loads(args.manifest.read_text(encoding="utf-8"))["requests"]
    if len(requests) != 206:
        raise RuntimeError(f"Expected 206 frozen requests, found {len(requests)}.")
    selected = [item for item in requests if item["request_id"] in set(args.only)] if args.only else requests[:args.limit]
    source_evidence = args.vision_root / "evidence"
    missing = [item["request_id"] for item in selected if not (source_evidence / f"{item['request_id']}.json").is_file()]
    if missing:
        raise FileNotFoundError(f"Vision-arm evidence missing for: {missing}")
    if args.preview:
        for item in selected:
            print(json.dumps({"request_id": item["request_id"], "command": llm_command(
                item, source_evidence / f"{item['request_id']}.json", args.output_root / "llm" / item["request_id"], args)}))
        return
    if args.output_root.exists() and any(args.output_root.iterdir()) and not args.resume:
        parser.error("--output-root must be empty for a new batch; use --resume to continue it.")

    evidence_dir, llm_root, logs_dir = (args.output_root / name for name in ("evidence", "llm", "logs"))
    for directory in (evidence_dir, llm_root, logs_dir):
        directory.mkdir(parents=True, exist_ok=True)
    results = []
    for position, item in enumerate(selected, start=1):
        request_id = item["request_id"]
        evidence_path = evidence_dir / f"{request_id}.json"
        if not evidence_path.is_file():
            shutil.copy2(source_evidence / f"{request_id}.json", evidence_path)
        llm_dir = llm_root / request_id
        print(f"[{position}/{len(selected)}] {request_id}", flush=True)
        if args.resume and not args.only and has_completed_llm_summary(llm_dir, args.model, args.reasoning_effort):
            results.append({"request_id": request_id, "status": "success", "resumed": True})
            continue
        try:
            run_logged(llm_command(item, evidence_path, llm_dir, args), logs_dir / f"{request_id}_llm.log")
            results.append({"request_id": request_id, "status": "success"})
        except (subprocess.CalledProcessError, OSError) as error:
            results.append({"request_id": request_id, "status": "failed", "error": str(error)})
            print(f"  failed: {error}", flush=True)

    summary_path = args.output_root / "batch_summary.json"
    previous = json.loads(summary_path.read_text(encoding="utf-8"))["results"] if summary_path.is_file() else []
    merged = {row["request_id"]: row for row in previous}
    merged.update({row["request_id"]: row for row in results})
    summary_path.write_text(json.dumps({
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "arm": "text_only",
        "vision_arm_root": str(args.vision_root),
        "evidence_source": str(source_evidence),
        "model": args.model, "reasoning_effort": args.reasoning_effort,
        "max_output_tokens_last_call": args.max_output_tokens, "dry_run": args.dry_run,
        "succeeded": sum(r["status"] == "success" for r in merged.values()),
        "failed": sum(r["status"] != "success" for r in merged.values()),
        "results": list(merged.values()),
    }, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
