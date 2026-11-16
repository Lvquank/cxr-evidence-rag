#!/usr/bin/env python3
"""Run the controlled all-label image rescue ablation on the 206-study cohort.

Protocol specified in HANDOFF_ALL_LABEL_PROBABILITY_IMAGE_RESCUE_ABLATION.md:
1. Two branches evaluated on the same 206-study manifest:
   - selected_only (control): allowed schema restricted to classifier.selected_labels (or normal)
   - all_six (rescue): allowed schema open to all 6 target pathologies (or normal)
2. Frozen configuration:
   - Same manifest, image, and DenseNet thresholds
   - Same frozen v5 compact evidence + 'clf' 6-pathology probabilities
   - Query image attached for all 206 cases (force-vision)
   - Same prompt (radiology_rescue), model (gpt-5.6-luna), reasoning effort (high)
3. Results layout under outputs/all_label_image_rescue_ablation_206/:
   - manifest.json
   - protocol.json
   - evidence_snapshot/
   - selected_only/ (llm, logs, nlp_metrics.json, nlp_per_sample.csv)
   - all_six/ (llm, logs, nlp_metrics.json, nlp_per_sample.csv)
   - comparison/ (rescue_per_study.csv, rescue_summary.json, table_iv_target_label_bootstrap_ci.json)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "data" / "evaluations" / "leave_one_out_206_manifest.json"
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "all_label_image_rescue_ablation_206"
DEFAULT_EVIDENCE_SOURCE = ROOT / "outputs" / "luna_high_polarity_checklist_v5_206" / "evidence"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def query_image_path(item: Dict[str, Any]) -> Path:
    original = Path(item["image_path"])
    if original.is_file():
        return original
    relocated = ROOT / "data" / "reprocessed" / "images" / f"{item['image_id']}{original.suffix}"
    if relocated.is_file():
        return relocated
    raise FileNotFoundError(f"Query image unavailable for {item.get('request_id')}: {original}")


def run_logged(command: List[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        capture_output=True,
        stdin=subprocess.DEVNULL,
        encoding="utf-8",
        errors="replace",
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )
    log_path.write_text(
        "COMMAND:\n" + subprocess.list2cmdline(command) + "\n\nSTDOUT:\n" + completed.stdout + "\n\nSTDERR:\n" + completed.stderr,
        encoding="utf-8",
    )
    if completed.returncode != 0:
        raise subprocess.CalledProcessError(completed.returncode, command, completed.stdout, completed.stderr)


def has_completed_summary(
    llm_case_dir: Path,
    model: str,
    reasoning_effort: str,
    policy: str,
) -> bool:
    if not llm_case_dir.is_dir():
        return False
    for path in llm_case_dir.glob("summary_*.json"):
        try:
            summary = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if (
            summary.get("requested_model") == model
            and summary.get("model") == model
            and summary.get("reasoning_effort") == reasoning_effort
            and summary.get("problem_label_policy") == policy
            and summary.get("evidence_schema") == 5
            and isinstance(summary.get("response"), str)
            and summary.get("prediction") is not None
            and summary.get("source")
        ):
            return True
    return False


def build_pipeline_command(
    evidence_path: Path,
    output_dir: Path,
    image_path: Path,
    study_id: str,
    model: str,
    reasoning_effort: str,
    policy: str,
    max_output_tokens: int = 160,
    image_detail: str = "high",
    temperature: Optional[float] = None,
    api_key: Optional[str] = None,
    dry_run: bool = False,
    no_cache: bool = False,
) -> List[str]:
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_full_pipeline.py"),
        "--input", str(evidence_path),
        "--output", str(output_dir),
        "--image-path", str(image_path),
        "--study-id", str(study_id),
        "--model", model,
        "--escalation-model", model,
        "--reasoning-effort", reasoning_effort,
        "--max-output-tokens", str(max_output_tokens),
        "--image-detail", image_detail,
        "--force-vision",
        "--prompt", "radiology_rescue",
        "--include-all-classifier-probabilities",
        "--problem-label-policy", policy,
    ]
    if temperature is not None:
        cmd.extend(["--temperature", str(temperature)])
    if api_key:
        cmd.extend(["--api-key", api_key])
    if dry_run:
        cmd.append("--dry-run")
    if no_cache:
        cmd.append("--no-cache")
    return cmd


def ensure_evidence_snapshot(
    manifest_items: List[Dict[str, Any]],
    evidence_source_dir: Optional[Path],
    snapshot_dir: Path,
    device: str = "cpu",
    lambda_global: float = 0.9,
) -> None:
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    for item in manifest_items:
        req_id = item["request_id"]
        target = snapshot_dir / f"{req_id}.json"
        if target.is_file():
            continue
        source_candidate = (evidence_source_dir / f"{req_id}.json") if evidence_source_dir else None
        if source_candidate and source_candidate.is_file():
            shutil.copyfile(source_candidate, target)
            logger.info("Copied evidence snapshot: %s -> %s", source_candidate.name, target.name)
        else:
            logger.info("Generating evidence for %s via test_classifier_to_retriever.py...", req_id)
            img_path = query_image_path(item)
            gen_cmd = [
                sys.executable,
                str(ROOT / "scripts" / "test_classifier_to_retriever.py"),
                "--image_path", str(img_path),
                "--output_json", str(target),
                "--exclude-study-id", str(item["study_id"]),
                "--device", device,
                "--lambda_global", str(lambda_global),
            ]
            completed = subprocess.run(gen_cmd, cwd=ROOT, text=True, capture_output=True, stdin=subprocess.DEVNULL)
            if completed.returncode != 0:
                raise RuntimeError(f"Failed to generate evidence for {req_id}: {completed.stderr}")


def run_arm(
    arm: str,
    manifest_items: List[Dict[str, Any]],
    output_root: Path,
    snapshot_dir: Path,
    model: str,
    reasoning_effort: str,
    max_output_tokens: int,
    image_detail: str,
    temperature: Optional[float],
    api_key: Optional[str],
    dry_run: bool,
    no_cache: bool,
    preview: bool,
    stop_on_error: bool,
) -> None:
    arm_dir = output_root / arm
    llm_root = arm_dir / "llm"
    logs_root = arm_dir / "logs"
    llm_root.mkdir(parents=True, exist_ok=True)
    logs_root.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("Running arm: %s (%d requests)", arm, len(manifest_items))
    logger.info("=" * 70)

    for index, item in enumerate(manifest_items, 1):
        req_id = item["request_id"]
        study_id = str(item["study_id"])
        image = query_image_path(item)
        evidence = snapshot_dir / f"{req_id}.json"
        case_llm_dir = llm_root / req_id
        case_log = logs_root / f"{req_id}.log"

        if not preview and not no_cache and has_completed_summary(case_llm_dir, model, reasoning_effort, arm):
            logger.info("[%d/%d] Skipping completed case %s (%s)", index, len(manifest_items), req_id, arm)
            continue

        cmd = build_pipeline_command(
            evidence_path=evidence,
            output_dir=case_llm_dir,
            image_path=image,
            study_id=study_id,
            model=model,
            reasoning_effort=reasoning_effort,
            policy=arm,
            max_output_tokens=max_output_tokens,
            image_detail=image_detail,
            temperature=temperature,
            api_key=api_key,
            dry_run=dry_run,
            no_cache=no_cache,
        )

        if preview:
            print(f"[{arm} {index}/{len(manifest_items)}] " + subprocess.list2cmdline(cmd))
            continue

        logger.info("[%d/%d] Running %s (%s)...", index, len(manifest_items), req_id, arm)
        try:
            run_logged(cmd, case_log)
        except Exception as err:
            logger.error("Failed on %s (%s): %s", req_id, arm, err)
            if stop_on_error:
                raise


def run_evaluation(output_root: Path, num_studies: int, bertscore_device: str = "cpu") -> None:
    logger.info("\n" + "=" * 70)
    logger.info("Running offline evaluations for selected_only and all_six...")
    logger.info("=" * 70)

    for arm in ["selected_only", "all_six"]:
        arm_dir = output_root / arm
        llm_dir = arm_dir / "llm"
        metrics_json = arm_dir / "nlp_metrics.json"
        per_sample_csv = arm_dir / "nlp_per_sample.csv"

        eval_cmd = [
            sys.executable,
            str(ROOT / "scripts" / "evaluate_llm_outputs.py"),
            "--summaries-dir", str(llm_dir),
            "--summary-glob", "**/summary_*.json",
            "--output-json", str(metrics_json),
            "--per-sample-csv", str(per_sample_csv),
            "--bertscore-device", bertscore_device,
        ]
        logger.info("Evaluating %s: %s", arm, subprocess.list2cmdline(eval_cmd))
        subprocess.run(eval_cmd, cwd=ROOT, check=True, stdin=subprocess.DEVNULL)

    comparison_dir = output_root / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    # Paired Table IV bootstrap
    boot_cmd = [
        sys.executable,
        str(ROOT / "scripts" / "bootstrap_end_to_end_target_metrics.py"),
        "--input", f"all_six={output_root / 'all_six' / 'nlp_per_sample.csv'}",
        "--input", f"selected_only={output_root / 'selected_only' / 'nlp_per_sample.csv'}",
        "--required-study-count", str(num_studies),
        "--bootstrap-samples", "2000",
        "--seed", "20260904",
        "--output-dir", str(comparison_dir),
    ]
    logger.info("Running paired Table IV bootstrap: %s", subprocess.list2cmdline(boot_cmd))
    subprocess.run(boot_cmd, cwd=ROOT, check=True, stdin=subprocess.DEVNULL)

    # Detailed rescue & tradeoffs analysis
    rescue_cmd = [
        sys.executable,
        str(ROOT / "scripts" / "compute_rescue_ablation_comparison.py"),
        "--control-csv", str(output_root / "selected_only" / "nlp_per_sample.csv"),
        "--rescue-csv", str(output_root / "all_six" / "nlp_per_sample.csv"),
        "--evidence-dir", str(output_root / "evidence_snapshot"),
        "--output-dir", str(comparison_dir),
        "--bootstrap-samples", "2000",
        "--seed", "20260904",
    ]
    logger.info("Running rescue comparison & tradeoffs: %s", subprocess.list2cmdline(rescue_cmd))
    subprocess.run(rescue_cmd, cwd=ROOT, check=True, stdin=subprocess.DEVNULL)

    logger.info("✅ All evaluations completed successfully!")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run controlled all-label rescue ablation.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST, help="Path to manifest JSON")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="Ablation output directory")
    parser.add_argument("--evidence-source", type=Path, default=DEFAULT_EVIDENCE_SOURCE, help="Pre-generated evidence folder")
    parser.add_argument("--arm", choices=["both", "selected_only", "all_six"], default="both", help="Which arm to run")
    parser.add_argument("--model", default="gpt-5.6-luna", help="Model name")
    parser.add_argument("--reasoning-effort", default="high", choices=["none", "low", "medium", "high", "xhigh", "max"])
    parser.add_argument("--max-output-tokens", type=int, default=25000)
    parser.add_argument("--image-detail", default="high", choices=["low", "high"])
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--limit", type=int, default=None, help="Limit number of manifest items (e.g. 1 for smoke test)")
    parser.add_argument("--preview", action="store_true", help="Print planned commands without running")
    parser.add_argument("--dry-run", action="store_true", help="Build artifacts without sending LLM API requests")
    parser.add_argument("--no-cache", action="store_true", help="Do not reuse cached LLM responses")
    parser.add_argument("--stop-on-error", action="store_true", help="Stop execution on first error")
    parser.add_argument("--evaluate", action="store_true", help="Run offline metrics & paired bootstrap after inference")
    parser.add_argument("--bertscore-device", default="cpu", help="BERTScore computation device")
    args = parser.parse_args()

    manifest_data = json.loads(args.manifest.read_text(encoding="utf-8"))
    if isinstance(manifest_data, dict) and "requests" in manifest_data:
        items = manifest_data["requests"][: args.limit] if args.limit else manifest_data["requests"]
        manifest_payload = {**manifest_data, "requests": items, "total_requests": len(items)}
    elif isinstance(manifest_data, list):
        items = manifest_data[: args.limit] if args.limit else manifest_data
        manifest_payload = items
    else:
        raise ValueError(f"Unrecognized manifest format in {args.manifest}")

    protocol = {
        "title": "All-Label Probability and Image Rescue Controlled Ablation",
        "date": "2026-10-03",
        "manifest_file": str(args.manifest),
        "manifest_items_count": len(items),
        "arms": ["selected_only", "all_six"],
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "max_output_tokens": args.max_output_tokens,
        "image_detail": args.image_detail,
        "prompt_template": "radiology_rescue",
        "force_vision": True,
        "include_all_classifier_probabilities": True,
        "frozen_adaptive_thresholds": "config/densenet_adaptive_thresholds.json",
        "evidence_source": str(args.evidence_source) if args.evidence_source else "generated",
    }

    if not args.preview:
        args.output_root.mkdir(parents=True, exist_ok=True)
        manifest_dst = args.output_root / "manifest.json"
        manifest_dst.write_text(json.dumps(manifest_payload, indent=2, ensure_ascii=False), encoding="utf-8")
        protocol_dst = args.output_root / "protocol.json"
        protocol_dst.write_text(json.dumps(protocol, indent=2, ensure_ascii=False), encoding="utf-8")
        snapshot_dir = args.output_root / "evidence_snapshot"
        ensure_evidence_snapshot(items, args.evidence_source, snapshot_dir)
    else:
        snapshot_dir = args.output_root / "evidence_snapshot"

    arms_to_run = ["selected_only", "all_six"] if args.arm == "both" else [args.arm]
    for arm in arms_to_run:
        run_arm(
            arm=arm,
            manifest_items=items,
            output_root=args.output_root,
            snapshot_dir=snapshot_dir,
            model=args.model,
            reasoning_effort=args.reasoning_effort,
            max_output_tokens=args.max_output_tokens,
            image_detail=args.image_detail,
            temperature=args.temperature,
            api_key=args.api_key,
            dry_run=args.dry_run,
            no_cache=args.no_cache,
            preview=args.preview,
            stop_on_error=args.stop_on_error,
        )

    if args.evaluate and not args.preview and set(arms_to_run) == {"selected_only", "all_six"}:
        run_evaluation(args.output_root, len(items), bertscore_device=args.bertscore_device)


if __name__ == "__main__":
    main()
