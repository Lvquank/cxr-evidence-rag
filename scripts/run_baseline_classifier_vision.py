#!/usr/bin/env python3
"""Baseline A: image -> classifier probabilities -> Luna vision, without evidence structuring.

The request contains only a flat natural-language list of six classifier scores
and the original image.  It deliberately does not call the retriever, build
support/non-support evidence, assign evidence statuses, or create minimal v4 JSON.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from llm_client import create_llm_client
from prompts import MedicalPrompts, TARGET_PROBLEM_LABELS, validate_radiology_prediction
from classifier_thresholds import DEFAULT_THRESHOLDS_PATH, load_classifier_thresholds, select_labels
from scripts.predict_chexpert_6labels import MAIN_LABELS, load_model, predict


def resolve_device(value: str) -> torch.device:
    return torch.device("cuda" if value == "auto" and torch.cuda.is_available() else ("cpu" if value == "auto" else value))


def classifier_prompt(probabilities: Dict[str, float], selected_labels: tuple[str, ...]) -> Dict[str, str]:
    scores = "\n".join(f"- {label}: {probabilities[label]:.4f}" for label in MAIN_LABELS)
    return {
        "system": (
            "You are a radiologist reviewing a chest X-ray. The image is primary evidence. "
            "The classifier probabilities are auxiliary predictions, not verified findings. "
            "For problems, use only classifier-passed labels: " + (", ".join(selected_labels) or "none") + ", or exactly [normal] when no target pathology is present. "
            "Place non-target observations only in findings. Return only the requested JSON output; do not mention the classifier or the experiment."
        ),
        "user": (
            "Analyze the attached chest X-ray directly.\n\n"
            "Classifier probabilities:\n" + scores +
            "\n\nProduce concise problems, findings, and impression from the image."
        ),
    }


def parse_prediction(text: str, selected_labels: tuple[str, ...] = TARGET_PROBLEM_LABELS) -> Optional[Dict[str, Any]]:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    return validate_radiology_prediction(parsed, selected_labels)


def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline A: classifier probabilities plus Luna vision only.")
    parser.add_argument("--image-path", required=True, type=Path)
    parser.add_argument("--study-id", help="IU-Xray uid saved with the response for offline evaluation")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/baselines/classifier_vision")
    parser.add_argument("--classifier-checkpoint", default="model/best_E2_relation_aware_weighted_bce.pt")
    parser.add_argument("--thresholds-json", type=Path, default=DEFAULT_THRESHOLDS_PATH)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image-detail", choices=["low", "high"], default="high")
    parser.add_argument("--max-output-tokens", type=int, default=8192)
    parser.add_argument("--reasoning-effort", choices=["none", "low", "medium", "high", "xhigh", "max"], default="high")
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()
    if not args.image_path.is_file():
        raise FileNotFoundError(args.image_path)
    device = resolve_device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this experiment; CPU fallback is disabled.")
    classifier, _ = load_model(Path(args.classifier_checkpoint), device)
    probabilities = predict(args.image_path, classifier, device)
    thresholds, threshold_artifact = load_classifier_thresholds(args.thresholds_json)
    selected_labels = select_labels(probabilities, thresholds)
    del classifier
    if device.type == "cuda":
        torch.cuda.empty_cache()
    prompt = classifier_prompt(probabilities, selected_labels)
    client = create_llm_client(
        model="gpt-5.6-luna", max_tokens=args.max_output_tokens,
        reasoning_effort=args.reasoning_effort, cache_enabled=not args.no_cache,
    )
    response = client.send_request(
        prompt["user"], system=prompt["system"], image_path=str(args.image_path),
        image_detail=args.image_detail, response_schema=MedicalPrompts.get_output_schema("radiology", selected_labels),
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        "baseline": "classifier_vision_unstructured_input",
        "timestamp": timestamp,
        "model": "gpt-5.6-luna",
        "reasoning_effort": args.reasoning_effort,
        "max_output_tokens": args.max_output_tokens,
        "source": {
            "study_id": str(args.study_id) if args.study_id is not None else None,
            "image_name": args.image_path.name,
            "image_path": str(args.image_path.resolve()),
        },
        "classifier_probabilities": {label: round(float(probabilities[label]), 6) for label in MAIN_LABELS},
        "classifier_thresholds": thresholds,
        "classifier_threshold_version": threshold_artifact["version"],
        "classifier_selected_labels": list(selected_labels),
        "prompt": prompt,
        "response": response,
        "prediction": parse_prediction(response, selected_labels),
        "api": client.get_last_response_metadata(),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    destination = args.output / f"summary_{timestamp}.json"
    destination.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(destination)


if __name__ == "__main__":
    main()
