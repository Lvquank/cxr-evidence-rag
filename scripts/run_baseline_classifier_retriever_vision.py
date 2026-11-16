#!/usr/bin/env python3
"""Baseline B: image -> classifier -> top-5 retrieval summary -> Luna vision.

This is intentionally separate from the structured-evidence pipeline.  It never
calls ``search_label_evidence`` and never creates support/non-support sets,
evidence statuses, or a minimal v4 JSON payload.  Luna receives the original
image plus a flat summary of five globally nearest retrieved studies.
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
from src.config import load_config
from src.embeddings.encoder import DEFAULT_MODEL_NAME
from src.retriever.densenet_local_reranker import DenseNetLocalFeatureReranker
from src.retriever.faiss_retriever import FAISSImageRetriever


TOP_K = 5


def resolve_device(value: str) -> torch.device:
    return torch.device("cuda" if value == "auto" and torch.cuda.is_available() else ("cpu" if value == "auto" else value))


def _as_problem_list(value: Any) -> list[str]:
    """Normalize metadata only for a compact, readable retrieval summary."""
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return [value] if value.strip() else []
        if isinstance(parsed, list):
            return [str(item) for item in parsed if str(item).strip()]
    return []


def summarize_nearest_cases(
    image_path: Path,
    retriever: FAISSImageRetriever,
    classifier_backbone: torch.nn.Module,
    device: torch.device,
    exclude_study_id: Optional[str] = None,
) -> list[Dict[str, Any]]:
    """Get exactly five ANN-nearest studies and independently score local maps.

    The original global rank is retained.  Local feature-map similarity is a
    descriptive number, not a reranking or pathology/evidence decision.
    """
    nearest = retriever.search(
        str(image_path), top_k=TOP_K, candidate_k=50,
        exclude_study_ids=(str(exclude_study_id),) if exclude_study_id else (),
    )
    local_scores = DenseNetLocalFeatureReranker(
        classifier_backbone, device=device, batch_size=TOP_K
    ).score_paths(
        str(image_path), [str(item.get("processed_image_path", "")) for item in nearest]
    )
    summary: list[Dict[str, Any]] = []
    for item in nearest:
        metadata = item.get("metadata") or {}
        reference_problems = _as_problem_list(metadata.get("problems", metadata.get("problem", [])))
        processed_path = str(item.get("processed_image_path", ""))
        summary.append(
            {
                "global_rank": int(item["rank"]),
                "reference_id": str(item.get("id", "")),
                "reference_study_id": str(item.get("study_id", "")),
                "rad_dino_cosine_similarity": round(float(item["cosine_similarity"]), 6),
                "densenet_feature_map_similarity": (
                    round(float(local_scores[processed_path]), 6) if processed_path in local_scores else None
                ),
                "reference_problems": reference_problems,
            }
        )
    return summary


def classifier_retriever_prompt(
    probabilities: Dict[str, float], nearest_cases: list[Dict[str, Any]], selected_labels: tuple[str, ...]
) -> Dict[str, str]:
    scores = "\n".join(f"- {label}: {probabilities[label]:.4f}" for label in MAIN_LABELS)
    references = "\n".join(
        "- Case {global_rank}: RAD-DINO cosine={rad_dino_cosine_similarity:.4f}; "
        "DenseNet feature-map similarity={feature}; reference problems={problems}".format(
            global_rank=item["global_rank"],
            rad_dino_cosine_similarity=item["rad_dino_cosine_similarity"],
            feature=(f"{item['densenet_feature_map_similarity']:.4f}" if item["densenet_feature_map_similarity"] is not None else "unavailable"),
            problems=(", ".join(item["reference_problems"]) or "not recorded"),
        )
        for item in nearest_cases
    ) or "- No retrieved cases were available."
    return {
        "system": (
            "You are a radiologist reviewing a chest X-ray. The attached image is primary evidence. "
            "Classifier probabilities and nearest-case summaries are auxiliary, unverified context. "
            "Do not treat a reference problem as a finding in the current image. "
            "For problems, use only classifier-passed labels: " + (", ".join(selected_labels) or "none") + ", or exactly [normal] when no target pathology is present. "
            "Place non-target observations only in findings. Return only the requested JSON output; do not mention the classifier, retrieval, or experiment."
        ),
        "user": (
            "Analyze the attached chest X-ray directly.\n\n"
            "Classifier probabilities:\n" + scores +
            "\n\nFive globally nearest retrieved chest-X-ray cases (context only):\n" + references +
            "\n\nProduce concise problems, findings, and impression for the attached image."
        ),
    }


def parse_prediction(text: str, selected_labels: tuple[str, ...] = TARGET_PROBLEM_LABELS) -> Optional[Dict[str, Any]]:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    return validate_radiology_prediction(parsed, selected_labels)


def make_retriever(device: torch.device) -> FAISSImageRetriever:
    config = load_config(str(ROOT / "config" / "config.yaml"))
    retrieval = config["retrieval"]
    model = config["model"]

    def project_path(value: str) -> str:
        path = Path(value)
        return str(path if path.is_absolute() else ROOT / path)

    return FAISSImageRetriever(
        index_path=project_path(retrieval["index_path"]),
        metadata_path=project_path(retrieval["metadata_path"]),
        processed_dir=project_path(retrieval["processed_dir"]) if retrieval.get("processed_dir") else None,
        device=str(device),
        model_name=DEFAULT_MODEL_NAME,
        projection_checkpoint=(
            project_path(model["pathology_projection_checkpoint"])
            if model.get("pathology_projection_checkpoint") else None
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline B: classifier, five nearest retrieval cases, and Luna vision.")
    parser.add_argument("--image-path", required=True, type=Path)
    parser.add_argument("--study-id", help="IU-Xray uid excluded from retrieval and saved for offline evaluation")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/baselines/classifier_retriever_vision")
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
    retriever = make_retriever(device)
    nearest_cases = summarize_nearest_cases(
        args.image_path, retriever, classifier.backbone, device, exclude_study_id=args.study_id
    )
    del retriever, classifier
    if device.type == "cuda":
        torch.cuda.empty_cache()

    prompt = classifier_retriever_prompt(probabilities, nearest_cases, selected_labels)
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
        "baseline": "classifier_retriever_vision_unstructured_input",
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
        "five_nearest_cases": nearest_cases,
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
