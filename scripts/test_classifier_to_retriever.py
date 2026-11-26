"""Classify a chest X-ray, retrieve global and label-bank evidence, and write JSON.

The DenseNet classifier outputs all six raw probabilities without thresholding.
The RAD-DINO retriever always returns a global morphology neighborhood. For
selected labels, it unions global, positive-bank and non-support-bank candidates;
classifier probabilities route labels but do not rank reference images.
"""

import argparse
import json
import sys
from pathlib import Path

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.predict_chexpert_6labels import MAIN_LABELS, load_model, predict
from src.classifier_thresholds import DEFAULT_THRESHOLDS_PATH, load_classifier_thresholds, threshold_snapshot
from src.config import load_config
from src.embeddings.encoder import DEFAULT_MODEL_NAME
from src.retriever.densenet_local_reranker import DenseNetLocalFeatureReranker
from src.retriever.faiss_retriever import FAISSImageRetriever
from src.retriever.pathology_support import CHEXPERT_LABELS


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def compact_score(value: object) -> float:
    """Keep JSON scores readable without changing retrieval calculations."""
    return round(float(value), 6)


def compact_candidate(candidate: dict, include_label_support: bool = False) -> dict:
    """Return the evidence fields needed by downstream consumers and reviewers."""
    metadata = candidate.get("metadata") or {}
    result = {
        key: candidate[key]
        for key in ("rank", "id", "study_id", "image_path")
        if candidate.get(key) is not None
    }
    for key in ("cosine_similarity", "base_rank_score", "densenet_local_similarity", "final_rank_score"):
        if candidate.get(key) is not None:
            result[key] = compact_score(candidate[key])
    if include_label_support and candidate.get("label_support") is not None:
        result["label_support"] = compact_score(candidate["label_support"])
    # Keep the clinical annotation schema stable for every evidence image.
    # Empty lists explicitly distinguish "no annotation" from a missing field.
    result["mesh"] = metadata.get("mesh") or []
    result["problems"] = metadata.get("problems") or []
    return result


def compact_label_evidence(evidence: dict) -> dict:
    """Remove retrieval internals while retaining evidence quality and examples."""
    quality = evidence.get("retrieval_quality") or {}
    margins = (evidence.get("retrieval_statistics") or {}).get("margins") or {}
    return {
        "label": evidence["label"],
        "classifier_probability": compact_score(evidence["classifier_probability"]),
        "evidence_status": evidence["evidence_status"],
        "evidence_available": evidence["evidence_available"],
        "margins": {
            metric: compact_score(values["margin"])
            for metric, values in margins.items()
            if values.get("margin") is not None
        },
        "quality_checks": {
            "margin_threshold": quality.get("margin_threshold"),
            "m3_same_sign_as_m5": quality.get("margin_3_same_sign_as_margin_5"),
            "m10_strong_reversal": quality.get("margin_10_strong_reversal"),
            "m20_warning": quality.get("margin_20_warning"),
        },
        "supporting_evidence": [
            compact_candidate(candidate, include_label_support=True)
            for candidate in evidence.get("supporting_evidence", [])
        ],
        "matched_non_support_evidence": [
            compact_candidate(candidate)
            for candidate in evidence.get("matched_non_support_evidence", [])
        ],
    }


def main() -> None:
    cfg = load_config()
    retrieval_cfg = cfg.get("retrieval", {})
    model_cfg = cfg.get("model", {})
    parser = argparse.ArgumentParser(
        description="Run DenseNet raw-probability classification followed by label-specific RAD-DINO retrieval."
    )
    parser.add_argument("--image_path", required=True)
    parser.add_argument("--output_json", default="data/classifier_retriever_evidence.json")
    parser.add_argument(
        "--exclude-study-id",
        action="append",
        default=[],
        help="Study ID temporarily excluded from every retrieval bank; can be repeated.",
    )
    parser.add_argument("--classifier_checkpoint", default="model/best_E2_relation_aware_weighted_bce.pt")
    parser.add_argument("--thresholds-json", default=str(DEFAULT_THRESHOLDS_PATH), help="Frozen per-label DenseNet development thresholds")
    parser.add_argument("--index_path", default=retrieval_cfg.get("index_path"))
    parser.add_argument("--metadata_path", default=retrieval_cfg.get("metadata_path"))
    parser.add_argument("--processed_dir", default=retrieval_cfg.get("processed_dir"))
    parser.add_argument("--projection_checkpoint", default=model_cfg.get("pathology_projection_checkpoint"))
    parser.add_argument("--candidate_k", type=int, default=200)
    parser.add_argument("--top_k", type=int, default=3, help="Supporting and matched non-support images returned per selected label")
    parser.add_argument("--common_top_k", type=int, default=5, help="Nearest common evidence images returned")
    parser.add_argument("--bank_candidate_k", type=int, default=50, help="Candidates retrieved from each logical label bank")
    parser.add_argument("--statistics_k", type=int, default=20, help="Per-bank candidates used for retrieval statistics")
    parser.add_argument(
        "--evidence_threshold",
        type=float,
        default=None,
        help="Optional legacy common threshold override for every label; defaults to the frozen adaptive artifact.",
    )
    parser.add_argument(
        "--label_thresholds",
        type=float,
        nargs=len(CHEXPERT_LABELS),
        metavar="THRESHOLD",
        help=(
            "Six per-label evidence thresholds in this order: "
            + ", ".join(CHEXPERT_LABELS)
            + ". Overrides --evidence_threshold."
        ),
    )
    parser.add_argument("--support_similarity_threshold", type=float, default=0.10, help="Minimum cosine similarity for supporting evidence")
    parser.add_argument("--quality_margin_threshold", type=float, default=0.02, help="Minimum mean support/non-support gap for a directional status")
    parser.add_argument("--margin_10_reversal_ratio", type=float, default=0.5, help="M@10 opposite-sign magnitude allowed relative to |M@5|")
    parser.add_argument("--disable_densenet_local_rerank", action="store_true", help="Skip DenseNet feature-map reranking after ANN retrieval")
    parser.add_argument("--local_rerank_k", type=int, default=10, help="Candidates per branch reranked by DenseNet local features")
    parser.add_argument("--lambda_global", type=float, default=0.9, help="Global RAD-DINO weight in final score; DenseNet local weight is 1 - lambda_global")
    parser.add_argument("--local_rerank_batch_size", type=int, default=16, help="DenseNet feature-map batch size")
    parser.add_argument("--full_output", action="store_true", help="Write all retrieval internals and full metadata for debugging")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:0")
    args = parser.parse_args()

    if tuple(MAIN_LABELS) != tuple(CHEXPERT_LABELS):
        raise RuntimeError("DenseNet classifier label order differs from retriever support label order.")
    if args.candidate_k < max(args.top_k, args.common_top_k):
        parser.error("--candidate_k must be at least --top_k and --common_top_k.")
    if args.bank_candidate_k < 1 or args.statistics_k < 20:
        parser.error("--bank_candidate_k must be positive and --statistics_k must be at least 20.")
    if args.evidence_threshold is not None and not 0.0 <= args.evidence_threshold <= 1.0:
        parser.error("--evidence_threshold must be between 0 and 1.")
    if args.label_thresholds and any(not 0.0 <= value <= 1.0 for value in args.label_thresholds):
        parser.error("Every value in --label_thresholds must be between 0 and 1.")
    if not 0.0 <= args.support_similarity_threshold <= 1.0:
        parser.error("--support_similarity_threshold must be between 0 and 1.")
    if args.quality_margin_threshold < 0.0:
        parser.error("--quality_margin_threshold must be non-negative.")
    if args.margin_10_reversal_ratio < 0.0:
        parser.error("--margin_10_reversal_ratio must be non-negative.")
    if args.local_rerank_k < 1 or args.local_rerank_batch_size < 1:
        parser.error("--local_rerank_k and --local_rerank_batch_size must be at least 1.")
    if not 0.0 <= args.lambda_global <= 1.0:
        parser.error("--lambda_global must be between 0 and 1.")
    image_path = Path(args.image_path)
    classifier_checkpoint = Path(args.classifier_checkpoint)
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not classifier_checkpoint.is_file():
        raise FileNotFoundError(f"Classifier checkpoint not found: {classifier_checkpoint}")

    device = resolve_device(args.device)
    classifier, _ = load_model(classifier_checkpoint, device)
    raw_probabilities = predict(image_path, classifier, device)
    threshold_artifact = None
    label_thresholds = (
        dict(zip(CHEXPERT_LABELS, args.label_thresholds))
        if args.label_thresholds
        else ({label: args.evidence_threshold for label in CHEXPERT_LABELS}
              if args.evidence_threshold is not None else load_classifier_thresholds(args.thresholds_json)[0])
    )
    if not args.label_thresholds and args.evidence_threshold is None:
        _, threshold_artifact = load_classifier_thresholds(args.thresholds_json)
    threshold_metadata = threshold_snapshot(
        label_thresholds, threshold_artifact["version"] if threshold_artifact else "cli_override",
        args.thresholds_json if threshold_artifact else None,
    )
    selected_labels = tuple(
        label for label in CHEXPERT_LABELS
        if raw_probabilities[label] >= label_thresholds[label]
    )

    local_reranker = None
    if selected_labels and not args.disable_densenet_local_rerank:
        local_reranker = DenseNetLocalFeatureReranker(
            classifier.backbone,
            device=device,
            batch_size=args.local_rerank_batch_size,
        )
    # Keep only the DenseNet spatial backbone when local reranking is enabled.
    del classifier
    if device.type == "cuda":
        torch.cuda.empty_cache()
    retriever = FAISSImageRetriever(
        index_path=args.index_path,
        metadata_path=args.metadata_path,
        processed_dir=args.processed_dir,
        device=str(device),
        model_name=DEFAULT_MODEL_NAME,
        projection_checkpoint=args.projection_checkpoint,
    )
    retrieval = retriever.search_label_evidence(
        str(image_path),
        raw_probabilities=raw_probabilities,
        labels=selected_labels,
        top_k=args.top_k,
        common_top_k=args.common_top_k,
        candidate_k=args.candidate_k,
        bank_candidate_k=args.bank_candidate_k,
        statistics_k=args.statistics_k,
        support_similarity_threshold=args.support_similarity_threshold,
        quality_margin_threshold=args.quality_margin_threshold,
        margin_10_reversal_ratio=args.margin_10_reversal_ratio,
        local_reranker=local_reranker,
        local_rerank_k=args.local_rerank_k,
        lambda_global=args.lambda_global,
        exclude_study_ids=args.exclude_study_id,
    )

    full_result = {
        "schema_version": 1,
        "image_path": str(image_path.resolve()),
        "classifier": {
            "model": "local/chexpert-relation-aware-densenet121",
            "checkpoint": str(classifier_checkpoint.resolve()),
            "raw_probabilities": {label: float(raw_probabilities[label]) for label in CHEXPERT_LABELS},
            "probabilities_thresholded": False,
            "selected_labels": list(selected_labels),
            **threshold_metadata,
        },
        "retriever": {
            "model": retriever.model.model_name,
            "projection_checkpoint": (
                str(Path(args.projection_checkpoint).resolve())
                if args.projection_checkpoint else None
            ),
            "candidate_k": args.candidate_k,
            "common_top_k": args.common_top_k,
            "top_k_per_label": args.top_k,
            "bank_candidate_k": args.bank_candidate_k,
            "statistics_k": args.statistics_k,
            "primary_evidence_metric": "M@5",
            "margin_3_requirement": "M@3 must have the same non-zero sign as M@5.",
            "margin_10_requirement": "M@10 cannot reverse with magnitude >= margin_10_reversal_ratio * |M@5|.",
            "margin_20_usage": "Warning only; it does not change evidence_status.",
            "margin_10_reversal_ratio": args.margin_10_reversal_ratio,
            "label_thresholds": label_thresholds,
            "selected_labels": list(selected_labels),
            "excluded_study_ids": list(args.exclude_study_id),
            "no_label_specific_retrieval_reason": None if selected_labels else "No raw classifier probability met its label threshold.",
            "classifier_probability_usage": "Label routing only; excluded from reference ranking.",
            "support_rerank_formula": "cosine_similarity",
            "matched_non_support_rank_formula": "cosine_similarity",
            "densenet_local_reranker": {
                "enabled": local_reranker is not None,
                "feature_map": "DenseNet121 backbone.features before global pooling",
                "similarity": "symmetric maximum patch cosine similarity",
                "candidate_k_per_branch": args.local_rerank_k,
                "lambda_global": args.lambda_global,
                "lambda_local": 1.0 - args.lambda_global,
                "final_score_formula": "lambda_global * base_rank_score + (1 - lambda_global) * densenet_local_similarity",
            },
            **retrieval,
        },
    }
    result = full_result if args.full_output else {
        "schema_version": 2,
        "image_path": str(image_path.resolve()),
        "classifier": {
            "raw_probabilities": {
                label: compact_score(raw_probabilities[label])
                for label in CHEXPERT_LABELS
            },
            "selected_labels": list(selected_labels),
            **threshold_metadata,
        },
        "retriever": {
            "model": retriever.model.model_name,
            "local_rerank_enabled": local_reranker is not None,
            "lambda_global": args.lambda_global,
            "excluded_study_ids": list(args.exclude_study_id),
            "common_evidence": [
                compact_candidate(candidate)
                for candidate in retrieval["common_evidence"]
            ],
            "label_evidence": [
                compact_label_evidence(evidence)
                for evidence in retrieval["label_evidence"]
            ],
        },
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    available = [item["label"] for item in retrieval["label_evidence"] if item["evidence_available"]]
    print(json.dumps({"output_json": str(output_path), "raw_probabilities": result["classifier"]["raw_probabilities"], "evidence_labels": available}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
