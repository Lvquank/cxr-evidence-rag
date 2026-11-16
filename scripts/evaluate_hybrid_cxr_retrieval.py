"""Evaluate RAD-DINO retrieval with bounded DenseNet feature-map reranking.

Relevance is derived only from compact metadata ``support`` vectors, never from
classifier probabilities.  The script evaluates study-level rankings using
weighted-Jaccard relevance and writes one compact JSON report.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.predict_chexpert_6labels import load_model
from src.retriever.densenet_local_reranker import DenseNetLocalFeatureReranker


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_artifacts(directory: Path) -> tuple[np.ndarray, list[dict], np.ndarray]:
    embeddings = np.load(directory / "embeddings.npy").astype(np.float32)
    rows = read_jsonl(directory / "metadata.jsonl")
    ids = (directory / "ids.txt").read_text(encoding="utf-8").splitlines()
    if embeddings.ndim != 2 or len(embeddings) != len(rows) or len(ids) != len(rows):
        raise ValueError(f"Artifacts in {directory} have inconsistent row counts.")
    if ids != [str(row.get("id", "")) for row in rows]:
        raise ValueError(f"ids.txt and metadata.jsonl are not in the same order in {directory}.")
    embeddings /= np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-8)
    support = np.asarray([row.get("support", []) for row in rows], dtype=np.float32)
    if support.shape != (len(rows), 6):
        raise ValueError(f"Metadata in {directory} must contain six-value support vectors.")
    return embeddings, rows, support


def weighted_jaccard(query_support: np.ndarray, gallery_support: np.ndarray) -> np.ndarray:
    numerator = np.minimum(query_support, gallery_support).sum(axis=1)
    denominator = np.maximum(query_support, gallery_support).sum(axis=1)
    return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)


def normal_relevance(gallery_support: np.ndarray) -> np.ndarray:
    """Define normal-to-normal relevance; all-zero Jaccard is otherwise undefined."""
    return np.all(gallery_support == 0.0, axis=1).astype(np.float32)


def study_key(row: dict) -> tuple[str, str]:
    dataset = str(row.get("dataset", "") or "").strip()
    study_id = str(row.get("study_id", row.get("uid", "")) or "").strip()
    return dataset, study_id or f"image:{row.get('id', '')}"


def image_path(row: dict, processed_dir: Path) -> Path:
    value = row.get("path", row.get("image_path", ""))
    path = Path(str(value))
    return path if path.is_absolute() else processed_dir / path


def dcg(relevance: np.ndarray) -> float:
    if not len(relevance):
        return 0.0
    return float((relevance / np.log2(np.arange(2, len(relevance) + 2))).sum())


def precision_at_k(binary_relevance: np.ndarray, k: int) -> float:
    values = binary_relevance[:k]
    return float(values.mean()) if len(values) else float("nan")


def average_precision_at_k(binary_relevance: np.ndarray, total_relevant: int, k: int) -> float:
    if total_relevant == 0:
        return float("nan")
    values = binary_relevance[:k]
    hits = np.cumsum(values)
    return float((hits / np.arange(1, len(values) + 1) * values).sum() / min(total_relevant, k))


def ndcg_at_k(graded_relevance: np.ndarray, k: int) -> float:
    ideal_dcg = dcg(np.sort(graded_relevance)[::-1][:k])
    return dcg(graded_relevance[:k]) / ideal_dcg if ideal_dcg else float("nan")


def pathology_delta(graded_relevance: np.ndarray, ranked_scores: np.ndarray, threshold: float) -> float:
    support_scores = ranked_scores[graded_relevance >= threshold]
    zero_overlap_scores = ranked_scores[graded_relevance == 0.0]
    if not len(support_scores) or not len(zero_overlap_scores):
        return float("nan")
    return float(support_scores.mean() - zero_overlap_scores.mean())


def margin_at_k(graded_relevance: np.ndarray, ranked_scores: np.ndarray, threshold: float, k: int) -> float:
    support_scores = ranked_scores[graded_relevance >= threshold][:k]
    zero_overlap_scores = ranked_scores[graded_relevance == 0.0][:k]
    if not len(support_scores) or not len(zero_overlap_scores):
        return float("nan")
    return float(support_scores.mean() - zero_overlap_scores.mean())


def mean_or_none(values: list[float]) -> float | None:
    valid = [value for value in values if not math.isnan(value)]
    return float(np.mean(valid)) if valid else None


def json_safe(value):
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate hybrid RAD-DINO plus DenseNet-local CXR retrieval from embedding artifacts."
    )
    parser.add_argument("--gallery_dir", required=True, help="Directory containing embeddings.npy, ids.txt and metadata.jsonl")
    parser.add_argument("--processed_dir", required=True, help="Root containing gallery image paths from metadata")
    parser.add_argument("--query_dir", help="Optional query embedding artifact directory; defaults to --gallery_dir")
    parser.add_argument("--query_processed_dir", help="Optional root for query images; defaults to --processed_dir")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--classifier_checkpoint", default="model/best_E2_relation_aware_weighted_bce.pt")
    parser.add_argument("--k", type=int, nargs="+", default=[3, 5, 10, 20])
    parser.add_argument("--relevance_threshold", type=float, default=0.5)
    parser.add_argument("--local_rerank_candidates", type=int, default=50, help="Top study-level RAD-DINO candidates rescored with DenseNet")
    parser.add_argument("--lambda_global", type=float, default=0.9, help="Global RAD-DINO weight; DenseNet local weight is 1 - lambda_global")
    parser.add_argument("--local_rerank_batch_size", type=int, default=16)
    parser.add_argument("--disable_densenet_local_rerank", action="store_true")
    parser.add_argument("--max_queries", type=int, default=0, help="0 evaluates all eligible queries; otherwise sample this many")
    parser.add_argument(
        "--query-manifest",
        help="Optional frozen manifest with requests[].image_id; evaluates those exact query images, including normal studies.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include_per_query", action="store_true", help="Include compact query-level diagnostics in the JSON")
    parser.add_argument(
        "--similarity_cache_json",
        help=(
            "Optional path for a reusable cache of global and DenseNet-local scores. "
            "The cache contains the global top-N candidate pool per query and can be "
            "used by sweep_retrieval_lambda.py without recomputing either model."
        ),
    )
    parser.add_argument(
        "--cache_candidates",
        type=int,
        default=50,
        help="Global-ranked candidates saved per query when --similarity_cache_json is used.",
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:0")
    args = parser.parse_args()

    if not args.k or any(value < 1 for value in args.k):
        parser.error("--k must contain positive values.")
    if not 0.0 < args.relevance_threshold <= 1.0:
        parser.error("--relevance_threshold must be in (0, 1].")
    if args.local_rerank_candidates < 1 or args.local_rerank_batch_size < 1:
        parser.error("--local_rerank_candidates and --local_rerank_batch_size must be positive.")
    if args.max_queries < 0:
        parser.error("--max_queries must be non-negative.")
    if args.cache_candidates < max(args.k):
        parser.error("--cache_candidates must be at least the largest value in --k.")
    if not 0.0 <= args.lambda_global <= 1.0:
        parser.error("--lambda_global must be between 0 and 1.")

    gallery_dir = Path(args.gallery_dir)
    query_dir = Path(args.query_dir) if args.query_dir else gallery_dir
    processed_dir = Path(args.processed_dir)
    query_processed_dir = Path(args.query_processed_dir) if args.query_processed_dir else processed_dir
    gallery_embeddings, gallery_rows, gallery_support = load_artifacts(gallery_dir)
    if query_dir == gallery_dir:
        query_embeddings, query_rows, query_support = gallery_embeddings, gallery_rows, gallery_support
    else:
        query_embeddings, query_rows, query_support = load_artifacts(query_dir)
    if query_embeddings.shape[1] != gallery_embeddings.shape[1]:
        raise ValueError("Query and gallery embedding dimensions differ.")

    device = resolve_device(args.device)
    reranker = None
    if not args.disable_densenet_local_rerank:
        checkpoint = Path(args.classifier_checkpoint)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Classifier checkpoint not found: {checkpoint}")
        classifier, _ = load_model(checkpoint, device)
        reranker = DenseNetLocalFeatureReranker(classifier.backbone, device, args.local_rerank_batch_size)
        del classifier
        if device.type == "cuda":
            torch.cuda.empty_cache()

    gallery_studies = [study_key(row) for row in gallery_rows]
    query_studies = [study_key(row) for row in query_rows]
    eligible_all = np.flatnonzero(query_support.sum(axis=1) > 0)
    if args.query_manifest:
        manifest = json.loads(Path(args.query_manifest).read_text(encoding="utf-8"))
        requested_ids = [str(item.get("image_id", "")) for item in manifest.get("requests", [])]
        if not requested_ids or any(not value for value in requested_ids):
            raise ValueError("--query-manifest must contain requests with image_id values.")
        index_by_id = {str(row.get("id", "")): index for index, row in enumerate(query_rows)}
        missing_ids = [value for value in requested_ids if value not in index_by_id]
        if missing_ids:
            raise ValueError(f"{len(missing_ids)} manifest image IDs are absent from query artifacts; first: {missing_ids[0]}")
        eligible = np.asarray([index_by_id[value] for value in requested_ids], dtype=np.int64)
    else:
        eligible = eligible_all
        if args.max_queries and args.max_queries < len(eligible):
            eligible = np.sort(np.random.default_rng(args.seed).choice(eligible, size=args.max_queries, replace=False))

    metric_values = {
        **{f"Precision@{k}": [] for k in args.k},
        **{f"mAP@{k}": [] for k in args.k},
        **{f"nDCG@{k}": [] for k in args.k},
        **{f"Delta_path@{k}": [] for k in args.k},
        **{f"M@{k}": [] for k in args.k},
        "Delta_path": [],
    }
    records = []
    similarity_cache_records = []
    skipped_missing_query_image = 0
    local_scores_applied = 0

    for query_index in eligible:
        query_image = image_path(query_rows[query_index], query_processed_dir)
        if reranker is not None and not query_image.is_file():
            skipped_missing_query_image += 1
            continue

        base_scores = gallery_embeddings @ query_embeddings[query_index]
        relevance = (
            normal_relevance(gallery_support)
            if float(query_support[query_index].sum()) == 0.0
            else weighted_jaccard(query_support[query_index], gallery_support)
        )
        base_order = np.argsort(-base_scores, kind="stable")
        ranked_indices = []
        seen_studies = set()
        for candidate_index in base_order:
            candidate_study = gallery_studies[candidate_index]
            if candidate_study == query_studies[query_index] or candidate_study in seen_studies:
                continue
            seen_studies.add(candidate_study)
            ranked_indices.append(candidate_index)
        ranked_indices = np.asarray(ranked_indices, dtype=np.int64)
        ranking_scores = base_scores[ranked_indices].copy()
        local_scores_by_position: dict[int, float] = {}

        if reranker is not None:
            local_positions = []
            candidate_paths = []
            for position, candidate_index in enumerate(ranked_indices[:args.local_rerank_candidates]):
                path = image_path(gallery_rows[candidate_index], processed_dir)
                if path.is_file():
                    local_positions.append(position)
                    candidate_paths.append(str(path))
            local_scores = reranker.score_paths(str(query_image), candidate_paths)
            for position, path in zip(local_positions, candidate_paths):
                local_score = local_scores.get(path)
                if local_score is None:
                    continue
                base_score = float(ranking_scores[position])
                local_scores_by_position[position] = float(local_score)
                ranking_scores[position] = (
                    args.lambda_global * base_score
                    + (1.0 - args.lambda_global) * local_score
                )
                local_scores_applied += 1
            reranker.clear_cache()

        if args.similarity_cache_json:
            cache_limit = min(args.cache_candidates, len(ranked_indices))
            cache_relevance = relevance[ranked_indices]
            cache_binary = (cache_relevance >= args.relevance_threshold).astype(np.float32)
            similarity_cache_records.append({
                "query_id": str(query_rows[query_index].get("id", "")),
                "query_study_id": query_studies[query_index][1],
                "total_relevant_in_gallery": int(cache_binary.sum()),
                "ideal_dcg": {
                    str(k): dcg(np.sort(cache_relevance)[::-1][:k]) for k in args.k
                },
                "candidates": [
                    {
                        "global_rank": position + 1,
                        "id": str(gallery_rows[candidate_index].get("id", "")),
                        "study_id": gallery_studies[candidate_index][1],
                        "global_similarity": float(base_scores[candidate_index]),
                        "local_similarity": local_scores_by_position.get(position),
                        "relevance": float(cache_relevance[position]),
                    }
                    for position, candidate_index in enumerate(ranked_indices[:cache_limit])
                ],
            })

        order = np.argsort(-ranking_scores, kind="stable")
        ranked_indices = ranked_indices[order]
        ranked_scores = ranking_scores[order]
        ranked_relevance = relevance[ranked_indices]
        binary_relevance = (ranked_relevance >= args.relevance_threshold).astype(np.float32)
        total_relevant = int(binary_relevance.sum())
        if total_relevant == 0:
            continue

        record = {
            "query_id": str(query_rows[query_index].get("id", "")),
            "query_study_id": query_studies[query_index][1],
            "relevant_studies": total_relevant,
        }
        for k in args.k:
            precision = precision_at_k(binary_relevance, k)
            ap = average_precision_at_k(binary_relevance, total_relevant, k)
            ndcg = ndcg_at_k(ranked_relevance, k)
            delta_at_k = pathology_delta(ranked_relevance[:k], ranked_scores[:k], args.relevance_threshold)
            margin = margin_at_k(ranked_relevance, ranked_scores, args.relevance_threshold, k)
            metric_values[f"Precision@{k}"].append(precision)
            metric_values[f"mAP@{k}"].append(ap)
            metric_values[f"nDCG@{k}"].append(ndcg)
            metric_values[f"Delta_path@{k}"].append(delta_at_k)
            metric_values[f"M@{k}"].append(margin)
            record.update({
                f"Precision@{k}": precision,
                f"AP@{k}": ap,
                f"nDCG@{k}": ndcg,
                f"Delta_path@{k}": delta_at_k,
                f"M@{k}": margin,
            })
        delta = pathology_delta(ranked_relevance, ranked_scores, args.relevance_threshold)
        metric_values["Delta_path"].append(delta)
        record["Delta_path"] = delta
        records.append(record)

    report = {
        "schema_version": 1,
        "protocol": {
            "relevance": "weighted_jaccard(query_support, gallery_support)",
            "normal_relevance": "1 for gallery studies whose six-value support vector is all zero, otherwise 0.",
            "binary_relevance_threshold": args.relevance_threshold,
            "same_study_excluded": True,
            "deduplicate_by": "dataset + study_id",
            "ranking_score": (
                "lambda_global * rad_dino_cosine + (1 - lambda_global) * densenet_local_similarity"
                if reranker is not None else "rad_dino_cosine"
            ),
            "distance": "1 - ranking_score (smaller is closer)",
            "local_rerank_candidates": args.local_rerank_candidates if reranker is not None else 0,
            "lambda_global": args.lambda_global if reranker is not None else 1.0,
            "lambda_local": 1.0 - args.lambda_global if reranker is not None else 0.0,
            "delta_path_definition": "mean(score | relevance >= threshold) - mean(score | relevance == 0)",
            "m_at_k_definition": "mean(top-K score among relevant) - mean(top-K score among zero-overlap)",
        },
        "artifacts": {
            "gallery_dir": str(gallery_dir.resolve()),
            "query_dir": str(query_dir.resolve()),
            "embedding_dimension": int(gallery_embeddings.shape[1]),
        },
        "counts": {
            "gallery_rows": len(gallery_rows),
            "queries_with_support": int(len(eligible_all)),
            "selected_queries": int(len(eligible)),
            "evaluated_queries": len(records),
            "skipped_missing_query_image": skipped_missing_query_image,
            "densenet_local_scores_applied": local_scores_applied,
        },
        "metrics": {name: mean_or_none(values) for name, values in metric_values.items()},
        "caveat": "support=0 denotes no observed target support in metadata, not a clinically confirmed negative.",
    }
    if args.include_per_query:
        report["per_query"] = records

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(json_safe(report), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.similarity_cache_json:
        cache_path = Path(args.similarity_cache_json)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(json_safe({
            "schema_version": 1,
            "protocol": {
                "candidate_pool": "global top-N unique studies after same-study removal",
                "same_study_excluded": True,
                "deduplicate_by": "dataset + study_id",
                "global_score": "RAD-DINO cosine similarity",
                "local_score": "DenseNet symmetric feature-map similarity",
                "production_score": "lambda_global * global + (1 - lambda_global) * local",
                "lambda_global": args.lambda_global if reranker is not None else 1.0,
                "lambda_local": 1.0 - args.lambda_global if reranker is not None else 0.0,
                "local_rerank_candidates": args.local_rerank_candidates if reranker is not None else 0,
                "cache_candidates": args.cache_candidates,
                "relevance": "weighted_jaccard(query_support, gallery_support); normal queries use all-zero-support matching",
                "binary_relevance_threshold": args.relevance_threshold,
            },
            "queries": similarity_cache_records,
        }), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(json_safe({"output_json": str(output_path), "counts": report["counts"], "metrics": report["metrics"]}), indent=2))


if __name__ == "__main__":
    main()
