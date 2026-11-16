"""Evaluate pathology-aware image retrieval from saved embedding artifacts.

The evaluator uses compact six-label ``support`` vectors as weak ground truth.
It excludes same-study images and deduplicates gallery results by study before
calculating Precision@K, mAP@K, nDCG@K, and Delta_path.
"""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_artifacts(directory: Path):
    embeddings = np.load(directory / "embeddings.npy").astype(np.float32)
    metadata = read_jsonl(directory / "metadata.jsonl")
    ids = (directory / "ids.txt").read_text(encoding="utf-8").splitlines()
    if embeddings.ndim != 2 or len(embeddings) != len(metadata) or len(ids) != len(metadata):
        raise ValueError(f"Artifacts in {directory} have inconsistent row counts.")
    metadata_ids = [str(row.get("id", "")) for row in metadata]
    if ids != metadata_ids:
        raise ValueError(f"ids.txt and metadata.jsonl are not in the same order in {directory}.")
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.maximum(norms, 1e-8)
    support = np.asarray([row.get("support", []) for row in metadata], dtype=np.float32)
    if support.shape != (len(metadata), 6):
        raise ValueError(f"Metadata in {directory} must contain six-value support vectors.")
    return embeddings, metadata, support


def weighted_jaccard(query_support: np.ndarray, gallery_support: np.ndarray) -> np.ndarray:
    numerator = np.minimum(query_support, gallery_support).sum(axis=1)
    denominator = np.maximum(query_support, gallery_support).sum(axis=1)
    return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)


def study_key(row: dict) -> tuple[str, str]:
    dataset = str(row.get("dataset", "") or "").strip()
    study_id = str(row.get("study_id", row.get("uid", "")) or "").strip()
    if study_id:
        return dataset, study_id
    return dataset, f"image:{row.get('id', '')}"


def dcg(relevance: np.ndarray) -> float:
    if not len(relevance):
        return 0.0
    discounts = np.log2(np.arange(2, len(relevance) + 2))
    return float((relevance / discounts).sum())


def average_precision_at_k(binary_relevance: np.ndarray, total_relevant: int, k: int) -> float:
    if total_relevant == 0:
        return float("nan")
    binary_relevance = binary_relevance[:k]
    hits = np.cumsum(binary_relevance)
    precision_at_rank = hits / np.arange(1, len(binary_relevance) + 1)
    return float((precision_at_rank * binary_relevance).sum() / min(total_relevant, k))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate pathology-aware retrieval metrics from embedding artifacts.")
    parser.add_argument("--gallery_dir", required=True, help="Embedding artifact directory used as gallery")
    parser.add_argument("--query_dir", help="Optional different embedding artifact directory for cross-domain queries")
    parser.add_argument("--output_dir", required=True, help="Directory for summary.json and per_query.csv")
    parser.add_argument("--k", type=int, nargs="+", default=[5, 10])
    parser.add_argument("--relevance_threshold", type=float, default=0.5)
    args = parser.parse_args()

    if not 0 < args.relevance_threshold <= 1:
        raise ValueError("--relevance_threshold must be in (0, 1].")
    if not args.k or any(value < 1 for value in args.k):
        raise ValueError("--k must contain positive values.")

    gallery_dir = Path(args.gallery_dir)
    query_dir = Path(args.query_dir) if args.query_dir else gallery_dir
    gallery_embeddings, gallery_rows, gallery_support = load_artifacts(gallery_dir)
    query_embeddings, query_rows, query_support = (
        (gallery_embeddings, gallery_rows, gallery_support)
        if query_dir == gallery_dir
        else load_artifacts(query_dir)
    )
    if query_embeddings.shape[1] != gallery_embeddings.shape[1]:
        raise ValueError("Query and gallery embedding dimensions differ.")

    gallery_studies = [study_key(row) for row in gallery_rows]
    query_studies = [study_key(row) for row in query_rows]
    max_k = max(args.k)
    records = []
    precision_scores = {k: [] for k in args.k}
    ap_scores = {k: [] for k in args.k}
    ndcg_scores = {k: [] for k in args.k}
    delta_scores = []

    eligible_queries = np.flatnonzero(query_support.sum(axis=1) > 0)
    for query_index in eligible_queries:
        similarities = gallery_embeddings @ query_embeddings[query_index]
        relevance = weighted_jaccard(query_support[query_index], gallery_support)
        order = np.argsort(-similarities, kind="stable")

        ranked_indices = []
        seen_studies = set()
        for candidate_index in order:
            candidate_study = gallery_studies[candidate_index]
            if candidate_study == query_studies[query_index]:
                continue
            if candidate_study in seen_studies:
                continue
            seen_studies.add(candidate_study)
            ranked_indices.append(candidate_index)

        ranked_indices = np.asarray(ranked_indices, dtype=np.int64)
        ranked_relevance = relevance[ranked_indices]
        ranked_similarity = similarities[ranked_indices]
        binary_relevance = (ranked_relevance >= args.relevance_threshold).astype(np.float32)
        total_relevant = int(binary_relevance.sum())
        if total_relevant == 0:
            continue

        record = {
            "query_id": query_rows[query_index].get("id", ""),
            "query_study_id": query_studies[query_index][1],
            "relevant_studies": total_relevant,
        }
        for k in args.k:
            limited_binary = binary_relevance[:k]
            precision = float(limited_binary.mean()) if len(limited_binary) else float("nan")
            ap = average_precision_at_k(binary_relevance, total_relevant, k)
            actual = ranked_relevance[:k]
            ideal = np.sort(ranked_relevance)[::-1][:k]
            ideal_dcg = dcg(ideal)
            ndcg = dcg(actual) / ideal_dcg if ideal_dcg else float("nan")
            precision_scores[k].append(precision)
            ap_scores[k].append(ap)
            ndcg_scores[k].append(ndcg)
            record[f"precision_at_{k}"] = precision
            record[f"ap_at_{k}"] = ap
            record[f"ndcg_at_{k}"] = ndcg

        same = ranked_similarity[ranked_relevance >= args.relevance_threshold]
        different = ranked_similarity[ranked_relevance == 0]
        if len(same) and len(different):
            delta = float(same.mean() - different.mean())
            delta_scores.append(delta)
            record["delta_path"] = delta
        else:
            record["delta_path"] = float("nan")
        records.append(record)

    def mean(values):
        return float(np.nanmean(values)) if values else float("nan")

    metrics = {
        "gallery_dir": str(gallery_dir.resolve()),
        "query_dir": str(query_dir.resolve()),
        "embedding_dimension": int(gallery_embeddings.shape[1]),
        "relevance": "weighted_jaccard(support_query, support_gallery)",
        "relevance_threshold": args.relevance_threshold,
        "query_rows": int(len(query_rows)),
        "queries_with_support": int(len(eligible_queries)),
        "evaluated_queries": int(len(records)),
        "deduplicate_by": "dataset + study_id",
        "same_study_excluded": True,
        "metrics": {
            **{f"Precision@{k}": mean(precision_scores[k]) for k in args.k},
            **{f"mAP@{k}": mean(ap_scores[k]) for k in args.k},
            **{f"nDCG@{k}": mean(ndcg_scores[k]) for k in args.k},
            "Delta_path": mean(delta_scores),
        },
        "caveat": "support=0 means no observed target support in compact metadata, not a clinically confirmed negative.",
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    fields = ["query_id", "query_study_id", "relevant_studies"] + [
        field for k in args.k for field in (f"precision_at_{k}", f"ap_at_{k}", f"ndcg_at_{k}")
    ] + ["delta_path"]
    with (output_dir / "per_query.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    print(json.dumps(metrics, indent=2))
    print(f"Saved per-query metrics to {output_dir / 'per_query.csv'}")


if __name__ == "__main__":
    main()
