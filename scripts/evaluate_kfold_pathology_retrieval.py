"""Five-fold, study-disjoint evaluation for pathology and normal cohorts.

Each pathology has its own folds.  For a fold, all rows belonging to every
test-fold study are removed from the gallery before any query is ranked.  This
prevents image/study leakage while retaining the other pathology-positive
studies as valid retrieval evidence.
"""

import argparse
import json
import math
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.retriever.pathology_support import CHEXPERT_LABELS
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
        raise ValueError("embeddings.npy, ids.txt and metadata.jsonl have inconsistent row counts.")
    if ids != [str(row.get("id", "")) for row in rows]:
        raise ValueError("ids.txt and metadata.jsonl are not in the same order.")
    support = np.asarray([row.get("support", []) for row in rows], dtype=np.float32)
    if support.shape != (len(rows), len(CHEXPERT_LABELS)):
        raise ValueError(f"Every row must have a {len(CHEXPERT_LABELS)}-value support vector.")
    embeddings /= np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-8)
    return embeddings, rows, support


def study_key(row: dict) -> tuple[str, str]:
    dataset = str(row.get("dataset", "") or "").strip()
    patient_id = str(row.get("patient_id", "") or "").strip()
    study_id = str(row.get("study_id", row.get("uid", "")) or "").strip()
    return dataset, patient_id or study_id or f"image:{row.get('id', '')}"


def image_path(row: dict, processed_dir: Path) -> Path:
    value = row.get("path", row.get("image_path", ""))
    path = Path(str(value))
    return path if path.is_absolute() else processed_dir / path


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def weighted_jaccard(query_support: np.ndarray, gallery_support: np.ndarray) -> np.ndarray:
    numerator = np.minimum(query_support, gallery_support).sum(axis=1)
    denominator = np.maximum(query_support, gallery_support).sum(axis=1)
    return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)


def normal_relevance(gallery_support: np.ndarray) -> np.ndarray:
    """Normal queries need an explicit rule because all-zero Jaccard is undefined."""
    return np.all(gallery_support == 0.0, axis=1).astype(np.float32)


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


def pathology_delta(graded_relevance: np.ndarray, ranked_similarity: np.ndarray, threshold: float) -> float:
    support_similarity = ranked_similarity[graded_relevance >= threshold]
    zero_overlap_similarity = ranked_similarity[graded_relevance == 0.0]
    if not len(support_similarity) or not len(zero_overlap_similarity):
        return float("nan")
    return float(support_similarity.mean() - zero_overlap_similarity.mean())


def finite_mean(values: list[float]) -> float | None:
    valid = [float(value) for value in values if not math.isnan(float(value))]
    return float(np.mean(valid)) if valid else None


def finite_std(values: list[float]) -> float | None:
    valid = [float(value) for value in values if not math.isnan(float(value))]
    return float(np.std(valid, ddof=0)) if valid else None


def evaluate_fold(
    *,
    label: str,
    label_index: int | None,
    fold_number: int,
    test_groups: list[tuple[str, str]],
    groups: dict[tuple[str, str], list[int]],
    group_studies: list[tuple[str, str]],
    embeddings: np.ndarray,
    rows: list[dict],
    support: np.ndarray,
    ks: list[int],
    relevance_threshold: float,
    reranker: DenseNetLocalFeatureReranker | None,
    processed_dir: Path | None,
    local_rerank_candidates: int,
    lambda_global: float,
) -> tuple[dict, list[dict]]:
    gallery_mask = np.ones(len(rows), dtype=bool)
    for group in test_groups:
        gallery_mask[groups[group]] = False
    gallery_indices = np.flatnonzero(gallery_mask)
    gallery_embeddings = embeddings[gallery_indices]
    gallery_support = support[gallery_indices]
    gallery_studies = [group_studies[index] for index in gallery_indices]

    values = {
        **{f"WeightedJaccard@{k}": [] for k in ks},
        **{f"Precision@{k}": [] for k in ks},
        **{f"AP@{k}": [] for k in ks},
        **{f"nDCG@{k}": [] for k in ks},
        "Delta_path": [],
    }
    records = []
    local_scores_applied = 0
    skipped_missing_query_image = 0
    for group in test_groups:
        # One representative per study avoids giving multi-image studies extra weight.
        query_index = (
            groups[group][0]
            if label_index is None
            else max(groups[group], key=lambda index: float(support[index, label_index]))
        )
        similarities = gallery_embeddings @ embeddings[query_index]
        relevance = (
            normal_relevance(gallery_support)
            if label_index is None
            else weighted_jaccard(support[query_index], gallery_support)
        )
        order = np.argsort(-similarities, kind="stable")
        ranked_local_indices = []
        seen_studies = set()
        for local_index in order:
            candidate_group = gallery_studies[local_index]
            if candidate_group in seen_studies:
                continue
            seen_studies.add(candidate_group)
            ranked_local_indices.append(local_index)
        ranked_local_indices = np.asarray(ranked_local_indices, dtype=np.int64)
        ranked_relevance = relevance[ranked_local_indices]
        ranked_similarity = similarities[ranked_local_indices]
        ranking_scores = ranked_similarity.copy()

        if reranker is not None and processed_dir is not None:
            query_image = image_path(rows[query_index], processed_dir)
            if not query_image.is_file():
                skipped_missing_query_image += 1
                records.append({
                    "label": label,
                    "fold": fold_number,
                    "query_id": str(rows[query_index].get("id", "")),
                    "query_study_id": group[1],
                    "query_target_support": 1.0 if label_index is None else float(support[query_index, label_index]),
                    "gallery_studies": int(len(ranked_local_indices)),
                    "relevant_studies": None,
                    "evaluated": False,
                    "skip_reason": "missing_query_image",
                })
                continue
            local_positions = []
            candidate_paths = []
            for position, local_index in enumerate(ranked_local_indices[:local_rerank_candidates]):
                candidate_row = rows[gallery_indices[local_index]]
                candidate_image = image_path(candidate_row, processed_dir)
                if candidate_image.is_file():
                    local_positions.append(position)
                    candidate_paths.append(str(candidate_image))
            local_scores = reranker.score_paths(str(query_image), candidate_paths)
            for position, candidate_path in zip(local_positions, candidate_paths):
                local_score = local_scores.get(candidate_path)
                if local_score is None:
                    continue
                base_score = float(ranking_scores[position])
                ranking_scores[position] = (
                    lambda_global * base_score + (1.0 - lambda_global) * local_score
                )
                local_scores_applied += 1
            reranker.clear_cache()
            hybrid_order = np.argsort(-ranking_scores, kind="stable")
            ranked_relevance = ranked_relevance[hybrid_order]
            ranked_similarity = ranking_scores[hybrid_order]
        binary_relevance = (ranked_relevance >= relevance_threshold).astype(np.float32)
        total_relevant = int(binary_relevance.sum())

        record = {
            "label": label,
            "fold": fold_number,
            "query_id": str(rows[query_index].get("id", "")),
            "query_study_id": group[1],
            "query_target_support": 1.0 if label_index is None else float(support[query_index, label_index]),
            "gallery_studies": int(len(ranked_local_indices)),
            "relevant_studies": total_relevant,
            "evaluated": total_relevant > 0,
        }
        if total_relevant:
            for k in ks:
                weighted_jaccard_at_k = float(ranked_relevance[:k].mean()) if len(ranked_relevance[:k]) else float("nan")
                precision = precision_at_k(binary_relevance, k)
                ap = average_precision_at_k(binary_relevance, total_relevant, k)
                ndcg = ndcg_at_k(ranked_relevance, k)
                values[f"WeightedJaccard@{k}"].append(weighted_jaccard_at_k)
                values[f"Precision@{k}"].append(precision)
                values[f"AP@{k}"].append(ap)
                values[f"nDCG@{k}"].append(ndcg)
                record.update({
                    f"WeightedJaccard@{k}": weighted_jaccard_at_k,
                    f"Precision@{k}": precision,
                    f"AP@{k}": ap,
                    f"nDCG@{k}": ndcg,
                })
            delta = pathology_delta(ranked_relevance, ranked_similarity, relevance_threshold)
            values["Delta_path"].append(delta)
            record["Delta_path"] = delta
        records.append(record)

    metrics = {
        **{f"WeightedJaccard@{k}": finite_mean(values[f"WeightedJaccard@{k}"]) for k in ks},
        **{f"Precision@{k}": finite_mean(values[f"Precision@{k}"]) for k in ks},
        **{f"mAP@{k}": finite_mean(values[f"AP@{k}"]) for k in ks},
        **{f"nDCG@{k}": finite_mean(values[f"nDCG@{k}"]) for k in ks},
        "Delta_path": finite_mean(values["Delta_path"]),
    }
    return {
        "label": label,
        "fold": fold_number,
        "test_studies": len(test_groups),
        "gallery_rows": int(gallery_mask.sum()),
        "gallery_unique_studies": int(len(set(gallery_studies))),
        "evaluated_queries": sum(record["evaluated"] for record in records),
        "skipped_missing_query_image": skipped_missing_query_image,
        "densenet_local_scores_applied": local_scores_applied,
        "metrics": metrics,
    }, records


def write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Study-disjoint K-fold pathology retrieval evaluation.")
    parser.add_argument("--embeddings_dir", required=True, help="Directory containing embeddings.npy, ids.txt and metadata.jsonl")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k", type=int, nargs="+", default=[3, 5, 10])
    parser.add_argument("--labels", nargs="+", choices=[*CHEXPERT_LABELS, "normal"], help="Optional subset of cohorts; defaults to six pathologies plus normal")
    parser.add_argument("--exclude_normal", action="store_true", help="Do not evaluate the normal cohort")
    parser.add_argument("--target_support_threshold", type=float, default=0.2, help="Minimum target-label support to include a study as a query")
    parser.add_argument("--relevance_threshold", type=float, default=0.5, help="Weighted-Jaccard threshold used by Precision and AP")
    parser.add_argument("--processed_dir", help="Root containing metadata image paths; enables DenseNet local reranking")
    parser.add_argument("--classifier_checkpoint", default="model/best_E2_relation_aware_weighted_bce.pt")
    parser.add_argument("--local_rerank_candidates", type=int, default=50)
    parser.add_argument("--lambda_global", type=float, default=0.9, help="Global RAD-DINO weight; DenseNet local weight is 1 - lambda_global")
    parser.add_argument("--local_rerank_batch_size", type=int, default=16)
    parser.add_argument("--disable_densenet_local_rerank", action="store_true")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:0")
    parser.add_argument("--overwrite", action="store_true", help="Replace a non-empty output directory")
    args = parser.parse_args()

    if args.folds < 2:
        parser.error("--folds must be at least 2.")
    if not args.k or any(value < 1 for value in args.k):
        parser.error("--k must contain positive values.")
    if not 0.0 < args.target_support_threshold <= 1.0:
        parser.error("--target_support_threshold must be in (0, 1].")
    if not 0.0 < args.relevance_threshold <= 1.0:
        parser.error("--relevance_threshold must be in (0, 1].")
    if args.local_rerank_candidates < 1 or args.local_rerank_batch_size < 1:
        parser.error("--local_rerank_candidates and --local_rerank_batch_size must be positive.")
    if not 0.0 <= args.lambda_global <= 1.0:
        parser.error("--lambda_global must be between 0 and 1.")

    embeddings_dir = Path(args.embeddings_dir)
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"Output directory is not empty: {output_dir}. Use --overwrite to replace it.")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    embeddings, rows, support = load_artifacts(embeddings_dir)
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[study_key(row)].append(index)
    group_studies = [study_key(row) for row in rows]
    reranker = None
    processed_dir = Path(args.processed_dir) if args.processed_dir else None
    if processed_dir is not None and not args.disable_densenet_local_rerank:
        checkpoint = Path(args.classifier_checkpoint)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Classifier checkpoint not found: {checkpoint}")
        device = resolve_device(args.device)
        classifier, _ = load_model(checkpoint, device)
        reranker = DenseNetLocalFeatureReranker(classifier.backbone, device, args.local_rerank_batch_size)
        del classifier
        if device.type == "cuda":
            torch.cuda.empty_cache()

    all_fold_results = []
    all_query_records = []
    label_summaries = {}
    if args.labels:
        selected_labels = list(args.labels)
    else:
        selected_labels = [*CHEXPERT_LABELS, "normal"]
        if args.exclude_normal:
            selected_labels.pop()
    for label in selected_labels:
        label_index = CHEXPERT_LABELS.index(label) if label != "normal" else None
        positive_groups = (
            [
                group for group, indices in groups.items()
                if np.all(support[indices] == 0.0)
            ]
            if label_index is None
            else [
                group for group, indices in groups.items()
                if max(float(support[index, label_index]) for index in indices) >= args.target_support_threshold
            ]
        )
        if len(positive_groups) < args.folds:
            raise ValueError(f"{label} has only {len(positive_groups)} eligible studies; fewer than {args.folds} folds.")
        rng = np.random.default_rng(args.seed + (label_index if label_index is not None else len(CHEXPERT_LABELS)))
        shuffled_groups = list(positive_groups)
        rng.shuffle(shuffled_groups)
        fold_positions = np.array_split(np.arange(len(shuffled_groups)), args.folds)
        folds = [
            [shuffled_groups[int(position)] for position in positions]
            for positions in fold_positions
        ]
        fold_results = []
        for fold_number, test_groups in enumerate(folds, start=1):
            result, query_records = evaluate_fold(
                label=label,
                label_index=label_index,
                fold_number=fold_number,
                test_groups=test_groups,
                groups=groups,
                group_studies=group_studies,
                embeddings=embeddings,
                rows=rows,
                support=support,
                ks=args.k,
                relevance_threshold=args.relevance_threshold,
                reranker=reranker,
                processed_dir=processed_dir,
                local_rerank_candidates=args.local_rerank_candidates,
                lambda_global=args.lambda_global,
            )
            fold_results.append(result)
            all_fold_results.append(result)
            all_query_records.extend(query_records)

        metric_names = list(fold_results[0]["metrics"])
        label_summaries[label] = {
            "eligible_query_studies": len(positive_groups),
            "folds": fold_results,
            "mean_metrics": {
                name: finite_mean([fold["metrics"][name] for fold in fold_results if fold["metrics"][name] is not None])
                for name in metric_names
            },
            "std_metrics": {
                name: finite_std([fold["metrics"][name] for fold in fold_results if fold["metrics"][name] is not None])
                for name in metric_names
            },
        }

    summary = {
        "schema_version": 1,
        "protocol": {
            "folds": args.folds,
            "seed": args.seed,
            "query_selection": "One representative per target-label-positive study in its held-out fold.",
            "gallery_exclusion": "All rows from every held-out fold study are removed from that fold's gallery.",
            "gallery_deduplication": "dataset + patient_id when present; otherwise dataset + study_id",
            "graded_relevance": "Pathology: weighted_jaccard(query_support, gallery_support). Normal: 1 for all-zero support gallery rows, otherwise 0.",
            "target_support_threshold": args.target_support_threshold,
            "binary_relevance_threshold": args.relevance_threshold,
            "weighted_jaccard_at_k": "mean graded Weighted Jaccard relevance in the ranked Top-K.",
            "delta_path": "mean(ranking score | relevance >= threshold) - mean(ranking score | relevance == 0)",
            "ranking_score": (
                "lambda_global * rad_dino_cosine + (1 - lambda_global) * densenet_local_similarity"
                if reranker is not None else "rad_dino_cosine"
            ),
            "local_rerank_candidates": args.local_rerank_candidates if reranker is not None else 0,
            "lambda_global": args.lambda_global if reranker is not None else 1.0,
            "lambda_local": 1.0 - args.lambda_global if reranker is not None else 0.0,
        },
        "artifacts": {
            "embeddings_dir": str(embeddings_dir.resolve()),
            "embedding_dimension": int(embeddings.shape[1]),
            "rows": len(rows),
            "unique_studies": len(groups),
        },
        "labels": label_summaries,
        "macro_mean_over_labels": {
            name: finite_mean([
                label_summaries[label]["mean_metrics"][name]
                for label in selected_labels
                if label_summaries[label]["mean_metrics"][name] is not None
            ])
            for name in next(iter(label_summaries.values()))["mean_metrics"]
        },
        "caveat": "support=0 means no observed pathology support in metadata; it is not a clinically verified negative label.",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "fold_metrics.json").write_text(json.dumps(all_fold_results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_jsonl(output_dir / "per_query.jsonl", all_query_records)
    print(json.dumps({"output_dir": str(output_dir), "labels": list(label_summaries), "macro_mean_over_labels": summary["macro_mean_over_labels"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
