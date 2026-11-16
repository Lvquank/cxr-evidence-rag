"""Audit L2 normalization, anisotropy and collapse indicators for an embedding matrix."""

import argparse
import json
from pathlib import Path

import numpy as np


def summarize_embedding_geometry(embeddings: np.ndarray, sample_pairs: int, seed: int) -> dict:
    if embeddings.ndim != 2 or len(embeddings) < 2:
        raise ValueError("embeddings must be a two-dimensional matrix with at least two rows.")
    vectors = np.asarray(embeddings, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1)
    normalized = vectors / np.maximum(norms[:, None], 1e-12)

    rng = np.random.default_rng(seed)
    left = rng.integers(len(normalized), size=sample_pairs)
    right = rng.integers(len(normalized), size=sample_pairs)
    same = left == right
    while same.any():
        right[same] = rng.integers(len(normalized), size=int(same.sum()))
        same = left == right
    pairwise_cosines = (normalized[left] * normalized[right]).sum(axis=1)

    centered = normalized - normalized.mean(axis=0, keepdims=True)
    covariance = centered.T @ centered / max(1, len(centered) - 1)
    eigenvalues = np.maximum(np.linalg.eigvalsh(covariance), 0.0)
    total_variance = float(eigenvalues.sum())
    if total_variance:
        distribution = eigenvalues / total_variance
        nonzero = distribution[distribution > 0]
        effective_rank = float(np.exp(-(nonzero * np.log(nonzero)).sum()))
        top_component_ratio = float(distribution[-1])
        top_ten_ratio = float(distribution[-min(10, len(distribution)):].sum())
    else:
        effective_rank = 0.0
        top_component_ratio = 1.0
        top_ten_ratio = 1.0

    centroid_norm = float(np.linalg.norm(normalized.mean(axis=0)))
    pairwise_std = float(pairwise_cosines.std())
    collapse_flags = []
    if float(np.abs(norms - 1.0).max()) > 1e-3:
        collapse_flags.append("vectors_not_l2_normalized")
    if centroid_norm > 0.70:
        collapse_flags.append("high_centroid_anisotropy")
    if pairwise_std < 0.01:
        collapse_flags.append("low_pairwise_variance")
    if effective_rank < max(8.0, normalized.shape[1] * 0.10):
        collapse_flags.append("low_effective_rank")

    return {
        "num_vectors": int(len(normalized)),
        "dimension": int(normalized.shape[1]),
        "normalization": {
            "norm_min": float(norms.min()),
            "norm_mean": float(norms.mean()),
            "norm_max": float(norms.max()),
            "max_abs_error_from_one": float(np.abs(norms - 1.0).max()),
        },
        "random_pairwise_cosine": {
            "sample_pairs": int(sample_pairs),
            "mean": float(pairwise_cosines.mean()),
            "std": pairwise_std,
            "p01": float(np.quantile(pairwise_cosines, 0.01)),
            "p50": float(np.quantile(pairwise_cosines, 0.50)),
            "p99": float(np.quantile(pairwise_cosines, 0.99)),
        },
        "anisotropy": {
            "centroid_norm": centroid_norm,
            "effective_rank": effective_rank,
            "top_component_variance_ratio": top_component_ratio,
            "top_10_component_variance_ratio": top_ten_ratio,
        },
        "heuristic_flags": collapse_flags,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Check embedding normalization and geometry.")
    parser.add_argument("--embeddings_path", required=True)
    parser.add_argument("--sample_pairs", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_json")
    args = parser.parse_args()
    if args.sample_pairs < 1:
        parser.error("--sample_pairs must be at least 1.")

    path = Path(args.embeddings_path)
    if not path.is_file():
        raise FileNotFoundError(f"Embedding matrix not found: {path}")
    report = summarize_embedding_geometry(np.load(path), args.sample_pairs, args.seed)
    report["embeddings_path"] = str(path.resolve())
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
