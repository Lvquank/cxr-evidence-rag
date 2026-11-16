#!/usr/bin/env python3
"""Bootstrap confidence intervals for a retrieval lambda-sweep CSV.

The resampling unit is one query study, not individual candidates or labels.
All lambda configurations for a sampled study are kept together, which permits
paired confidence intervals for a reranked configuration minus global-only.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def as_float(value: str | None) -> float | None:
    try:
        numeric = float(value or "")
    except ValueError:
        return None
    return numeric if math.isfinite(numeric) else None


def bootstrap_mean(values: np.ndarray, iterations: int, rng: np.random.Generator) -> dict[str, Any]:
    indices = rng.integers(0, len(values), size=(iterations, len(values)))
    distribution = values[indices].mean(axis=1)
    return {
        "point": float(values.mean()),
        "lower": float(np.quantile(distribution, 0.025)),
        "upper": float(np.quantile(distribution, 0.975)),
        "n_studies": int(len(values)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Add study-level paired bootstrap CIs to retrieval lambda-sweep metrics.")
    parser.add_argument("--per-query-csv", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument(
        "--unit-column",
        default="query_study_id",
        help="Study-level resampling key; defaults to query_study_id from the lambda-sweep CSV.",
    )
    parser.add_argument("--metrics", nargs="+", default=["P@5", "AP@5", "nDCG@5", "Delta_path@5"])
    parser.add_argument("--reference-lambda", type=float, default=1.0, help="Global-only configuration under lambda_global notation.")
    parser.add_argument("--comparison-lambdas", nargs="*", type=float, help="Defaults to all non-reference configurations.")
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260904)
    args = parser.parse_args()
    if args.iterations < 100:
        parser.error("--iterations must be at least 100.")

    with args.per_query_csv.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("The per-query CSV is empty.")
    required = {args.unit_column, "lambda_global", *args.metrics}
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

    values: dict[float, dict[str, dict[str, float]]] = {}
    for row in rows:
        unit = str(row[args.unit_column]).strip()
        weight = as_float(row["lambda_global"])
        if not unit or weight is None:
            continue
        bucket = values.setdefault(weight, {})
        if unit in bucket:
            raise ValueError(f"Duplicate unit '{unit}' for lambda_global={weight}.")
        bucket[unit] = {metric: as_float(row.get(metric)) for metric in args.metrics}

    lambda_values = sorted(values)
    if args.reference_lambda not in values:
        raise ValueError(f"reference lambda {args.reference_lambda} is absent from CSV; found {lambda_values}")
    reference_units = set(values[args.reference_lambda])
    for weight, bucket in values.items():
        if set(bucket) != reference_units:
            raise ValueError(f"Study set for lambda {weight} does not match reference lambda {args.reference_lambda}.")
    compare = args.comparison_lambdas or [weight for weight in lambda_values if weight != args.reference_lambda]
    absent = sorted(set(compare) - set(lambda_values))
    if absent:
        raise ValueError(f"Comparison lambda values absent from CSV: {absent}")

    rng = np.random.default_rng(args.seed)
    absolute: dict[str, Any] = {}
    for weight in lambda_values:
        metric_results: dict[str, Any] = {}
        for metric in args.metrics:
            vector = np.asarray(
                [values[weight][unit][metric] for unit in reference_units if values[weight][unit][metric] is not None],
                dtype=np.float64,
            )
            metric_results[metric] = bootstrap_mean(vector, args.iterations, rng) if len(vector) else None
        absolute[str(weight)] = metric_results

    paired: dict[str, Any] = {}
    for weight in compare:
        metric_results = {}
        for metric in args.metrics:
            differences = np.asarray([
                values[weight][unit][metric] - values[args.reference_lambda][unit][metric]
                for unit in reference_units
                if values[weight][unit][metric] is not None
                and values[args.reference_lambda][unit][metric] is not None
            ], dtype=np.float64)
            if not len(differences):
                metric_results[metric] = None
                continue
            result = bootstrap_mean(differences, args.iterations, rng)
            result["ci_excludes_zero"] = bool(result["lower"] > 0.0 or result["upper"] < 0.0)
            metric_results[metric] = result
        paired[str(weight)] = metric_results

    output = {
        "schema_version": 1,
        "method": "nonparametric percentile bootstrap",
        "resampling_unit": args.unit_column,
        "paired_across_lambda": True,
        "iterations": args.iterations,
        "seed": args.seed,
        "input_csv": str(args.per_query_csv.resolve()),
        "score_convention": "lambda_global * S_global + (1 - lambda_global) * S_local",
        "reference_lambda_global": args.reference_lambda,
        "absolute_metrics": absolute,
        "paired_difference_vs_reference": paired,
        "interpretation": "A paired-difference CI excluding zero supports a directional difference within this cached candidate-pool protocol.",
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output_json": str(args.output_json),
        "reference_lambda_global": args.reference_lambda,
        "paired_difference_vs_reference": paired,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
