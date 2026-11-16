#!/usr/bin/env python3
"""Run RadGraph and paired study-level bootstrap confidence intervals.

This evaluator takes the evaluator-neutral CSVs from
``prepare_clinical_text_pairs.py``.  It writes raw RadGraph input/output,
per-study scores, aggregate scores, absolute CIs, and paired CIs versus one
pre-specified reference pipeline.  No reference text is fabricated.

Run normally with ``--config``.  The ``--radgraph-worker-*`` arguments are an
internal subprocess entry point that isolates RadGraph's dependencies.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from validate_common_studies import build_canonical_pairs, joined_report, sha256, write_canonical


ROOT = Path(__file__).resolve().parents[1]
SCORE_NAMES = ("radgraph_entity_f1", "radgraph_relation_f1", "radgraph_complete_f1")


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle) or {}
    if not isinstance(value, dict):
        raise ValueError("Configuration root must be a mapping.")
    return value


def path_from_config(value: str | None, fallback: Path) -> Path:
    return (ROOT / value).resolve() if value else fallback


def percentile_ci(values: np.ndarray) -> tuple[float, float]:
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def bootstrap_scores(
    scores_by_pipeline: dict[str, np.ndarray], reference_pipeline: str, iterations: int, seed: int,
) -> dict[str, Any]:
    """Use the same sampled study indices for every pipeline (paired bootstrap)."""
    names = sorted(scores_by_pipeline)
    if reference_pipeline not in scores_by_pipeline:
        raise ValueError(f"Reference pipeline {reference_pipeline!r} is not present.")
    count = next(iter(scores_by_pipeline.values())).shape[0]
    if any(value.shape != (count, 3) for value in scores_by_pipeline.values()):
        raise ValueError("Each pipeline must have one entity/relation/complete score per common study.")
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, count, size=(iterations, count))
    sampled_means = {name: values[samples].mean(axis=1) for name, values in scores_by_pipeline.items()}
    absolute: dict[str, Any] = {}
    for name in names:
        observed = scores_by_pipeline[name].mean(axis=0)
        absolute[name] = {}
        for index, metric in enumerate(SCORE_NAMES):
            low, high = percentile_ci(sampled_means[name][:, index])
            absolute[name][metric] = {"estimate": float(observed[index]), "ci_95_lower": low, "ci_95_upper": high}
    comparisons: dict[str, Any] = {}
    for name in names:
        if name == reference_pipeline:
            continue
        delta = scores_by_pipeline[name].mean(axis=0) - scores_by_pipeline[reference_pipeline].mean(axis=0)
        delta_samples = sampled_means[name] - sampled_means[reference_pipeline]
        comparisons[f"{name}_minus_{reference_pipeline}"] = {}
        for index, metric in enumerate(SCORE_NAMES):
            low, high = percentile_ci(delta_samples[:, index])
            comparisons[f"{name}_minus_{reference_pipeline}"][metric] = {
                "delta": float(delta[index]),
                "bootstrap_mean_delta": float(delta_samples[:, index].mean()),
                "ci_95_lower": low,
                "ci_95_upper": high,
                "ci_contains_zero": bool(low <= 0.0 <= high),
            }
    return {
        "schema_version": 1,
        "method": "paired nonparametric percentile bootstrap",
        "resampling_unit": "study_id",
        "iterations": iterations,
        "seed": seed,
        "confidence_level": 0.95,
        "reference_pipeline": reference_pipeline,
        "absolute": absolute,
        "comparisons": comparisons,
    }


def build_batches(canonical: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    batches: dict[str, list[dict[str, str]]] = {}
    for row in canonical:
        batches.setdefault(row["pipeline"], []).append({
            "study_id": row["study_id"],
            "hypothesis": joined_report(row["generated_findings"], row["generated_impression"]),
            "reference": joined_report(row["reference_findings"], row["reference_impression"]),
        })
    return batches


def _construct_radgraph(model_type: str, cache_dir: str | None):
    try:
        from radgraph import F1RadGraph  # type: ignore
    except ImportError as error:
        raise RuntimeError(
            "RadGraph is not installed. Clone the pinned repository into third_party/radgraph "
            "and install it with the dedicated requirements file."
        ) from error
    kwargs: dict[str, Any] = {"reward_level": "all", "model_type": model_type}
    parameters = inspect.signature(F1RadGraph).parameters
    if cache_dir:
        for key in ("cache_dir", "model_cache_dir"):
            if key in parameters:
                kwargs[key] = cache_dir
                break
    return F1RadGraph(**kwargs)


def run_radgraph_worker(input_path: Path, output_path: Path, model_type: str, cache_dir: str | None, repo_path: Path | None) -> None:
    if repo_path and repo_path.is_dir():
        sys.path.insert(0, str(repo_path.resolve()))
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    batches = payload.get("batches")
    if not isinstance(batches, dict) or not batches:
        raise ValueError("RadGraph worker input must contain non-empty 'batches'.")
    scorer = _construct_radgraph(model_type, cache_dir)
    result: dict[str, Any] = {"model_type": model_type, "score_order": ["entity", "relation", "complete"], "pipelines": {}}
    for pipeline, rows in batches.items():
        studies = [str(row["study_id"]) for row in rows]
        hypotheses = [str(row["hypothesis"]) for row in rows]
        references = [str(row["reference"]) for row in rows]
        raw = scorer(hyps=hypotheses, refs=references)
        if not isinstance(raw, tuple) or len(raw) != 4:
            raise RuntimeError("Unexpected F1RadGraph return value; expected mean, scores, hypothesis annotations, reference annotations.")
        mean, scores, hypothesis_annotations, reference_annotations = raw
        score_rows = [[float(value) for value in row] for row in scores]
        if len(score_rows) != len(studies) or any(len(row) != 3 for row in score_rows):
            raise RuntimeError("RadGraph returned a score count/order inconsistent with input studies.")
        result["pipelines"][pipeline] = {
            "study_ids": studies,
            "mean": [float(value) for value in mean],
            "scores": score_rows,
            "hypothesis_annotations": hypothesis_annotations,
            "reference_annotations": reference_annotations,
        }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_worker_scores(path: Path, batches: dict[str, list[dict[str, str]]]) -> dict[str, np.ndarray]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    pipelines = payload.get("pipelines")
    if not isinstance(pipelines, dict) or set(pipelines) != set(batches):
        raise ValueError("RadGraph worker output pipelines do not match canonical input.")
    output: dict[str, np.ndarray] = {}
    for name, expected_rows in batches.items():
        item = pipelines[name]
        expected_ids = [row["study_id"] for row in expected_rows]
        if item.get("study_ids") != expected_ids:
            raise ValueError(f"RadGraph output study order differs for pipeline {name}.")
        values = np.asarray(item.get("scores"), dtype=float)
        if values.shape != (len(expected_ids), 3) or not np.isfinite(values).all():
            raise ValueError(f"Invalid RadGraph scores for pipeline {name}.")
        output[name] = values
    return output


def worker_command(args: argparse.Namespace) -> int:
    run_radgraph_worker(args.radgraph_worker_input, args.radgraph_worker_output, args.radgraph_model_type, args.radgraph_cache_dir, args.radgraph_repo)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "radgraph_bootstrap.yaml")
    parser.add_argument("--pairs-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--reuse-radgraph-raw", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--radgraph-worker-input", type=Path)
    parser.add_argument("--radgraph-worker-output", type=Path)
    parser.add_argument("--radgraph-repo", type=Path)
    parser.add_argument("--radgraph-model-type")
    parser.add_argument("--radgraph-cache-dir")
    args = parser.parse_args()
    if args.radgraph_worker_input or args.radgraph_worker_output:
        if not all((args.radgraph_worker_input, args.radgraph_worker_output, args.radgraph_model_type)):
            parser.error("Worker mode requires --radgraph-worker-input, --radgraph-worker-output, and --radgraph-model-type.")
        return worker_command(args)

    config = load_yaml(args.config)
    evaluation = config.get("evaluation", {})
    radgraph = config.get("radgraph", {})
    bootstrap = config.get("bootstrap", {})
    pairs_dir = args.pairs_dir or path_from_config(config.get("pairs_dir"), ROOT / "data" / "evaluations" / "clinical_text_pairs_4pipelines")
    output_dir = args.output_dir or path_from_config(config.get("output_dir"), ROOT / "outputs" / "radgraph_bootstrap")
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        parser.error(f"Output directory already contains files: {output_dir}. Use --overwrite to replace them.")
    expected = tuple(str(item) for item in evaluation.get("expected_pipelines", [])) or None
    canonical, validation = build_canonical_pairs(
        pairs_dir,
        required_count=evaluation.get("required_study_count"),
        expected_pipelines=expected,
        empty_reference_policy=str(evaluation.get("empty_reference_policy", "fail")),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_canonical(canonical, validation, output_dir / "canonical_pairs.csv", output_dir / "validation_manifest.json")
    batches = build_batches(canonical)
    raw_dir = output_dir / "radgraph_raw"
    input_path, result_path = raw_dir / "radgraph_input.json", raw_dir / "radgraph_output.json"
    input_path.parent.mkdir(parents=True, exist_ok=True)
    input_path.write_text(json.dumps({"batches": batches}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    repo_path = path_from_config(radgraph.get("repo_path"), ROOT / "third_party" / "radgraph")
    cache_dir = path_from_config(radgraph.get("cache_dir"), ROOT / "models" / "radgraph")
    model_type = str(radgraph.get("model_type", "radgraph-xl"))
    if not args.reuse_radgraph_raw:
        executable = str(radgraph.get("python_executable") or sys.executable)
        command = [
            executable, str(Path(__file__).resolve()),
            "--radgraph-worker-input", str(input_path),
            "--radgraph-worker-output", str(result_path),
            "--radgraph-repo", str(repo_path),
            "--radgraph-model-type", model_type,
            "--radgraph-cache-dir", str(cache_dir),
        ]
        completed = subprocess.run(command, text=True, capture_output=True)
        (raw_dir / "radgraph.log").write_text(
            "COMMAND\n" + " ".join(command) + "\n\nSTDOUT\n" + completed.stdout + "\nSTDERR\n" + completed.stderr,
            encoding="utf-8",
        )
        if completed.returncode:
            raise RuntimeError(f"RadGraph worker failed; inspect {raw_dir / 'radgraph.log'}")
    if not result_path.is_file():
        raise FileNotFoundError(f"Missing RadGraph output: {result_path}")
    scores = load_worker_scores(result_path, batches)
    reference_pipeline = str(bootstrap.get("reference_pipeline", "structured_rule"))
    iterations = int(bootstrap.get("samples", 2000))
    seed = int(bootstrap.get("seed", 20260904))
    if iterations < 100:
        raise ValueError("Bootstrap samples must be at least 100.")
    ci = bootstrap_scores(scores, reference_pipeline, iterations, seed)

    per_sample: list[dict[str, Any]] = []
    aggregate: dict[str, Any] = {}
    for pipeline, rows in batches.items():
        values = scores[pipeline]
        aggregate[pipeline] = {metric: float(values[:, index].mean()) for index, metric in enumerate(SCORE_NAMES)}
        for row, score in zip(rows, values, strict=True):
            per_sample.append({"pipeline": pipeline, "study_id": row["study_id"], **{metric: float(score[index]) for index, metric in enumerate(SCORE_NAMES)}})
    with (output_dir / "radgraph_per_sample.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["pipeline", "study_id", *SCORE_NAMES])
        writer.writeheader()
        writer.writerows(per_sample)
    with (output_dir / "radgraph_final_table.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["pipeline", "study_count", *SCORE_NAMES])
        writer.writeheader()
        for pipeline in sorted(aggregate):
            writer.writerow({"pipeline": pipeline, "study_count": validation["study_count"], **aggregate[pipeline]})
    metrics_payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "study_count": validation["study_count"],
        "report_join": "findings_then_impression",
        "radgraph": {"model_type": model_type, "score_order": ["entity", "relation", "complete"]},
        "aggregate": aggregate,
    }
    (output_dir / "radgraph_metrics.json").write_text(json.dumps(metrics_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "radgraph_bootstrap_ci.json").write_text(json.dumps(ci, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "created_at": metrics_payload["created_at"],
        "config": {"path": str(args.config.resolve()), "sha256": sha256(args.config)},
        "validation_manifest": str((output_dir / "validation_manifest.json").resolve()),
        "radgraph": {"repo_path": str(repo_path), "expected_repo_commit": radgraph.get("expected_repo_commit"), "model_type": model_type, "cache_dir": str(cache_dir)},
        "bootstrap": {"samples": iterations, "seed": seed, "resampling_unit": "study_id", "reference_pipeline": reference_pipeline},
        "outputs": ["radgraph_metrics.json", "radgraph_per_sample.csv", "radgraph_bootstrap_ci.json", "radgraph_final_table.csv", "radgraph_raw/radgraph_input.json", "radgraph_raw/radgraph_output.json"],
    }
    (output_dir / "radgraph_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "aggregate": aggregate}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
