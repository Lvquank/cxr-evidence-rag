"""Build a complete, paired-bootstrap Table IV for the four 206-study runs.

The script deliberately reads the per-study target-label files rather than
summary JSON files.  Consequently the same bootstrap resample of study IDs is
applied to every pipeline, making a confidence interval for a between-pipeline
difference a *paired* interval.

Default inputs are the four completed experimental outputs in this project.
No LLM request, classifier inference, or retriever inference is performed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_INPUTS = {
    "structured_rule": "outputs/leave_one_out_206/nlp_per_sample_target6.csv",
    "structured_learned_gate": "src/learned_gate_oof_206/metrics/end_to_end_per_sample.csv",
    "baseline_a_classifier_vision": "outputs/baselines/classifier_vision_leave_one_out_206/nlp_per_sample_target6.csv",
    "baseline_b_classifier_retriever_vision": "outputs/baselines/classifier_retriever_vision_leave_one_out_206/nlp_per_sample_target6.csv",
}

DEFAULT_REFERENCE_METRICS = {
    "structured_rule": "outputs/leave_one_out_206/nlp_metrics_target6.json",
    "structured_learned_gate": "src/learned_gate_oof_206/metrics/end_to_end_metrics.json",
    "baseline_a_classifier_vision": "outputs/baselines/classifier_vision_leave_one_out_206/nlp_metrics_target6.json",
    "baseline_b_classifier_retriever_vision": "outputs/baselines/classifier_retriever_vision_leave_one_out_206/nlp_metrics_target6.json",
}

METRICS = (
    "precision",
    "recall",
    "micro_f1",
    "sample_f1",
    "label_groundedness",
    "unmatched_rate",
    "normal_f1",
)


def _as_float(value: Any) -> float:
    if value is None or str(value).strip() == "":
        return 0.0
    return float(value)


def _as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _f1(tp: float, fp: float, fn: float) -> float:
    return _safe_div(2.0 * tp, 2.0 * tp + fp + fn)


def read_per_sample(path: Path) -> dict[str, dict[str, float | bool]]:
    required = {
        "study_id",
        "problem_tp",
        "problem_fp",
        "problem_fn",
        "problem_f1",
        "gold_normal",
        "predicted_normal",
    }
    records: dict[str, dict[str, float | bool]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        for row in reader:
            study_id = str(row["study_id"]).strip()
            if not study_id:
                raise ValueError(f"{path}: encountered an empty study_id")
            if study_id in records:
                raise ValueError(f"{path}: duplicate study_id {study_id}")
            records[study_id] = {
                "tp": _as_float(row["problem_tp"]),
                "fp": _as_float(row["problem_fp"]),
                "fn": _as_float(row["problem_fn"]),
                "sample_f1": _as_float(row["problem_f1"]),
                "gold_normal": _as_bool(row["gold_normal"]),
                "predicted_normal": _as_bool(row["predicted_normal"]),
            }
    if not records:
        raise ValueError(f"{path}: no per-study rows")
    return records


def build_arrays(records: dict[str, dict[str, float | bool]], ids: list[str]) -> dict[str, np.ndarray]:
    return {
        "tp": np.asarray([float(records[x]["tp"]) for x in ids], dtype=np.float64),
        "fp": np.asarray([float(records[x]["fp"]) for x in ids], dtype=np.float64),
        "fn": np.asarray([float(records[x]["fn"]) for x in ids], dtype=np.float64),
        "sample_f1": np.asarray([float(records[x]["sample_f1"]) for x in ids], dtype=np.float64),
        "gold_normal": np.asarray([bool(records[x]["gold_normal"]) for x in ids], dtype=bool),
        "predicted_normal": np.asarray([bool(records[x]["predicted_normal"]) for x in ids], dtype=bool),
    }


def calculate_metrics(values: dict[str, np.ndarray], index: np.ndarray) -> dict[str, float]:
    tp = float(values["tp"][index].sum())
    fp = float(values["fp"][index].sum())
    fn = float(values["fn"][index].sum())
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    gold_normal = values["gold_normal"][index]
    predicted_normal = values["predicted_normal"][index]
    normal_tp = float(np.logical_and(gold_normal, predicted_normal).sum())
    normal_fp = float(np.logical_and(~gold_normal, predicted_normal).sum())
    normal_fn = float(np.logical_and(gold_normal, ~predicted_normal).sum())
    return {
        "precision": precision,
        "recall": recall,
        "micro_f1": _f1(tp, fp, fn),
        "sample_f1": float(values["sample_f1"][index].mean()),
        # In the target-label protocol, these are respectively TP/(TP+FP)
        # and FP/(TP+FP), so they complement one another.
        "label_groundedness": precision,
        "unmatched_rate": _safe_div(fp, tp + fp),
        "normal_f1": _f1(normal_tp, normal_fp, normal_fn),
    }


def percentile_interval(samples: np.ndarray) -> tuple[float, float]:
    lower, upper = np.quantile(samples, [0.025, 0.975])
    return float(lower), float(upper)


def expected_metrics(path: Path) -> dict[str, float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    target = data["metrics"]["target_problems"]
    return {
        "precision": float(target["micro_precision"]),
        "recall": float(target["micro_recall"]),
        "micro_f1": float(target["micro_f1"]),
        # Existing files call this macro_f1, but it is a mean of per-study F1.
        "sample_f1": float(target["macro_f1"]),
        "label_groundedness": float(target["target_label_groundedness"]),
        "unmatched_rate": float(target["target_label_unmatched_rate"]),
        "normal_f1": float(target["normal_decision"]["f1"]),
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fmt_ci(point: float, lower: float, upper: float) -> str:
    return f"{point:.3f} [{lower:.3f}, {upper:.3f}]"


def parse_named_paths(values: list[str], flag: str) -> dict[str, Path]:
    parsed: dict[str, Path] = {}
    for value in values:
        name, separator, raw_path = value.partition("=")
        if not separator or not name.strip() or not raw_path.strip():
            raise ValueError(f"{flag} entries must use NAME=PATH.")
        if name.strip() in parsed:
            raise ValueError(f"Duplicate pipeline name in {flag}: {name.strip()}")
        parsed[name.strip()] = Path(raw_path.strip())
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="outputs/table_iv_target_label_206")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--skip-reference-validation", action="store_true")
    parser.add_argument("--input", action="append", default=[], metavar="NAME=PER_SAMPLE_CSV", help="Override defaults; may be repeated for one or more aligned pipelines.")
    parser.add_argument("--reference-metrics", action="append", default=[], metavar="NAME=METRICS_JSON", help="Optional saved point estimates for --input validation.")
    parser.add_argument("--required-study-count", type=int, default=206)
    args = parser.parse_args()
    if args.bootstrap_samples < 100:
        raise ValueError("--bootstrap-samples must be at least 100")

    root = Path.cwd()
    custom_inputs = parse_named_paths(args.input, "--input")
    custom_references = parse_named_paths(args.reference_metrics, "--reference-metrics")
    inputs = custom_inputs if custom_inputs else {name: root / relative for name, relative in DEFAULT_INPUTS.items()}
    references = custom_references if custom_inputs else {name: root / relative for name, relative in DEFAULT_REFERENCE_METRICS.items()}
    if custom_references and set(custom_references) != set(inputs):
        raise ValueError("--reference-metrics must cover exactly the same pipeline names as --input.")
    if custom_inputs and not custom_references:
        args.skip_reference_validation = True
    for path in [*inputs.values(), *references.values()]:
        if not path.is_file():
            raise FileNotFoundError(path)

    raw_records = {name: read_per_sample(path) for name, path in inputs.items()}
    reference_name = "structured_rule" if "structured_rule" in raw_records else next(iter(raw_records))
    study_ids = sorted(raw_records[reference_name], key=lambda x: (not x.isdigit(), int(x) if x.isdigit() else x))
    expected_ids = set(study_ids)
    for name, records in raw_records.items():
        current = set(records)
        if current != expected_ids:
            missing = sorted(expected_ids - current)
            extra = sorted(current - expected_ids)
            raise ValueError(f"study ID mismatch for {name}: missing={missing[:5]} extra={extra[:5]}")
    if len(study_ids) != args.required_study_count:
        raise ValueError(f"Expected {args.required_study_count} studies; found {len(study_ids)} studies")

    arrays = {name: build_arrays(records, study_ids) for name, records in raw_records.items()}
    full_index = np.arange(len(study_ids))
    points = {name: calculate_metrics(values, full_index) for name, values in arrays.items()}

    validation: dict[str, dict[str, Any]] = {}
    if not args.skip_reference_validation:
        for name, path in references.items():
            saved = expected_metrics(path)
            differences = {metric: points[name][metric] - saved[metric] for metric in METRICS}
            max_abs_difference = max(abs(x) for x in differences.values())
            validation[name] = {
                "reference": str(path),
                "max_abs_difference": max_abs_difference,
                "passed": max_abs_difference < 1e-10,
                "differences": differences,
            }
            if max_abs_difference >= 1e-10:
                raise RuntimeError(
                    f"Recomputed point estimates do not match {path} for {name}; "
                    f"maximum absolute difference={max_abs_difference:.12g}"
                )

    pipeline_names = list(inputs)
    pair_names = [(left, right, f"{left}_minus_{right}") for left, right in combinations(pipeline_names, 2)]
    absolute_draws = {
        name: {metric: np.empty(args.bootstrap_samples, dtype=np.float64) for metric in METRICS}
        for name in pipeline_names
    }
    delta_draws = {
        name: {metric: np.empty(args.bootstrap_samples, dtype=np.float64) for metric in METRICS}
        for _, _, name in pair_names
    }
    rng = np.random.default_rng(args.seed)
    for draw_number in range(args.bootstrap_samples):
        index = rng.integers(0, len(study_ids), size=len(study_ids))
        draw_metrics = {name: calculate_metrics(arrays[name], index) for name in pipeline_names}
        for name in pipeline_names:
            for metric in METRICS:
                absolute_draws[name][metric][draw_number] = draw_metrics[name][metric]
        for left, right, pair_name in pair_names:
            for metric in METRICS:
                delta_draws[pair_name][metric][draw_number] = draw_metrics[left][metric] - draw_metrics[right][metric]

    absolute_ci = {
        name: {metric: percentile_interval(draws) for metric, draws in metric_draws.items()}
        for name, metric_draws in absolute_draws.items()
    }
    deltas: dict[str, dict[str, dict[str, float | bool]]] = {}
    for left, right, pair_name in pair_names:
        deltas[pair_name] = {}
        for metric, draws in delta_draws[pair_name].items():
            lower, upper = percentile_interval(draws)
            delta = points[left][metric] - points[right][metric]
            deltas[pair_name][metric] = {
                "delta": delta,
                "ci95_lower": lower,
                "ci95_upper": upper,
                "ci_contains_zero": lower <= 0.0 <= upper,
            }

    output_dir = root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    metric_rows: list[dict[str, Any]] = []
    for name in pipeline_names:
        row: dict[str, Any] = {"pipeline": name, "study_count": len(study_ids)}
        for metric in METRICS:
            row[metric] = points[name][metric]
            row[f"{metric}_ci95_lower"] = absolute_ci[name][metric][0]
            row[f"{metric}_ci95_upper"] = absolute_ci[name][metric][1]
        metric_rows.append(row)
    metric_fields = ["pipeline", "study_count"] + [field for metric in METRICS for field in (metric, f"{metric}_ci95_lower", f"{metric}_ci95_upper")]
    write_csv(output_dir / "table_iv_target_label_full.csv", metric_rows, metric_fields)

    delta_rows: list[dict[str, Any]] = []
    for pair_name, results in deltas.items():
        for metric, result in results.items():
            delta_rows.append({"comparison": pair_name, "metric": metric, **result})
    write_csv(
        output_dir / "table_iv_target_label_paired_deltas.csv",
        delta_rows,
        ["comparison", "metric", "delta", "ci95_lower", "ci95_upper", "ci_contains_zero"],
    )

    markdown = [
        "# Table IV — End-to-end target-label performance (N=206)",
        "",
        "Values are point estimates with percentile 95% bootstrap CIs (2,000 paired study resamples; seed 20260904). "
        "`Sample F1` is the mean of the 206 per-study F1 values; it is not the six-label macro F1.",
        "",
        "| Pipeline | Precision | Recall | Micro F1 | Sample F1 | Label groundedness | Unmatched rate ↓ | Normal F1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in pipeline_names:
        ci = absolute_ci[name]
        markdown.append(
            "| " + name.replace("_", " ") + " | " + " | ".join(
                fmt_ci(points[name][metric], *ci[metric]) for metric in METRICS
            ) + " |"
        )
    markdown.extend([
        "",
        "## Paired differences",
        "",
        "Each difference is left pipeline minus right pipeline. An interval excluding zero provides bootstrap evidence of a directional difference at the stated percentile CI level.",
        "",
        "| Comparison | Precision Δ | Recall Δ | Micro F1 Δ | Sample F1 Δ | Groundedness Δ | Unmatched Δ | Normal F1 Δ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for pair_name, results in deltas.items():
        markdown.append(
            "| " + pair_name.replace("_", " ") + " | " + " | ".join(
                fmt_ci(float(results[metric]["delta"]), float(results[metric]["ci95_lower"]), float(results[metric]["ci95_upper"]))
                for metric in METRICS
            ) + " |"
        )
    (output_dir / "table_iv_target_label_full.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")

    payload = {
        "cohort": {"study_count": len(study_ids), "study_ids": study_ids},
        "bootstrap": {"method": "paired nonparametric percentile bootstrap by study_id", "samples": args.bootstrap_samples, "seed": args.seed, "ci_level": 0.95},
        "metric_definitions": {
            "sample_f1": "mean of per-study target-label F1 values",
            "label_groundedness": "pooled target-label TP / (TP + FP)",
            "unmatched_rate": "pooled target-label FP / (TP + FP)",
        },
        "input_files": {name: {"path": str(path), "sha256": sha256(path)} for name, path in inputs.items()},
        "point_estimates": points,
        "absolute_ci95": {name: {metric: {"lower": bounds[0], "upper": bounds[1]} for metric, bounds in ci.items()} for name, ci in absolute_ci.items()},
        "paired_differences": deltas,
        "reference_validation": validation,
    }
    (output_dir / "table_iv_target_label_bootstrap_ci.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    manifest = {
        "outputs": ["table_iv_target_label_full.csv", "table_iv_target_label_paired_deltas.csv", "table_iv_target_label_full.md", "table_iv_target_label_bootstrap_ci.json"],
        "study_count": len(study_ids),
        "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed,
        "point_estimates_validated_against_existing_json": not args.skip_reference_validation,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Completed Table IV with {len(study_ids)} paired studies and {args.bootstrap_samples} bootstrap resamples.")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
