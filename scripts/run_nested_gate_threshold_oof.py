"""Nested, study-level OOF selection of the learned evidence-gate threshold.

This script is deliberately limited to the frozen gate dataset.  It does not
call an LLM, alter the existing 206-report results, or use a test study while
choosing that study's operating threshold.

Protocol
--------
* Reuse the five stored outer study folds from ``folds.json``.
* For each outer fold, make deterministic 4-fold multilabel inner folds from
  the remaining studies.
* Fit the original StandardScaler + balanced liblinear LogisticRegression on
  each inner-training partition, choose the threshold exclusively from its
  inner OOF predictions, then refit on the whole outer-training partition.
* Apply the selected threshold only to the held-out outer studies.

The merged predictions are an internally validated nested-OOF result.  They
are not an independent external-test estimate, and a median of outer-fold
thresholds is reported only as a deployment candidate, not as a re-optimised
score on all 206 studies.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "src" / "learned_gate_oof_206" / "gate_dataset.csv"
DEFAULT_FOLDS = ROOT / "src" / "learned_gate_oof_206" / "folds.json"
DEFAULT_OUTPUT = ROOT / "outputs" / "learned_gate_nested_oof_206"

PATHOLOGIES = (
    "Atelectasis",
    "Cardiomegaly",
    "Edema",
    "Pleural Effusion",
    "Pneumonia",
    "Pneumothorax",
)
NUMERIC_FEATURES = (
    "raw_prob",
    "m5",
    "support_mean",
    "non_support_mean",
    "support_count",
    "non_support_count",
)


def parse_thresholds(value: str) -> list[float]:
    """Accept ``start:stop:step`` (inclusive) or a comma-separated list."""
    if ":" in value:
        parts = [float(item) for item in value.split(":")]
        if len(parts) != 3 or parts[2] <= 0:
            raise argparse.ArgumentTypeError("Threshold range must be start:stop:positive_step.")
        start, stop, step = parts
        values = []
        current = start
        while current <= stop + 1e-10:
            values.append(round(current, 10))
            current += step
    else:
        values = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not values or any(item <= 0 or item >= 1 for item in values):
        raise argparse.ArgumentTypeError("Every threshold must be strictly between 0 and 1.")
    return sorted(set(values))


def make_feature_matrix(frame: pd.DataFrame) -> np.ndarray:
    numeric = frame.loc[:, NUMERIC_FEATURES].astype(float).to_numpy()
    one_hot = np.column_stack([(frame["pathology"] == pathology).to_numpy(dtype=float) for pathology in PATHOLOGIES])
    return np.column_stack([numeric, one_hot])


def build_model(seed: int) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=1.0,
                    class_weight="balanced",
                    solver="liblinear",
                    max_iter=2000,
                    tol=1e-4,
                    random_state=seed,
                ),
            ),
        ]
    )


def contingency(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, int]:
    return {
        "tp": int(np.sum((y_true == 1) & (y_pred == 1))),
        "fp": int(np.sum((y_true == 0) & (y_pred == 1))),
        "fn": int(np.sum((y_true == 1) & (y_pred == 0))),
        "tn": int(np.sum((y_true == 0) & (y_pred == 0))),
    }


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    result: dict[str, float | int] = contingency(y_true, y_pred)
    tp, fp, fn = int(result["tp"]), int(result["fp"]), int(result["fn"])
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    result.update(
        {
            "precision": precision,
            "recall": recall,
            "micro_f1": f1,
            "emitted": int(np.sum(y_pred == 1)),
            "positives": int(np.sum(y_true == 1)),
        }
    )
    return result


def choose_threshold(y_true: np.ndarray, probabilities: np.ndarray, thresholds: Iterable[float]) -> tuple[float, list[dict[str, float | int]]]:
    """Maximise micro-F1; ties prefer higher precision, then the threshold nearest .5."""
    sweep: list[dict[str, float | int]] = []
    for threshold in thresholds:
        row = {"threshold": threshold, **metrics(y_true, (probabilities >= threshold).astype(int))}
        sweep.append(row)
    best = max(
        sweep,
        key=lambda item: (
            float(item["micro_f1"]),
            float(item["precision"]),
            -abs(float(item["threshold"]) - 0.5),
        ),
    )
    return float(best["threshold"]), sweep


def study_label_matrix(frame: pd.DataFrame, study_ids: list[str]) -> np.ndarray:
    lookup = frame.pivot(index="study_id", columns="pathology", values="gold").reindex(study_ids)
    return lookup.loc[:, PATHOLOGIES].astype(int).to_numpy()


def deterministic_multilabel_folds(frame: pd.DataFrame, study_ids: list[str], n_folds: int, seed: int) -> dict[str, int]:
    """Small deterministic greedy multilabel splitter, keeping all rows of a study together."""
    if len(study_ids) < n_folds:
        raise ValueError(f"Cannot make {n_folds} folds from only {len(study_ids)} studies.")
    labels = study_label_matrix(frame, study_ids)
    totals = labels.sum(axis=0).astype(float)
    rarity = np.divide(1.0, totals, out=np.zeros_like(totals), where=totals > 0)
    rng = np.random.default_rng(seed)
    jitter = {study_id: float(rng.random()) for study_id in study_ids}
    order = sorted(
        range(len(study_ids)),
        key=lambda index: (-int(labels[index].sum()), -float(labels[index] @ rarity), jitter[study_ids[index]], study_ids[index]),
    )
    desired = totals / n_folds
    fold_counts = np.zeros((n_folds, len(PATHOLOGIES)), dtype=float)
    fold_sizes = np.zeros(n_folds, dtype=int)
    mapping: dict[str, int] = {}
    for index in order:
        label_vector = labels[index]
        candidate_scores = []
        for fold in range(n_folds):
            proposed = fold_counts.copy()
            proposed[fold] += label_vector
            label_error = np.sum(np.square(proposed - desired) / np.maximum(desired, 1.0))
            size_error = ((fold_sizes[fold] + 1) - len(study_ids) / n_folds) ** 2 / max(len(study_ids) / n_folds, 1.0)
            candidate_scores.append((float(label_error + 0.15 * size_error), int(fold_sizes[fold]), fold))
        _, _, selected_fold = min(candidate_scores)
        mapping[study_ids[index]] = selected_fold
        fold_counts[selected_fold] += label_vector
        fold_sizes[selected_fold] += 1
    return mapping


def study_bootstrap(
    predictions: pd.DataFrame,
    action_column: str,
    samples: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    rng = np.random.default_rng(seed)
    groups = [group for _, group in predictions.groupby("study_id", sort=True)]
    if len({len(group) for group in groups}) != 1:
        raise ValueError("Study bootstrap requires the same number of label rows per study.")
    gold_by_study = np.vstack([group["gold"].to_numpy(int) for group in groups])
    action_by_study = np.vstack([group[action_column].to_numpy(int) for group in groups])
    observed = metrics(predictions["gold"].to_numpy(int), predictions[action_column].to_numpy(int))
    draws: dict[str, list[float]] = {key: [] for key in ("precision", "recall", "micro_f1")}
    for _ in range(samples):
        chosen = rng.integers(0, len(groups), size=len(groups))
        value = metrics(gold_by_study[chosen].ravel(), action_by_study[chosen].ravel())
        for key in draws:
            draws[key].append(float(value[key]))
    return {
        key: {
            "estimate": float(observed[key]),
            "ci95_low": float(np.quantile(values, 0.025)),
            "ci95_high": float(np.quantile(values, 0.975)),
        }
        for key, values in draws.items()
    }


def paired_study_bootstrap_delta(
    predictions: pd.DataFrame,
    left_action: str,
    right_action: str,
    samples: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    """Percentile CI for left minus right, resampling whole studies jointly."""
    rng = np.random.default_rng(seed)
    groups = [group for _, group in predictions.groupby("study_id", sort=True)]
    if len({len(group) for group in groups}) != 1:
        raise ValueError("Study bootstrap requires the same number of label rows per study.")
    gold_by_study = np.vstack([group["gold"].to_numpy(int) for group in groups])
    left_by_study = np.vstack([group[left_action].to_numpy(int) for group in groups])
    right_by_study = np.vstack([group[right_action].to_numpy(int) for group in groups])
    observed_left = metrics(predictions["gold"].to_numpy(int), predictions[left_action].to_numpy(int))
    observed_right = metrics(predictions["gold"].to_numpy(int), predictions[right_action].to_numpy(int))
    draws: dict[str, list[float]] = {key: [] for key in ("precision", "recall", "micro_f1")}
    for _ in range(samples):
        chosen = rng.integers(0, len(groups), size=len(groups))
        left = metrics(gold_by_study[chosen].ravel(), left_by_study[chosen].ravel())
        right = metrics(gold_by_study[chosen].ravel(), right_by_study[chosen].ravel())
        for key in draws:
            draws[key].append(float(left[key]) - float(right[key]))
    return {
        key: {
            "estimate": float(observed_left[key]) - float(observed_right[key]),
            "ci95_low": float(np.quantile(values, 0.025)),
            "ci95_high": float(np.quantile(values, 0.975)),
        }
        for key, values in draws.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--folds", type=Path, default=DEFAULT_FOLDS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--inner-folds", type=int, default=4)
    parser.add_argument("--thresholds", type=parse_thresholds, default=parse_thresholds("0.05:0.95:0.05"))
    parser.add_argument("--seed", type=int, default=20260928)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--overwrite", action="store_true", help="Replace files in an existing output directory.")
    args = parser.parse_args()
    if args.inner_folds < 2:
        parser.error("--inner-folds must be at least 2.")
    if args.bootstrap_samples < 100:
        parser.error("--bootstrap-samples must be at least 100.")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        parser.error(f"Output directory already contains results: {args.output_dir}. Use --overwrite to replace them.")

    dataset = pd.read_csv(args.dataset, dtype={"study_id": str})
    missing = set(NUMERIC_FEATURES + ("study_id", "pathology", "gold")).difference(dataset.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")
    if set(dataset["pathology"].unique()) != set(PATHOLOGIES):
        raise ValueError("Dataset pathology ontology differs from the frozen six-label ontology.")
    per_study = dataset.groupby("study_id")["pathology"].nunique()
    if not (per_study == len(PATHOLOGIES)).all():
        raise ValueError("Every study must contain exactly six pathology rows.")

    fold_payload = json.loads(args.folds.read_text(encoding="utf-8"))
    outer_map = {str(study_id): int(fold) for study_id, fold in fold_payload["study_to_fold"].items()}
    study_ids = sorted(dataset["study_id"].unique(), key=lambda item: int(item))
    if set(study_ids) != set(outer_map):
        raise ValueError("Frozen outer-fold study IDs do not exactly match the gate dataset.")
    outer_folds = sorted(set(outer_map.values()))
    if len(outer_folds) != 5 or outer_folds != list(range(5)):
        raise ValueError("Expected five outer folds numbered 0 through 4.")

    all_outer_predictions: list[pd.DataFrame] = []
    sweep_rows: list[dict[str, float | int]] = []
    selected_rows: list[dict[str, float | int]] = []
    inner_fold_rows: list[dict[str, str | int]] = []

    for outer_fold in outer_folds:
        test_studies = [study_id for study_id in study_ids if outer_map[study_id] == outer_fold]
        train_studies = [study_id for study_id in study_ids if outer_map[study_id] != outer_fold]
        inner_map = deterministic_multilabel_folds(dataset, train_studies, args.inner_folds, args.seed + outer_fold)
        inner_fold_rows.extend(
            {"outer_fold": outer_fold, "study_id": study_id, "inner_fold": fold}
            for study_id, fold in sorted(inner_map.items(), key=lambda item: int(item[0]))
        )

        inner_predictions: list[pd.DataFrame] = []
        for inner_fold in range(args.inner_folds):
            validation_studies = [study_id for study_id in train_studies if inner_map[study_id] == inner_fold]
            fitting_studies = [study_id for study_id in train_studies if inner_map[study_id] != inner_fold]
            train_frame = dataset[dataset["study_id"].isin(fitting_studies)]
            validation_frame = dataset[dataset["study_id"].isin(validation_studies)].copy()
            model = build_model(args.seed + outer_fold * 100 + inner_fold)
            model.fit(make_feature_matrix(train_frame), train_frame["gold"].to_numpy(int))
            validation_frame["gate_probability"] = model.predict_proba(make_feature_matrix(validation_frame))[:, 1]
            inner_predictions.append(validation_frame)
        inner_oof = pd.concat(inner_predictions, ignore_index=True).sort_values(["study_id", "pathology"])
        selected_threshold, sweep = choose_threshold(
            inner_oof["gold"].to_numpy(int), inner_oof["gate_probability"].to_numpy(float), args.thresholds
        )
        for entry in sweep:
            sweep_rows.append({"outer_fold": outer_fold, **entry})
        selected = next(entry for entry in sweep if float(entry["threshold"]) == selected_threshold)
        selected_rows.append(
            {
                "outer_fold": outer_fold,
                "outer_train_studies": len(train_studies),
                "outer_test_studies": len(test_studies),
                "inner_folds": args.inner_folds,
                "selected_threshold": selected_threshold,
                **{f"inner_{key}": value for key, value in selected.items() if key != "threshold"},
            }
        )

        outer_train = dataset[dataset["study_id"].isin(train_studies)]
        outer_test = dataset[dataset["study_id"].isin(test_studies)].copy()
        final_model = build_model(args.seed + outer_fold * 100 + 99)
        final_model.fit(make_feature_matrix(outer_train), outer_train["gold"].to_numpy(int))
        outer_test["outer_fold"] = outer_fold
        outer_test["gate_probability"] = final_model.predict_proba(make_feature_matrix(outer_test))[:, 1]
        outer_test["selected_threshold"] = selected_threshold
        outer_test["nested_action"] = (outer_test["gate_probability"] >= selected_threshold).astype(int)
        outer_test["fixed_0_5_action"] = (outer_test["gate_probability"] >= 0.5).astype(int)
        all_outer_predictions.append(outer_test)

    predictions = pd.concat(all_outer_predictions, ignore_index=True).sort_values(["study_id", "pathology"])
    if predictions["study_id"].nunique() != len(study_ids) or len(predictions) != len(dataset):
        raise RuntimeError("Nested OOF predictions do not cover the frozen dataset exactly once.")

    nested_metrics = metrics(predictions["gold"].to_numpy(int), predictions["nested_action"].to_numpy(int))
    fixed_metrics = metrics(predictions["gold"].to_numpy(int), predictions["fixed_0_5_action"].to_numpy(int))
    per_pathology = {
        pathology: metrics(
            group["gold"].to_numpy(int), group["nested_action"].to_numpy(int)
        )
        for pathology, group in predictions.groupby("pathology", sort=False)
    }
    thresholds_by_fold = pd.DataFrame(selected_rows).sort_values("outer_fold")
    recommended_threshold = float(np.median(thresholds_by_fold["selected_threshold"]))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output_dir / "nested_oof_predictions.csv", index=False)
    pd.DataFrame(sweep_rows).sort_values(["outer_fold", "threshold"]).to_csv(args.output_dir / "inner_threshold_sweep.csv", index=False)
    thresholds_by_fold.to_csv(args.output_dir / "outer_fold_thresholds.csv", index=False)
    pd.DataFrame(inner_fold_rows).sort_values(["outer_fold", "study_id"]).to_csv(args.output_dir / "inner_folds.csv", index=False)

    bootstrap = {
        "nested_selected_threshold": study_bootstrap(predictions, "nested_action", args.bootstrap_samples, args.seed + 7000),
        "fixed_threshold_0_5": study_bootstrap(predictions, "fixed_0_5_action", args.bootstrap_samples, args.seed + 8000),
        "paired_delta_nested_minus_fixed_0_5": paired_study_bootstrap_delta(
            predictions, "nested_action", "fixed_0_5_action", args.bootstrap_samples, args.seed + 9000
        ),
    }
    summary = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "outer_folds": 5,
            "outer_fold_source": str(args.folds.relative_to(ROOT)),
            "outer_split_level": "study_id",
            "inner_folds": args.inner_folds,
            "inner_split_level": "study_id",
            "inner_split_strategy": "deterministic greedy multilabel stratification",
            "seed": args.seed,
            "model": "StandardScaler + LogisticRegression(C=1.0, class_weight=balanced, solver=liblinear, max_iter=2000, tol=1e-4)",
            "features": [*NUMERIC_FEATURES, *[f"pathology={item}" for item in PATHOLOGIES]],
            "threshold_grid": args.thresholds,
            "selection_objective": "maximise inner-OOF micro F1; ties: higher precision, then threshold nearest 0.5",
        },
        "coverage": {"study_count": len(study_ids), "pair_count": len(predictions), "pathology_count": len(PATHOLOGIES)},
        "nested_oof": nested_metrics,
        "same_outer_models_fixed_threshold_0_5": fixed_metrics,
        "per_pathology_nested_oof": per_pathology,
        "outer_fold_selected_thresholds": selected_rows,
        "deployment_candidate": {
            "threshold": recommended_threshold,
            "rule": "median of outer-fold selected thresholds",
            "warning": "A deployment candidate only; its performance is not re-estimated on the same 206 studies.",
        },
        "interpretation": "Each outer test study was never used to fit its gate model or select its threshold. This is an internally validated nested-OOF result, not an external test result.",
    }
    (args.output_dir / "nested_oof_metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output_dir / "bootstrap_ci.json").write_text(json.dumps(bootstrap, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = {
        "created_at": summary["created_at"],
        "inputs": {"dataset": str(args.dataset), "outer_folds": str(args.folds)},
        "outputs": [
            "nested_oof_predictions.csv",
            "inner_threshold_sweep.csv",
            "outer_fold_thresholds.csv",
            "inner_folds.csv",
            "nested_oof_metrics.json",
            "bootstrap_ci.json",
        ],
        "no_llm_calls": True,
        "no_existing_outputs_overwritten": not args.overwrite,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({"output_dir": str(args.output_dir), "nested_oof": nested_metrics, "deployment_candidate": recommended_threshold}, indent=2))


if __name__ == "__main__":
    main()
