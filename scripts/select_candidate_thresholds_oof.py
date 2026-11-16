"""Select review-candidate thresholds on an independent classifier development cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ONTOLOGY = ROOT / "config" / "evaluation_ontology.json"
BENCHMARK = ROOT / "data" / "evaluations" / "leave_one_out_206_manifest.json"
FREEZE = ROOT / "config" / "densenet_candidate_thresholds_oof_v1.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_budgets(value: str, labels: list[str], default: int) -> dict[str, int]:
    budgets = dict.fromkeys(labels, default)
    if value:
        for part in value.split(","):
            name, separator, count = part.partition("=")
            if not separator or name.strip() not in budgets or not count.strip().isdigit():
                raise ValueError(f"Invalid per-label FP budget: {part!r}")
            budgets[name.strip()] = int(count.strip())
    return budgets


def multilabel_folds(frame: pd.DataFrame, ids: list[str], n_folds: int, seed: int,
                     ontology: list[str]) -> dict[str, int]:
    if len(ids) < n_folds:
        raise ValueError("Too few studies for requested folds.")
    lookup = frame.pivot(index="study_id", columns="pathology", values="gold").reindex(ids)
    matrix = lookup.loc[:, ontology].to_numpy(dtype=int)
    totals = matrix.sum(axis=0).astype(float)
    rarity = np.divide(1.0, totals, out=np.zeros_like(totals), where=totals > 0)
    rng = np.random.default_rng(seed)
    jitter = rng.random(len(ids))
    order = sorted(range(len(ids)), key=lambda index: (
        -int(matrix[index].sum()), -float(matrix[index] @ rarity), float(jitter[index]), ids[index]))
    desired = totals / n_folds
    counts = np.zeros((n_folds, len(ontology)), dtype=float)
    sizes = np.zeros(n_folds, dtype=int)
    mapping = {}
    for index in order:
        vector = matrix[index]
        def score(fold: int) -> tuple[float, int, int]:
            proposed = counts.copy()
            proposed[fold] += vector
            label_error = np.sum((proposed - desired) ** 2 / np.maximum(desired, 1.0))
            proposed_sizes = sizes.copy()
            proposed_sizes[fold] += 1
            size_error = np.sum((proposed_sizes - len(ids) / n_folds) ** 2) / max(len(ids) / n_folds, 1.0)
            return float(label_error + 0.15 * size_error), int(sizes[fold]), fold
        fold = min(range(n_folds), key=score)
        mapping[ids[index]] = fold
        counts[fold] += vector
        sizes[fold] += 1
    return mapping


def sweep_thresholds(frame: pd.DataFrame, main: float, fp_budget: int,
                     rate_cap: float, outer_fold: int) -> tuple[float | None, list[dict]]:
    below = frame.loc[frame.raw_probability < main, ["raw_probability", "gold"]]
    grouped = below.groupby("raw_probability", sort=False)["gold"].agg(["sum", "count"])
    tp = fp = candidates = 0
    records = []
    best = None
    for threshold, row in grouped.sort_index(ascending=False).iterrows():
        positives = int(row["sum"])
        total = int(row["count"])
        tp += positives
        fp += total - positives
        candidates += total
        rate = candidates / len(frame)
        eligible = fp <= fp_budget and rate <= rate_cap
        record = {"outer_fold": outer_fold, "threshold": float(threshold),
                  "additional_TP": tp, "additional_FP": fp,
                  "candidate_count": candidates, "candidate_rate": rate,
                  "eligible": eligible}
        records.append(record)
        if eligible and tp > 0:
            key = (tp, -fp, float(threshold))
            if best is None or key > best[0]:
                best = (key, float(threshold))
    return (best[1] if best else None), records


def bootstrap(rows: pd.DataFrame, labels: list[str], seed: int, draws: int = 1000) -> dict:
    ids = sorted(rows.study_id.unique())
    values = {}
    for label in labels:
        part = rows.loc[rows.pathology == label].set_index("study_id").loc[ids]
        values[label] = np.column_stack([
            (part.gold.to_numpy(int) * part.review_candidate.to_numpy(int)),
            ((1 - part.gold.to_numpy(int)) * part.review_candidate.to_numpy(int)),
            part.review_candidate.to_numpy(int),
        ])
    rng = np.random.default_rng(seed)
    result = {}
    for label, matrix in values.items():
        samples = np.empty((draws, 3), dtype=float)
        for draw in range(draws):
            indices = rng.integers(0, len(ids), len(ids))
            samples[draw] = matrix[indices].sum(axis=0)
        result[label] = {name: {"estimate": float(matrix[:, index].sum()),
                                "ci95_low": float(np.quantile(samples[:, index], 0.025)),
                                "ci95_high": float(np.quantile(samples[:, index], 0.975))}
                         for index, name in enumerate(("additional_TP", "additional_FP", "candidate_count"))}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=ROOT / "data" / "classifier_dev_oof_predictions.csv")
    parser.add_argument("--main-thresholds", type=Path, default=ROOT / "config" / "densenet_adaptive_thresholds.json")
    parser.add_argument("--labels", nargs="+", default=["Atelectasis", "Pneumothorax"])
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=4)
    parser.add_argument("--candidate-rate-cap", type=float, default=0.15)
    parser.add_argument("--fp-budget", type=int, default=4)
    parser.add_argument("--fp-budgets", default="", help="Comma-separated mapping, e.g. Atelectasis=4,Pneumothorax=3")
    parser.add_argument("--seed", type=int, default=20260930)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "candidate_threshold_oof_v1")
    args = parser.parse_args()
    if not (0 < args.candidate_rate_cap <= 1) or args.fp_budget < 0 or args.outer_folds < 2 or args.inner_folds < 2:
        parser.error("Invalid cap, FP budget, or fold count.")
    if args.output_dir.exists() or FREEZE.exists():
        parser.error("OOF output or frozen artifact exists; do not overwrite an earlier selection.")
    ontology = json.loads(ONTOLOGY.read_text(encoding="utf-8"))["target_labels"]
    if len(set(args.labels)) != len(args.labels) or not set(args.labels) <= {"Atelectasis", "Pneumothorax"}:
        parser.error("Only distinct Atelectasis and Pneumothorax are permitted in v1.")
    budgets = parse_budgets(args.fp_budgets, args.labels, args.fp_budget)
    mains = json.loads(args.main_thresholds.read_text(encoding="utf-8"))["labels"]
    manifest_path = args.predictions.with_name(args.predictions.stem + "_manifest.json")
    source_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if source_manifest["sha256"]["predictions"] != sha256(args.predictions):
        raise ValueError("Prediction CSV hash differs from its provenance manifest.")
    if "independent" not in source_manifest["training_fold_mapping"]:
        raise ValueError("Prediction manifest does not establish independent development scoring.")
    frame = pd.read_csv(args.predictions, dtype={"study_id": str, "pathology": str, "fold": str, "checkpoint_id": str})
    required = {"study_id", "pathology", "raw_probability", "gold", "fold", "checkpoint_id"}
    if not required <= set(frame) or frame[list(required)].isna().any().any():
        raise ValueError("Prediction CSV has missing required columns/values.")
    if frame.duplicated(["study_id", "pathology"]).any() or set(frame.pathology) != set(ontology):
        raise ValueError("Every study must have exactly one row per six ontology labels.")
    ids = sorted(frame.study_id.unique())
    if len(frame) != len(ids) * len(ontology) or frame.groupby("study_id").pathology.nunique().ne(len(ontology)).any():
        raise ValueError("Incomplete study-label grid.")
    frozen_ids = {str(item["study_id"]) for item in json.loads(BENCHMARK.read_text(encoding="utf-8"))["requests"]}
    if set(ids) & frozen_ids or len(frozen_ids) != 206:
        raise ValueError("Development predictions overlap the frozen 206 cohort.")
    if not frame.raw_probability.between(0, 1).all() or not frame.gold.isin([0, 1]).all():
        raise ValueError("Invalid probabilities or gold labels.")
    if frame.groupby("study_id").fold.nunique().gt(1).any() or frame.checkpoint_id.nunique() != 1:
        raise ValueError("Inconsistent study folds or checkpoint identity.")
    if set(frame.fold) != {"independent_dev"} or frame.checkpoint_id.iloc[0] != source_manifest["checkpoint_id"]:
        raise ValueError("Predictions do not match independent-development manifest.")
    outer = multilabel_folds(frame, ids, args.outer_folds, args.seed, ontology)
    output_rows = []
    sweeps = []
    folds = []
    metrics = {"input_source": str(args.predictions), "source_manifest": str(manifest_path),
               "fold_ids": {str(fold): [study for study in ids if outer[study] == fold] for fold in range(args.outer_folds)},
               "main_thresholds": {label: float(mains[label]["threshold"]) for label in args.labels},
               "fp_budgets": budgets, "candidate_rate_cap": args.candidate_rate_cap,
               "labels": {}}
    for label in args.labels:
        main = float(mains[label]["threshold"])
        if not 0 < main <= 1:
            raise ValueError(f"Invalid main threshold for {label}.")
        selected_thresholds = []
        for fold in range(args.outer_folds):
            train_ids = [study for study in ids if outer[study] != fold]
            inner = multilabel_folds(frame[frame.study_id.isin(train_ids)], train_ids,
                                     args.inner_folds, args.seed + fold + 1, ontology)
            # Every development prediction is independent of its study. Pool the
            # held-out inner partitions to select only from the outer-training set.
            train = frame[(frame.study_id.isin(train_ids)) & (frame.pathology == label)].copy()
            train["inner_fold"] = train.study_id.map(inner)
            if set(train.inner_fold) != set(range(args.inner_folds)):
                raise ValueError("Empty inner validation fold.")
            threshold, sweep = sweep_thresholds(train, main, budgets[label], args.candidate_rate_cap, fold)
            for item in sweep:
                item["pathology"] = label
            sweeps.extend(sweep)
            if threshold is not None and not 0 <= threshold < main:
                raise ValueError("Selected candidate threshold is outside [0, main threshold).")
            selected_thresholds.append(threshold)
            folds.append({"pathology": label, "outer_fold": fold, "selected_threshold": threshold,
                          "enabled": threshold is not None, "training_studies": len(train_ids),
                          "held_out_studies": len(ids) - len(train_ids),
                          "inner_fold_sizes": [sum(value == inner_fold for value in inner.values())
                                               for inner_fold in range(args.inner_folds)]})
            held_out = frame[(frame.study_id.map(outer) == fold) & (frame.pathology == label)]
            for row in held_out.itertuples(index=False):
                selected = bool(row.raw_probability >= main)
                candidate = bool(threshold is not None and threshold <= row.raw_probability < main)
                output_rows.append({"study_id": row.study_id, "pathology": label, "gold": int(row.gold),
                                    "raw_probability": float(row.raw_probability), "outer_fold": fold,
                                    "main_threshold": main, "selected_threshold": threshold,
                                    "main_positive": int(selected), "review_candidate": int(candidate)})
        part = pd.DataFrame(output_rows).query("pathology == @label")
        tp = int(((part.gold == 1) & (part.review_candidate == 1)).sum())
        fp = int(((part.gold == 0) & (part.review_candidate == 1)).sum())
        fn = int(((part.gold == 1) & (part.main_positive == 0) & (part.review_candidate == 0)).sum())
        rate = float(part.review_candidate.mean())
        enabled = all(value is not None for value in selected_thresholds) and tp > 0 and fp <= budgets[label] and rate <= args.candidate_rate_cap
        deployment = float(np.median(selected_thresholds)) if enabled else None
        metrics["labels"][label] = {"enabled": enabled, "outer_thresholds": selected_thresholds,
                                    "OOF_additional_TP": tp, "OOF_additional_FP": fp,
                                    "OOF_remaining_FN_below_main": fn,
                                    "OOF_candidate_count": int(part.review_candidate.sum()),
                                    "OOF_candidate_rate": rate,
                                    "deployment_candidate": deployment,
                                    "development_positive_studies": int(part.gold.sum()),
                                    "disable_reason": None if enabled else "Insufficient nested OOF TP or exceeded preregistered FP/rate cap"}
    output = pd.DataFrame(output_rows).sort_values(["study_id", "pathology"])
    if len(output) != len(ids) * len(args.labels) or output.duplicated(["study_id", "pathology"]).any():
        raise ValueError("Nested OOF output is incomplete.")
    args.output_dir.mkdir(parents=True)
    output.to_csv(args.output_dir / "nested_oof_predictions.csv", index=False)
    pd.DataFrame(sweeps).to_csv(args.output_dir / "inner_threshold_sweep.csv", index=False)
    pd.DataFrame(folds).to_csv(args.output_dir / "outer_fold_thresholds.csv", index=False)
    (args.output_dir / "candidate_threshold_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "bootstrap_ci.json").write_text(json.dumps(bootstrap(output, args.labels, args.seed), indent=2) + "\n", encoding="utf-8")
    provenance = {"created_at": datetime.now(timezone.utc).isoformat(), "seed": args.seed,
                  "predictions_sha256": sha256(args.predictions), "source_manifest_sha256": sha256(manifest_path),
                  "main_thresholds_sha256": sha256(args.main_thresholds), "benchmark_manifest_sha256": sha256(BENCHMARK),
                  "selector_code_sha256": sha256(Path(__file__)), "study_count": len(ids),
                  "input_fold": "independent_dev", "outer_folds": args.outer_folds, "inner_folds": args.inner_folds}
    (args.output_dir / "manifest.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    freeze = {"version": "densenet-review-candidate-oof-v1", "source": str(args.predictions.relative_to(ROOT)),
              "source_sha256": provenance["predictions_sha256"], "seed": args.seed,
              "selection": {"unit": "study_id", "outer_folds": args.outer_folds,
                            "inner_folds": args.inner_folds, "candidate_rate_cap": args.candidate_rate_cap,
                            "objective": "maximize incremental TP subject to per-label FP budget"},
              "labels": {label: {"enabled": item["enabled"], "main_threshold": metrics["main_thresholds"][label],
                                 "candidate_threshold": item["deployment_candidate"], "fp_budget": budgets[label]}
                         for label, item in metrics["labels"].items()}}
    if any(item["enabled"] for item in metrics["labels"].values()):
        FREEZE.write_text(json.dumps(freeze, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), "artifact": str(FREEZE) if FREEZE.exists() else None,
                      "labels": metrics["labels"]}, indent=2))


if __name__ == "__main__":
    main()
