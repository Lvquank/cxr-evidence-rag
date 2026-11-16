"""Evaluate completed ablation summaries, including paired target-label bootstrap."""

import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_module_improvements import ARMS, DEFAULT_ROOT, LABELS, read, write, freeze
from scripts.evaluate_llm_outputs import (load_reports, load_ontology, evaluate_summaries,
    add_bertscore, aggregate, write_per_sample_csv)


def label_metrics(gold, predicted):
    tp = (gold & predicted).sum(axis=0)
    fp = (~gold & predicted).sum(axis=0)
    fn = (gold & ~predicted).sum(axis=0)
    denominator = 2 * tp + fp + fn
    per_label = np.divide(2 * tp, denominator, out=np.zeros(len(LABELS)), where=denominator > 0)
    sample_denominator = gold.sum(axis=1) + predicted.sum(axis=1)
    sample_f1 = np.divide(2 * (gold & predicted).sum(axis=1), sample_denominator,
                          out=np.ones(len(gold)), where=sample_denominator > 0)
    return {"micro_f1": float(2 * tp.sum() / denominator.sum()) if denominator.sum() else 0.,
            "pathology_macro_f1": float(per_label.mean()), "sample_f1": float(sample_f1.mean()),
            "precision": float(tp.sum() / (tp + fp).sum()) if (tp + fp).sum() else 0.,
            "recall": float(tp.sum() / (tp + fn).sum()) if (tp + fn).sum() else 0.,
            "tp": int(tp.sum()), "fp": int(fp.sum()), "fn": int(fn.sum())}


def error_sources(gold, selected, predicted):
    if np.any(predicted & ~selected):
        raise ValueError("Prediction contains a label outside the selected-only schema")
    masks = {
        "preselection_fn": gold & ~selected,
        "rejected_true_labels": gold & selected & ~predicted,
        "removed_false_labels": ~gold & selected & ~predicted,
        "retained_false_labels": ~gold & predicted,
        "retained_true_labels": gold & predicted,
    }
    return {"totals": {name: int(mask.sum()) for name, mask in masks.items()},
            "per_label": {label: {name: int(mask[:, i].sum()) for name, mask in masks.items()}
                          for i, label in enumerate(LABELS)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--skip-bertscore", action="store_true")
    parser.add_argument("--clinical", action="store_true")
    args = parser.parse_args()
    requests = freeze(args.output_root)
    expected = {str(item["study_id"]) for item in requests}
    reports = load_reports(ROOT / "src/data/indiana_reports.csv")
    ontology = load_ontology()
    prediction_arrays, metrics_by_arm, attribution = {}, {}, {}
    prepared = {row["study_id"]: row for row in read(args.output_root / "preparation_complete.json")["results"]}
    clinical_dir = args.output_root / "clinical_pairs"
    gold_array = None
    studies = sorted(expected, key=int)
    for arm in ARMS:
        files = sorted((args.output_root / arm / "llm").glob("*/summary_frozen.json"))
        records, skipped = evaluate_summaries(files, reports, {}, ontology=ontology)
        if skipped or len(records) != 206 or {r["study_id"] for r in records} != expected:
            raise ValueError(f"Incomplete cohort in {arm}; completed {len(records)}/206")
        records.sort(key=lambda row: int(row["study_id"]))
        score_metadata = ({"computed": False, "reason": "explicit_skip"} if args.skip_bertscore else
                          add_bertscore(records, "roberta-large", batch_size=8, device="cuda"))
        for record in records:
            for section in ("findings", "impression"):
                for field in ("bertscore_precision", "bertscore_recall", "bertscore_f1"):
                    record[section].setdefault(field, None)
        metrics_by_arm[arm] = aggregate(records, LABELS)
        write(args.output_root / arm / "nlp_metrics_target6.json", {
            "metrics": metrics_by_arm[arm], "records": records, "skipped": skipped, "bertscore": score_metadata})
        write_per_sample_csv(args.output_root / arm / "nlp_per_sample_target6.csv", records)
        gold = np.array([[label in r["gold_target_problems"] for label in LABELS] for r in records])
        if gold_array is not None and not np.array_equal(gold_array, gold):
            raise ValueError("Reference labels differ across arms")
        gold_array = gold
        prediction_arrays[arm] = np.array([[label in r["predicted_target_problems"] for label in LABELS] for r in records])
        selection_key = "baseline_labels" if arm == "control" else "candidate_labels"
        selected = np.array([[label in prepared[study][selection_key] for label in LABELS] for study in studies])
        attribution[arm] = error_sources(gold, selected, prediction_arrays[arm])
        pairs = []
        excluded = []
        for row in records:
            if not any(row["reference_text"][s].strip() for s in ("findings", "impression")):
                excluded.append(row["study_id"])
                continue
            pairs.append({"pipeline": arm, "study_id": row["study_id"], "model": "gpt-5.6-luna",
                "generated_findings": row["prediction_text"]["findings"],
                "generated_impression": row["prediction_text"]["impression"],
                "reference_findings": row["reference_text"]["findings"],
                "reference_impression": row["reference_text"]["impression"]})
        clinical_dir.mkdir(exist_ok=True)
        pd.DataFrame(pairs).to_csv(clinical_dir / f"{arm}.csv", index=False)
        write(clinical_dir / f"{arm}_coverage.json", {"included": len(pairs), "excluded_empty_reference": excluded})
    estimates = {arm: label_metrics(gold_array, pred) for arm, pred in prediction_arrays.items()}
    differences = {}
    rng = np.random.default_rng(20261018)
    sample_ids = rng.integers(0, len(studies), (2000, len(studies)))
    for left, right in itertools.combinations(ARMS, 2):
        delta = {key: estimates[right][key] - estimates[left][key] for key in estimates[left]}
        samples = {key: [] for key in delta}
        for indices in sample_ids:
            l = label_metrics(gold_array[indices], prediction_arrays[left][indices])
            r = label_metrics(gold_array[indices], prediction_arrays[right][indices])
            for key in samples:
                samples[key].append(r[key] - l[key])
        differences[f"{right}_minus_{left}"] = {
            key: {"delta": value, "ci95": np.quantile(samples[key], [.025, .975]).tolist()}
            for key, value in delta.items()}
    write(args.output_root / "end_to_end_comparison.json", {"n_studies": len(studies), "estimates": estimates,
        "error_attribution": attribution,
        "paired_bootstrap": differences, "bootstrap_draws": 2000, "seed": 20261018,
        "clinical_reference_count": len(pairs), "note": "Historical cohort; exploratory evidence only."})
    if args.clinical:
        command = [sys.executable, str(ROOT / "scripts/evaluate_clinical_metrics_for_pipeline.py"),
            "--target-pipeline", "hybrid_consistent", "--target-pairs", str(clinical_dir / "hybrid_consistent.csv"),
            "--baseline-pairs", f"control={clinical_dir / 'control.csv'}",
            "--baseline-pairs", f"query_first={clinical_dir / 'query_first.csv'}",
            "--output-dir", str(args.output_root / "clinical_evaluation")]
        subprocess.run(command, check=True, cwd=ROOT)
    print(json.dumps({"estimates": estimates, "output": str(args.output_root)}))


if __name__ == "__main__":
    main()
