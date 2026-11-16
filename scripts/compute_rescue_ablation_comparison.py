"""Compute detailed comparative rescue metrics, tradeoffs, and paired bootstrap CIs.

Implements the exact evaluation protocol from Section 9 & 10 of
HANDOFF_ALL_LABEL_PROBABILITY_IMAGE_RESCUE_ABLATION.md:
- G: Gold target pathologies
- C: Classifier selected labels
- P0: Control (selected_only) predicted labels
- P1: Rescue (all_six) predicted labels

Calculates:
- Classifier FN: G - C
- Rescued classifier FN: (G - C) intersect P1
- Added below-threshold FP: (P1 - C) - G
- Recovered end-to-end FN: (G - P0) intersect P1
- Lost previous TP: (G intersect P0) - P1
- New FP versus control: (P1 - G) - P0
- Removed FP versus control: (P0 - G) - P1
- Rescue precision: rescued_classifier_fn / (rescued_classifier_fn + added_below_threshold_fp)
- Rescue recall: rescued_classifier_fn / classifier_fn
- Micro F1, Sample F1, Pathology Macro F1
- Per-pathology breakdown
- Paired bootstrap 95% CIs across study IDs
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ONTOLOGY = ROOT / "config" / "evaluation_ontology.json"
TARGET_PATHOLOGIES = (
    "Atelectasis",
    "Cardiomegaly",
    "Edema",
    "Pleural Effusion",
    "Pneumonia",
    "Pneumothorax",
)


def _safe_div(num: float, den: float) -> Optional[float]:
    return float(num / den) if den != 0.0 else None


def _f1(tp: int | float, fp: int | float, fn: int | float) -> float:
    denom = 2.0 * tp + fp + fn
    return float(2.0 * tp / denom) if denom > 0 else 0.0


def load_evidence_records(evidence_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Map study_id -> evidence dict."""
    evidence_by_study: Dict[str, Dict[str, Any]] = {}
    for path in evidence_dir.glob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        study_id = None
        # Check retriever excluded_study_ids or study_id in data
        ret = data.get("retriever", {})
        if ret.get("excluded_study_ids"):
            study_id = str(ret["excluded_study_ids"][0])
        elif data.get("study_id"):
            study_id = str(data["study_id"])
        if study_id:
            evidence_by_study[study_id] = data
    return evidence_by_study


def load_arm_predictions_from_csv(csv_path: Path) -> Dict[str, Dict[str, Any]]:
    """Read nlp_per_sample.csv produced by evaluate_llm_outputs.py."""
    rows: Dict[str, Dict[str, Any]] = {}
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            study_id = str(row["study_id"]).strip()
            gold_str = str(row.get("gold_target_problems", "")).strip()
            pred_str = str(row.get("predicted_target_problems", "")).strip()
            gold_labels = {l.strip() for l in gold_str.split(";") if l.strip() and l.strip() in TARGET_PATHOLOGIES}
            pred_labels = {l.strip() for l in pred_str.split(";") if l.strip() and l.strip() in TARGET_PATHOLOGIES}
            rows[study_id] = {
                "study_id": study_id,
                "image_name": row.get("image_name", ""),
                "gold_labels": gold_labels,
                "predicted_labels": pred_labels,
                "sample_f1": float(row.get("problem_f1", 0.0)),
            }
    return rows


def load_arm_predictions_from_llm_dir(llm_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Fallback: read summary_*.json files in llm directory."""
    records: Dict[str, Dict[str, Any]] = {}
    for path in llm_dir.glob("**/summary_*.json"):
        try:
            summary = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        source = summary.get("source") or {}
        study_id = str(source.get("study_id") or "").strip()
        if not study_id:
            continue
        pred_obj = summary.get("prediction") or {}
        pred_problems = pred_obj.get("problems") or []
        pred_labels = {l for l in pred_problems if l in TARGET_PATHOLOGIES}
        records[study_id] = {
            "study_id": study_id,
            "image_name": source.get("image_name", ""),
            "predicted_labels": pred_labels,
            "prediction": pred_obj,
        }
    return records


def compute_comparison_records(
    control_data: Dict[str, Dict[str, Any]],
    rescue_data: Dict[str, Dict[str, Any]],
    evidence_by_study: Dict[str, Dict[str, Any]],
    gold_by_study: Optional[Dict[str, Set[str]]] = None,
) -> List[Dict[str, Any]]:
    """Compute per-study comparison records across common studies."""
    common_studies = sorted(set(control_data.keys()) & set(rescue_data.keys()))
    if not common_studies:
        raise ValueError("No overlapping study IDs found between control and rescue datasets.")

    records: List[Dict[str, Any]] = []
    for study_id in common_studies:
        ctrl = control_data[study_id]
        resc = rescue_data[study_id]

        if gold_by_study and study_id in gold_by_study:
            G = set(gold_by_study[study_id])
        else:
            G = set(ctrl.get("gold_labels") or resc.get("gold_labels") or [])

        # Classifier selected labels
        evidence = evidence_by_study.get(study_id, {})
        clf_selected = evidence.get("classifier", {}).get("selected_labels") or []
        C = {label for label in clf_selected if label in TARGET_PATHOLOGIES}

        P0 = set(ctrl["predicted_labels"])
        P1 = set(resc["predicted_labels"])

        classifier_fn = G - C
        rescued_classifier_fn = (G - C) & P1
        added_below_threshold_fp = (P1 - C) - G
        recovered_end_to_end_fn = (G - P0) & P1
        lost_previous_tp = (G & P0) - P1
        new_fp_vs_control = (P1 - G) - P0
        removed_fp_vs_control = (P0 - G) - P1

        image_name = ctrl.get("image_name") or resc.get("image_name") or ""

        # Study-level metrics for control
        ctrl_tp = len(G & P0)
        ctrl_fp = len(P0 - G)
        ctrl_fn = len(G - P0)
        ctrl_sample_f1 = _f1(ctrl_tp, ctrl_fp, ctrl_fn) if (ctrl_tp or ctrl_fp or ctrl_fn) else 1.0

        # Study-level metrics for rescue
        resc_tp = len(G & P1)
        resc_fp = len(P1 - G)
        resc_fn = len(G - P1)
        resc_sample_f1 = _f1(resc_tp, resc_fp, resc_fn) if (resc_tp or resc_fp or resc_fn) else 1.0

        records.append({
            "study_id": study_id,
            "image_name": image_name,
            "gold_labels": sorted(G),
            "classifier_selected_labels": sorted(C),
            "control_predicted_labels": sorted(P0),
            "rescue_predicted_labels": sorted(P1),
            "classifier_fn": sorted(classifier_fn),
            "rescued_classifier_fn": sorted(rescued_classifier_fn),
            "added_below_threshold_fp": sorted(added_below_threshold_fp),
            "recovered_end_to_end_fn": sorted(recovered_end_to_end_fn),
            "lost_previous_tp": sorted(lost_previous_tp),
            "new_fp_vs_control": sorted(new_fp_vs_control),
            "removed_fp_vs_control": sorted(removed_fp_vs_control),
            "num_classifier_fn": len(classifier_fn),
            "num_rescued_classifier_fn": len(rescued_classifier_fn),
            "num_added_below_threshold_fp": len(added_below_threshold_fp),
            "num_recovered_end_to_end_fn": len(recovered_end_to_end_fn),
            "num_lost_previous_tp": len(lost_previous_tp),
            "num_new_fp_vs_control": len(new_fp_vs_control),
            "num_removed_fp_vs_control": len(removed_fp_vs_control),
            "control_tp": ctrl_tp,
            "control_fp": ctrl_fp,
            "control_fn": ctrl_fn,
            "control_sample_f1": ctrl_sample_f1,
            "rescue_tp": resc_tp,
            "rescue_fp": resc_fp,
            "rescue_fn": resc_fn,
            "rescue_sample_f1": resc_sample_f1,
        })

    return records


def calculate_headline_metrics(
    records: List[Dict[str, Any]],
    arm_pred_key: str,
) -> Dict[str, Any]:
    """Calculate TP, FP, FN, TN, precision, recall, Micro F1, Sample F1, Pathology Macro F1."""
    total_tp = sum(r[f"{arm_pred_key}_tp"] for r in records)
    total_fp = sum(r[f"{arm_pred_key}_fp"] for r in records)
    total_fn = sum(r[f"{arm_pred_key}_fn"] for r in records)
    total_tn = len(records) * len(TARGET_PATHOLOGIES) - (total_tp + total_fp + total_fn)

    micro_prec = _safe_div(total_tp, total_tp + total_fp) or 0.0
    micro_rec = _safe_div(total_tp, total_tp + total_fn) or 0.0
    micro_f1 = _f1(total_tp, total_fp, total_fn)

    sample_f1 = float(np.mean([r[f"{arm_pred_key}_sample_f1"] for r in records]))

    # Per-pathology breakdown
    per_path: Dict[str, Dict[str, Any]] = {}
    f1_list: List[float] = []
    pred_field = "control_predicted_labels" if arm_pred_key == "control" else "rescue_predicted_labels"

    for path in TARGET_PATHOLOGIES:
        p_tp = sum(1 for r in records if path in r["gold_labels"] and path in r[pred_field])
        p_fp = sum(1 for r in records if path not in r["gold_labels"] and path in r[pred_field])
        p_fn = sum(1 for r in records if path in r["gold_labels"] and path not in r[pred_field])
        p_tn = len(records) - (p_tp + p_fp + p_fn)
        p_prec = _safe_div(p_tp, p_tp + p_fp) or 0.0
        p_rec = _safe_div(p_tp, p_tp + p_fn) or 0.0
        p_f1 = _f1(p_tp, p_fp, p_fn)
        gold_pos = sum(1 for r in records if path in r["gold_labels"])
        per_path[path] = {
            "gold_positives": gold_pos,
            "tp": p_tp,
            "fp": p_fp,
            "fn": p_fn,
            "tn": p_tn,
            "precision": round(p_prec, 4),
            "recall": round(p_rec, 4),
            "f1": round(p_f1, 4),
        }
        f1_list.append(p_f1)

    pathology_macro_f1 = float(np.mean(f1_list))

    return {
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "tn": total_tn,
        "precision": round(micro_prec, 4),
        "recall": round(micro_rec, 4),
        "micro_f1": round(micro_f1, 4),
        "sample_f1": round(sample_f1, 4),
        "pathology_macro_f1": round(pathology_macro_f1, 4),
        "per_pathology": per_path,
    }


def run_paired_bootstrap(
    records: List[Dict[str, Any]],
    n_samples: int = 2000,
    seed: int = 20260904,
) -> Dict[str, Any]:
    """Perform paired bootstrap over study records with fixed seed."""
    rng = np.random.default_rng(seed)
    n = len(records)
    indices = rng.integers(0, n, size=(n_samples, n))

    ctrl_micro_f1s = []
    ctrl_sample_f1s = []
    ctrl_macro_f1s = []

    resc_micro_f1s = []
    resc_sample_f1s = []
    resc_macro_f1s = []

    diff_micro_f1s = []
    diff_sample_f1s = []
    diff_macro_f1s = []

    rescued_fn_counts = []
    added_fp_counts = []
    net_gain_counts = []

    for b in range(n_samples):
        b_idx = indices[b]
        b_records = [records[i] for i in b_idx]

        # Control
        c_tp = sum(r["control_tp"] for r in b_records)
        c_fp = sum(r["control_fp"] for r in b_records)
        c_fn = sum(r["control_fn"] for r in b_records)
        c_micro = _f1(c_tp, c_fp, c_fn)
        c_sample = float(np.mean([r["control_sample_f1"] for r in b_records]))
        c_path_f1s = []
        for path in TARGET_PATHOLOGIES:
            p_tp = sum(1 for r in b_records if path in r["gold_labels"] and path in r["control_predicted_labels"])
            p_fp = sum(1 for r in b_records if path not in r["gold_labels"] and path in r["control_predicted_labels"])
            p_fn = sum(1 for r in b_records if path in r["gold_labels"] and path not in r["control_predicted_labels"])
            c_path_f1s.append(_f1(p_tp, p_fp, p_fn))
        c_macro = float(np.mean(c_path_f1s))

        # Rescue
        r_tp = sum(r["rescue_tp"] for r in b_records)
        r_fp = sum(r["rescue_fp"] for r in b_records)
        r_fn = sum(r["rescue_fn"] for r in b_records)
        r_micro = _f1(r_tp, r_fp, r_fn)
        r_sample = float(np.mean([r["rescue_sample_f1"] for r in b_records]))
        r_path_f1s = []
        for path in TARGET_PATHOLOGIES:
            p_tp = sum(1 for r in b_records if path in r["gold_labels"] and path in r["rescue_predicted_labels"])
            p_fp = sum(1 for r in b_records if path not in r["gold_labels"] and path in r["rescue_predicted_labels"])
            p_fn = sum(1 for r in b_records if path in r["gold_labels"] and path not in r["rescue_predicted_labels"])
            r_path_f1s.append(_f1(p_tp, p_fp, p_fn))
        r_macro = float(np.mean(r_path_f1s))

        resc_fn = sum(r["num_rescued_classifier_fn"] for r in b_records)
        add_fp = sum(r["num_added_below_threshold_fp"] for r in b_records)

        ctrl_micro_f1s.append(c_micro)
        ctrl_sample_f1s.append(c_sample)
        ctrl_macro_f1s.append(c_macro)

        resc_micro_f1s.append(r_micro)
        resc_sample_f1s.append(r_sample)
        resc_macro_f1s.append(r_macro)

        diff_micro_f1s.append(r_micro - c_micro)
        diff_sample_f1s.append(r_sample - c_sample)
        diff_macro_f1s.append(r_macro - c_macro)

        rescued_fn_counts.append(resc_fn)
        added_fp_counts.append(add_fp)
        net_gain_counts.append(resc_fn - add_fp)

    def _ci(arr: List[float | int]) -> Dict[str, Any]:
        np_arr = np.array(arr)
        lower, upper = float(np.percentile(np_arr, 2.5)), float(np.percentile(np_arr, 97.5))
        return {
            "mean": round(float(np.mean(np_arr)), 4),
            "ci_95": [round(lower, 4), round(upper, 4)],
            "contains_zero": bool(lower <= 0.0 <= upper),
        }

    return {
        "bootstrap_samples": n_samples,
        "seed": seed,
        "control": {
            "micro_f1": _ci(ctrl_micro_f1s),
            "sample_f1": _ci(ctrl_sample_f1s),
            "pathology_macro_f1": _ci(ctrl_macro_f1s),
        },
        "rescue": {
            "micro_f1": _ci(resc_micro_f1s),
            "sample_f1": _ci(resc_sample_f1s),
            "pathology_macro_f1": _ci(resc_macro_f1s),
            "rescued_classifier_fn_count": _ci(rescued_fn_counts),
            "added_below_threshold_fp_count": _ci(added_fp_counts),
            "net_gain_count": _ci(net_gain_counts),
        },
        "difference_rescue_minus_control": {
            "delta_micro_f1": _ci(diff_micro_f1s),
            "delta_sample_f1": _ci(diff_sample_f1s),
            "delta_pathology_macro_f1": _ci(diff_macro_f1s),
        },
    }


def write_rescue_per_study_csv(output_path: Path, records: List[Dict[str, Any]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "study_id",
        "image_name",
        "gold_labels",
        "classifier_selected_labels",
        "control_predicted_labels",
        "rescue_predicted_labels",
        "classifier_fn",
        "rescued_classifier_fn",
        "added_below_threshold_fp",
        "recovered_end_to_end_fn",
        "lost_previous_tp",
        "new_fp_vs_control",
        "removed_fp_vs_control",
        "num_classifier_fn",
        "num_rescued_classifier_fn",
        "num_added_below_threshold_fp",
        "num_recovered_end_to_end_fn",
        "num_lost_previous_tp",
        "num_new_fp_vs_control",
        "num_removed_fp_vs_control",
        "control_tp",
        "control_fp",
        "control_fn",
        "control_sample_f1",
        "rescue_tp",
        "rescue_fp",
        "rescue_fn",
        "rescue_sample_f1",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            row = dict(r)
            for k in [
                "gold_labels", "classifier_selected_labels", "control_predicted_labels",
                "rescue_predicted_labels", "classifier_fn", "rescued_classifier_fn",
                "added_below_threshold_fp", "recovered_end_to_end_fn", "lost_previous_tp",
                "new_fp_vs_control", "removed_fp_vs_control",
            ]:
                row[k] = "; ".join(row[k])
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute all-label rescue ablation comparison and bootstrap CIs.")
    parser.add_argument("--control-csv", type=Path, help="Path to selected_only/nlp_per_sample.csv")
    parser.add_argument("--rescue-csv", type=Path, help="Path to all_six/nlp_per_sample.csv")
    parser.add_argument("--control-dir", type=Path, help="Path to selected_only/llm directory")
    parser.add_argument("--rescue-dir", type=Path, help="Path to all_six/llm directory")
    parser.add_argument("--evidence-dir", type=Path, required=True, help="Directory containing evidence snapshot JSONs")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to save rescue comparison outputs")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260904)
    args = parser.parse_args()

    # Load data
    if args.control_csv and args.control_csv.is_file():
        control_data = load_arm_predictions_from_csv(args.control_csv)
    elif args.control_dir and args.control_dir.is_dir():
        control_data = load_arm_predictions_from_llm_dir(args.control_dir)
    else:
        raise ValueError("Must provide either valid --control-csv or --control-dir.")

    if args.rescue_csv and args.rescue_csv.is_file():
        rescue_data = load_arm_predictions_from_csv(args.rescue_csv)
    elif args.rescue_dir and args.rescue_dir.is_dir():
        rescue_data = load_arm_predictions_from_llm_dir(args.rescue_dir)
    else:
        raise ValueError("Must provide either valid --rescue-csv or --rescue-dir.")

    evidence_by_study = load_evidence_records(args.evidence_dir)

    records = compute_comparison_records(control_data, rescue_data, evidence_by_study)
    print(f"Computed comparison across {len(records)} matching studies.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_study_csv = args.output_dir / "rescue_per_study.csv"
    write_rescue_per_study_csv(per_study_csv, records)
    print(f"Saved: {per_study_csv}")

    ctrl_metrics = calculate_headline_metrics(records, "control")
    resc_metrics = calculate_headline_metrics(records, "rescue")

    total_clf_fn = sum(r["num_classifier_fn"] for r in records)
    total_resc_clf_fn = sum(r["num_rescued_classifier_fn"] for r in records)
    total_added_fp = sum(r["num_added_below_threshold_fp"] for r in records)
    total_rec_e2e_fn = sum(r["num_recovered_end_to_end_fn"] for r in records)
    total_lost_tp = sum(r["num_lost_previous_tp"] for r in records)
    total_new_fp = sum(r["num_new_fp_vs_control"] for r in records)
    total_rem_fp = sum(r["num_removed_fp_vs_control"] for r in records)

    rescue_prec = _safe_div(total_resc_clf_fn, total_resc_clf_fn + total_added_fp)
    rescue_rec = _safe_div(total_resc_clf_fn, total_clf_fn)

    # Per-pathology rescue details
    pathology_tradeoffs = {}
    for path in TARGET_PATHOLOGIES:
        p_clf_fn = sum(1 for r in records if path in r["classifier_fn"])
        p_resc_fn = sum(1 for r in records if path in r["rescued_classifier_fn"])
        p_add_fp = sum(1 for r in records if path in r["added_below_threshold_fp"])
        p_rec_e2e = sum(1 for r in records if path in r["recovered_end_to_end_fn"])
        p_lost_tp = sum(1 for r in records if path in r["lost_previous_tp"])
        pathology_tradeoffs[path] = {
            "gold_positives": sum(1 for r in records if path in r["gold_labels"]),
            "classifier_fn": p_clf_fn,
            "rescued_classifier_fn": p_resc_fn,
            "added_below_threshold_fp": p_add_fp,
            "recovered_end_to_end_fn": p_rec_e2e,
            "lost_previous_tp": p_lost_tp,
            "control_f1": ctrl_metrics["per_pathology"][path]["f1"],
            "rescue_f1": resc_metrics["per_pathology"][path]["f1"],
        }

    # Bootstrap
    bootstrap_results = run_paired_bootstrap(
        records, n_samples=args.bootstrap_samples, seed=args.seed
    )

    summary_payload = {
        "total_studies": len(records),
        "headline_counts": {
            "classifier_fn_count": total_clf_fn,
            "rescued_classifier_fn_count": total_resc_clf_fn,
            "added_below_threshold_fp_count": total_added_fp,
            "recovered_end_to_end_fn_count": total_rec_e2e_fn,
            "lost_previous_tp_count": total_lost_tp,
            "new_fp_vs_control_count": total_new_fp,
            "removed_fp_vs_control_count": total_rem_fp,
            "rescue_precision": round(rescue_prec, 4) if rescue_prec is not None else None,
            "rescue_recall": round(rescue_rec, 4) if rescue_rec is not None else None,
        },
        "control_selected_only": ctrl_metrics,
        "rescue_all_six": resc_metrics,
        "per_pathology_tradeoffs": pathology_tradeoffs,
        "paired_bootstrap": bootstrap_results,
    }

    summary_file = args.output_dir / "rescue_summary.json"
    summary_file.write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved: {summary_file}")


if __name__ == "__main__":
    main()
