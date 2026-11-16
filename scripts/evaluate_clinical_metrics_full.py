#!/usr/bin/env python3
"""Run clinical NLP metrics (CheXbert and RadGraph) + paired Bootstrap CI

Evaluates polarity_checklist_v5 vs fresh_v5_baseline on the 206-study cohort.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party" / "CheXbert" / "src"))
sys.path.insert(0, str(ROOT))
from src.chexbert_metrics import joined_report, compute_chexbert_aggregate, compute_chexbert_per_sample

CHEXBERT_14_LABELS = [
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
    "No Finding",
]

CHEXBERT_HEADLINE_LABELS = [
    "Atelectasis",
    "Cardiomegaly",
    "Edema",
    "Pleural Effusion",
    "Pneumonia",
    "Pneumothorax",
]

RADGRAPH_SCORE_NAMES = ("radgraph_entity_f1", "radgraph_relation_f1", "radgraph_complete_f1")


def percentile_ci(values: np.ndarray) -> tuple[float, float]:
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def run_chexbert_labeler(reports_csv: Path, output_dir: Path, checkpoint: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    labeled_path = output_dir / "labeled_reports.csv"
    if labeled_path.is_file():
        cached = pd.read_csv(labeled_path)["Report Impression"].fillna("").tolist()
        current = pd.read_csv(reports_csv)["Report Impression"].fillna("").tolist()
        if cached != current:
            raise ValueError("CheXbert cache report text/order differs; use a fresh output directory.")
        print(f"Reusing existing CheXbert output: {labeled_path}")
        return labeled_path

    print(f"Running CheXbert inference on {reports_csv}...")
    from label import label, save_preds
    preds = label(str(checkpoint), str(reports_csv))
    save_preds(preds, str(reports_csv), str(output_dir))
    return labeled_path


def run_radgraph(hyps: list[str], refs: list[str], model_type: str = "radgraph-xl") -> tuple[np.ndarray, Any]:
    from radgraph import F1RadGraph

    print(f"Running RadGraph ({model_type}) on {len(hyps)} report pairs...")
    scorer = F1RadGraph(reward_level="all", model_type=model_type,
                        model_cache_dir=os.environ.get("RADGRAPH_MODEL_CACHE"))
    mean, scores, hyp_annos, ref_annos = scorer(hyps=hyps, refs=refs)
    score_arr = np.asarray(scores, dtype=float).T
    assert score_arr.shape == (len(hyps), 3), f"Expected shape ({len(hyps)}, 3), got {score_arr.shape}"
    return score_arr, {"hypothesis_annotations": hyp_annos, "reference_annotations": ref_annos}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checklist-pairs", type=Path, default=ROOT / "outputs/luna_high_polarity_checklist_v5_206/clinical_pairs/polarity_checklist_v5_clinical_text_pairs.csv")
    parser.add_argument("--baseline-pairs", type=Path, default=ROOT / "outputs/luna_high_polarity_checklist_v5_206/clinical_pairs/fresh_v5_baseline_clinical_text_pairs.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/luna_high_polarity_checklist_v5_206/clinical_evaluation")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260904)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    chexbert_checkpoint = ROOT / "models/chexbert/chexbert.pth"

    # 1. Load clinical text pairs
    df_chk = pd.read_csv(args.checklist_pairs)
    df_base = pd.read_csv(args.baseline_pairs)

    studies = df_chk["study_id"].astype(str).tolist()
    assert studies == df_base["study_id"].astype(str).tolist(), "Study ordering mismatch between pipelines"

    # Build report text for CheXbert and RadGraph
    ref_reports = [joined_report(r["reference_findings"], r["reference_impression"]) for _, r in df_chk.iterrows()]
    chk_reports = [joined_report(r["generated_findings"], r["generated_impression"]) for _, r in df_chk.iterrows()]
    base_reports = [joined_report(r["generated_findings"], r["generated_impression"]) for _, r in df_base.iterrows()]

    # 2. CheXbert inference
    chexbert_dir = args.output_dir / "chexbert_raw"
    chexbert_dir.mkdir(parents=True, exist_ok=True)

    ref_csv = chexbert_dir / "reference_reports.csv"
    chk_csv = chexbert_dir / "polarity_checklist_reports.csv"
    base_csv = chexbert_dir / "fresh_v5_baseline_reports.csv"

    pd.DataFrame({"Report Impression": ref_reports}).to_csv(ref_csv, index=False)
    pd.DataFrame({"Report Impression": chk_reports}).to_csv(chk_csv, index=False)
    pd.DataFrame({"Report Impression": base_reports}).to_csv(base_csv, index=False)

    ref_labeled_path = run_chexbert_labeler(ref_csv, chexbert_dir / "reference", chexbert_checkpoint)
    chk_labeled_path = run_chexbert_labeler(chk_csv, chexbert_dir / "polarity_checklist", chexbert_checkpoint)
    base_labeled_path = run_chexbert_labeler(base_csv, chexbert_dir / "fresh_v5_baseline", chexbert_checkpoint)

    ref_chex = pd.read_csv(ref_labeled_path)
    chk_chex = pd.read_csv(chk_labeled_path)
    base_chex = pd.read_csv(base_labeled_path)

    # Compute CheXbert aggregates
    chk_chex_14 = compute_chexbert_aggregate(ref_chex, chk_chex, CHEXBERT_14_LABELS)
    base_chex_14 = compute_chexbert_aggregate(ref_chex, base_chex, CHEXBERT_14_LABELS)

    chk_chex_h = compute_chexbert_aggregate(ref_chex, chk_chex, CHEXBERT_HEADLINE_LABELS)
    base_chex_h = compute_chexbert_aggregate(ref_chex, base_chex, CHEXBERT_HEADLINE_LABELS)

    # Compute per-sample CheXbert
    chk_acc_14, chk_f1_14 = compute_chexbert_per_sample(ref_chex, chk_chex, CHEXBERT_14_LABELS)
    base_acc_14, base_f1_14 = compute_chexbert_per_sample(ref_chex, base_chex, CHEXBERT_14_LABELS)

    chk_acc_h, chk_f1_h = compute_chexbert_per_sample(ref_chex, chk_chex, CHEXBERT_HEADLINE_LABELS)
    base_acc_h, base_f1_h = compute_chexbert_per_sample(ref_chex, base_chex, CHEXBERT_HEADLINE_LABELS)

    # 3. RadGraph inference
    radgraph_dir = args.output_dir / "radgraph_raw"
    radgraph_dir.mkdir(parents=True, exist_ok=True)

    chk_rad_scores, chk_rad_raw = run_radgraph(chk_reports, ref_reports, model_type="radgraph-xl")
    base_rad_scores, base_rad_raw = run_radgraph(base_reports, ref_reports, model_type="radgraph-xl")

    # 4. Save per-sample CSV
    per_sample_rows = []
    for i, sid in enumerate(studies):
        per_sample_rows.append({
            "study_id": sid,
            "chk_chexbert_all_label_acc": chk_acc_14[i],
            "chk_chexbert_all_sample_f1": chk_f1_14[i],
            "chk_chexbert_headline_acc": chk_acc_h[i],
            "chk_chexbert_headline_sample_f1": chk_f1_h[i],
            "chk_radgraph_entity_f1": chk_rad_scores[i, 0],
            "chk_radgraph_relation_f1": chk_rad_scores[i, 1],
            "chk_radgraph_complete_f1": chk_rad_scores[i, 2],
            "base_chexbert_all_label_acc": base_acc_14[i],
            "base_chexbert_all_sample_f1": base_f1_14[i],
            "base_chexbert_headline_acc": base_acc_h[i],
            "base_chexbert_headline_sample_f1": base_f1_h[i],
            "base_radgraph_entity_f1": base_rad_scores[i, 0],
            "base_radgraph_relation_f1": base_rad_scores[i, 1],
            "base_radgraph_complete_f1": base_rad_scores[i, 2],
        })
    df_per_sample = pd.DataFrame(per_sample_rows)
    df_per_sample.to_csv(args.output_dir / "clinical_per_sample.csv", index=False)

    # 5. Bootstrap Paired CI
    n = len(studies)
    rng = np.random.default_rng(args.seed)
    samples = rng.integers(0, n, size=(args.bootstrap_samples, n))

    # Metric functions for CheXbert micro/macro under sample resample
    # For sample-level metrics (Sample F1, RadGraph Entity/Relation/Complete F1)
    resample_metrics = {}

    for name, chk_vals, base_vals in [
        ("chexbert_all_sample_f1", chk_f1_14, base_f1_14),
        ("chexbert_headline_sample_f1", chk_f1_h, base_f1_h),
        ("radgraph_entity_f1", chk_rad_scores[:, 0], base_rad_scores[:, 0]),
        ("radgraph_relation_f1", chk_rad_scores[:, 1], base_rad_scores[:, 1]),
        ("radgraph_complete_f1", chk_rad_scores[:, 2], base_rad_scores[:, 2]),
    ]:
        chk_means = np.nanmean(chk_vals[samples], axis=1)
        base_means = np.nanmean(base_vals[samples], axis=1)
        deltas = chk_means - base_means
        low, high = percentile_ci(deltas)
        observed_delta = float(np.nanmean(chk_vals) - np.nanmean(base_vals))
        chk_low, chk_high = percentile_ci(chk_means)
        base_low, base_high = percentile_ci(base_means)
        resample_metrics[name] = {
            "chk_estimate": float(np.nanmean(chk_vals)),
            "chk_ci_95": [chk_low, chk_high],
            "base_estimate": float(np.nanmean(base_vals)),
            "base_ci_95": [base_low, base_high],
            "delta": observed_delta,
            "bootstrap_mean_delta": float(deltas.mean()),
            "ci_95": [low, high],
            "ci_contains_zero": bool(low <= 0.0 <= high),
        }

    # Resample CheXbert micro F1 and macro F1 (computed directly over sampled matrices)
    chk_micro_14_samples = []
    base_micro_14_samples = []
    chk_macro_14_samples = []
    base_macro_14_samples = []
    chk_micro_h_samples = []
    base_micro_h_samples = []
    chk_macro_h_samples = []
    base_macro_h_samples = []

    print("Running bootstrap resamples for CheXbert Micro/Macro F1...")
    for idx_row in samples:
        r_samp = ref_chex.iloc[idx_row].reset_index(drop=True)
        c_samp = chk_chex.iloc[idx_row].reset_index(drop=True)
        b_samp = base_chex.iloc[idx_row].reset_index(drop=True)

        c_14 = compute_chexbert_aggregate(r_samp, c_samp, CHEXBERT_14_LABELS)
        b_14 = compute_chexbert_aggregate(r_samp, b_samp, CHEXBERT_14_LABELS)
        c_h = compute_chexbert_aggregate(r_samp, c_samp, CHEXBERT_HEADLINE_LABELS)
        b_h = compute_chexbert_aggregate(r_samp, b_samp, CHEXBERT_HEADLINE_LABELS)

        chk_micro_14_samples.append(c_14["micro_f1"])
        base_micro_14_samples.append(b_14["micro_f1"])
        chk_macro_14_samples.append(c_14["macro_f1"])
        base_macro_14_samples.append(b_14["macro_f1"])

        chk_micro_h_samples.append(c_h["micro_f1"])
        base_micro_h_samples.append(b_h["micro_f1"])
        chk_macro_h_samples.append(c_h["macro_f1"])
        base_macro_h_samples.append(b_h["macro_f1"])

    for name, chk_obs, base_obs, chk_s, base_s in [
        ("chexbert_all_micro_f1", chk_chex_14["micro_f1"], base_chex_14["micro_f1"], np.array(chk_micro_14_samples), np.array(base_micro_14_samples)),
        ("chexbert_all_macro_f1", chk_chex_14["macro_f1"], base_chex_14["macro_f1"], np.array(chk_macro_14_samples), np.array(base_macro_14_samples)),
        ("chexbert_headline_micro_f1", chk_chex_h["micro_f1"], base_chex_h["micro_f1"], np.array(chk_micro_h_samples), np.array(base_micro_h_samples)),
        ("chexbert_headline_macro_f1", chk_chex_h["macro_f1"], base_chex_h["macro_f1"], np.array(chk_macro_h_samples), np.array(base_macro_h_samples)),
    ]:
        deltas = chk_s - base_s
        low, high = percentile_ci(deltas)
        chk_low, chk_high = percentile_ci(chk_s)
        base_low, base_high = percentile_ci(base_s)
        resample_metrics[name] = {
            "chk_estimate": float(chk_obs),
            "chk_ci_95": [chk_low, chk_high],
            "base_estimate": float(base_obs),
            "base_ci_95": [base_low, base_high],
            "delta": float(chk_obs - base_obs),
            "bootstrap_mean_delta": float(deltas.mean()),
            "ci_95": [low, high],
            "ci_contains_zero": bool(low <= 0.0 <= high),
        }

    # Summary JSON
    summary = {
        "schema_version": 1,
        "cohort": f"{n}-study leave-one-out cohort",
        "study_count": n,
        "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed,
        "pipelines": {
            "polarity_checklist_v5": {
                "chexbert_all_14": chk_chex_14,
                "chexbert_headline": chk_chex_h,
                "radgraph_mean": [float(v) for v in chk_rad_scores.mean(axis=0)],
            },
            "fresh_v5_baseline": {
                "chexbert_all_14": base_chex_14,
                "chexbert_headline": base_chex_h,
                "radgraph_mean": [float(v) for v in base_rad_scores.mean(axis=0)],
            },
        },
        "paired_comparisons_checklist_minus_baseline": resample_metrics,
    }

    (args.output_dir / "clinical_metrics_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # Generate Markdown Table
    md_lines = [
        f"# Clinical NLP Metrics & Paired Bootstrap CI (N={n})",
        "",
        "Comparison: `polarity_checklist_v5` vs `fresh_v5_baseline`",
        f"Bootstrap: {args.bootstrap_samples} paired study resamples (seed {args.seed}, 95% percentile CI).",
        "",
        "| Metric | fresh_v5_baseline | polarity_checklist_v5 | Delta (Checklist - Baseline) | 95% CI | Contains 0? |",
        "|---|---:|---:|---:|---:|:---:|",
    ]

    metric_display_names = [
        ("chexbert_all_micro_f1", "CheXbert 14-label Micro F1"),
        ("chexbert_all_macro_f1", "CheXbert 14-label Macro F1"),
        ("chexbert_headline_micro_f1", "CheXbert 6-headline Micro F1"),
        ("chexbert_headline_macro_f1", "CheXbert 6-headline Macro F1"),
        ("chexbert_all_sample_f1", "CheXbert 14-label Sample F1"),
        ("chexbert_headline_sample_f1", "CheXbert 6-headline Sample F1"),
        ("radgraph_entity_f1", "RadGraph Entity F1"),
        ("radgraph_relation_f1", "RadGraph Relation F1"),
        ("radgraph_complete_f1", "RadGraph Complete F1"),
    ]

    for key, label in metric_display_names:
        item = resample_metrics[key]
        b_est = f"{item['base_estimate']:.4f} [{item['base_ci_95'][0]:.4f}, {item['base_ci_95'][1]:.4f}]"
        c_est = f"{item['chk_estimate']:.4f} [{item['chk_ci_95'][0]:.4f}, {item['chk_ci_95'][1]:.4f}]"
        d = f"{item['delta']:+.4f}"
        ci = f"[{item['ci_95'][0]:+.4f}, {item['ci_95'][1]:+.4f}]"
        c0 = "Yes" if item["ci_contains_zero"] else "**No**"
        md_lines.append(f"| {label} | {b_est} | {c_est} | {d} | {ci} | {c0} |")

    md_content = "\n".join(md_lines) + "\n"
    (args.output_dir / "clinical_metrics_table.md").write_text(md_content, encoding="utf-8")
    print(md_content)


if __name__ == "__main__":
    main()
