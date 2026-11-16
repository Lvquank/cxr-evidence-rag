#!/usr/bin/env python3
"""Run clinical NLP metrics (CheXbert and RadGraph) + Bootstrap CIs for a target pipeline.

Computes point estimates, 95% bootstrap CIs, per-label metrics, and paired
comparisons against one or more baselines on the 206-study cohort.
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
    parser.add_argument("--target-pipeline", default="luna_high_evidence_v5_206")
    parser.add_argument("--target-pairs", type=Path, default=ROOT / "outputs/luna_high_evidence_v5_206/clinical_pairs/luna_high_evidence_v5_206_clinical_text_pairs.csv")
    parser.add_argument("--baseline-pairs", action="append", help="NAME=PATH_TO_CSV", default=[])
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/luna_high_evidence_v5_206/clinical_evaluation")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260904)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    chexbert_checkpoint = ROOT / "models/chexbert/chexbert.pth"

    # Default baselines if none specified
    if not args.baseline_pairs:
        args.baseline_pairs = [
            f"terra_medium_bottleneck_rerun={ROOT / 'outputs/luna_high_evidence_v5_206/clinical_pairs/terra_medium_bottleneck_rerun_clinical_text_pairs.csv'}",
            f"fresh_v5_baseline={ROOT / 'outputs/luna_high_evidence_v5_206/clinical_pairs/fresh_v5_baseline_clinical_text_pairs.csv'}",
        ]

    # Load target
    df_target = pd.read_csv(args.target_pairs)
    target_studies = df_target["study_id"].astype(str).tolist()

    # Parse baselines
    baselines: dict[str, pd.DataFrame] = {}
    for item in args.baseline_pairs:
        b_name, b_path = item.split("=", 1)
        b_df = pd.read_csv(Path(b_path))
        assert target_studies == b_df["study_id"].astype(str).tolist(), f"Study mismatch for baseline {b_name}"
        baselines[b_name] = b_df

    # Prepare texts
    ref_reports = [joined_report(r["reference_findings"], r["reference_impression"]) for _, r in df_target.iterrows()]
    target_reports = [joined_report(r["generated_findings"], r["generated_impression"]) for _, r in df_target.iterrows()]
    baseline_reports = {
        b_name: [joined_report(r["generated_findings"], r["generated_impression"]) for _, r in b_df.iterrows()]
        for b_name, b_df in baselines.items()
    }

    # Setup directories
    chexbert_dir = args.output_dir / "chexbert_raw"
    chexbert_dir.mkdir(parents=True, exist_ok=True)
    radgraph_dir = args.output_dir / "radgraph_raw"
    radgraph_dir.mkdir(parents=True, exist_ok=True)

    # Reusable reference CheXbert
    ref_csv = chexbert_dir / "reference_reports.csv"
    pd.DataFrame({"Report Impression": ref_reports}).to_csv(ref_csv, index=False)
    ref_labeled_path = run_chexbert_labeler(ref_csv, chexbert_dir / "reference", chexbert_checkpoint)
    ref_chex = pd.read_csv(ref_labeled_path)

    # Target CheXbert
    target_csv = chexbert_dir / f"{args.target_pipeline}_reports.csv"
    pd.DataFrame({"Report Impression": target_reports}).to_csv(target_csv, index=False)
    target_labeled_path = run_chexbert_labeler(target_csv, chexbert_dir / args.target_pipeline, chexbert_checkpoint)
    target_chex = pd.read_csv(target_labeled_path)

    # Baselines CheXbert
    baselines_chex = {}
    for b_name, b_rep in baseline_reports.items():
        b_csv = chexbert_dir / f"{b_name}_reports.csv"
        pd.DataFrame({"Report Impression": b_rep}).to_csv(b_csv, index=False)
        b_labeled_path = run_chexbert_labeler(b_csv, chexbert_dir / b_name, chexbert_checkpoint)
        baselines_chex[b_name] = pd.read_csv(b_labeled_path)

    # CheXbert aggregates
    target_chex_14 = compute_chexbert_aggregate(ref_chex, target_chex, CHEXBERT_14_LABELS)
    target_chex_h = compute_chexbert_aggregate(ref_chex, target_chex, CHEXBERT_HEADLINE_LABELS)
    target_acc_14, target_f1_14 = compute_chexbert_per_sample(ref_chex, target_chex, CHEXBERT_14_LABELS)
    target_acc_h, target_f1_h = compute_chexbert_per_sample(ref_chex, target_chex, CHEXBERT_HEADLINE_LABELS)

    baselines_chex_14 = {k: compute_chexbert_aggregate(ref_chex, v, CHEXBERT_14_LABELS) for k, v in baselines_chex.items()}
    baselines_chex_h = {k: compute_chexbert_aggregate(ref_chex, v, CHEXBERT_HEADLINE_LABELS) for k, v in baselines_chex.items()}
    baselines_sample_scores = {}
    for k, v in baselines_chex.items():
        acc_14, f1_14 = compute_chexbert_per_sample(ref_chex, v, CHEXBERT_14_LABELS)
        acc_h, f1_h = compute_chexbert_per_sample(ref_chex, v, CHEXBERT_HEADLINE_LABELS)
        baselines_sample_scores[k] = {
            "acc_14": acc_14,
            "f1_14": f1_14,
            "acc_h": acc_h,
            "f1_h": f1_h,
        }

    # RadGraph
    target_rad_scores, target_rad_raw = run_radgraph(target_reports, ref_reports, model_type="radgraph-xl")
    (radgraph_dir / f"{args.target_pipeline}_raw.json").write_text(json.dumps(target_rad_raw, indent=2, ensure_ascii=False), encoding="utf-8")

    baselines_rad_scores = {}
    for b_name, b_rep in baseline_reports.items():
        b_rad_scores, b_rad_raw = run_radgraph(b_rep, ref_reports, model_type="radgraph-xl")
        (radgraph_dir / f"{b_name}_raw.json").write_text(json.dumps(b_rad_raw, indent=2, ensure_ascii=False), encoding="utf-8")
        baselines_rad_scores[b_name] = b_rad_scores

    # Save per-sample CSV
    per_sample_rows = []
    for i, sid in enumerate(target_studies):
        row = {
            "study_id": sid,
            f"{args.target_pipeline}_chexbert_all_label_acc": target_acc_14[i],
            f"{args.target_pipeline}_chexbert_all_sample_f1": target_f1_14[i],
            f"{args.target_pipeline}_chexbert_headline_acc": target_acc_h[i],
            f"{args.target_pipeline}_chexbert_headline_sample_f1": target_f1_h[i],
            f"{args.target_pipeline}_radgraph_entity_f1": target_rad_scores[i, 0],
            f"{args.target_pipeline}_radgraph_relation_f1": target_rad_scores[i, 1],
            f"{args.target_pipeline}_radgraph_complete_f1": target_rad_scores[i, 2],
        }
        for b_name in baselines:
            row.update({
                f"{b_name}_chexbert_all_label_acc": baselines_sample_scores[b_name]["acc_14"][i],
                f"{b_name}_chexbert_all_sample_f1": baselines_sample_scores[b_name]["f1_14"][i],
                f"{b_name}_chexbert_headline_acc": baselines_sample_scores[b_name]["acc_h"][i],
                f"{b_name}_chexbert_headline_sample_f1": baselines_sample_scores[b_name]["f1_h"][i],
                f"{b_name}_radgraph_entity_f1": baselines_rad_scores[b_name][i, 0],
                f"{b_name}_radgraph_relation_f1": baselines_rad_scores[b_name][i, 1],
                f"{b_name}_radgraph_complete_f1": baselines_rad_scores[b_name][i, 2],
            })
        per_sample_rows.append(row)
    df_per_sample = pd.DataFrame(per_sample_rows)
    df_per_sample.to_csv(args.output_dir / "clinical_per_sample.csv", index=False)

    # Bootstrap CIs
    n = len(target_studies)
    rng = np.random.default_rng(args.seed)
    samples = rng.integers(0, n, size=(args.bootstrap_samples, n))

    # Single-pipeline bootstrap for target
    target_boot_samples = {}
    for name, vals in [
        ("chexbert_all_sample_f1", target_f1_14),
        ("chexbert_headline_sample_f1", target_f1_h),
        ("radgraph_entity_f1", target_rad_scores[:, 0]),
        ("radgraph_relation_f1", target_rad_scores[:, 1]),
        ("radgraph_complete_f1", target_rad_scores[:, 2]),
    ]:
        means = np.nanmean(vals[samples], axis=1)
        low, high = percentile_ci(means)
        target_boot_samples[name] = {
            "estimate": float(np.nanmean(vals)),
            "ci_95": [low, high],
            "boot_samples": means,
        }

    # Aggregate metric bootstrap
    target_micro_14_samples = []
    target_macro_14_samples = []
    target_micro_h_samples = []
    target_macro_h_samples = []

    print("Running bootstrap resamples for CheXbert Micro/Macro F1...")
    for idx_row in samples:
        r_samp = ref_chex.iloc[idx_row].reset_index(drop=True)
        t_samp = target_chex.iloc[idx_row].reset_index(drop=True)

        t_14 = compute_chexbert_aggregate(r_samp, t_samp, CHEXBERT_14_LABELS)
        t_h = compute_chexbert_aggregate(r_samp, t_samp, CHEXBERT_HEADLINE_LABELS)

        target_micro_14_samples.append(t_14["micro_f1"])
        target_macro_14_samples.append(t_14["macro_f1"])
        target_micro_h_samples.append(t_h["micro_f1"])
        target_macro_h_samples.append(t_h["macro_f1"])

    target_micro_14_arr = np.array(target_micro_14_samples)
    target_macro_14_arr = np.array(target_macro_14_samples)
    target_micro_h_arr = np.array(target_micro_h_samples)
    target_macro_h_arr = np.array(target_macro_h_samples)

    for name, obs, arr in [
        ("chexbert_all_micro_f1", target_chex_14["micro_f1"], target_micro_14_arr),
        ("chexbert_all_macro_f1", target_chex_14["macro_f1"], target_macro_14_arr),
        ("chexbert_headline_micro_f1", target_chex_h["micro_f1"], target_micro_h_arr),
        ("chexbert_headline_macro_f1", target_chex_h["macro_f1"], target_macro_h_arr),
    ]:
        low, high = percentile_ci(arr)
        target_boot_samples[name] = {
            "estimate": float(obs),
            "ci_95": [low, high],
            "boot_samples": arr,
        }

    # Single-pipeline bootstrap for baselines & paired comparisons
    baseline_boot_samples = {}
    paired_comparisons = {}

    for b_name in baselines:
        b_chex = baselines_chex[b_name]
        b_14 = baselines_chex_14[b_name]
        b_h = baselines_chex_h[b_name]
        b_scores = baselines_sample_scores[b_name]
        b_rad = baselines_rad_scores[b_name]

        b_boot = {}
        for name, vals in [
            ("chexbert_all_sample_f1", b_scores["f1_14"]),
            ("chexbert_headline_sample_f1", b_scores["f1_h"]),
            ("radgraph_entity_f1", b_rad[:, 0]),
            ("radgraph_relation_f1", b_rad[:, 1]),
            ("radgraph_complete_f1", b_rad[:, 2]),
        ]:
            means = np.nanmean(vals[samples], axis=1)
            low, high = percentile_ci(means)
            b_boot[name] = {
                "estimate": float(np.nanmean(vals)),
                "ci_95": [low, high],
                "boot_samples": means,
            }

        b_micro_14_s, b_macro_14_s = [], []
        b_micro_h_s, b_macro_h_s = [], []
        for idx_row in samples:
            r_samp = ref_chex.iloc[idx_row].reset_index(drop=True)
            bs_samp = b_chex.iloc[idx_row].reset_index(drop=True)
            res_14 = compute_chexbert_aggregate(r_samp, bs_samp, CHEXBERT_14_LABELS)
            res_h = compute_chexbert_aggregate(r_samp, bs_samp, CHEXBERT_HEADLINE_LABELS)
            b_micro_14_s.append(res_14["micro_f1"])
            b_macro_14_s.append(res_14["macro_f1"])
            b_micro_h_s.append(res_h["micro_f1"])
            b_macro_h_s.append(res_h["macro_f1"])

        for name, obs, arr in [
            ("chexbert_all_micro_f1", b_14["micro_f1"], np.array(b_micro_14_s)),
            ("chexbert_all_macro_f1", b_14["macro_f1"], np.array(b_macro_14_s)),
            ("chexbert_headline_micro_f1", b_h["micro_f1"], np.array(b_micro_h_s)),
            ("chexbert_headline_macro_f1", b_h["macro_f1"], np.array(b_macro_h_s)),
        ]:
            low, high = percentile_ci(arr)
            b_boot[name] = {
                "estimate": float(obs),
                "ci_95": [low, high],
                "boot_samples": arr,
            }

        baseline_boot_samples[b_name] = b_boot

        # Paired delta
        paired_dict = {}
        for m_name in [
            "chexbert_all_micro_f1",
            "chexbert_all_macro_f1",
            "chexbert_headline_micro_f1",
            "chexbert_headline_macro_f1",
            "chexbert_all_sample_f1",
            "chexbert_headline_sample_f1",
            "radgraph_entity_f1",
            "radgraph_relation_f1",
            "radgraph_complete_f1",
        ]:
            t_s = target_boot_samples[m_name]["boot_samples"]
            b_s = b_boot[m_name]["boot_samples"]
            deltas = t_s - b_s
            low, high = percentile_ci(deltas)
            obs_delta = float(target_boot_samples[m_name]["estimate"] - b_boot[m_name]["estimate"])
            paired_dict[m_name] = {
                "target_estimate": target_boot_samples[m_name]["estimate"],
                "target_ci_95": target_boot_samples[m_name]["ci_95"],
                "baseline_estimate": b_boot[m_name]["estimate"],
                "baseline_ci_95": b_boot[m_name]["ci_95"],
                "delta": obs_delta,
                "bootstrap_mean_delta": float(deltas.mean()),
                "ci_95": [low, high],
                "ci_contains_zero": bool(low <= 0.0 <= high),
            }
        paired_comparisons[f"{args.target_pipeline}_minus_{b_name}"] = paired_dict

    # Clean target bootstrap dict for JSON serialization (remove numpy arrays)
    clean_target_boot = {k: {"estimate": v["estimate"], "ci_95": v["ci_95"]} for k, v in target_boot_samples.items()}
    clean_baseline_boot = {
        b_name: {k: {"estimate": v["estimate"], "ci_95": v["ci_95"]} for k, v in b_data.items()}
        for b_name, b_data in baseline_boot_samples.items()
    }

    # Summary JSON
    summary = {
        "schema_version": 1,
        "cohort": f"{n}-study leave-one-out cohort",
        "study_count": n,
        "target_pipeline": args.target_pipeline,
        "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed,
        "target_metrics": {
            "chexbert_all_14": target_chex_14,
            "chexbert_headline": target_chex_h,
            "radgraph_mean": [float(v) for v in target_rad_scores.mean(axis=0)],
            "bootstrap_ci": clean_target_boot,
        },
        "baseline_metrics": {
            b_name: {
                "chexbert_all_14": baselines_chex_14[b_name],
                "chexbert_headline": baselines_chex_h[b_name],
                "radgraph_mean": [float(v) for v in baselines_rad_scores[b_name].mean(axis=0)],
                "bootstrap_ci": clean_baseline_boot[b_name],
            }
            for b_name in baselines
        },
        "paired_comparisons": paired_comparisons,
    }

    (args.output_dir / "clinical_metrics_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # Markdown Table
    md_lines = [
        f"# Clinical NLP Metrics & Bootstrap CI (N={n})",
        "",
        f"Target Pipeline: `{args.target_pipeline}`",
        f"Bootstrap: {args.bootstrap_samples} paired study resamples (seed {args.seed}, 95% percentile CI).",
        "",
        "## 1. Absolute Performance",
        "",
        "| Metric | Point Estimate | 95% Bootstrap CI |",
        "|---|---:|---:|",
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
        item = clean_target_boot[key]
        md_lines.append(f"| {label} | {item['estimate']:.4f} | [{item['ci_95'][0]:.4f}, {item['ci_95'][1]:.4f}] |")

    # Add Headline Per-Label Table
    md_lines.extend([
        "",
        "## 2. CheXbert 6-Headline Pathology Breakdown",
        "",
        "| Pathology | TP | FP | FN | F1 Score |",
        "|---|---:|---:|---:|---:|",
    ])
    for label in CHEXBERT_HEADLINE_LABELS:
        c = target_chex_h["per_label_counts"][label]
        md_lines.append(f"| {label} | {c['tp']} | {c['fp']} | {c['fn']} | {c['f1']:.4f} |")

    # Add Paired Comparisons
    for comp_name, comp_data in paired_comparisons.items():
        base_name = comp_name.replace(f"{args.target_pipeline}_minus_", "")
        md_lines.extend([
            "",
            f"## 3. Paired Comparison: `{args.target_pipeline}` vs `{base_name}`",
            "",
            f"| Metric | {base_name} | {args.target_pipeline} | Delta ({args.target_pipeline} - {base_name}) | 95% CI | Contains 0? |",
            "|---|---:|---:|---:|---:|:---:|",
        ])
        for key, label in metric_display_names:
            item = comp_data[key]
            b_est = f"{item['baseline_estimate']:.4f} [{item['baseline_ci_95'][0]:.4f}, {item['baseline_ci_95'][1]:.4f}]"
            t_est = f"{item['target_estimate']:.4f} [{item['target_ci_95'][0]:.4f}, {item['target_ci_95'][1]:.4f}]"
            d = f"{item['delta']:+.4f}"
            ci = f"[{item['ci_95'][0]:+.4f}, {item['ci_95'][1]:+.4f}]"
            c0 = "Yes" if item["ci_contains_zero"] else "**No**"
            md_lines.append(f"| {label} | {b_est} | {t_est} | {d} | {ci} | {c0} |")

    md_content = "\n".join(md_lines) + "\n"
    (args.output_dir / "clinical_metrics_table.md").write_text(md_content, encoding="utf-8")
    print("\n" + md_content)


if __name__ == "__main__":
    main()
