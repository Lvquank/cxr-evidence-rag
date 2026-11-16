"""Audit frozen module outputs without training or calling an LLM."""

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from statistics import mean

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.generate_minimal_json_for_llm import build_minimal_data
from scripts.sweep_retrieval_lambda import dcg
from src.chexbert_metrics import compute_chexbert_aggregate

LABELS = json.loads((ROOT / "config/evaluation_ontology.json").read_text(encoding="utf-8"))["target_labels"]
RUNS = ["luna_high_evidence_v5_fresh_20260930", "luna_high_polarity_checklist_v5_206",
        "selective_thresholds_v1_206", "text_only_ablation_luna_high_206"]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def scores(tp, fp, fn):
    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": tp / (tp + fp) if tp + fp else 0,
            "recall": tp / (tp + fn) if tp + fn else 0,
            "micro_f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0}


def audit_run(name, requests, gold_by_study):
    root = ROOT / "outputs" / name
    result = read_json(root / "nlp_metrics_target6.json")
    records = result["records"]
    studies = [str(row["study_id"]) for row in records]
    if len(studies) != len(set(studies)) or set(studies) != set(requests):
        raise ValueError(f"Incomplete or duplicate cohort in {name}")
    rows, totals, routing, usage = [], Counter(), Counter(), Counter()
    old_mismatch_studies = set()
    for record in records:
        study = str(record["study_id"])
        item = requests[study]
        gold = set(record["gold_target_problems"])
        if gold != gold_by_study[study]:
            raise ValueError(f"Gold labels changed in {name}: {study}")
        evidence = read_json(root / "evidence" / f"{item['request_id']}.json")
        selected = set(evidence["classifier"]["selected_labels"])
        predicted = set(record["predicted_target_problems"])
        if not predicted <= selected:
            raise ValueError(f"Selected-only schema violated in {name}: {study}")
        threshold_path = ROOT / "config/densenet_selective_iu_thresholds_v1.json" if name.startswith("selective_") else None
        compact = build_minimal_data(evidence, thresholds_path=threshold_path)
        llm_dir = root / "llm" / item["request_id"]
        summary = read_json(llm_dir / record["summary_file"])
        saved_compact_path = llm_dir / record["summary_file"].replace("summary_", "minimal_")
        saved_compact = read_json(saved_compact_path) if saved_compact_path.exists() else None
        if saved_compact:
            old_thresholds = {row["l"]: row["t"] for row in saved_compact["f"]}
            mismatches = sum(old_thresholds.get(row["l"]) != row["t"] for row in compact["f"])
            totals["stored_compact_threshold_mismatch_rows"] += mismatches
            if mismatches:
                old_mismatch_studies.add(study)
        else:
            totals["missing_saved_compact"] += 1
        route = "vision" if summary["route"].get("image_attached") else "text_only"
        routing[route] += 1
        usage.update({key: int(value) for key, value in (summary.get("api", {}).get("usage") or {}).items()
                      if key in ("input_tokens", "output_tokens", "total_tokens")})
        evidence_by_label = {row["label"]: row for row in evidence["retriever"]["label_evidence"]}
        for label in LABELS:
            ev = evidence_by_label.get(label, {})
            row = {"run": name, "study_id": study, "label": label, "gold": label in gold,
                   "selected": label in selected, "emitted": label in predicted,
                   "state": ev.get("evidence_status", "NOT_SELECTED"), "route": route,
                   "margin": (ev.get("margins") or {}).get("M@5")}
            row["error_stage"] = ("classifier_fn" if row["gold"] and not row["selected"] else
                                  "post_selection_fn" if row["gold"] and not row["emitted"] else
                                  "final_fp" if row["emitted"] and not row["gold"] else "correct")
            rows.append(row)
            totals[row["error_stage"]] += 1
    classifier = scores(sum(r["gold"] and r["selected"] for r in rows),
                        sum(not r["gold"] and r["selected"] for r in rows), totals["classifier_fn"])
    final = scores(sum(r["gold"] and r["emitted"] for r in rows), totals["final_fp"],
                   totals["classifier_fn"] + totals["post_selection_fn"])
    stored = result["metrics"]["target_problems"]
    if any(final[key] != stored[key] for key in ("tp", "fp", "fn")):
        raise ValueError(f"Stored confusion counts do not match records: {name}")
    per_label = {}
    for label in LABELS:
        label_rows = [r for r in rows if r["label"] == label]
        counts = Counter(r["error_stage"] for r in label_rows)
        per_label[label] = {**stored["per_target_label"][label],
                            "classifier_fn": counts["classifier_fn"],
                            "post_selection_fn": counts["post_selection_fn"]}
    selected_rows = [r for r in rows if r["selected"]]
    states = {}
    for state in sorted({r["state"] for r in selected_rows}):
        group = [r for r in selected_rows if r["state"] == state]
        positives = sum(r["gold"] for r in group)
        states[state] = {"rows": len(group), "gold_positive": positives,
                         "positive_fraction": positives / len(group),
                         "missed_after_selection": sum(r["error_stage"] == "post_selection_fn" for r in group)}
    return {"studies": len(studies), "classifier": classifier, "final": final,
            "per_label": per_label, "sample_f1": stored["macro_f1"],
            "pathology_macro_f1": mean(v["f1"] for v in per_label.values()),
            "classifier_recall_ceiling_selected_only": classifier["recall"],
            "oracle_recover_post_selection_fn_at_fixed_fp": scores(
                final["tp"] + totals["post_selection_fn"], final["fp"], totals["classifier_fn"]),
            "stored_compact_threshold_mismatch_studies": len(old_mismatch_studies),
            "counts": dict(totals), "states": states, "routes": dict(routing), "api_tokens": dict(usage),
            "text_metrics": {section: result["metrics"][section] for section in ("findings", "impression")}}, rows


def audit_retrieval():
    results = {}
    for variant in ("projected", "raw"):
        root = ROOT / "outputs/flow1"
        if variant == "raw":
            root /= "raw_rad_dino_densenet"
        cache = read_json(root / "retrieval_similarity_cache.json")
        queries = cache["queries"]
        ceilings, no_hits, below_five = [], 0, 0
        for query in queries:
            candidates = query["candidates"]
            if any(str(c["study_id"]) == str(query["query_study_id"]) for c in candidates):
                raise ValueError("Query study leaked into cached retrieval candidates")
            relevance = sorted((float(c["relevance"]) for c in candidates), reverse=True)
            hits = sum(value >= 0.5 for value in relevance)
            no_hits += hits == 0
            below_five += hits < 5
            ideal = float(query["ideal_dcg"]["5"])
            ceilings.append({"P@5": min(hits, 5) / 5,
                             "nDCG@5": dcg(relevance[:5]) / ideal if ideal else 0})
        results[variant] = {"queries": len(queries), "zero_relevant_in_top50": no_hits,
                            "fewer_than_five_relevant_in_top50": below_five,
                            "oracle_reranking_top50": {k: mean(r[k] for r in ceilings) for k in ceilings[0]},
                            "lambda_sweep": read_json(root / "lambda_sweep_results.json")["summary"],
                            "note": "Oracle uses gold relevance, diagnostic ceiling only; cached global top50 is not the production bank union."}
    return results


def audit_clinical():
    root = ROOT / "outputs/luna_high_polarity_checklist_v5_206"
    pairs = pd.read_csv(root / "clinical_pairs/polarity_checklist_v5_clinical_text_pairs.csv")
    eligible = pairs[["reference_findings", "reference_impression"]].fillna("").apply(
        lambda row: any(str(v).strip() for v in row), axis=1).to_numpy()
    chex_root = root / "clinical_evaluation/chexbert_raw"
    reference = pd.read_csv(chex_root / "reference/labeled_reports.csv")
    # Cached labels must retain the exact report order before applying the cohort mask.
    text = pairs[["reference_findings", "reference_impression"]].fillna("").apply(
        lambda row: "\n".join(str(v).strip() for v in row if str(v).strip()), axis=1)
    if len(reference) != len(pairs) or reference.loc[eligible, "Report Impression"].tolist() != text[eligible].tolist():
        raise ValueError("Cached reference CheXbert report order/text mismatch")
    result = {"excluded_empty_reference_studies": pairs.loc[~eligible, "study_id"].astype(str).tolist(),
              "retained_studies": int(eligible.sum()), "pipelines": {}}
    result["reference_label_coverage"] = {}
    for label in LABELS:
        values = reference.loc[eligible, label]
        result["reference_label_coverage"][label] = {
            "positive": int((values == 1).sum()), "uncertain": int((values == -1).sum()),
            "explicit_negative": int((values == 0).sum()), "blank": int(values.isna().sum())}
    for pipeline in ("polarity_checklist", "fresh_v5_baseline"):
        prediction = pd.read_csv(chex_root / pipeline / "labeled_reports.csv")
        pair_name = "polarity_checklist_v5" if pipeline == "polarity_checklist" else pipeline
        generated_pairs = pd.read_csv(root / "clinical_pairs" / f"{pair_name}_clinical_text_pairs.csv")
        generated_text = generated_pairs[["generated_findings", "generated_impression"]].fillna("").apply(
            lambda row: "\n".join(str(v).strip() for v in row if str(v).strip()), axis=1)
        if (pairs["study_id"].tolist() != generated_pairs["study_id"].tolist() or
                prediction["Report Impression"].tolist() != generated_text.tolist()):
            raise ValueError(f"Cached prediction CheXbert report order/text mismatch: {pipeline}")
        result["pipelines"][pipeline] = {
            "target6": compute_chexbert_aggregate(reference.loc[eligible], prediction.loc[eligible], LABELS),
            "all14": compute_chexbert_aggregate(reference.loc[eligible], prediction.loc[eligible], list(reference.columns[1:]))}
    result["historical_metrics_206"] = read_json(root / "clinical_evaluation/clinical_metrics_summary.json")["pipelines"]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/module_bottleneck_audit_20261018")
    args = parser.parse_args()
    if (args.output_dir / "audit.json").exists():
        parser.error("Use a fresh --output-dir to preserve the audit already saved.")
    requests = {str(r["study_id"]): r for r in read_json(ROOT / "data/evaluations/leave_one_out_206_manifest.json")["requests"]}
    if len(requests) != 206:
        raise ValueError("Expected 206 unique frozen studies")
    baseline = read_json(ROOT / "outputs" / RUNS[0] / "nlp_metrics_target6.json")
    gold = {str(r["study_id"]): set(r["gold_target_problems"]) for r in baseline["records"]}
    output = {"schema_version": 1, "note": "Historical cohorts, descriptive diagnosis; no new independent test or model improvement claimed.", "runs": {}}
    all_rows = []
    for name in RUNS:
        output["runs"][name], rows = audit_run(name, requests, gold)
        all_rows.extend(rows)
    output["retrieval"] = audit_retrieval()
    output["clinical"] = audit_clinical()
    output["gate"] = read_json(ROOT / "outputs/learned_gate_nested_oof_206/nested_oof_metrics.json")["nested_oof"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "label_errors.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    (args.output_dir / "audit.json").write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output": str(args.output_dir), "runs": len(RUNS), "label_decisions": len(all_rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
