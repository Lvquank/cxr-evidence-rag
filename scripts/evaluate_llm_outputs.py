"""Evaluate structured LLM radiology outputs against IU-Xray reports.

It evaluates the ``prediction`` object saved by ``run_full_pipeline.py``:

* ``problems``: precision/recall/F1 and label hallucination rate against the
  report ``Problems`` column;
* ``findings`` and ``impression``: BLEU-4, ROUGE-L F1, and BERTScore P/R/F1.

Install the optional BERTScore dependency with ``pip install -r requirements.txt``.

New pipeline summaries should include ``source.study_id`` (pass ``--study-id``
when running the pipeline).  As a convenience, an original image basename is
joined through ``indiana_projections.csv`` and a processed ``<uuid>.png`` name
through metadata.jsonl.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
TOKEN_RE = re.compile(r"[a-z0-9]+")
DEFAULT_ONTOLOGY_PATH = ROOT / "config" / "evaluation_ontology.json"


def normalize_label(value: str) -> str:
    return " ".join(str(value).lower().split())


def split_problems(value: Any) -> List[str]:
    if isinstance(value, list):
        values = value
    else:
        values = str(value or "").split(";")
    unique: Dict[str, str] = {}
    for item in values:
        label = " ".join(str(item).split())
        if label:
            unique.setdefault(normalize_label(label), label)
    return list(unique.values())


def load_ontology(path: Path = DEFAULT_ONTOLOGY_PATH) -> Dict[str, Any]:
    """Load the frozen target-label ontology used for headline pathology metrics."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    targets = [str(label) for label in raw.get("target_labels", [])]
    aliases = raw.get("aliases", {})
    if not raw.get("version") or not targets or set(targets) != set(aliases):
        raise ValueError("Ontology must define a version and aliases for every target label.")
    alias_to_target: Dict[str, str] = {}
    for target in targets:
        for alias in [target, *aliases[target]]:
            normalized = normalize_label(alias)
            previous = alias_to_target.setdefault(normalized, target)
            if previous != target:
                raise ValueError(f"Ontology alias '{alias}' maps to both {previous} and {target}.")
    return {
        "version": str(raw["version"]),
        "target_labels": targets,
        "alias_to_target": alias_to_target,
        "normal_aliases": {normalize_label(value) for value in raw.get("normal_aliases", ["normal"])},
    }


def normalize_target_problems(values: Iterable[str], ontology: Dict[str, Any]) -> Dict[str, Any]:
    """Separate target pathologies, normal, and open-vocabulary findings."""
    targets: Dict[str, str] = {}
    other: Dict[str, str] = {}
    normal = False
    for raw in values:
        label = " ".join(str(raw).split())
        normalized = normalize_label(label)
        canonical = ontology["alias_to_target"].get(normalized)
        if canonical:
            targets.setdefault(canonical, canonical)
        elif normalized in ontology["normal_aliases"]:
            normal = True
        elif label:
            other.setdefault(normalized, label)
    return {
        "target_problems": [label for label in ontology["target_labels"] if label in targets],
        "normal": normal,
        "other_findings": list(other.values()),
    }


def tokens(text: Any) -> List[str]:
    return TOKEN_RE.findall(str(text or "").lower())


def _clipped_ngram_matches(prediction: Sequence[str], reference: Sequence[str], order: int) -> Tuple[int, int]:
    predicted = Counter(tuple(prediction[index:index + order]) for index in range(len(prediction) - order + 1))
    reference_counts = Counter(tuple(reference[index:index + order]) for index in range(len(reference) - order + 1))
    return sum((predicted & reference_counts).values()), sum(predicted.values())


def bleu4(prediction: str, reference: str) -> float:
    """Sentence BLEU-4 with add-one smoothing and effective order."""
    pred, ref = tokens(prediction), tokens(reference)
    if not pred or not ref:
        return float(pred == ref)
    precisions = []
    for order in range(1, min(4, len(pred)) + 1):
        matched, total = _clipped_ngram_matches(pred, ref, order)
        precisions.append((matched + 1) / (total + 1))
    geometric_mean = math.exp(sum(math.log(score) for score in precisions) / len(precisions))
    brevity_penalty = 1.0 if len(pred) >= len(ref) else math.exp(1 - len(ref) / len(pred))
    return brevity_penalty * geometric_mean


def lcs_length(left: Sequence[str], right: Sequence[str]) -> int:
    """LCS using one row of dynamic-programming state."""
    if len(left) < len(right):
        left, right = right, left
    previous = [0] * (len(right) + 1)
    for first in left:
        current = [0]
        for index, second in enumerate(right, start=1):
            current.append(previous[index - 1] + 1 if first == second else max(previous[index], current[-1]))
        previous = current
    return previous[-1]


def rouge_l_f1(prediction: str, reference: str) -> float:
    pred, ref = tokens(prediction), tokens(reference)
    if not pred or not ref:
        return float(pred == ref)
    common = lcs_length(pred, ref)
    precision, recall = common / len(pred), common / len(ref)
    return 0.0 if not precision + recall else 2 * precision * recall / (precision + recall)


def text_metrics(prediction: str, reference: str) -> Dict[str, Any]:
    """Score only when a textual reference exists; missing gold is not a zero score."""
    if not str(reference or "").strip():
        return {
            "included": False,
            "bleu4": None,
            "rouge_l_f1": None,
            "bertscore_precision": None,
            "bertscore_recall": None,
            "bertscore_f1": None,
        }
    return {
        "included": True,
        "bleu4": bleu4(prediction, reference),
        "rouge_l_f1": rouge_l_f1(prediction, reference),
        "bertscore_precision": None,
        "bertscore_recall": None,
        "bertscore_f1": None,
    }


def problem_metrics(predicted: Iterable[str], reference: Iterable[str]) -> Dict[str, float]:
    """Legacy exact-string metric retained only for backward-compatible callers.

    The batch evaluator uses ``target_problem_metrics`` below instead, because
    raw IU-Xray Problems labels have a broader ontology than this study.
    """
    prediction_by_normalized = {normalize_label(item): item for item in predicted}
    pred_set = set(prediction_by_normalized)
    ref_set = {normalize_label(item) for item in reference}
    tp, fp, fn = len(pred_set & ref_set), len(pred_set - ref_set), len(ref_set - pred_set)
    precision = tp / (tp + fp) if tp + fp else float(not ref_set)
    recall = tp / (tp + fn) if tp + fn else float(not pred_set)
    f1 = 0.0 if not precision + recall else 2 * precision * recall / (precision + recall)
    hallucinated = [prediction_by_normalized[label] for label in sorted(pred_set - ref_set)]
    return {
        "tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1,
        "hallucinated_labels": hallucinated,
        "hallucination_rate": fp / len(pred_set) if pred_set else 0.0,
    }


def target_problem_metrics(
    predicted: Dict[str, Any], reference: Dict[str, Any]
) -> Dict[str, Any]:
    """Compute six-label metrics after fixed synonym canonicalization.

    Open-vocabulary findings do not enter TP/FP/FN.  A normal gold case is
    correct only when the model predicts no target pathology; it is reported as
    a separate binary decision rather than fabricated as a seventh pathology.
    """
    predicted_set = set(predicted["target_problems"])
    reference_set = set(reference["target_problems"])
    tp, fp, fn = len(predicted_set & reference_set), len(predicted_set - reference_set), len(reference_set - predicted_set)
    precision = tp / (tp + fp) if tp + fp else float(not reference_set)
    recall = tp / (tp + fn) if tp + fn else float(not predicted_set)
    f1 = 0.0 if not precision + recall else 2 * precision * recall / (precision + recall)
    gold_normal = bool(reference["normal"] and not reference_set)
    predicted_normal = not predicted_set
    normal_decision = {
        "gold_normal": gold_normal,
        "predicted_normal": predicted_normal,
        "tp": int(gold_normal and predicted_normal),
        "fp": int(not gold_normal and predicted_normal),
        "fn": int(gold_normal and not predicted_normal),
        "tn": int(not gold_normal and not predicted_normal),
    }
    unmatched = sorted(predicted_set - reference_set)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "target_label_unmatched": unmatched,
        "target_label_unmatched_rate": fp / len(predicted_set) if predicted_set else 0.0,
        # This is a pathology-label proxy grounded in report Problems, not a
        # claim-level clinical factuality measurement.
        "target_label_groundedness": tp / len(predicted_set) if predicted_set else 1.0,
        "normal_decision": normal_decision,
        "predicted_other_findings": predicted["other_findings"],
        "reference_other_findings": reference["other_findings"],
    }


def load_reports(path: Path) -> Dict[str, Dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return {
            str(row["uid"]): {
                "problems": split_problems(row.get("Problems")),
                "findings": row.get("findings", "") or "",
                "impression": row.get("impression", "") or "",
            }
            for row in csv.DictReader(handle)
            if row.get("uid")
        }


def load_image_to_study(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    result: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            study_id = row.get("study_id")
            image_path = row.get("path")
            if study_id and image_path:
                result.setdefault(Path(image_path).name, str(study_id))
    return result


def load_projection_image_to_study(path: Path) -> Dict[str, str]:
    """Map original IU-Xray filenames (from indiana_projections.csv) to uid."""
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return {
            Path(row["filename"]).name: str(row["uid"])
            for row in csv.DictReader(handle)
            if row.get("uid") and row.get("filename")
        }


def get_prediction(summary: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    prediction = summary.get("prediction")
    if not isinstance(prediction, dict):
        # Supports old summaries if their raw response happened to be JSON.
        try:
            prediction = json.loads(summary.get("response", ""))
        except (TypeError, json.JSONDecodeError):
            return None
    if not isinstance(prediction, dict):
        return None
    findings, impression = prediction.get("findings"), prediction.get("impression")
    if not isinstance(findings, str) or not isinstance(impression, str):
        return None
    return {"problems": split_problems(prediction.get("problems")), "findings": findings, "impression": impression}


def add_bertscore(records: Sequence[Dict[str, Any]], model: str, batch_size: int, device: Optional[str]) -> Dict[str, Any]:
    """Attach BERTScore precision/recall/F1 to each text field in-place."""
    if not records:
        return {"model": model, "computed": False, "reason": "no_evaluated_samples", "sections": {}}
    try:
        from bert_score import score as bertscore_score
    except ImportError:
        for record in records:
            for section in ("findings", "impression"):
                record[section]["bertscore_precision"] = None
                record[section]["bertscore_recall"] = None
                record[section]["bertscore_f1"] = None
        return {
            "model": model,
            "computed": False,
            "reason": "dependency_not_installed",
            "sections": {},
        }

    sections: Dict[str, Any] = {}
    for section in ("findings", "impression"):
        eligible = [
            record for record in records
            if record[section].get("included", bool(str(record["reference_text"][section] or "").strip()))
        ]
        for record in records:
            record[section]["bertscore_precision"] = None
            record[section]["bertscore_recall"] = None
            record[section]["bertscore_f1"] = None
        sections[section] = {
            "eligible": len(eligible),
            "excluded_empty_reference": len(records) - len(eligible),
        }
        if not eligible:
            continue
        candidates = [record["prediction_text"][section] for record in eligible]
        references = [record["reference_text"][section] for record in eligible]
        precision, recall, f1 = bertscore_score(
            candidates, references, model_type=model, lang="en", batch_size=batch_size,
            device=device, verbose=False,
        )
        for index, record in enumerate(eligible):
            record[section]["bertscore_precision"] = float(precision[index])
            record[section]["bertscore_recall"] = float(recall[index])
            record[section]["bertscore_f1"] = float(f1[index])
    return {"model": model, "computed": any(info["eligible"] for info in sections.values()), "rescale_with_baseline": False, "sections": sections}


def corpus_bleu4(records: Sequence[Dict[str, Any]], section: str) -> float:
    matches = [0, 0, 0, 0]
    totals = [0, 0, 0, 0]
    candidate_length = reference_length = 0
    for record in records:
        predicted, reference = tokens(record["prediction_text"][section]), tokens(record["reference_text"][section])
        candidate_length += len(predicted)
        reference_length += len(reference)
        for order in range(1, 5):
            match, total = _clipped_ngram_matches(predicted, reference, order)
            matches[order - 1] += match
            totals[order - 1] += total
    if not candidate_length or not reference_length:
        return float(candidate_length == reference_length)
    active_orders = [index for index, total in enumerate(totals) if total]
    precisions = [(matches[index] + 1) / (totals[index] + 1) for index in active_orders]
    geometric_mean = math.exp(sum(math.log(score) for score in precisions) / len(precisions))
    brevity_penalty = 1.0 if candidate_length >= reference_length else math.exp(1 - reference_length / candidate_length)
    return brevity_penalty * geometric_mean


def evaluate_summaries(
    summary_files: Sequence[Path], reports: Dict[str, Dict[str, Any]], image_to_study: Dict[str, str],
    forced_study_id: Optional[str] = None, ontology: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, str]]]:
    ontology = ontology or load_ontology()
    records: List[Dict[str, Any]] = []
    skipped: List[Dict[str, str]] = []
    for summary_file in summary_files:
        try:
            summary = json.loads(summary_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            skipped.append({"summary_file": summary_file.name, "reason": f"invalid_summary: {error}"})
            continue
        prediction = get_prediction(summary)
        if prediction is None:
            skipped.append({"summary_file": summary_file.name, "reason": "missing_structured_prediction"})
            continue
        source = summary.get("source") if isinstance(summary.get("source"), dict) else {}
        image_name = source.get("image_name")
        study_id = source.get("study_id") or (forced_study_id if len(summary_files) == 1 else None)
        study_id = str(study_id) if study_id is not None else image_to_study.get(str(image_name or ""))
        gold = reports.get(study_id or "")
        if gold is None:
            skipped.append({"summary_file": summary_file.name, "reason": "study_id_not_resolved_or_missing_in_reports"})
            continue
        predicted_labels = normalize_target_problems(prediction["problems"], ontology)
        reference_labels = normalize_target_problems(gold["problems"], ontology)
        route = summary.get("route") if isinstance(summary.get("route"), dict) else {}
        evidence_states = sorted({str(state) for state in route.get("statuses", []) if str(state)})
        record = {
            "summary_file": summary_file.name,
            "study_id": study_id,
            "image_name": image_name or "",
            "predicted_problems_raw": prediction["problems"],
            "gold_problems_raw": gold["problems"],
            "predicted_target_problems": predicted_labels["target_problems"],
            "gold_target_problems": reference_labels["target_problems"],
            "problems": target_problem_metrics(predicted_labels, reference_labels),
            "findings": text_metrics(prediction["findings"], gold["findings"]),
            "impression": text_metrics(prediction["impression"], gold["impression"]),
            "evidence_states": evidence_states,
            "evidence_state_signature": "+".join(evidence_states) if evidence_states else "unknown",
            "prediction_text": {"findings": prediction["findings"], "impression": prediction["impression"]},
            "reference_text": {"findings": gold["findings"], "impression": gold["impression"]},
        }
        records.append(record)
    return records, skipped


def aggregate_target_problems(
    records: Sequence[Dict[str, Any]], target_labels: Optional[Sequence[str]] = None
) -> Dict[str, Any]:
    if not records:
        return {"evaluated": 0}
    target_labels = list(target_labels or load_ontology()["target_labels"])
    problem_totals = {key: sum(record["problems"][key] for record in records) for key in ("tp", "fp", "fn")}
    tp, fp, fn = problem_totals["tp"], problem_totals["fp"], problem_totals["fn"]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    micro_f1 = 0.0 if not precision + recall else 2 * precision * recall / (precision + recall)
    normal_counts = {
        key: sum(record["problems"]["normal_decision"][key] for record in records)
        for key in ("tp", "fp", "fn", "tn")
    }
    normal_precision = normal_counts["tp"] / (normal_counts["tp"] + normal_counts["fp"]) if normal_counts["tp"] + normal_counts["fp"] else 0.0
    normal_recall = normal_counts["tp"] / (normal_counts["tp"] + normal_counts["fn"]) if normal_counts["tp"] + normal_counts["fn"] else 0.0
    normal_f1 = 0.0 if not normal_precision + normal_recall else 2 * normal_precision * normal_recall / (normal_precision + normal_recall)
    per_target_label = {}
    for label in target_labels:
        label_tp = sum(label in record["predicted_target_problems"] and label in record["gold_target_problems"] for record in records)
        label_fp = sum(label in record["predicted_target_problems"] and label not in record["gold_target_problems"] for record in records)
        label_fn = sum(label not in record["predicted_target_problems"] and label in record["gold_target_problems"] for record in records)
        label_precision = label_tp / (label_tp + label_fp) if label_tp + label_fp else 0.0
        label_recall = label_tp / (label_tp + label_fn) if label_tp + label_fn else 0.0
        per_target_label[label] = {
            "support": label_tp + label_fn,
            "predicted": label_tp + label_fp,
            "tp": label_tp,
            "fp": label_fp,
            "fn": label_fn,
            "precision": label_precision,
            "recall": label_recall,
            "f1": 0.0 if not label_precision + label_recall else 2 * label_precision * label_recall / (label_precision + label_recall),
        }
    return {
        "evaluated": len(records),
        **problem_totals,
        "micro_precision": precision,
        "micro_recall": recall,
        "micro_f1": micro_f1,
        "sample_f1": mean(record["problems"]["f1"] for record in records),
        "pathology_macro_f1": mean(item["f1"] for item in per_target_label.values()),
        "macro_f1_definition": "Deprecated alias of sample_f1; use pathology_macro_f1 for label macro averaging.",
        "macro_f1": mean(record["problems"]["f1"] for record in records),
        "target_label_unmatched_rate": fp / (tp + fp) if tp + fp else 0.0,
        "macro_target_label_unmatched_rate": mean(record["problems"]["target_label_unmatched_rate"] for record in records),
        "target_label_groundedness": tp / (tp + fp) if tp + fp else 1.0,
        "normal_decision": {
            **normal_counts,
            "precision": normal_precision,
            "recall": normal_recall,
            "f1": normal_f1,
            "accuracy": (normal_counts["tp"] + normal_counts["tn"]) / len(records),
        },
        "per_target_label": per_target_label,
    }


def aggregate_text(records: Sequence[Dict[str, Any]], section: str) -> Dict[str, Any]:
    eligible = [record for record in records if record[section]["included"]]
    result: Dict[str, Any] = {
        "all_requests": len(records),
        "reference_nonempty": len(eligible),
        "reference_empty": len(records) - len(eligible),
        "reference_coverage": len(eligible) / len(records) if records else 0.0,
        "metrics_on_nonempty_reference": None,
    }
    if not eligible:
        return result
    metrics: Dict[str, Any] = {
        "bleu4": corpus_bleu4(eligible, section),
        "rouge_l_f1": mean(record[section]["rouge_l_f1"] for record in eligible),
    }
    for key in ("bertscore_precision", "bertscore_recall", "bertscore_f1"):
        values = [record[section][key] for record in eligible if record[section][key] is not None]
        metrics[key] = mean(values) if values else None
    result["metrics_on_nonempty_reference"] = metrics
    return result


def aggregate_by_evidence_state(
    records: Sequence[Dict[str, Any]], target_labels: Sequence[str]
) -> Dict[str, Any]:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for record in records:
        groups.setdefault(record["evidence_state_signature"], []).append(record)
    return {signature: aggregate_target_problems(group, target_labels) for signature, group in sorted(groups.items())}


def aggregate(records: Sequence[Dict[str, Any]], target_labels: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    if not records:
        return {"evaluated": 0}
    target_labels = list(target_labels or load_ontology()["target_labels"])
    return {
        "evaluated": len(records),
        "target_problems": aggregate_target_problems(records, target_labels),
        "findings": aggregate_text(records, "findings"),
        "impression": aggregate_text(records, "impression"),
        "by_evidence_state": aggregate_by_evidence_state(records, target_labels),
    }


def write_per_sample_csv(path: Path, records: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["summary_file", "study_id", "image_name", "evidence_states", "predicted_problems_raw", "gold_problems_raw", "predicted_target_problems", "gold_target_problems", "predicted_other_findings", "reference_other_findings", "target_label_unmatched", "problem_tp", "problem_fp", "problem_fn", "problem_precision", "problem_recall", "problem_f1", "problem_target_label_unmatched_rate", "problem_target_label_groundedness", "gold_normal", "predicted_normal", "findings_included", "findings_bleu4", "findings_rouge_l_f1", "findings_bertscore_precision", "findings_bertscore_recall", "findings_bertscore_f1", "impression_included", "impression_bleu4", "impression_rouge_l_f1", "impression_bertscore_precision", "impression_bertscore_recall", "impression_bertscore_f1"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            problems, findings, impression = record["problems"], record["findings"], record["impression"]
            writer.writerow({
                "summary_file": record["summary_file"], "study_id": record["study_id"], "image_name": record["image_name"],
                "evidence_states": "; ".join(record["evidence_states"]),
                "predicted_problems_raw": "; ".join(record["predicted_problems_raw"]), "gold_problems_raw": "; ".join(record["gold_problems_raw"]),
                "predicted_target_problems": "; ".join(record["predicted_target_problems"]), "gold_target_problems": "; ".join(record["gold_target_problems"]),
                "predicted_other_findings": "; ".join(problems["predicted_other_findings"]), "reference_other_findings": "; ".join(problems["reference_other_findings"]),
                "target_label_unmatched": "; ".join(problems["target_label_unmatched"]),
                **{f"problem_{key}": problems[key] for key in ("tp", "fp", "fn", "precision", "recall", "f1", "target_label_unmatched_rate", "target_label_groundedness")},
                "gold_normal": problems["normal_decision"]["gold_normal"], "predicted_normal": problems["normal_decision"]["predicted_normal"],
                "findings_included": findings["included"],
                **{f"findings_{key}": findings[key] for key in findings},
                "impression_included": impression["included"],
                **{f"impression_{key}": impression[key] for key in impression},
            })


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate LLM radiology JSON predictions against IU-Xray CSV reports.")
    parser.add_argument("--summaries-dir", type=Path, default=ROOT / "data/llm_responses")
    parser.add_argument("--summary-glob", default="summary_*.json")
    parser.add_argument("--reports-csv", type=Path, default=ROOT / "src/data/indiana_reports.csv")
    parser.add_argument("--projections-csv", type=Path, default=ROOT / "src/data/indiana_projections.csv")
    parser.add_argument("--metadata", type=Path, default=ROOT / "data/embeddings_chexpert_projection/metadata.jsonl")
    parser.add_argument("--study-id", help="Fallback uid, allowed only when exactly one summary is selected")
    parser.add_argument("--ontology-json", type=Path, default=DEFAULT_ONTOLOGY_PATH, help="Frozen target-label ontology for headline pathology metrics")
    parser.add_argument("--bertscore-model", default="roberta-large", help="Hugging Face model used by BERTScore")
    parser.add_argument("--bertscore-batch-size", type=int, default=8)
    parser.add_argument("--bertscore-device", help="For example cuda, cuda:0, or cpu; defaults to BERTScore auto-selection")
    parser.add_argument("--output-json", type=Path, default=ROOT / "data/evaluations/llm_nlp_metrics.json")
    parser.add_argument("--per-sample-csv", type=Path, default=ROOT / "data/evaluations/llm_nlp_per_sample.csv")
    args = parser.parse_args()

    summaries = sorted(args.summaries_dir.glob(args.summary_glob))
    reports = load_reports(args.reports_csv)
    image_to_study = load_projection_image_to_study(args.projections_csv)
    image_to_study.update(load_image_to_study(args.metadata))
    ontology = load_ontology(args.ontology_json)
    records, skipped = evaluate_summaries(summaries, reports, image_to_study, args.study_id, ontology)
    bertscore = add_bertscore(records, args.bertscore_model, args.bertscore_batch_size, args.bertscore_device)
    result = {
        "inputs": {"summaries_dir": str(args.summaries_dir), "summary_glob": args.summary_glob, "reports_csv": str(args.reports_csv), "projections_csv": str(args.projections_csv), "metadata": str(args.metadata)},
        "ontology": {
            "path": str(args.ontology_json),
            "version": ontology["version"],
            "target_labels": ontology["target_labels"],
            "normal_aliases": sorted(ontology["normal_aliases"]),
        },
        "metric_notes": {
            "target_problem_scope": "Only the frozen six-label ontology is used for headline pathology P/R/F1.",
            "unmatched_rate": "Target labels predicted but absent from canonicalized report Problems; label-level proxy only, not claim-level clinical hallucination.",
            "text_reference_policy": "Text metrics exclude empty reference fields and report coverage; an empty reference is never converted to a zero score.",
            "normal_policy": "A gold normal case is correct when no target pathology is predicted; normal is reported separately from the six pathology labels.",
        },
        "metrics": aggregate(records, ontology["target_labels"]), "bertscore": bertscore, "skipped": skipped, "records": records,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_per_sample_csv(args.per_sample_csv, records)
    print(json.dumps({"metrics": result["metrics"], "skipped": len(skipped), "output_json": str(args.output_json), "per_sample_csv": str(args.per_sample_csv)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
