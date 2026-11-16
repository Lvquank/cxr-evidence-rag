"""Build compact v5 reference evidence for radiology report generation."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))
from src.classifier_thresholds import evidence_thresholds
REPORTS = ROOT / "src" / "data" / "indiana_reports.csv"
STATUS_CODES = {"SUPPORTED": "S", "MIXED": "M", "NON_SUPPORT_DOMINANT": "D", "INSUFFICIENT": "I"}
ALIASES = {
    "Atelectasis": (
        r"\batelecta(?:sis|tic)\b",
        r"\bvolume loss\b",
        r"\b(?:subsegmental )?collapse\b",
        r"\bbasilar linear opacit(?:y|ies)\b",
    ),
    "Cardiomegaly": (
        r"\bcardiomegaly\b",
        r"\benlarged (?:heart|cardiac silhouette)\b",
        r"\bheart enlargement\b",
        r"\benlargement of the (?:heart|cardiac silhouette)\b",
    ),
    "Edema": (
        r"\b(?:pulmonary |interstitial )?edema\b",
        r"\b(?:pulmonary|vascular) congestion\b",
    ),
    "Pleural Effusion": (
        r"\bpleural effusions?\b",
        r"\bpleural fluid\b",
        r"\bblunted costophrenic angles?\b",
        r"\bblunting of the (?:costophrenic )?angles?\b",
    ),
    "Pneumonia": (
        r"\bpneumoni(?:a|c)\b",
        r"\b(?:focal |airspace )?consolidation\b",
        r"\b(?:focal )?airspace disease\b",
        r"\binfiltrat(?:e|ion|es)\b",
    ),
    "Pneumothorax": (
        r"\bpneumothorax\b",
        r"\bpleural line\b",
        r"\bapical lucency\b",
        r"\bpleural air\b",
    ),
}
SURROGATE_NEGATIVES = {
    "Cardiomegaly": (r"\b(?:normal|within normal limits) (?:heart size|cardiac silhouette)\b",
                     r"\bheart size (?:is )?(?:normal|within normal limits)\b"),
}
NEGATION = re.compile(r"\b(?:no|without|absent|negative for|absence of|free of|not|neither)\b", re.I)
HEDGE = re.compile(r"\b(?:possible|possibly|may represent|may reflect|may be|may|cannot exclude|cannot be excluded|no definite|questionable|suspected|probable)\b", re.I)
POST_NEGATION = re.compile(r"^\s*(?:is |are )?(?:not seen|not identified|absent|unremarkable)\b", re.I)


def _round(value: Any) -> float:
    return round(float(value or 0.0), 3)


@lru_cache(maxsize=1)
def _reports() -> dict[str, dict[str, str]]:
    with REPORTS.open(encoding="utf-8-sig", newline="") as handle:
        return {str(row["uid"]): {key: row.get(key, "") or "" for key in ("findings", "impression")}
                for row in csv.DictReader(handle)}


def _sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+|[;\n]+", text) if part.strip()]


def _mentions(sentence: str, label: str) -> list[str]:
    states = []
    for index, pattern in enumerate(ALIASES[label]):
        for match in re.finditer(pattern, sentence, re.I):
            before = re.split(r"\b(?:but|however|although|though|yet|whereas)\b",
                              sentence[:match.start()], flags=re.I)[-1]
            after = sentence[match.end():]
            if HEDGE.search(before) or re.search(r"\bcannot be excluded\b", after, re.I):
                state = "uncertain"
            elif NEGATION.search(before) or POST_NEGATION.match(after):
                state = "explicit_negative" if index == 0 else "surrogate_negative"
            else:
                state = "positive"
            states.append(state)
    for pattern in SURROGATE_NEGATIVES.get(label, ()):
        if re.search(pattern, sentence, re.I):
            states.append("surrogate_negative")
    return states


def _polarity(sentences: list[str], label: str) -> str:
    states = {state for sentence in sentences for state in _mentions(sentence, label)}
    negative = bool(states & {"explicit_negative", "surrogate_negative"})
    if "positive" in states and negative:
        return "conflicted"
    if "positive" in states:
        return "positive"
    if "uncertain" in states:
        return "uncertain"
    if "explicit_negative" in states:
        return "explicit_negative"
    if "surrogate_negative" in states:
        return "surrogate_negative"
    return "unmentioned"


def _reference(case: dict[str, Any], label: str, support: bool = False) -> dict[str, Any]:
    report = _reports().get(str(case.get("study_id", "")), {})
    findings = report.get("findings", "").strip()
    impression = report.get("impression", "").strip()
    sentences = _sentences(findings) + _sentences(impression)
    if sentences:
        candidates = [sentence for sentence in sentences if _mentions(sentence, label)]
        relevant = min(candidates, key=len) if candidates else ""
        terms = case.get("mesh") or case.get("problems") or []
        finding = relevant or "; ".join(str(term) for term in terms if term) or (sentences[0] if sentences else "")
        polarity = _polarity(sentences, label)
    else:
        terms = case.get("mesh") or case.get("problems") or []
        finding = "; ".join(str(term) for term in terms if term)
        polarity = "unmentioned"
    result: dict[str, Any] = {
        "q": _round(case.get("final_rank_score", case.get("cosine_similarity"))),
        "f": finding[:240],
        "z": polarity,
    }
    if support and case.get("label_support") is not None:
        result["u"] = _round(case["label_support"])
    return result


def _status(item: dict[str, Any]) -> str:
    raw = str(item.get("evidence_status", "INSUFFICIENT")).upper()
    return STATUS_CODES.get(raw, raw if raw in {"S", "M", "D", "I"} else "I")


def _strength(item: dict[str, Any], margin: float | None, support: dict[str, Any] | None,
              non_support: dict[str, Any] | None) -> str:
    status = _status(item)
    if status == "I" or not support or item.get("evidence_available") is False or margin is None:
        return "insufficient"
    threshold = float((item.get("quality_checks") or {}).get("margin_threshold") or 0.02)
    if status == "D" or margin <= -threshold:
        return "non_support_favored"
    if margin >= threshold and support.get("u", 0) >= 0.5 and (
        not non_support or support["q"] >= non_support["q"]
    ):
        return "strong_support"
    if margin > 0:
        return "weak_support"
    if abs(margin) < threshold or status == "M":
        return "ambiguous"
    return "non_support_favored"


TARGET_PATHOLOGY_ORDER = (
    "Atelectasis",
    "Cardiomegaly",
    "Edema",
    "Pleural Effusion",
    "Pneumonia",
    "Pneumothorax",
)


def build_minimal_data(
    data: dict[str, Any],
    max_labels: int = 6,
    fallback_hits: int = 1,
    include_all_classifier_probabilities: bool = False,
    thresholds_path: str | Path | None = None,
) -> dict[str, Any]:
    """Return v5 compact evidence, optionally including 6-label classifier probabilities."""
    if max_labels < 1 or fallback_hits < 0:
        raise ValueError("max_labels must be >= 1 and fallback_hits must be >= 0")
    thresholds = evidence_thresholds(data, thresholds_path)
    classifier = data.get("classifier", {})
    selection_scores = classifier.get("selection_probabilities", classifier.get("raw_probabilities", {}))
    selected = set(data.get("classifier", {}).get("selected_labels") or [])
    records = data.get("retriever", {}).get("label_evidence", []) or []
    if not selected and "selected_labels" not in data.get("classifier", {}):
        selected = {item.get("label") for item in records}
    rows = []
    for item in records:
        label = item.get("label")
        if label not in selected:
            continue
        margin = (item.get("margins") or {}).get("M@5")
        supports = item.get("supporting_evidence") or []
        contrasts = item.get("matched_non_support_evidence") or []
        support = _reference(supports[0], label, True) if supports else None
        contrast = _reference(contrasts[0], label) if contrasts else None
        quality = item.get("quality_checks") or {}
        row: dict[str, Any] = {
            "l": label,
            "p": _round(item.get("classifier_probability",
                                 selection_scores.get(label))),
            "t": _round(thresholds[label]),
            "s": _status(item),
            "m": _round(margin) if margin is not None else None,
            "e": _strength(item, margin, support, contrast),
            "c": "stable" if all((quality.get("m3_same_sign_as_m5") is True,
                                  quality.get("m10_strong_reversal") is False,
                                  quality.get("m20_warning") is False)) else "unstable",
        }
        if support:
            row["a"] = support
        if contrast:
            row["n"] = contrast
        score_source = (classifier.get("score_sources") or {}).get(label, "densenet")
        if score_source != "densenet":
            row["o"] = score_source
        rows.append(row)
    priority = {"S": 0, "M": 1, "D": 2, "I": 3}
    rows.sort(key=lambda row: (priority[row["s"]], -row["p"], row["l"]))
    final_f = rows[:min(max_labels, 6)]

    result: dict[str, Any] = {"v": 5}
    if include_all_classifier_probabilities:
        raw_probs = classifier.get("selection_probabilities", classifier.get("raw_probabilities"))
        if not isinstance(raw_probs, dict):
            raise ValueError("Classifier raw_probabilities dictionary missing from input data.")
        retrieved_set = {r["l"] for r in final_f}
        clf_rows = []
        for path_label in TARGET_PATHOLOGY_ORDER:
            if path_label not in raw_probs:
                raise ValueError(f"Required label '{path_label}' missing from classifier raw_probabilities.")
            if path_label not in thresholds:
                raise ValueError(f"Required label '{path_label}' missing from adaptive thresholds.")
            try:
                p_val = float(raw_probs[path_label])
                t_val = float(thresholds[path_label])
            except (TypeError, ValueError) as err:
                raise ValueError(f"Invalid numeric score for label '{path_label}': {err}")
            import math
            if not math.isfinite(p_val) or not (0.0 <= p_val <= 1.0):
                raise ValueError(f"Probability for '{path_label}' must be finite and within [0, 1], got {p_val}")
            if not math.isfinite(t_val) or not (0.0 <= t_val <= 1.0):
                raise ValueError(f"Threshold for '{path_label}' must be finite and within [0, 1], got {t_val}")
            is_selected = bool(path_label in selected)
            if (p_val >= t_val) != is_selected:
                import logging
                logging.getLogger(__name__).warning(
                    "Provenance check discrepancy for '%s': p=%.6f, t=%.6f (p >= t is %s), but stored selected is %s",
                    path_label, p_val, t_val, (p_val >= t_val), is_selected,
                )
            is_retrieved = bool(path_label in retrieved_set)
            clf_rows.append({
                "l": path_label,
                "p": p_val,
                "t": t_val,
                "selected": is_selected,
                "retrieved": is_retrieved,
            })
        result["clf"] = clf_rows

    result["f"] = final_f
    if not result["f"] and fallback_hits:
        common = data.get("retriever", {}).get("common_evidence") or []
        result["g"] = [{key: val for key, val in _reference(case, "Pneumonia").items() if key != "z"}
                       for case in common[:1]]
    return result


def generate_minimal_json(input_file: str, output_file: str, top_k_common: int = 1,
                          top_k_label_evidence: int = 6, *, max_labels: int | None = None,
                          include_all_classifier_probabilities: bool = False,
                          thresholds_path: str | Path | None = None) -> dict[str, Any]:
    source, destination = Path(input_file), Path(output_file)
    data = json.loads(source.read_text(encoding="utf-8"))
    result = build_minimal_data(
        data,
        max_labels=max_labels if max_labels is not None else top_k_label_evidence,
        fallback_hits=top_k_common,
        include_all_classifier_probabilities=include_all_classifier_probabilities,
        thresholds_path=thresholds_path,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(destination), "bytes": destination.stat().st_size,
                      "reduction_percent": round(100 * (1 - destination.stat().st_size /
                                                         max(source.stat().st_size, 1)), 1)}))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Create compact v5 LLM reference evidence")
    parser.add_argument("--input", "-i", default="data/classifier_retriever_evidence.json")
    parser.add_argument("--output", "-o", default="data/classifier_retriever_evidence_minimal.json")
    parser.add_argument("--top-k-common", type=int, default=1)
    parser.add_argument("--top-k-label", type=int, default=6)
    parser.add_argument("--thresholds-json", type=Path, help="Selection-time artifact for legacy evidence without a threshold snapshot")
    parser.add_argument("--include-all-classifier-probabilities", action="store_true",
                        help="Include all six target pathology scores and thresholds under clf")
    args = parser.parse_args()
    generate_minimal_json(
        args.input, args.output, args.top_k_common, args.top_k_label,
        include_all_classifier_probabilities=args.include_all_classifier_probabilities,
        thresholds_path=args.thresholds_json,
    )


if __name__ == "__main__":
    main()
