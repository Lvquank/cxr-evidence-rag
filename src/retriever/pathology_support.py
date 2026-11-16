"""Rule-based pathology support from IU X-Ray metadata and report text.

This module implements retrieval relevance, not clinical diagnosis.  Exact terms,
strong supporting findings, and contextual findings deliberately receive different
scores so related pathology is not promoted as an exact label match.
"""

import re
from typing import Dict, Iterable, List, Mapping, Tuple


CHEXPERT_LABELS = (
    "Atelectasis",
    "Cardiomegaly",
    "Edema",
    "Pleural Effusion",
    "Pneumonia",
    "Pneumothorax",
)


# Scores follow the evaluation framework: exact=1.0, strong=0.5-0.8,
# and contextual co-occurrence=0.1-0.3.
SUPPORT_RULES = {
    "Atelectasis": {
        "exact": (1.0, (r"\batelecta(?:sis|tic)\b",)),
        "strong": (0.65, (r"\b(?:opacity compatible with atelectasis|low lung volume|hypoinflation|bronchovascular crowding|basilar linear opacity)\b",)),
        "context": (0.2, (r"\b(?:hypoinflation|crowding)\b",)),
    },
    "Cardiomegaly": {
        "exact": (1.0, (r"\bcardiomegaly\b", r"\b(?:enlarged heart|enlarged cardiac silhouette)\b")),
        "strong": (0.7, (r"\bprominent cardiac silhouette\b",)),
        "context": (0.2, (r"\b(?:pulmonary congestion|pleural effusion)\b",)),
    },
    "Edema": {
        "exact": (1.0, (r"\b(?:pulmonary )?edema\b",)),
        "strong": (0.65, (r"\b(?:pulmonary congestion|vascular congestion|vascular redistribution|pulmonary vascular indistinctness)\b",)),
        "context": (0.2, (r"\b(?:cardiomegaly|pleural effusion)\b",)),
    },
    "Pleural Effusion": {
        "exact": (1.0, (r"\bpleural effusions?\b", r"\b(?:right|left|bilateral) pleural effusions?\b")),
        "strong": (0.65, (r"\b(?:pleural fluid|blunted costophrenic angle|hydropneumothorax)\b",)),
        "context": (0.2, (r"\bpleural thickening\b",)),
    },
    "Pneumonia": {
        "exact": (1.0, (r"\bpneumonia\b", r"\bpneumonic\b")),
        "strong": (0.7, (r"\b(?:airspace disease|focal airspace consolidation|consolidation|infiltrate|lung opacity)\b",)),
        "context": (0.2, (r"\b(?:airspace opacity|pulmonary opacity)\b",)),
    },
    "Pneumothorax": {
        "exact": (1.0, (r"\bpneumothorax\b",)),
        "strong": (0.7, (r"\b(?:pleural air|hydropneumothorax)\b",)),
        "context": (0.2, (r"\b(?:chest tube|subcutaneous emphysema|trauma|post[- ]operative)\b",)),
    },
}

SOURCE_PRIORITIES = {
    "mesh": 1,
    "problems": 1,
    "disease": 1,
    "impression": 2,
    "findings": 3,
    "report": 3,
    "query_report": 2,
}
TAG_SOURCES = {"mesh", "problems", "disease"}

NEGATION_RE = re.compile(
    r"\b(?:no|without|absent|negative for|free of|no evidence of|no acute)\b(?:\W+\w+){0,5}\W*$",
    re.IGNORECASE,
)
UNCERTAINTY_RE = re.compile(
    r"\b(?:possible|possibly|may represent|may reflect|cannot exclude|questionable|suspected|likely|probable)\b(?:\W+\w+){0,5}\W*$",
    re.IGNORECASE,
)
POST_NEGATION_RE = re.compile(
    r"^\W*(?:is )?(?:not seen|not identified|absent)\b", re.IGNORECASE
)


def _sentence_bounds(text: str, start: int, end: int) -> Tuple[str, str]:
    before = text[max(text.rfind(mark, 0, start) for mark in ".;\n") + 1:start]
    after_candidates = [position for position in (text.find(mark, end) for mark in ".;\n") if position >= 0]
    after = text[end:min(after_candidates) if after_candidates else len(text)]
    return before[-120:], after[:48]


def _mention_state(text: str, start: int, end: int, source: str) -> str:
    if source in TAG_SOURCES:
        return "asserted"
    before, after = _sentence_bounds(text, start, end)
    if NEGATION_RE.search(before) or POST_NEGATION_RE.search(after):
        return "negated"
    if UNCERTAINTY_RE.search(before):
        return "uncertain"
    return "asserted"


def _iter_sources(metadata: Mapping[str, object]) -> Iterable[Tuple[str, str]]:
    keys_by_source = {
        "mesh": ("mesh", "Mesh", "MeSH"),
        "problems": ("problems", "Problems"),
        "disease": ("disease", "diagnosis", "pathology", "labels"),
        "impression": ("impression", "Impression"),
        "findings": ("findings", "Findings"),
        "report": ("report_text", "report", "final"),
    }
    seen = set()
    for source, keys in keys_by_source.items():
        for key in keys:
            value = str(metadata.get(key, "") or "").strip()
            if value and (source, value) not in seen:
                seen.add((source, value))
                yield source, value


def support_from_metadata(metadata: Mapping[str, object]) -> Tuple[Dict[str, float], List[Dict[str, object]]]:
    """Return a six-label support vector and auditable evidence records."""
    scores = {label: 0.0 for label in CHEXPERT_LABELS}
    evidence = []
    seen_evidence = set()

    for source, text in _iter_sources(metadata):
        for label, levels in SUPPORT_RULES.items():
            for level, (base_score, patterns) in levels.items():
                for pattern in patterns:
                    for match in re.finditer(pattern, text, re.IGNORECASE):
                        state = _mention_state(text, match.start(), match.end(), source)
                        if state == "negated":
                            continue
                        score = base_score * (0.5 if state == "uncertain" else 1.0)
                        key = (label, source, level, match.group(0).lower(), state)
                        if key in seen_evidence:
                            continue
                        seen_evidence.add(key)
                        scores[label] = max(scores[label], score)
                        evidence.append({
                            "label": label,
                            "source": source,
                            "source_priority": SOURCE_PRIORITIES[source],
                            "level": level,
                            "score": round(score, 3),
                            "state": state,
                            "matched_text": match.group(0),
                        })

    evidence.sort(key=lambda item: (item["label"], -item["score"], item["source_priority"]))
    return {label: round(scores[label], 3) for label in CHEXPERT_LABELS}, evidence


def support_from_report(report_text: str) -> Dict[str, float]:
    """Build a query support vector from direct/strong report evidence only.

    Context is useful for grading candidates, but treating it as a query pathology
    would create false query labels and dilute the support-overlap denominator.
    """
    _, evidence = support_from_metadata({"report_text": report_text})
    scores = {label: 0.0 for label in CHEXPERT_LABELS}
    for item in evidence:
        if item["level"] in {"exact", "strong"}:
            scores[item["label"]] = max(scores[item["label"]], item["score"])
    return scores


def pathology_support_score(query_support: Mapping[str, float], candidate_support: Mapping[str, float]) -> float:
    """Weighted support overlap from the evaluation framework, with equal label weights."""
    numerator = 0.0
    denominator = 0.0
    for label in CHEXPERT_LABELS:
        query_score = max(0.0, float(query_support.get(label, 0.0)))
        numerator += query_score * max(0.0, float(candidate_support.get(label, 0.0)))
        denominator += query_score
    return numerator / denominator if denominator else 0.0
