"""Normalize saved research artefacts for display, without evaluating or changing them."""
from __future__ import annotations


def present_result(method, summary, evidence, image_index):
    classifier = evidence.get("classifier", {})
    probabilities = classifier.get("raw_probabilities", summary.get("classifier_probabilities", {}))
    thresholds = classifier.get("thresholds", evidence.get("retriever", {}).get("label_thresholds", summary.get("classifier_thresholds", {})))
    if not isinstance(thresholds, dict):
        thresholds = {}
    labels = classifier.get("selected_labels", summary.get("classifier_selected_labels", []))
    references = []
    for row in evidence.get("retriever", {}).get("label_evidence", []):
        def candidate(item):
            value = dict(item)
            if item.get("id"):
                value["reference_id"] = str(item["id"])
            return value
        references.append({"label": row["label"], "status": row.get("evidence_status", "INSUFFICIENT"),
                           "margin": row.get("margins", {}).get("M@5"),
                           "support": [candidate(x) for x in row.get("supporting_evidence", [])],
                           "non_support": [candidate(x) for x in row.get("matched_non_support_evidence", [])]})
    neighbors = summary.get("five_nearest_cases", summary.get("nearest_cases", summary.get("retrieval_summary", [])))
    return {"method": method, "image_index": image_index, "prediction": summary.get("prediction"),
            "probabilities": probabilities, "thresholds": thresholds, "selected_labels": labels,
            "evidence": references, "neighbors": neighbors, "route": summary.get("route", {"name": "vision_always", "image_attached": True}),
            "api": summary.get("api", summary.get("llm_metadata", {})),
            "model": summary.get("model", summary.get("requested_model")),
            "classifier_model": summary.get("classifier_model"),
            "classifier_provenance": summary.get("classifier_provenance"),
            "reasoning_effort": summary.get("reasoning_effort", "high"),
            "gate": evidence.get("learned_gate"), "status": "success"}
