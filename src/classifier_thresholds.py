"""Shared frozen per-label DenseNet thresholds for every experiment arm."""

from __future__ import annotations

import json
import hashlib
import math
from pathlib import Path
from typing import Mapping


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_THRESHOLDS_PATH = ROOT / "config" / "densenet_adaptive_thresholds.json"


def load_classifier_thresholds(path: str | Path = DEFAULT_THRESHOLDS_PATH) -> tuple[dict[str, float], dict]:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    labels = payload.get("labels")
    if not payload.get("version") or not isinstance(labels, dict):
        raise ValueError(f"Invalid classifier threshold artifact: {source}")
    thresholds = {str(label): float(values["threshold"]) for label, values in labels.items()}
    if any(not 0.0 <= threshold <= 1.0 for threshold in thresholds.values()):
        raise ValueError(f"Classifier thresholds must be in [0, 1]: {source}")
    return thresholds, payload


def select_labels(probabilities: Mapping[str, float], thresholds: Mapping[str, float]) -> tuple[str, ...]:
    missing = set(thresholds).difference(probabilities)
    if missing:
        raise ValueError(f"Classifier probabilities are missing threshold labels: {sorted(missing)}")
    return tuple(label for label in thresholds if float(probabilities[label]) >= float(thresholds[label]))


def threshold_snapshot(thresholds: Mapping[str, float], version: str, path: str | Path | None = None) -> dict:
    values = {label: float(value) for label, value in thresholds.items()}
    digest = hashlib.sha256(json.dumps(values, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"thresholds": values, "threshold_version": version,
            "thresholds_sha256": digest, "thresholds_path": str(Path(path).resolve()) if path else None}


def evidence_thresholds(data: Mapping, path: str | Path | None = None) -> dict[str, float]:
    """Use the selection-time snapshot; legacy non-default versions need an explicit artifact."""
    classifier = data.get("classifier") or {}
    embedded = classifier.get("thresholds")
    if embedded is None:
        embedded = (data.get("retriever") or {}).get("label_thresholds")
    configured, artifact = load_classifier_thresholds(path or DEFAULT_THRESHOLDS_PATH)
    version = classifier.get("threshold_version")
    if path is not None or embedded is None:
        if version and version != artifact["version"]:
            raise ValueError(f"Threshold version {version!r} differs from {artifact['version']!r}; "
                             "provide the matching --thresholds-json or a selection-time snapshot.")
    values = {label: float(value) for label, value in embedded.items()} if embedded is not None else configured
    if set(values) != set(configured) or any(not math.isfinite(v) or not 0 <= v <= 1 for v in values.values()):
        raise ValueError("Evidence thresholds must contain all six labels with finite values in [0, 1].")
    if path is not None and values != configured:
        raise ValueError("Evidence threshold snapshot differs from --thresholds-json.")
    if embedded is not None and classifier.get("thresholds_sha256"):
        if threshold_snapshot(values, version or "")['thresholds_sha256'] != classifier["thresholds_sha256"]:
            raise ValueError("Evidence threshold snapshot hash mismatch.")
    if (embedded is not None or path is not None) and "raw_probabilities" in classifier:
        probabilities = classifier.get("selection_probabilities", classifier["raw_probabilities"])
        # A learned gate changes output labels after raw classifier thresholding.
        selected = set(classifier.get("rule_selected_labels") if data.get("learned_gate") else
                       classifier.get("selected_labels", (data.get("retriever") or {}).get("selected_labels")) or [])
        if data.get("learned_gate"):
            learned = data["learned_gate"]
            if set(classifier.get("selected_labels", [])) != set(learned["learned_selected_labels"]):
                raise ValueError("Selected labels disagree with learned-gate decisions.")
        if set(probabilities) != set(values) or not selected <= set(values):
            raise ValueError("Classifier probabilities/selected labels do not match threshold labels.")
        for label, threshold in values.items():
            probability = float(probabilities[label])
            if not math.isfinite(probability) or not 0 <= probability <= 1:
                raise ValueError(f"Invalid classifier probability for {label}.")
            # Evidence probabilities are rounded to six decimals after selection.
            if (probability >= threshold) != (label in selected) and abs(probability - threshold) > 0.0000005:
                raise ValueError(f"Selected label disagrees with its threshold: {label}.")
    return values
