#!/usr/bin/env python3
"""Build the reproducible 206-query, leave-one-study-out test manifest.

The gallery is never rewritten.  The runner will pass each row's study_id to
the retriever as a temporary exclusion, then the same gallery is used again for
the next row.  Every selected study_id occurs only once in the manifest.
"""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List


ROOT = Path(__file__).resolve().parents[1]
LABELS = (
    "Atelectasis", "Cardiomegaly", "Edema", "Pleural Effusion",
    "Pneumonia", "Pneumothorax", "normal",
)
CHEXPERT_LABELS = LABELS[:-1]


def is_normal(row: Dict[str, Any]) -> bool:
    problems = {str(value).strip().lower() for value in row.get("problems", [])}
    return "normal" in problems and not any(float(value) > 0.0 for value in row.get("support", []))


def candidate_rows(metadata_path: Path, processed_dir: Path) -> Dict[str, List[Dict[str, Any]]]:
    """Return one valid frontal image candidate per study and label."""
    by_label_and_study: Dict[str, Dict[str, Dict[str, Any]]] = {label: {} for label in LABELS}
    with metadata_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            study_id = str(row.get("study_id", "")).strip()
            image_id = str(row.get("id", "")).strip()
            raw_image_path = Path(str(row.get("path", "")))
            image_path = raw_image_path if raw_image_path.is_absolute() else processed_dir / raw_image_path
            if not study_id or not image_id or not image_path.is_file():
                continue
            support = list(row.get("support", []))
            for index, label in enumerate(CHEXPERT_LABELS):
                score = float(support[index]) if index < len(support) else 0.0
                if score > 0.0:
                    by_label_and_study[label].setdefault(study_id, {
                        "study_id": study_id, "image_id": image_id, "image_path": str(image_path.resolve()),
                        "support_score": score,
                    })
            if is_normal(row):
                by_label_and_study["normal"].setdefault(study_id, {
                    "study_id": study_id, "image_id": image_id, "image_path": str(image_path.resolve()),
                    "support_score": 1.0,
                })
    return {label: list(rows.values()) for label, rows in by_label_and_study.items()}


def select_manifest(pool: Dict[str, List[Dict[str, Any]]], per_class: int, seed: int) -> List[Dict[str, Any]]:
    """Sample without replacement globally, prioritizing the smallest pools."""
    rng = random.Random(seed)
    used_studies = set()
    selections: List[Dict[str, Any]] = []
    # Pneumothorax is intentionally short at 26; all other labels have >=30.
    for label in sorted(LABELS, key=lambda item: (len(pool[item]), item)):
        candidates = list(pool[label])
        rng.shuffle(candidates)
        desired = min(per_class, len(candidates))
        picked = [candidate for candidate in candidates if candidate["study_id"] not in used_studies][:desired]
        if len(picked) != desired:
            raise RuntimeError(f"Cannot select {desired} unique studies for {label}; cross-label overlap exhausted the pool.")
        for candidate in picked:
            used_studies.add(candidate["study_id"])
            selections.append({"case_label": label, **candidate})
    if len(selections) != 206:
        raise RuntimeError(f"Expected the fixed 206-query protocol, selected {len(selections)}.")
    rng.shuffle(selections)
    for index, row in enumerate(selections, start=1):
        row["request_id"] = f"loo_{index:03d}_{row['case_label'].lower().replace(' ', '_')}"
    return selections


def main() -> None:
    parser = argparse.ArgumentParser(description="Create the fixed 206-query leave-one-study-out manifest.")
    parser.add_argument("--metadata", type=Path, default=ROOT / "data/embeddings_chexpert_projection/metadata.jsonl")
    parser.add_argument("--processed-dir", type=Path, default=ROOT / "data/reprocessed")
    parser.add_argument("--per-class", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--output", type=Path, default=ROOT / "data/evaluations/leave_one_out_206_manifest.json")
    args = parser.parse_args()
    if args.per_class != 30:
        raise ValueError("This protocol is fixed at 30 per class; Pneumothorax is automatically capped at its 26 available studies.")
    pool = candidate_rows(args.metadata, args.processed_dir)
    selections = select_manifest(pool, args.per_class, args.seed)
    payload = {
        "protocol": "leave-one-study-out; each query study is excluded only during its own retrieval and remains available for later queries",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "requested_per_class": args.per_class,
        "available_studies": {label: len(pool[label]) for label in LABELS},
        "selected_per_class": {label: sum(item["case_label"] == label for item in selections) for label in LABELS},
        "total_requests": len(selections),
        "requests": selections,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "total_requests": len(selections), "selected_per_class": payload["selected_per_class"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
