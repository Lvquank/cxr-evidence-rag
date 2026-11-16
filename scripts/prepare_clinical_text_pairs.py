#!/usr/bin/env python3
"""Export aligned generated/reference Findings and Impression for clinical NLP evaluators.

The output is deliberately evaluator-neutral: CheXbert and RadGraph installations
have different APIs, but both need a study-level prediction/reference pairing.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def read_references(path: Path) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            study_id = str(row.get("study_id", row.get("uid", ""))).strip()
            if study_id:
                result[study_id] = {
                    "reference_findings": str(row.get("findings", "") or "").strip(),
                    "reference_impression": str(row.get("impression", "") or "").strip(),
                }
    return result


def latest_summary_by_study(llm_root: Path) -> dict[str, tuple[Path, dict[str, Any]]]:
    selected: dict[str, tuple[Path, dict[str, Any]]] = {}
    for path in llm_root.rglob("summary_*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        study_id = str(data.get("source", {}).get("study_id", "")).strip()
        if not study_id:
            continue
        current = selected.get(study_id)
        if current is None or path.name > current[0].name:
            selected[study_id] = (path, data)
    return selected


def export_pipeline(name: str, llm_root: Path, references: dict[str, dict[str, str]], output_dir: Path) -> dict[str, Any]:
    selected = latest_summary_by_study(llm_root)
    rows: list[dict[str, str]] = []
    missing_reference: list[str] = []
    for study_id, (path, data) in sorted(selected.items(), key=lambda item: item[0]):
        reference = references.get(study_id)
        if reference is None:
            missing_reference.append(study_id)
            continue
        prediction = data.get("prediction", {}) or {}
        rows.append({
            "pipeline": name,
            "study_id": study_id,
            "generated_findings": str(prediction.get("findings", "") or "").strip(),
            "reference_findings": reference["reference_findings"],
            "generated_impression": str(prediction.get("impression", "") or "").strip(),
            "reference_impression": reference["reference_impression"],
            "model": str(data.get("api", {}).get("model") or data.get("model") or ""),
            "source_summary": str(path.resolve()),
        })
    destination = output_dir / f"{name}_clinical_text_pairs.csv"
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["pipeline", "study_id"])
        writer.writeheader()
        writer.writerows(rows)
    return {
        "pipeline": name,
        "llm_root": str(llm_root.resolve()),
        "rows_exported": len(rows),
        "missing_reference_study_ids": missing_reference,
        "output_csv": str(destination.resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Create study-aligned input CSVs for CheXbert/RadGraph evaluation.")
    parser.add_argument("--pipeline", action="append", required=True, metavar="NAME=LLM_ROOT")
    parser.add_argument("--reports-csv", type=Path, default=ROOT / "src" / "data" / "indiana_reports.csv")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    references = read_references(args.reports_csv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    for value in args.pipeline:
        if "=" not in value:
            parser.error("Every --pipeline must be NAME=LLM_ROOT.")
        name, raw_root = value.split("=", 1)
        root = Path(raw_root)
        if not root.is_dir():
            raise FileNotFoundError(f"LLM output root does not exist: {root}")
        reports.append(export_pipeline(name, root, references, args.output_dir))
    summary = {
        "schema_version": 1,
        "purpose": "Evaluator-neutral, per-study Findings/Impression pairs for CheXbert or RadGraph.",
        "reports_csv": str(args.reports_csv.resolve()),
        "pipelines": reports,
        "caveat": "This export does not itself calculate a clinical factuality metric.",
    }
    (args.output_dir / "clinical_text_pairs_manifest.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
