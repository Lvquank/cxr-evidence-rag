#!/usr/bin/env python3
"""Validate and canonicalize generated/reference report pairs across pipelines.

The validator is evaluator-neutral: it prepares one identical, study-aligned
input set for RadGraph, CheXbert, or other clinical text metrics.  It never
creates a reference report or silently replaces an empty reference.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


REQUIRED_COLUMNS = {
    "pipeline",
    "study_id",
    "generated_findings",
    "reference_findings",
    "generated_impression",
    "reference_impression",
}


def joined_report(findings: str, impression: str) -> str:
    return "\n".join(part.strip() for part in (findings, impression) if part and part.strip()).strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_pairs(path: Path) -> tuple[str, dict[str, dict[str, str]]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        missing = REQUIRED_COLUMNS.difference(fieldnames)
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path} contains no report pairs.")
    names = {str(row["pipeline"]).strip() for row in rows}
    if len(names) != 1 or not next(iter(names)):
        raise ValueError(f"{path} must contain exactly one non-empty pipeline name.")
    pipeline = next(iter(names))
    by_study: dict[str, dict[str, str]] = {}
    for raw in rows:
        study_id = str(raw["study_id"]).strip()
        if not study_id:
            raise ValueError(f"{path} contains an empty study_id.")
        if study_id in by_study:
            raise ValueError(f"{path} contains duplicate study_id={study_id}.")
        row = {column: str(raw.get(column, "") or "").strip() for column in reader.fieldnames or []}
        row["pipeline"] = pipeline
        row["study_id"] = study_id
        if not joined_report(row["generated_findings"], row["generated_impression"]):
            raise ValueError(f"Generated report is empty for {pipeline}/{study_id}.")
        by_study[study_id] = row
    return pipeline, by_study


def _study_sort_key(value: str) -> tuple[int, str]:
    return (0, f"{int(value):012d}") if value.isdigit() else (1, value)


def build_canonical_pairs(
    pairs_dir: Path,
    required_count: int | None = None,
    expected_pipelines: tuple[str, ...] | None = None,
    empty_reference_policy: str = "fail",
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Load all ``*_clinical_text_pairs.csv`` files and enforce identical studies."""
    files = sorted(pairs_dir.glob("*_clinical_text_pairs.csv"))
    if not files:
        raise FileNotFoundError(f"No *_clinical_text_pairs.csv files found in {pairs_dir}")
    loaded: dict[str, dict[str, dict[str, str]]] = {}
    inputs: dict[str, dict[str, str]] = {}
    for path in files:
        pipeline, rows = read_pairs(path)
        if pipeline in loaded:
            raise ValueError(f"Pipeline {pipeline!r} appears in more than one CSV.")
        loaded[pipeline] = rows
        inputs[pipeline] = {"path": str(path.resolve()), "sha256": sha256(path)}
    names = tuple(sorted(loaded))
    if expected_pipelines and set(names) != set(expected_pipelines):
        raise ValueError(f"Pipeline names differ. Expected {sorted(expected_pipelines)}, found {list(names)}")
    first_name = names[0]
    canonical_ids = set(loaded[first_name])
    for name in names[1:]:
        if set(loaded[name]) != canonical_ids:
            missing = sorted(canonical_ids.difference(loaded[name]), key=_study_sort_key)
            extra = sorted(set(loaded[name]).difference(canonical_ids), key=_study_sort_key)
            raise ValueError(f"Study set mismatch for {name}: missing={missing[:10]}, extra={extra[:10]}")
    if required_count is not None and len(canonical_ids) != required_count:
        raise ValueError(f"Expected {required_count} studies, found {len(canonical_ids)}.")

    excluded_empty_reference: list[str] = []
    for study_id in sorted(canonical_ids, key=_study_sort_key):
        reference = (
            loaded[first_name][study_id]["reference_findings"],
            loaded[first_name][study_id]["reference_impression"],
        )
        for name in names[1:]:
            other = (loaded[name][study_id]["reference_findings"], loaded[name][study_id]["reference_impression"])
            if other != reference:
                raise ValueError(f"Reference text differs between pipelines for study_id={study_id}.")
        if not joined_report(*reference):
            if empty_reference_policy == "fail":
                raise ValueError(f"Reference report is empty for study_id={study_id}.")
            excluded_empty_reference.append(study_id)
    kept_ids = [study_id for study_id in sorted(canonical_ids, key=_study_sort_key) if study_id not in excluded_empty_reference]
    canonical = [loaded[name][study_id] for name in names for study_id in kept_ids]
    manifest: dict[str, Any] = {
        "validation": "same study set, no duplicate IDs, identical references, non-empty joined reports",
        "pipeline_count": len(names),
        "pipelines": list(names),
        "study_count": len(kept_ids),
        "row_count": len(canonical),
        "study_ids": kept_ids,
        "empty_reference_policy": empty_reference_policy,
        "excluded_empty_reference_study_ids": excluded_empty_reference,
        "inputs": inputs,
    }
    return canonical, manifest


def write_canonical(canonical: list[dict[str, str]], manifest: dict[str, Any], output_csv: Path, output_manifest: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "pipeline", "study_id", "generated_findings", "reference_findings", "generated_impression", "reference_impression",
        "model", "source_summary",
    ]
    extra = sorted({key for row in canonical for key in row}.difference(fields))
    fields.extend(extra)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(canonical)
    manifest = {**manifest, "canonical_csv": str(output_csv.resolve()), "canonical_csv_sha256": sha256(output_csv)}
    output_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs-dir", type=Path, required=True)
    parser.add_argument("--required-count", type=int)
    parser.add_argument("--expected-pipeline", action="append", default=[])
    parser.add_argument("--empty-reference-policy", choices=("fail", "exclude"), default="fail")
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    args = parser.parse_args()
    canonical, manifest = build_canonical_pairs(
        args.pairs_dir,
        required_count=args.required_count,
        expected_pipelines=tuple(args.expected_pipeline) or None,
        empty_reference_policy=args.empty_reference_policy,
    )
    write_canonical(canonical, manifest, args.output_csv, args.output_manifest)
    print(json.dumps({"study_count": manifest["study_count"], "pipelines": manifest["pipelines"], "output_csv": str(args.output_csv)}, indent=2))


if __name__ == "__main__":
    main()
