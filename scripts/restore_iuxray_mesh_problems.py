"""Restore compact IU X-Ray MeSH/Problems metadata from indiana_reports.csv.

This updates metadata only. It never changes processed images, embeddings, or FAISS.
"""

import argparse
import csv
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.metadata import read_metadata_jsonl, write_metadata_jsonl


def _items(value: str) -> list[str]:
    return [item.strip() for item in (value or "").split(";") if item.strip()]


def _study_key(value) -> str:
    text = str(value or "").strip()
    return text[:-2] if text.endswith(".0") else text


def load_report_labels(reports_csv: Path) -> dict[str, tuple[list[str], list[str]]]:
    with reports_csv.open(encoding="utf-8", newline="") as handle:
        rows = csv.DictReader(handle)
        required = {"uid", "MeSH", "Problems"}
        missing = required.difference(rows.fieldnames or [])
        if missing:
            raise ValueError(f"Reports CSV is missing columns: {sorted(missing)}")
        return {
            _study_key(row["uid"]): (_items(row["MeSH"]), _items(row["Problems"]))
            for row in rows
        }


def restore(metadata_path: Path, labels_by_study: dict[str, tuple[list[str], list[str]]]) -> tuple[int, int]:
    rows = read_metadata_jsonl(metadata_path)
    updated = 0
    missing = 0
    for row in rows:
        labels = labels_by_study.get(_study_key(row.get("study_id")))
        if labels is None:
            missing += 1
            continue
        mesh, problems = labels
        if row.get("mesh") != mesh or row.get("problems") != problems:
            row["mesh"] = mesh
            row["problems"] = problems
            updated += 1
    write_metadata_jsonl(rows, metadata_path)
    return updated, missing


def main() -> None:
    parser = argparse.ArgumentParser(description="Restore compact IU X-Ray MeSH/Problems fields without reprocessing images.")
    parser.add_argument("--reports_csv", required=True)
    parser.add_argument(
        "--metadata_path",
        action="append",
        required=True,
        help="Compact metadata.jsonl to update; repeat for processed and embedding metadata.",
    )
    args = parser.parse_args()
    labels_by_study = load_report_labels(Path(args.reports_csv))
    for raw_path in args.metadata_path:
        metadata_path = Path(raw_path)
        updated, missing = restore(metadata_path, labels_by_study)
        print(f"Updated {updated} rows in {metadata_path}; {missing} study IDs were not found.")


if __name__ == "__main__":
    main()
