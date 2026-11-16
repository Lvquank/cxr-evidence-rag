"""Add complete derived metadata to an existing processed IU Xray metadata CSV."""

import argparse
import csv
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.metadata import enrich_metadata_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_dir", default="data/processed")
    parser.add_argument("--metadata_path", help="Defaults to <processed_dir>/metadata.csv")
    args = parser.parse_args()

    processed_dir = Path(args.processed_dir)
    metadata_path = Path(args.metadata_path) if args.metadata_path else processed_dir / "metadata.csv"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Metadata CSV was not found: {metadata_path}")

    with metadata_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    enriched = enrich_metadata_rows(processed_dir, rows)

    fieldnames = []
    for row in enriched:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with metadata_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(enriched)

    print(f"Enriched {len(enriched)} metadata rows in {metadata_path}")


if __name__ == "__main__":
    main()
