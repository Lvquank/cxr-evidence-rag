"""Copy complete processed-image metadata into an existing embedding metadata CSV."""

import argparse
import csv
from pathlib import Path


def _read_csv(path: Path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_metadata", default="data/processed/metadata.csv")
    parser.add_argument("--embedding_metadata", default="data/embeddings/metadata.csv")
    args = parser.parse_args()

    processed_path = Path(args.processed_metadata)
    embedding_path = Path(args.embedding_metadata)
    processed_rows = _read_csv(processed_path)
    embedding_rows = _read_csv(embedding_path)
    processed_by_id = {row["id"]: row for row in processed_rows}
    embedding_ids = [row["id"] for row in embedding_rows]

    missing = [image_id for image_id in embedding_ids if image_id not in processed_by_id]
    if missing:
        raise ValueError(f"{len(missing)} embedding IDs do not exist in processed metadata.")
    if len(set(embedding_ids)) != len(embedding_ids):
        raise ValueError("Embedding metadata contains duplicate IDs.")

    merged_rows = []
    for index, embedding_row in enumerate(embedding_rows):
        merged = dict(processed_by_id[embedding_row["id"]])
        merged.update({key: value for key, value in embedding_row.items() if value})
        merged["embedding_index"] = str(index)
        merged_rows.append(merged)

    fieldnames = []
    for row in merged_rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with embedding_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(merged_rows)

    print(f"Synced {len(merged_rows)} rows to {embedding_path}")


if __name__ == "__main__":
    main()
