"""Copy compact label metadata into an embedding directory without rebuilding vectors."""

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.metadata import read_metadata_jsonl, write_metadata_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Synchronize mesh/problems/support fields by image ID without changing vectors or FAISS."
    )
    parser.add_argument("--source_metadata", required=True, help="Complete compact metadata.jsonl")
    parser.add_argument("--target_metadata", required=True, help="Embedding directory metadata.jsonl to update")
    args = parser.parse_args()

    source_rows = read_metadata_jsonl(Path(args.source_metadata))
    target_path = Path(args.target_metadata)
    target_rows = read_metadata_jsonl(target_path)
    labels_by_id = {
        str(row["id"]): (
            row.get("mesh", []),
            row.get("problems", []),
            row.get("support", [0.0] * 6),
        )
        for row in source_rows
        if row.get("id")
    }

    updated = 0
    missing = []
    for row in target_rows:
        image_id = str(row.get("id", ""))
        labels = labels_by_id.get(image_id)
        if labels is None:
            missing.append(image_id)
            continue
        mesh, problems, support = labels
        if (
            row.get("mesh") != mesh
            or row.get("problems") != problems
            or row.get("support") != support
        ):
            row["mesh"] = mesh
            row["problems"] = problems
            row["support"] = support
            updated += 1
    if missing:
        raise ValueError(f"{len(missing)} target IDs were absent from source metadata; first ID: {missing[0]}")
    write_metadata_jsonl(target_rows, target_path)
    print(f"Synchronized mesh/problems/support for {updated} rows in {target_path}.")


if __name__ == "__main__":
    main()
