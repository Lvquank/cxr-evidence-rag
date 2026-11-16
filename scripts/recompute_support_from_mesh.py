"""Recompute pathology support using only the metadata mesh field."""
import argparse
import json
from pathlib import Path

from src.retriever.pathology_support import CHEXPERT_LABELS, support_from_metadata


def recompute_metadata(path: Path) -> int:
    rows = []
    changed = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            support, _ = support_from_metadata({"mesh": row.get("mesh", [])})
            new_support = [float(support[label]) for label in CHEXPERT_LABELS]
            if row.get("support") != new_support:
                row["support"] = new_support
                changed += 1
            rows.append(row)

    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    return changed


def main():
    parser = argparse.ArgumentParser(
        description="Recompute support vectors from mesh only."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data"),
        help="Root directory to scan for metadata.jsonl files",
    )
    args = parser.parse_args()

    paths = sorted(args.root.rglob("metadata.jsonl"))
    if not paths:
        raise FileNotFoundError(f"No metadata.jsonl found under {args.root}")

    total_changed = 0
    for path in paths:
        changed = recompute_metadata(path)
        total_changed += changed
        print(f"{path}: recomputed {changed} rows")
    print(f"Total changed rows: {total_changed}")


if __name__ == "__main__":
    main()
