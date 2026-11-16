"""Reset all pathology support for metadata rows marked as normal."""
import argparse
import json
from pathlib import Path

from src.retriever.pathology_support import CHEXPERT_LABELS


def is_normal_value(value) -> bool:
    if isinstance(value, list):
        return any(str(item).strip().lower() == "normal" for item in value)
    return str(value or "").strip().lower() == "normal"


def reset_metadata(path: Path) -> int:
    rows = []
    changed = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if is_normal_value(row.get("mesh")) or is_normal_value(row.get("problems")):
                zero_support = [0.0] * len(CHEXPERT_LABELS)
                if row.get("support") != zero_support:
                    row["support"] = zero_support
                    changed += 1
            rows.append(row)

    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    return changed


def main():
    parser = argparse.ArgumentParser(
        description="Set support to zero when mesh or problems contains normal."
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
        changed = reset_metadata(path)
        total_changed += changed
        print(f"{path}: reset {changed} rows")
    print(f"Total reset rows: {total_changed}")


if __name__ == "__main__":
    main()
