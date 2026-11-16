"""List metadata samples with a positive disease-label support."""
import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.retriever.pathology_support import CHEXPERT_LABELS


def load_metadata(metadata_path: str):
    path = Path(metadata_path)
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main():
    cfg = load_config()
    retrieval_cfg = cfg.get("retrieval", {})
    parser = argparse.ArgumentParser(
        description="List samples that have positive support for a disease label."
    )
    parser.add_argument(
        "--disease_label",
        required=True,
        choices=CHEXPERT_LABELS,
        help="Disease label to filter in metadata",
    )
    parser.add_argument(
        "--metadata_path",
        default=retrieval_cfg.get("metadata_path", "data/embeddings/metadata.jsonl"),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Maximum number of samples to print; 0 prints all matches",
    )
    args = parser.parse_args()

    if args.limit < 0:
        parser.error("--limit must be non-negative")

    label_index = CHEXPERT_LABELS.index(args.disease_label)
    rows = load_metadata(args.metadata_path)
    matches = [
        row for row in rows
        if isinstance(row.get("support"), list)
        and len(row["support"]) > label_index
        and float(row["support"][label_index]) > 0.0
    ]
    matches.sort(key=lambda row: float(row["support"][label_index]), reverse=True)
    selected = matches if args.limit == 0 else matches[:args.limit]

    print({
        "disease_label": args.disease_label,
        "metadata_path": args.metadata_path,
        "matched": len(matches),
        "returned": len(selected),
    })
    for rank, row in enumerate(selected, start=1):
        print({
            "rank": rank,
            "id": row.get("id"),
            "study_id": row.get("study_id", row.get("uid", "")),
            "label_support": row["support"][label_index],
            "path": row.get("path", row.get("image_path", "")),
            "metadata": row,
        })


if __name__ == "__main__":
    main()
