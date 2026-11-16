"""Create a compact, group-disjoint CXR retrieval evaluation subset.

The output keeps embedding artifacts aligned with compact metadata.  Positive
cohorts are sampled per pathology using metadata support; normal cohorts have
an all-zero support vector.  By default a group is assigned to one cohort only.
"""

import argparse
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.retriever.pathology_support import CHEXPERT_LABELS


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def group_key(row: dict) -> tuple[str, str]:
    """Use patient_id when available; IU-Xray compact metadata falls back to study_id."""
    dataset = str(row.get("dataset", "") or "").strip()
    patient_id = str(row.get("patient_id", "") or "").strip()
    study_id = str(row.get("study_id", row.get("uid", "")) or "").strip()
    identifier = patient_id or study_id or f"image:{row.get('id', '')}"
    return dataset, identifier


def choose_representative(indices: list[int], rows: list[dict], label_index: int | None, rng: np.random.Generator) -> int:
    """Prefer the strongest support for a pathology; randomly break exact ties."""
    if label_index is None:
        scores = np.zeros(len(indices), dtype=np.float32)
    else:
        scores = np.asarray([rows[index]["support"][label_index] for index in indices], dtype=np.float32)
    best = scores.max()
    ties = [index for index, score in zip(indices, scores) if score == best]
    return int(rng.choice(ties))


def source_image_path(row: dict, processed_dir: Path) -> Path:
    value = row.get("path", row.get("image_path", ""))
    path = Path(str(value))
    return path if path.is_absolute() else processed_dir / path


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a group-disjoint balanced retrieval evaluation subset from embedding artifacts."
    )
    parser.add_argument("--input_dir", required=True, help="Embedding artifact directory with embeddings.npy, ids.txt and metadata.jsonl")
    parser.add_argument("--out_dir", required=True, help="New directory for the compact evaluation artifacts")
    parser.add_argument("--per_label", type=int, default=50, help="Maximum groups selected for each of six pathologies")
    parser.add_argument("--normal_count", type=int, default=50, help="Maximum all-zero-support groups selected as normal")
    parser.add_argument("--positive_support_threshold", type=float, default=0.2, help="Minimum support value for a pathology-positive group")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow_group_overlap", action="store_true", help="Allow one patient/study group to count for multiple pathology cohorts")
    parser.add_argument("--processed_dir", help="Required with --copy_images; root containing the relative image paths")
    parser.add_argument("--copy_images", action="store_true", help="Copy selected processed images into out_dir so the subset is portable")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing out_dir")
    args = parser.parse_args()

    if args.per_label < 1 or args.normal_count < 1:
        parser.error("--per_label and --normal_count must be positive.")
    if not 0.0 < args.positive_support_threshold <= 1.0:
        parser.error("--positive_support_threshold must be in (0, 1].")
    if args.copy_images and not args.processed_dir:
        parser.error("--processed_dir is required with --copy_images.")

    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    embeddings = np.load(input_dir / "embeddings.npy")
    rows = read_jsonl(input_dir / "metadata.jsonl")
    ids = (input_dir / "ids.txt").read_text(encoding="utf-8").splitlines()
    if embeddings.ndim != 2 or len(embeddings) != len(rows) or len(ids) != len(rows):
        raise ValueError("Input embeddings, metadata and IDs have inconsistent row counts.")
    if ids != [str(row.get("id", "")) for row in rows]:
        raise ValueError("ids.txt and metadata.jsonl are not in the same order.")
    support = np.asarray([row.get("support", []) for row in rows], dtype=np.float32)
    if support.shape != (len(rows), len(CHEXPERT_LABELS)):
        raise ValueError(f"Expected a {len(CHEXPERT_LABELS)}-value support vector for every row.")

    if out_dir.exists() and any(out_dir.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"Output directory is not empty: {out_dir}. Use --overwrite to replace it.")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[group_key(row)].append(index)

    rng = np.random.default_rng(args.seed)
    group_indices = list(groups.values())
    rng.shuffle(group_indices)
    assigned_groups: set[tuple[str, str]] = set()
    selected_indices: list[int] = []
    cohort_manifest: dict[str, dict] = {}

    # Process rarer labels first so common multi-label cases do not consume them.
    label_order = sorted(
        range(len(CHEXPERT_LABELS)),
        key=lambda label_index: sum(
            max(float(support[index, label_index]) for index in indices) >= args.positive_support_threshold
            for indices in group_indices
        ),
    )
    selected_by_label: dict[str, list[int]] = {}
    for label_index in label_order:
        label = CHEXPERT_LABELS[label_index]
        eligible_groups = [
            indices for indices in group_indices
            if max(float(support[index, label_index]) for index in indices) >= args.positive_support_threshold
            and (args.allow_group_overlap or group_key(rows[indices[0]]) not in assigned_groups)
        ]
        rng.shuffle(eligible_groups)
        chosen_groups = eligible_groups[:args.per_label]
        chosen_indices = [choose_representative(indices, rows, label_index, rng) for indices in chosen_groups]
        selected_by_label[label] = chosen_indices
        selected_indices.extend(chosen_indices)
        if not args.allow_group_overlap:
            assigned_groups.update(group_key(rows[indices[0]]) for indices in chosen_groups)
        cohort_manifest[label] = {
            "requested_groups": args.per_label,
            "available_groups": len(eligible_groups),
            "selected_groups": len(chosen_groups),
            "shortfall": max(0, args.per_label - len(chosen_groups)),
        }

    normal_groups = [
        indices for indices in group_indices
        if np.all(support[indices] == 0.0)
        and (args.allow_group_overlap or group_key(rows[indices[0]]) not in assigned_groups)
    ]
    normal_available_groups = len(normal_groups)
    rng.shuffle(normal_groups)
    normal_groups = normal_groups[:args.normal_count]
    normal_indices = [choose_representative(indices, rows, None, rng) for indices in normal_groups]
    selected_indices.extend(normal_indices)
    cohort_manifest["normal"] = {
        "requested_groups": args.normal_count,
        "available_groups": normal_available_groups,
        "selected_groups": len(normal_groups),
        "shortfall": max(0, args.normal_count - len(normal_groups)),
    }

    # Preserve a stable cohort order, then original row order within each cohort.
    unique_indices = []
    seen_indices = set()
    for index in selected_indices:
        if index not in seen_indices:
            seen_indices.add(index)
            unique_indices.append(index)
    selected_indices = unique_indices
    selected_rows = [rows[index] for index in selected_indices]
    np.save(out_dir / "embeddings.npy", embeddings[selected_indices])
    (out_dir / "ids.txt").write_text("\n".join(row["id"] for row in selected_rows) + "\n", encoding="utf-8")
    write_jsonl(out_dir / "metadata.jsonl", selected_rows)

    copied_images = 0
    missing_images = []
    if args.copy_images:
        processed_dir = Path(args.processed_dir)
        for row in selected_rows:
            source = source_image_path(row, processed_dir)
            relative = Path(str(row.get("path", row.get("image_path", ""))))
            target = out_dir / relative
            if not source.is_file():
                missing_images.append(str(source))
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            copied_images += 1

    manifest = {
        "schema_version": 1,
        "source_dir": str(input_dir.resolve()),
        "grouping": "dataset + patient_id when available; otherwise dataset + study_id",
        "positive_support_threshold": args.positive_support_threshold,
        "allow_group_overlap": args.allow_group_overlap,
        "seed": args.seed,
        "cohorts": {label: cohort_manifest[label] for label in [*CHEXPERT_LABELS, "normal"]},
        "selected_rows": len(selected_rows),
        "selected_unique_groups": len({group_key(row) for row in selected_rows}),
        "requested_max_rows": len(CHEXPERT_LABELS) * args.per_label + args.normal_count,
        "copied_images": copied_images,
        "missing_images": missing_images,
        "note": "With six pathology labels, per_label=50 and normal_count=50 requests at most 350 groups, not 400.",
    }
    (out_dir / "subset_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
