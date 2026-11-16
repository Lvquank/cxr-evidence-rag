"""Utilities for keeping image-retrieval metadata complete and consistent."""

import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Mapping

from PIL import Image

from src.retriever.pathology_support import CHEXPERT_LABELS, support_from_metadata


def _text(value) -> str:
    if value is None:
        return ""
    try:
        if value != value:  # NaN without introducing a pandas dependency here.
            return ""
    except TypeError:
        pass
    return str(value).strip()


def _field_name(name: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")
    return value or "unnamed"


def source_columns(row) -> Dict[str, str]:
    """Preserve all source CSV values under stable, collision-free field names."""
    fields = {}
    for column, value in row.items():
        base_name = f"source_{_field_name(column)}"
        field_name = base_name
        suffix = 2
        while field_name in fields:
            field_name = f"{base_name}_{suffix}"
            suffix += 1
        fields[field_name] = _text(value)
    return fields


def image_details(image_path: Path) -> Dict[str, str]:
    """Return portable technical metadata for an already processed image."""
    with Image.open(image_path) as image:
        width, height = image.size
        return {
            "image_width": str(width),
            "image_height": str(height),
            "image_mode": image.mode,
            "image_format": image.format or image_path.suffix.lstrip(".").upper(),
        }


def _string_list(value) -> List[str]:
    if isinstance(value, list):
        return [_text(item) for item in value if _text(item)]
    text = _text(value)
    if not text:
        return []
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [_text(item) for item in parsed if _text(item)]
    except json.JSONDecodeError:
        pass
    return [item.strip() for item in text.split(";") if item.strip()]


def _support_list(row: Mapping[str, object]) -> List[float]:
    existing = row.get("support")
    if isinstance(existing, list) and len(existing) == len(CHEXPERT_LABELS):
        return [float(value) for value in existing]
    if existing:
        try:
            parsed = json.loads(str(existing))
            if isinstance(parsed, list) and len(parsed) == len(CHEXPERT_LABELS):
                return [float(value) for value in parsed]
        except json.JSONDecodeError:
            pass
    vector, _ = support_from_metadata(row)
    return [float(vector[label]) for label in CHEXPERT_LABELS]


def compact_metadata_row(row: Mapping[str, object]) -> Dict[str, object]:
    """Map raw IU metadata into the compact retrieval schema."""
    path = _text(row.get("path") or row.get("image_path")).replace("\\", "/")
    return {
        "id": _text(row.get("id")),
        "path": path,
        "study_id": _text(row.get("study_id") or row.get("uid")),
        "dataset": _text(row.get("dataset") or row.get("dataset_name")) or "IU-Xray",
        "view": _text(row.get("view") or row.get("projection")),
        # CSV source values are namespaced by source_columns() during ingestion.
        "mesh": _string_list(row.get("mesh") or row.get("MeSH") or row.get("source_mesh")),
        "problems": _string_list(
            row.get("problems") or row.get("Problems") or row.get("source_problems")
        ),
        "support": _support_list(row),
    }


def compact_metadata_rows(rows: Iterable[Mapping[str, object]]) -> List[Dict[str, object]]:
    return [compact_metadata_row(row) for row in rows]


def read_metadata_jsonl(path: Path) -> List[Dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_metadata_jsonl(rows: Iterable[Mapping[str, object]], path: Path):
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def enrich_metadata_rows(processed_dir: Path, rows: Iterable[Dict]) -> List[Dict[str, str]]:
    """Add derived fields while preserving all existing metadata values unchanged."""
    enriched = []
    for embedding_index, source_row in enumerate(rows):
        row = {key: _text(value) for key, value in source_row.items()}
        image_path = row.get("image_path", "")
        absolute_image_path = Path(image_path)
        if not absolute_image_path.is_absolute():
            absolute_image_path = processed_dir / absolute_image_path

        report = row.get("report_text", "")
        row.update({
            "dataset_name": row.get("dataset_name") or "IU-Xray",
            "embedding_index": str(embedding_index),
            "image_filename": absolute_image_path.name,
            "report_char_count": str(len(report)),
            "report_word_count": str(len(report.split())),
            "has_report": str(bool(report)).lower(),
        })
        if absolute_image_path.is_file():
            row.update(image_details(absolute_image_path))
        support_vector, support_evidence = support_from_metadata(row)
        row["support_vector"] = json.dumps(support_vector, sort_keys=True)
        row["support_evidence"] = json.dumps(support_evidence, ensure_ascii=True)
        for label in CHEXPERT_LABELS:
            row[f"support_{_field_name(label)}"] = str(support_vector[label])
        enriched.append(row)
    return enriched
