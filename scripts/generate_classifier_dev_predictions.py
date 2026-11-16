"""Score study-disjoint IU-Xray development images with the frozen CheXpert DenseNet."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.predict_chexpert_6labels import (
    DEFAULT_CHECKPOINT, EVAL_TRANSFORM, MAIN_LABELS, load_model,
    positive_probability_from_logits,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "classifier_dev_oof_predictions.csv")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260930)
    args = parser.parse_args()
    manifest_path = args.output.with_name(args.output.stem + "_manifest.json")
    if args.output.exists() or manifest_path.exists():
        parser.error("Development output already exists; choose a new path rather than overwrite it.")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive.")

    reports_path = ROOT / "src" / "data" / "indiana_reports.csv"
    metadata_path = ROOT / "data" / "reprocessed" / "metadata.jsonl"
    benchmark_path = ROOT / "data" / "evaluations" / "leave_one_out_206_manifest.json"
    ontology_path = ROOT / "config" / "evaluation_ontology.json"
    notebook_path = ROOT / "model" / "CheXpert_relation_aware_Pneumonia_Pneumothorax.ipynb"
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    held_out = {str(item["study_id"]) for item in benchmark["requests"]}
    ontology = json.loads(ontology_path.read_text(encoding="utf-8"))
    if tuple(ontology["target_labels"]) != MAIN_LABELS or len(held_out) != 206:
        raise ValueError("Frozen six-label ontology or 206-study benchmark changed.")
    with reports_path.open(newline="", encoding="utf-8-sig") as handle:
        reports = {str(row["uid"]): row for row in csv.DictReader(handle)}
    images: dict[str, list[dict]] = {}
    with metadata_path.open(encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            study_id = str(item["study_id"])
            if study_id in held_out or item.get("view") != "Frontal":
                continue
            path = ROOT / "data" / "reprocessed" / item["path"]
            if path.is_file() and study_id in reports and reports[study_id]["Problems"].strip():
                images.setdefault(study_id, []).append(item)
    studies = sorted(images, key=lambda value: (not value.isdigit(), int(value) if value.isdigit() else value))
    if not studies or set(studies) & held_out:
        raise ValueError("No study-disjoint development images available.")
    selected_images = {study_id: min(images[study_id], key=lambda item: item["id"]) for study_id in studies}
    aliases = {label: {label.casefold(), *(name.casefold() for name in ontology["aliases"][label])}
               for label in MAIN_LABELS}
    gold = {
        study_id: {label: int(bool({part.strip().casefold() for part in reports[study_id]["Problems"].split(";")} & aliases[label]))
                   for label in MAIN_LABELS}
        for study_id in studies
    }

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    model, checkpoint = load_model(args.checkpoint, device)
    checkpoint_hash = sha256(args.checkpoint)
    checkpoint_id = f"{args.checkpoint.name}@{checkpoint_hash[:12]}"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["study_id", "pathology", "raw_probability", "gold", "fold", "checkpoint_id"])
        writer.writeheader()
        with torch.inference_mode():
            for start in range(0, len(studies), args.batch_size):
                batch_ids = studies[start:start + args.batch_size]
                tensors = []
                for study_id in batch_ids:
                    image_path = ROOT / "data" / "reprocessed" / selected_images[study_id]["path"]
                    with Image.open(image_path) as source:
                        tensors.append(EVAL_TRANSFORM(source.convert("RGB")))
                outputs = model(torch.stack(tensors).to(device))
                scores = {label: positive_probability_from_logits(label, outputs[label], model.policies).cpu().tolist()
                          for label in MAIN_LABELS}
                for index, study_id in enumerate(batch_ids):
                    for label in MAIN_LABELS:
                        writer.writerow({"study_id": study_id, "pathology": label,
                                         "raw_probability": scores[label][index], "gold": gold[study_id][label],
                                         "fold": "independent_dev", "checkpoint_id": checkpoint_id})
                if (start // args.batch_size + 1) % 25 == 0 or start + args.batch_size >= len(studies):
                    print(f"Scored {min(start + args.batch_size, len(studies))}/{len(studies)} studies", flush=True)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "prediction_source": str(args.output.relative_to(ROOT)),
        "source_dataset": "IU-Xray reports and reprocessed frontal images, excluding all frozen 206 benchmark study IDs",
        "reference_labels": "Exact Problems aliases in config/evaluation_ontology.json; blank Problems excluded",
        "training_source_provenance": "The checkpoint training notebook loads CheXpert train.csv, splits CheXpert patients for training/development, and selects the E2 checkpoint on CheXpert development AUROC. IU-Xray studies are an independent development domain, not classifier-training samples.",
        "training_fold_mapping": "independent_dev for every IU-Xray study; the frozen checkpoint was not trained on IU-Xray",
        "checkpoint_id": checkpoint_id,
        "checkpoint_experiment": checkpoint.get("experiment_name"),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "seed": args.seed,
        "device": str(device),
        "batch_size": args.batch_size,
        "study_count": len(studies),
        "row_count": len(studies) * len(MAIN_LABELS),
        "positive_studies": dict(Counter({label: sum(gold[study_id][label] for study_id in studies) for label in MAIN_LABELS})),
        "study_to_image_id": {study_id: selected_images[study_id]["id"] for study_id in studies},
        "sha256": {name: sha256(path) for name, path in {
            "predictions": args.output, "reports": reports_path, "metadata": metadata_path,
            "benchmark": benchmark_path, "ontology": ontology_path, "checkpoint": args.checkpoint,
            "training_notebook": notebook_path, "generator_code": Path(__file__),
        }.items()},
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "manifest": str(manifest_path),
                      "studies": len(studies), "positives": manifest["positive_studies"]}))


if __name__ == "__main__":
    main()
