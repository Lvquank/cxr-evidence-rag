"""Generate complete 206-case evidence files using the frozen selective thresholds artifact.

Loads DenseNet, local reranker, and FAISS retriever once on GPU, then processes all
206 leave-one-study-out benchmark cases in a single fast session.
"""

import json
import sys
import time
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.predict_chexpert_6labels import MAIN_LABELS, load_model, predict
from src.classifier_thresholds import load_classifier_thresholds, threshold_snapshot
from src.config import load_config
from src.embeddings.encoder import DEFAULT_MODEL_NAME
from src.retriever.densenet_local_reranker import DenseNetLocalFeatureReranker
from src.retriever.faiss_retriever import FAISSImageRetriever
from src.retriever.pathology_support import CHEXPERT_LABELS
from scripts.test_classifier_to_retriever import (
    compact_score,
    compact_candidate,
    compact_label_evidence,
)

DEFAULT_MANIFEST = ROOT / "data" / "evaluations" / "leave_one_out_206_manifest.json"
DEFAULT_THRESHOLDS = ROOT / "config" / "densenet_selective_iu_thresholds_v1.json"
DEFAULT_OUT_DIR = ROOT / "outputs" / "selective_thresholds_v1_206" / "evidence"


def resolve_image_path(item: dict) -> Path:
    p = Path(item["image_path"])
    if p.is_file():
        return p
    relocated = ROOT / "data" / "reprocessed" / "images" / f"{item['image_id']}{p.suffix}"
    if relocated.is_file():
        return relocated
    raise FileNotFoundError(f"Image not found for {item['request_id']}: {p}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load thresholds
    thresholds, th_payload = load_classifier_thresholds(DEFAULT_THRESHOLDS)
    print(f"Loaded selective thresholds from {DEFAULT_THRESHOLDS.name}:")
    for k, v in thresholds.items():
        print(f"  {k}: {v}")

    # Load retrieval config
    app_cfg = load_config()
    retrieval_cfg = app_cfg.get("retrieval", {})
    model_cfg = app_cfg.get("model", {})

    # Load classifier
    ckpt_path = ROOT / "model" / "best_E2_relation_aware_weighted_bce.pt"
    print(f"Loading DenseNet from {ckpt_path.name}...")
    classifier, _ = load_model(ckpt_path, device)

    # Local reranker
    print("Initializing DenseNet local feature reranker...")
    local_reranker = DenseNetLocalFeatureReranker(
        classifier.backbone,
        device=device,
        batch_size=16,
    )

    # Retriever
    print("Initializing FAISS retriever...")
    retriever = FAISSImageRetriever(
        index_path=retrieval_cfg.get("index_path"),
        metadata_path=retrieval_cfg.get("metadata_path"),
        processed_dir=retrieval_cfg.get("processed_dir"),
        device=str(device),
        model_name=DEFAULT_MODEL_NAME,
        projection_checkpoint=model_cfg.get("pathology_projection_checkpoint"),
    )

    # Manifest
    manifest = json.loads(DEFAULT_MANIFEST.read_text(encoding="utf-8"))
    requests = manifest["requests"]
    print(f"Loaded manifest with {len(requests)} requests.")

    DEFAULT_OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    for idx, item in enumerate(requests, start=1):
        req_id = item["request_id"]
        study_id = str(item["study_id"])
        image_path = resolve_image_path(item)
        out_file = DEFAULT_OUT_DIR / f"{req_id}.json"

        # Predict raw probabilities
        raw_probabilities = predict(image_path, classifier, device)

        # Select labels
        selected_labels = tuple(
            l for l in CHEXPERT_LABELS
            if raw_probabilities[l] >= thresholds[l]
        )

        # Retrieve evidence
        retrieval = retriever.search_label_evidence(
            str(image_path),
            raw_probabilities=raw_probabilities,
            labels=selected_labels,
            top_k=3,
            common_top_k=5,
            candidate_k=200,
            bank_candidate_k=50,
            statistics_k=20,
            support_similarity_threshold=0.10,
            quality_margin_threshold=0.02,
            margin_10_reversal_ratio=0.5,
            local_reranker=local_reranker,
            local_rerank_k=10,
            lambda_global=0.9,
            exclude_study_ids=[study_id],
        )

        evidence = {
            "schema_version": 2,
            "image_path": str(image_path.resolve()),
            "classifier": {
                "raw_probabilities": {
                    label: compact_score(raw_probabilities[label])
                    for label in CHEXPERT_LABELS
                },
                "selected_labels": list(selected_labels),
                **threshold_snapshot(thresholds, th_payload["version"], DEFAULT_THRESHOLDS),
            },
            "retriever": {
                "model": retriever.model.model_name,
                "local_rerank_enabled": True,
                "excluded_study_ids": [study_id],
                "common_evidence": [
                    compact_candidate(cand)
                    for cand in retrieval["common_evidence"]
                ],
                "label_evidence": [
                    compact_label_evidence(ev)
                    for ev in retrieval["label_evidence"]
                ],
            },
        }

        out_file.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        if idx % 25 == 0 or idx == len(requests):
            elapsed = time.time() - t0
            print(f"[{idx}/{len(requests)}] Processed {req_id} ({elapsed:.1f}s elapsed, ~{elapsed/idx:.2f}s/case)")

    total_time = time.time() - t0
    print(f"\nFinished all {len(requests)} cases in {total_time:.1f}s (~{total_time/len(requests):.2f}s/case).")
    print(f"Evidence files written to: {DEFAULT_OUT_DIR}")


if __name__ == "__main__":
    main()
