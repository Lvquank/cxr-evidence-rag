"""Prepare, run, and evaluate the frozen module-improvement ablation."""

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.classifier_thresholds import load_classifier_thresholds, threshold_snapshot
from src.classifier_refinement import refinement_scores
from src.prompts import MedicalPrompts, TARGET_PROBLEM_LABELS, build_prompt_with_evidence, validate_radiology_prediction
from scripts.generate_minimal_json_for_llm import build_minimal_data
from scripts.run_terra_agreement_vision_batch import query_image_path

LABELS = list(TARGET_PROBLEM_LABELS)
DEFAULT_ROOT = ROOT / "outputs/module_improvements_20261018"
BASELINE = ROOT / "outputs/luna_high_evidence_v5_fresh_20260930"
CANDIDATE = ROOT / "outputs/classifier_fn_fp_benchmark_20261015/targeted_hybrid_live/candidate_classifier.joblib"
ARMS = {"control": "radiology", "hybrid_consistent": "radiology", "query_first": "radiology_query_first"}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def freeze(root):
    manifest_path = ROOT / "data/evaluations/leave_one_out_206_manifest.json"
    payload = {"manifest_sha256": digest(manifest_path), "candidate_sha256": digest(CANDIDATE),
               "model": "gpt-5.6-luna", "reasoning_effort": "high", "image_policy": "all_vision_high",
               "selected_only": True, "minimum_support": 0.5, "lambda_global": 0.9,
               "arms": {name: {"prompt": prompt, "system_sha256": hashlib.sha256(
                   MedicalPrompts.get_prompt(prompt).system.encode()).hexdigest()} for name, prompt in ARMS.items()},
               "status": "exploratory historical cohort; no independent confirmation claimed"}
    path = root / "protocol.json"
    if path.exists():
        if read(path) != payload:
            raise ValueError("Frozen experiment inputs changed. Use a fresh output root.")
    else:
        write(path, payload)
    requests = read(manifest_path)["requests"]
    if len(requests) != 206 or len({str(r["study_id"]) for r in requests}) != 206:
        raise ValueError("Expected 206 unique frozen studies")
    return requests


def prepare(root, requests):
    import joblib
    import torch
    from PIL import Image
    from scripts.predict_chexpert_6labels import load_model, predict
    from scripts.test_classifier_to_retriever import compact_candidate, compact_label_evidence, compact_score
    from src.embeddings.encoder import load_image_processor
    from src.retriever.faiss_retriever import FAISSImageRetriever
    from src.retriever.densenet_local_reranker import DenseNetLocalFeatureReranker

    if (root / "preparation_complete.json").exists():
        raise ValueError("Preparation already completed; preserve its evidence.")
    gallery = root / "consistent_gallery"
    if not (gallery / "faiss.index").exists():
        raise FileNotFoundError("Build the consistent gallery before preparing evidence.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dense_path = ROOT / "model/best_E2_relation_aware_weighted_bce.pt"
    bundle = joblib.load(CANDIDATE)
    feature_manifest = bundle["image_feature_manifest"]
    if digest(dense_path) != bundle["densenet_checkpoint_sha256"]:
        raise ValueError("DenseNet checkpoint does not match the frozen candidate")
    classifier, _ = load_model(dense_path, device)
    reranker = DenseNetLocalFeatureReranker(classifier.backbone, device, batch_size=8)
    retriever = FAISSImageRetriever(str(gallery / "faiss.index"), str(gallery / "metadata.jsonl"),
        processed_dir=str(ROOT / "data/reprocessed"), device=str(device),
        projection_checkpoint=feature_manifest["checkpoint_path"], minimum_label_support=0.5)
    if digest(feature_manifest["checkpoint_path"]) != feature_manifest["checkpoint_sha256"]:
        raise ValueError("Candidate encoder checkpoint hash mismatch")
    processor = load_image_processor(retriever.model.model_name)
    if processor.to_dict() != feature_manifest["processor"]:
        raise ValueError("Candidate processor differs from frozen training features")
    consistency = []
    for gallery_index in (0, 100, 1000):
        row = retriever.metadata[gallery_index]
        image = ROOT / "data/reprocessed" / row["path"]
        value = retriever.embed_image(str(image))
        saved = retriever.index.reconstruct(gallery_index)
        cosine = float(value @ saved / np.linalg.norm(saved))
        consistency.append({"gallery_row": gallery_index, "same_image_cosine": cosine})
    if min(row["same_image_cosine"] for row in consistency) < .9999:
        raise ValueError("New gallery does not reproduce query preprocessing")
    write(root / "self_consistency.json", consistency)
    thresholds, threshold_artifact = load_classifier_thresholds()
    baseline_snapshot = threshold_snapshot(thresholds, threshold_artifact["version"])
    results = []
    start = time.monotonic()
    for index, item in enumerate(requests, 1):
        image_path = query_image_path(item)
        raw = predict(image_path, classifier, device)
        with Image.open(image_path) as image:
            pixels = processor(images=image.convert("RGB"), return_tensors="pt")["pixel_values"].to(device)
        with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            features = retriever.model(pixels)
        feature_array = torch.nn.functional.normalize(features.float(), dim=1).cpu().numpy()
        selected, score_array, candidate_thresholds, sources = refinement_scores(
            bundle, np.array([[raw[name] for name in LABELS]]), {"chexpert_rad": feature_array})
        selected_labels = [name for name, keep in zip(LABELS, selected[0]) if keep]
        scores = dict(zip(LABELS, score_array[0].tolist()))
        retrieval = retriever.search_label_evidence(str(image_path), raw_probabilities=scores,
            labels=tuple(selected_labels), local_reranker=reranker, lambda_global=0.9,
            exclude_study_ids=[str(item["study_id"])])
        candidate_evidence = {"schema_version": 2, "image_path": str(image_path),
            "classifier": {"raw_probabilities": {name: compact_score(raw[name]) for name in LABELS},
                "selection_probabilities": scores, "score_sources": sources,
                "selected_labels": selected_labels,
                **threshold_snapshot(candidate_thresholds, "targeted-hybrid-live-v1"),
                "candidate_sha256": digest(CANDIDATE)},
            "retriever": {"model": retriever.model.model_name, "excluded_study_ids": [str(item["study_id"])],
                "local_rerank_enabled": True, "minimum_label_support": 0.5,
                "gallery": str(gallery),
                "common_evidence": [compact_candidate(c) for c in retrieval["common_evidence"]],
                "label_evidence": [compact_label_evidence(e) for e in retrieval["label_evidence"]]}}
        baseline = read(BASELINE / "evidence" / f"{item['request_id']}.json")
        baseline["image_path"] = str(image_path)
        baseline["classifier"].update(baseline_snapshot)
        for arm, evidence in (("control", baseline), ("hybrid_consistent", candidate_evidence)):
            write(root / arm / "evidence" / f"{item['request_id']}.json", evidence)
            write(root / arm / "compact" / f"{item['request_id']}.json", build_minimal_data(evidence))
        results.append({"request_id": item["request_id"], "study_id": str(item["study_id"]),
                        "control_evidence_sha256": digest(root / "control/evidence" / f"{item['request_id']}.json"),
                        "candidate_evidence_sha256": digest(root / "hybrid_consistent/evidence" / f"{item['request_id']}.json"),
                        "baseline_labels": baseline["classifier"]["selected_labels"],
                        "candidate_labels": selected_labels,
                        "max_dense_score_change": max(abs(raw[n] - baseline["classifier"]["raw_probabilities"][n]) for n in LABELS)})
        reranker.clear_cache()
        if index % 10 == 0 or index == len(requests):
            print(f"evidence {index}/{len(requests)}, elapsed {time.monotonic()-start:.1f}s", flush=True)
    write(root / "preparation_complete.json", {"studies": len(results), "elapsed_seconds": time.monotonic()-start,
          "query_first_evidence": "identical to hybrid_consistent", "results": results})


def run_llm(root, requests, limit, dry_run, arm=None):
    from src.llm_client import LLMConfig, OpenAIResponsesClient
    if not (root / "preparation_complete.json").exists():
        raise ValueError("Prepare and validate all evidence before sending requests")
    frozen_inputs = {row["request_id"]: row for row in read(root / "preparation_complete.json")["results"]}
    selected_arms = ARMS if arm is None else {arm: ARMS[arm]}
    completed = []
    for item in requests[:limit]:
        for arm, prompt_name in selected_arms.items():
            evidence_arm = "hybrid_consistent" if arm == "query_first" else arm
            evidence_path = root / evidence_arm / "evidence" / f"{item['request_id']}.json"
            hash_key = "control_evidence_sha256" if arm == "control" else "candidate_evidence_sha256"
            if digest(evidence_path) != frozen_inputs[item["request_id"]][hash_key]:
                raise ValueError("Evidence changed after preparation was frozen")
            evidence = read(evidence_path)
            compact = build_minimal_data(evidence)
            allowed = evidence["classifier"]["selected_labels"]
            schema = MedicalPrompts.get_output_schema(prompt_name, allowed)
            prompt = build_prompt_with_evidence(MedicalPrompts.get_prompt(prompt_name), compact)
            request_hash = hashlib.sha256(json.dumps({"prompt": prompt, "schema": schema,
                "image_sha256": digest(query_image_path(item)), "model": "gpt-5.6-luna", "effort": "high"},
                sort_keys=True).encode()).hexdigest()
            folder = root / arm / "llm" / item["request_id"]
            destination = folder / "summary_frozen.json"
            if destination.exists():
                saved = read(destination)
                if saved.get("request_sha256") != request_hash or validate_radiology_prediction(saved.get("prediction"), allowed) is None:
                    raise ValueError("Saved summary differs from the frozen request")
                completed.append({"arm": arm, "request_id": item["request_id"], "resumed": True})
                continue
            write(folder / "minimal_frozen.json", compact)
            write(folder / "request_manifest.json", {"request_sha256": request_hash, "prompt": prompt,
                "schema": schema, "image_path": str(query_image_path(item)), "dry_run": dry_run})
            if dry_run:
                continue
            for attempt, cap in enumerate((2048, 4096, 4096), 1):
                client = OpenAIResponsesClient(LLMConfig(model="gpt-5.6-luna", reasoning_effort="high",
                    max_tokens=cap, retry_attempts=1, timeout=120, cache_enabled=False))
                try:
                    response = client.send_request(prompt["user"], system=prompt["system"],
                        image_path=str(query_image_path(item)), image_detail="high", response_schema=schema)
                    prediction = validate_radiology_prediction(json.loads(response), allowed)
                    if prediction is None:
                        raise ValueError("Response violates the selected-only schema")
                    metadata = client.get_last_response_metadata()
                    summary = {"timestamp": datetime.now(timezone.utc).isoformat(), "requested_model": "gpt-5.6-luna",
                        "model": metadata.get("model"), "reasoning_effort": "high", "request_sha256": request_hash,
                        "prompt_type": prompt_name, "evidence_schema": 5, "prediction": prediction,
                        "response": response, "api": metadata, "attempt": attempt, "max_output_tokens": cap,
                        "route": {"name": "vision_forced", "image_attached": True, "image_detail": "high",
                                  "statuses": sorted({row["s"] for row in compact["f"]})},
                        "source": {"study_id": str(item["study_id"]), "image_name": query_image_path(item).name}}
                    write(destination, summary)
                    completed.append({"arm": arm, "request_id": item["request_id"], "attempt": attempt})
                    print(f"LLM {len(completed)}/{limit*len(selected_arms)}: {arm} {item['request_id']}", flush=True)
                    break
                except Exception as error:
                    write(root / arm / "failures" / item["request_id"] / f"attempt_{attempt}.json",
                          {"error": str(error), "cap": cap, "request_sha256": request_hash})
                    if attempt == 3 or "HTTP 4" in str(error) and "429" not in str(error):
                        raise
    progress_name = "dry_run_complete" if dry_run else "llm_progress"
    if len(selected_arms) == 1:
        progress_name += f"_{next(iter(selected_arms))}"
    write(root / f"{progress_name}.json",
          {"requested_studies": limit, "dry_run": dry_run, "completed": completed})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["prepare", "llm"])
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--arm", choices=list(ARMS), help="Run one arm for independent resumable processes")
    args = parser.parse_args()
    if not 1 <= args.limit <= 206:
        parser.error("--limit must be between 1 and 206")
    requests = freeze(args.output_root)
    if args.phase == "prepare":
        prepare(args.output_root, requests)
    else:
        run_llm(args.output_root, requests, args.limit, args.dry_run, args.arm)


if __name__ == "__main__":
    main()
