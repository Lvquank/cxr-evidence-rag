"""Adapter to the frozen CXR-RAG code. Never trains models or edits research artefacts."""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
import traceback
import warnings
from datetime import datetime, timezone
from pathlib import Path

if __package__:
    from .presentation import present_result
    from .classifiers import DEFAULT_CLASSIFIER, classifier_checkpoint, load_classifier, load_retrieval_classifier
else:
    from presentation import present_result
    from classifiers import DEFAULT_CLASSIFIER, classifier_checkpoint, load_classifier, load_retrieval_classifier


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def prepare(root):
    os.chdir(root)
    sys.path.insert(0, str(root / "src"))
    sys.path.insert(0, str(root))
    from src.llm_client import _load_project_env
    _load_project_env()
    if (root / "hf_cache").is_dir():
        os.environ.setdefault("HF_HUB_CACHE", str(root / "hf_cache"))


def endpoint():
    return (os.environ.get("LLM_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
            or os.environ.get("LLM_API_BASE") or "https://api.openai.com/v1")


def probe(root):
    import torch
    missing = [name for name in ["torchvision", "faiss", "transformers", "pandas", "sklearn", "yaml", "docx"] if not importlib.util.find_spec(name)]
    required = ["model/best_E2_relation_aware_weighted_bce.pt", "model/rad_dino_chexpert_projection_best.pt",
                "data/embeddings_chexpert_projection/faiss.index", "config/densenet_adaptive_thresholds.json",
                "src/learned_gate_oof_206/folds.json", *[f"src/learned_gate_oof_206/models/fold_{i}.pkl" for i in range(5)]]
    missing += [name for name in required if not (root / name).is_file()]
    key = bool(os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY") or os.environ.get("GPT_TOKEN"))
    ready = bool(torch.cuda.is_available() and not missing and key)
    return {"status": "ready" if ready else "error", "ready": ready, "cuda": torch.cuda.is_available(),
            "resnet50_available": classifier_checkpoint("resnet50", root).is_file(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "torch": torch.__version__, "api_configured": key, "missing": missing,
            "message": "CUDA và pipeline đã sẵn sàng" if ready else ("Thiếu CUDA. Không chạy bằng CPU." if not torch.cuda.is_available() else ("Thiếu cấu hình API trong .env của nghiên cứu." if not key else "Thiếu thư viện hoặc artefact: " + ", ".join(missing)))}


def identify_images(root, folder, state):
    metadata = root / "data/embeddings_chexpert_projection/metadata.jsonl"
    cache_path = folder.parent / "gallery_hashes.json"
    signature = sha(metadata)
    cache = read(cache_path) if cache_path.exists() else {}
    if cache.get("metadata_sha256") != signature:
        matches = {}
        for line in metadata.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            path = root / "data/reprocessed" / record["path"]
            if path.is_file():
                matches.setdefault(sha(path), []).append(str(record["study_id"]))
        cache = {"metadata_sha256": signature, "hash_to_studies": matches}
        save(cache_path, cache)
    known = set()
    for item in state["images"]:
        digest = sha(folder / "uploads" / item["stored_name"])
        item["sha256"] = digest
        item["matched_study_ids"] = cache["hash_to_studies"].get(digest, [])
        known.update(item["matched_study_ids"])
    provided = state.get("study_id")
    if len(known) > 1 or (known and provided and provided not in known):
        raise ValueError("Các ảnh khớp nhiều study hoặc Study ID nhập không khớp ảnh. Chỉ tải ảnh của cùng một ca.")
    excluded = sorted(known | ({provided} if provided else set()))
    state["excluded_study_ids"] = excluded
    state["identified_study_id"] = next(iter(known), None)
    return excluded


def frozen_gate(root, raw, study_id, external_fold):
    import joblib
    from scripts.profile_learned_gate_latency import apply_gate, feature_rows
    from scripts.run_nested_gate_threshold_oof import NUMERIC_FEATURES, PATHOLOGIES, make_feature_matrix
    gate_root = root / "src/learned_gate_oof_206"
    folds = read(gate_root / "folds.json")["study_to_fold"]
    fold = int(folds[study_id]) if study_id in folds else external_fold
    file = gate_root / "models" / f"fold_{fold}.pkl"
    with warnings.catch_warnings(record=True) as load_warnings:
        warnings.simplefilter("always")
        artifact = joblib.load(file)
    if study_id in folds:
        result = apply_gate(raw, artifact, study_id)
        mode = "study_held_out_oof"
    else:
        expected = list(NUMERIC_FEATURES) + [f"pathology={label}" for label in PATHOLOGIES]
        if artifact["feature_names"] != expected or artifact["threshold"] != 0.5:
            raise ValueError("Frozen gate không khớp feature order/threshold của nghiên cứu.")
        probabilities = artifact["pipeline"].predict_proba(make_feature_matrix(feature_rows(raw)))[:, 1]
        result = copy.deepcopy(raw)
        by_label = {row["label"]: row for row in raw["retriever"]["label_evidence"]}
        selected, decisions = [], []
        for label, probability in zip(PATHOLOGIES, probabilities):
            decision = {"pathology": label, "probability": float(probability), "threshold": 0.5,
                        "action": "EMIT" if probability >= 0.5 else "SUPPRESS", "fold": fold}
            decisions.append(decision)
            if probability < 0.5:
                continue
            row = copy.deepcopy(by_label.get(label, {"label": label,
                "classifier_probability": raw["classifier"]["raw_probabilities"][label],
                "evidence_status": "INSUFFICIENT", "evidence_available": False, "margins": {},
                "quality_checks": {}, "supporting_evidence": [], "matched_non_support_evidence": []}))
            row["learned_gate"] = decision
            selected.append(row)
        selected.sort(key=lambda row: -float(row["classifier_probability"]))
        rule_labels = raw["classifier"]["selected_labels"]
        result["classifier"].update(rule_selected_labels=rule_labels, selected_labels=[row["label"] for row in selected])
        result["retriever"]["label_evidence"] = selected
        result["learned_gate"] = {"fold": fold, "threshold": 0.5, "decisions": decisions,
                                  "rule_selected_labels": rule_labels, "learned_selected_labels": [row["label"] for row in selected]}
        mode = "external_inference_single_frozen_fold"
    result["learned_gate"].update(inference_mode=mode, artifact_sha256=sha(file),
        artifact_path=str(file), retrained=False,
        sklearn_version=__import__("sklearn").__version__,
        load_warnings=list(dict.fromkeys(str(warning.message) for warning in load_warnings)),
        note="Ca mới dùng một checkpoint đóng băng; không phải kết quả OOF hoặc kiểm định ngoài tập dữ liệu." if study_id not in folds else "Đúng checkpoint held-out của study trong bộ 206.")
    return result


def retrieve(root, classifier_name):
    from scripts import test_classifier_to_retriever as research_retrieval
    # Override only the loader in this subprocess, leaving research code untouched.
    research_retrieval.load_model = lambda checkpoint, device: load_retrieval_classifier(
        classifier_name, checkpoint, root, device)
    research_retrieval.main()


def classifier_provenance(root, name):
    checkpoint = classifier_checkpoint(name, root)
    return {"model": name, "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha(checkpoint), "device": "cuda",
        "preprocessing": "Research EVAL_TRANSFORM: RGB, 320x320, ImageNet normalization",
        "thresholds_path": str(root / "config/densenet_adaptive_thresholds.json"),
        "threshold_artifact_sha256": sha(root / "config/densenet_adaptive_thresholds.json"),
        "threshold_policy": "Frozen DenseNet development thresholds; no ResNet-specific calibration supplied"
            if name != DEFAULT_CLASSIFIER else "Frozen DenseNet development thresholds",
        "retrieval_local_features": "Frozen DenseNet121; unchanged research retrieval",
        "gate_policy": "Frozen research gate; no retraining" +
            ("; no ResNet-specific validation" if name != DEFAULT_CLASSIFIER else "")}


def run(root, folder, external_fold):
    import torch
    state = read(folder / "run.json")
    started = time.perf_counter()
    total = len(state["images"]) * len(state["methods"])
    def update(stage):
        state.update(stage=stage, progress=round(100 * (len(state["results"]) + len(state["errors"])) / total))
        save(folder / "run.json", state)
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA không khả dụng; không fallback CPU.")
        update("Đối chiếu ảnh với kho IU-Xray để loại self-retrieval")
        excluded = identify_images(root, folder, state)
        study_id = state.get("identified_study_id") or state.get("study_id") or "external-" + state["id"]
        state["gpu"] = torch.cuda.get_device_name(0)
        classifier_name = state.setdefault("classifier_model", DEFAULT_CLASSIFIER)
        checkpoint = classifier_checkpoint(classifier_name, root)
        classifier_metadata = classifier_provenance(root, classifier_name)
        state["classifier_provenance"] = classifier_metadata
        state["provenance"] = {name: sha(root / name) for name in ["model/best_E2_relation_aware_weighted_bce.pt",
            "model/rad_dino_chexpert_projection_best.pt", "config/densenet_adaptive_thresholds.json",
            "scripts/run_full_pipeline.py", "src/prompts.py", "src/llm_client.py"]}
        state["provenance"].update({str(checkpoint): classifier_metadata["checkpoint_sha256"],
            str(Path(__file__).resolve()): sha(Path(__file__)),
            str(Path(__file__).with_name("classifiers.py")): sha(Path(__file__).with_name("classifiers.py"))})
        from src.llm_client import LLMConfig, create_llm_client
        from src.prompts import MedicalPrompts, validate_radiology_prediction
        from src.classifier_thresholds import load_classifier_thresholds, select_labels
        from scripts.predict_chexpert_6labels import load_model, predict
        thresholds, threshold_artifact = load_classifier_thresholds(root / "config/densenet_adaptive_thresholds.json")
        response_ids = set()
        for image_index, image in enumerate(state["images"]):
            path = folder / "uploads" / image["stored_name"]
            case_folder = folder / f"image_{image_index}"
            case_folder.mkdir(exist_ok=True)
            raw = None
            if any(method.startswith("structured") for method in state["methods"]):
                update(f"Ảnh {image_index + 1}: classifier và truy hồi bằng chứng")
                command = [sys.executable, str(Path(__file__).resolve()), "--research-root", str(root),
                    "--retrieve", "--classifier-model", classifier_name,
                    "--classifier_checkpoint", str(checkpoint),
                    "--image_path", str(path), "--output_json", str(case_folder / "raw_evidence.json"),
                    "--device", "cuda", "--lambda_global", "0.9"]
                for excluded_id in excluded:
                    command += ["--exclude-study-id", excluded_id]
                process = subprocess.run(command, cwd=root, stdin=subprocess.DEVNULL, timeout=900)
                if process.returncode:
                    raise RuntimeError("Classifier/retrieval thất bại. Xem worker.log.")
                raw = read(case_folder / "raw_evidence.json")
                raw["classifier"].update(classifier_metadata)
                save(case_folder / "raw_evidence.json", raw)
            probabilities = raw["classifier"]["raw_probabilities"] if raw else None
            neighbor_cases = []
            if probabilities is None or "baseline_b" in state["methods"]:
                update(f"Ảnh {image_index + 1}: chuẩn bị baseline")
                classifier, _ = load_classifier(classifier_name, checkpoint, torch.device("cuda"))
                if probabilities is None:
                    probabilities = predict(path, classifier, torch.device("cuda"))
                if "baseline_b" in state["methods"]:
                    from scripts.run_baseline_classifier_retriever_vision import make_retriever, summarize_nearest_cases
                    retriever = make_retriever(torch.device("cuda"))
                    reranker = classifier if classifier_name == DEFAULT_CLASSIFIER else load_model(
                        classifier_checkpoint(DEFAULT_CLASSIFIER, root), torch.device("cuda"))[0]
                    neighbor_cases = summarize_nearest_cases(path, retriever, reranker.backbone, torch.device("cuda"), exclude_study_id=excluded[0] if excluded else None)
                    del reranker
                    del retriever
                del classifier
                gc.collect()
                torch.cuda.empty_cache()
            selected = select_labels(probabilities, thresholds)
            for method in state["methods"]:
                method_started = time.perf_counter()
                update(f"Ảnh {image_index + 1}: đang sinh báo cáo {method}")
                output = case_folder / method
                output.mkdir(exist_ok=True)
                client, pipeline, response = None, None, None
                try:
                    evidence = raw or {"classifier": {**classifier_metadata, "raw_probabilities": probabilities, "selected_labels": list(selected), "thresholds": thresholds}}
                    if method.startswith("baseline"):
                        if method == "baseline_a":
                            from scripts.run_baseline_classifier_vision import classifier_prompt
                            prompt = classifier_prompt(probabilities, selected)
                        else:
                            from scripts.run_baseline_classifier_retriever_vision import classifier_retriever_prompt
                            prompt = classifier_retriever_prompt(probabilities, neighbor_cases, selected)
                        client = create_llm_client(model=state["model"], base_url=endpoint(), reasoning_effort=state["reasoning_effort"],
                            max_tokens=8192, cache_enabled=False, timeout=120)
                        response = client.send_request(prompt["user"], system=prompt["system"], image_path=str(path),
                            image_detail="high", response_schema=MedicalPrompts.get_output_schema("radiology", selected))
                        prediction = validate_radiology_prediction(json.loads(response), selected)
                        if prediction is None:
                            raise ValueError("Báo cáo không khớp ontology/schema nghiên cứu.")
                        summary = {"model": state["model"], "reasoning_effort": state["reasoning_effort"], "max_output_tokens": 8192,
                            "baseline": method, "source": {"study_id": study_id, "image_name": path.name, "image_path": str(path)},
                            "prediction": prediction, "response": response, "prompt": prompt,
                            "classifier_probabilities": probabilities, "classifier_thresholds": thresholds,
                            "classifier_selected_labels": list(selected), "five_nearest_cases": neighbor_cases if method == "baseline_b" else [],
                            "api": client.get_last_response_metadata(), "route": {"name": "vision_always", "image_attached": True}}
                        summary_file = output / "summary.json"
                        evidence = {"classifier": evidence["classifier"]}
                    else:
                        from scripts.run_full_pipeline import MedicalImageRAGPipeline
                        if method == "structured_learned_gate":
                            evidence = frozen_gate(root, raw, study_id, external_fold)
                        evidence_file = output / "evidence.json"
                        save(evidence_file, evidence)
                        pipeline = MedicalImageRAGPipeline(str(evidence_file), str(output),
                            llm_config=LLMConfig(model=state["model"], base_url=endpoint(), reasoning_effort=state["reasoning_effort"],
                                max_tokens=8192, cache_enabled=False, timeout=120),
                            image_path=str(path), study_id=study_id, escalation_model=state["model"])
                        outcome = pipeline.run()
                        if outcome["status"] != "success":
                            raise RuntimeError(outcome.get("error", "Structured pipeline thất bại."))
                        summaries = list(output.glob("summary_*.json"))
                        if len(summaries) != 1:
                            raise RuntimeError("Không tìm thấy một báo cáo mới duy nhất.")
                        summary = read(summaries[0])
                        summary_file = summaries[0]
                    summary["classifier_model"] = classifier_name
                    summary["classifier_provenance"] = classifier_metadata
                    save(summary_file, summary)
                    api = summary["api"]
                    if api.get("cached") or not api.get("response_id") or api["response_id"] in response_ids:
                        raise ValueError("API response phải mới, không cache, có ID duy nhất.")
                    response_ids.add(api["response_id"])
                    display = present_result(method, summary, evidence, image_index)
                    display["elapsed_s"] = round(time.perf_counter() - method_started, 3)
                    display["summary_path"] = str(summary_file.resolve())
                    display["summary_sha256"] = sha(summary_file)
                    if (output / "evidence.json").is_file():
                        display["evidence_path"] = str((output / "evidence.json").resolve())
                        display["evidence_sha256"] = sha(output / "evidence.json")
                    display["timing_scope"] = "Branch processing after shared image inference; not benchmark end-to-end latency."
                    state["results"].append(display)
                except Exception as error:
                    traceback.print_exc()
                    failure = {"method": method, "image_index": image_index, "error": str(error),
                        "model": state["model"], "reasoning_effort": state["reasoning_effort"],
                        "max_output_tokens": 8192, "device": "cuda", "cache_enabled": False,
                        "source_image": str(path), "failed_at": datetime.now(timezone.utc).isoformat()}
                    failed_client = client or (getattr(pipeline, "llm", None) if pipeline else None)
                    if failed_client:
                        failure["api"] = failed_client.get_last_response_metadata()
                    if response is not None:
                        failure["response"] = response
                    state["errors"].append(failure)
                    save(output / "failure.json", failure)
                update(f"Đã xử lý {len(state['results']) + len(state['errors'])}/{total} báo cáo")
        state["status"] = "complete" if not state["errors"] else ("partial" if state["results"] else "failed")
        state["elapsed_s"] = round(time.perf_counter() - started, 3)
        state["finished_at"] = datetime.now(timezone.utc).isoformat()
        update("Hoàn tất" if state["status"] == "complete" else "Hoàn tất với lỗi — xem chi tiết từng phương pháp")
    except Exception as error:
        traceback.print_exc()
        state.update(status="failed", error=str(error), elapsed_s=round(time.perf_counter() - started, 3))
        save(folder / "run.json", state)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--research-root", type=Path, required=True)
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--run", type=Path)
    parser.add_argument("--retrieve", action="store_true")
    parser.add_argument("--classifier-model", choices=("densenet121", "resnet50"), default=DEFAULT_CLASSIFIER)
    parser.add_argument("--external-fold", type=int, choices=range(5), default=0)
    args, remaining = parser.parse_known_args()
    prepare(args.research_root)
    if args.retrieve:
        sys.argv = [sys.argv[0], *remaining]
        retrieve(args.research_root, args.classifier_model)
    elif remaining:
        parser.error("unrecognized arguments: " + " ".join(remaining))
    elif args.probe:
        print("CXR_PROBE:" + json.dumps(probe(args.research_root), ensure_ascii=True))
    else:
        run(args.research_root, args.run, args.external_fold)
