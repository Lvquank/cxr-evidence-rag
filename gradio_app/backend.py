"""Local Gradio callbacks; neural inference stays in the research subprocess."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import subprocess
import threading
import time
import uuid
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path, PureWindowsPath

from PIL import Image, UnidentifiedImageError

if __package__:
    from .classifiers import CLASSIFIERS, DEFAULT_CLASSIFIER, classifier_checkpoint
    from .presentation import present_result
else:
    from classifiers import CLASSIFIERS, DEFAULT_CLASSIFIER, classifier_checkpoint
    from presentation import present_result

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "outputs/gradio_cases"
RESEARCH_PYTHON = Path(os.environ.get("CXR_RESEARCH_PYTHON", str(ROOT / ".venv-bertscore/Scripts/python.exe")))
BENCHMARK = ROOT / "outputs/high_effort_4branches_20261031"
METHODS = {
    "structured_rule": "Structured rule",
    "structured_learned_gate": "Structured learned gate",
    "baseline_a": "Baseline A",
    "baseline_b": "Baseline B",
}
REASONING_EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh", "max")
_RUN_LOCK = threading.Lock()


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _write(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _manifest():
    return _read(ROOT / "data/evaluations/leave_one_out_206_manifest.json")["requests"]


def sample_choices():
    return [(f"Study {r['study_id']} · {r['case_label']}", str(r["study_id"])) for r in _manifest()]


def sample_image(study_id):
    item = next((r for r in _manifest() if str(r["study_id"]) == str(study_id)), None)
    if not item:
        raise ValueError("Không tìm thấy study trong manifest 206 ca.")
    path = ROOT / "data/reprocessed/images" / (item["image_id"] + ".png")
    if not path.is_file():
        raise FileNotFoundError("Không tìm thấy ảnh của study trên máy.")
    return str(path)


@lru_cache(maxsize=1)
def _references():
    path = ROOT / "data/embeddings_chexpert_projection/metadata.jsonl"
    return {str(row["id"]): row for row in (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines())}


def reference_image(reference_id):
    record = _references().get(str(reference_id))
    if record is None:
        return None
    folder = (ROOT / "data/reprocessed").resolve()
    path = (folder / record["path"]).resolve()
    return str(path) if path.is_relative_to(folder) and path.is_file() else None


@lru_cache(maxsize=1)
def _reference_reports():
    with (ROOT / "src/data/indiana_reports.csv").open(encoding="utf-8-sig", newline="") as handle:
        return {str(row["uid"]): row for row in csv.DictReader(handle)}


def _evidence_display(state):
    for result in state.get("results", []):
        for label in result.get("evidence", []):
            for kind in ("support", "non_support"):
                for candidate in label[kind]:
                    candidate["source_image_path"] = candidate.get("source_image_path", candidate.get("image_path"))
                    candidate["image_path"] = reference_image(candidate.get("reference_id", candidate.get("id")))
                    report = _reference_reports().get(str(candidate.get("study_id")))
                    if report is not None:
                        candidate["findings"] = report.get("findings", "") or ""
                        candidate["impression"] = report.get("impression", "") or ""
                        candidate["reference_text_source"] = str(ROOT / "src/data/indiana_reports.csv")
    return state


def _environment():
    return {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1",
            "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}


def probe_backend():
    try:
        process = subprocess.run([str(RESEARCH_PYTHON), str(Path(__file__).with_name("worker.py")),
            "--research-root", str(ROOT), "--probe"], cwd=ROOT, capture_output=True,
            text=True, encoding="utf-8", errors="replace", timeout=120, env=_environment(),
            stdin=subprocess.DEVNULL)
        lines = [line[len("CXR_PROBE:"):] for line in process.stdout.splitlines() if line.startswith("CXR_PROBE:")]
        if process.returncode or not lines:
            raise RuntimeError("Không khởi tạo được Python nghiên cứu. Kiểm tra .venv-bertscore và các thư viện CUDA.")
        return json.loads(lines[-1])
    except (OSError, ValueError, subprocess.SubprocessError, RuntimeError) as error:
        return {"status": "error", "ready": False, "message": str(error),
                "resnet50_available": classifier_checkpoint("resnet50", ROOT).is_file()}


def create_case(files, case_name="", model="gpt-5.6-luna", methods=None,
                reasoning_effort="high", study_id="", classifier_model=DEFAULT_CLASSIFIER):
    files = files or []
    if not 1 <= len(files) <= 10:
        raise ValueError("Chọn từ 1 đến 10 ảnh PNG/JPG của cùng một ca.")
    if not re.fullmatch(r"gpt-[A-Za-z0-9._-]{1,75}", str(model)):
        raise ValueError("Mã model GPT không hợp lệ.")
    if reasoning_effort not in REASONING_EFFORTS:
        raise ValueError("Mức reasoning không hợp lệ.")
    if classifier_model not in CLASSIFIERS or not classifier_checkpoint(classifier_model, ROOT).is_file():
        raise ValueError("Không tìm thấy checkpoint của classifier đã chọn.")
    methods = [methods] if isinstance(methods, str) else methods
    selected = list(dict.fromkeys(next((k for k, label in METHODS.items() if m in (k, label)), m)
                                 for m in (methods if methods is not None else ["structured_rule"])))
    if not selected or any(method not in METHODS for method in selected):
        raise ValueError("Chọn ít nhất một phương pháp hợp lệ.")
    study_id = str(study_id or "").strip()
    if study_id and not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", study_id):
        raise ValueError("Study ID chỉ gồm chữ, số, dấu gạch ngang hoặc gạch dưới.")
    incoming, total = [], 0
    for file in files:
        path = Path(file)
        size = path.stat().st_size
        total += size
        if size > 10 * 1024 * 1024 or total > 50 * 1024 * 1024:
            raise ValueError("Tối đa 10 MB mỗi ảnh và 50 MB cho một ca.")
        data = path.read_bytes()
        try:
            with Image.open(io.BytesIO(data)) as image:
                if image.format not in {"PNG", "JPEG"} or min(image.size) < 32 or image.width * image.height > 30_000_000:
                    raise ValueError()
                width, height = image.size
                suffix = ".png" if image.format == "PNG" else ".jpg"
                image.verify()
        except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as error:
            raise ValueError(f"{path.name}: cần ảnh PNG/JPG hợp lệ, tối thiểu 32px và tối đa 30 megapixel.") from error
        incoming.append((path.name, data, suffix, width, height))
    run_id = uuid.uuid4().hex
    folder = RUNTIME / run_id
    uploads = folder / "uploads"
    uploads.mkdir(parents=True, exist_ok=False)
    images = []
    for index, (name, data, suffix, width, height) in enumerate(incoming):
        path = uploads / f"image_{index}{suffix}"
        path.write_bytes(data)
        images.append({"name": name, "stored_name": path.name, "path": str(path.resolve()),
                       "width": width, "height": height, "bytes": len(data),
                       "sha256": hashlib.sha256(data).hexdigest()})
    state = {"id": run_id, "case_name": str(case_name or "").strip()[:100] or "Ca X-quang " + run_id[:8],
             "created_at": datetime.now(timezone.utc).isoformat(), "status": "queued", "progress": 0,
             "stage": "Đang chuẩn bị", "model": model, "methods": selected, "images": images,
             "classifier_model": classifier_model, "study_id": study_id or None,
             "reasoning_effort": reasoning_effort, "max_output_tokens": 8192, "device": "cuda",
             "cache_enabled": False, "results": [], "errors": [], "external_gate_fold": 0,
             "multi_image_policy": "Independent per-image inference, grouped by case; no multi-view fusion.",
             "research_root": str(ROOT), "historical": False}
    if classifier_model != DEFAULT_CLASSIFIER:
        state["classifier_note"] = "ResNet50 dùng ngưỡng DenseNet và gate đã đóng băng; chưa hiệu chuẩn/kiểm định riêng cho ResNet50."
    _write(folder / "run.json", state)
    return state


def _case_folder(run_id):
    if not re.fullmatch(r"[0-9a-f]{32}", str(run_id)):
        raise ValueError("Mã lượt chạy không hợp lệ.")
    folder = RUNTIME / run_id
    if not (folder / "run.json").is_file():
        raise ValueError("Không tìm thấy lượt chạy trên máy.")
    return folder


def load_case(run_id):
    return _evidence_display(_read(_case_folder(run_id) / "run.json"))


def run_case(run_id):
    if not _RUN_LOCK.acquire(blocking=False):
        raise ValueError("GPU đang xử lý ca khác. Chờ lượt chạy hoàn tất.")
    process = None
    folder = None
    try:
        folder = _case_folder(run_id)
        state = load_case(run_id)
        if state["status"] != "queued":
            raise ValueError("Lượt chạy đã bắt đầu. Tạo ca mới để sinh báo cáo mới.")
        state.update(status="running", stage="Khởi tạo môi trường CUDA")
        _write(folder / "run.json", state)
        with (folder / "worker.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen([str(RESEARCH_PYTHON), "-u", str(Path(__file__).with_name("worker.py")),
                "--research-root", str(ROOT), "--run", str(folder), "--external-fold", str(state["external_gate_fold"])],
                cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, env=_environment())
            started, previous = time.monotonic(), None
            while process.poll() is None:
                state = load_case(run_id)
                signature = json.dumps(state, sort_keys=True)
                if signature != previous:
                    previous = signature
                    yield state
                if time.monotonic() - started > 7200:
                    raise TimeoutError("Lượt chạy quá 2 giờ. Xem worker.log và tạo lượt mới.")
                time.sleep(1)
        state = load_case(run_id)
        if process.returncode or state["status"] not in {"complete", "partial", "failed"}:
            state.update(status="failed", error="Pipeline dừng trước khi hoàn tất. Xem worker.log.", stage="Pipeline thất bại")
            _write(folder / "run.json", state)
        yield state
    except (OSError, subprocess.SubprocessError, TimeoutError) as error:
        if folder:
            state = load_case(run_id)
            state.update(status="failed", error=str(error), stage="Pipeline thất bại")
            _write(folder / "run.json", state)
            yield state
        else:
            raise
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            state = load_case(run_id)
            if state["status"] == "running":
                state.update(status="interrupted", stage="Lượt chạy bị gián đoạn")
                _write(folder / "run.json", state)
        _RUN_LOCK.release()


def list_cases():
    rows = []
    for path in RUNTIME.glob("*/run.json"):
        try:
            state = _read(path)
            rows.append({key: state.get(key) for key in ("id", "case_name", "created_at", "status", "model",
                "classifier_model", "reasoning_effort", "methods", "images", "progress", "stage")})
            rows[-1]["result_count"] = len(state.get("results", []))
        except (OSError, ValueError):
            continue
    return sorted(rows, key=lambda row: row["created_at"], reverse=True)


def historical_case(study_id):
    study_id = str(study_id)
    item = next((r for r in _manifest() if str(r["study_id"]) == study_id), None)
    if item is None:
        raise ValueError("Không tìm thấy study trong manifest 206 ca.")
    with (BENCHMARK / "metrics/report_provenance.csv").open(encoding="utf-8-sig", newline="") as handle:
        provenance = list(csv.DictReader(handle))
    checksums = _read(BENCHMARK / "artifact_checksums.json")
    expected_ids = {str(r["study_id"]) for r in _manifest()}
    if len(expected_ids) != 206 or len(provenance) != 824:
        raise ValueError("Provenance lịch sử không đủ 4 × 206 report.")
    for method in METHODS:
        branch_rows = [r for r in provenance if r["pipeline"] == method]
        if len(branch_rows) != 206 or {r["study_id"] for r in branch_rows} != expected_ids:
            raise ValueError("Cohort lịch sử không đồng nhất giữa bốn nhánh.")
    results = []
    for method in METHODS:
        row = next(r for r in provenance if r["pipeline"] == method and r["study_id"] == study_id)
        branch = BENCHMARK / ("baselines/" + method if method.startswith("baseline") else method)
        if row["request_id"] != item["request_id"]:
            raise ValueError("Request ID của report lịch sử không khớp manifest.")
        path = branch / "llm" / item["request_id"] / PureWindowsPath(row["summary_path"]).name
        if not path.resolve().is_relative_to((branch / "llm").resolve()) or _sha(path) != row["summary_sha256"]:
            raise ValueError("Report lịch sử không khớp checksum/provenance: " + method)
        summary = _read(path)
        if str(summary["source"]["study_id"]) != study_id or summary["api"]["response_id"] != row["response_id"]:
            raise ValueError("Study ID hoặc response ID của report lịch sử không khớp.")
        evidence, selected_evidence_path, selected_evidence_sha = {}, None, None
        for name in ("evidence", "raw_evidence"):
            evidence_path = branch / name / (item["request_id"] + ".json")
            if evidence_path.is_file():
                expected_sha = checksums.get(str(evidence_path.relative_to(BENCHMARK)))
                if not expected_sha or _sha(evidence_path) != expected_sha:
                    raise ValueError("Evidence lịch sử không khớp checksum: " + method)
                evidence = _read(evidence_path)
                selected_evidence_path, selected_evidence_sha = str(evidence_path.resolve()), expected_sha
                break
        result = present_result(method, summary, evidence, 0)
        result.update(historical=True, summary_path=str(path.resolve()), summary_sha256=row["summary_sha256"])
        result.update(evidence_path=selected_evidence_path, evidence_sha256=selected_evidence_sha)
        results.append(result)
    return _evidence_display({"id": "historical-" + study_id, "case_name": "IU-Xray · Study " + study_id,
            "study_id": study_id, "status": "complete", "historical": True, "model": "gpt-5.6-luna",
            "classifier_model": DEFAULT_CLASSIFIER, "created_at": "2026-11-01T01:09:58+07:00",
            "methods": list(METHODS), "results": results, "images": [{"name": item["image_id"] + ".png",
                "path": sample_image(study_id), "stored_name": item["image_id"] + ".png"}],
            "reasoning_effort": "high", "max_output_tokens": 8192, "errors": [], "progress": 100})


def research_tables():
    result = {}
    for name in ("table5", "table6"):
        with (BENCHMARK / "metrics" / (name + ".csv")).open(encoding="utf-8-sig", newline="") as handle:
            result[name] = [{key: value if key == "pipeline" else float(value) for key, value in row.items()}
                            for row in csv.DictReader(handle)]
    text_metrics = ROOT / "outputs/text_similarity_4branches_high_20261113/table_4_13_full_precision.json"
    if text_metrics.is_file():
        result["table4_13"] = _read(text_metrics)
    result.update(source=str(BENCHMARK), completed_at="2026-11-01", pathology_studies=206,
                  clinical_studies=205, excluded_reference="1615",
                  settings={"model": "gpt-5.6-luna", "reasoning_effort": "high", "max_output_tokens": 8192, "device": "CUDA"})
    return result
