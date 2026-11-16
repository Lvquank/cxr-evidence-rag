"""Check the local UI boundary without loading neural models or calling APIs."""
import hashlib
import json
import shutil
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from PIL import Image

from gradio_app import backend, worker


class GradioBackendTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.runtime = self.folder / "runtime"
        self.runtime_patch = patch.object(backend, "RUNTIME", self.runtime)
        self.runtime_patch.start()
        self.addCleanup(self.runtime_patch.stop)
        self.image = self.folder / "upload.png"
        Image.new("RGB", (64, 64), color="black").save(self.image)

    def test_fresh_case_preserves_original_bytes_and_defaults_without_api(self):
        original = self.image.read_bytes()
        with patch.object(backend.subprocess, "Popen") as popen, patch.object(backend.subprocess, "run") as run:
            first = backend.create_case([str(self.image)], "Ca kiểm tra")
            second = backend.create_case([str(self.image)], "Ca kiểm tra")
            loaded = backend.load_case(first["id"])
            self.assertEqual(2, len(backend.list_cases()))
            self.assertEqual([0, 0], [row["result_count"] for row in backend.list_cases()])
            popen.assert_not_called()
            run.assert_not_called()
        self.assertNotEqual(first["id"], second["id"])
        self.assertEqual(first, loaded)
        self.assertEqual("high", first["reasoning_effort"])
        self.assertEqual("gpt-5.6-luna", first["model"])
        self.assertEqual(8192, first["max_output_tokens"])
        self.assertEqual("cuda", first["device"])
        self.assertFalse(first["cache_enabled"])
        self.assertFalse(first["historical"])
        self.assertEqual("queued", first["status"])
        self.assertEqual(original, Path(first["images"][0]["path"]).read_bytes())
        self.assertEqual(hashlib.sha256(original).hexdigest(), first["images"][0]["sha256"])

    def test_upload_rejects_invalid_content_tiny_images_large_files_and_count(self):
        invalid = self.folder / "fake.png"
        invalid.write_text("not an image", encoding="utf-8")
        tiny = self.folder / "tiny.png"
        Image.new("RGB", (16, 16)).save(tiny)
        large = self.folder / "large.png"
        with large.open("wb") as handle:
            handle.seek(10 * 1024 * 1024)
            handle.write(b"x")
        for files in ([], [str(self.image)] * 11, [str(invalid)], [str(tiny)], [str(large)]):
            with self.subTest(files=files[:1]):
                with self.assertRaises(ValueError):
                    backend.create_case(files)
        self.assertFalse(self.runtime.exists())

    def test_selected_classifier_and_reasoning_are_preserved(self):
        state = backend.create_case([str(self.image)], model="gpt-6-astra", methods=["Baseline A", "baseline_b"],
                                    reasoning_effort="medium", classifier_model="resnet50")
        self.assertEqual(["baseline_a", "baseline_b"], state["methods"])
        self.assertEqual("medium", state["reasoning_effort"])
        self.assertEqual("gpt-6-astra", state["model"])
        self.assertEqual("resnet50", state["classifier_model"])
        self.assertIn("chưa hiệu chuẩn", state["classifier_note"])
        for values in ({"methods": []}, {"methods": ["unknown"]}, {"model": "not-gpt"},
                       {"reasoning_effort": "default"}, {"study_id": "../../outside"}, {"classifier_model": "unknown"}):
            with self.assertRaises(ValueError):
                backend.create_case([str(self.image)], **values)

    def test_historical_uses_four_exact_verified_provenance_paths(self):
        with patch.object(Path, "glob", side_effect=AssertionError("Do not select reports by latest glob")), \
             patch.object(backend.subprocess, "run") as run:
            state = backend.historical_case("1614")
            run.assert_not_called()
        self.assertTrue(state["historical"])
        self.assertEqual(set(backend.METHODS), {r["method"] for r in state["results"]})
        self.assertEqual(4, len({r["api"]["response_id"] for r in state["results"]}))
        self.assertTrue(Path(state["images"][0]["path"]).is_file())
        for result in state["results"]:
            self.assertTrue(Path(result["summary_path"]).is_file())
            self.assertEqual(64, len(result["summary_sha256"]))
            self.assertTrue({"prediction", "probabilities", "thresholds", "api", "route", "gate"} <= result.keys())
        learned = next(r for r in state["results"] if r["method"] == "structured_learned_gate")
        self.assertIn("structured_learned_gate", learned["evidence_path"])
        self.assertEqual(64, len(learned["evidence_sha256"]))
        self.assertEqual(0.5, learned["gate"]["threshold"])
        self.assertEqual(6, len(learned["gate"]["decisions"]))
        for label in learned["evidence"]:
            for kind in ("support", "non_support"):
                for candidate in label[kind]:
                    self.assertTrue(Path(candidate["image_path"]).is_file())
                    report = backend._reference_reports()[str(candidate["study_id"])]
                    self.assertEqual(report["findings"], candidate["findings"])
                    self.assertEqual(report["impression"], candidate["impression"])
                    self.assertTrue(Path(candidate["reference_text_source"]).is_file())
        with patch.object(backend, "_sha", return_value="changed"):
            with self.assertRaisesRegex(ValueError, "checksum"):
                backend.historical_case("1614")

    def test_historical_reports_follow_the_project_when_it_moves(self):
        original_root = backend.ROOT
        original = backend.historical_case("1614")
        relocated = self.folder / "another_drive" / "image-rag"
        sources = [
            original_root / "data/evaluations/leave_one_out_206_manifest.json",
            original_root / "data/embeddings_chexpert_projection/metadata.jsonl",
            original_root / "src/data/indiana_reports.csv",
            Path(backend.sample_image("1614")),
            backend.BENCHMARK / "artifact_checksums.json",
            backend.BENCHMARK / "metrics/report_provenance.csv",
        ]
        for result in original["results"]:
            sources.append(Path(result["summary_path"]))
            if result["evidence_path"]:
                sources.append(Path(result["evidence_path"]))
        for source in sources:
            destination = relocated / source.relative_to(original_root)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        with patch.object(backend, "ROOT", relocated), \
             patch.object(backend, "BENCHMARK", relocated / "outputs/high_effort_4branches_20261031"), \
             patch.object(backend.subprocess, "run") as run:
            moved = backend.historical_case("1614")
            run.assert_not_called()
        self.assertEqual(4, len(moved["results"]))
        for old, new in zip(original["results"], moved["results"]):
            self.assertTrue(Path(new["summary_path"]).is_relative_to(relocated))
            self.assertEqual(old["summary_sha256"], new["summary_sha256"])
            self.assertEqual(old["api"]["response_id"], new["api"]["response_id"])

    def test_run_case_streams_subprocess_state_and_cannot_reuse_a_run(self):
        case = backend.create_case([str(self.image)])
        captured = {}

        class FakeWorker:
            returncode = 0

            def __init__(self, command, **kwargs):
                captured.update(command=command, options=kwargs)
                self.calls = 0

            def poll(self):
                self.calls += 1
                if self.calls == 1:
                    return None
                state = backend.load_case(case["id"])
                state.update(status="complete", progress=100, stage="Hoàn tất", results=[{"method": "structured_rule", "evidence": []}])
                backend._write(self_folder / "run.json", state)
                return 0

        self_folder = self.runtime / case["id"]
        with patch.object(backend.subprocess, "Popen", FakeWorker), patch.object(backend.time, "sleep"):
            states = list(backend.run_case(case["id"]))
        self.assertEqual(["running", "complete"], [s["status"] for s in states])
        self.assertIn("--run", captured["command"])
        self.assertEqual("1", captured["options"]["env"]["HF_HUB_OFFLINE"])
        self.assertTrue((self_folder / "worker.log").is_file())
        self.assertEqual(1, backend.list_cases()[0]["result_count"])
        with self.assertRaisesRegex(ValueError, "đã bắt đầu"):
            list(backend.run_case(case["id"]))
        self.assertFalse(backend._RUN_LOCK.locked())

    def test_run_case_records_failed_worker_exit_and_releases_gpu_lock(self):
        case = backend.create_case([str(self.image)])
        process = Mock(returncode=3)
        process.poll.return_value = 3
        with patch.object(backend.subprocess, "Popen", return_value=process):
            states = list(backend.run_case(case["id"]))
        self.assertEqual("failed", states[-1]["status"])
        self.assertIn("worker.log", states[-1]["error"])
        self.assertEqual("failed", backend.load_case(case["id"])["status"])
        self.assertFalse(backend._RUN_LOCK.locked())

    def test_research_tables_and_sample_cohort_are_read_only(self):
        with patch.object(backend.subprocess, "run") as run:
            tables = backend.research_tables()
            choices = backend.sample_choices()
            run.assert_not_called()
        self.assertEqual(206, len(choices))
        self.assertEqual(206, len({sid for _, sid in choices}))
        self.assertEqual(4, len(tables["table5"]))
        self.assertEqual(4, len(tables["table6"]))
        self.assertEqual({206.0}, {r["study_count"] for r in tables["table5"]})
        self.assertEqual({205.0}, {r["study_count"] for r in tables["table6"]})
        self.assertEqual("1615", tables["excluded_reference"])
        self.assertEqual(set(backend.METHODS), set(tables["table4_13"]["results"]))

    def test_frozen_gate_chooses_correct_held_out_fold(self):
        root = self.folder / "research"
        gate = root / "src/learned_gate_oof_206"
        (gate / "models").mkdir(parents=True)
        (gate / "folds.json").write_text(json.dumps({"study_to_fold": {"1614": 2}}), encoding="utf-8")
        checkpoint = gate / "models/fold_2.pkl"
        checkpoint.write_bytes(b"frozen-test-artifact")
        artifact = {"held_out_study_ids": ["1614"], "train_study_ids": ["other"]}
        apply = Mock(return_value={"learned_gate": {"fold": 2}})
        joblib = types.SimpleNamespace(load=Mock(return_value=artifact))
        modules = {
            "joblib": joblib,
            "scripts.profile_learned_gate_latency": types.SimpleNamespace(apply_gate=apply, feature_rows=Mock()),
            "scripts.run_nested_gate_threshold_oof": types.SimpleNamespace(NUMERIC_FEATURES=[], PATHOLOGIES=[], make_feature_matrix=Mock()),
            "sklearn": types.SimpleNamespace(__version__="test"),
        }
        raw = {"classifier": {}}
        with patch.dict(sys.modules, modules):
            result = worker.frozen_gate(root, raw, "1614", external_fold=0)
        joblib.load.assert_called_once_with(checkpoint)
        apply.assert_called_once_with(raw, artifact, "1614")
        self.assertEqual("study_held_out_oof", result["learned_gate"]["inference_mode"])
        self.assertFalse(result["learned_gate"]["retrained"])
        self.assertEqual(hashlib.sha256(checkpoint.read_bytes()).hexdigest(), result["learned_gate"]["artifact_sha256"])


if __name__ == "__main__":
    unittest.main()
