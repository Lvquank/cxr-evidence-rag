"""Reject cached/mislabeled reports and retrieval leakage in latency runs."""

import copy
import json
import unittest

from scripts.profile_baseline_latency import ROOT, validate_summary, query_image_path
from src.classifier_thresholds import load_classifier_thresholds
from scripts.run_baseline_leave_one_out_batch import command_for_request
from pathlib import Path


class BaselineLatencyTests(unittest.TestCase):
    def setUp(self):
        self.item = json.loads((ROOT / "data/evaluations/leave_one_out_206_manifest.json").read_text())["requests"][0]
        thresholds, metadata = load_classifier_thresholds()
        probabilities = {label: 0.0 for label in thresholds}
        probabilities["Atelectasis"] = 1.0
        self.report = {
            "reasoning_effort": "high", "max_output_tokens": 8192,
            "model": "gpt-5.6-luna", "baseline": "classifier_vision_unstructured_input",
            "source": {"study_id": str(self.item["study_id"]), "image_name": query_image_path(self.item).name},
            "classifier_probabilities": probabilities, "classifier_thresholds": thresholds,
            "classifier_threshold_version": metadata["version"], "classifier_selected_labels": ["Atelectasis"],
            "prediction": {"problems": ["Atelectasis"], "findings": "Basal opacity.", "impression": "Atelectasis."},
            "api": {"cached": False, "response_id": "test-response"},
        }

    def test_accepts_both_baselines_and_rejects_query_leakage(self):
        validate_summary(self.report, self.item, "A")
        report = copy.deepcopy(self.report)
        report["baseline"] = "classifier_retriever_vision_unstructured_input"
        report["five_nearest_cases"] = [{"reference_study_id": f"reference-{index}", "densenet_feature_map_similarity": 0.5} for index in range(5)]
        validate_summary(report, self.item, "B")
        report["five_nearest_cases"][0]["reference_study_id"] = self.item["study_id"]
        with self.assertRaisesRegex(ValueError, "excluding the query"):
            validate_summary(report, self.item, "B")

    def test_rejects_cache_and_wrong_study(self):
        self.report["api"]["cached"] = True
        with self.assertRaisesRegex(ValueError, "fresh API"):
            validate_summary(self.report, self.item, "A")
        self.report["api"]["cached"] = False
        self.report["source"]["study_id"] = "wrong-study"
        with self.assertRaisesRegex(ValueError, "study ID"):
            validate_summary(self.report, self.item, "A")

    def test_rejects_predictions_outside_classifier_passed_labels(self):
        self.report["prediction"]["problems"] = ["Pneumonia"]
        with self.assertRaises(ValueError):
            validate_summary(self.report, self.item, "A")

    def test_rejects_wrong_reasoning_effort_or_budget(self):
        self.report["reasoning_effort"] = "none"
        with self.assertRaisesRegex(ValueError, "reasoning effort"):
            validate_summary(self.report, self.item, "A")
        self.report["reasoning_effort"] = "high"
        self.report["max_output_tokens"] = 160
        with self.assertRaisesRegex(ValueError, "token budget"):
            validate_summary(self.report, self.item, "A")

    def test_child_command_passes_requested_effort_and_budget(self):
        command = command_for_request(self.item, "classifier_vision", Path("out"), "cuda", "high", 8192, "high")
        self.assertEqual(command[command.index("--reasoning-effort")+1], "high")
        self.assertEqual(command[command.index("--max-output-tokens")+1], "8192")


if __name__ == "__main__":
    unittest.main()
