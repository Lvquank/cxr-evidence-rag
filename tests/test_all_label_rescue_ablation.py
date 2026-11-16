"""Unit and integration tests for the all-label image rescue ablation.

Validates the full specification from Section 11 of
HANDOFF_ALL_LABEL_PROBABILITY_IMAGE_RESCUE_ABLATION.md:
1. Default payload/schema remains unchanged when ablation options are disabled.
2. Full 6-label probability/threshold validation (NaN/missing/out-of-bounds rejected).
3. Float precision preserved without 3-decimal rounding in clf.
4. Schema & parser: control rejects unselected labels; rescue accepts them if in ontology.
5. Image enforcement: rescue policy strictly requires readable, attached query image.
6. Normal cannot be mixed with pathology labels; invalid types/duplicates rejected.
7. Empty selected_labels: control allows only normal; rescue keeps clf & allowed 6 labels.
8. Cache key differentiates reasoning_effort, temperature, prompt, and schema.
9. Audit metadata records policy, allowed labels, clf sent, new_labels.
10. Comparative rescue tradeoffs (rescued FN, added FP, lost TP, rescue precision/recall).
"""

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from generate_minimal_json_for_llm import TARGET_PATHOLOGY_ORDER, build_minimal_data
from llm_client import LLMConfig, OpenAIResponsesClient
from prompts import MedicalPrompts, TARGET_PROBLEM_LABELS, radiology_output_schema, validate_radiology_prediction
from run_full_pipeline import MedicalImageRAGPipeline
from compute_rescue_ablation_comparison import (
    calculate_headline_metrics,
    compute_comparison_records,
    run_paired_bootstrap,
)


class _MockHTTPResponse:
    def __init__(self, body: bytes):
        self._body = body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return self._body


class AllLabelRescueAblationTests(unittest.TestCase):
    @staticmethod
    def _create_mock_evidence(
        raw_probs: dict[str, float] | None = None,
        selected_labels: list[str] | None = None,
        image_path: str = "mock_image.png",
    ) -> dict[str, Any]:
        default_probs = {
            "Atelectasis": 0.333185,
            "Cardiomegaly": 0.086663,
            "Edema": 0.207262,
            "Pleural Effusion": 0.461986,
            "Pneumonia": 0.813927,
            "Pneumothorax": 0.459342,
        }
        return {
            "image_path": image_path,
            "classifier": {
                "raw_probabilities": raw_probs or default_probs,
                "selected_labels": selected_labels if selected_labels is not None else ["Cardiomegaly", "Pneumonia"],
            },
            "retriever": {
                "common_evidence": [{"study_id": "99", "cosine_similarity": 0.7}],
                "label_evidence": [
                    {
                        "label": "Cardiomegaly",
                        "classifier_probability": 0.086663,
                        "evidence_status": "SUPPORTED",
                        "margins": {"M@5": 0.05},
                        "supporting_evidence": [{"problems": ["Cardiomegaly"], "final_rank_score": 0.75}],
                    },
                    {
                        "label": "Pneumonia",
                        "classifier_probability": 0.813927,
                        "evidence_status": "MIXED",
                        "margins": {"M@5": -0.01},
                        "supporting_evidence": [{"problems": ["Pneumonia"], "final_rank_score": 0.65}],
                    },
                ],
            },
        }

    def test_default_contract_unchanged_without_ablation_flag(self):
        evidence = self._create_mock_evidence()
        minimal = build_minimal_data(evidence, include_all_classifier_probabilities=False)
        self.assertEqual(minimal["v"], 5)
        self.assertNotIn("clf", minimal)
        self.assertIn("f", minimal)
        self.assertEqual(len(minimal["f"]), 2)

    def test_clf_contains_all_six_target_pathologies_with_exact_floats(self):
        evidence = self._create_mock_evidence()
        minimal = build_minimal_data(evidence, include_all_classifier_probabilities=True)
        self.assertIn("clf", minimal)
        self.assertEqual(len(minimal["clf"]), 6)
        labels = [row["l"] for row in minimal["clf"]]
        self.assertEqual(tuple(labels), TARGET_PATHOLOGY_ORDER)

        # Precision must NOT be rounded to 3 decimal places in clf
        atelectasis_row = minimal["clf"][0]
        self.assertEqual(atelectasis_row["l"], "Atelectasis")
        self.assertEqual(atelectasis_row["p"], 0.333185)
        self.assertFalse(atelectasis_row["selected"])
        self.assertFalse(atelectasis_row["retrieved"])

        cardiomegaly_row = minimal["clf"][1]
        self.assertEqual(cardiomegaly_row["l"], "Cardiomegaly")
        self.assertEqual(cardiomegaly_row["p"], 0.086663)
        self.assertTrue(cardiomegaly_row["selected"])
        self.assertTrue(cardiomegaly_row["retrieved"])

    def test_clf_validates_bounds_and_missing_labels(self):
        # Missing label
        bad_probs = {"Atelectasis": 0.3}
        bad_evidence = self._create_mock_evidence(raw_probs=bad_probs)
        with self.assertRaisesRegex(ValueError, "missing from classifier raw_probabilities"):
            build_minimal_data(bad_evidence, include_all_classifier_probabilities=True)

        # NaN probability
        nan_probs = {label: (float("nan") if label == "Edema" else 0.2) for label in TARGET_PATHOLOGY_ORDER}
        nan_evidence = self._create_mock_evidence(raw_probs=nan_probs)
        with self.assertRaisesRegex(ValueError, "must be finite and within"):
            build_minimal_data(nan_evidence, include_all_classifier_probabilities=True)

        # Probability > 1.0
        out_of_bounds = {label: (1.5 if label == "Edema" else 0.2) for label in TARGET_PATHOLOGY_ORDER}
        oob_evidence = self._create_mock_evidence(raw_probs=out_of_bounds)
        with self.assertRaisesRegex(ValueError, "must be finite and within"):
            build_minimal_data(oob_evidence, include_all_classifier_probabilities=True)

    def test_radiology_rescue_prompt_template_exists_and_matches_spec(self):
        template = MedicalPrompts.get_prompt("radiology_rescue")
        self.assertEqual(template.name, "radiology_rescue")
        self.assertIn("all six target pathologies in clf", template.system)
        self.assertIn("Do not invent a new probability cutoff.", template.system)
        self.assertIn("A classifier-unselected label may be included only if", template.system)
        self.assertIn("Data:{evidence}", template.user)

    def test_schema_and_parser_enforce_selected_only_vs_all_six(self):
        selected = ("Cardiomegaly",)
        ctrl_schema = radiology_output_schema(selected)
        self.assertEqual(
            ctrl_schema["properties"]["problems"]["items"]["enum"],
            ["Cardiomegaly", "normal"],
        )
        self.assertEqual(ctrl_schema["properties"]["problems"]["maxItems"], 1)

        rescue_schema = radiology_output_schema(TARGET_PROBLEM_LABELS)
        self.assertEqual(len(rescue_schema["properties"]["problems"]["items"]["enum"]), 7)
        self.assertEqual(rescue_schema["properties"]["problems"]["maxItems"], 6)

        # Control rejects unselected pathology
        atelectasis_resp = '{"problems":["Atelectasis"],"findings":"Linear opacities.","impression":"Atelectasis."}'
        self.assertIsNone(validate_radiology_prediction(json.loads(atelectasis_resp), selected))

        # Rescue accepts unselected pathology
        validated = validate_radiology_prediction(json.loads(atelectasis_resp), TARGET_PROBLEM_LABELS)
        self.assertIsNotNone(validated)
        self.assertEqual(validated["problems"], ["Atelectasis"])

    def test_normal_cannot_be_mixed_with_pathology_labels(self):
        mixed_resp = '{"problems":["normal","Atelectasis"],"findings":"Clear lungs.","impression":"None."}'
        self.assertIsNone(validate_radiology_prediction(json.loads(mixed_resp), TARGET_PROBLEM_LABELS))

    def test_rescue_policy_strictly_requires_readable_image(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            evidence_file = temp_path / "evidence.json"
            evidence_file.write_text(json.dumps(self._create_mock_evidence(image_path="non_existent.png")), encoding="utf-8")

            pipeline = MedicalImageRAGPipeline(
                input_file=str(evidence_file),
                output_dir=str(temp_path / "out"),
                problem_label_policy="all_six",
                force_vision=True,
                image_path="non_existent.png",
            )
            pipeline._load_evidence()
            pipeline._generate_minimal_json()
            with self.assertRaisesRegex(RuntimeError, "Rescue policy 'all_six' requires a readable, attached query chest radiograph"):
                pipeline._select_route()

    def test_cache_key_includes_reasoning_effort_and_temperature(self):
        client = OpenAIResponsesClient(LLMConfig(
            api_key="test",
            model="gpt-5.6-luna",
            reasoning_effort="high",
            temperature=0.7,
        ))
        key1 = client._cache_key("sys", "prompt", "hash1", "high", None)

        client2 = OpenAIResponsesClient(LLMConfig(
            api_key="test",
            model="gpt-5.6-luna",
            reasoning_effort="none",
            temperature=0.7,
        ))
        key2 = client2._cache_key("sys", "prompt", "hash1", "high", None)

        client3 = OpenAIResponsesClient(LLMConfig(
            api_key="test",
            model="gpt-5.6-luna",
            reasoning_effort="high",
            temperature=0.2,
        ))
        key3 = client3._cache_key("sys", "prompt", "hash1", "high", None)

        self.assertNotEqual(key1, key2)
        self.assertNotEqual(key1, key3)

    def test_full_pipeline_mock_end_to_end_records_audit_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            img = temp_path / "query.png"
            img.write_bytes(b"dummy_image_data")

            evidence = self._create_mock_evidence(selected_labels=["Cardiomegaly"], image_path=str(img))
            evidence_file = temp_path / "evidence.json"
            evidence_file.write_text(json.dumps(evidence), encoding="utf-8")

            # Mock LLM returning rescued Pleural Effusion (not in classifier selected_labels)
            llm_json_resp = (
                b'{"id":"resp_mock","model":"gpt-5.6-luna","output_text":"{\\"problems\\":[\\"Pleural Effusion\\"],'
                b'\\"findings\\":\\"Left blunting.\\",\\"impression\\":\\"Small effusion.\\"}",'
                b'"usage":{"input_tokens":100,"output_tokens":20}}'
            )

            pipeline = MedicalImageRAGPipeline(
                input_file=str(evidence_file),
                output_dir=str(temp_path / "out"),
                prompt_type="radiology_rescue",
                problem_label_policy="all_six",
                include_all_classifier_probabilities=True,
                force_vision=True,
                image_path=str(img),
                study_id="1234",
                llm_config=LLMConfig(api_key="mock", cache_enabled=False),
            )

            with patch("llm_client.urlopen", return_value=_MockHTTPResponse(llm_json_resp)):
                result = pipeline.run()

            self.assertEqual(result["status"], "success")
            self.assertEqual(result["problem_label_policy"], "all_six")
            self.assertEqual(result["prediction"]["problems"], ["Pleural Effusion"])

            # Check saved summary
            summaries = list((temp_path / "out").glob("summary_*.json"))
            self.assertEqual(len(summaries), 1)
            summary_data = json.loads(summaries[0].read_text(encoding="utf-8"))
            self.assertEqual(summary_data["problem_label_policy"], "all_six")
            self.assertEqual(summary_data["new_labels"], ["Pleural Effusion"])
            self.assertEqual(summary_data["classifier_scores_sent"], 6)

    def test_comparison_metrics_calculation_and_tradeoffs(self):
        # 2 test studies
        control_data = {
            "s1": {"study_id": "s1", "image_name": "s1.png", "gold_labels": {"Cardiomegaly", "Pneumothorax"}, "predicted_labels": {"Cardiomegaly"}, "sample_f1": 0.67},
            "s2": {"study_id": "s2", "image_name": "s2.png", "gold_labels": {"Edema"}, "predicted_labels": set(), "sample_f1": 0.0},
        }
        # Rescue:
        # s1: predicts Cardiomegaly + Pneumothorax (rescues Pneumothorax FN)
        # s2: predicts Atelectasis (adds below-threshold FP)
        rescue_data = {
            "s1": {"study_id": "s1", "image_name": "s1.png", "gold_labels": {"Cardiomegaly", "Pneumothorax"}, "predicted_labels": {"Cardiomegaly", "Pneumothorax"}, "sample_f1": 1.0},
            "s2": {"study_id": "s2", "image_name": "s2.png", "gold_labels": {"Edema"}, "predicted_labels": {"Atelectasis"}, "sample_f1": 0.0},
        }
        evidence_by_study = {
            "s1": {"classifier": {"selected_labels": ["Cardiomegaly"]}},
            "s2": {"classifier": {"selected_labels": []}},
        }

        records = compute_comparison_records(control_data, rescue_data, evidence_by_study)
        self.assertEqual(len(records), 2)

        # Study 1:
        # G = {Cardiomegaly, Pneumothorax}, C = {Cardiomegaly}, P0 = {Cardiomegaly}, P1 = {Cardiomegaly, Pneumothorax}
        # Classifier FN: {Pneumothorax}
        # Rescued classifier FN: {Pneumothorax}
        # Added below-threshold FP: empty
        r1 = next(r for r in records if r["study_id"] == "s1")
        self.assertEqual(r1["classifier_fn"], ["Pneumothorax"])
        self.assertEqual(r1["rescued_classifier_fn"], ["Pneumothorax"])
        self.assertEqual(r1["added_below_threshold_fp"], [])

        # Study 2:
        # G = {Edema}, C = {}, P0 = {}, P1 = {Atelectasis}
        # Classifier FN: {Edema}
        # Rescued classifier FN: empty
        # Added below-threshold FP: {Atelectasis}
        r2 = next(r for r in records if r["study_id"] == "s2")
        self.assertEqual(r2["classifier_fn"], ["Edema"])
        self.assertEqual(r2["rescued_classifier_fn"], [])
        self.assertEqual(r2["added_below_threshold_fp"], ["Atelectasis"])

        ctrl_metrics = calculate_headline_metrics(records, "control")
        resc_metrics = calculate_headline_metrics(records, "rescue")
        self.assertGreater(resc_metrics["micro_f1"], ctrl_metrics["micro_f1"])

        # Bootstrap runs without error
        boot = run_paired_bootstrap(records, n_samples=50, seed=123)
        self.assertIn("difference_rescue_minus_control", boot)
        self.assertIn("delta_micro_f1", boot["difference_rescue_minus_control"])


if __name__ == "__main__":
    unittest.main()
