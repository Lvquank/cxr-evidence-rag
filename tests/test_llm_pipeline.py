import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from generate_minimal_json_for_llm import build_minimal_data, _polarity, _sentences
from llm_client import LLMConfig, OpenAIResponsesClient
from prompts import MedicalPrompts
from run_full_pipeline import MedicalImageRAGPipeline, select_llm_route
from run_terra_agreement_vision_batch import has_completed_llm_summary, llm_command, run_logged


class _HTTPResponse:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return b'{"id":"resp_1","model":"test-model","output_text":"Concise impression.","usage":{"input_tokens":12,"output_tokens":3}}'


class MinimalEvidenceTests(unittest.TestCase):
    def setUp(self):
        reports = patch("generate_minimal_json_for_llm._reports", return_value={
            "synthetic-reference": {"findings": "Airspace disease.", "impression": ""},
        })
        reports.start()
        self.addCleanup(reports.stop)

    @staticmethod
    def _six_label_evidence():
        labels = [
            "Atelectasis", "Cardiomegaly", "Edema", "Pleural Effusion",
            "Pneumonia", "Pneumothorax",
        ]
        return {
            "retriever": {
                "label_evidence": [
                    {
                        "label": label,
                        "classifier_probability": 0.9 - index / 100,
                        "evidence_status": "SUPPORTED",
                        "supporting_evidence": [{"problems": [label]}],
                    }
                    for index, label in enumerate(labels)
                ]
            }
        }

    def test_keeps_only_high_value_label_summary(self):
        source = {
            "retriever": {
                "label_evidence": [
                    {
                        "label": "Pneumonia",
                        "classifier_probability": 0.9,
                        "evidence_status": "MIXED",
                        "margins": {"M@5": 0.01},
                        "supporting_evidence": [{"problems": ["Opacity"], "label_support": 0.7, "final_rank_score": 0.65}],
                        "matched_non_support_evidence": [{"problems": ["Pulmonary Edema"], "final_rank_score": 0.66}],
                    },
                    {"label": "Edema", "classifier_probability": 0.5, "evidence_status": "SUPPORTED",
                     "margins": {"M@5": 0.05},
                     "supporting_evidence": [{"problems": ["Edema"], "label_support": 1, "final_rank_score": 0.75}]},
                ]
            }
        }
        result = build_minimal_data(source, max_labels=2)
        self.assertEqual(result["v"], 5)
        self.assertEqual([row["l"] for row in result["f"]], ["Edema", "Pneumonia"])
        self.assertEqual(result["f"][0]["e"], "strong_support")
        self.assertEqual(result["f"][1]["e"], "weak_support")
        self.assertAlmostEqual(result["f"][1]["t"], 0.106)
        self.assertEqual(result["f"][1]["a"]["u"], 0.7)
        self.assertEqual(result["f"][1]["n"]["q"], 0.66)

    def test_default_compact_contract_keeps_all_six_label_rows(self):
        result = build_minimal_data(self._six_label_evidence())
        self.assertEqual(len(result["f"]), 6)
        self.assertEqual({row["l"] for row in result["f"]}, {
            "Atelectasis", "Cardiomegaly", "Edema", "Pleural Effusion",
            "Pneumonia", "Pneumothorax",
        })

    def test_explicit_smaller_label_limit_remains_available(self):
        result = build_minimal_data(self._six_label_evidence(), max_labels=3)
        self.assertEqual(len(result["f"]), 3)

    def test_full_pipeline_compact_generation_keeps_all_six_label_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            evidence_file = Path(directory) / "evidence.json"
            evidence_file.write_text(json.dumps(self._six_label_evidence()), encoding="utf-8")
            pipeline = MedicalImageRAGPipeline(
                input_file=str(evidence_file), output_dir=str(Path(directory) / "output")
            )
            pipeline._generate_minimal_json()
            self.assertEqual(len(pipeline.minimal_json["f"]), 6)
            self.assertEqual({row["l"] for row in pipeline.minimal_json["f"]}, {
                "Atelectasis", "Cardiomegaly", "Edema", "Pleural Effusion",
                "Pneumonia", "Pneumothorax",
            })

    def test_calls_responses_api_once_and_exposes_usage(self):
        with tempfile.TemporaryDirectory() as directory:
            client = OpenAIResponsesClient(LLMConfig(
                api_key="test-key",
                model="test-model",
                cache_dir=directory,
                cache_enabled=False,
                retry_attempts=1,
            ))
            with patch("llm_client.urlopen", return_value=_HTTPResponse()) as request:
                self.assertEqual(client.send_request("Data:{}", system="System"), "Concise impression.")
            payload = json.loads(request.call_args.args[0].data.decode("utf-8"))
            self.assertEqual(payload["input"], "Data:{}")
            self.assertEqual(payload["instructions"], "System")
            self.assertFalse(payload["store"])
            self.assertEqual(payload["reasoning"]["effort"], "high")
            self.assertEqual(client.get_last_response_metadata()["usage"]["output_tokens"], 3)

    def test_incomplete_response_does_not_accept_partial_json(self):
        with self.assertRaisesRegex(RuntimeError, "response incomplete"):
            OpenAIResponsesClient._extract_output_text({
                "status": "incomplete", "output_text": '{"problems":[',
                "incomplete_details": {"reason": "max_output_tokens"},
            })

    def test_attaches_image_only_when_requested(self):
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "source.png"
            image.write_bytes(b"not-decoded-by-the-mocked-api")
            client = OpenAIResponsesClient(LLMConfig(
                api_key="test-key",
                model="test-model",
                cache_dir=directory,
                cache_enabled=False,
                retry_attempts=1,
            ))
            with patch("llm_client.urlopen", return_value=_HTTPResponse()) as request:
                client.send_request("Data:{}", system="System", image_path=str(image), image_detail="high")
            payload = json.loads(request.call_args.args[0].data.decode("utf-8"))
            content = payload["input"][0]["content"]
            self.assertEqual(content[0], {"type": "input_text", "text": "Data:{}"})
            self.assertEqual(content[1]["type"], "input_image")
            self.assertEqual(content[1]["detail"], "high")
            self.assertTrue(content[1]["image_url"].startswith("data:image/png;base64,"))

    def test_radiology_uses_strict_structured_output(self):
        schema = MedicalPrompts.get_output_schema("radiology")
        with tempfile.TemporaryDirectory() as directory:
            client = OpenAIResponsesClient(LLMConfig(
                api_key="test-key",
                model="test-model",
                cache_dir=directory,
                cache_enabled=False,
                retry_attempts=1,
            ))
            with patch("llm_client.urlopen", return_value=_HTTPResponse()) as request:
                client.send_request("Data:{}", system="System", response_schema=schema)
            payload = json.loads(request.call_args.args[0].data.decode("utf-8"))
            self.assertEqual(payload["text"]["format"]["type"], "json_schema")
            self.assertTrue(payload["text"]["format"]["strict"])
            self.assertEqual(payload["text"]["format"]["schema"]["required"], ["problems", "findings", "impression"])
            self.assertEqual(
                payload["text"]["format"]["schema"]["properties"]["problems"]["items"]["enum"],
                ["Atelectasis", "Cardiomegaly", "Edema", "Pleural Effusion", "Pneumonia", "Pneumothorax", "normal"],
            )
            self.assertNotIn("uniqueItems", payload["text"]["format"]["schema"]["properties"]["problems"])
        self.assertEqual(
            MedicalImageRAGPipeline._parse_prediction(
                '{"problems":["Edema"],"findings":"Mild bilateral opacity.","impression":"Mild edema."}', schema
            ),
            {"problems": ["Edema"], "findings": "Mild bilateral opacity.", "impression": "Mild edema."},
        )
        self.assertIsNone(
            MedicalImageRAGPipeline._parse_prediction(
                '{"problems":["normal","Edema"],"findings":"Normal lungs.","impression":"No acute finding."}', schema
            )
        )

    def test_radiology_prompt_contains_a_format_example(self):
        prompt = MedicalPrompts.get_prompt("radiology").format(evidence="{}")
        self.assertIn('"problems":["Cardiomegaly"]', prompt["system"])
        self.assertIn('"findings"', prompt["system"])

    def test_cost_aware_model_routes(self):
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "source.png"
            image.write_bytes(b"image")
            uncertain = {"v": 4, "f": [["Pneumonia", 0.82, "M", ["Consolidation"], ["Pulmonary Edema"]]]}
            supported = {"v": 4, "f": [["Edema", 0.98, "S", ["Pulmonary Edema"]]]}

            escalated = select_llm_route(uncertain, str(image), "gpt-5.6-luna")
            self.assertEqual(escalated["name"], "vision_escalation")
            self.assertEqual(escalated["model"], "gpt-5.6-luna")
            self.assertTrue(escalated["image_path"])

            fallback = select_llm_route(uncertain, None, "gpt-5.6-luna")
            self.assertEqual(fallback["name"], "text_fallback")
            self.assertEqual(fallback["model"], "gpt-5.6-luna")

            default = select_llm_route(supported, str(image), "gpt-5.6-luna")
            self.assertEqual(default["name"], "text_default")
            self.assertIsNone(default["image_path"])

            v5 = {"v": 5, "f": [{"l": "Pneumonia", "p": 0.82, "t": 0.106, "s": "D", "m": -0.05}]}
            self.assertEqual(select_llm_route(v5, str(image), "gpt-5.6-luna")["name"], "vision_escalation")

    def test_reference_polarity_and_relevant_sentence(self):
        self.assertEqual(_polarity(_sentences("No pneumothorax."), "Pneumothorax"), "explicit_negative")
        self.assertEqual(_polarity(_sentences("No focal consolidation."), "Pneumonia"), "surrogate_negative")
        self.assertEqual(_polarity(_sentences("No pneumothorax or pleural effusion."), "Pleural Effusion"),
                         "explicit_negative")
        self.assertEqual(_polarity(_sentences("Possible pneumonia."), "Pneumonia"), "uncertain")
        self.assertEqual(_polarity(_sentences("No definite pneumonia."), "Pneumonia"), "uncertain")
        self.assertEqual(_polarity(_sentences("Pneumonia. No pneumonia."), "Pneumonia"), "conflicted")
        self.assertEqual(_polarity(_sentences("No acute osseous abnormality."), "Pneumonia"), "unmentioned")
        source = {"retriever": {"label_evidence": [{
            "label": "Pneumonia", "classifier_probability": 0.8, "evidence_status": "MIXED",
            "margins": {"M@5": -0.056},
            "quality_checks": {"m3_same_sign_as_m5": True, "m10_strong_reversal": False, "m20_warning": False},
            "supporting_evidence": [{"study_id": "synthetic-reference", "label_support": 0.7, "final_rank_score": 0.652}],
        }]}}
        row = build_minimal_data(source)["f"][0]
        self.assertIn("airspace disease", row["a"]["f"].lower())
        self.assertEqual(row["e"], "non_support_favored")
        self.assertEqual(row["c"], "stable")

    def test_batch_command_uses_luna_high(self):
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "query.png"
            image.write_bytes(b"image")
            command = llm_command({"image_path": str(image), "study_id": "1"}, Path("evidence.json"),
                                  Path("llm"), 1024, False, "gpt-5.6-luna", "high")
        self.assertEqual(command[command.index("--model") + 1], "gpt-5.6-luna")
        self.assertEqual(command[command.index("--reasoning-effort") + 1], "high")
        self.assertIn("--force-vision", command)

    def test_batch_resume_requires_valid_prediction(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "summary_test.json"
            summary = {"requested_model": "gpt-5.6-luna", "model": "gpt-5.6-luna",
                       "reasoning_effort": "high", "evidence_schema": 5,
                       "response": "partial", "prediction": None, "source": {"study_id": "1"}}
            path.write_text(json.dumps(summary), encoding="utf-8")
            self.assertFalse(has_completed_llm_summary(Path(directory), "gpt-5.6-luna", "high"))
            summary["prediction"] = {"problems": ["normal"], "findings": "Clear lungs.",
                                     "impression": "No target pathology."}
            path.write_text(json.dumps(summary), encoding="utf-8")
            self.assertTrue(has_completed_llm_summary(Path(directory), "gpt-5.6-luna", "high"))

    def test_batch_log_reads_utf8_output(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "child.log"
            run_logged([sys.executable, "-c", "print(chr(0x2713))"], log)
            self.assertIn(chr(0x2713), log.read_text(encoding="utf-8"))



    def test_extended_polarity_rules(self):
        # 1. Conflicted
        self.assertEqual(_polarity(_sentences("Atelectasis. No atelectasis."), "Atelectasis"), "conflicted")
        # 2. Uncertain
        self.assertEqual(_polarity(_sentences("Pneumothorax cannot be excluded."), "Pneumothorax"), "uncertain")
        self.assertEqual(_polarity(_sentences("Questionable pleural effusion."), "Pleural Effusion"), "uncertain")
        self.assertEqual(_polarity(_sentences("May represent pulmonary edema."), "Edema"), "uncertain")
        # 3. Direct negative
        self.assertEqual(_polarity(_sentences("Without pneumothorax."), "Pneumothorax"), "explicit_negative")
        self.assertEqual(_polarity(_sentences("Negative for acute cardiomegaly."), "Cardiomegaly"), "explicit_negative")
        self.assertEqual(_polarity(_sentences("Pneumothorax is absent."), "Pneumothorax"), "explicit_negative")
        # 4. Surrogate negative
        self.assertEqual(_polarity(_sentences("No focal consolidation."), "Pneumonia"), "surrogate_negative")
        self.assertEqual(_polarity(_sentences("Normal cardiac silhouette."), "Cardiomegaly"), "surrogate_negative")
        # 5. Positive
        self.assertEqual(_polarity(_sentences("Small right apical pneumothorax."), "Pneumothorax"), "positive")
        self.assertEqual(_polarity(_sentences("Bibasilar subsegmental collapse."), "Atelectasis"), "positive")
        self.assertEqual(_polarity(_sentences("Persistent heart enlargement."), "Cardiomegaly"), "positive")
        # 6. Unmentioned
        self.assertEqual(_polarity(_sentences("No acute osseous abnormality."), "Pneumothorax"), "unmentioned")

    def test_shortest_relevant_sentence_selected(self):
        from generate_minimal_json_for_llm import _reference
        with patch("generate_minimal_json_for_llm._reports", return_value={"999": {
            "findings": "There is extensive bilateral chronic fibrotic change and pulmonary volume loss with severe apical scarring. Bibasilar volume loss.",
            "impression": "No pneumothorax."
        }}):
            ref = _reference({"study_id": "999", "cosine_similarity": 0.8}, "Atelectasis")
            self.assertEqual(ref["f"], "Bibasilar volume loss.")
            self.assertEqual(ref["z"], "positive")

    def test_empty_reference_never_fabricates_normal(self):
        from generate_minimal_json_for_llm import _reference
        with patch("generate_minimal_json_for_llm._reports", return_value={"999": {
            "findings": "",
            "impression": ""
        }}):
            ref = _reference({"study_id": "999", "mesh": ["Normal"], "cosine_similarity": 0.8}, "Pneumothorax")
            self.assertEqual(ref["f"], "Normal")
            self.assertEqual(ref["z"], "unmentioned")

            ref_empty = _reference({"study_id": "999", "cosine_similarity": 0.8}, "Pneumothorax")
            self.assertEqual(ref_empty["f"], "")
            self.assertEqual(ref_empty["z"], "unmentioned")

    def test_deterministic_evidence_strength_and_stability(self):
        from generate_minimal_json_for_llm import _strength
        # Insufficient
        self.assertEqual(_strength({"evidence_status": "I"}, 0.05, {"q": 0.8, "u": 1.0}, None), "insufficient")
        self.assertEqual(_strength({"evidence_status": "S", "evidence_available": False}, 0.05, None, None), "insufficient")
        # Non-support favored
        self.assertEqual(_strength({"evidence_status": "D"}, -0.05, {"q": 0.7, "u": 0.5}, {"q": 0.75}), "non_support_favored")
        # Ambiguous
        self.assertEqual(_strength({"evidence_status": "M"}, 0.01, {"q": 0.7, "u": 0.5}, {"q": 0.69}), "weak_support")
        self.assertEqual(_strength({"evidence_status": "M"}, -0.01, {"q": 0.7, "u": 0.5}, {"q": 0.71}), "ambiguous")
        # Strong support
        self.assertEqual(_strength({"evidence_status": "S"}, 0.04, {"q": 0.8, "u": 0.9}, {"q": 0.7}), "strong_support")

    def test_radiology_prompt_contains_selected_label_checklist(self):
        prompt = MedicalPrompts.get_prompt("radiology").format(evidence="{}")
        self.assertIn("For every compact item, the label has already passed its classifier selection threshold", prompt["system"])
        self.assertIn("non-DenseNet score source", prompt["system"])
        self.assertIn("independently inspect the query image", prompt["system"])
        self.assertIn("when s is M or D, do not reject a selected label solely because non-support", prompt["system"])
        self.assertIn("Resolve the disagreement from visible query-image signs", prompt["system"])
        self.assertIn("visceral pleural line", prompt["system"])
        self.assertIn("absent peripheral lung markings", prompt["system"])

    def test_agreement_routing_policy(self):
        from run_terra_agreement_vision_batch import is_classifier_retriever_agreement
        all_s = {
            "classifier": {"selected_labels": ["Cardiomegaly", "Effusion"]},
            "retriever": {"label_evidence": [
                {"label": "Cardiomegaly", "evidence_status": "SUPPORTED"},
                {"label": "Effusion", "evidence_status": "SUPPORTED"},
            ]}
        }
        agreed, reason, _, _ = is_classifier_retriever_agreement(all_s)
        self.assertTrue(agreed)

        mixed = {
            "classifier": {"selected_labels": ["Cardiomegaly", "Pneumothorax"]},
            "retriever": {"label_evidence": [
                {"label": "Cardiomegaly", "evidence_status": "SUPPORTED"},
                {"label": "Pneumothorax", "evidence_status": "NON_SUPPORT_DOMINANT"},
            ]}
        }
        agreed_m, reason_m, _, _ = is_classifier_retriever_agreement(mixed)
        self.assertFalse(agreed_m)
        self.assertIn("retriever_not_supported", reason_m)


if __name__ == "__main__":
    unittest.main()
