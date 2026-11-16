import csv
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_llm_outputs import (
    add_bertscore, bleu4, evaluate_summaries, load_ontology, load_reports,
    normalize_target_problems, problem_metrics, target_problem_metrics, text_metrics, write_per_sample_csv,
)
from create_leave_one_out_manifest import LABELS, select_manifest
from run_leave_one_out_batch import command_for_request
from run_single_llm import load_minimal_json, validate_prediction


class LLMEvaluationTests(unittest.TestCase):
    def test_joins_study_and_scores_structured_prediction(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            reports_csv = folder / "reports.csv"
            with reports_csv.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["uid", "Problems", "findings", "impression"])
                writer.writeheader()
                writer.writerow({
                    "uid": "7", "Problems": "Cardiomegaly;Pulmonary Edema",
                    "findings": "Mild cardiomegaly with edema.", "impression": "Mild pulmonary edema.",
                })
            summary = folder / "summary_test.json"
            summary.write_text(json.dumps({
                "source": {"study_id": "7", "image_name": "scan.png"},
                "prediction": {
                    "problems": ["Cardiomegaly", "Pulmonary Edema"],
                    "findings": "Mild cardiomegaly with edema.",
                    "impression": "Mild pulmonary edema.",
                },
            }), encoding="utf-8")

            records, skipped = evaluate_summaries([summary], load_reports(reports_csv), {})
            self.assertFalse(skipped)
            self.assertEqual(records[0]["problems"]["f1"], 1.0)
            self.assertEqual(records[0]["findings"]["rouge_l_f1"], 1.0)
            self.assertEqual(records[0]["impression"]["bleu4"], 1.0)

    def test_hallucination_rate_marks_unsupported_labels(self):
        result = problem_metrics(["Cardiomegaly", "Pneumonia"], ["Cardiomegaly"])
        self.assertEqual(result["hallucinated_labels"], ["Pneumonia"])
        self.assertEqual(result["hallucination_rate"], 0.5)
        self.assertGreater(bleu4("mild edema", "mild pulmonary edema"), 0.0)

    def test_target_ontology_maps_synonyms_and_handles_normal_separately(self):
        ontology = load_ontology()
        predicted = normalize_target_problems(["Atelectasis", "Catheters, Indwelling"], ontology)
        reference = normalize_target_problems(["Pulmonary Atelectasis", "normal"], ontology)
        result = target_problem_metrics(predicted, reference)
        self.assertEqual(result["tp"], 1)
        self.assertEqual(result["fp"], 0)
        self.assertFalse(result["normal_decision"]["gold_normal"])
        self.assertEqual(predicted["other_findings"], ["Catheters, Indwelling"])

        normal_result = target_problem_metrics(
            normalize_target_problems([], ontology), normalize_target_problems(["normal"], ontology)
        )
        self.assertEqual(normal_result["normal_decision"]["tp"], 1)

    def test_empty_text_reference_is_excluded_not_scored_as_zero(self):
        metrics = text_metrics("Generated finding.", "")
        self.assertFalse(metrics["included"])
        self.assertIsNone(metrics["bleu4"])

    def test_writes_target_problem_columns_to_csv(self):
        ontology = load_ontology()
        predicted = normalize_target_problems(["Atelectasis"], ontology)
        reference = normalize_target_problems(["Pulmonary Atelectasis"], ontology)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "metrics.csv"
            write_per_sample_csv(output, [{
                "summary_file": "summary.json", "study_id": "1", "image_name": "scan.png",
                "evidence_states": ["S"], "predicted_problems_raw": ["Atelectasis"], "gold_problems_raw": ["Pulmonary Atelectasis"],
                "predicted_target_problems": predicted["target_problems"], "gold_target_problems": reference["target_problems"],
                "problems": target_problem_metrics(predicted, reference),
                "findings": text_metrics("finding", "finding"), "impression": text_metrics("impression", "impression"),
            }])
            with output.open(encoding="utf-8", newline="") as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 1)

    def test_adds_bertscore_to_findings_and_impression(self):
        records = [{
            "prediction_text": {"findings": "a", "impression": "b"},
            "reference_text": {"findings": "a", "impression": "b"},
            "findings": {}, "impression": {},
        }]
        fake_module = types.SimpleNamespace(score=lambda *args, **kwargs: ([0.8], [0.9], [0.85]))
        with patch.dict(sys.modules, {"bert_score": fake_module}):
            metadata = add_bertscore(records, "roberta-large", 8, None)
        self.assertTrue(metadata["computed"])
        self.assertEqual(records[0]["findings"]["bertscore_f1"], 0.85)
        self.assertEqual(records[0]["impression"]["bertscore_precision"], 0.8)

    def test_bertscore_excludes_empty_references(self):
        records = [
            {
                "prediction_text": {"findings": "a", "impression": "b"},
                "reference_text": {"findings": "a", "impression": "b"},
                "findings": text_metrics("a", "a"), "impression": text_metrics("b", "b"),
            },
            {
                "prediction_text": {"findings": "ignored", "impression": "c"},
                "reference_text": {"findings": "", "impression": "c"},
                "findings": text_metrics("ignored", ""), "impression": text_metrics("c", "c"),
            },
        ]
        calls = []

        def fake_score(candidates, references, **kwargs):
            calls.append((candidates, references))
            return ([0.8] * len(candidates), [0.9] * len(candidates), [0.85] * len(candidates))

        fake_module = types.SimpleNamespace(score=fake_score)
        with patch.dict(sys.modules, {"bert_score": fake_module}):
            metadata = add_bertscore(records, "roberta-large", 8, None)
        self.assertEqual(metadata["sections"]["findings"]["eligible"], 1)
        self.assertEqual(calls[0][0], ["a"])
        self.assertIsNone(records[1]["findings"]["bertscore_f1"])

    def test_single_runner_accepts_minimal_input_and_validates_output(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "minimal.json"
            source.write_text('{"v":4,"f":[["Edema",0.9,"S",["Pulmonary Edema"]]]}', encoding="utf-8")
            self.assertEqual(load_minimal_json(source)["v"], 4)
        self.assertEqual(
            validate_prediction('{"problems":["Edema"],"findings":"Edema.","impression":"Edema."}'),
            {"problems": ["Edema"], "findings": "Edema.", "impression": "Edema."},
        )

    def test_leave_one_out_selection_is_206_unique_studies(self):
        pool = {}
        for label in LABELS:
            count = 26 if label == "Pneumothorax" else 30
            pool[label] = [{"study_id": f"{label}-{index}", "image_id": str(index), "image_path": "test.png", "support_score": 1.0} for index in range(count)]
        selected = select_manifest(pool, per_class=30, seed=1)
        self.assertEqual(len(selected), 206)
        self.assertEqual(len({item["study_id"] for item in selected}), 206)
        self.assertEqual(sum(item["case_label"] == "Pneumothorax" for item in selected), 26)

    def test_batch_command_temporarily_excludes_its_query_study(self):
        commands = command_for_request(
            {"request_id": "loo_001_normal", "study_id": "99", "image_path": "E:/query.png", "case_label": "normal"},
            Path("out"), "cpu", 160,
        )
        self.assertIn("--exclude-study-id", commands[0])
        self.assertEqual(commands[0][commands[0].index("--exclude-study-id") + 1], "99")
        self.assertIn("--study-id", commands[1])
        self.assertIn("gpt-5.6-luna", commands[1])

    def test_skips_dry_run_without_prediction(self):
        with tempfile.TemporaryDirectory() as directory:
            summary = Path(directory) / "summary_test.json"
            summary.write_text(json.dumps({"response": "[dry-run: request was not sent]"}), encoding="utf-8")
            records, skipped = evaluate_summaries([summary], {"1": {}}, {})
            self.assertEqual(records, [])
            self.assertEqual(skipped[0]["reason"], "missing_structured_prediction")


if __name__ == "__main__":
    unittest.main()
