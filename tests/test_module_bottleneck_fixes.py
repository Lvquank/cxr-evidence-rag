import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.generate_minimal_json_for_llm import build_minimal_data
from src.classifier_thresholds import load_classifier_thresholds, threshold_snapshot
from src.chexbert_metrics import compute_chexbert_aggregate, compute_chexbert_per_sample, joined_report


class ThresholdPropagationTests(unittest.TestCase):
    def test_selective_thresholds_survive_compaction_and_legacy_requires_artifact(self):
        path = ROOT / "config/densenet_selective_iu_thresholds_v1.json"
        thresholds, artifact = load_classifier_thresholds(path)
        probabilities = dict.fromkeys(thresholds, 0.0)
        probabilities.update({"Atelectasis": 0.333185, "Pneumonia": 0.15})
        source = {
            "classifier": {"raw_probabilities": probabilities, "selected_labels": ["Atelectasis"],
                           **threshold_snapshot(thresholds, artifact["version"], path)},
            "retriever": {"label_evidence": [{"label": "Atelectasis", "evidence_status": "MIXED"}]},
        }
        compact = build_minimal_data(source, include_all_classifier_probabilities=True)
        self.assertEqual(compact["f"][0]["t"], 0.3)
        self.assertEqual(next(r for r in compact["clf"] if r["l"] == "Pneumonia")["t"], 0.203)
        changed = copy.deepcopy(source)
        changed["classifier"]["thresholds"]["Atelectasis"] = 0.31
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            build_minimal_data(changed)
        changed = copy.deepcopy(source)
        changed["classifier"]["selected_labels"].append("Pneumonia")
        with self.assertRaisesRegex(ValueError, "disagrees"):
            build_minimal_data(changed)
        legacy = copy.deepcopy(source)
        for key in ("thresholds", "thresholds_sha256", "thresholds_path"):
            del legacy["classifier"][key]
        with self.assertRaisesRegex(ValueError, "matching --thresholds-json"):
            build_minimal_data(legacy)
        self.assertEqual(build_minimal_data(legacy, thresholds_path=path)["f"][0]["t"], 0.3)

    def test_historical_default_compact_is_unchanged(self):
        root = ROOT / "outputs/luna_high_evidence_v5_fresh_20260930"
        evidence = json.loads((root / "evidence/loo_001_cardiomegaly.json").read_text(encoding="utf-8"))
        existing = next((root / "llm/loo_001_cardiomegaly").glob("minimal_*.json"))
        self.assertEqual(build_minimal_data(evidence), json.loads(existing.read_text(encoding="utf-8")))


class ClinicalMetricTests(unittest.TestCase):
    def test_omission_and_uncertainty_cannot_remove_reference_positive(self):
        reference = pd.DataFrame({"Edema": [1, np.nan, 1, -1]})
        predicted = pd.DataFrame({"Edema": [np.nan, 1, -1, 1]})
        result = compute_chexbert_aggregate(reference, predicted, ["Edema"])
        self.assertEqual((result["tp"], result["fp"], result["fn"]), (0, 1, 2))
        self.assertEqual(result["per_label_counts"]["Edema"], {"tp": 0, "fp": 1, "fn": 2, "f1": 0.0})
        self.assertEqual(result["evaluated_label_decisions"], 3)
        accuracy, f1 = compute_chexbert_per_sample(reference, predicted, ["Edema"])
        np.testing.assert_array_equal(accuracy[:3], [0, 0, 0])
        np.testing.assert_array_equal(f1[:3], [0, 0, 0])
        self.assertTrue(np.isnan(f1[3]))
        with self.assertRaisesRegex(ValueError, "Empty clinical report"):
            joined_report(np.nan, "")
        self.assertEqual(joined_report("", "No effusion."), "No effusion.")

    def test_evaluators_reject_stale_cached_report_text(self):
        from scripts.evaluate_clinical_metrics_for_pipeline import run_chexbert_labeler as target_labeler
        from scripts.evaluate_clinical_metrics_full import run_chexbert_labeler as comparison_labeler
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reports = root / "reports.csv"
            cached = root / "labeled_reports.csv"
            pd.DataFrame({"Report Impression": ["No edema."]}).to_csv(reports, index=False)
            pd.DataFrame({"Report Impression": ["Pulmonary edema."], "Edema": [1]}).to_csv(cached, index=False)
            for labeler in (target_labeler, comparison_labeler):
                with self.assertRaisesRegex(ValueError, "cache report text/order differs"):
                    labeler(reports, root, root / "unused_checkpoint")

    def test_sample_and_label_macro_are_named_separately(self):
        from scripts.evaluate_llm_outputs import aggregate_target_problems
        result = json.loads((ROOT / "outputs/luna_high_evidence_v5_fresh_20260930/nlp_metrics_target6.json").read_text(encoding="utf-8"))
        aggregate = aggregate_target_problems(result["records"])
        self.assertEqual(aggregate["sample_f1"], aggregate["macro_f1"])
        self.assertAlmostEqual(aggregate["pathology_macro_f1"], np.mean([
            label["f1"] for label in aggregate["per_target_label"].values()]))
        self.assertNotAlmostEqual(aggregate["sample_f1"], aggregate["pathology_macro_f1"])


if __name__ == "__main__":
    unittest.main()
