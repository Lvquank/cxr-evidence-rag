import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.classifier_refinement import refinement_scores
from src.classifier_thresholds import load_classifier_thresholds, threshold_snapshot
from scripts.generate_minimal_json_for_llm import build_minimal_data


class ProbabilityProbe:
    def predict_proba(self, features):
        return np.column_stack([1 - features[:, 0], features[:, 0]])


class ModuleImprovementTests(unittest.TestCase):
    def test_probe_score_is_used_with_probe_threshold_and_raw_dense_is_preserved(self):
        thresholds, artifact = load_classifier_thresholds()
        labels = list(thresholds)
        raw = np.zeros((1, 6))
        raw[0, labels.index("Atelectasis")] = .333185
        raw[0, labels.index("Pneumonia")] = .10
        bundle = {"labels": labels, "baseline_thresholds": artifact, "models": {
            "Atelectasis": {"method": "threshold_tuned", "threshold": float(np.log(.3/.7)),
                            "raw_threshold": .3, "model": None},
            "Pneumonia": {"method": "chexpert_rad_probe", "threshold": .67, "model": ProbabilityProbe()}}}
        selected, scores, updated, sources = refinement_scores(bundle, raw, {"chexpert_rad": np.array([[.8]])})
        source = {"classifier": {"raw_probabilities": dict(zip(labels, raw[0])),
                    "selection_probabilities": dict(zip(labels, scores[0])), "score_sources": sources,
                    "selected_labels": [n for n, value in zip(labels, selected[0]) if value],
                    **threshold_snapshot(updated, "candidate-v1")},
                  "retriever": {"label_evidence": [{"label": "Pneumonia", "evidence_status": "MIXED"}]}}
        compact = build_minimal_data(source, include_all_classifier_probabilities=True)
        self.assertEqual(source["classifier"]["raw_probabilities"]["Pneumonia"], .1)
        self.assertEqual(compact["f"][0]["p"], .8)
        self.assertEqual(compact["f"][0]["t"], .67)
        self.assertEqual(compact["f"][0]["o"], "chexpert_rad_probe")
        self.assertIn("Atelectasis", source["classifier"]["selected_labels"])

    def test_context_support_is_excluded_from_both_strong_and_absence_banks(self):
        from src.retriever.faiss_retriever import FAISSImageRetriever
        retriever = object.__new__(FAISSImageRetriever)
        retriever.minimum_label_support = .5
        retriever.metadata = [{"support": [value] * 6} for value in (.2, .65, 1., 0.)]
        retriever._build_label_banks()
        bank = retriever.label_banks["Pneumonia"]
        np.testing.assert_array_equal(bank["positive"], [1, 2])
        np.testing.assert_array_equal(bank["non_support"], [3])

    def test_end_to_end_bootstrap_statistics_include_empty_normal_samples(self):
        from scripts.evaluate_module_end_to_end import label_metrics
        gold = np.zeros((2, 6), dtype=bool)
        predicted = gold.copy()
        gold[0, 0] = predicted[0, 0] = True
        predicted[1, 1] = True
        result = label_metrics(gold, predicted)
        self.assertEqual((result["tp"], result["fp"], result["fn"]), (1, 1, 0))
        self.assertAlmostEqual(result["micro_f1"], 2/3)
        self.assertAlmostEqual(result["pathology_macro_f1"], 1/6)
        self.assertEqual(result["sample_f1"], .5)

    def test_error_attribution_partitions_false_negatives_and_filtered_false_positives(self):
        from scripts.evaluate_module_end_to_end import error_sources
        gold = np.array([[True, True, False, False, True, False]])
        selected = np.array([[False, True, True, True, True, False]])
        predicted = np.array([[False, False, False, True, True, False]])
        result = error_sources(gold, selected, predicted)
        self.assertEqual(result["totals"], dict.fromkeys([
            "preselection_fn", "rejected_true_labels", "removed_false_labels",
            "retained_false_labels", "retained_true_labels"], 1))
        with self.assertRaisesRegex(ValueError, "outside the selected-only schema"):
            error_sources(gold, selected, gold)


if __name__ == "__main__":
    unittest.main()
