"""Check the FP-constrained decision rule used by classifier experiments."""

import unittest

import numpy as np

from scripts.benchmark_classifier_refinements import constrained_threshold, decide, choose
from src.classifier_refinement import predict_refinement


class RefinementTests(unittest.TestCase):
    def test_threshold_maximizes_tp_without_exceeding_fp_budget(self):
        gold = np.array([1, 0, 1, 0, 1], dtype=bool)
        scores = np.array([.9, .8, .7, .6, .5])
        threshold, quality = constrained_threshold(gold, scores, 1)
        self.assertEqual(threshold, .7)
        self.assertEqual(quality, (2, 1))

    def test_tied_scores_cannot_split_positives_from_negatives(self):
        threshold, quality = constrained_threshold(np.array([1, 0], dtype=bool), np.array([.5, .5]), 0)
        self.assertIsNone(threshold)
        self.assertEqual(quality, (0, 0))
        self.assertFalse(decide(np.array([.5, .9]), threshold).any())

    def test_selector_rejects_recall_gain_paid_with_extra_fp(self):
        gold = np.array([1, 1, 0, 0], dtype=bool)
        baseline = np.array([1, 0, 0, 0], dtype=bool)
        candidates = {"unsafe":np.array([1, 1, 1, 0], dtype=bool),
                      "better":np.array([1, 1, 0, 0], dtype=bool)}
        self.assertEqual(choose(gold, baseline, candidates), "better")
        self.assertEqual(choose(gold, baseline, {"unsafe":candidates["unsafe"]}), "baseline")

    def test_refinement_preserves_untrained_label(self):
        bundle={"labels":["Atelectasis","Pneumothorax"],
                "baseline_thresholds":{"labels":{"Atelectasis":{"threshold":.3},"Pneumothorax":{"threshold":.46}}},
                "models":{"Atelectasis":{"method":"threshold_tuned","model":None,"threshold":0.}}}
        pred,_=predict_refinement(bundle,np.array([[.4,.5],[.8,.2]]),{})
        self.assertEqual(pred.tolist(),[[False,True],[True,False]])
        bundle["mode"]="add_only"
        pred,_=predict_refinement(bundle,np.array([[.4,.5],[.8,.2]]),{})
        self.assertEqual(pred.tolist(),[[True,True],[True,False]])
        with self.assertRaises(ValueError):
            predict_refinement(bundle,np.array([[float("nan"),.5]]),{})


if __name__ == "__main__":
    unittest.main()
