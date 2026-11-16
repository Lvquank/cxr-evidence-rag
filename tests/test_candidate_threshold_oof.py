import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from select_candidate_thresholds_oof import multilabel_folds, sweep_thresholds


class CandidateThresholdSelectionTests(unittest.TestCase):
    def test_sweep_uses_incremental_region_and_fp_budget(self):
        frame = pd.DataFrame({
            "raw_probability": [0.8, 0.49, 0.48, 0.47, 0.2],
            "gold": [1, 0, 1, 0, 1],
        })
        threshold, sweep = sweep_thresholds(frame, main=0.5, fp_budget=1,
                                            rate_cap=0.6, outer_fold=0)
        self.assertEqual(threshold, 0.48)
        self.assertEqual(sweep[-1]["additional_TP"], 2)
        self.assertEqual(sweep[-1]["additional_FP"], 2)
        self.assertFalse(sweep[-1]["eligible"])

    def test_sweep_disables_when_no_true_positive_is_feasible(self):
        frame = pd.DataFrame({"raw_probability": [0.49, 0.48, 0.47],
                              "gold": [0, 0, 1]})
        threshold, _ = sweep_thresholds(frame, main=0.5, fp_budget=1,
                                        rate_cap=1.0, outer_fold=0)
        self.assertIsNone(threshold)

    def test_multilabel_folds_are_deterministic_and_study_level(self):
        labels = ["Atelectasis", "Pneumothorax"]
        frame = pd.DataFrame([
            {"study_id": str(study), "pathology": label,
             "gold": int(label == labels[study % 2])}
            for study in range(20) for label in labels
        ])
        ids = sorted(frame.study_id.unique())
        first = multilabel_folds(frame, ids, 5, 17, labels)
        self.assertEqual(first, multilabel_folds(frame, ids, 5, 17, labels))
        self.assertEqual(len(first), 20)
        self.assertEqual(sorted(first.values()).count(0), 4)
        self.assertEqual({fold: list(first.values()).count(fold) for fold in range(5)},
                         {fold: 4 for fold in range(5)})


if __name__ == "__main__":
    unittest.main()
