"""Check live gate reconstruction against all 1,236 frozen OOF decisions."""

import json
import tempfile
import unittest

import joblib
import numpy as np
import pandas as pd

from pathlib import Path

from scripts.profile_learned_gate_latency import GATE_ROOT, ROOT, apply_gate, feature_rows, save_results, resume_rows
from scripts.run_nested_gate_threshold_oof import NUMERIC_FEATURES
from src.classifier_thresholds import evidence_thresholds, load_classifier_thresholds, threshold_snapshot


class LearnedGateLatencyTests(unittest.TestCase):
    def test_resume_keeps_successes_and_archives_failed_attempt(self):
        config = dict(branch="rule", requested=206, device="cuda", model="gpt-5.6-luna",
                      reasoning_effort="high", max_output_tokens=8192, lambda_global=0.6, manifest_sha256="frozen")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertEqual(resume_rows(root, config, False), ([], 0.0))
            (root / "llm/loo_002").mkdir(parents=True)
            (root / "llm/loo_002/invalid.txt").write_text("failed")
            (root / "latency_summary.json").write_text(json.dumps({**config, "batch_wall_s": 12., "results": [
                {"request_id":"loo_001", "status":"success"}, {"request_id":"loo_002", "status":"failed"}]}))
            with self.assertRaisesRegex(ValueError, "reasoning_effort"):
                resume_rows(root, {**config, "reasoning_effort":"none"}, True)
            with self.assertRaisesRegex(ValueError, "lambda_global"):
                resume_rows(root, {**config, "lambda_global": 0.9}, True)
            rows, elapsed = resume_rows(root, config, True)
            self.assertEqual([r["request_id"] for r in rows], ["loo_001"])
            self.assertEqual(elapsed, 12.)
            self.assertFalse((root / "llm/loo_002").exists())
            self.assertEqual(len(list((root / "failed_attempts").rglob("invalid.txt"))), 1)

    def test_baseline_resume_without_rerank_weight(self):
        config = dict(branch="baseline_a", requested=206, device="cuda", model="gpt-5.6-luna",
                      reasoning_effort="high", max_output_tokens=8192, manifest_sha256="frozen")
        rows = [{"request_id": "loo_001", "status": "success"}]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "latency_summary.json").write_text(json.dumps({**config, "batch_wall_s": 2., "results": rows}))
            self.assertEqual(resume_rows(root, config, True), (rows, 2.0))

    def test_rule_summary_does_not_report_io_as_learned_gate_inference(self):
        timings = ["classifier_retriever_s", "rule_evidence_io_s", "llm_pipeline_s", "end_to_end_s"]
        rows = [dict(status="success", classifier_retriever_s=10.0, rule_evidence_io_s=0.001,
                     llm_pipeline_s=value, end_to_end_s=10.001 + value) for value in (2.0, 4.0)]
        with tempfile.TemporaryDirectory() as directory:
            result = save_results(Path(directory), rows, {"branch": "rule", "timing_fields": timings}, 26.002, True)
            self.assertNotIn("learned_gate_s", result["latency"])
            self.assertAlmostEqual(result["latency"]["end_to_end_s"]["mean_s"], 13.001)
            self.assertAlmostEqual(result["latency"]["end_to_end_s"]["p95_s"], 13.901)
            self.assertEqual(result["succeeded"], 2)
            self.assertEqual(result["failed"], 0)

    def test_replays_all_frozen_features_probabilities_and_output_labels(self):
        reference = pd.read_csv(GATE_ROOT / "oof_gate_predictions.csv", dtype={"study_id": str})
        folds = json.loads((GATE_ROOT / "folds.json").read_text())["study_to_fold"]
        artifacts = {fold: joblib.load(GATE_ROOT / "models" / f"fold_{fold}.pkl") for fold in range(5)}
        thresholds, metadata = load_classifier_thresholds()
        for request_id, group in reference.groupby("request_id", sort=False):
            evidence = json.loads((ROOT / "outputs/leave_one_out_206/evidence" / f"{request_id}.json").read_text())
            actual = feature_rows(evidence)
            expected = group.set_index("pathology").loc[actual.pathology]
            np.testing.assert_allclose(actual.loc[:, NUMERIC_FEATURES].to_numpy(float),
                                       expected.loc[:, NUMERIC_FEATURES].to_numpy(float), atol=1e-12, rtol=0)
            study_id = str(group.study_id.iloc[0])
            gated = apply_gate(evidence, artifacts[folds[study_id]], study_id)
            np.testing.assert_allclose([row["probability"] for row in gated["learned_gate"]["decisions"]],
                                       expected.gate_probability.to_numpy(), atol=1e-12, rtol=0)
            historical = json.loads((GATE_ROOT / "evidence" / f"{request_id}.json").read_text())
            self.assertEqual(gated["classifier"]["selected_labels"], historical["classifier"]["selected_labels"])
            gated["classifier"].update(threshold_snapshot(thresholds, metadata["version"]))
            self.assertEqual(evidence_thresholds(gated), thresholds)
        with self.assertRaisesRegex(ValueError, "not held out"):
            apply_gate(evidence, artifacts[(folds[study_id] + 1) % 5], study_id)
        gated["classifier"]["selected_labels"] = ["Invalid"]
        with self.assertRaisesRegex(ValueError, "learned-gate"):
            evidence_thresholds(gated)


if __name__ == "__main__":
    unittest.main()
