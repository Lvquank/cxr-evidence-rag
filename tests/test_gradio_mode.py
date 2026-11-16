"""Check that comparison submits all four branches regardless of the saved radio."""
import unittest
from unittest.mock import patch

from gradio_app import app


class AnalysisModeTests(unittest.TestCase):
    def test_comparison_ignores_previous_single_method_and_preserves_settings(self):
        state = {"id": "test", "case_name": "Comparison", "results": []}
        with patch.object(app.backend, "create_case", return_value=state) as create, patch.object(app.backend, "run_case", return_value=iter([])):
            list(app.run_analysis(["image.png"], "Comparison", "gpt-5.6-terra", app.MODE_COMPARE, "structured_learned_gate", "medium", "1614", "resnet50"))
        self.assertEqual(["structured_rule", "structured_learned_gate", "baseline_a", "baseline_b"], create.call_args.args[3])
        self.assertEqual(("gpt-5.6-terra", "medium", "1614"), (create.call_args.args[2], *create.call_args.args[4:6]))
        self.assertEqual("resnet50", create.call_args.kwargs["classifier_model"])

    def test_single_mode_submits_only_the_preserved_method(self):
        state = {"id": "test", "case_name": "Single", "results": []}
        with patch.object(app.backend, "create_case", return_value=state) as create, patch.object(app.backend, "run_case", return_value=iter([])):
            list(app.run_analysis(["image.png"], "Single", "gpt-5.6-luna", "Chọn một", "baseline_b", "high", "", "densenet121"))
        self.assertEqual(["baseline_b"], create.call_args.args[3])


if __name__ == "__main__":
    unittest.main()
