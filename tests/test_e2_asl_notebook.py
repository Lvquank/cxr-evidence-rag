"""Run with .venv-bertscore/Scripts/python.exe -m unittest discover -s tests -p test_e2_asl_notebook.py."""

import ast
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from sklearn.metrics import (
    average_precision_score, confusion_matrix, f1_score, precision_recall_curve,
    precision_score, recall_score, roc_auc_score,
)


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/train_chexpert_e2_asl_compare_bce.ipynb"


def notebook_functions():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    namespace = dict(globals())
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        tree = ast.parse(source)
        definitions = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))]
        exec(compile(ast.Module(body=definitions, type_ignores=[]), f"cell-{index}", "exec"), namespace)
    return namespace, notebook


class E2ASLNotebookTests(unittest.TestCase):
    def setUp(self):
        self.ns, self.notebook = notebook_functions()
        self.ns["METRIC_COLUMNS"] = ["AUROC", "AUPRC", "Precision", "Recall", "Specificity", "F1"]

    def test_asl_reduces_to_bce_without_focusing_or_clipping(self):
        logits = torch.tensor([-4., -1., 0., 1., 4.], requires_grad=True)
        targets = torch.tensor([0., 1., 0., 1., 1.])
        actual = self.ns["AsymmetricLoss"](gamma_neg=0, gamma_pos=0, clip=0)(logits, targets)
        expected = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        torch.testing.assert_close(actual, expected)
        actual.sum().backward()
        torch.testing.assert_close(logits.grad, logits.detach().sigmoid() - targets)

    def test_asl_discards_easy_negatives_and_remains_finite(self):
        logits = torch.tensor([-1000., -5., 0., 1000.], requires_grad=True)
        targets = torch.tensor([0., 0., 1., 1.])
        values = self.ns["AsymmetricLoss"]()(logits, targets)
        self.assertTrue(torch.isfinite(values).all())
        torch.testing.assert_close(values[:2], torch.zeros(2))
        values.sum().backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        torch.testing.assert_close(logits.grad[:2], torch.zeros(2))
        self.assertTrue(torch.is_grad_enabled())

    def test_masked_binary_loss_and_multiclass_ce(self):
        ns = self.ns
        ns.update(
            DEVICE=torch.device("cpu"), MAIN_LABELS=["Pneumonia", "Cardiomegaly"],
            BINARY_LABELS=["Pneumonia"], MULTICLASS_LABELS=["Cardiomegaly"],
            LOSS_MODE="asl", AUX_LOSS_WEIGHT=0.35,
            asl_loss=ns["AsymmetricLoss"](),
        )
        binary = torch.tensor([[0.5], [-0.7]], requires_grad=True)
        multiclass = torch.tensor([[0.1, 0.5, -0.3], [0.2, -0.2, 0.9]], requires_grad=True)
        batch = {
            "binary_targets": {"Pneumonia": torch.tensor([1., 0.])},
            "binary_masks": {"Pneumonia": torch.tensor([1., 0.])},
            "multiclass_targets": {"Cardiomegaly": torch.tensor([1, 2])},
            "multiclass_masks": {"Cardiomegaly": torch.ones(2)},
        }
        loss, parts = ns["compute_loss"]({"Pneumonia": binary, "Cardiomegaly": multiclass}, batch)
        ce = F.cross_entropy(multiclass, torch.tensor([1, 2]))
        self.assertAlmostEqual(parts["Cardiomegaly"], ce.item())
        expected = (ns["asl_loss"](binary[0, 0], torch.tensor(1.)) + ce) / 2
        torch.testing.assert_close(loss, expected)
        loss.backward()
        self.assertEqual(binary.grad[1, 0].item(), 0.)
        self.assertLess(binary.grad[0, 0].item(), 0.)

    @staticmethod
    def predictions(probabilities):
        return {label: {
            "y": np.array([0, 1, 0, 1, 0, 1]), "p": np.array(probabilities),
            "path": np.array([f"image-{i}" for i in range(6)]),
            "patient_id": np.array(["a", "a", "b", "b", "c", "c"]),
        } for label in ["Pneumonia", "Pneumothorax"]}

    def test_paired_comparison_and_patient_bootstrap(self):
        ns = self.ns
        a = self.predictions([0.1, 0.9, 0.2, 0.8, 0.3, 0.7])
        b = self.predictions([0.9, 0.1, 0.8, 0.2, 0.7, 0.3])
        thresholds = {label: 0.5 for label in a}
        am = ns["threshold_metrics"](a, thresholds, list(a))
        bm = ns["threshold_metrics"](b, thresholds, list(b))
        result = ns["compare_metrics"](am, bm).set_index("Label")
        self.assertEqual(result.loc["Macro", "Delta_AUROC"], 1.)
        self.assertEqual(result.loc["Pneumonia", "Delta_F1"], 1.)
        ci = ns["paired_patient_bootstrap"](a, b, list(a), repeats=30)
        self.assertTrue(ci["CI95_low"].gt(0).all())
        self.assertTrue(ci["N_bootstrap"].eq(30).all())
        identical = ns["paired_patient_bootstrap"](a, a, list(a), repeats=30)
        np.testing.assert_array_equal(identical[["CI95_low", "CI95_high"]], 0.)
        b["Pneumonia"]["path"][0] = "other"
        with self.assertRaisesRegex(ValueError, "path"):
            ns["paired_patient_bootstrap"](a, b, list(a), repeats=2)

    def test_single_class_counts_and_development_threshold(self):
        ns = self.ns
        pred = {"Pneumonia": {"y": np.array([1, 1]), "p": np.array([0.9, 0.1])}}
        row = ns["threshold_metrics"](pred, {"Pneumonia": 0.5}, ["Pneumonia"]).iloc[0]
        self.assertEqual((row["TP"], row["FN"], row["TN"], row["FP"]), (1, 1, 0, 0))
        self.assertTrue(np.isnan(row["AUROC"]))
        y = np.array([0, 0, 1, 1, 1, 1])
        p = np.array([0.2, 0.6, 0.3, 0.5, 0.7, 0.9])
        info = ns["choose_threshold_for_recall"](y, p, 0.85)
        self.assertGreaterEqual(recall_score(y, p >= info["threshold"]), 0.85)
        with self.assertRaises(ValueError):
            ns["choose_threshold_for_recall"](np.ones(3), np.arange(3), 0.85)

    def test_report_generation(self):
        ns = self.ns
        a = self.predictions([0.1, 0.9, 0.2, 0.8, 0.3, 0.7])
        b = self.predictions([0.9, 0.1, 0.8, 0.2, 0.7, 0.3])
        thresholds = {label: 0.5 for label in a}
        am = ns["threshold_metrics"](a, thresholds, list(a))
        bm = ns["threshold_metrics"](b, thresholds, list(b))
        ns.update(
            comparison_df=ns["compare_metrics"](am, bm).assign(Split="internal_valid"),
            bootstrap_df=ns["paired_patient_bootstrap"](a, b, list(a), repeats=30).assign(Split="internal_valid"),
            BOOTSTRAP_REPEATS=30, CHECKPOINT_PATH=Path("asl.pt"),
            BCE_CHECKPOINT_PATH=Path("bce.pt"), ckpt={"epoch": 1}, bce_ckpt={"epoch": 2}, run_config={},
        )
        with tempfile.TemporaryDirectory(dir=ROOT / "scratch", prefix="test_e2_asl_") as folder:
            ns["OUT_DIR"] = Path(folder)
            self.assertTrue(ns["OUT_DIR"].resolve().is_relative_to(ROOT.resolve()))
            with contextlib.redirect_stdout(io.StringIO()):
                exec("".join(self.notebook["cells"][-1]["source"]), ns)
            report = (Path(folder) / "comparison_report.md").read_text(encoding="utf-8")
            self.assertIn("\n| Macro | AUROC", report)
            self.assertIn("BCE thiếu metadata", report)
            self.assertIn("ASL cao hơn; CI không chứa 0", report)


if __name__ == "__main__":
    unittest.main()
