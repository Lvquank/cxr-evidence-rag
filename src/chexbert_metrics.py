"""Positive-label agreement: blank is not positive; uncertain references are excluded."""

import numpy as np
import pandas as pd


def joined_report(findings, impression) -> str:
    text = "\n".join(str(value).strip() for value in (findings, impression)
                     if pd.notna(value) and str(value).strip() and str(value).lower() != "nan")
    if not text:
        raise ValueError("Empty clinical report; exclude it explicitly instead of inventing a normal report.")
    return text


def _decisions(ref_df, pred_df, labels):
    if len(ref_df) != len(pred_df):
        raise ValueError("CheXbert reference and prediction row counts differ.")
    reference = ref_df[labels].to_numpy(dtype=float)
    prediction = pred_df[labels].to_numpy(dtype=float)
    if any(not np.all(np.isnan(values) | np.isin(values, [-1, 0, 1])) for values in (reference, prediction)):
        raise ValueError("CheXbert labels must be blank, -1, 0, or 1.")
    # The denominator depends only on the reference, so omissions cannot hide FN.
    return reference == 1, prediction == 1, reference != -1


def compute_chexbert_aggregate(ref_df, pred_df, labels) -> dict:
    reference, prediction, mask = _decisions(ref_df, pred_df, labels)
    tp = np.sum(reference & prediction & mask, axis=0)
    fp = np.sum(~reference & prediction & mask, axis=0)
    fn = np.sum(reference & ~prediction & mask, axis=0)
    denominators = 2 * tp + fp + fn
    per_label = np.divide(2 * tp, denominators, out=np.zeros(len(labels)), where=denominators != 0)
    total = int(denominators.sum())
    return {
        "micro_f1": float(2 * tp.sum() / total) if total else 0.0,
        "macro_f1": float(per_label.mean()) if len(labels) else 0.0,
        "per_label_f1": dict(zip(labels, per_label.tolist())),
        "per_label_counts": {
            label: {"tp": int(tp[i]), "fp": int(fp[i]), "fn": int(fn[i]), "f1": float(per_label[i])}
            for i, label in enumerate(labels)
        },
        "evaluated_label_decisions": int(mask.sum()),
        "tp": int(tp.sum()), "fp": int(fp.sum()), "fn": int(fn.sum()),
        "uncertain_policy": "exclude_reference_uncertain; prediction_uncertain_is_not_positive",
        "blank_policy": "not_positive_including_omitted_prediction",
        "protocol_version": "chexbert-positive-agreement-v2",
        "averaging": "positive-class micro F1 and mean positive-class F1 across labels; zero division=0",
    }


def compute_chexbert_per_sample(ref_df, pred_df, labels):
    reference, prediction, mask = _decisions(ref_df, pred_df, labels)
    tp = np.sum(reference & prediction & mask, axis=1)
    fp = np.sum(~reference & prediction & mask, axis=1)
    fn = np.sum(reference & ~prediction & mask, axis=1)
    count = mask.sum(axis=1)
    accuracy = np.divide(((reference == prediction) & mask).sum(axis=1), count,
                         out=np.full(len(ref_df), np.nan), where=count != 0)
    denominator = 2 * tp + fp + fn
    f1 = np.divide(2 * tp, denominator, out=np.ones(len(ref_df)), where=denominator != 0)
    f1[count == 0] = np.nan
    return accuracy, f1
