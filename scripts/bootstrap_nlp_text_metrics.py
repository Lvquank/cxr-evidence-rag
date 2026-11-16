#!/usr/bin/env python3
"""Compute paired bootstrap CI for NLP text metrics (BLEU-4, ROUGE-L, BERTScore F1)."""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

df_chk = pd.read_csv(ROOT / "outputs/luna_high_polarity_checklist_v5_206/nlp_per_sample_target6.csv")
df_base = pd.read_csv(ROOT / "outputs/luna_high_evidence_v5_fresh_20260930/nlp_per_sample_target6.csv")

df_chk = df_chk.sort_values("study_id").reset_index(drop=True)
df_base = df_base.sort_values("study_id").reset_index(drop=True)
assert (df_chk["study_id"] == df_base["study_id"]).all()

nlp_cols = [
    "findings_bleu4",
    "findings_rouge_l_f1",
    "findings_bertscore_precision",
    "findings_bertscore_recall",
    "findings_bertscore_f1",
    "impression_bleu4",
    "impression_rouge_l_f1",
    "impression_bertscore_precision",
    "impression_bertscore_recall",
    "impression_bertscore_f1",
]

n = len(df_chk)
rng = np.random.default_rng(20260904)
samples = rng.integers(0, n, size=(2000, n))


def percentile_ci(vals: np.ndarray) -> tuple[float, float]:
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


results = {}
for col in nlp_cols:
    c_vals = df_chk[col].values
    b_vals = df_base[col].values

    c_means = np.array([np.nanmean(c_vals[idx]) for idx in samples])
    b_means = np.array([np.nanmean(b_vals[idx]) for idx in samples])
    deltas = c_means - b_means

    obs_c = float(np.nanmean(c_vals))
    obs_b = float(np.nanmean(b_vals))
    d_obs = obs_c - obs_b

    low, high = percentile_ci(deltas)
    c_low, c_high = percentile_ci(c_means)
    b_low, b_high = percentile_ci(b_means)

    results[col] = {
        "chk_estimate": obs_c,
        "chk_ci_95": [c_low, c_high],
        "base_estimate": obs_b,
        "base_ci_95": [b_low, b_high],
        "delta": d_obs,
        "bootstrap_mean_delta": float(deltas.mean()),
        "ci_95": [low, high],
        "ci_contains_zero": bool(low <= 0.0 <= high),
    }

out_dir = ROOT / "outputs/luna_high_polarity_checklist_v5_206/paired_bootstrap_vs_fresh_v5"
out_dir.mkdir(parents=True, exist_ok=True)
with open(out_dir / "nlp_text_metrics_bootstrap_ci.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print("NLP Bootstrap CI results saved.")
for k, v in results.items():
    print(f"{k}: base={v['base_estimate']:.4f}, chk={v['chk_estimate']:.4f}, delta={v['delta']:+.4f}, ci=[{v['ci_95'][0]:+.4f}, {v['ci_95'][1]:+.4f}], zero={v['ci_contains_zero']}")
