"""Compare two audited high-effort retrieval λ runs on the saved paired study draws."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from finalize_high_effort_comparison import BRANCHES, scores, sufficient
from recompute_latency_tables import (
    LABELS14, SEED, SAMPLES, T5, T6, read_csv, read_json, require_cuda, save_csv, save_json, tensor,
)
from validate_common_studies import sha256


def load_audited_metrics(root, weight):
    audit = read_json(root / "final_audit.json")
    assert audit["passed"] and audit["source_reports"] == audit["unique_response_ids"] == 824, root
    assert audit["all_model_forwards_cuda"] and audit["table6_excluded_study_ids"] == ["1615"], root
    digests = read_json(root / "output_checksums.json")
    for path, digest in digests.items():
        assert sha256(root / path) == digest, f"Modified metrics artifact: {root / path}"

    manifest = read_json(root / "input_manifest.json")
    assert manifest["source_cohort_count"] == 206 and manifest["clinical_study_count"] == 205, root
    assert manifest["excluded_empty_reference_study_ids"] == ["1615"], root
    assert set(manifest["branches"]) == set(BRANCHES), root
    for name, branch in manifest["branches"].items():
        config = branch["run_config"]
        assert branch["selected_reports"] == 206 and branch["unselected_reports"] == 0, (root, name)
        assert config["model"] == "gpt-5.6-luna" and config["reasoning_effort"] == "high", (root, name)
        assert config["max_output_tokens"] == 8192 and config["device"] == "cuda", (root, name)
        if name.startswith("structured_"):
            assert config["lambda_global"] == weight, (root, name, config["lambda_global"], weight)

    rows = read_csv(root / "report_provenance.csv")
    responses = {row["response_id"] for row in rows}
    assert len(rows) == len(responses) == 824 and "" not in responses, root
    refs = {
        (row["pipeline"], row["study_id"], row["request_id"]):
            (row["reference_problems"], row["reference_text_sha256"])
        for row in rows
    }
    assert len(refs) == len(rows), root
    return manifest, responses, refs


def compare(reference, run, reference_lambda, run_lambda):
    hardware = require_cuda()
    old, old_responses, old_references = load_audited_metrics(reference, reference_lambda)
    new, new_responses, new_references = load_audited_metrics(run, run_lambda)
    assert reference == run or old_responses.isdisjoint(new_responses), "Cross-run response IDs overlap"
    assert old_references == new_references, "Paired requests or reference reports differ"
    assert old["source_study_ids"] == new["source_study_ids"], "Source cohorts differ"
    assert old["metric_policy"] == new["metric_policy"], "Metric policies differ"
    for name in BRANCHES:
        old_branch, new_branch = old["branches"][name], new["branches"][name]
        assert old_branch["prompt_variants"] == new_branch["prompt_variants"], f"Prompt variants differ for {name}"
        for key in ("manifest_sha256", "compact_schema", "gate_threshold", "image_detail", "baseline"):
            assert old_branch["run_config"].get(key) == new_branch["run_config"].get(key), (name, key)

    ref_columns = ["Report Impression", *LABELS14]
    old_labels, new_labels = (
        pd.read_csv(path / "chexbert_raw/labeled_reports.csv", usecols=ref_columns, nrows=205)
        for path in (reference, run)
    )
    pd.testing.assert_frame_equal(old_labels, new_labels)
    checkpoints = {}
    for stage, keys in (("chexbert_raw", ("checkpoint_sha256",)), ("radgraph_raw", ("weights_sha256", "config_sha256"))):
        old_eval, new_eval = (read_json(path / stage / "inference_manifest.json") for path in (reference, run))
        assert all(old_eval[key] == new_eval[key] for key in keys), f"Evaluator inputs differ for {stage}"
        checkpoints[stage] = {key: new_eval[key] for key in keys}

    rows, index_hashes = [], {}
    clinical_ids = [sid for sid in new["source_study_ids"] if sid != "1615"]
    for table, metrics, ids in (("table5", T5, new["source_study_ids"]), ("table6", T6, clinical_ids)):
        old_result, new_result = (read_json(path / f"{table}_results.json") for path in (reference, run))
        assert old_result["study_ids"] == new_result["study_ids"] == ids, f"{table} study orders differ"
        old_indices, new_indices = (path / f"{table}_bootstrap_indices.npy" for path in (reference, run))
        index_array = np.load(new_indices)
        np.testing.assert_array_equal(index_array, np.load(old_indices))
        expected = np.random.default_rng(SEED).integers(0, len(ids), size=(SAMPLES, len(ids)))
        np.testing.assert_array_equal(index_array, expected)
        index_hashes[table] = sha256(new_indices)
        index = torch.as_tensor(index_array, dtype=torch.long, device="cuda:0")
        for name in BRANCHES:
            before, after = (scores(sufficient(path, table, name, ids), index, table) for path in (reference, run))
            deltas = after - before
            lower, upper = torch.quantile(deltas[1:], tensor([0.025, 0.975]), dim=0).tolist()
            for i, metric in enumerate(metrics):
                old_score, new_score = float(before[0, i]), float(after[0, i])
                assert abs(old_score - old_result["absolute"][name][metric]["estimate"]) < 1e-12
                assert abs(new_score - new_result["absolute"][name][metric]["estimate"]) < 1e-12
                rows.append(dict(
                    table=table, pipeline=name, metric=metric, study_count=len(ids),
                    reference_lambda=reference_lambda, run_lambda=run_lambda,
                    structured_rerank=name.startswith("structured_"), reference=old_score, run=new_score,
                    delta=float(deltas[0, i]), ci95_lower=lower[i], ci95_upper=upper[i],
                    ci_contains_zero=lower[i] <= 0 <= upper[i],
                ))
    protocol = dict(
        reference_metrics=str(reference), run_metrics=str(run), reference_lambda=reference_lambda, run_lambda=run_lambda,
        model="gpt-5.6-luna", reasoning_effort="high", max_output_tokens=8192, hardware=hardware,
        samples=SAMPLES, seed=SEED, method="Paired 95% percentile bootstrap by study ID", delta_direction="run minus reference",
        target_study_count=206, clinical_study_count=205, excluded_clinical_study_ids=["1615"],
        identical_reference_reports_and_chexbert_labels=True, evaluator_checkpoints=checkpoints,
        bootstrap_indices_sha256=index_hashes, disjoint_response_ids=old_responses.isdisjoint(new_responses),
        interpretation="The structured branches rerank at the specified λ and generate fresh reports. Baseline A has no retrieval. Baseline B uses global retrieval. Differences for A and B measure repeated generation with the same pipeline settings.",
    )
    return rows, protocol


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-metrics", required=True, type=Path)
    parser.add_argument("--run-metrics", required=True, type=Path)
    parser.add_argument("--reference-lambda", type=float, default=0.9)
    parser.add_argument("--run-lambda", type=float, default=0.6)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    reference, run, output = (path.resolve() for path in (args.reference_metrics, args.run_metrics, args.output_dir))
    if output.exists() or any(output.is_relative_to(path) for path in (reference, run)):
        parser.error("Use a new output directory outside both metrics directories.")
    rows, protocol = compare(reference, run, args.reference_lambda, args.run_lambda)
    output.mkdir(parents=True)
    save_csv(output / "lambda_paired_ci.csv", rows)
    save_json(output / "lambda_comparison_protocol.json", protocol)
    lines = [
        "# Micro F1 trên sáu nhãn mục tiêu",
        "",
        f"206 study. Δ = lượt λ {args.run_lambda:g} trừ lượt λ {args.reference_lambda:g}. "
        f"CI percentile 95% từ {SAMPLES:,} mẫu bootstrap ghép cặp theo study, seed {SEED}.",
        "",
        f"| Pipeline | Lượt λ {args.reference_lambda:g} | Lượt λ {args.run_lambda:g} | Δ [CI 95%] |",
        "|---|---:|---:|---:|",
    ]
    for row in rows:
        if row["metric"] == "micro_f1":
            lines.append(
                f"| {row['pipeline']} | {row['reference']:.4f} | {row['run']:.4f} | "
                f"{row['delta']:+.4f} [{row['ci95_lower']:+.4f}, {row['ci95_upper']:+.4f}] |"
            )
    lines += [
        "",
        "Pipeline Rule và Learned dùng λ khi xếp hạng lại. Baseline A chỉ dùng classifier và ảnh. "
        "Baseline B dùng truy xuất global. Chênh lệch A và B phản ánh các lượt sinh report riêng của cùng cấu hình.",
        "",
        "[48 kết quả ghép cặp](lambda_paired_ci.csv) gồm các chỉ số clinical trên 205 study. "
        "Study 1615 thiếu cả Findings và Impression tham chiếu.",
    ]
    (output / "LAMBDA_COMPARISON.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved {len(rows)} paired estimates to {output}")


if __name__ == "__main__":
    main()
