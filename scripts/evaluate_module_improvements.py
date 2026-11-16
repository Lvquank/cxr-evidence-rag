"""Evaluate classifier and gallery changes offline, with paired study bootstrap."""

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.benchmark_classifier_refinements import load_data, metrics, bootstrap
from scripts.evaluate_hybrid_cxr_retrieval import weighted_jaccard, normal_relevance, dcg
from scripts.run_module_improvements import DEFAULT_ROOT, CANDIDATE, read, write, LABELS, freeze
from src.classifier_refinement import refinement_scores


def paired(values, baseline):
    difference = np.asarray(values) - np.asarray(baseline)
    rng = np.random.default_rng(20261018)
    draws = np.array([difference[rng.integers(0, len(difference), len(difference))].mean() for _ in range(2000)])
    return {"delta": float(difference.mean()), "ci95": np.quantile(draws, [.025, .975]).tolist(),
            "unit": "study", "draws": 2000}


def classifier_inputs():
    frame, ids, names, raw, gold, _, baseline, _, _ = load_data()
    bundle = joblib.load(CANDIDATE)
    bank = Path(bundle["feature_banks"]["chexpert_rad"])
    index = {image_id: i for i, image_id in enumerate((bank / "ids.txt").read_text().splitlines())}
    image_map = read(ROOT / "data/classifier_dev_oof_predictions_manifest.json")["study_to_image_id"]
    features = np.load(bank / "embeddings.npy")[[index[image_map[sid]] for sid in ids]]
    candidate, _, _, _ = refinement_scores(bundle, raw, {"chexpert_rad": features})
    return ids, names, gold, baseline, candidate


def evaluate_classifier(root, requests):
    ids, names, gold, baseline, candidate = classifier_inputs()
    image_map = read(ROOT / "data/classifier_dev_oof_predictions_manifest.json")["study_to_image_id"]
    split = pd.read_csv(ROOT / "outputs/classifier_fn_fp_benchmark_20261015/splits.csv", dtype={"study_id": str}).set_index("study_id")
    holdout = np.array([split.loc[sid, "partition"] != "development" for sid in ids])
    report = {"870_historical_holdout": {"baseline": metrics(gold[holdout], baseline[holdout]),
        "candidate": metrics(gold[holdout], candidate[holdout]),
        "paired_bootstrap": bootstrap(gold[holdout], candidate[holdout], baseline[holdout]),
        "note": "Previously inspected cohort; this rerun verifies integration, not new independent validation."}}
    from scripts.evaluate_llm_outputs import load_reports, load_ontology, normalize_target_problems
    reports = load_reports(ROOT / "src/data/indiana_reports.csv")
    ontology = load_ontology()
    truth, controls, candidates = [], [], []
    for item in requests:
        sid = str(item["study_id"])
        labels = normalize_target_problems(reports[sid]["problems"], ontology)["target_problems"]
        truth.append([n in labels for n in names])
        for arm, target in (("control", controls), ("hybrid_consistent", candidates)):
            selected = read(root / arm / "evidence" / f"{item['request_id']}.json")["classifier"]["selected_labels"]
            target.append([n in selected for n in names])
    y, old, new = map(np.asarray, (truth, controls, candidates))
    report["206_fresh_inference"] = {"baseline": metrics(y, old), "candidate": metrics(y, new),
        "paired_bootstrap": bootstrap(y, new, old), "per_label": {
            n: {"baseline": metrics(y[:, j], old[:, j]), "candidate": metrics(y[:, j], new[:, j])}
            for j, n in enumerate(names)}}
    positives = {n: int(gold[~holdout, j].sum()) for j, n in enumerate(names)}
    report["development_positive_counts"] = positives
    review = []
    for i, sid in enumerate(ids):
        if holdout[i]:
            continue
        for label in ("Edema", "Pneumonia", "Pneumothorax"):
            j = names.index(label)
            if gold[i, j] or baseline[i, j] != gold[i, j] or candidate[i, j] != gold[i, j]:
                review.append({"study_id": sid, "image_id": image_map[sid], "label": label,
                    "metadata_gold": int(gold[i, j]), "baseline": int(baseline[i, j]),
                    "candidate": int(candidate[i, j]), "needs_expert_review": True})
    pd.DataFrame(review).to_csv(root / "data_review_development.csv", index=False)
    write(root / "classifier_evaluation.json", report)
    return report


def evaluate_retrieval(root, requests, query_labels=None):
    legacy = ROOT / "data/embeddings_chexpert_projection"
    consistent = root / "consistent_gallery"
    old_rows = [json.loads(line) for line in (legacy / "metadata.jsonl").read_text(encoding="utf-8").splitlines()]
    new_rows = [json.loads(line) for line in (consistent / "metadata.jsonl").read_text(encoding="utf-8").splitlines()]
    new_index = {r["id"]: i for i, r in enumerate(new_rows)}
    ordered = [new_index[r["id"]] for r in old_rows]
    old_vectors = np.load(legacy / "embeddings.npy").astype(np.float32)
    new_vectors = np.load(consistent / "embeddings.npy")[ordered].astype(np.float32)
    old_vectors /= np.linalg.norm(old_vectors, axis=1, keepdims=True)
    new_vectors /= np.linalg.norm(new_vectors, axis=1, keepdims=True)
    studies = np.array([str(r["study_id"]) for r in old_rows])
    image_index = {r["id"]: i for i, r in enumerate(old_rows)}
    support = np.asarray([r["support"] for r in old_rows], dtype=np.float32)
    image_map = read(ROOT / "data/classifier_dev_oof_predictions_manifest.json")["study_to_image_id"]
    split = pd.read_csv(ROOT / "outputs/classifier_fn_fp_benchmark_20261015/splits.csv", dtype={"study_id": str})
    cohorts = {partition: [image_map[sid] for sid in split.loc[split.partition == partition, "study_id"]]
               for partition in ("development", "locked_holdout")}
    cohorts["historical_206"] = [item["image_id"] for item in requests]
    output, per_query = {}, []
    for cohort, image_ids in cohorts.items():
        variants = ["legacy_gallery", "consistent_gallery"] + (["consistent_guided"] if query_labels is not None else [])
        results = {name: [] for name in variants}
        with threadpool_limits(limits=1):
            for image_id in image_ids:
                i = image_index[image_id]
                relevance = normal_relevance(support) if not support[i].sum() else weighted_jaccard(support[i], support)
                valid = studies != studies[i]
                ideal_by_study = {}
                for candidate in np.flatnonzero(valid):
                    sid = studies[candidate]
                    ideal_by_study[sid] = max(ideal_by_study.get(sid, 0), float(relevance[candidate]))
                ideal = dcg(np.array(sorted(ideal_by_study.values(), reverse=True)[:5]))
                for name in variants:
                    vectors = old_vectors if name == "legacy_gallery" else new_vectors
                    similarity = vectors @ new_vectors[i]
                    if name == "consistent_guided":
                        selected = np.array(query_labels[studies[i]], dtype=np.float32)
                        if selected.any():
                            affinity = weighted_jaccard(selected, support)
                            similarity = .9 * similarity + .1 * affinity
                    ranked, seen = [], set()
                    for candidate in np.argsort(-similarity, kind="stable"):
                        sid = studies[candidate]
                        if not valid[candidate] or sid in seen:
                            continue
                        seen.add(sid)
                        ranked.append(candidate)
                        if len(ranked) == 5:
                            break
                    values = relevance[ranked]
                    row = {"cohort": cohort, "study_id": studies[i], "variant": name,
                           "P@5": float((values >= .5).mean()),
                           "nDCG@5": dcg(values) / ideal if ideal else 0.0}
                    results[name].append(row)
                    per_query.append(row)
        output[cohort] = {name: {metric: float(np.mean([r[metric] for r in rows]))
                                 for metric in ("P@5", "nDCG@5")} for name, rows in results.items()}
        output[cohort]["n_studies"] = len(image_ids)
        output[cohort]["paired_difference"] = {metric: paired(
            [r[metric] for r in results["consistent_gallery"]], [r[metric] for r in results["legacy_gallery"]])
            for metric in ("P@5", "nDCG@5")}
        if query_labels is not None:
            output[cohort]["guided_minus_consistent"] = {metric: paired(
                [r[metric] for r in results["consistent_guided"]], [r[metric] for r in results["consistent_gallery"]])
                for metric in ("P@5", "nDCG@5")}
    output["protocol"] = "Same freshly encoded query for both galleries; global cosine only; query study removed; unique study rankings; metadata relevance frozen."
    output["mean_old_new_same_image_cosine"] = float(np.mean(np.sum(old_vectors * new_vectors, axis=1)))
    prefix = "guided_" if query_labels is not None else ""
    pd.DataFrame(per_query).to_csv(root / f"{prefix}retrieval_per_query.csv", index=False)
    write(root / f"{prefix}retrieval_evaluation.json", output)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--retrieval-only", action="store_true")
    parser.add_argument("--guided-rerank", action="store_true")
    args = parser.parse_args()
    requests = freeze(args.output_root)
    query_labels = None
    if args.guided_rerank:
        write(args.output_root / "guided_reranker_protocol.json", {
            "score": "0.9*cosine + 0.1*weighted_jaccard(predicted_selected_labels, reference_support)",
            "no_selected_labels": "retain global cosine ranking",
            "weights": "fixed before evaluating this branch; no parameter sweep",
            "status": "exploratory follow-up after inspecting gallery results; not an independent test",
            "query_gold_used_for_ranking": False})
        ids, names, _, _, predicted = classifier_inputs()
        query_labels = dict(zip(ids, predicted.tolist()))
        for item in requests:
            evidence = read(args.output_root / "hybrid_consistent/evidence" / f"{item['request_id']}.json")
            selected = evidence["classifier"]["selected_labels"]
            query_labels[str(item["study_id"])] = [label in selected for label in names]
    retrieval = evaluate_retrieval(args.output_root, requests, query_labels)
    if not args.retrieval_only:
        evaluate_classifier(args.output_root, requests)
    print(json.dumps({"output": str(args.output_root), "retrieval_206": retrieval["historical_206"]}))


if __name__ == "__main__":
    main()
