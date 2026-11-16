"""Fit one Edema probe on development OOF; preserve all other frozen decisions."""

import argparse
import copy
import json
import shutil
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.benchmark_classifier_refinements import load_data, metrics, bootstrap
from scripts.run_module_improvements import DEFAULT_ROOT, CANDIDATE, read, write, digest
from src.classifier_refinement import predict_refinement


def select_threshold(gold, scores):
    candidates = []
    for threshold in np.unique(scores):
        result = metrics(gold, scores >= threshold)
        if result["recall"] >= .85:
            candidates.append(((result["micro_f1"], result["precision"], threshold), threshold, result))
    if not candidates:
        raise ValueError("No development OOF threshold satisfies the recall floor")
    _, threshold, result = max(candidates, key=lambda candidate: candidate[0])
    return float(threshold), result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ROOT / "edema_probe")
    parser.add_argument("--method", choices=["probe", "densenet_threshold"], default="probe")
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("Use a fresh output directory")
    args.output_dir.mkdir(parents=True)
    shutil.copyfile(__file__, args.output_dir / "train_source.py")
    write(args.output_dir / "protocol.json", {
        "head": "Edema only; preserve Hybrid Live decisions for the other five labels",
        "model": "StandardScaler + LogisticRegression(C=.001,class_weight=balanced,max_iter=2000)" if args.method == "probe" else "Frozen DenseNet score; threshold calibration only",
        "threshold": "maximize development OOF F1 subject to recall>=.85; tie precision then higher threshold",
        "folds": "5 stratified folds by study; seed 20261018; patient IDs unavailable" if args.method == "probe" else "Existing frozen development predictions; no model refit",
        "status": "exploratory after historical test inspection; requires independent confirmation",
        "initial_candidate_sha256": digest(CANDIDATE), "code_sha256": digest(Path(__file__))})
    _, ids, names, raw, gold, _, _, _, _ = load_data()
    bundle = joblib.load(CANDIDATE)
    bank = Path(bundle["feature_banks"]["chexpert_rad"])
    indices = {image_id: index for index, image_id in enumerate((bank / "ids.txt").read_text().splitlines())}
    vectors = np.load(bank / "embeddings.npy")
    image_map = read(ROOT / "data/classifier_dev_oof_predictions_manifest.json")["study_to_image_id"]
    x = vectors[[indices[image_map[sid]] for sid in ids]]
    training_ids = set(bundle["training_study_ids"])
    development = np.array([sid in training_ids for sid in ids])
    heldout = ~development
    target = names.index("Edema")
    y = gold[development, target]
    if int(y.sum()) < 5:
        raise ValueError("Insufficient development positives for five folds")
    model = None
    if args.method == "probe":
        oof = np.zeros(len(y))
        folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=20261018)
        with threadpool_limits(limits=1):
            for train, valid in folds.split(x[development], y):
                model = make_pipeline(StandardScaler(), LogisticRegression(C=.001, class_weight="balanced", max_iter=2000))
                model.fit(x[development][train], y[train])
                oof[valid] = model.predict_proba(x[development][valid])[:, 1]
            threshold, selection = select_threshold(y, oof)
            model = make_pipeline(StandardScaler(), LogisticRegression(C=.001, class_weight="balanced", max_iter=2000))
            model.fit(x[development], y)
    else:
        threshold, selection = select_threshold(y, raw[development, target])
    updated = copy.deepcopy(bundle)
    updated["models"]["Edema"] = ({"method": "chexpert_rad_probe", "model": model,
        "threshold": threshold, "parameter": .001, "selection_source": "development OOF only"} if model is not None else
        {"method": "threshold_tuned", "model": None, "raw_threshold": threshold,
         "threshold": float(np.log(threshold / (1-threshold))), "selection_source": "frozen development predictions only"})
    updated["edema_protocol"] = read(args.output_dir / "protocol.json")
    joblib.dump(updated, args.output_dir / "candidate_classifier.joblib")
    base, _ = predict_refinement(bundle, raw, {"chexpert_rad": x})
    prediction, _ = predict_refinement(updated, raw, {"chexpert_rad": x})
    result = {"development_edema_selection": selection, "threshold": threshold,
        "development_positives": int(y.sum()), "holdout_count": int(heldout.sum()),
        "holdout_baseline_hybrid": metrics(gold[heldout], base[heldout]),
        "holdout_edema_candidate": metrics(gold[heldout], prediction[heldout]),
        "holdout_edema_only": {"baseline": metrics(gold[heldout, target], base[heldout, target]),
                               "candidate": metrics(gold[heldout, target], prediction[heldout, target])},
        "paired_bootstrap": bootstrap(gold[heldout], prediction[heldout], base[heldout])}
    requests = read(ROOT / "data/evaluations/leave_one_out_206_manifest.json")["requests"]
    old_by_study = {r["study_id"]: r for r in read(ROOT / "outputs/luna_high_evidence_v5_fresh_20260930/nlp_metrics_target6.json")["records"]}
    p = np.array([[read(ROOT / "outputs/luna_high_evidence_v5_fresh_20260930/evidence" / f"{item['request_id']}.json")["classifier"]["raw_probabilities"][name] for name in names] for item in requests])
    xx = vectors[[indices[item["image_id"]] for item in requests]]
    yy = np.array([[name in old_by_study[str(item["study_id"])] ["gold_target_problems"] for name in names] for item in requests])
    b, _ = predict_refinement(bundle, p, {"chexpert_rad": xx})
    n, _ = predict_refinement(updated, p, {"chexpert_rad": xx})
    result["historical_206_cached_features"] = {"baseline_hybrid": metrics(yy, b), "candidate": metrics(yy, n),
        "edema_baseline": metrics(yy[:, target], b[:, target]), "edema_candidate": metrics(yy[:, target], n[:, target]),
        "paired_bootstrap": bootstrap(yy, n, b)}
    write(args.output_dir / "evaluation.json", result)
    write(args.output_dir / "completion.json", {"candidate_sha256": digest(args.output_dir / "candidate_classifier.joblib"),
          "llm_api_calls": 0, "default_checkpoint_changed": False})
    print(json.dumps(result))


if __name__ == "__main__":
    main()
