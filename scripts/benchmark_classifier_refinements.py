"""Compare FN/FP-constrained classifier refinements without LLM calls or test tuning."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.select_candidate_thresholds_oof import multilabel_folds, sha256

SEED = 20261015
FEATURE_BANKS = {
    "rad": "embeddings",
    "dense": "embeddings_densenet_raw",
    "chexpert_rad": "embeddings_chexpert_projection",
}
METHODS = ["threshold_tuned", "platt", "score_fusion", "densenet_probe",
           "rad_dino_probe", "chexpert_rad_probe", "hybrid_probe", "score_boost"]
GRID = {name: [0.001, 0.01, 0.1] for name in METHODS}
GRID.update(threshold_tuned=[None], platt=[1.0], score_fusion=[0.01, 0.1, 1.0], score_boost=[7, 15])


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def counts(y, predicted):
    return np.array([(y & predicted).sum(), (~y & predicted).sum(), (y & ~predicted).sum()], dtype=int)


def metrics(y, predicted):
    tp, fp, fn = map(int, counts(y, predicted))
    return dict(tp=tp, fp=fp, fn=fn, precision=tp/(tp+fp) if tp+fp else 0.,
                recall=tp/(tp+fn) if tp+fn else 0.,
                micro_f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.)


def decide(scores, threshold):
    return np.zeros(len(scores), dtype=bool) if threshold is None else scores >= threshold


def constrained_threshold(y, scores, max_fp):
    """Maximize TP within an integer FP budget; ties prefer fewer FP and higher t."""
    if len(y) == 0 or not np.isfinite(scores).all() or max_fp < 0:
        raise ValueError("Invalid calibration scores or FP budget")
    order = np.argsort(-scores, kind="stable")
    ordered = scores[order]
    ends = np.r_[np.flatnonzero(ordered[:-1] != ordered[1:]), len(y)-1]
    tp = np.cumsum(y[order])[ends]
    fp = ends+1-tp
    eligible = np.flatnonzero(fp <= max_fp)
    if not len(eligible) or tp[eligible].max() == 0:
        return None, (0, 0)
    best = max(eligible, key=lambda k: (int(tp[k]), -int(fp[k]), float(ordered[ends[k]])))
    return float(ordered[ends[best]]), (int(tp[best]), int(fp[best]))


def design(features, method, label_index):
    if method in {"threshold_tuned", "platt"}:
        return features["logits"][:, [label_index]]
    if method in {"score_fusion", "score_boost"}:
        return features["logits"]
    return features[{"densenet_probe": "dense", "rad_dino_probe": "rad",
                     "chexpert_rad_probe": "chexpert_rad", "hybrid_probe": "hybrid"}[method]]


def fit(method, parameter, x, y):
    if method == "threshold_tuned":
        return None
    if len(np.unique(y)) < 2:
        raise ValueError("Both classes are required for a learned refinement")
    if method == "score_boost":
        model = HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=int(parameter),
                    min_samples_leaf=30, l2_regularization=5., class_weight="balanced",
                    early_stopping=False, random_state=SEED)
    else:
        model = make_pipeline(StandardScaler(), LogisticRegression(C=parameter,
                    class_weight=None if method == "platt" else "balanced", solver="liblinear",
                    max_iter=1500, tol=1e-5, random_state=SEED))
    return model.fit(x, y.astype(int))


def score(model, x):
    if model is None:
        return x[:, 0]
    return model.predict_proba(x)[:, 1]


def tune(method, x, y, baseline, folds):
    max_fp = int((~y & baseline).sum())
    best = None
    for parameter in GRID[method]:
        oof = np.full(len(y), np.nan)
        for fold in sorted(set(folds)):
            valid = folds == fold
            model = fit(method, parameter, x[~valid], y[~valid])
            oof[valid] = score(model, x[valid])
        threshold, quality = constrained_threshold(y, oof, max_fp)
        key = (quality[0], -quality[1])
        if best is None or key > best[0]:
            best = (key, dict(parameter=parameter, threshold=threshold), decide(oof, threshold))
    return best[1], best[2]


def choose(y, baseline, candidates):
    """Choose using calibration predictions only, retaining the frozen baseline option."""
    base = counts(y, baseline)
    choices = [(int(base[0]), -int(base[1]), "baseline")]
    for name, pred in candidates.items():
        tp, fp, _ = counts(y, pred)
        if fp <= base[1] and tp >= base[0]:
            choices.append((int(tp), -int(fp), name))
    return max(choices, key=lambda v: (v[0], v[1]))[2]


def load_data():
    source = ROOT / "data/classifier_dev_oof_predictions.csv"
    provenance = json.loads(source.with_name(source.stem+"_manifest.json").read_text())
    if provenance["sha256"]["predictions"] != sha256(source):
        raise ValueError("Development prediction hash differs from provenance")
    frame = pd.read_csv(source, dtype={"study_id": str})
    names = json.loads((ROOT / "config/evaluation_ontology.json").read_text())["target_labels"]
    if frame.duplicated(["study_id", "pathology"]).any():
        raise ValueError("Duplicate study/label rows")
    ids = sorted(frame.study_id.unique(), key=int)
    if len(frame) != len(ids)*len(names):
        raise ValueError("Incomplete six-label development grid")
    benchmark = json.loads((ROOT / "data/evaluations/leave_one_out_206_manifest.json").read_text())
    if set(ids) & {str(r["study_id"]) for r in benchmark["requests"]}:
        raise ValueError("Development data overlap the frozen benchmark")
    p = frame.pivot(index="study_id", columns="pathology", values="raw_probability").loc[ids,names].to_numpy()
    y = frame.pivot(index="study_id", columns="pathology", values="gold").loc[ids,names].to_numpy(bool)
    features = {"logits": np.log(np.clip(p,1e-6,1-1e-6)/np.clip(1-p,1e-6,1-1e-6))}
    hashes = {str(source.relative_to(ROOT)): sha256(source)}
    for filename in ["data/classifier_dev_oof_predictions_manifest.json",
                     "config/densenet_adaptive_thresholds.json", "config/densenet_selective_iu_thresholds_v1.json"]:
        hashes[filename] = sha256(ROOT/filename)
    for key, directory in FEATURE_BANKS.items():
        bank = ROOT / "data" / directory
        image_ids = (bank/"ids.txt").read_text().splitlines()
        metadata = [json.loads(line) for line in (bank/"metadata.jsonl").read_text(encoding="utf-8").splitlines()]
        index = {v:i for i,v in enumerate(image_ids)}
        values = np.load(bank/"embeddings.npy")
        indices = [index[provenance["study_to_image_id"][sid]] for sid in ids]
        assert all(str(metadata[k]["study_id"]) == sid and metadata[k]["id"] == image_ids[k]
                   for k,sid in zip(indices,ids))
        features[key] = values[indices]
        if len(image_ids) != len(index) or values.shape[0] != len(image_ids):
            raise ValueError("Feature bank alignment failed")
        for filename in ["embeddings.npy", "ids.txt", "embedding_manifest.json"]:
            hashes[str((bank/filename).relative_to(ROOT))] = sha256(bank/filename)
    if len(np.unique(features["rad"],axis=0)) != len(ids):
        raise ValueError("Duplicate image feature rows require grouped splitting")
    features["hybrid"] = np.column_stack([features["dense"],features["rad"],features["logits"]])
    if any(not np.isfinite(value).all() for value in features.values()):
        raise ValueError("Nonfinite input features")
    thresholds = [json.loads((ROOT / path).read_text())["labels"] for path in
                  ["config/densenet_adaptive_thresholds.json", "config/densenet_selective_iu_thresholds_v1.json"]]
    baseline, selective = [p >= np.array([t[n]["threshold"] for n in names]) for t in thresholds]
    return frame, ids, names, p, y, features, baseline, selective, hashes


def bootstrap(y, prediction, baseline, seed=SEED, draws=2000):
    rng = np.random.default_rng(seed)
    delta = []
    for _ in range(draws):
        ix = rng.integers(0,len(y),len(y))
        a,b = metrics(y[ix],prediction[ix]), metrics(y[ix],baseline[ix])
        delta.append([a[k]-b[k] for k in ["tp","fp","fn","micro_f1"]])
    return {k:dict(delta=metrics(y,prediction)[k]-metrics(y,baseline)[k],
                   ci95=np.quantile(np.array(delta)[:,j],[.025,.975]).tolist())
            for j,k in enumerate(["tp","fp","fn","micro_f1"])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/classifier_fn_fp_benchmark_20261015")
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("Use a fresh output directory; previous experiments are preserved")
    frame, ids, names, p, y, features, baseline, selective, hashes = load_data()
    split = multilabel_folds(frame,ids,4,SEED,names)
    holdout = np.array([split[sid] == 3 for sid in ids])
    development = ~holdout
    train_ids = [sid for sid,keep in zip(ids,development) if keep]
    cvmap = multilabel_folds(frame[frame.study_id.isin(train_ids)],train_ids,5,SEED+1,names)
    outer = np.array([cvmap[sid] for sid in train_ids])
    yy, bb = y[development], baseline[development]
    active = [j for j in range(len(names)) if yy[:,j].sum() >= 20]
    args.output_dir.mkdir(parents=True)
    protocol = dict(seed=SEED,study_count=len(ids),development_studies=int(development.sum()),
        holdout_studies=int(holdout.sum()),outer_folds=5,inner_folds=3,
        labels=names,active_labels=[names[j] for j in active],methods=METHODS,parameter_grid=GRID,
        objective="maximize TP; then minimize FP; per-label FP no greater than baseline on tuning data",
        selector="per-label method/hyperparameter/threshold selected strictly inside each outer training fold",
        final_winner_rule="among outer-OOF methods with no per-label TP loss or FP increase, maximize pooled TP then minimize FP; baseline fallback",
        split_unit="study_id (patient IDs unavailable)",holdout_rule="not used for model/threshold/winner selection",
        missing_positive_policy="fewer than 20 development positives: preserve frozen DenseNet decision",
        methods_not_selectable=["selective_v1"],input_sha256=hashes,
        forbidden_inputs="benchmark 206 gold, report text, metadata mesh/problems/support and retrieval labels",
        feature_banks=FEATURE_BANKS,code_sha256=sha256(Path(__file__)))
    write_json(args.output_dir/"protocol.json",protocol)
    pd.DataFrame(dict(study_id=ids,partition=np.where(holdout,"locked_holdout","development"),
                      outer_fold=[None if h else cvmap[sid] for sid,h in zip(ids,holdout)])).to_csv(args.output_dir/"splits.csv",index=False)
    oof = {m:bb.copy() for m in ["baseline", "selective_v1", *METHODS,"labelwise_selector"]}
    oof["selective_v1"] = selective[development]
    choices = []
    print(json.dumps({"event":"protocol_frozen", "output":str(args.output_dir),
                      "development":int(development.sum()),"locked_holdout":int(holdout.sum()),
                      "active_labels":protocol["active_labels"]}),flush=True)
    with threadpool_limits(limits=1):
        for fold in range(5):
            train,valid = outer != fold,outer == fold
            fold_ids = [sid for sid,keep in zip(train_ids,train) if keep]
            inner_map = multilabel_folds(frame[frame.study_id.isin(fold_ids)],fold_ids,3,SEED+20+fold,names)
            inner = np.array([inner_map[sid] for sid in fold_ids])
            for j in active:
                tuned, inner_predictions = {}, {}
                for method in METHODS:
                    x = design(features,method,j)[development]
                    settings,pred = tune(method,x[train],yy[train,j],bb[train,j],inner)
                    model = fit(method,settings["parameter"],x[train],yy[train,j])
                    oof[method][valid,j] = decide(score(model,x[valid]),settings["threshold"])
                    tuned[method],inner_predictions[method] = settings,pred
                selected = choose(yy[train,j],bb[train,j],inner_predictions)
                oof["labelwise_selector"][valid,j] = bb[valid,j] if selected == "baseline" else oof[selected][valid,j]
                choices.append(dict(outer_fold=fold,label=names[j],selected_method=selected,settings=tuned))
                print(f"outer={fold+1}/5 label={names[j]} selected={selected}",flush=True)
    table = []
    for method,predicted in oof.items():
        per_label = {name:metrics(yy[:,j],predicted[:,j]) for j,name in enumerate(names)}
        eligible = method != "selective_v1" and all(counts(yy[:,j],predicted[:,j])[0] >= counts(yy[:,j],bb[:,j])[0]
                    and counts(yy[:,j],predicted[:,j])[1] <= counts(yy[:,j],bb[:,j])[1] for j in active)
        table.append(dict(method=method,eligible=eligible,**metrics(yy,predicted),per_label=per_label))
    eligible = [row for row in table if row["eligible"]]
    winner = max(eligible,key=lambda row:(row["tp"],-row["fp"]))["method"]
    write_json(args.output_dir/"development_comparison.json",table)
    write_json(args.output_dir/"outer_choices.json",choices)
    write_json(args.output_dir/"winner_before_holdout.json",dict(method=winner,selection_source="development outer-OOF only"))
    np.savez_compressed(args.output_dir/"development_oof_predictions.npz",gold=yy,**oof)
    print(f"Winner frozen before holdout: {winner}",flush=True)
    final_predictions = baseline.copy()
    fitted = {}
    final_choices = {}
    inner_map = multilabel_folds(frame[frame.study_id.isin(train_ids)],train_ids,5,SEED+99,names)
    folds = np.array([inner_map[sid] for sid in train_ids])
    with threadpool_limits(limits=1):
        for j in active:
            if winner == "baseline":
                continue
            methods = METHODS if winner == "labelwise_selector" else [winner]
            settings,preds = {},{}
            for method in methods:
                x=design(features,method,j)
                settings[method],preds[method]=tune(method,x[development],yy[:,j],bb[:,j],folds)
            selected=choose(yy[:,j],bb[:,j],preds) if winner == "labelwise_selector" else winner
            if selected == "baseline":
                final_choices[names[j]]={"method":"baseline"}
                continue
            setting=settings[selected]
            x=design(features,selected,j)
            model=fit(selected,setting["parameter"],x[development],yy[:,j])
            final_predictions[:,j]=decide(score(model,x),setting["threshold"])
            fitted[names[j]]=dict(model=model,method=selected,**setting)
            final_choices[names[j]]=dict(method=selected,**setting)
    holdout_result=dict(baseline=metrics(y[holdout],baseline[holdout]),winner=metrics(y[holdout],final_predictions[holdout]),
        paired_bootstrap=bootstrap(y[holdout],final_predictions[holdout],baseline[holdout]),
        per_label={name:dict(support=int(y[holdout,j].sum()),baseline=metrics(y[holdout,j],baseline[holdout,j]),
                            winner=metrics(y[holdout,j],final_predictions[holdout,j])) for j,name in enumerate(names)})
    holdout_result["all_active_labels_nonworse_point_counts"] = all(
        counts(y[holdout,j],final_predictions[holdout,j])[0] >= counts(y[holdout,j],baseline[holdout,j])[0]
        and counts(y[holdout,j],final_predictions[holdout,j])[1] <= counts(y[holdout,j],baseline[holdout,j])[1] for j in active)
    bundle=dict(schema_version=1,labels=names,models=fitted,feature_banks=FEATURE_BANKS,
                baseline_thresholds=json.loads((ROOT/"config/densenet_adaptive_thresholds.json").read_text()),
                training_study_ids=train_ids,holdout_study_ids=[sid for sid,h in zip(ids,holdout) if h],
                feature_order="hybrid: dense1024, raw_rad768, six DenseNet logits in labels order")
    joblib.dump(bundle,args.output_dir/"candidate_classifier.joblib")
    write_json(args.output_dir/"final_model_choices.json",final_choices)
    write_json(args.output_dir/"holdout_evaluation.json",holdout_result)
    records=[]
    for i in np.flatnonzero(holdout):
        for j,name in enumerate(names):
            records.append(dict(study_id=ids[i],label=name,gold=int(y[i,j]),
                                baseline=int(baseline[i,j]),winner=int(final_predictions[i,j])))
    pd.DataFrame(records).to_csv(args.output_dir/"holdout_predictions.csv",index=False)
    write_json(args.output_dir/"completion.json",dict(winner=winner,holdout_evaluated_once=True,
        candidate_sha256=sha256(args.output_dir/"candidate_classifier.joblib"),
        holdout=holdout_result,full_backbone_finetuning=False,llm_api_calls=0))
    print(json.dumps(dict(winner=winner,holdout=holdout_result),indent=2),flush=True)


if __name__ == "__main__":
    main()
