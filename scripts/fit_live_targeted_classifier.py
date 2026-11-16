"""Refit the targeted classifier on features matching new-image inference."""

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.benchmark_classifier_refinements import (
    SEED, bootstrap, fit, load_data, metrics, multilabel_folds, sha256, tune, write_json,
)
from src.classifier_refinement import predict_refinement


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment",type=Path,required=True)
    args=parser.parse_args()
    bank=(args.experiment/"live_feature_bank").resolve()
    out=args.experiment/"targeted_hybrid_live"
    if out.exists():
        raise ValueError("Output exists")
    completed=json.loads((bank/"completed.json").read_text())
    if completed["embeddings_sha256"] != sha256(bank/"embeddings.npy"):
        raise ValueError("Feature bank hash mismatch")
    manifest=json.loads((bank/"embedding_manifest.json").read_text())
    frame,ids,names,p,y,_,base,_,_=load_data()
    image_map=json.loads((ROOT/"data/classifier_dev_oof_predictions_manifest.json").read_text())["study_to_image_id"]
    image_ids=(bank/"ids.txt").read_text().splitlines()
    index={v:i for i,v in enumerate(image_ids)}
    features={"chexpert_rad":np.load(bank/"embeddings.npy")[[index[image_map[sid]] for sid in ids]]}
    split=pd.read_csv(args.experiment/"splits.csv",dtype={"study_id":str}).set_index("study_id")
    dev=np.array([split.loc[sid,"partition"] == "development" for sid in ids])
    heldout=~dev
    train_ids=[sid for sid,k in zip(ids,dev) if k]
    folds_map=multilabel_folds(frame[frame.study_id.isin(train_ids)],train_ids,5,SEED+99,names)
    folds=np.array([folds_map[sid] for sid in train_ids])
    out.mkdir()
    write_json(out/"protocol.json",dict(phase="exploratory refit to remove feature-cache/new-image mismatch",
        family="Atelectasis DenseNet threshold .300; Pneumonia CheXpert-adapted RAD-DINO linear probe; other four baseline",
        parameters_and_thresholds="5-fold development OOF only; same FP-constrained rule as phase 1",
        feature_bank_sha256=completed["embeddings_sha256"],code_sha256=sha256(Path(__file__)),
        validation_status="870 already inspected during research; requires new independent confirmation"))
    j=names.index("Pneumonia")
    x=features["chexpert_rad"]
    with threadpool_limits(limits=1):
        setting,oof=tune("chexpert_rad_probe",x[dev],y[dev,j],base[dev,j],folds)
        model=fit("chexpert_rad_probe",setting["parameter"],x[dev],y[dev,j])
    models={"Pneumonia":dict(model=model,method="chexpert_rad_probe",**setting),
            "Atelectasis":dict(model=None,method="threshold_tuned",threshold=float(np.log(.3/.7)),raw_threshold=.3)}
    bundle=dict(schema_version=1,labels=names,models=models,feature_banks={"chexpert_rad":str(bank)},
        image_feature_manifest=manifest,training_study_ids=train_ids,
        densenet_checkpoint_sha256=json.loads((ROOT/"data/classifier_dev_oof_predictions_manifest.json").read_text())["sha256"]["checkpoint"],
        holdout_study_ids=[sid for sid,k in zip(ids,heldout) if k],
        baseline_thresholds=json.loads((ROOT/"config/densenet_adaptive_thresholds.json").read_text()))
    write_json(out/"winner_before_holdout.json",dict(method="targeted_hybrid_live",settings=setting,
        selection_source="architecture fixed before refit; C and threshold fitted using development OOF only",
        pneumonia_development_oof=metrics(y[dev,j],oof)))
    joblib.dump(bundle,out/"candidate_classifier.joblib")
    write_json(out/"final_model_choices.json",{n:{k:v for k,v in entry.items() if k != "model"} for n,entry in models.items()})
    pred,_=predict_refinement(bundle,p,features)
    result=dict(baseline=metrics(y[heldout],base[heldout]),winner=metrics(y[heldout],pred[heldout]),
        paired_bootstrap=bootstrap(y[heldout],pred[heldout],base[heldout]),
        per_label={name:dict(support=int(y[heldout,k].sum()),baseline=metrics(y[heldout,k],base[heldout,k]),
                            winner=metrics(y[heldout,k],pred[heldout,k])) for k,name in enumerate(names)})
    write_json(out/"holdout_evaluation.json",result)
    pd.DataFrame([dict(study_id=ids[i],label=name,gold=int(y[i,k]),baseline=int(base[i,k]),winner=int(pred[i,k]))
                  for i in np.flatnonzero(heldout) for k,name in enumerate(names)]).to_csv(out/"holdout_predictions.csv",index=False)
    write_json(out/"completion.json",dict(winner="targeted_hybrid_live",exploratory=True,
        candidate_sha256=sha256(out/"candidate_classifier.joblib"),holdout=result,llm_api_calls=0))
    print(json.dumps(dict(settings=setting,holdout=result),indent=2))


if __name__ == "__main__":
    main()
