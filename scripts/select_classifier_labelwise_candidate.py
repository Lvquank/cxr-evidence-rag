"""Execute the pre-registered labelwise selection using development OOF only."""

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
    METHODS, SEED, bootstrap, choose, design, fit, load_data, metrics, multilabel_folds,
    sha256, tune, write_json,
)
from src.classifier_refinement import predict_refinement


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment",type=Path,required=True)
    args=parser.parse_args()
    protocol=json.loads((args.experiment/"labelwise_ensemble_protocol.json").read_text())
    assert protocol["registered_before_primary_experiment_completion"]
    out=args.experiment/"labelwise_ensemble"
    if out.exists():
        raise ValueError("Candidate output already exists")
    frame,ids,names,p,y,features,baseline,_,_=load_data()
    split=pd.read_csv(args.experiment/"splits.csv",dtype={"study_id":str}).set_index("study_id")
    development=np.array([split.loc[sid,"partition"] == "development" for sid in ids])
    holdout=~development
    train_ids=[sid for sid,keep in zip(ids,development) if keep]
    oo=np.load(args.experiment/"development_oof_predictions.npz")
    assert np.array_equal(oo["gold"],y[development])
    families={}
    combined=oo["baseline"].copy()
    for j,name in enumerate(names):
        selected="baseline" if y[development,j].sum()<20 else choose(
            oo["gold"][:,j],oo["baseline"][:,j],{m:oo[m][:,j] for m in METHODS})
        families[name]=selected
        if selected != "baseline":
            combined[:,j]=oo[selected][:,j]
    out.mkdir()
    write_json(out/"winner_before_holdout.json",dict(method="oof_labelwise_ensemble",families=families,
        selection_source="development nested-OOF only; no locked-holdout or historical benchmark metrics read",
        oof=metrics(y[development],combined),oof_baseline=metrics(y[development],baseline[development])))
    mapping=multilabel_folds(frame[frame.study_id.isin(train_ids)],train_ids,5,SEED+99,names)
    folds=np.array([mapping[sid] for sid in train_ids])
    bundle=dict(schema_version=1,labels=names,models={},
        baseline_thresholds=json.loads((ROOT/"config/densenet_adaptive_thresholds.json").read_text()),
        training_study_ids=train_ids,holdout_study_ids=[sid for sid,h in zip(ids,holdout) if h],
        feature_banks=json.loads((args.experiment/"protocol.json").read_text())["feature_banks"])
    settings={}
    with threadpool_limits(limits=1):
        for j,name in enumerate(names):
            method=families[name]
            if method == "baseline":
                continue
            x=design(features,method,j)
            setting,_=tune(method,x[development],y[development,j],baseline[development,j],folds)
            model=fit(method,setting["parameter"],x[development],y[development,j])
            bundle["models"][name]=dict(model=model,method=method,**setting)
            settings[name]=dict(method=method,**setting)
            print(f"Fitted {name}: {method}",flush=True)
    write_json(out/"final_model_choices.json",settings)
    joblib.dump(bundle,out/"candidate_classifier.joblib")
    pred,_=predict_refinement(bundle,p,features)
    heldout=metrics(y[holdout],pred[holdout])
    result=dict(baseline=metrics(y[holdout],baseline[holdout]),winner=heldout,
        paired_bootstrap=bootstrap(y[holdout],pred[holdout],baseline[holdout]),
        per_label={name:dict(support=int(y[holdout,j].sum()),baseline=metrics(y[holdout,j],baseline[holdout,j]),
                            winner=metrics(y[holdout,j],pred[holdout,j])) for j,name in enumerate(names)})
    write_json(out/"holdout_evaluation.json",result)
    pd.DataFrame([dict(study_id=ids[i],label=name,gold=int(y[i,j]),baseline=int(baseline[i,j]),winner=int(pred[i,j]))
                  for i in np.flatnonzero(holdout) for j,name in enumerate(names)]).to_csv(out/"holdout_predictions.csv",index=False)
    write_json(out/"completion.json",dict(winner="oof_labelwise_ensemble",holdout_evaluated_once=True,
        candidate_sha256=sha256(out/"candidate_classifier.joblib"),holdout=result,llm_api_calls=0))
    print(json.dumps(dict(families=families,holdout=result),indent=2))


if __name__ == "__main__":
    main()
