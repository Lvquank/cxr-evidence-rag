"""Explore a two-label hybrid, selecting its composition from development OOF."""

import argparse
import itertools
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
    METHODS, SEED, bootstrap, counts, design, fit, load_data, metrics,
    multilabel_folds, sha256, tune, write_json,
)
from src.classifier_refinement import predict_refinement


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment",type=Path,required=True)
    args=parser.parse_args()
    out=args.experiment/"targeted_hybrid"
    if out.exists():
        raise ValueError("Candidate output exists")
    frame,ids,names,p,y,features,baseline,_,_=load_data()
    split=pd.read_csv(args.experiment/"splits.csv",dtype={"study_id":str}).set_index("study_id")
    dev=np.array([split.loc[sid,"partition"] == "development" for sid in ids])
    heldout=~dev
    train_ids=[sid for sid,k in zip(ids,dev) if k]
    targets=[names.index(n) for n in ["Atelectasis","Pneumonia"]]
    methods=["baseline","selective_v1",*METHODS]
    oo=np.load(args.experiment/"development_oof_predictions.npz")
    assert np.array_equal(oo["gold"],y[dev])
    out.mkdir()
    write_json(out/"protocol.json",dict(phase="exploratory after initial holdout inspection; new independent confirmation is required",
        targets=[names[j] for j in targets],reason="restrict changes to the two labels changed by the existing selective-v1 artifact",
        selection_source="development nested-OOF predictions only",methods=methods,
        objective="maximize total TP, then minimize total FP; FP <= frozen baseline overall; TP cannot decrease for either target on development OOF",
        other_labels="preserve frozen baseline decisions",source_code_sha256=sha256(Path(__file__))))
    choices=[]
    reference=counts(y[dev],baseline[dev])
    for pair in itertools.product(methods,repeat=2):
        pred=baseline[dev].copy()
        admissible=True
        for j,method in zip(targets,pair):
            pred[:,j]=oo[method][:,j]
            if counts(y[dev,j],pred[:,j])[0] < counts(y[dev,j],baseline[dev,j])[0]:
                admissible=False
        tp,fp,fn=counts(y[dev],pred)
        choices.append(dict(atelectasis=pair[0],pneumonia=pair[1],eligible=bool(admissible and fp <= reference[1]),
                            tp=int(tp),fp=int(fp),fn=int(fn)))
    best=max([c for c in choices if c["eligible"]],key=lambda c:(c["tp"],-c["fp"]))
    families=dict(zip([names[j] for j in targets],[best["atelectasis"],best["pneumonia"]]))
    write_json(out/"development_comparison.json",choices)
    write_json(out/"winner_before_holdout.json",dict(method="targeted_hybrid",families=families,
        oof=best,oof_baseline=metrics(y[dev],baseline[dev])))
    mapping=multilabel_folds(frame[frame.study_id.isin(train_ids)],train_ids,5,SEED+99,names)
    folds=np.array([mapping[sid] for sid in train_ids])
    bundle=dict(schema_version=1,labels=names,models={},training_study_ids=train_ids,
        holdout_study_ids=[sid for sid,k in zip(ids,heldout) if k],
        baseline_thresholds=json.loads((ROOT/"config/densenet_adaptive_thresholds.json").read_text()),
        feature_banks=json.loads((args.experiment/"protocol.json").read_text())["feature_banks"])
    selective=json.loads((ROOT/"config/densenet_selective_iu_thresholds_v1.json").read_text())
    with threadpool_limits(limits=1):
        for j in targets:
            name=names[j]
            method=families[name]
            if method == "baseline":
                continue
            if method == "selective_v1":
                t=selective["labels"][name]["threshold"]
                bundle["models"][name]=dict(model=None,method="threshold_tuned",threshold=float(np.log(t/(1-t))),
                                            raw_threshold=t,threshold_source="frozen selective v1")
            else:
                x=design(features,method,j)
                setting,_=tune(method,x[dev],y[dev,j],baseline[dev,j],folds)
                model=fit(method,setting["parameter"],x[dev],y[dev,j])
                bundle["models"][name]=dict(model=model,method=method,**setting)
    joblib.dump(bundle,out/"candidate_classifier.joblib")
    write_json(out/"final_model_choices.json",{n:{k:v for k,v in entry.items() if k != "model"}
                                               for n,entry in bundle["models"].items()})
    pred,_=predict_refinement(bundle,p,features)
    result=dict(baseline=metrics(y[heldout],baseline[heldout]),winner=metrics(y[heldout],pred[heldout]),
        paired_bootstrap=bootstrap(y[heldout],pred[heldout],baseline[heldout]),
        per_label={name:dict(support=int(y[heldout,j].sum()),baseline=metrics(y[heldout,j],baseline[heldout,j]),
                            winner=metrics(y[heldout,j],pred[heldout,j])) for j,name in enumerate(names)})
    write_json(out/"holdout_evaluation.json",result)
    pd.DataFrame([dict(study_id=ids[i],label=name,gold=int(y[i,j]),baseline=int(baseline[i,j]),winner=int(pred[i,j]))
                  for i in np.flatnonzero(heldout) for j,name in enumerate(names)]).to_csv(out/"holdout_predictions.csv",index=False)
    write_json(out/"completion.json",dict(winner="targeted_hybrid",candidate_sha256=sha256(out/"candidate_classifier.joblib"),
        exploratory=True,holdout=result,llm_api_calls=0))
    print(json.dumps(dict(families=families,holdout=result),indent=2),flush=True)


if __name__ == "__main__":
    main()
