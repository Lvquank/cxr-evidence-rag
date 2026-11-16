"""Explore add-only FN rescue with predeclared incremental FP budgets."""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.benchmark_classifier_refinements import (
    SEED, bootstrap, constrained_threshold, counts, decide, design, fit, load_data,
    metrics, multilabel_folds, score, sha256, write_json,
)
from src.classifier_refinement import predict_refinement

METHODS=["threshold_tuned","score_fusion","densenet_probe","chexpert_rad_probe",
         "hybrid_probe","residual_densenet_probe","residual_score_fusion"]
BUDGETS=[0.,.0025,.005,.01]
PRIMARY=.005


def fit_rescue(method,parameter,x,y,baseline):
    train=~baseline if method.startswith("residual_") else np.ones(len(y),dtype=bool)
    return fit(method.removeprefix("residual_"),parameter,x[train],y[train])


def parameter_for(method,settings):
    if method == "residual_densenet_probe":
        return .01
    if method == "residual_score_fusion":
        return .1
    return settings[method]["parameter"]


def calibrate(y,baseline,scores,budget):
    region=~baseline
    fp_cap=int(np.floor(budget*(~y & region).sum()))
    threshold,quality=constrained_threshold(y[region],scores[region],fp_cap)
    return threshold,quality


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment",type=Path,required=True)
    args=parser.parse_args()
    out=args.experiment/"add_only_rescue"
    if out.exists():
        raise ValueError("Use a fresh experiment output")
    frame,ids,names,p,y,features,baseline,_,_=load_data()
    split=pd.read_csv(args.experiment/"splits.csv",dtype={"study_id":str}).set_index("study_id")
    dev=np.array([split.loc[sid,"partition"] == "development" for sid in ids])
    heldout=~dev
    train_ids=[sid for sid,keep in zip(ids,dev) if keep]
    folds=np.array([int(split.loc[sid,"outer_fold"]) for sid in train_ids])
    yy,bb=y[dev],baseline[dev]
    active=[j for j in range(len(names)) if (yy[:,j] & ~bb[:,j]).sum() >= 20]
    original=json.loads((args.experiment/"outer_choices.json").read_text())
    settings={(r["outer_fold"],r["label"]):r["settings"] for r in original}
    out.mkdir()
    write_json(out/"protocol.json",dict(method="add-only rescue; never remove baseline positives",
        phase="exploratory extension; 870 holdout already inspected in phase 1, not a fresh confirmatory set",
        method_selection_data="development nested OOF only",forbidden_selection_data="870 holdout and 206 historical outcomes",
        primary_fp_rate_budget=PRIMARY,sensitivity_budgets=BUDGETS,
        budget_denominator="gold-negative AND baseline-negative studies, per label, calibration folds only",
        active_labels=[names[j] for j in active],minimum_development_fn=20,methods=METHODS,
        hyperparameters="full-data methods reuse phase-1 inner-selected C; residual heads fixed C=.01/.1",
        final_selection="highest additional TP then lowest FP among development OOF arms within primary FP cap; no rescue fallback",
        code_sha256=sha256(Path(__file__)),source_protocol_sha256=sha256(args.experiment/"protocol.json")))
    rescues={(m,q):np.zeros_like(bb) for m in METHODS for q in BUDGETS}
    for q in BUDGETS:
        rescues[("inner_selector",q)]=np.zeros_like(bb)
    with threadpool_limits(limits=1):
        for fold in range(5):
            tr,va=folds != fold,folds == fold
            fold_ids=[sid for sid,keep in zip(train_ids,tr) if keep]
            mapping=multilabel_folds(frame[frame.study_id.isin(fold_ids)],fold_ids,3,SEED+20+fold,names)
            inner=np.array([mapping[sid] for sid in fold_ids])
            for j in active:
                quality={q:[] for q in BUDGETS}
                for method in METHODS:
                    base_method=method.removeprefix("residual_")
                    x=design(features,base_method,j)[dev]
                    parameter=parameter_for(method,settings[(fold,names[j])])
                    cal=np.full(tr.sum(),np.nan)
                    for k in range(3):
                        valid=inner == k
                        model=fit_rescue(method,parameter,x[tr][~valid],yy[tr,j][~valid],bb[tr,j][~valid])
                        cal[valid]=score(model,x[tr][valid])
                    model=fit_rescue(method,parameter,x[tr],yy[tr,j],bb[tr,j])
                    test_scores=score(model,x[va])
                    for q in BUDGETS:
                        threshold,(tp,fp)=calibrate(yy[tr,j],bb[tr,j],cal,q)
                        rescues[(method,q)][va,j]=~bb[va,j] & decide(test_scores,threshold)
                        quality[q].append((tp,-fp,method))
                for q in BUDGETS:
                    chosen=max(quality[q],key=lambda v:(v[0],v[1]))[2]
                    rescues[("inner_selector",q)][va,j]=rescues[(chosen,q)][va,j]
                print(f"rescue outer={fold+1}/5 label={names[j]}",flush=True)
    rows=[]
    for (method,q),rescue in rescues.items():
        pred=bb | rescue
        assert np.all(pred | ~bb)
        rows.append(dict(method=method,fp_rate_budget=q,additional_tp=int((yy & rescue).sum()),
            additional_fp=int((~yy & rescue).sum()),**metrics(yy,pred),
            per_label={name:dict(additional_tp=int((yy[:,j] & rescue[:,j]).sum()),
                                additional_fp=int((~yy[:,j] & rescue[:,j]).sum())) for j,name in enumerate(names)}))
    write_json(out/"development_comparison.json",rows)
    np.savez_compressed(out/"development_rescues.npz",**{f"{m}__{q}":a for (m,q),a in rescues.items()})
    selected={}
    combined=np.zeros_like(bb)
    for j in active:
        fp_cap=int(np.floor(PRIMARY*(~yy[:,j] & ~bb[:,j]).sum()))
        choices=[(0,0,"disabled")]
        for method in METHODS:
            r=rescues[(method,PRIMARY)][:,j]
            tp,fp=int((r & yy[:,j]).sum()),int((r & ~yy[:,j]).sum())
            if fp <= fp_cap:
                choices.append((tp,-fp,method))
        chosen=max(choices,key=lambda v:(v[0],v[1]))[2]
        selected[names[j]]=chosen
        if chosen != "disabled":
            combined[:,j]=rescues[(chosen,PRIMARY)][:,j]
    write_json(out/"winner_before_holdout.json",dict(method="add_only_rescue",families=selected,
        fp_rate_budget=PRIMARY,oof=metrics(yy,bb|combined),oof_baseline=metrics(yy,bb)))
    bundle=dict(schema_version=1,mode="add_only",labels=names,models={},
        baseline_thresholds=json.loads((ROOT/"config/densenet_adaptive_thresholds.json").read_text()),
        training_study_ids=train_ids,holdout_study_ids=[sid for sid,h in zip(ids,heldout) if h],
        feature_banks=json.loads((args.experiment/"protocol.json").read_text())["feature_banks"])
    fullmap=multilabel_folds(frame[frame.study_id.isin(train_ids)],train_ids,5,SEED+99,names)
    fullfolds=np.array([fullmap[sid] for sid in train_ids])
    final_settings={}
    with threadpool_limits(limits=1):
        for j in active:
            method=selected[names[j]]
            if method == "disabled":
                continue
            parameters=[parameter_for(method,settings[(fold,names[j])]) for fold in range(5)]
            parameter=Counter(parameters).most_common(1)[0][0]
            base_method=method.removeprefix("residual_")
            x=design(features,base_method,j)[dev]
            cal=np.full(len(yy),np.nan)
            for fold in range(5):
                valid=fullfolds == fold
                model=fit_rescue(method,parameter,x[~valid],yy[~valid,j],bb[~valid,j])
                cal[valid]=score(model,x[valid])
            threshold,quality=calibrate(yy[:,j],bb[:,j],cal,PRIMARY)
            model=fit_rescue(method,parameter,x,yy[:,j],bb[:,j])
            bundle["models"][names[j]]=dict(model=model,method=base_method,threshold=threshold,parameter=parameter)
            final_settings[names[j]]=dict(method=method,threshold=threshold,parameter=parameter,calibration_tp_fp=quality)
    joblib.dump(bundle,out/"candidate_classifier.joblib")
    write_json(out/"final_model_choices.json",final_settings)
    pred,_=predict_refinement(bundle,p,features)
    assert np.all(pred | ~baseline)
    result=dict(baseline=metrics(y[heldout],baseline[heldout]),winner=metrics(y[heldout],pred[heldout]),
        paired_bootstrap=bootstrap(y[heldout],pred[heldout],baseline[heldout]),
        per_label={name:dict(support=int(y[heldout,j].sum()),baseline=metrics(y[heldout,j],baseline[heldout,j]),
                            winner=metrics(y[heldout,j],pred[heldout,j])) for j,name in enumerate(names)})
    write_json(out/"holdout_evaluation.json",result)
    pd.DataFrame([dict(study_id=ids[i],label=name,gold=int(y[i,j]),baseline=int(baseline[i,j]),winner=int(pred[i,j]))
                  for i in np.flatnonzero(heldout) for j,name in enumerate(names)]).to_csv(out/"holdout_predictions.csv",index=False)
    write_json(out/"completion.json",dict(winner="add_only_rescue",candidate_sha256=sha256(out/"candidate_classifier.joblib"),
                holdout_evaluated_once_for_this_candidate=True,exploratory=True,holdout=result,llm_api_calls=0))
    print(json.dumps(dict(selected=selected,holdout=result),indent=2),flush=True)


if __name__ == "__main__":
    main()
