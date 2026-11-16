"""Diagnose a frozen candidate on the historical 206 cohort, without selecting it."""

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.benchmark_classifier_refinements import FEATURE_BANKS, bootstrap, metrics, sha256, write_json
from scripts.evaluate_llm_outputs import load_reports, load_ontology, normalize_target_problems
from src.classifier_refinement import predict_refinement


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment",type=Path,required=True)
    args=parser.parse_args()
    completion=json.loads((args.experiment/"completion.json").read_text())
    frozen=json.loads((args.experiment/"winner_before_holdout.json").read_text())
    candidate=args.experiment/"candidate_classifier.joblib"
    if completion["candidate_sha256"] != sha256(candidate) or frozen["method"] != completion["winner"]:
        raise ValueError("Candidate identity differs from frozen selection")
    output=args.experiment/"historical_206_diagnostic"
    if output.exists():
        raise ValueError("Historical cohort is evaluated only once per frozen candidate")
    bundle=joblib.load(candidate)
    requests=json.loads((ROOT/"data/evaluations/leave_one_out_206_manifest.json").read_text())["requests"]
    ids=[str(r["study_id"]) for r in requests]
    assert not set(ids) & (set(bundle["training_study_ids"]) | set(bundle["holdout_study_ids"]))
    names=bundle["labels"]
    ontology=load_ontology()
    reports=load_reports(ROOT/"src/data/indiana_reports.csv")
    gold=[set(normalize_target_problems(reports[sid]["problems"],ontology)["target_problems"]) for sid in ids]
    y=np.array([[name in g for name in names] for g in gold])
    probabilities=[]
    for item in requests:
        evidence=json.loads((ROOT/"outputs/luna_high_evidence_v5_206/evidence"/f"{item['request_id']}.json").read_text())
        assert str(evidence["retriever"]["excluded_study_ids"][0]) == str(item["study_id"])
        probabilities.append([evidence["classifier"]["raw_probabilities"][name] for name in names])
    p=np.array(probabilities)
    features={}
    for key,directory in bundle.get("feature_banks",FEATURE_BANKS).items():
        bank=Path(directory)
        if not bank.is_absolute():
            bank=ROOT/"data"/directory
        image_ids=(bank/"ids.txt").read_text().splitlines()
        index={v:i for i,v in enumerate(image_ids)}
        values=np.load(bank/"embeddings.npy")
        features[key]=values[[index[item["image_id"]] for item in requests]]
    pred,details=predict_refinement(bundle,p,features)
    base=p >= np.array([bundle["baseline_thresholds"]["labels"][name]["threshold"] for name in names])
    result=dict(status="historical diagnostic only; not used to select the method or tune thresholds",
                studies=len(ids),winner=frozen["method"],baseline=metrics(y,base),candidate=metrics(y,pred),
                paired_bootstrap=bootstrap(y,pred,base),
                per_label={name:dict(baseline=metrics(y[:,j],base[:,j]),candidate=metrics(y[:,j],pred[:,j]))
                           for j,name in enumerate(names)})
    output.mkdir()
    write_json(output/"metrics.json",result)
    write_json(output/"score_provenance.json",details)
    pd.DataFrame([dict(study_id=sid,label=name,gold=int(y[i,j]),baseline=int(base[i,j]),candidate=int(pred[i,j]))
                  for i,sid in enumerate(ids) for j,name in enumerate(names)]).to_csv(output/"predictions.csv",index=False)
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
