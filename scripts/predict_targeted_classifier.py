"""Run the opt-in targeted hybrid on a new chest image, without retrieval or LLM."""

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE","1")
os.environ.setdefault("TRANSFORMERS_OFFLINE","1")
import joblib
import numpy as np
import torch
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.predict_chexpert_6labels import DEFAULT_CHECKPOINT, load_model, predict
from scripts.select_candidate_thresholds_oof import sha256
from src.classifier_refinement import predict_refinement
from src.embeddings.encoder import load_encoder, load_image_processor


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-path",type=Path,required=True)
    parser.add_argument("--candidate",type=Path,default=ROOT/"outputs/classifier_fn_fp_benchmark_20261015/targeted_hybrid_live/candidate_classifier.joblib")
    parser.add_argument("--output-json",type=Path,required=True)
    args=parser.parse_args()
    bundle=joblib.load(args.candidate)
    manifest=bundle["image_feature_manifest"]
    checkpoint=Path(manifest["checkpoint_path"])
    if sha256(checkpoint) != manifest["checkpoint_sha256"]:
        raise ValueError("Encoder checkpoint differs from training features")
    if sha256(DEFAULT_CHECKPOINT) != bundle["densenet_checkpoint_sha256"]:
        raise ValueError("DenseNet checkpoint differs from development scoring")
    device=torch.device("cuda")
    classifier,_=load_model(DEFAULT_CHECKPOINT,device)
    raw=predict(args.image_path,classifier,device)
    del classifier
    torch.cuda.empty_cache()
    encoder=load_encoder(device=device,projection_checkpoint=checkpoint)
    processor=load_image_processor(encoder.model_name)
    if processor.to_dict() != manifest["processor"]:
        raise ValueError("Image processor differs from training features")
    with Image.open(args.image_path) as im:
        pixels=processor(images=im.convert("RGB"),return_tensors="pt")["pixel_values"].to(device)
    with torch.inference_mode(),torch.autocast(device_type="cuda",dtype=torch.float16):
        value=encoder(pixels)
    features=torch.nn.functional.normalize(value.float(),dim=1).cpu().numpy()
    predicted,details=predict_refinement(bundle,np.array([[raw[n] for n in bundle["labels"]]]),{"chexpert_rad":features})
    result=dict(version="targeted-hybrid-live-exploratory-v1",image_path=str(args.image_path.resolve()),
        raw_densenet_probabilities=raw,selected_labels=[n for n,k in zip(bundle["labels"],predicted[0]) if k],
        decision_sources=details,candidate_sha256=sha256(args.candidate),
        feature_bank_sha256=sha256(Path(bundle["feature_banks"]["chexpert_rad"])/"embeddings.npy"))
    args.output_json.parent.mkdir(parents=True,exist_ok=True)
    args.output_json.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
