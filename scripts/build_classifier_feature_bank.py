"""Build constant-size RAD-DINO features matching standalone classifier inference."""

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE","1")
os.environ.setdefault("TRANSFORMERS_OFFLINE","1")
import numpy as np
import torch
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.select_candidate_thresholds_oof import sha256
from src.embeddings.encoder import load_encoder, load_image_processor


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--batch-size",type=int,default=8)
    args=parser.parse_args()
    if args.output_dir.exists() or args.batch_size<1:
        parser.error("Use a fresh output directory and a positive batch size")
    dev=json.loads((ROOT/"data/classifier_dev_oof_predictions_manifest.json").read_text())
    benchmark=json.loads((ROOT/"data/evaluations/leave_one_out_206_manifest.json").read_text())["requests"]
    wanted=set(dev["study_to_image_id"].values()) | {r["image_id"] for r in benchmark}
    source=[json.loads(line) for line in (ROOT/"data/embeddings_chexpert_projection/metadata.jsonl").read_text().splitlines()]
    rows=[{key:r[key] for key in ["id","study_id","path"]} for r in source if r["id"] in wanted]
    if len(rows)!=len(wanted):
        raise ValueError("Requested images missing from feature metadata")
    checkpoint=ROOT/"model/rad_dino_chexpert_projection_best.pt"
    model=load_encoder(device="cuda",projection_checkpoint=checkpoint)
    processor=load_image_processor(model.model_name)
    args.output_dir.mkdir(parents=True)
    manifest=dict(embedding_dimension=512,embedding_model=model.model_name,
        model_revision=model.model_revision,checkpoint_sha256=sha256(checkpoint),
        checkpoint_path=str(checkpoint),processor=processor.to_dict(),
        preprocessing="Hugging Face processor defaults: shortest edge 518, center crop 518x518; no variable padding",
        cuda_autocast_dtype="float16",output_normalization="float32 L2 after encoder",
        batch_size=args.batch_size,images=len(rows),code_sha256=sha256(Path(__file__)))
    (args.output_dir/"embedding_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    vectors=np.lib.format.open_memmap(args.output_dir/"embeddings.npy",mode="w+",dtype="float32",shape=(len(rows),512))
    start=time.monotonic()
    for offset in range(0,len(rows),args.batch_size):
        images=[]
        for row in rows[offset:offset+args.batch_size]:
            with Image.open(ROOT/"data/reprocessed"/row["path"]) as im:
                images.append(im.convert("RGB").copy())
        pixels=processor(images=images,return_tensors="pt")["pixel_values"].to("cuda")
        with torch.inference_mode(),torch.autocast(device_type="cuda",dtype=torch.float16):
            value=model(pixels)
        vectors[offset:offset+len(images)]=torch.nn.functional.normalize(value.float(),dim=1).cpu().numpy()
        if offset//args.batch_size%25==0 or offset+args.batch_size>=len(rows):
            print(f"features {min(offset+args.batch_size,len(rows))}/{len(rows)}, elapsed {time.monotonic()-start:.0f}s",flush=True)
    vectors.flush()
    (args.output_dir/"ids.txt").write_text("\n".join(r["id"] for r in rows)+"\n")
    (args.output_dir/"metadata.jsonl").write_text("\n".join(json.dumps(r) for r in rows)+"\n")
    (args.output_dir/"completed.json").write_text(json.dumps(dict(images=len(rows),embeddings_sha256=sha256(args.output_dir/"embeddings.npy")),indent=2)+"\n")


if __name__ == "__main__":
    main()
