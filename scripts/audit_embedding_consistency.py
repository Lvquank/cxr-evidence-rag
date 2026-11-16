"""Compare production and Hugging Face preprocessing against a frozen FAISS gallery."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import load_config
from src.embeddings.encoder import load_image_processor
from src.retriever.faiss_retriever import FAISSImageRetriever


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    if args.output_json.exists():
        parser.error("Use a new output path to preserve previous measurements.")
    config = load_config()
    started = time.perf_counter()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    retriever = FAISSImageRetriever(
        **{key: config["retrieval"][key] for key in ("index_path", "metadata_path", "processed_dir")},
        device=device, projection_checkpoint=config["model"]["pathology_projection_checkpoint"])
    load_seconds = time.perf_counter() - started
    processor = load_image_processor(retriever.model.model_name)
    records = []
    for index in (0, 100, 1000):
        metadata = retriever.metadata[index]
        image_id = metadata.get("id") or Path(metadata["path"]).stem
        image_path = ROOT / "data/reprocessed/images" / f"{image_id}.png"
        cached = retriever.index.reconstruct(index)
        cached /= np.linalg.norm(cached)
        started = time.perf_counter()
        production = retriever.embed_image(str(image_path))
        elapsed = time.perf_counter() - started
        with Image.open(image_path) as image:
            pixels = processor(images=image.convert("RGB"), return_tensors="pt")["pixel_values"].to(device)
        with torch.inference_mode():
            default = retriever.model(pixels).float()
            default = torch.nn.functional.normalize(default, dim=1).cpu().numpy()[0]
        _, indices = retriever.index.search(production[None, :], 10)
        records.append({"gallery_row": index, "image_id": str(image_id),
                        "production_cosine_to_gallery": float(production @ cached),
                        "hf_default_center_crop_cosine_to_gallery": float(default @ cached),
                        "self_rank_production_top10": indices[0].tolist().index(index) + 1 if index in indices[0] else None,
                        "production_embed_seconds": elapsed})
    output = {"device": device, "model_load_seconds": load_seconds,
              "index_type": type(retriever.index).__name__, "probes": records,
              "note": "Three deterministic gallery examples; diagnostic, not a retrieval quality benchmark."}
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output))


if __name__ == "__main__":
    main()
