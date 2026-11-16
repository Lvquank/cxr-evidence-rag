"""Apply a trained pathology projection to existing raw image-encoder embeddings."""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.embeddings.artifacts import clean_embedding_artifacts
from src.embeddings.encoder import DEFAULT_MODEL_NAME
from src.embeddings.pathology_projection import load_projection_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a 512-D pathology-aware FAISS index from raw embeddings.")
    parser.add_argument("--source_dir", default="data/embeddings", help="Directory containing raw 768-D artifacts")
    parser.add_argument("--out_dir", default="data/embeddings_pathology", help="Output directory for projected artifacts")
    parser.add_argument("--projection_checkpoint", required=True)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    source_dir, out_dir = Path(args.source_dir), Path(args.out_dir)
    source_manifest = json.loads((source_dir / "embedding_manifest.json").read_text(encoding="utf-8"))
    if source_manifest.get("projection"):
        raise ValueError("--source_dir must contain raw, unprojected encoder embeddings.")
    embeddings = np.load(source_dir / "embeddings.npy").astype(np.float32)
    ids = (source_dir / "ids.txt").read_text(encoding="utf-8").splitlines()
    if len(embeddings) != len(ids):
        raise ValueError("Source embeddings and IDs have different lengths.")

    device = torch.device(args.device)
    projection, projection_metadata = load_projection_checkpoint(
        args.projection_checkpoint,
        expected_model_name=source_manifest.get("embedding_model", DEFAULT_MODEL_NAME),
        expected_input_dim=embeddings.shape[1],
        device=device,
    )
    batches = []
    with torch.inference_mode():
        for start in range(0, len(embeddings), args.batch_size):
            batch = torch.from_numpy(embeddings[start:start + args.batch_size]).to(device)
            batches.append(projection(batch).cpu().numpy())
    projected = np.ascontiguousarray(np.vstack(batches), dtype=np.float32)

    out_dir.mkdir(parents=True, exist_ok=True)
    if args.clean:
        clean_embedding_artifacts(out_dir)
    np.save(out_dir / "embeddings.npy", projected)
    (out_dir / "ids.txt").write_text("\n".join(ids) + "\n", encoding="utf-8")
    shutil.copy2(source_dir / "metadata.jsonl", out_dir / "metadata.jsonl")
    manifest = dict(source_manifest)
    manifest.update(
        {
            "embedding_dimension": int(projected.shape[1]),
            "embedding_strategy": "pathology_aware_projection",
            "projection": projection_metadata,
        }
    )
    (out_dir / "embedding_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    import faiss

    index = faiss.IndexFlatIP(projected.shape[1])
    index.add(projected)
    faiss.write_index(index, str(out_dir / "faiss.index"))
    print(f"Saved {len(projected)} projected 512-D embeddings to {out_dir}")


if __name__ == "__main__":
    main()
