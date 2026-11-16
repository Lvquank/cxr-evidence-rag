"""Compute image embeddings for processed IU Xray images and save to VectorDB (Chroma) or FAISS.

Example:
  python scripts/compute_image_embeddings.py --processed_dir data/processed --out_dir data/embeddings --batch_size 32 --device cpu --vector_db faiss
"""
import argparse
import json
from pathlib import Path
import csv
import sys
from tqdm import tqdm
import numpy as np

import torch
import torch.nn.functional as F

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.embeddings.encoder import (
    DEFAULT_MODEL_NAME,
    RAD_DINO_MODEL_NAME,
    ImagePreprocessor,
    build_patch_mask,
    load_encoder,
)
from src.embeddings.artifacts import clean_embedding_artifacts
from src.config import load_config
from src.data.dataset import ImageDataset
from src.data.metadata import compact_metadata_rows, read_metadata_jsonl, write_metadata_jsonl
from torch.utils.data import DataLoader


def iter_metadata(processed_dir: Path, metadata_path: Path | None = None):
    # Prefer the compact JSONL schema; retain CSV support for existing outputs.
    if metadata_path is not None:
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        if metadata_path.suffix.lower() == ".jsonl":
            yield from read_metadata_jsonl(metadata_path)
            return
        if metadata_path.suffix.lower() == ".csv":
            with metadata_path.open('r', encoding='utf-8') as f:
                yield from csv.DictReader(f)
            return
        raise ValueError("--metadata_path must point to a .jsonl or .csv file")

    jsonl_path = processed_dir / "metadata.jsonl"
    if jsonl_path.exists():
        yield from read_metadata_jsonl(jsonl_path)
        return
    meta_path = processed_dir / "metadata.csv"
    if meta_path.exists():
        with meta_path.open('r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                yield row
    else:
        imgs = list((processed_dir / 'images').glob('*.png'))
        for img in imgs:
            yield { 'id': img.stem, 'path': str(Path('images') / img.name), 'dataset': 'IU-Xray', 'view': '', 'mesh': [], 'problems': [], 'support': [0.0] * 6 }


def save_numpy(embeddings: np.ndarray, ids, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / 'embeddings.npy', embeddings)
    # save ids
    with (out_dir / 'ids.txt').open('w', encoding='utf-8') as f:
        for _id in ids:
            f.write(f"{_id}\n")


def save_metadata_jsonl(metadata_rows, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    write_metadata_jsonl(metadata_rows, out_dir / "metadata.jsonl")


def save_embedding_manifest(out_dir: Path, model, embedding_dimension: int, preprocessor, batch_size: int):
    manifest = {
        "schema_version": 1,
        "embedding_model": model.model_name,
        "model_revision": model.model_revision,
        "embedding_dimension": embedding_dimension,
        "normalization": "L2",
        "image_processor": model.model_name,
        "embedding_strategy": "pathology_aware_projection" if model.projection_metadata else "rad_dino_cls",
        "projection": model.projection_metadata,
        "preprocessing": preprocessor.signature(),
        "embedding_batch_size": batch_size,
        "inference_precision": "float32",
    }
    with (out_dir / "embedding_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)


def collate_rows(batch):
    tensors, rows = zip(*batch)
    max_height = max(tensor.shape[-2] for tensor in tensors)
    max_width = max(tensor.shape[-1] for tensor in tensors)
    padded_tensors = [
        F.pad(tensor, (0, max_width - tensor.shape[-1], 0, max_height - tensor.shape[-2]))
        for tensor in tensors
    ]
    spatial_sizes = [(tensor.shape[-2], tensor.shape[-1]) for tensor in tensors]
    return torch.stack(padded_tensors), list(rows), spatial_sizes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--processed_dir', required=True)
    parser.add_argument(
        '--metadata_path',
        help='Optional metadata .jsonl/.csv file; paths inside remain relative to --processed_dir.',
    )
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--projection_checkpoint', help='Trained pathology projection checkpoint; produces 512-D embeddings.')
    parser.add_argument('--densenet_checkpoint', help='Trusted local DenseNet checkpoint when using the DenseNet backend.')
    parser.add_argument('--vector_db', choices=['none','faiss','chroma'], default='none')
    parser.add_argument(
        '--clean',
        action='store_true',
        help='Remove existing generated embedding artifacts in --out_dir before rebuilding.',
    )
    args = parser.parse_args()

    processed_dir = Path(args.processed_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.clean:
        removed = clean_embedding_artifacts(out_dir)
        print(f"Removed {len(removed)} existing embedding artifact(s) from {out_dir}")

    # config
    cfg = load_config()

    device = torch.device(args.device)

    model_name = cfg['model'].get('image_model', DEFAULT_MODEL_NAME)
    projection_checkpoint = args.projection_checkpoint or cfg['model'].get('pathology_projection_checkpoint')
    densenet_checkpoint = args.densenet_checkpoint or cfg['model'].get('densenet_checkpoint')
    model = load_encoder(
        device=device,
        model_name=model_name,
        projection_checkpoint=projection_checkpoint,
        densenet_checkpoint=densenet_checkpoint,
    )
    preprocessor = ImagePreprocessor(model_name=model.model_name)

    metadata_path = Path(args.metadata_path) if args.metadata_path else None
    metadata = compact_metadata_rows(list(iter_metadata(processed_dir, metadata_path)))

    # RAD-DINO's Hugging Face processor owns resizing and normalization.
    dataset = ImageDataset(
        processed_dir,
        metadata,
        resize=None,
        train=False,
        use_augmentation=False,
        image_transform=preprocessor,
    )

    # Use num_workers=0 on Windows to avoid multiprocessing pickling issues with nested collate functions.
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
        collate_fn=collate_rows,
    )

    ids = []
    embeds = []

    for batch_tensors, batch_rows, spatial_sizes in tqdm(dataloader, desc='Embedding'):
        batch = batch_tensors.to(device)
        patch_mask = None
        if model.model_name == RAD_DINO_MODEL_NAME and model.feature_fusion is not None:
            patch_mask = build_patch_mask(spatial_sizes, batch.shape[-2], batch.shape[-1], device)
        with torch.inference_mode():
            emb = model(batch, patch_mask=patch_mask).cpu().numpy()
        embeds.append(emb)
        ids.extend([r['id'] for r in batch_rows])

    if embeds:
        embeddings = np.ascontiguousarray(np.vstack(embeds), dtype=np.float32)
    else:
        embeddings = np.zeros((0, model.embedding_dimension), dtype=np.float32)

    # Save as numpy and ids
    if len(ids) != len(set(ids)):
        raise ValueError("Metadata contains duplicate image IDs; embeddings cannot be indexed safely.")
    save_numpy(embeddings, ids, out_dir)

    save_metadata_jsonl(metadata, out_dir)
    save_embedding_manifest(out_dir, model, embeddings.shape[1], preprocessor, args.batch_size)

    print(f"Saved embeddings to {out_dir / 'embeddings.npy'} (shape={embeddings.shape})")
    print(f"Saved metadata index to {out_dir / 'metadata.jsonl'} ({len(metadata)} rows)")
    print(f"Saved embedding manifest to {out_dir / 'embedding_manifest.json'}")

    # Optionally build FAISS index or ChromaDB collection
    if args.vector_db == 'faiss':
        try:
            import faiss
            d = embeddings.shape[1]
            # using inner product on L2-normalized vectors is equivalent to cosine
            index = faiss.IndexFlatIP(d)
            index.add(embeddings.astype('float32'))
            faiss.write_index(index, str(out_dir / 'faiss.index'))
            print(f"Saved FAISS index to {out_dir / 'faiss.index'}")
        except Exception as e:
            print('Failed to build FAISS index:', e)

    if args.vector_db == 'chroma':
        try:
            import chromadb
            persist_dir = Path('data/chroma_db')
            persist_dir.mkdir(parents=True, exist_ok=True)
            client = chromadb.PersistentClient(path=str(persist_dir))
            try:
                coll = client.get_collection(name="iuxray_images")
            except Exception:
                coll = client.create_collection(name="iuxray_images")

            id_to_meta = {str(m['id']): m for m in metadata}
            metadatas = []
            for _id in ids:
                meta = id_to_meta.get(str(_id), {})
                safe_meta = {
                    key: value if isinstance(value, (str, int, float, bool)) else json.dumps(value)
                    for key, value in meta.items()
                }
                metadatas.append(safe_meta)

            coll.upsert(ids=ids, embeddings=embeddings.tolist(), metadatas=metadatas)
            print('Saved embeddings to ChromaDB collection iuxray_images at', persist_dir)
        except Exception as e:
            print('Failed to save to ChromaDB:', e)


if __name__ == '__main__':
    main()
