"""Test retrieval of the top-k most similar images sharing a disease label."""
import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.embeddings.encoder import DEFAULT_MODEL_NAME
from src.retriever.faiss_retriever import FAISSImageRetriever
from src.retriever.pathology_support import CHEXPERT_LABELS


def main():
    cfg = load_config()
    retrieval_cfg = cfg.get("retrieval", {})
    model_cfg = cfg.get("model", {})
    parser = argparse.ArgumentParser(
        description="Retrieve visually similar images with the same disease label."
    )
    parser.add_argument("--image_path", required=True, help="Path to query image")
    parser.add_argument(
        "--disease_label",
        required=True,
        choices=CHEXPERT_LABELS,
        help="Disease label shared by the retrieved images",
    )
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument(
        "--candidate_k",
        type=int,
        default=0,
        help="Number of FAISS candidates to scan; 0 scans the full gallery",
    )
    parser.add_argument(
        "--index_path",
        default=retrieval_cfg.get("index_path", "data/embeddings/faiss.index"),
    )
    parser.add_argument(
        "--metadata_path",
        default=retrieval_cfg.get("metadata_path", "data/embeddings/metadata.jsonl"),
    )
    parser.add_argument(
        "--processed_dir",
        default=retrieval_cfg.get("processed_dir", "data/processed"),
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--model_name", default=DEFAULT_MODEL_NAME)
    parser.add_argument(
        "--projection_checkpoint",
        default=model_cfg.get("pathology_projection_checkpoint"),
    )
    parser.add_argument("--densenet_checkpoint")
    args = parser.parse_args()

    retriever = FAISSImageRetriever(
        index_path=args.index_path,
        metadata_path=args.metadata_path,
        processed_dir=args.processed_dir,
        device=args.device,
        model_name=args.model_name,
        projection_checkpoint=args.projection_checkpoint,
        densenet_checkpoint=args.densenet_checkpoint,
    )
    candidate_k = retriever.index.ntotal if args.candidate_k == 0 else args.candidate_k
    results = retriever.search(
        args.image_path,
        top_k=args.top_k,
        candidate_k=candidate_k,
        required_label=args.disease_label,
    )

    print({
        "image_path": args.image_path,
        "disease_label": args.disease_label,
        "candidate_k": candidate_k,
        "returned": len(results),
    })
    for item in results:
        print({
            "rank": item["rank"],
            "cosine_similarity": item["cosine_similarity"],
            "label_support": item["metadata"]["support"][CHEXPERT_LABELS.index(args.disease_label)],
            "processed_image_path": item["processed_image_path"],
            "id": item["id"],
            "metadata": item["metadata"],
        })


if __name__ == "__main__":
    main()