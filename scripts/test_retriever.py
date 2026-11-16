"""Test FAISS retriever using a query image path.

Example:
    python -m scripts.test_retriever --image_path "data/processed/images/xxx.png" --disease_label "Edema"
"""
import argparse
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.retriever.faiss_retriever import FAISSImageRetriever
from src.retriever.pathology_support import CHEXPERT_LABELS, support_from_report
from src.embeddings.encoder import DEFAULT_MODEL_NAME
from src.config import load_config


def main():
    cfg = load_config()
    retrieval_cfg = cfg.get("retrieval", {})
    model_cfg = cfg.get("model", {})
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', required=True, help='Path to query image')
    parser.add_argument('--index_path', default=retrieval_cfg.get('index_path', 'data/embeddings/faiss.index'), help='FAISS index path')
    parser.add_argument('--metadata_path', default=retrieval_cfg.get('metadata_path', 'data/embeddings/metadata.jsonl'), help='Compact JSONL metadata for lookup')
    parser.add_argument('--processed_dir', default=retrieval_cfg.get('processed_dir', 'data/processed'), help='Processed images directory used to resolve matched paths')
    parser.add_argument('--top_k', type=int, default=10, help='Number of unique studies returned after reranking')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--model_name', default=DEFAULT_MODEL_NAME, help='Retrieval image embedding model')
    parser.add_argument(
        '--projection_checkpoint',
        default=model_cfg.get('pathology_projection_checkpoint'),
        help='Trained pathology projection checkpoint for 512-D retrieval',
    )
    parser.add_argument('--densenet_checkpoint', help='Trusted local DenseNet checkpoint for the DenseNet backend')
    query_report_group = parser.add_mutually_exclusive_group()
    query_report_group.add_argument('--disease_label', choices=CHEXPERT_LABELS, help='Disease label used for label-based reranking')
    query_report_group.add_argument('--query_report_path', help='UTF-8 report text file for support-aware reranking')
    query_report_group.add_argument('--query_report_text', help='Inline report text for support-aware reranking')
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.2,
        help='Visual-score weight during pathology reranking; lower values prioritize disease-label support.',
    )
    parser.add_argument('--candidate_k', type=int, default=200, help='FAISS cosine shortlist size before reranking')
    parser.add_argument(
        '--strict_disease_label',
        action='store_true',
        help='Exclude shortlist candidates without the selected disease label; may return fewer than --top_k results.',
    )
    args = parser.parse_args()
    if args.candidate_k < args.top_k:
        parser.error('--candidate_k must be at least --top_k.')

    retriever = FAISSImageRetriever(
        index_path=args.index_path,
        metadata_path=args.metadata_path,
        processed_dir=args.processed_dir,
        device=args.device,
        model_name=args.model_name,
        projection_checkpoint=args.projection_checkpoint,
        densenet_checkpoint=args.densenet_checkpoint,
    )
    query_support = None
    if args.disease_label:
        query_support = {label: float(label == args.disease_label) for label in CHEXPERT_LABELS}
        print({'disease_label': args.disease_label, 'query_support': query_support, 'alpha': args.alpha})
    elif args.query_report_path:
        query_report = Path(args.query_report_path).read_text(encoding='utf-8')
        query_support = support_from_report(query_report)
        print({'query_support': query_support, 'alpha': args.alpha})
    elif args.query_report_text:
        query_support = support_from_report(args.query_report_text)
        print({'query_support': query_support, 'alpha': args.alpha})

    results = retriever.search(
        args.image_path,
        top_k=args.top_k,
        query_support=query_support,
        alpha=args.alpha,
        candidate_k=args.candidate_k,
        required_label=args.disease_label if args.strict_disease_label else None,
    )

    if args.disease_label and not any((item['pathology_support'] or 0.0) > 0.0 for item in results):
        print(
            f"Warning: no '{args.disease_label}' support was found in the cosine Top-{args.candidate_k} shortlist; "
            "returned results fall back to visual similarity."
        )

    print(f"Top {args.top_k} nearest images for: {args.image_path}")
    for item in results:
        print({
            'rank': item['rank'],
            'distance': item['distance'],
            'pathology_support': item['pathology_support'],
            'final_score': item['final_score'],
            'processed_image_path': item['processed_image_path'],
            'id': item['id'],
            'metadata': item['metadata'],
        })


if __name__ == '__main__':
    main()
