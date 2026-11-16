"""Reset or clear the ChromaDB collection used for image embeddings.

Usage:
    python -m scripts.reset_chromadb --collection iuxray_images --reset
    python -m scripts.reset_chromadb --collection iuxray_images --delete
"""
import argparse
from pathlib import Path

import chromadb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", default="iuxray_images", help="Collection name in ChromaDB")
    parser.add_argument("--delete", action="store_true", help="Delete the whole collection")
    parser.add_argument("--reset", action="store_true", help="Delete and recreate collection")
    parser.add_argument("--persist_dir", default="data/chroma_db", help="Directory for Chroma local storage")
    args = parser.parse_args()

    persist_dir = Path(args.persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(persist_dir))

    try:
        existing = client.get_collection(args.collection)
        print(f"Found collection: {args.collection} with {existing.count()} items")
    except Exception:
        existing = None
        print(f"Collection {args.collection} does not exist yet.")

    if args.delete or args.reset:
        if existing is not None:
            client.delete_collection(args.collection)
            print(f"Deleted collection: {args.collection}")
        else:
            print(f"No collection to delete: {args.collection}")

    if args.reset:
        client.create_collection(args.collection)
        print(f"Recreated collection: {args.collection}")

    if not args.delete and not args.reset:
        if existing is not None:
            print(f"Collection {args.collection} remains. No reset performed.")
        else:
            print(f"Nothing to do. Collection {args.collection} does not exist.")


if __name__ == "__main__":
    main()
