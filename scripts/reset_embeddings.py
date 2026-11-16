"""Remove generated embedding artifacts without deleting processed images or other files."""

import argparse
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.embeddings.artifacts import clean_embedding_artifacts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="data/embeddings")
    args = parser.parse_args()

    removed = clean_embedding_artifacts(Path(args.out_dir))
    for path in removed:
        print(f"Removed {path}")
    print(f"Removed {len(removed)} embedding artifact(s).")


if __name__ == "__main__":
    main()
