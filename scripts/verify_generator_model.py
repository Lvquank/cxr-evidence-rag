#!/usr/bin/env python3
"""Verify the generator model string from saved run summaries, not the current .env."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def collect(root: Path) -> dict[str, Any]:
    summaries = sorted(root.rglob("summary_*.json"))
    models: Counter[str] = Counter()
    requested: Counter[str] = Counter()
    response_ids = 0
    invalid: list[str] = []
    for path in summaries:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            invalid.append(str(path))
            continue
        model = data.get("api", {}).get("model") or data.get("model")
        if model:
            models[str(model)] += 1
        else:
            invalid.append(str(path))
        if data.get("requested_model"):
            requested[str(data["requested_model"])] += 1
        if data.get("api", {}).get("response_id"):
            response_ids += 1
    return {
        "run_root": str(root.resolve()),
        "summary_files": len(summaries),
        "models_from_saved_api_metadata": dict(models),
        "requested_models": dict(requested),
        "summaries_with_response_id": response_ids,
        "invalid_or_missing_model": invalid,
        "verified_model": next(iter(models)) if len(models) == 1 and not invalid else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify exact LLM model names in saved run metadata.")
    parser.add_argument("--run", action="append", required=True, metavar="NAME=PATH", help="Repeatable pipeline name and LLM output root.")
    parser.add_argument("--output-json", required=True, type=Path)
    args = parser.parse_args()
    runs: dict[str, Any] = {}
    for value in args.run:
        if "=" not in value:
            parser.error("Every --run must be NAME=PATH.")
        name, raw_path = value.split("=", 1)
        root = Path(raw_path)
        if not root.is_dir():
            raise FileNotFoundError(f"Run root does not exist: {root}")
        runs[name] = collect(root)
    output = {
        "schema_version": 1,
        "verification_method": "Saved API response metadata is preferred over current configuration and README text.",
        "runs": runs,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
