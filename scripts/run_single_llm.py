#!/usr/bin/env python3
"""Run one minimal/evidence JSON file through gpt-5.6-luna and save JSON output."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from generate_minimal_json_for_llm import build_minimal_data
from llm_client import create_llm_client
from prompts import MedicalPrompts, build_prompt_with_evidence, validate_radiology_prediction


def load_project_env() -> None:
    """Load .env without printing its secret values."""
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    try:
        from dotenv import load_dotenv
        load_dotenv(env_path, override=False)
        return
    except ImportError:
        pass
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def load_minimal_json(path: Path) -> Dict[str, Any]:
    try:
        source = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Input is not valid JSON: {path}") from error
    if not isinstance(source, dict):
        raise ValueError("Input JSON must be an object.")
    if source.get("v") in (4, 5) and isinstance(source.get("f"), list):
        return source
    try:
        return build_minimal_data(source)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Input must be a v4/v5 minimal JSON or classifier/retriever evidence JSON.") from error


def validate_prediction(text: str) -> Dict[str, Any]:
    try:
        prediction = json.loads(text)
    except json.JSONDecodeError as error:
        raise RuntimeError("LLM returned invalid JSON despite the structured-output contract.") from error
    validated = validate_radiology_prediction(prediction)
    if validated is None:
        raise RuntimeError("LLM JSON does not match the six-pathology-or-normal output contract.")
    return validated


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one JSON input through gpt-5.6-luna.")
    parser.add_argument("--input", "-i", type=Path, required=True, help="v4/v5 minimal JSON or classifier/retriever evidence JSON")
    parser.add_argument("--output", "-o", type=Path, help="Output JSON path; defaults to data/llm_responses/single_<timestamp>.json")
    parser.add_argument("--max-output-tokens", type=int, default=8192)
    parser.add_argument("--reasoning-effort", choices=["none", "low", "medium", "high", "xhigh", "max"], default="high")
    parser.add_argument("--no-cache", action="store_true", help="Do not reuse an identical local request")
    args = parser.parse_args()

    load_project_env()
    minimal = load_minimal_json(args.input)
    api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("GPT_TOKEN")
    if not api_key:
        raise RuntimeError("No API key configured. Add OPENAI_API_KEY (recommended), LLM_API_KEY, or legacy GPT_TOKEN to .env.")

    template = MedicalPrompts.get_prompt("radiology")
    prompt = build_prompt_with_evidence(template, minimal)
    client = create_llm_client(
        model="gpt-5.6-luna",
        api_key=api_key,
        max_tokens=args.max_output_tokens,
        reasoning_effort=args.reasoning_effort,
        cache_enabled=not args.no_cache,
    )
    response = client.send_request(
        prompt["user"], system=prompt["system"],
        response_schema=MedicalPrompts.get_output_schema("radiology"),
    )
    output = {
        "timestamp": datetime.now().astimezone().isoformat(),
        "model": "gpt-5.6-luna",
        "input_file": str(args.input),
        "minimal_json": minimal,
        "prediction": validate_prediction(response),
        "api": client.get_last_response_metadata(),
    }
    destination = args.output or ROOT / "data/llm_responses" / f"single_{datetime.now():%Y%m%d_%H%M%S}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved LLM JSON result: {destination}")


if __name__ == "__main__":
    main()
