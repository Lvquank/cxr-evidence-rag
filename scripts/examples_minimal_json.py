#!/usr/bin/env python3
"""
Examples sử dụng generate_minimal_json_for_llm script
"""

import sys
import json
from pathlib import Path

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent))

from generate_minimal_json_for_llm import generate_minimal_json, batch_process


def example_1_single_file():
    """Example 1: Generate minimal JSON cho 1 file"""
    print("=" * 60)
    print("EXAMPLE 1: Tối ưu hóa 1 file")
    print("=" * 60)

    generate_minimal_json(
        input_file="data/classifier_retriever_evidence.json",
        output_file="data/classifier_retriever_evidence_minimal.json",
        top_k_common=3,
        top_k_label_evidence=2
    )


def example_2_aggressive_compression():
    """Example 2: Aggressive compression (very minimal)"""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Aggressive compression (siêu gọn cho token)")
    print("=" * 60)

    generate_minimal_json(
        input_file="data/classifier_retriever_evidence.json",
        output_file="data/classifier_retriever_evidence_ultra_minimal.json",
        top_k_common=1,
        top_k_label_evidence=1
    )

    # Show comparison
    with open("data/classifier_retriever_evidence_ultra_minimal.json") as f:
        ultra_minimal = json.load(f)

    print("\n📊 Ultra Minimal Format:")
    print(json.dumps(ultra_minimal, indent=2)[:500] + "...")


def example_3_rich_compression():
    """Example 3: Rich compression (more details for reasoning)"""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Rich compression (chi tiết hơn cho reasoning)")
    print("=" * 60)

    generate_minimal_json(
        input_file="data/classifier_retriever_evidence.json",
        output_file="data/classifier_retriever_evidence_rich_minimal.json",
        top_k_common=5,
        top_k_label_evidence=3
    )


def example_4_batch_processing():
    """Example 4: Batch process multiple files"""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Batch processing (xử lý nhiều file)")
    print("=" * 60)

    # Tạo output directory nếu chưa có
    Path("data/minimal").mkdir(exist_ok=True)

    batch_process(
        input_dir="data/",
        output_dir="data/minimal/",
        pattern="classifier_retriever_evidence.json"
    )


def example_5_compare_formats():
    """Example 5: So sánh formats"""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: So sánh Original vs Minimal")
    print("=" * 60)

    with open("data/classifier_retriever_evidence.json") as f:
        original = json.load(f)

    with open("data/classifier_retriever_evidence_minimal.json") as f:
        minimal = json.load(f)

    original_size = len(json.dumps(original).encode('utf-8'))
    minimal_size = len(json.dumps(minimal).encode('utf-8'))
    compression = (1 - minimal_size / original_size) * 100

    print(f"\n📊 COMPARISON:")
    print(f"Original size: {original_size:,} bytes")
    print(f"Minimal size:  {minimal_size:,} bytes")
    print(f"Compression:   {compression:.1f}%")
    print(f"Ratio:         {original_size/minimal_size:.1f}x")

    print(f"\n🔍 ORIGINAL STRUCTURE:")
    print(f"  - Keys: {list(original.keys())}")
    print(f"  - Classifier labels: {list(original.get('classifier', {}).get('raw_probabilities', {}).keys())}")
    print(f"  - Retriever model: {original.get('retriever', {}).get('model')}")
    print(f"  - Common evidence count: {len(original.get('retriever', {}).get('common_evidence', []))}")
    print(f"  - Label evidence count: {len(original.get('retriever', {}).get('label_evidence', []))}")

    print(f"\n🔍 MINIMAL STRUCTURE:")
    print(f"  - Keys: {list(minimal.keys())}")
    print(f"  - Classifier labels: {list(minimal.get('clf', {}).keys())}")
    print(f"  - Retriever model: {minimal.get('ret', {}).get('model')}")
    print(f"  - Hits count: {len(minimal.get('ret', {}).get('hits', []))}")
    print(f"  - Labels count: {len(minimal.get('ret', {}).get('lbl', {}))}")

    print(f"\n📝 SAMPLE MINIMAL JSON:")
    print(json.dumps(minimal, indent=2)[:800] + "\n...")


def example_6_for_llm():
    """Example 6: Prepare JSON cho LLM prompt"""
    print("\n" + "=" * 60)
    print("EXAMPLE 6: Prepare for LLM Prompt")
    print("=" * 60)

    with open("data/classifier_retriever_evidence_minimal.json") as f:
        minimal = json.load(f)

    llm_context = f"""
Medical Image Analysis Report:
- Image: {minimal['img']}
- Classifier Predictions: {json.dumps(minimal['clf'], indent=2)}
- Top Retrieval Evidence: {len(minimal['ret']['hits'])} cases
- Label Evidence: {len(minimal['ret']['lbl'])} labels analyzed

Evidence Summary:
"""

    for label, evidence in minimal['ret']['lbl'].items():
        llm_context += f"\n{label}: (confidence={evidence['p']}, status={evidence['s']})"
        for i, sup in enumerate(evidence['sup'], 1):
            problems_str = ", ".join(sup['problems'][:3])
            llm_context += f"\n  ✓ Supporting{i}: {problems_str}"

    print(llm_context)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run examples")
    parser.add_argument("--example", type=int, choices=[1, 2, 3, 4, 5, 6, 0],
                       default=0, help="Run specific example (0=all)")

    args = parser.parse_args()

    if args.example in [1, 0]:
        example_1_single_file()

    if args.example in [2, 0]:
        example_2_aggressive_compression()

    if args.example in [3, 0]:
        example_3_rich_compression()

    if args.example in [4, 0]:
        try:
            example_4_batch_processing()
        except Exception as e:
            print(f"⚠️  Batch processing example skipped: {e}")

    if args.example in [5, 0]:
        try:
            example_5_compare_formats()
        except Exception as e:
            print(f"⚠️  Comparison example skipped: {e}")

    if args.example in [6, 0]:
        try:
            example_6_for_llm()
        except Exception as e:
            print(f"⚠️  LLM prompt example skipped: {e}")

    print("\n" + "=" * 60)
    print("✨ Examples completed!")
    print("=" * 60)
