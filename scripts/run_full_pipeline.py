#!/usr/bin/env python3
"""
Full Pipeline: Classifier → Retriever → Minimal JSON → LLM Analysis

Thực thi toàn bộ pipeline từ input image đến LLM response
"""

import json
import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from llm_client import create_llm_client, LLMConfig
from prompts import MedicalPrompts, CustomPromptBuilder, build_prompt_with_evidence, TARGET_PROBLEM_LABELS, validate_radiology_prediction
from generate_minimal_json_for_llm import generate_minimal_json
from case_report_docx import build_case_report_docx

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

UNCERTAIN_EVIDENCE_STATUSES = {"M", "I", "D"}


def select_llm_route(
    minimal_json: Dict[str, Any],
    image_path: Optional[str],
    default_model: str,
    escalation_model: str = "gpt-5.6-luna",
    vision_escalation: bool = True,
    image_detail: str = "high",
    force_vision: bool = False,
) -> Dict[str, Any]:
    """Choose the least expensive route that has enough input to resolve uncertainty."""
    statuses: List[str] = sorted({
        str(row.get("s") if isinstance(row, dict) else row[2])
        for row in minimal_json.get("f", [])
        if (isinstance(row, dict) and row.get("s"))
        or (isinstance(row, list) and len(row) >= 3)
    })
    uncertain = sorted(set(statuses) & UNCERTAIN_EVIDENCE_STATUSES)
    resolved_image = Path(image_path).expanduser() if image_path else None
    image_available = bool(resolved_image and resolved_image.is_file())

    if force_vision and vision_escalation and image_available:
        return {
            "name": "vision_forced",
            "model": escalation_model,
            "image_path": str(resolved_image.resolve()),
            "image_detail": image_detail,
            "statuses": statuses,
            "reason": "forced_vision_with_source_image",
        }

    if force_vision and not image_available:
        return {
            "name": "text_fallback",
            "model": default_model,
            "image_path": None,
            "image_detail": None,
            "statuses": statuses,
            "reason": "forced_vision_without_source_image",
        }

    if vision_escalation and uncertain and image_available:
        return {
            "name": "vision_escalation",
            "model": escalation_model,
            "image_path": str(resolved_image.resolve()),
            "image_detail": image_detail,
            "statuses": statuses,
            "reason": "uncertain_evidence_with_source_image",
        }

    if uncertain:
        return {
            "name": "text_fallback",
            "model": default_model,
            "image_path": None,
            "image_detail": None,
            "statuses": statuses,
            "reason": (
                "vision_escalation_disabled"
                if not vision_escalation else "uncertain_evidence_without_source_image"
            ),
        }

    return {
        "name": "text_default",
        "model": default_model,
        "image_path": None,
        "image_detail": None,
        "statuses": statuses,
        "reason": "supported_evidence_only",
    }


class MedicalImageRAGPipeline:
    """Orchestrator cho full pipeline"""

    def __init__(self, input_file: str, output_dir: str = "data/llm_responses",
                 llm_config: Optional[LLMConfig] = None, prompt_type: str = "radiology",
                 image_path: Optional[str] = None,
                 study_id: Optional[str] = None,
                 escalation_model: str = "gpt-5.6-luna",
                 vision_escalation: bool = True,
                 image_detail: str = "high",
                 force_vision: bool = False,
                 report_docx: Optional[str] = None,
                 problem_label_policy: str = "selected_only",
                 include_all_classifier_probabilities: bool = False,
                 thresholds_path: Optional[str] = None):
        """
        Khởi tạo pipeline
        
        Args:
            input_file: Path to original classifier_retriever_evidence.json
            output_dir: Output directory
            llm_config: LLM configuration
            prompt_type: Loại prompt (basic, detailed, reasoning, treatment, risk, qa)
        """
        self.input_file = input_file
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if problem_label_policy not in {"selected_only", "all_six"}:
            raise ValueError(f"Unknown problem_label_policy: {problem_label_policy}")
        self.problem_label_policy = problem_label_policy
        self.include_all_classifier_probabilities = include_all_classifier_probabilities
        self.thresholds_path = thresholds_path
        self.allowed_problem_labels: tuple[str, ...] = ()

        self.prompt_type = prompt_type
        self.llm_config = llm_config or LLMConfig()
        self.active_llm_config = self.llm_config
        self.image_path = image_path
        self.study_id = str(study_id) if study_id is not None else None
        self.escalation_model = escalation_model
        self.vision_escalation = vision_escalation
        self.image_detail = image_detail
        self.force_vision = force_vision
        self.report_docx = report_docx
        self.report_file: Optional[Path] = None
        self.llm = None
        
        self.evidence = None
        self.minimal_json = None
        self.llm_response = None
        self.llm_metadata = None
        self.prediction = None
        self.route = None
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def run(self) -> Dict[str, Any]:
        """
        Chạy full pipeline
        
        Returns:
            Dictionary chứa toàn bộ results
        """
        logger.info("=" * 70)
        logger.info("🚀 STARTING MEDICAL IMAGE RAG PIPELINE")
        logger.info("=" * 70)

        try:
            # Step 1: Load evidence
            logger.info("\n[Step 1/6] Loading evidence data...")
            self._load_evidence()

            # Step 2: Generate minimal JSON
            logger.info("\n[Step 2/6] Generating minimal JSON...")
            self._generate_minimal_json()

            # Step 3: Route to the lowest-cost sufficient model/input combination
            logger.info("\n[Step 3/6] Selecting model route...")
            self._select_route()

            # Step 4: Initialize LLM client
            logger.info("\n[Step 4/6] Initializing LLM client...")
            self._init_llm()

            # Step 5: Build prompt
            logger.info("\n[Step 5/6] Building prompt...")
            prompt = self._build_prompt()

            # Step 6: Send to LLM and get response
            logger.info("\n[Step 6/6] Sending to LLM...")
            self._send_to_llm(prompt)

            # Save results
            logger.info("\n[Save] Saving results...")
            self._save_results()

            logger.info("\n" + "=" * 70)
            logger.info("✅ PIPELINE COMPLETED SUCCESSFULLY")
            logger.info("=" * 70)

            return {
                "status": "success",
                "evidence": self.evidence,
                "minimal_json": self.minimal_json,
                "llm_response": self.llm_response,
                "llm_metadata": self.llm_metadata,
                "prediction": self.prediction,
                "route": self._route_for_output(),
                "problem_label_policy": self.problem_label_policy,
                "allowed_problem_labels": list(self.allowed_problem_labels),
                "source": self._source_for_output(),
                "report_docx": str(self.report_file) if self.report_file else None,
                "output_dir": str(self.output_dir)
            }

        except Exception as e:
            logger.error(f"\n❌ PIPELINE FAILED: {e}", exc_info=True)
            return {
                "status": "failed",
                "error": str(e)
            }

    def _load_evidence(self):
        """Load evidence từ JSON file"""
        logger.info(f"Reading: {self.input_file}")
        
        if not os.path.exists(self.input_file):
            raise FileNotFoundError(f"Evidence file not found: {self.input_file}")
        
        with open(self.input_file, 'r', encoding='utf-8') as f:
            self.evidence = json.load(f)
        
        logger.info(f"✅ Loaded evidence")
        self._print_evidence_summary()

    def _print_evidence_summary(self):
        """Print summary của evidence"""
        clf = self.evidence.get("classifier", {}).get("raw_probabilities", {})
        ret = self.evidence.get("retriever", {})
        
        logger.info(f"  - Image: {self.evidence.get('image_path')}")
        logger.info(f"  - Classifier predictions: {len(clf)} labels")
        logger.info(f"  - Retriever evidence: {len(ret.get('common_evidence', []))} hits")
        logger.info(f"  - Label evidence: {len(ret.get('label_evidence', []))} labels")

    def _generate_minimal_json(self):
        """Generate minimal JSON từ evidence"""
        minimal_file = self.output_dir / f"minimal_{self.timestamp}.json"
        logger.info(f"Generating minimal JSON → {minimal_file}")
        
        # Generate minimal
        generate_minimal_json(
            input_file=self.input_file,
            output_file=str(minimal_file),
            top_k_common=1,
            top_k_label_evidence=6,
            include_all_classifier_probabilities=self.include_all_classifier_probabilities,
            thresholds_path=self.thresholds_path,
        )
        
        # Load minimal JSON
        with open(minimal_file, 'r', encoding='utf-8') as f:
            self.minimal_json = json.load(f)
        
        logger.info(f"✅ Minimal JSON created: {minimal_file.name}")

    def _select_route(self):
        """Use image input only when the compact evidence is uncertain."""
        source_image = self.image_path or self.evidence.get("image_path")
        self.route = select_llm_route(
            minimal_json=self.minimal_json,
            image_path=source_image,
            default_model=self.llm_config.model,
            escalation_model=self.escalation_model,
            vision_escalation=self.vision_escalation,
            image_detail=self.image_detail,
            force_vision=self.force_vision,
        )
        if self.problem_label_policy == "all_six":
            attached_image = self.route.get("image_path")
            if not attached_image or not Path(attached_image).is_file():
                raise RuntimeError(
                    "Rescue policy 'all_six' requires a readable, attached query chest radiograph. "
                    "Running all_six rescue without an image is strictly prohibited."
                )
        self.active_llm_config = replace(self.llm_config, model=self.route["model"])
        logger.info(
            "Route=%s; model=%s; statuses=%s; image_attached=%s (%s)",
            self.route["name"], self.route["model"],
            ",".join(self.route["statuses"]) or "none",
            bool(self.route["image_path"]), self.route["reason"],
        )

    def _route_for_output(self) -> Dict[str, Any]:
        """Keep audit metadata useful without persisting an image filesystem path."""
        return {
            "name": self.route["name"],
            "reason": self.route["reason"],
            "statuses": self.route["statuses"],
            "model": self.route["model"],
            "image_attached": bool(self.route["image_path"]),
            "image_detail": self.route["image_detail"],
        }

    def _init_llm(self):
        """Initialize LLM client"""
        logger.info(f"Creating LLM client: {self.active_llm_config.model}")
        
        self.llm = create_llm_client(**self.active_llm_config.__dict__)
        
        logger.info(f"✅ LLM client ready")

    def _build_prompt(self) -> str:
        """Build prompt từ template + evidence"""
        logger.info(f"Building prompt: {self.prompt_type}")
        
        # Get template
        template = MedicalPrompts.get_prompt(self.prompt_type)
        
        # Check if prompt uses {evidence} placeholder (new style)
        if "{evidence}" in template.user:
            # Use helper function for new-style prompts
            formatted = build_prompt_with_evidence(
                template=template,
                evidence=self.minimal_json,
            )
            prompt = formatted
        else:
            # Use old-style formatting for backward compatibility
            clf_info = self._format_classifier_info()
            ret_info = self._format_retrieval_info()
            lbl_info = self._format_label_evidence()
            
            prompt = template.user.format(
                image=self.minimal_json.get("img", "unknown"),
                predictions=clf_info,
                classifier_info=clf_info,
                retrieval_info=ret_info,
                label_evidence=lbl_info,
                num_cases=len(self.minimal_json.get("ret", {}).get("hits", [])),
                evidence_status=lbl_info,
                supporting_evidence=ret_info,
                contradicting_evidence="None identified",
                diagnosis="Pending",
                findings=ret_info,
                severity="Moderate",
                case_outcomes="Similar cases show positive treatment response",
                confidence_scores=clf_info,
                similar_cases_summary=ret_info,
                analysis_results="Analysis in progress",
                classifier_scores=clf_info,
                evidence_consistency="Consistent"
            )

        if self.route and self.route.get("image_path") and isinstance(prompt, dict) and self.prompt_type != "radiology_rescue":
            prompt = dict(prompt)
            prompt["system"] = (
                prompt["system"]
                + "\nFor this case, directly assess the attached query chest X-ray and reconcile it with "
                "the supporting and contrasting retrieved evidence. Do not assert a target pathology "
                "from retrieval similarity alone. Resolve disagreement using visible query-specific signs: "
                "mixed or non-support-dominant retrieval alone does not rule out a pathology. "
                "Include it only when the query assessment supports it and the output schema permits it. "
                "When image quality or signs are inconclusive, preserve uncertainty without inventing "
                "a positive or negative finding. Return only the required JSON; do not expose step-by-step reasoning."
            )
        
        prompt_size = sum(len(value) for value in prompt.values()) if isinstance(prompt, dict) else len(prompt)
        logger.info(f"Prompt created ({prompt_size} characters)")
        return prompt

    def _format_classifier_info(self) -> str:
        """Format classifier info"""
        clf = self.minimal_json.get("clf", {})
        lines = []
        if isinstance(clf, list):
            for row in clf:
                lines.append(f"- {row.get('l')}: {row.get('p', 0.0):.1%} confidence")
        elif isinstance(clf, dict):
            for label, prob in sorted(clf.items(), key=lambda x: x[1], reverse=True):
                lines.append(f"- {label}: {prob:.1%} confidence")
        return "\n".join(lines) if lines else "No predictions"

    def _format_retrieval_info(self) -> str:
        """Format retrieval info"""
        hits = self.minimal_json.get("ret", {}).get("hits", [])
        lines = []
        for hit in hits:
            r = hit.get("r", "?")
            sim = hit.get("sim", "?")
            problems = ", ".join(hit.get("problems", [])[:3])
            lines.append(f"- Case {r}: Similarity {sim:.2f} → {problems}")
        return "\n".join(lines) if lines else "No similar cases"

    def _format_label_evidence(self) -> str:
        """Format label evidence"""
        lbl = self.minimal_json.get("ret", {}).get("lbl", {})
        lines = []
        for label, evidence in lbl.items():
            p = evidence.get("p", "?")
            s = evidence.get("s", "?")
            sup_count = len(evidence.get("sup", []))
            lines.append(f"- {label}: confidence={p:.1%}, status={s}, supporting={sup_count}")
        return "\n".join(lines) if lines else "No label evidence"

    def _send_to_llm(self, prompt: Any):
        """Send prompt to LLM"""
        logger.info(f"Sending request to {self.active_llm_config.model}...")
        selected = (self.evidence.get("classifier", {}).get("selected_labels", []) if self.evidence else [])
        if self.problem_label_policy == "all_six":
            allowed_problem_labels = TARGET_PROBLEM_LABELS
        else:
            allowed_problem_labels = tuple(label for label in selected if label in TARGET_PROBLEM_LABELS)
        self.allowed_problem_labels = allowed_problem_labels
        response_schema = MedicalPrompts.get_output_schema(self.prompt_type, allowed_problem_labels)
        
        if isinstance(prompt, dict):
            self.llm_response = self.llm.send_request(
                prompt["user"],
                system=prompt["system"],
                image_path=self.route["image_path"],
                image_detail=self.route["image_detail"] or "high",
                response_schema=response_schema,
            )
        else:
            self.llm_response = self.llm.send_request(
                prompt,
                image_path=self.route["image_path"],
                image_detail=self.route["image_detail"] or "high",
                response_schema=response_schema,
            )
        self.prediction = self._parse_prediction(self.llm_response, response_schema, allowed_problem_labels)
        if hasattr(self.llm, "get_last_response_metadata"):
            self.llm_metadata = self.llm.get_last_response_metadata()
        if response_schema is not None and self.prediction is None:
            if self.active_llm_config.dry_run:
                logger.info("Dry-run mode: prediction was not generated because request was not sent.")
            else:
                failure = {
                    "reason": "structured_prediction_schema_validation_failed",
                    "model": self.active_llm_config.model,
                    "reasoning_effort": self.active_llm_config.reasoning_effort,
                    "response": self.llm_response,
                    "api": self.llm_metadata,
                    "allowed_problem_labels": list(allowed_problem_labels),
                }
                (self.output_dir / f"invalid_response_{self.timestamp}.json").write_text(
                    json.dumps(failure, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                raise ValueError("LLM response did not match the structured prediction schema.")
        
        logger.info(f"✅ Received response from LLM ({len(self.llm_response)} characters)")

    @staticmethod
    def _parse_prediction(
        response: str, response_schema: Optional[Dict[str, Any]],
        allowed_problem_labels: Optional[tuple[str, ...]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Validate the structured radiology response before saving it for evaluation."""
        if response_schema is None:
            return None
        try:
            parsed = json.loads(response)
        except (TypeError, json.JSONDecodeError):
            logger.warning("Structured response was not JSON; prediction will not be exported.")
            return None
        if not isinstance(parsed, dict):
            logger.warning("Structured response was not an object; prediction will not be exported.")
            return None
        prediction = validate_radiology_prediction(parsed, allowed_problem_labels)
        if prediction is None:
            logger.warning("Structured response did not match the six-pathology-or-normal contract.")
        return prediction

    def _save_results(self):
        """Save all results to files"""
        # Save response
        response_file = self.output_dir / f"response_{self.timestamp}.txt"
        with open(response_file, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("MEDICAL IMAGE ANALYSIS - LLM RESPONSE\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Model: {self.active_llm_config.model}\n")
            f.write(f"Route: {self.route['name']} ({self.route['reason']})\n")
            f.write(f"Image attached: {bool(self.route['image_path'])}\n")
            f.write(f"Prompt Type: {self.prompt_type}\n")
            f.write(f"Timestamp: {self.timestamp}\n\n")
            f.write("RESPONSE:\n")
            f.write("-" * 70 + "\n")
            f.write(self.llm_response)
        
        logger.info(f"✅ Response saved: {response_file.name}")

        if self.report_docx is not None:
            requested_path = Path(self.report_docx).expanduser() if self.report_docx else None
            report_path = requested_path or (self.output_dir / f"case_report_{self.timestamp}.docx")
            self.report_file = build_case_report_docx(
                evidence=self.evidence,
                prediction=self.prediction,
                raw_response=self.llm_response,
                output_path=report_path,
                study_id=self.study_id,
                route=self._route_for_output(),
                timestamp=self.timestamp,
            )
            logger.info(f"✅ DOCX case report saved: {self.report_file.name}")
        
        # Save summary JSON
        summary_file = self.output_dir / f"summary_{self.timestamp}.json"
        selected_raw = (self.evidence.get("classifier", {}).get("selected_labels", []) if self.evidence else [])
        predicted_problems = (self.prediction.get("problems", []) if self.prediction else [])
        new_labels = [label for label in predicted_problems if label != "normal" and label not in selected_raw]
        summary = {
            "timestamp": self.timestamp,
            "requested_model": self.llm_config.model,
            "model": self.active_llm_config.model,
            "reasoning_effort": self.active_llm_config.reasoning_effort,
            "route": self._route_for_output(),
            "prompt_type": self.prompt_type,
            "problem_label_policy": self.problem_label_policy,
            "allowed_problem_labels": list(self.allowed_problem_labels),
            "evidence_schema": self.minimal_json.get("v"),
            "evidence_enrichment": "polarity-strength-selected-label-checklist-v1",
            "findings_sent": len(self.minimal_json.get("f", [])),
            "fallback_hits_sent": len(self.minimal_json.get("g", [])),
            "classifier_scores_sent": len(self.minimal_json.get("clf", [])),
            "new_labels": new_labels,
            "response": self.llm_response,
            "prediction": self.prediction,
            "source": self._source_for_output(),
            "report_docx": str(self.report_file) if self.report_file else None,
            "api": self.llm_metadata,
        }
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"✅ Summary saved: {summary_file.name}")
        
        # Save minimal JSON
        minimal_file = self.output_dir / f"minimal_data_{self.timestamp}.json"
        with open(minimal_file, 'w', encoding='utf-8') as f:
            json.dump(self.minimal_json, f, indent=2)
        
        logger.info(f"✅ Minimal JSON saved: {minimal_file.name}")

    def _source_for_output(self) -> Dict[str, Optional[str]]:
        """Identifiers for matching this prediction to the IU-Xray report."""
        source_image = self.image_path or (self.evidence or {}).get("image_path")
        return {
            "study_id": self.study_id,
            "image_name": Path(source_image).name if source_image else None,
        }


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run full Medical Image RAG pipeline"
    )
    parser.add_argument(
        "--input", "-i",
        default="data/classifier_retriever_evidence.json",
        help="Input evidence file"
    )
    parser.add_argument(
        "--output", "-o",
        default="data/llm_responses",
        help="Output directory"
    )
    parser.add_argument(
        "--prompt", "-p",
        default="radiology",
        choices=["radiology", "radiology_query_first", "radiology_rescue", "basic", "detailed", "reasoning", "treatment", "risk", "qa"],
        help="Prompt type"
    )
    parser.add_argument(
        "--model", "-m",
        default="gpt-5.6-luna",
        help="LLM model name"
    )
    parser.add_argument(
        "--temperature", "-t",
        type=float,
        default=None,
        help="LLM temperature; omit to use the model default"
    )
    parser.add_argument(
        "--api-key", "-k",
        default=None,
        help="LLM API key"
    )
    parser.add_argument("--max-output-tokens", type=int, default=8192)
    parser.add_argument("--reasoning-effort", default="high", choices=["none", "low", "medium", "high", "xhigh", "max"])
    parser.add_argument("--image-path", help="Original image path; overrides evidence.image_path")
    parser.add_argument(
        "--study-id",
        help="IU-Xray uid saved with the response for offline evaluation",
    )
    parser.add_argument(
        "--escalation-model",
        default="gpt-5.6-luna",
        help="Model for M/I/D evidence when the original image is available (defaults to Luna for experiments)",
    )
    parser.add_argument(
        "--image-detail",
        default="high",
        choices=["low", "high"],
        help="Vision detail used only by the escalation route",
    )
    parser.add_argument(
        "--disable-vision-escalation",
        action="store_true",
        help="Never attach an image or switch to the escalation model",
    )
    parser.add_argument(
        "--force-vision",
        action="store_true",
        help="Attach the original image whenever it is available, irrespective of evidence status.",
    )
    parser.add_argument(
        "--include-all-classifier-probabilities",
        action="store_true",
        help="Include full 6-label classifier scores and thresholds under 'clf' in minimal JSON",
    )
    parser.add_argument(
        "--problem-label-policy",
        choices=["selected_only", "all_six"],
        default="selected_only",
        help="Allowed label policy for structured output: selected_only (default) or all_six (rescue ablation)",
    )
    parser.add_argument("--no-cache", action="store_true", help="Do not reuse exact requests")
    parser.add_argument("--thresholds-json", help="Selection-time threshold artifact for legacy evidence")
    parser.add_argument("--dry-run", action="store_true", help="Build artifacts without sending an API request")
    parser.add_argument(
        "--report-docx",
        nargs="?",
        const="",
        metavar="PATH",
        help=(
            "Write a DOCX case report after the LLM response. Omit PATH to save "
            "case_report_<timestamp>.docx under --output."
        ),
    )
    
    args = parser.parse_args()
    if args.force_vision and args.disable_vision_escalation:
        parser.error("--force-vision cannot be combined with --disable-vision-escalation.")
    
    # Create config
    llm_config = LLMConfig(
        model=args.model,
        api_key=args.api_key,
        temperature=args.temperature,
        max_tokens=args.max_output_tokens,
        reasoning_effort=args.reasoning_effort,
        cache_enabled=not args.no_cache,
        dry_run=args.dry_run,
    )
    
    # Run pipeline
    pipeline = MedicalImageRAGPipeline(
        input_file=args.input,
        output_dir=args.output,
        llm_config=llm_config,
        prompt_type=args.prompt,
        image_path=args.image_path,
        study_id=args.study_id,
        escalation_model=args.escalation_model,
        vision_escalation=not args.disable_vision_escalation,
        image_detail=args.image_detail,
        force_vision=args.force_vision,
        report_docx=args.report_docx,
        problem_label_policy=args.problem_label_policy,
        include_all_classifier_probabilities=args.include_all_classifier_probabilities,
        thresholds_path=args.thresholds_json,
    )
    
    result = pipeline.run()
    
    # Print result
    if result["status"] == "success":
        print("\n" + "=" * 70)
        print("RESULTS")
        print("=" * 70)
        print("\nLLM RESPONSE:")
        print("-" * 70)
        print(result["llm_response"])
        print("-" * 70)
        print(f"\nOutput directory: {result['output_dir']}")
        if result.get("report_docx"):
            print(f"DOCX report: {result['report_docx']}")
    else:
        print(f"\n❌ Error: {result['error']}")
        sys.exit(1)


if __name__ == "__main__":
    main()
