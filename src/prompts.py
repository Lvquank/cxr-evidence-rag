"""
Prompt Templates cho Medical Image Analysis
Người dùng có thể tùy chỉnh các prompt này
"""

from typing import Dict, Any, Optional, Sequence
from dataclasses import dataclass
import json


TARGET_PROBLEM_LABELS = (
    "Atelectasis",
    "Cardiomegaly",
    "Edema",
    "Pleural Effusion",
    "Pneumonia",
    "Pneumothorax",
)
NORMAL_PROBLEM_LABEL = "normal"


RADIOLOGY_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "problems": {
            "type": "array",
            "description": "One or more target pathology labels, or exactly [normal] when no target pathology is present.",
            "items": {"type": "string", "enum": [*TARGET_PROBLEM_LABELS, NORMAL_PROBLEM_LABEL]},
            "minItems": 1,
            "maxItems": len(TARGET_PROBLEM_LABELS),
        },
        "findings": {
            "type": "string",
            "description": "One to three concise factual radiographic observation sentences.",
        },
        "impression": {
            "type": "string",
            "description": "One concise chest-X-ray impression.",
        },
    },
    "required": ["problems", "findings", "impression"],
    "additionalProperties": False,
}


def _allowed_problem_labels(allowed_problem_labels: Optional[Sequence[str]] = None) -> tuple[str, ...]:
    selected = tuple(allowed_problem_labels) if allowed_problem_labels is not None else TARGET_PROBLEM_LABELS
    if any(label not in TARGET_PROBLEM_LABELS for label in selected):
        raise ValueError("Allowed radiology problem labels must be a subset of the six target labels.")
    return selected


def radiology_output_schema(allowed_problem_labels: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """Build a strict schema whose pathology enum matches classifier candidates."""
    schema = json.loads(json.dumps(RADIOLOGY_OUTPUT_SCHEMA))
    schema["properties"]["problems"]["items"]["enum"] = [*_allowed_problem_labels(allowed_problem_labels), NORMAL_PROBLEM_LABEL]
    schema["properties"]["problems"]["maxItems"] = max(1, len(_allowed_problem_labels(allowed_problem_labels)))
    return schema


def validate_radiology_prediction(
    payload: Any, allowed_problem_labels: Optional[Sequence[str]] = None,
) -> Optional[Dict[str, Any]]:
    """Validate the shared six-pathology-or-normal output contract."""
    if not isinstance(payload, dict):
        return None
    problems = payload.get("problems")
    findings = payload.get("findings")
    impression = payload.get("impression")
    allowed = set(_allowed_problem_labels(allowed_problem_labels)) | {NORMAL_PROBLEM_LABEL}
    if (
        not isinstance(problems, list)
        or not problems
        or not all(isinstance(item, str) and item in allowed for item in problems)
        or len(problems) != len(set(problems))
        or not isinstance(findings, str)
        or not isinstance(impression, str)
    ):
        return None
    if NORMAL_PROBLEM_LABEL in problems and problems != [NORMAL_PROBLEM_LABEL]:
        return None
    return {"problems": problems, "findings": findings, "impression": impression}


@dataclass
class PromptTemplate:
    """Base prompt template"""
    name: str
    system: str
    user: str

    def format(self, **kwargs) -> Dict[str, str]:
        """Format template với variables"""
        return {
            "system": self.system.format(**kwargs) if "{" in self.system else self.system,
            "user": self.user.format(**kwargs) if "{" in self.user else self.user
        }


class MedicalPrompts:
    """Tập hợp các medical prompts"""

    # ==================== PROMPT 1: RADIOLOGY REPORT (MAIN) ====================
    RADIOLOGY_REPORT = PromptTemplate(
        name="radiology_report",
        system="""Generate a concise English chest-X-ray report from the supplied evidence and any attached query image.

Input contract:
In v5 data, f is a list of at most six selected-label dictionaries. Each has
l=pathology label, p=classifier selection score rounded for display,
o optionally identifies a non-DenseNet score source; scores are not calibrated diagnostic probabilities,
t=frozen label-specific threshold from the same score source, s=retrieval status
(S=supported, M=mixed, D=non-support-dominant, I=insufficient),
m=M@5 support-minus-non-support mean similarity,
e=summary of retrieval strength (strong_support, weak_support, ambiguous,
non_support_favored, insufficient), and c=margin stability (stable/unstable).
a is the best support reference and n the best non-support reference.
In each, q=final retrieval score, f=short reference finding, z=reference
report polarity (positive, explicit_negative, surrogate_negative, unmentioned,
uncertain, conflicted). a.u is support strength. The optional g contains one
global reference q/f when f is empty. Missing reference terms do not establish
absence; a non-support reference is not necessarily an explicit negative.
Every a.f, n.f and g.f describes a REFERENCE case, not the query image.
s/m/q/u/e/c are retrieval evidence, never a diagnostic verdict; z describes
only the reference report. No value confirms or excludes query pathology.
Historical v4 arrays use [label, probability, status, support_problem_terms]
and M/D/I append non-support problem terms.

Assessment:
For every compact item, the label has already passed its classifier selection threshold.
Before composing the final JSON, independently inspect the query image for
that label's compatible radiographic signs.
Assess every supplied label independently, including rows after the first three;
probabilities are rounded. Do not apply a new 0.5 cutoff or treat them as diagnostic certainty.
Use the output schema's permitted labels; do not add labels from reference problem lists.

S/M/D/I, m, e, c, q, u, and reference findings describe retrieved reference
studies only. They are not a diagnostic verdict for the query. In particular,
when s is M or D, do not reject a selected label solely because non-support
references rank well. Resolve the disagreement from visible query-image signs.

Include a selected label in problems only if the query image supports it;
omit it if the image is not compatible. Do not copy laterality, severity, or
diagnosis from a reference case.
Key radiographic cue: for Pneumothorax, look for a visceral pleural line with
absent peripheral lung markings and apical or lateral hyperlucency; do not state
these signs are present unless visible on the query image.
If the image is unclear, do not treat uncertainty as either confirmation or definite absence.
Without an image, use only the supplied evidence and do not claim visual inspection.
Do not copy location, severity, chronicity, or other details from reference cases.
Do not invent findings or mention the data source in the report.

Output:
Return exactly one JSON object with problems, findings, and impression; no Markdown,
extra keys, or step-by-step reasoning.
`problems`: include all supported labels permitted by the schema, with no duplicates.
The target vocabulary is Atelectasis, Cardiomegaly, Edema, Pleural Effusion,
Pneumonia, Pneumothorax. Use exactly ["normal"] only when no permitted target
pathology is supported. This fallback does not establish an entirely normal image.
Never mix normal with pathology labels or use an empty list, synonyms, or anatomy/device labels.
`findings`: 1-3 concise sentences describing supported observations. Preserve uncertainty
when necessary; do not add recommendations or unsupported negative findings.
Non-target observations belong here only when supported by the query image.
`impression`: one concise conclusion consistent with findings.
Check that every problems label is supported by the report and no label is both
affirmed and negated across the three fields. Do not convert uncertain possibilities
into definite problems merely to fill the schema.
Format example only (not case content):
{{"problems":["Cardiomegaly"],"findings":"Mild enlargement of the cardiac silhouette.","impression":"Mild cardiomegaly."}}""",
        user="Data:{evidence}\nReturn the report JSON:"
    )

    RADIOLOGY_QUERY_FIRST = PromptTemplate(
        name="radiology_query_first",
        system=RADIOLOGY_REPORT.system.replace("1-3 concise sentences", "2-5 concise sentences") + """

Image-first verification:
Before using reference descriptions, examine the query image independently.
Check each permitted pathology against the visible query findings; do not let a
negative reference determine the query decision. For a selected Pneumothorax or
Pneumonia, verify the compatible visible signs independently even when s is D or M.
Do not promote an equivocal sign to a definite diagnosis. Describe supported
anatomy, location, laterality, and uncertainty explicitly when visible; omit
details that cannot be established from this query. Then use reference evidence
only as context. Return the required JSON only, without the verification steps.""",
        user="Data:{evidence}\nReturn the report JSON:"
    )

    # ==================== PROMPT 1B: RADIOLOGY RESCUE ABLATION ====================
    RADIOLOGY_RESCUE = PromptTemplate(
        name="radiology_rescue",
        system="""You receive the original query chest radiograph, classifier information for
all six target pathologies in clf, and retrieved reference evidence in f.

clf.l is the target label; p is the classifier score; t is its frozen
decision threshold; selected records the classifier's original decision.
retrieved indicates whether label-specific reference evidence is supplied.
Classifier scores are not calibrated diagnostic probabilities. A score
below t does not establish absence of disease; a score above t does not
establish its presence. Do not invent a new probability cutoff.

Assess the six target pathologies independently using the query image.
Use supplied reference evidence only as contextual support or contrast.
Do not copy a reference finding into the query report, and do not treat
an unmentioned reference label as an explicit negative.

A classifier-unselected label may be included only if it is permitted by
the response schema and the query image provides sufficiently clear,
label-specific evidence. A near-threshold score, a similar reference,
or the lack of an explicit negative is not sufficient by itself.
Conversely, do not exclude a permitted label solely because its score
is below t or because no label-specific reference evidence is supplied.
Preserve uncertainty in findings/impression; do not turn an uncertain
appearance into a definite problems label merely to increase recall.

Return only the required JSON object with problems, findings, impression.
All problems must belong to the supplied schema. Use normal alone when
no permitted target pathology is supported; it does not assert that the
entire radiograph is normal. Never mix normal with pathology labels.
Keep findings and impression clinically consistent with the reported labels.
Do not output reasoning traces, reference IDs, scores, or audit fields.""",
        user="Data:{evidence}\nReturn the report JSON:"
    )

    # ==================== PROMPT 2: BASIC DIAGNOSIS ====================
    BASIC_DIAGNOSIS = PromptTemplate(
        name="basic_diagnosis",
        system="""You are an expert radiologist analyzing medical images.
Provide clear, evidence-based diagnostic recommendations.
Format your response with:
- Primary Diagnosis
- Confidence Level
- Supporting Evidence
- Clinical Recommendations""",
        user="""Analyze this chest X-ray and provide diagnostic assessment.

{evidence}

Provide your diagnostic assessment."""
    )

    # ==================== PROMPT 3: DETAILED ANALYSIS ====================
    DETAILED_ANALYSIS = PromptTemplate(
        name="detailed_analysis",
        system="""You are a senior radiologist with 20+ years of experience.
Provide comprehensive, detailed analysis of medical images.
Consider differential diagnoses and clinical context.
Include evidence from retrieval system and confidence levels.""",
        user="""Please provide a detailed radiological analysis.

{evidence}

Please provide:
1. Primary and differential diagnoses
2. Confidence assessment
3. Supporting radiological findings
4. Clinical recommendations
5. Risk factors and considerations"""
    )

    # ==================== PROMPT 4: CLINICAL REASONING ====================
    CLINICAL_REASONING = PromptTemplate(
        name="clinical_reasoning",
        system="""You are an AI physician assistant with expertise in medical reasoning.
Analyze the evidence systematically and provide step-by-step reasoning.
Consider all supporting and contradicting evidence.
Provide probability estimates where appropriate.""",
        user="""Apply clinical reasoning to this medical case:

{evidence}

Answer these questions:
1. What is the most likely diagnosis and why?
2. What is the confidence level in this diagnosis?
3. What evidence supports or contradicts it?
4. What follow-up tests or clinical actions are recommended?
5. Are there any urgent concerns?"""
    )

    # ==================== PROMPT 5: TREATMENT PLANNING ====================
    TREATMENT_PLANNING = PromptTemplate(
        name="treatment_planning",
        system="""You are a clinical decision support system.
Based on radiological findings and evidence, suggest treatment plans.
Consider evidence-based guidelines and patient safety.""",
        user="""Suggest a treatment plan based on this analysis:

{evidence}

Please recommend:
1. Immediate interventions needed (if any)
2. First-line treatment options
3. Follow-up imaging schedule
4. Monitoring parameters
5. Expected outcomes and timeline"""
    )

    # ==================== PROMPT 6: RISK ASSESSMENT ====================
    RISK_ASSESSMENT = PromptTemplate(
        name="risk_assessment",
        system="""You are a medical risk assessment specialist.
Analyze the radiological findings to assess clinical risk.
Consider severity, comorbidities, and potential complications.""",
        user="""Assess the clinical risk for this patient:

{evidence}

Provide risk assessment for:
1. Acute complications (0-7 days)
2. Short-term complications (1-4 weeks)
3. Long-term complications (1-6 months)
4. Mortality risk estimation
5. Recommended monitoring frequency"""
    )

    # ==================== PROMPT 7: QUALITY ASSURANCE ====================
    QUALITY_ASSURANCE = PromptTemplate(
        name="quality_assurance",
        system="""You are a quality assurance expert in radiology.
Verify the accuracy of the analysis and flag any concerns.
Check for inconsistencies or contradictions in evidence.""",
        user="""Perform quality assurance check:

{evidence}

Please check for:
1. Consistency between classifier and retrieval evidence
2. Any contradictions or concerns
3. Confidence in the analysis
4. Areas needing additional investigation
5. Quality score (1-10) and recommendations"""
    )

    @staticmethod
    def get_prompt(name: str) -> PromptTemplate:
        """Get prompt by name"""
        prompts = {
            "radiology": MedicalPrompts.RADIOLOGY_REPORT,
            "radiology_query_first": MedicalPrompts.RADIOLOGY_QUERY_FIRST,
            "radiology_rescue": MedicalPrompts.RADIOLOGY_RESCUE,
            "basic": MedicalPrompts.BASIC_DIAGNOSIS,
            "detailed": MedicalPrompts.DETAILED_ANALYSIS,
            "reasoning": MedicalPrompts.CLINICAL_REASONING,
            "treatment": MedicalPrompts.TREATMENT_PLANNING,
            "risk": MedicalPrompts.RISK_ASSESSMENT,
            "qa": MedicalPrompts.QUALITY_ASSURANCE,
        }
        return prompts.get(name, MedicalPrompts.RADIOLOGY_REPORT)

    @staticmethod
    def list_prompts() -> Dict[str, str]:
        """List available prompts"""
        return {
            "radiology": "Radiology Report Generation (RECOMMENDED)",
            "radiology_query_first": "Query-image-first verification experiment",
            "radiology_rescue": "Radiology All-Label Image Rescue Ablation",
            "basic": "Basic Diagnosis",
            "detailed": "Detailed Analysis",
            "reasoning": "Clinical Reasoning",
            "treatment": "Treatment Planning",
            "risk": "Risk Assessment",
            "qa": "Quality Assurance",
        }

    @staticmethod
    def get_output_schema(
        name: str, allowed_problem_labels: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any] | None:
        """Return a strict machine-readable output contract when the prompt has one."""
        if name in {"radiology", "radiology_query_first", "radiology_rescue"}:
            return radiology_output_schema(allowed_problem_labels)
        return None


class CustomPromptBuilder:
    """Fluent builder cho custom prompts"""
    
    def __init__(self):
        self.system_prompt = ""
        self.user_prompt = ""
        self.evidence_section = ""

    def set_system_prompt(self, prompt: str) -> "CustomPromptBuilder":
        """Set system prompt"""
        self.system_prompt = prompt
        return self

    def set_user_prompt(self, prompt: str) -> "CustomPromptBuilder":
        """Set user prompt"""
        self.user_prompt = prompt
        return self

    def add_evidence_section(self, evidence: Any) -> "CustomPromptBuilder":
        """Add evidence section"""
        if isinstance(evidence, dict):
            self.evidence_section = json.dumps(evidence, indent=2, ensure_ascii=False)
        else:
            self.evidence_section = str(evidence)
        return self

    def build(self) -> PromptTemplate:
        """Build prompt template"""
        return PromptTemplate(
            name="custom",
            system=self.system_prompt,
            user=self.user_prompt
        )


# ==================== HELPER FUNCTIONS ====================

def format_classifier_info(clf_data: Dict) -> str:
    """Format classifier predictions"""
    if not clf_data:
        return "No classifier predictions"
    
    lines = ["Classifier Predictions:"]
    for label, prob in clf_data.items():
        confidence = "High" if prob > 0.8 else "Medium" if prob > 0.5 else "Low"
        lines.append(f"  - {label}: {prob:.2f} ({confidence})")
    return "\n".join(lines)


def format_retrieval_info(ret_data: Dict) -> str:
    """Format retrieval evidence"""
    if not ret_data or "hits" not in ret_data:
        return "No retrieval evidence"
    
    lines = [f"Retrieved Cases (Model: {ret_data.get('model', 'unknown')}):"]
    hits = ret_data.get("hits", [])
    
    for i, hit in enumerate(hits[:5], 1):  # Top 5
        similarity = hit.get("sim", 0)
        problems = hit.get("problems", [])
        lines.append(f"  {i}. Similarity: {similarity:.2f} | Problems: {', '.join(problems)}")
    
    return "\n".join(lines)


def format_label_evidence(lbl_data: Dict) -> str:
    """Format label-specific evidence"""
    if not lbl_data:
        return "No label-specific evidence"
    
    lines = ["Label-Specific Evidence:"]
    for label, evidence in lbl_data.items():
        supporting = evidence.get("supporting", [])
        non_supporting = evidence.get("non_supporting", [])
        
        lines.append(f"  {label}:")
        if supporting:
            lines.append(f"    Supporting: {supporting[:2]}")
        if non_supporting:
            lines.append(f"    Non-supporting: {non_supporting[:2]}")
    
    return "\n".join(lines)


def build_prompt_with_evidence(
    template: PromptTemplate,
    evidence: Dict[str, Any],
    include_examples: bool = False,
) -> Dict[str, str]:
    """Build one compact prompt. Evidence is never duplicated as API context."""
    del include_examples  # Kept only so old callers do not break.
    evidence_text = json.dumps(evidence, ensure_ascii=False, separators=(",", ":"))
    return template.format(evidence=evidence_text)


# ==================== MAIN & TESTING ====================

if __name__ == "__main__":
    print("=" * 70)
    print("📝 AVAILABLE PROMPTS")
    print("=" * 70)
    
    for name, desc in MedicalPrompts.list_prompts().items():
        print(f"  • {name:15} - {desc}")
    
    print("\n" + "=" * 70)
    print("🧪 PROMPT EXAMPLE")
    print("=" * 70)
    
    # Get radiology prompt
    prompt = MedicalPrompts.get_prompt("radiology")
    print(f"\nPrompt: {prompt.name}")
    print(f"\nSystem:\n{prompt.system[:200]}...")
    print(f"\nUser:\n{prompt.user[:200]}...")
    
    print("\n" + "=" * 70)
    print("✅ Prompts loaded successfully!")
    print("=" * 70)
    print("\nUsage in pipeline:")
    print("  from src.prompts import MedicalPrompts")
    print("  prompt = MedicalPrompts.get_prompt('radiology')")
    print("  response = llm.send_request(prompt.user)")
