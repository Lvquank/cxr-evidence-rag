"""Create a compact single-case DOCX report from full-pipeline evidence."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence


ROOT = Path(__file__).resolve().parents[1]


def _require_docx() -> None:
    try:
        import docx  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("DOCX report generation requires python-docx. Install requirements.txt first.") from exc


def _text(value: Any, empty: str = "Not available") -> str:
    if value is None or value == "":
        return empty
    if isinstance(value, (list, tuple)):
        return ", ".join(str(item) for item in value) or empty
    return str(value)


def _format_score(value: Any) -> str:
    return f"{float(value):.4f}" if isinstance(value, (int, float)) else "Not available"


def _format_probability(value: Any) -> str:
    return f"{100.0 * float(value):.2f}%" if isinstance(value, (int, float)) else "Not available"


def _resolve_image_path(raw_path: Any, query_image: Optional[Path] = None) -> Optional[Path]:
    """Resolve absolute and retriever-relative image paths without assuming one layout."""
    if not raw_path:
        return None
    candidate = Path(str(raw_path)).expanduser()
    candidates = [candidate]
    if not candidate.is_absolute():
        candidates.extend((
            ROOT / candidate,
            ROOT / "data" / "reprocessed" / candidate,
            ROOT / "data" / "reprocessed" / "images" / candidate.name,
        ))
        if query_image:
            candidates.append(query_image.parent / candidate.name)
    for path in candidates:
        if path.is_file():
            return path.resolve()
    return None


def _candidate_score(candidate: Dict[str, Any]) -> float:
    for key in ("final_rank_score", "densenet_local_similarity", "cosine_similarity", "base_rank_score"):
        value = candidate.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return float("-inf")


def _best_candidate(candidates: Sequence[Dict[str, Any]], exclude_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Select the strongest reference, optionally excluding a reference ID."""
    choices = [
        candidate for candidate in candidates
        if isinstance(candidate, dict) and str(candidate.get("id")) != str(exclude_id)
    ]
    if not choices:
        return None
    return max(choices, key=lambda candidate: (_candidate_score(candidate), -int(candidate.get("rank", 999999))))


def _set_cell_shading(cell: Any, fill: str) -> None:
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn

    properties = cell._tc.get_or_add_tcPr()
    shading = OxmlElement("w:shd")
    shading.set(qn("w:fill"), fill)
    properties.append(shading)


def _set_cell_width(cell: Any, width_twips: int) -> None:
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn

    properties = cell._tc.get_or_add_tcPr()
    width = OxmlElement("w:tcW")
    width.set(qn("w:w"), str(width_twips))
    width.set(qn("w:type"), "dxa")
    properties.append(width)


def _set_table_layout_fixed(table: Any, widths_twips: Sequence[int]) -> None:
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn

    table.autofit = False
    layout = OxmlElement("w:tblLayout")
    layout.set(qn("w:type"), "fixed")
    table._tbl.tblPr.append(layout)
    for row in table.rows:
        for cell, width in zip(row.cells, widths_twips):
            _set_cell_width(cell, width)


def _set_repeat_table_header(row: Any) -> None:
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn

    repeat = OxmlElement("w:tblHeader")
    repeat.set(qn("w:val"), "true")
    row._tr.get_or_add_trPr().append(repeat)


def _add_label(paragraph: Any, label: str, value: Any) -> None:
    from docx.shared import Pt

    run = paragraph.add_run(f"{label}: ")
    run.bold = True
    run.font.size = Pt(9)
    value_run = paragraph.add_run(_text(value))
    value_run.font.size = Pt(9)


def _add_picture_with_alt(paragraph: Any, image_path: Path, width: Any, description: str) -> None:
    """Add an inline image with Word alternative text for screen readers."""
    shape = paragraph.add_run().add_picture(str(image_path), width=width)
    shape._inline.docPr.set("descr", description)
    shape._inline.docPr.set("title", description)


def _add_reference_image(cell: Any, candidate: Optional[Dict[str, Any]], query_image: Optional[Path], role: str) -> None:
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.shared import Inches, Pt

    paragraph = cell.paragraphs[0]
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    if candidate is None:
        paragraph.add_run(f"No distinct {role}\nreference")
        return
    image_path = _resolve_image_path(candidate.get("image_path"), query_image)
    if image_path:
        _add_picture_with_alt(
            paragraph,
            image_path,
            Inches(1.15),
            f"{role} reference chest X-ray {image_path.name}",
        )
        caption = cell.add_paragraph(image_path.name)
        caption.alignment = WD_ALIGN_PARAGRAPH.CENTER
        for run in caption.runs:
            run.italic = True
            run.font.size = Pt(7)
    else:
        paragraph.add_run("Reference image unavailable")


def _add_reference_facts(cell: Any, candidate: Optional[Dict[str, Any]], role: str) -> None:
    from docx.shared import Pt

    paragraph = cell.add_paragraph()
    if candidate is None:
        paragraph.add_run(f"{role}: no distinct retrieved case")
        return
    problems = list(dict.fromkeys(candidate.get("problems") or []))[:3]
    facts = (
        f"{role}: Study {_text(candidate.get('study_id'), '?')} | Retrieval rank {_text(candidate.get('rank'), '?')}",
        f"Similarity: final {_format_score(candidate.get('final_rank_score'))}; global {_format_score(candidate.get('cosine_similarity'))}; local {_format_score(candidate.get('densenet_local_similarity'))}",
        f"Annotated problems: {_text(problems, 'None')}",
    )
    for index, text in enumerate(facts):
        if index:
            paragraph.add_run("\n")
        run = paragraph.add_run(text)
        run.font.size = Pt(7.5)


def _add_label_evidence_row(
    document: Any,
    label: str,
    context: Optional[Dict[str, Any]],
    probability: Any,
    query_image: Optional[Path],
) -> None:
    from docx.shared import Pt, RGBColor

    status = _text((context or {}).get("evidence_status"), "No evidence")
    margin = _format_score(((context or {}).get("margins") or {}).get("M@5"))
    heading = document.add_paragraph()
    heading.paragraph_format.space_before = Pt(3)
    heading.paragraph_format.space_after = Pt(1)
    run = heading.add_run(f"{label} ({_format_probability(probability)}) | Retrieval status: {status} | M@5 {margin}")
    run.bold = True
    run.font.size = Pt(9)
    run.font.color.rgb = RGBColor(31, 77, 120)

    support = _best_candidate((context or {}).get("supporting_evidence") or [])
    non_support = _best_candidate(
        (context or {}).get("matched_non_support_evidence") or [],
        exclude_id=str(support.get("id")) if support else None,
    )
    # Intentional two-panel layout: readable support/contrast comparison in portrait orientation.
    panel = document.add_table(rows=1, cols=2)
    panel.style = "Table Grid"
    _set_table_layout_fixed(panel, [5040, 5040])
    cells = panel.rows[0].cells
    for cell in cells:
        _set_cell_shading(cell, "F4F6F9")
    _add_reference_image(cells[0], support, query_image, "Supporting")
    _add_reference_facts(cells[0], support, "Supporting reference")
    _add_reference_image(cells[1], non_support, query_image, "Contrasting")
    _add_reference_facts(cells[1], non_support, "Contrasting reference")


def _compact_text(value: Any, limit: int) -> str:
    text = _text(value, "Not reported")
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."


def _add_report_section(document: Any, prediction: Optional[Dict[str, Any]], raw_response: str) -> None:
    from docx.shared import Pt

    document.add_heading("Radiology report", level=1)
    if not prediction:
        document.add_paragraph(f"Raw model response: {_compact_text(raw_response or 'No response available', 420)}")
        return
    for label, key in (("PROBLEMS", "problems"), ("FINDINGS", "findings"), ("IMPRESSION", "impression")):
        paragraph = document.add_paragraph()
        title = paragraph.add_run(f"{label}: ")
        title.bold = True
        title.font.size = Pt(10)
        limit = 160 if key == "problems" else 260
        paragraph.add_run(_compact_text(prediction.get(key), limit))


def build_case_report_docx(
    evidence: Dict[str, Any],
    prediction: Optional[Dict[str, Any]],
    raw_response: str,
    output_path: Path,
    *,
    study_id: Optional[str] = None,
    route: Optional[Dict[str, Any]] = None,
    timestamp: Optional[str] = None,
) -> Path:
    """Create a DOCX audit report for one query image and return its path."""
    _require_docx()
    from docx import Document
    from docx.enum.table import WD_TABLE_ALIGNMENT
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.oxml.ns import qn
    from docx.shared import Inches, Pt, RGBColor

    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    document = Document()
    section = document.sections[0]
    section.top_margin = Inches(0.45)
    section.bottom_margin = Inches(0.45)
    section.left_margin = Inches(0.75)
    section.right_margin = Inches(0.75)
    normal = document.styles["Normal"]
    normal.font.name = "Calibri"
    normal._element.rPr.rFonts.set(qn("w:ascii"), "Calibri")
    normal._element.rPr.rFonts.set(qn("w:hAnsi"), "Calibri")
    normal.font.size = Pt(8)
    normal.paragraph_format.space_after = Pt(1)
    for style_name, size, color in (("Heading 1", 12, "2E74B5"), ("Heading 2", 10, "1F4D78")):
        style = document.styles[style_name]
        style.font.name = "Calibri"
        style._element.rPr.rFonts.set(qn("w:ascii"), "Calibri")
        style._element.rPr.rFonts.set(qn("w:hAnsi"), "Calibri")
        style.font.size = Pt(size)
        style.font.color.rgb = RGBColor.from_string(color)

    title = document.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = title.add_run("Chest X-ray AI-assisted Research Report")
    run.bold = True
    run.font.size = Pt(15)
    run.font.color.rgb = RGBColor(11, 37, 69)
    title.paragraph_format.space_after = Pt(8)

    query_image = _resolve_image_path(evidence.get("image_path"))
    metadata = document.add_table(rows=1, cols=2)
    metadata.alignment = WD_TABLE_ALIGNMENT.CENTER
    metadata.style = "Table Grid"
    _set_table_layout_fixed(metadata, [2000, 8080])
    left, right = metadata.rows[0].cells
    _set_cell_shading(left, "E8EEF5")
    _set_cell_shading(right, "E8EEF5")
    if query_image:
        image_paragraph = left.paragraphs[0]
        image_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        _add_picture_with_alt(
            image_paragraph,
            query_image,
            Inches(1.1),
            f"Query chest X-ray {query_image.name}",
        )
    else:
        left.paragraphs[0].add_run("Query image unavailable")
    source = right.paragraphs[0]
    source_fields = (
        ("Image file", query_image.name if query_image else evidence.get("image_path")),
        ("Study ID", study_id),
        ("Generated", timestamp),
        ("System route", (route or {}).get("name")),
    )
    for index, (label, value) in enumerate(source_fields):
        if index:
            source.add_run("\n")
        _add_label(source, label, value)

    classifier = evidence.get("classifier") or {}
    selected_labels = list(classifier.get("selected_labels") or [])
    probabilities = classifier.get("raw_probabilities") or {}
    document.add_heading("AI triage: target pathologies above threshold", level=1)
    if not selected_labels:
        document.add_paragraph("No target pathology exceeded its classifier threshold.")
    else:
        probability_line = document.add_paragraph()
        probability_line.paragraph_format.space_after = Pt(2)
        probability_line.add_run(" | ".join(
            f"{label}: {_format_probability(probabilities.get(label))}"
            for label in selected_labels
        ))

    contexts_by_label = {
        str(context.get("label")): context
        for context in ((evidence.get("retriever") or {}).get("label_evidence") or [])
        if isinstance(context, dict)
    }
    for position, label in enumerate(selected_labels, start=1):
        if position == 4:
            document.add_page_break()
        context = contexts_by_label.get(str(label))
        _add_label_evidence_row(document, label, context, probabilities.get(label), query_image)
    _add_report_section(document, prediction, raw_response)

    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    footer_run = footer.add_run("Research use only. This output does not replace radiologist interpretation.")
    footer_run.font.size = Pt(8)
    footer_run.font.color.rgb = RGBColor(100, 100, 100)
    document.save(output_path)
    return output_path
