"""Create a Word-ready appendix describing implemented manuscript revisions."""

from __future__ import annotations

import json
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.table import WD_ALIGN_VERTICAL
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "thesis" / "PhuLuc_CacNoiDungDaCapNhat_DeChenBaiBao_CXR_RAG.docx"


def set_cell_shading(cell, fill: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shading = OxmlElement("w:shd")
    shading.set(qn("w:fill"), fill)
    tc_pr.append(shading)


def set_cell_width(cell, width_inches: float) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    tc_w = tc_pr.find(qn("w:tcW"))
    if tc_w is None:
        tc_w = OxmlElement("w:tcW")
        tc_pr.append(tc_w)
    tc_w.set(qn("w:w"), str(round(width_inches * 1440)))
    tc_w.set(qn("w:type"), "dxa")


def set_repeat_table_header(row) -> None:
    tr_pr = row._tr.get_or_add_trPr()
    header = OxmlElement("w:tblHeader")
    header.set(qn("w:val"), "true")
    tr_pr.append(header)


def set_font(run, size: float | None = None, bold: bool | None = None, color: str | None = None) -> None:
    run.font.name = "Calibri"
    run._element.rPr.rFonts.set(qn("w:ascii"), "Calibri")
    run._element.rPr.rFonts.set(qn("w:hAnsi"), "Calibri")
    if size:
        run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    if color:
        run.font.color.rgb = RGBColor.from_string(color)


def add_text(paragraph, text: str, *, bold: bool = False, italic: bool = False, color: str | None = None) -> None:
    run = paragraph.add_run(text)
    set_font(run, bold=bold, color=color)
    run.italic = italic


def add_heading(document: Document, text: str, level: int = 1) -> None:
    paragraph = document.add_paragraph(style=f"Heading {level}")
    paragraph.add_run(text)


def add_bullet(document: Document, text: str) -> None:
    paragraph = document.add_paragraph(style="List Bullet")
    paragraph.paragraph_format.space_after = Pt(4)
    add_text(paragraph, text)


def add_number(document: Document, text: str) -> None:
    paragraph = document.add_paragraph(style="List Number")
    paragraph.paragraph_format.space_after = Pt(4)
    add_text(paragraph, text)


def add_table(document: Document, headers: list[str], rows: list[list[str]], widths: list[float]) -> None:
    table = document.add_table(rows=1, cols=len(headers))
    table.style = "Table Grid"
    table.autofit = False
    header_cells = table.rows[0].cells
    set_repeat_table_header(table.rows[0])
    for index, (cell, text) in enumerate(zip(header_cells, headers)):
        set_cell_width(cell, widths[index])
        set_cell_shading(cell, "E8EEF5")
        cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        paragraph = cell.paragraphs[0]
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        add_text(paragraph, text, bold=True, color="1F4D78")
    for row_values in rows:
        cells = table.add_row().cells
        for index, (cell, text) in enumerate(zip(cells, row_values)):
            set_cell_width(cell, widths[index])
            cell.vertical_alignment = WD_ALIGN_VERTICAL.TOP
            paragraph = cell.paragraphs[0]
            add_text(paragraph, text)
    document.add_paragraph().paragraph_format.space_after = Pt(2)


def configure(document: Document) -> None:
    section = document.sections[0]
    section.top_margin = Inches(1)
    section.bottom_margin = Inches(1)
    section.left_margin = Inches(1)
    section.right_margin = Inches(1)
    section.header_distance = Inches(0.492)
    section.footer_distance = Inches(0.492)
    styles = document.styles
    normal = styles["Normal"]
    normal.font.name = "Calibri"
    normal._element.rPr.rFonts.set(qn("w:ascii"), "Calibri")
    normal._element.rPr.rFonts.set(qn("w:hAnsi"), "Calibri")
    normal.font.size = Pt(11)
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.line_spacing = 1.25
    for name, size, color, before, after in [
        ("Heading 1", 16, "2E74B5", 18, 10),
        ("Heading 2", 13, "2E74B5", 14, 7),
        ("Heading 3", 12, "1F4D78", 10, 5),
    ]:
        style = styles[name]
        style.font.name = "Calibri"
        style._element.rPr.rFonts.set(qn("w:ascii"), "Calibri")
        style._element.rPr.rFonts.set(qn("w:hAnsi"), "Calibri")
        style.font.size = Pt(size)
        style.font.color.rgb = RGBColor.from_string(color)
        style.font.bold = True
        style.paragraph_format.space_before = Pt(before)
        style.paragraph_format.space_after = Pt(after)
    for name in ("List Bullet", "List Number"):
        style = styles[name]
        style.font.name = "Calibri"
        style._element.rPr.rFonts.set(qn("w:ascii"), "Calibri")
        style._element.rPr.rFonts.set(qn("w:hAnsi"), "Calibri")
        style.font.size = Pt(11)

    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    add_text(footer, "CXR-RAG manuscript revision appendix | 2026-09-05", color="666666")


def ci_cell(point: float, bounds: dict) -> str:
    return f"{point:.3f} [{bounds['lower']:.3f}, {bounds['upper']:.3f}]"


def delta_cell(result: dict) -> str:
    return f"{result['delta']:+.3f} [{result['ci95_lower']:.3f}, {result['ci95_upper']:.3f}]"


def reversed_delta(result: dict) -> dict:
    """Express the same paired difference in the opposite pipeline order."""
    return {
        "delta": -result["delta"],
        "ci95_lower": -result["ci95_upper"],
        "ci95_upper": -result["ci95_lower"],
    }


def main() -> None:
    bootstrap_path = ROOT / "outputs" / "table_iv_target_label_206" / "table_iv_target_label_bootstrap_ci.json"
    data = json.loads(bootstrap_path.read_text(encoding="utf-8"))
    points = data["point_estimates"]
    cis = data["absolute_ci95"]
    deltas = data["paired_differences"]

    document = Document()
    configure(document)
    title = document.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.paragraph_format.space_after = Pt(6)
    title_run = title.add_run("PHỤ LỤC CẬP NHẬT BÀI BÁO CXR-RAG")
    set_font(title_run, size=18, bold=True, color="0B2545")
    subtitle = document.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    subtitle.paragraph_format.space_after = Pt(16)
    add_text(subtitle, "Nội dung đã triển khai, số liệu có thể chèn và các giới hạn bắt buộc phải công bố", italic=True, color="555555")

    add_heading(document, "1. Mục đích và phạm vi", 1)
    p = document.add_paragraph()
    add_text(p, "Tài liệu này tổng hợp các thay đổi kỹ thuật và đánh giá đã được thực hiện trong project sau bản thảo hướng dẫn FAIR2026. ")
    add_text(p, "Nó là phụ lục hỗ trợ cập nhật manuscript, không thay thế việc biên tập trực tiếp bản thảo.", bold=True)
    add_bullet(document, "Cohort end-to-end cố định: 206 study, đánh giá đa nhãn trên sáu pathology mục tiêu và normal.")
    add_bullet(document, "Mô hình sinh báo cáo đã được xác nhận trong manifest/log chạy là gpt-5.6-luna.")
    add_bullet(document, "Khoảng tin cậy dùng paired nonparametric percentile bootstrap, resample theo study_id, 2,000 lần, seed 20260904.")

    add_heading(document, "2. Checklist yêu cầu và trạng thái triển khai", 1)
    checklist = [
        ["Learned evidence gate", "Hoàn thành", "Đã huấn luyện/đánh giá OOF trên 206 study và so sánh trực tiếp với deterministic rule gate."],
        ["Local reranking + lambda sensitivity", "Hoàn thành về thực nghiệm", "Có global-only và các lambda trong sweep; cần chèn kết quả/diễn giải vào manuscript."],
        ["Table IV paired bootstrap", "Hoàn thành", "Có CI tuyệt đối cho bốn pipeline và đủ sáu paired comparison."],
        ["Claim-level CheXbert/RadGraph", "Có output, chưa chốt", "Cần xử lý minh bạch override study 1615 và khôi phục source/dependency để tái lập."],
        ["Pilot reader study", "Chưa thực hiện", "Giữ là future work; không mô tả như một kết quả đã có."],
        ["Abstract/Conclusion/Limitations", "Chưa sửa bản thảo", "Cần viết lại sau khi chốt claim-level evaluation và chèn bảng/CI mới."],
    ]
    add_table(document, ["Hạng mục", "Trạng thái", "Nội dung cần đưa vào bài"], checklist, [1.65, 1.15, 3.70])

    add_heading(document, "3. Cập nhật phương pháp có thể chèn", 1)
    add_heading(document, "3.1. Hybrid retrieval và local reranking", 2)
    p = document.add_paragraph()
    add_text(p, "Công thức thống nhất trong code là: ")
    add_text(p, "S_final = lambda S_global + (1 - lambda) S_local.", bold=True)
    add_text(p, " S_global là similarity của pathology-aware RAD-DINO projection; S_local là điểm matching hai chiều từ DenseNet spatial feature map. Local reranking chỉ áp dụng trên top-50 ứng viên global, sau đó khử trùng study ID và lấy top-k.")
    add_heading(document, "3.2. Learned evidence gate", 2)
    p = document.add_paragraph()
    add_text(p, "Gate học được thay thế quyết định ngưỡng rule-based bằng logistic-regression OOF theo nhãn. Đầu vào gồm xác suất DenseNet, thống kê top-5 support/non-support (similarity, số lượng) và pathology identity. Gate chỉ điều khiển evidence state/đường đưa thông tin vào LLM; không thay đổi classifier hay gallery retrieval.")
    add_heading(document, "3.3. Thống kê end-to-end", 2)
    p = document.add_paragraph()
    add_text(p, "Mỗi bootstrap draw resample cùng một danh sách study_id cho mọi pipeline. Vì vậy CI của delta là paired CI, phù hợp để diễn giải chênh lệch giữa các pipeline trên đúng cohort 206 study. ")
    add_text(p, "Sample F1", bold=True)
    add_text(p, " được định nghĩa là trung bình F1 theo từng study, không phải macro-F1 trung bình theo sáu pathology.")

    add_heading(document, "4. Table IV mở rộng: kết quả có thể chèn", 1)
    p = document.add_paragraph()
    add_text(p, "Bảng dưới đây có point estimate và percentile 95% CI. Groundedness = pooled TP/(TP+FP); unmatched rate = pooled FP/(TP+FP).", italic=True)
    table_rows = []
    display_names = {
        "structured_rule": "Structured (rule gate)",
        "structured_learned_gate": "Structured (learned gate)",
        "baseline_a_classifier_vision": "A: Classifier + vision",
        "baseline_b_classifier_retriever_vision": "B: Classifier + retrieval + vision",
    }
    for name in display_names:
        table_rows.append([
            display_names[name],
            ci_cell(points[name]["precision"], cis[name]["precision"]),
            ci_cell(points[name]["recall"], cis[name]["recall"]),
            ci_cell(points[name]["micro_f1"], cis[name]["micro_f1"]),
            ci_cell(points[name]["sample_f1"], cis[name]["sample_f1"]),
            ci_cell(points[name]["label_groundedness"], cis[name]["label_groundedness"]),
            ci_cell(points[name]["unmatched_rate"], cis[name]["unmatched_rate"]),
            ci_cell(points[name]["normal_f1"], cis[name]["normal_f1"]),
        ])
    add_table(
        document,
        ["Pipeline", "P [CI]", "R [CI]", "Micro F1 [CI]", "Sample F1 [CI]", "Grounded [CI]", "Unmatched [CI]", "Normal F1 [CI]"],
        table_rows,
        [1.30, 0.74, 0.74, 0.87, 0.87, 0.87, 0.87, 0.87],
    )

    add_heading(document, "5. Diễn giải paired CI nên dùng trong Results", 1)
    p = document.add_paragraph()
    add_text(p, "Chênh lệch được tính theo thứ tự pipeline ở cột trái trừ pipeline ở cột phải. CI không chứa 0 là bằng chứng bootstrap về khác biệt có hướng; CI chứa 0 không chứng minh tương đương, chỉ cho thấy dữ liệu hiện tại chưa xác lập được hướng chênh lệch với mức CI này.", italic=True)
    comparison_rows = [
        ["Learned gate − rule gate", *[delta_cell(reversed_delta(deltas["structured_rule_minus_structured_learned_gate"][metric])) for metric in ("precision", "micro_f1", "sample_f1")]],
        ["B − A", *[delta_cell(reversed_delta(deltas["baseline_a_classifier_vision_minus_baseline_b_classifier_retriever_vision"][metric])) for metric in ("precision", "micro_f1", "sample_f1")]],
        ["B − structured rule", *[delta_cell(reversed_delta(deltas["structured_rule_minus_baseline_b_classifier_retriever_vision"][metric])) for metric in ("precision", "micro_f1", "sample_f1")]],
    ]
    add_table(document, ["Comparison", "Precision delta [CI]", "Micro F1 delta [CI]", "Sample F1 delta [CI]"], comparison_rows, [1.65, 1.62, 1.62, 1.61])
    add_bullet(document, "Learned gate tăng precision/groundedness so với rule gate: +0.064 [0.017, 0.108], đồng thời giảm recall; Micro F1 chưa có CI tách 0.")
    add_bullet(document, "B có point estimate Micro F1 và Sample F1 cao nhất; B−A Sample F1 = +0.028 [0.006, 0.052], còn Micro F1 CI vẫn sát/chứa 0: +0.021 [-0.001, 0.046].")
    add_bullet(document, "Không dùng cụm từ 'statistically superior' cho các delta CI chứa 0. Mô tả đúng là 'higher point estimate, but paired CI included zero'.")

    add_heading(document, "6. Đoạn tiếng Anh gợi ý để chèn manuscript", 1)
    add_heading(document, "6.1. Methods — learned gate", 2)
    p = document.add_paragraph()
    add_text(p, "“In addition to the deterministic evidence rule, we evaluated an out-of-fold learned evidence gate. The logistic model used the classifier probability, top-5 support/non-support retrieval statistics, and pathology identity to determine the evidence state. It controls the evidence presented to the report generator but does not modify the classifier or retrieval gallery.”", italic=True)
    add_heading(document, "6.2. Methods — paired bootstrap", 2)
    p = document.add_paragraph()
    add_text(p, "“For end-to-end comparisons, we used a paired nonparametric percentile bootstrap at the study level (2,000 resamples; 95% confidence intervals; seed 20260904). Each resample used the same study IDs across pipelines. Sample F1 denotes the mean of the 206 per-study multilabel F1 values, rather than a macro-average over pathologies.”", italic=True)
    add_heading(document, "6.3. Results — calibrated interpretation", 2)
    p = document.add_paragraph()
    add_text(p, "“Baseline B had the highest point estimates for micro F1 (0.642) and sample F1 (0.651). Relative to Baseline A, its paired sample-F1 difference was +0.028 (95% CI 0.006 to 0.052), whereas the paired micro-F1 interval included zero (+0.021, 95% CI -0.001 to 0.046). The learned gate improved precision over the rule gate (+0.064, 95% CI 0.017 to 0.108), but its micro-F1 difference was not resolved by the paired interval.”", italic=True)
    add_heading(document, "6.4. Limitations — clinical metric reference", 2)
    p = document.add_paragraph()
    add_text(p, "“For the claim-level clinical-metric workflow, study 1615 had empty source findings and impression fields despite a normal label. It was retained under a documented normal-reference override. This sentence is an evaluation convention rather than a recovered original report; analyses using this convention should be disclosed and complemented by a sensitivity analysis excluding the study.”", italic=True)

    add_heading(document, "7. Việc còn lại trước khi nộp", 1)
    for text in [
        "Rerun/restore clinical-metric workflow with versioned scripts, config, dependencies and checkpoints; export a primary 205-study analysis that excludes study 1615, then report the documented 206-study override analysis as sensitivity.",
        "Chèn Table IV mở rộng hoặc tách bảng point estimate và bảng paired delta để tránh bảng IEEE quá rộng.",
        "Viết lại Abstract, Results, Limitations và Conclusion theo số liệu cuối cùng; loại mọi câu nói bootstrap/claim-level evaluation chưa có nếu chưa tái lập được workflow.",
        "Giữ reader study và external validation là future work, trừ khi có protocol và dữ liệu đánh giá thực tế.",
    ]:
        add_number(document, text)

    add_heading(document, "8. Nguồn artifact trong project", 1)
    sources = [
        "outputs/table_iv_target_label_206/table_iv_target_label_bootstrap_ci.json (point estimate, absolute CI, paired CI)",
        "outputs/table_iv_target_label_206/table_iv_target_label_full.md (Table IV mở rộng)",
        "src/learned_gate_oof_206/metrics/gate_metrics.json và end_to_end_metrics.json",
        "outputs/flow1/lambda_sweep_results.json và outputs/flow1/raw_rad_dino_densenet/lambda_sweep_results.json",
        "CLINICAL_METRICS_RUN_SUMMARY.md (policy override study 1615; cần được công bố minh bạch)",
    ]
    for source in sources:
        add_bullet(document, source)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    document.save(OUT)
    print(OUT)


if __name__ == "__main__":
    main()
