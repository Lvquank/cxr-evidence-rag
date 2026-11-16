"""Create a Word brief comparing the current retriever with the proposed strategy."""

from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


OUT_PATH = Path("docs/Tom_tat_So_sanh_Chien_luoc_Retriever.docx")
BLUE = "2E74B5"
DARK_BLUE = "1F4D78"
INK = "0B2545"
MUTED = "667085"
HEADER_FILL = "F2F4F7"
CALLOUT_FILL = "F4F6F9"
CAUTION_FILL = "FFF8E8"
TABLE_WIDTH_DXA = 9360
TABLE_INDENT_DXA = 120


def set_run_font(run, size=11, bold=False, color=INK, italic=False):
    run.font.name = "Calibri"
    run._element.rPr.rFonts.set(qn("w:ascii"), "Calibri")
    run._element.rPr.rFonts.set(qn("w:hAnsi"), "Calibri")
    run.font.size = Pt(size)
    run.bold = bold
    run.italic = italic
    run.font.color.rgb = RGBColor.from_string(color)


def set_cell_shading(cell, fill):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = tc_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tc_pr.append(shd)
    shd.set(qn("w:fill"), fill)


def set_cell_margins(cell, top=80, start=120, bottom=80, end=120):
    tc_pr = cell._tc.get_or_add_tcPr()
    tc_mar = tc_pr.first_child_found_in("w:tcMar")
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for name, value in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        node = tc_mar.find(qn(f"w:{name}"))
        if node is None:
            node = OxmlElement(f"w:{name}")
            tc_mar.append(node)
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")


def set_table_geometry(table, widths_dxa):
    table.autofit = False
    table.alignment = WD_TABLE_ALIGNMENT.LEFT
    tbl_pr = table._tbl.tblPr
    tbl_w = tbl_pr.first_child_found_in("w:tblW")
    if tbl_w is None:
        tbl_w = OxmlElement("w:tblW")
        tbl_pr.append(tbl_w)
    tbl_w.set(qn("w:w"), str(sum(widths_dxa)))
    tbl_w.set(qn("w:type"), "dxa")
    tbl_ind = tbl_pr.first_child_found_in("w:tblInd")
    if tbl_ind is None:
        tbl_ind = OxmlElement("w:tblInd")
        tbl_pr.append(tbl_ind)
    tbl_ind.set(qn("w:w"), str(TABLE_INDENT_DXA))
    tbl_ind.set(qn("w:type"), "dxa")
    grid = table._tbl.tblGrid
    for column, width in zip(grid.gridCol_lst, widths_dxa):
        column.set(qn("w:w"), str(width))
    for row in table.rows:
        for cell, width in zip(row.cells, widths_dxa):
            cell.width = Inches(width / 1440)
            tc_w = cell._tc.tcPr.tcW
            tc_w.set(qn("w:w"), str(width))
            tc_w.set(qn("w:type"), "dxa")
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            set_cell_margins(cell)


def set_cell_text(cell, text, bold=False, color=INK, size=9.5):
    cell.text = ""
    paragraph = cell.paragraphs[0]
    paragraph.paragraph_format.space_before = Pt(0)
    paragraph.paragraph_format.space_after = Pt(0)
    paragraph.paragraph_format.line_spacing = 1.08
    run = paragraph.add_run(text)
    set_run_font(run, size=size, bold=bold, color=color)


def add_table(doc, headers, rows, widths_dxa):
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = "Table Grid"
    set_table_geometry(table, widths_dxa)
    for cell, text in zip(table.rows[0].cells, headers):
        set_cell_shading(cell, HEADER_FILL)
        set_cell_text(cell, text, bold=True, color=DARK_BLUE)
    for row_data in rows:
        cells = table.add_row().cells
        for cell, text in zip(cells, row_data):
            set_cell_text(cell, text)
    doc.add_paragraph().paragraph_format.space_after = Pt(1)
    return table


def add_body(doc, text, before=0, after=6):
    paragraph = doc.add_paragraph(style="Normal")
    paragraph.paragraph_format.space_before = Pt(before)
    paragraph.paragraph_format.space_after = Pt(after)
    paragraph.paragraph_format.line_spacing = 1.1
    set_run_font(paragraph.add_run(text))
    return paragraph


def add_label_paragraph(doc, label, text, color=INK):
    paragraph = doc.add_paragraph(style="Normal")
    paragraph.paragraph_format.space_after = Pt(6)
    paragraph.paragraph_format.line_spacing = 1.1
    set_run_font(paragraph.add_run(label), bold=True, color=color)
    set_run_font(paragraph.add_run(text), color=color)
    return paragraph


def add_callout(doc, title, text, fill=CALLOUT_FILL, color=DARK_BLUE):
    table = doc.add_table(rows=1, cols=1)
    table.style = "Table Grid"
    set_table_geometry(table, [TABLE_WIDTH_DXA])
    cell = table.cell(0, 0)
    set_cell_shading(cell, fill)
    cell.text = ""
    paragraph = cell.paragraphs[0]
    paragraph.paragraph_format.space_after = Pt(2)
    run = paragraph.add_run(title)
    set_run_font(run, size=10.5, bold=True, color=color)
    paragraph = cell.add_paragraph()
    paragraph.paragraph_format.space_after = Pt(0)
    paragraph.paragraph_format.line_spacing = 1.08
    set_run_font(paragraph.add_run(text), size=10, color=INK)
    doc.add_paragraph().paragraph_format.space_after = Pt(1)


def add_page_number(paragraph):
    run = paragraph.add_run()
    field = OxmlElement("w:fldSimple")
    field.set(qn("w:instr"), "PAGE")
    run._r.addnext(field)


def configure_document(doc):
    section = doc.sections[0]
    section.top_margin = Inches(1)
    section.bottom_margin = Inches(1)
    section.left_margin = Inches(1)
    section.right_margin = Inches(1)
    section.header_distance = Inches(0.492)
    section.footer_distance = Inches(0.492)

    normal = doc.styles["Normal"]
    normal.font.name = "Calibri"
    normal._element.rPr.rFonts.set(qn("w:ascii"), "Calibri")
    normal._element.rPr.rFonts.set(qn("w:hAnsi"), "Calibri")
    normal.font.size = Pt(11)
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.line_spacing = 1.1

    title = doc.styles["Title"]
    title.font.name = "Calibri"
    title.font.size = Pt(22)
    title.font.color.rgb = RGBColor.from_string(INK)
    title.font.bold = True
    title.paragraph_format.space_before = Pt(12)
    title.paragraph_format.space_after = Pt(4)

    subtitle = doc.styles["Subtitle"]
    subtitle.font.name = "Calibri"
    subtitle.font.size = Pt(12)
    subtitle.font.color.rgb = RGBColor.from_string(MUTED)
    subtitle.paragraph_format.space_after = Pt(14)

    for style_name, size, color, before, after in (
        ("Heading 1", 16, BLUE, 16, 8),
        ("Heading 2", 13, BLUE, 12, 6),
        ("Heading 3", 12, DARK_BLUE, 8, 4),
    ):
        style = doc.styles[style_name]
        style.font.name = "Calibri"
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = RGBColor.from_string(color)
        style.paragraph_format.space_before = Pt(before)
        style.paragraph_format.space_after = Pt(after)
        style.paragraph_format.keep_with_next = True

    header = section.header.paragraphs[0]
    header.alignment = WD_ALIGN_PARAGRAPH.LEFT
    header.paragraph_format.space_after = Pt(0)
    set_run_font(header.add_run("TECHNICAL BRIEF | IU-Xray Evidence Retrieval"), size=8.5, bold=True, color=MUTED)

    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    footer.paragraph_format.space_before = Pt(0)
    set_run_font(footer.add_run("Trang "), size=8.5, color=MUTED)
    add_page_number(footer)


def build_document():
    doc = Document()
    configure_document(doc)

    kicker = doc.add_paragraph()
    kicker.paragraph_format.space_before = Pt(16)
    kicker.paragraph_format.space_after = Pt(4)
    set_run_font(kicker.add_run("BÁO CÁO TÓM TẮT KỸ THUẬT"), size=10, bold=True, color=BLUE)

    title = doc.add_paragraph(style="Title")
    set_run_font(title.add_run("Phương pháp truy hồi bằng chứng của dự án và so sánh với chiến lược đề xuất"), size=22, bold=True)
    subtitle = doc.add_paragraph(style="Subtitle")
    set_run_font(subtitle.add_run("DenseNet classifier, RAD-DINO MAIRA-2 gốc và evidence theo nhãn bệnh"), size=12, color=MUTED)

    add_table(
        doc,
        ["Hạng mục", "Thông tin"],
        [
            ("Phạm vi", "Pipeline classifier → retriever của dự án IU-Xray hiện tại."),
            ("Tài liệu so sánh", "quy_trinh_evidence_retrieval_reasoning_cxr.docx, từ Mục 3 trở đi."),
            ("Cấu hình retrieval", "microsoft/rad-dino-maira-2 gốc, embedding 768-D, không checkpoint projection + FAISS."),
            ("Ngày lập", "24/08/2026"),
        ],
        [2700, 6660],
    )

    add_callout(
        doc,
        "Kết luận ngắn",
        "Dự án đã có morphology retriever mạnh, bank search logic theo nhãn, candidate union, DenseNet local feature-map rerank và quality gate rule-based. Các khoảng cách còn lại là calibration, negative/uncertain metadata đáng tin và validation của quality gate.",
    )

    doc.add_heading("1. Phương pháp hiện tại của dự án", level=1)
    add_body(doc, "Pipeline online bắt đầu bằng DenseNet relation-aware để tạo sáu xác suất thô theo thứ tự Atelectasis, Cardiomegaly, Edema, Pleural Effusion, Pneumonia và Pneumothorax. Mảng --label_thresholds cho phép đặt ngưỡng độc lập cho từng nhãn. Chỉ nhãn có xác suất thô lớn hơn hoặc bằng ngưỡng tương ứng mới kích hoạt retrieval theo nhãn.")
    add_body(doc, "Retriever dùng microsoft/rad-dino-maira-2 gốc, không nạp checkpoint projection hoặc fine-tune trên gallery. Vector 768-D L2-normalized được tìm bằng FAISS IndexFlatIP, vì vậy inner product tương đương cosine similarity.")
    add_table(
        doc,
        ["Bước", "Cơ chế đang chạy", "Đầu ra"],
        [
            ("1. Classifier", "DenseNet trả probability thô cho 6 nhãn.", "raw_probabilities"),
            ("2. Chọn nhãn", "So p_raw,c với label_thresholds[c].", "selected_labels"),
            ("3. Morphology search", "RAD-DINO gốc tạo query vector; FAISS luôn lấy Top-200 global.", "common_evidence"),
            ("4. Evidence chung", "Khử trùng study_id và giữ 5 ảnh cosine cao nhất.", "common_evidence"),
            ("5. Evidence theo nhãn", "Search positive/non-support bank Top-50, union với global, DenseNet local rerank Top-10, rồi giữ 3 ảnh.", "supporting_evidence / matched_non_support_evidence"),
        ],
        [1650, 4900, 2810],
    )

    doc.add_heading("2. Quy tắc xếp hạng evidence hiện tại", level=1)
    add_label_paragraph(doc, "Stage một. ", "Candidate có label_support > 0 hoặc = 0 được lấy từ logical bank và candidate union. Xác suất classifier và mức support chỉ dùng để routing/lọc candidate, không dùng để xếp hạng reference.")
    add_label_paragraph(doc, "Stage hai DenseNet. ", "Top candidate mỗi nhánh được so khớp bằng symmetric maximum patch cosine trên feature map DenseNet121 trước global pooling. Điểm cuối trộn base rank với DenseNet local similarity; các danh sách đều khử trùng theo study_id.")
    add_callout(
        doc,
        "Giới hạn diễn giải",
        "Trong metadata hiện tại, support = 0 chỉ nói rằng không có match support dương từ MeSH/problems. Nó không phải nhãn negative đã được xác minh từ báo cáo hoặc chuyên gia. Vì vậy matched_non_support_evidence không phải phản chứng lâm sàng chắc chắn.",
        fill=CAUTION_FILL,
        color="7A5A00",
    )

    doc.add_heading("3. Chiến lược trong tài liệu tham chiếu", level=1)
    add_body(doc, "Từ Mục 3, tài liệu tách rõ morphology similarity khỏi disease evidence. Classifier sau calibration xác định các nhãn cần kiểm chứng bằng threshold tau_c, vùng bất định delta_c và Top-M. Morphology retrieval chạy độc lập để cung cấp global neighborhood, kể cả khi classifier không chọn nhãn.")
    add_body(doc, "Với từng nhãn, tài liệu yêu cầu search riêng positive bank và negative bank, sau đó hợp candidate pool: C_global ∪ C_positive ∪ C_negative. Disease-conditioned reranker được dùng để đánh giá hình thái liên quan đúng target pathology. Top-3 chỉ để hiển thị; backend phải tính statistics trên danh sách rộng hơn và chỉ kết luận Supported, Contradicted, Inconclusive hoặc Insufficient sau quality gate Q_ret.")

    doc.add_heading("4. So sánh trực tiếp", level=1)
    add_table(
        doc,
        ["Tiêu chí", "Dự án hiện tại", "Tài liệu đề xuất", "Đánh giá"],
        [
            ("Xác suất và ngưỡng", "Ngưỡng riêng nhưng dùng p_raw.", "p_cal, tau_c, delta_c, Top-M.", "Thiếu calibration và uncertainty band."),
            ("Global retrieval", "Top-200 luôn chạy, kể cả khi không nhãn đạt ngưỡng.", "Luôn chạy độc lập với classifier.", "Đã phù hợp ở mức morphology neighborhood."),
            ("Candidate pool", "Union global + positive bank + non-support bank logic.", "Union global + positive + verified-negative bank.", "Đã giải bottleneck; negative semantics còn hạn chế."),
            ("Nhánh negative", "support = 0.", "Negative label đáng tin, tách uncertain.", "Chưa đủ để gọi Contradicted."),
            ("Reranker", "DenseNet spatial feature-map rerank trên Top-10; p_classifier chỉ routing.", "Disease-conditioned vector score S_c(q,r).", "Local rerank đã có; chưa train gating riêng theo nhãn."),
            ("Chất lượng retrieval", "M@5 primary; M@3/M@10 stability; M@20 warning; density và coverage.", "Quality gate được calibration trước reasoning.", "Đã có MVP gate; cần validation."),
            ("Diversity", "Khử trùng study_id.", "Patient diversity/MMR và kiểm tra stability.", "Đã có bước cơ bản, chưa đủ."),
            ("Output", "Common Top-5; support/refute Top-3.", "Evidence packet, status tách diagnosis.", "Phù hợp để hiển thị, thiếu status chuẩn hóa."),
        ],
        [1700, 2550, 2550, 2560],
    )

    doc.add_heading("5. Phân tích các khoảng cách kỹ thuật", level=1)
    add_label_paragraph(doc, "Candidate recall. ", "Positive bank và non-support bank dùng selector logic trên index FAISS gốc, không nhân bản embedding. Candidate union bảo đảm ca cùng bệnh vẫn được xét dù đứng ngoài Top-200 morphology toàn cục.")
    add_label_paragraph(doc, "Matched non-support rank. ", "Nhánh non-support xếp theo morphology cosine, không dùng công thức chứa xác suất classifier. Điều này đúng semantic hơn, nhưng vẫn không biến candidate thành verified negative.")
    add_label_paragraph(doc, "Rerank disease-specific. ", "DenseNet local reranker đã bổ sung so khớp vùng đặc trưng cho candidate hậu ANN. Tuy vậy, score này dùng cùng feature map cho mọi nhãn; low-rank projection hoặc gating W_c/g_c theo sáu nhãn vẫn là bước tiếp theo nếu cần phân biệt bệnh-specific mạnh hơn.")
    add_label_paragraph(doc, "An toàn quyết định. ", "Không nên dùng absence of support để đảo classifier. Nếu quality thấp hoặc support/counter không tách biệt, trạng thái phù hợp là Insufficient hoặc Inconclusive, không phải Contradicted.")

    doc.add_heading("6. Lộ trình đề xuất", level=1)
    add_table(
        doc,
        ["Ưu tiên", "Thay đổi", "Lợi ích"],
        [
            ("P0", "Thêm label state positive/negative/uncertain, nguồn nhãn và reliability vào metadata.", "Định nghĩa support/counter có ý nghĩa và audit được."),
            ("P0", "Calibrate xác suất classifier, tau_c và delta_c trên validation set; thêm Top-M routing.", "Giảm trigger retrieval không cần thiết và xử lý ca borderline."),
            ("P1", "Validation rule-based quality gate, bao gồm threshold similarity và margin theo từng nhãn.", "Đưa Q_ret từ heuristic sang tiêu chí có cơ sở."),
            ("P1", "Tạo verified-negative/uncertain metadata và label reliability.", "Cho phép counter-evidence có nghĩa lâm sàng hơn."),
            ("P2", "Huấn luyện label-conditioned low-rank reranker/gating và MMR diversity.", "Tăng phân biệt bệnh-specific, giảm duplicate evidence."),
            ("P2", "Chuẩn hóa evidence status Supported/Inconclusive/Insufficient; chỉ mở Contradicted sau validation.", "Tách evidence assessment khỏi diagnosis."),
        ],
        [850, 5020, 3490],
    )

    doc.add_heading("7. Kết luận", level=1)
    add_body(doc, "Chiến lược hiện tại đã vượt qua MVP global-only: embedding bệnh lý mạnh, ANN nhanh, bank search logic, candidate union, evidence Top-3 và quality statistics theo M@5 đã được tách trong JSON. M@3 kiểm tra cùng dấu, M@10 kiểm tra đảo chiều và M@20 chỉ tạo cảnh báo. So với tài liệu, hạng mục cần ưu tiên là calibration, verified-negative/uncertain metadata, quality gate được validation và label-conditioned reranker. Những thay đổi này sẽ tiếp tục đưa retrieval từ tìm ảnh tương tự sang đánh giá evidence có thể audit.")

    doc.add_paragraph().paragraph_format.space_after = Pt(0)
    source = doc.add_paragraph()
    source.paragraph_format.space_before = Pt(4)
    source.paragraph_format.space_after = Pt(4)
    set_run_font(source.add_run("Nguồn: "), size=9, bold=True, color=MUTED)
    set_run_font(source.add_run("quy_trinh_evidence_retrieval_reasoning_cxr.docx; config/config.yaml; src/retriever/faiss_retriever.py; scripts/test_classifier_to_retriever.py; scripts/train_pathology_projection.py."), size=9, color=MUTED)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    doc.save(OUT_PATH)
    print(OUT_PATH.resolve())


if __name__ == "__main__":
    build_document()
