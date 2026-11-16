"""Build the Word report for the embedding and vector-database implementation."""

from pathlib import Path

from docx import Document
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


OUT_PATH = Path("docs/Bao_cao_Embedding_Vector_Database_IU_XRay.docx")
BLUE = "2E74B5"
DARK_BLUE = "1F4D78"
INK = "0B2545"
LIGHT_BLUE = "E8EEF5"
LIGHT_GRAY = "F2F4F7"
MUTED = "667085"


def set_run_font(run, size=11, bold=None, color=None, italic=None):
    run.font.name = "Calibri"
    run._element.rPr.rFonts.set(qn("w:ascii"), "Calibri")
    run._element.rPr.rFonts.set(qn("w:hAnsi"), "Calibri")
    run._element.rPr.rFonts.set(qn("w:eastAsia"), "Calibri")
    run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    if color:
        run.font.color.rgb = RGBColor.from_string(color)
    if italic is not None:
        run.italic = italic


def set_cell_shading(cell, fill):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), fill)
    tc_pr.append(shd)


def set_cell_width(cell, width_dxa):
    tc_pr = cell._tc.get_or_add_tcPr()
    tc_w = tc_pr.find(qn("w:tcW"))
    if tc_w is None:
        tc_w = OxmlElement("w:tcW")
        tc_pr.append(tc_w)
    tc_w.set(qn("w:w"), str(width_dxa))
    tc_w.set(qn("w:type"), "dxa")


def set_cell_margins(cell, top=80, start=120, bottom=80, end=120):
    tc_pr = cell._tc.get_or_add_tcPr()
    margins = tc_pr.first_child_found_in("w:tcMar")
    if margins is None:
        margins = OxmlElement("w:tcMar")
        tc_pr.append(margins)
    for side, value in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        node = margins.find(qn(f"w:{side}"))
        if node is None:
            node = OxmlElement(f"w:{side}")
            margins.append(node)
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")


def set_table_geometry(table, widths):
    table.autofit = False
    tbl_pr = table._tbl.tblPr
    tbl_w = tbl_pr.first_child_found_in("w:tblW")
    tbl_w.set(qn("w:w"), "9360")
    tbl_w.set(qn("w:type"), "dxa")
    tbl_ind = OxmlElement("w:tblInd")
    tbl_ind.set(qn("w:w"), "120")
    tbl_ind.set(qn("w:type"), "dxa")
    tbl_pr.append(tbl_ind)
    grid = table._tbl.tblGrid
    for col, width in zip(grid.gridCol_lst, widths):
        col.set(qn("w:w"), str(width))
    for row in table.rows:
        for cell, width in zip(row.cells, widths):
            set_cell_width(cell, width)
            set_cell_margins(cell)


def set_repeat_table_header(row):
    tr_pr = row._tr.get_or_add_trPr()
    marker = OxmlElement("w:tblHeader")
    marker.set(qn("w:val"), "true")
    tr_pr.append(marker)


def add_table(doc, headers, rows, widths):
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = "Table Grid"
    set_table_geometry(table, widths)
    header_cells = table.rows[0].cells
    for cell, text in zip(header_cells, headers):
        set_cell_shading(cell, LIGHT_BLUE)
        paragraph = cell.paragraphs[0]
        paragraph.paragraph_format.space_after = Pt(0)
        run = paragraph.add_run(text)
        set_run_font(run, size=10, bold=True, color=INK)
    set_repeat_table_header(table.rows[0])
    for row_values in rows:
        cells = table.add_row().cells
        for cell, text in zip(cells, row_values):
            paragraph = cell.paragraphs[0]
            paragraph.paragraph_format.space_after = Pt(0)
            run = paragraph.add_run(text)
            set_run_font(run, size=10)
    set_table_geometry(table, widths)
    return table


def add_body(doc, text):
    p = doc.add_paragraph(style="Normal")
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.line_spacing = 1.10
    p.add_run(text)
    return p


def add_label_paragraph(doc, label, text):
    p = doc.add_paragraph(style="Normal")
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.line_spacing = 1.10
    run = p.add_run(label)
    set_run_font(run, bold=True, color=DARK_BLUE)
    p.add_run(text)
    return p


def configure_styles(doc):
    normal = doc.styles["Normal"]
    normal.font.name = "Calibri"
    normal._element.rPr.rFonts.set(qn("w:ascii"), "Calibri")
    normal._element.rPr.rFonts.set(qn("w:hAnsi"), "Calibri")
    normal.font.size = Pt(11)
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.line_spacing = 1.10

    for style_name, size, color, before, after in (
        ("Title", 23, INK, 0, 4),
        ("Subtitle", 13, MUTED, 0, 16),
        ("Heading 1", 16, BLUE, 16, 8),
        ("Heading 2", 13, BLUE, 12, 6),
        ("Heading 3", 12, DARK_BLUE, 8, 4),
    ):
        style = doc.styles[style_name]
        style.font.name = "Calibri"
        style._element.rPr.rFonts.set(qn("w:ascii"), "Calibri")
        style._element.rPr.rFonts.set(qn("w:hAnsi"), "Calibri")
        style.font.size = Pt(size)
        style.font.color.rgb = RGBColor.from_string(color)
        style.paragraph_format.space_before = Pt(before)
        style.paragraph_format.space_after = Pt(after)
    doc.styles["Title"].font.bold = True
    doc.styles["Heading 1"].font.bold = True
    doc.styles["Heading 2"].font.bold = True
    doc.styles["Heading 3"].font.bold = True

    if "Report Note" not in doc.styles:
        note = doc.styles.add_style("Report Note", WD_STYLE_TYPE.PARAGRAPH)
        note.font.name = "Calibri"
        note.font.size = Pt(10)
        note.font.color.rgb = RGBColor.from_string(MUTED)
        note.paragraph_format.space_after = Pt(8)
        note.paragraph_format.line_spacing = 1.10


def set_header_footer(section):
    header = section.header.paragraphs[0]
    header.alignment = WD_ALIGN_PARAGRAPH.LEFT
    header.paragraph_format.space_after = Pt(0)
    run = header.add_run("IU-XRAY RETRIEVAL | Technical Report")
    set_run_font(run, size=9, color=MUTED)

    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    run = footer.add_run("Image RAG Project")
    set_run_font(run, size=9, color=MUTED)


def build_document():
    doc = Document()
    section = doc.sections[0]
    section.top_margin = Inches(1)
    section.bottom_margin = Inches(1)
    section.left_margin = Inches(1)
    section.right_margin = Inches(1)
    section.header_distance = Inches(0.492)
    section.footer_distance = Inches(0.492)
    configure_styles(doc)
    set_header_footer(section)

    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(16)
    p.paragraph_format.space_after = Pt(4)
    run = p.add_run("BÁO CÁO KỸ THUẬT")
    set_run_font(run, size=10, bold=True, color=BLUE)

    title = doc.add_paragraph(style="Title")
    title.add_run("Hệ thống trích xuất embedding và vector database cho IU-Xray")
    subtitle = doc.add_paragraph(style="Subtitle")
    subtitle.add_run("RAD-DINO pathology-aware image retrieval")

    add_table(
        doc,
        ["Hạng mục", "Cấu hình hiện tại"],
        [
            ("Dữ liệu", "IU-Xray, 3.818 ảnh Frontal đã xử lý"),
            ("Image encoder", "microsoft/rad-dino-maira-2"),
            ("Embedding retrieval", "512-D, L2-normalized"),
            ("Vector index", "FAISS IndexFlatIP"),
            ("Checkpoint", "pathology_projection_rad_dino_pretrained.pt"),
            ("Huấn luyện checkpoint", "10 epoch trên Kaggle GPU"),
        ],
        [2700, 6660],
    )

    doc.add_heading("Mục tiêu", level=1)
    add_body(
        doc,
        "Hệ thống hỗ trợ truy xuất các ảnh X-quang ngực có biểu diễn hình ảnh và mức hỗ trợ bệnh lý tương đồng với ảnh truy vấn. Mục tiêu là tạo nền tảng retrieval phục vụ phân tích báo cáo và đối chiếu ca bệnh; kết quả truy xuất không được dùng như kết luận chẩn đoán lâm sàng độc lập.",
    )

    doc.add_heading("1. Phương pháp trích xuất embedding", level=1)
    doc.add_heading("1.1. Dữ liệu đầu vào và tiền xử lý", level=2)
    add_body(
        doc,
        "Dữ liệu được chuẩn hóa về ảnh PNG và metadata JSONL. Mỗi bản ghi sử dụng định danh ảnh, đường dẫn tương đối, study ID, view, MeSH, problems và vector support sáu phần tử. Chỉ các ảnh Frontal được đưa vào kho retrieval để giữ tính nhất quán về góc chụp.",
    )
    add_body(
        doc,
        "Cả gallery và ảnh truy vấn dùng chung ImagePreprocessor của RAD-DINO: ảnh được chuyển sang RGB, resize theo cạnh ngắn 518 px với tỷ lệ gốc được giữ nguyên, chuẩn hóa bằng AutoImageProcessor của model và padding về bội số patch size 14. Quy trình không center-crop nên không chủ động loại bỏ vùng giải phẫu khỏi ảnh.",
    )

    doc.add_heading("1.2. Backbone và biểu diễn đặc trưng", level=2)
    add_body(
        doc,
        "Backbone là microsoft/rad-dino-maira-2. RAD-DINO tạo hidden representation 768-D từ ảnh X-quang. Hai transformer block cuối, encoder.layer.10 và encoder.layer.11, được fine-tune; các block trước giữ frozen để giảm mức thay đổi biểu diễn nền và giảm chi phí bộ nhớ.",
    )
    add_body(
        doc,
        "Biểu diễn cuối kết hợp CLS token, đại diện toàn cục của ảnh, với trung bình các local patch token. Khi batch chứa ảnh có tỷ lệ khác nhau, patch mask được áp dụng trước khi lấy trung bình để token thuộc phần padding không ảnh hưởng đến feature fusion.",
    )

    doc.add_heading("1.3. Projection và pathology-aware learning", level=2)
    add_body(
        doc,
        "Feature fusion 768-D được đưa qua MLP projection head gồm Linear, GELU và Linear để tạo vector retrieval 512-D. Đầu ra được L2-normalize, cho phép dùng inner product như cosine similarity trong FAISS.",
    )
    add_body(
        doc,
        "Supervision bệnh lý sử dụng vector support sáu nhãn từ metadata. Weighted Jaccard giữa hai vector support tạo relevance liên tục; graded weighted loss buộc cosine similarity của embedding phản ánh mức độ overlap này. Balanced sampling tăng tần suất các nhãn ít gặp, còn online hard-negative mining chọn các cặp không trùng support nhưng có độ giống hình ảnh cao để tối ưu biên phân tách.",
    )

    add_table(
        doc,
        ["Thành phần", "Thiết kế", "Vai trò"],
        [
            ("RAD-DINO", "768-D hidden state", "Feature X-quang nền"),
            ("Fine-tuning", "2 block cuối", "Điều chỉnh theo IU-Xray"),
            ("CLS/local fusion", "CLS + masked mean patch", "Kết hợp ngữ cảnh toàn cục và cục bộ"),
            ("Projection head", "768-D -> 512-D", "Embedding retrieval"),
            ("Normalization", "L2", "So sánh cosine ổn định"),
        ],
        [2100, 2700, 4560],
    )

    doc.add_heading("1.4. Artefact model hiện dùng", level=2)
    add_label_paragraph(doc, "Checkpoint: ", "model/pathology_projection_rad_dino_pretrained.pt.")
    add_label_paragraph(doc, "Nội dung checkpoint: ", "projection head, module CLS/local-patch fusion và state của hai RAD-DINO block đã fine-tune.")
    add_label_paragraph(doc, "Thông tin huấn luyện: ", "10 epoch, batch size 16, learning rate projection 1e-4, learning rate encoder 1e-5.")

    doc.add_heading("2. Thiết kế vector database", level=1)
    doc.add_heading("2.1. Lưu trữ vector và metadata", level=2)
    add_body(
        doc,
        "Vector database triển khai bằng FAISS IndexFlatIP. Vì mọi embedding đã L2-normalize, inner product của FAISS tương đương cosine similarity. Cấu trúc này là exact search, phù hợp quy mô hiện tại 3.818 ảnh, không yêu cầu xấp xỉ hoặc huấn luyện index riêng.",
    )
    add_table(
        doc,
        ["Artefact", "Nội dung", "Mục đích"],
        [
            ("embeddings.npy", "3.818 x 512 float32", "Ma trận vector gallery"),
            ("faiss.index", "IndexFlatIP 512-D", "Tìm kiếm top-k theo cosine"),
            ("ids.txt", "3.818 image ID", "Ánh xạ vị trí FAISS sang ảnh"),
            ("metadata.jsonl", "Metadata compact từng ảnh", "Hiển thị, support, study-level dedupe"),
            ("embedding_manifest.json", "Model, dimension, checkpoint, normalization", "Kiểm tra tương thích artefact"),
        ],
        [2200, 3600, 3560],
    )

    doc.add_heading("2.2. Metadata và tính toàn vẹn", level=2)
    add_body(
        doc,
        "Metadata compact có dạng id, path, study_id, dataset, view, mesh, problems và support. Không lưu report đầy đủ trong FAISS; metadata giữ đủ thông tin để truy vết ảnh, áp dụng logic hỗ trợ bệnh lý và trình bày kết quả. Support là danh sách số thay vì chuỗi để tránh phụ thuộc vào parsing khi rerank hoặc đánh giá.",
    )
    add_body(
        doc,
        "Khi khởi tạo retriever, hệ thống kiểm tra embedding dimension, số lượng vector, số dòng metadata, model revision và đường dẫn checkpoint trong manifest. Các kiểm tra này ngăn việc dùng nhầm index 768-D, index DenseNet hoặc checkpoint projection khác với index đã tạo.",
    )

    doc.add_heading("2.3. Luồng truy vấn", level=2)
    add_label_paragraph(doc, "Bước 1. ", "Ảnh truy vấn đi qua đúng preprocessing và encoder/projection như gallery.")
    add_label_paragraph(doc, "Bước 2. ", "FAISS tìm ứng viên theo inner product trên vector L2-normalized.")
    add_label_paragraph(doc, "Bước 3. ", "Retriever lấy nhiều ứng viên hơn top-k, sau đó khử trùng study ID để một study chỉ xuất hiện một lần.")
    add_label_paragraph(doc, "Bước 4. ", "Nếu có report query, pathology support có thể được suy ra bằng regex và kết hợp với visual score khi rerank.")
    add_label_paragraph(doc, "Diễn giải score. ", "Trường distance trong kết quả là cosine similarity, không phải Euclidean distance; giá trị lớn hơn biểu thị vector gần nhau hơn.")

    doc.add_heading("2.4. Đánh giá", level=2)
    add_body(
        doc,
        "Script evaluate_pathology_retrieval.py đo Precision@K, mAP@K, nDCG@K và pathology-support delta. Khi đánh giá, các ảnh cùng dataset và study ID với query được loại khỏi gallery để tránh trường hợp hai ảnh thuộc cùng một study làm tăng điểm không thực chất.",
    )

    doc.add_heading("3. Phạm vi và lưu ý", level=1)
    add_body(
        doc,
        "Hệ thống được đánh giá trên metadata support của IU-Xray và checkpoint hiện tại mới train 10 epoch. Các metric phản ánh retrieval theo proxy pathology support trong dữ liệu nội bộ, không thay thế validation lâm sàng, test cross-domain hoặc đánh giá bởi bác sĩ chẩn đoán hình ảnh. Trước khi sử dụng ngoài phạm vi nghiên cứu, cần tách train/validation/test theo study hoặc patient, đánh giá cross-domain và kiểm tra các nhãn support bằng review chuyên môn.",
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    doc.save(OUT_PATH)
    print(OUT_PATH.resolve())


if __name__ == "__main__":
    build_document()
