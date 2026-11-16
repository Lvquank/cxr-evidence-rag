"""Create a Vietnamese implementation guide for four-pipeline clinical report metrics."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "thesis" / "HuongDan_TrienKhai_DanhGia_Clinical_Metrics_4_Pipelines.docx"

BLUE = "2E74B5"
DARK = "1F4D78"
INK = "0B2545"
MUTED = "5B6573"
LIGHT = "E8EEF5"
PALE = "F4F6F9"
CAUTION = "FFF5E5"


def set_font(run, name="Calibri", size=None, color=None, bold=None, italic=None):
    run.font.name = name
    run._element.rPr.rFonts.set(qn("w:ascii"), name)
    run._element.rPr.rFonts.set(qn("w:hAnsi"), name)
    if size is not None:
        run.font.size = Pt(size)
    if color:
        run.font.color.rgb = RGBColor.from_string(color)
    if bold is not None:
        run.bold = bold
    if italic is not None:
        run.italic = italic


def shade(cell, fill):
    props = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), fill)
    props.append(shd)


def set_cell_margins(cell, top=80, start=120, bottom=80, end=120):
    props = cell._tc.get_or_add_tcPr()
    margins = props.first_child_found_in("w:tcMar")
    if margins is None:
        margins = OxmlElement("w:tcMar")
        props.append(margins)
    for side, value in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        node = margins.find(qn(f"w:{side}"))
        if node is None:
            node = OxmlElement(f"w:{side}")
            margins.append(node)
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")


def set_table_geometry(table, widths):
    total = sum(widths)
    table.autofit = False
    table.alignment = WD_TABLE_ALIGNMENT.LEFT
    tbl = table._tbl
    tbl_pr = tbl.tblPr
    tbl_w = tbl_pr.first_child_found_in("w:tblW")
    tbl_w.set(qn("w:w"), str(total))
    tbl_w.set(qn("w:type"), "dxa")
    tbl_ind = tbl_pr.first_child_found_in("w:tblInd")
    if tbl_ind is None:
        tbl_ind = OxmlElement("w:tblInd")
        tbl_pr.append(tbl_ind)
    tbl_ind.set(qn("w:w"), "120")
    tbl_ind.set(qn("w:type"), "dxa")
    grid = tbl.tblGrid
    for col, width in zip(grid.gridCol_lst, widths):
        col.set(qn("w:w"), str(width))
    for row in table.rows:
        for cell, width in zip(row.cells, widths):
            cell.width = Inches(width / 1440)
            tc_w = cell._tc.tcPr.tcW
            tc_w.set(qn("w:w"), str(width))
            tc_w.set(qn("w:type"), "dxa")
            cell.vertical_alignment = WD_ALIGN_VERTICAL.TOP
            set_cell_margins(cell)


def set_repeat_header(row):
    tr_pr = row._tr.get_or_add_trPr()
    node = OxmlElement("w:tblHeader")
    node.set(qn("w:val"), "true")
    tr_pr.append(node)


def add_table(doc, headers, rows, widths):
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = "Table Grid"
    set_table_geometry(table, widths)
    header = table.rows[0]
    set_repeat_header(header)
    for cell, value in zip(header.cells, headers):
        shade(cell, LIGHT)
        p = cell.paragraphs[0]
        p.paragraph_format.space_after = Pt(0)
        r = p.add_run(value)
        set_font(r, size=9.5, color=INK, bold=True)
    for row_values in rows:
        cells = table.add_row().cells
        for cell, value in zip(cells, row_values):
            p = cell.paragraphs[0]
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.line_spacing = 1.1
            r = p.add_run(value)
            set_font(r, size=9.4, color="202020")
    set_table_geometry(table, widths)
    doc.add_paragraph().paragraph_format.space_after = Pt(1)
    return table


def add_callout(doc, heading, text, fill=PALE):
    table = doc.add_table(rows=1, cols=1)
    set_table_geometry(table, [9360])
    cell = table.cell(0, 0)
    shade(cell, fill)
    p = cell.paragraphs[0]
    p.paragraph_format.space_after = Pt(2)
    r = p.add_run(heading + " ")
    set_font(r, size=10.5, color=DARK, bold=True)
    r = p.add_run(text)
    set_font(r, size=10.5, color="303030")
    doc.add_paragraph().paragraph_format.space_after = Pt(1)


def add_body(doc, text, bold_prefix=None):
    p = doc.add_paragraph(style="Normal")
    if bold_prefix and text.startswith(bold_prefix):
        r = p.add_run(bold_prefix)
        set_font(r, bold=True, color=INK)
        r = p.add_run(text[len(bold_prefix):])
        set_font(r, color="242424")
    else:
        r = p.add_run(text)
        set_font(r, color="242424")
    return p


def add_code(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Inches(0.18)
    p.paragraph_format.right_indent = Inches(0.18)
    p.paragraph_format.space_before = Pt(3)
    p.paragraph_format.space_after = Pt(5)
    p.paragraph_format.line_spacing = 1.0
    p_pr = p._p.get_or_add_pPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), "F2F4F7")
    p_pr.append(shd)
    for line_no, line in enumerate(text.splitlines()):
        if line_no:
            p.add_run().add_break()
        r = p.add_run(line)
        set_font(r, name="Consolas", size=8.7, color="202020")


def setup_document(doc):
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
    normal.font.size = Pt(10.5)
    normal.paragraph_format.space_after = Pt(7)
    normal.paragraph_format.line_spacing = 1.25

    for name, size, color, before, after in (
        ("Heading 1", 16, BLUE, 18, 10),
        ("Heading 2", 13, BLUE, 12, 6),
        ("Heading 3", 11.5, DARK, 8, 4),
    ):
        style = doc.styles[name]
        style.font.name = "Calibri"
        style._element.rPr.rFonts.set(qn("w:ascii"), "Calibri")
        style._element.rPr.rFonts.set(qn("w:hAnsi"), "Calibri")
        style.font.size = Pt(size)
        style.font.color.rgb = RGBColor.from_string(color)
        style.font.bold = True
        style.paragraph_format.space_before = Pt(before)
        style.paragraph_format.space_after = Pt(after)
        style.paragraph_format.keep_with_next = True

    header = section.header.paragraphs[0]
    header.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    header.paragraph_format.space_after = Pt(0)
    r = header.add_run("CXR-RAG | Clinical metrics implementation guide")
    set_font(r, size=8.5, color=MUTED)
    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = footer.add_run("Implementation guide | Internal research document")
    set_font(r, size=8.5, color=MUTED)


def heading(doc, text, level=1):
    p = doc.add_paragraph(style=f"Heading {level}")
    p.add_run(text)
    return p


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    doc = Document()
    setup_document(doc)

    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(18)
    p.paragraph_format.space_after = Pt(4)
    r = p.add_run("IMPLEMENTATION GUIDE")
    set_font(r, size=10, color=BLUE, bold=True)

    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(7)
    r = p.add_run("Triển khai đánh giá CheXbert và RadGraph\ncho bốn pipeline sinh báo cáo X-quang ngực")
    set_font(r, size=23, color=INK, bold=True)

    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(14)
    r = p.add_run("Phạm vi: Structured Rule, Structured Learned Gate, Baseline A và Baseline B | Phiên bản triển khai đề xuất")
    set_font(r, size=11, color=MUTED, italic=True)

    add_table(doc, ["Trường", "Nội dung"], [
        ("Mục tiêu", "Tạo một protocol đánh giá lâm sàng tái lập, dùng cùng 206 study cho bốn pipeline."),
        ("Đầu vào", "Bốn bộ summary_*.json đã có và IU-Xray reference Findings/Impression."),
        ("Đầu ra", "Bảng CheXbert, Entity/Relation/Complete RadGraph, per-sample CSV và bootstrap CI."),
        ("Không thuộc phạm vi", "Không chạy lại DenseNet, retriever hoặc 206 yêu cầu LLM."),
    ], [2250, 7110])

    add_callout(doc, "Quyết định cần chốt trước khi chạy.", "Không so sánh điểm giữa các pipeline nếu khác study set, khác cách ghép report, khác checkpoint, hoặc khác quy tắc xử lý nhãn uncertain.", CAUTION)

    heading(doc, "1. Hiện trạng dự án và khoảng trống", 1)
    add_body(doc, "Bốn pipeline đã có đủ 206 file summary_*.json, cùng dữ liệu IU-Xray reference, ontology sáu bệnh và script prepare_clinical_text_pairs.py để xuất các cặp sinh-tham chiếu theo study. Hệ thống cũng đã có đánh giá label-target, BLEU, ROUGE-L và BERTScore.")
    add_table(doc, ["Thành phần", "Trạng thái", "Vai trò"], [
        ("Structured Rule", "Sẵn sàng - 206 reports", "Classifier + retriever + structured evidence + rule gate."),
        ("Structured Learned Gate", "Sẵn sàng - 206 reports", "Cùng luồng structured, dùng learned gate OOF."),
        ("Baseline A", "Sẵn sàng - 206 reports", "Classifier probability + image + LLM."),
        ("Baseline B", "Sẵn sàng - 206 reports", "Classifier + retrieval summary + image + LLM."),
        ("CheXbert/RadGraph runner", "Chưa có", "Phần cần tích hợp để tạo clinical metrics so sánh được."),
    ], [2250, 2050, 5060])

    heading(doc, "2. Protocol đánh giá cần được cố định", 1)
    heading(doc, "2.1 Đơn vị phân tích và tập mẫu", 2)
    add_body(doc, "Đơn vị bootstrap là study, không phải câu hay nhãn. Mỗi pipeline chỉ được giữ các study xuất hiện ở cả bốn bộ kết quả và có reference tương ứng. Với run hiện tại, kỳ vọng tập giao là 206 study. Nếu thấp hơn 206, manifest phải ghi rõ các ID bị loại và lý do.")
    heading(doc, "2.2 Chuẩn hoá văn bản", 2)
    add_body(doc, "Với mỗi study, tạo cùng một trường report_text cho generated và reference: Findings, xuống dòng, rồi Impression. Chuẩn hoá khoảng trắng và giá trị rỗng; không được tự dịch, thêm hậu xử lý lâm sàng hoặc thay đổi nội dung report.")
    add_code(doc, "report_text = normalize(findings) + '\\n' + normalize(impression)\n# Giữ study_id, pipeline, generated_text, reference_text trong mọi artifact.")
    heading(doc, "2.3 Metric và quy ước", 2)
    add_table(doc, ["Metric", "Cách dùng đề xuất", "Điều phải báo cáo"], [
        ("CheXbert", "Label cả generated/reference cùng checkpoint; tính micro F1 và macro F1 trên các quan sát được chọn.", "Danh sách nhãn, xử lý -1 uncertain và NaN blank."),
        ("RadGraph", "Trích entity và relation cho cả hai report; tính Entity, Relation và Complete F1.", "Model type, phiên bản và quy tắc matching."),
        ("Bootstrap", "Paired bootstrap theo study để so sánh từng pipeline với Structured Rule hoặc Baseline B.", "Số lần lặp, seed, CI 95% và delta."),
    ], [1700, 4700, 2960])
    add_callout(doc, "Khuyến nghị báo cáo.", "Gọi CheXbert/RadGraph là clinical text metrics, không gọi là chẩn đoán lâm sàng hay chứng minh an toàn lâm sàng. Đây là phép đối chiếu report sinh với report tham chiếu.")
    heading(doc, "2.4 Ghi nhận cấu hình retrieval", 2)
    add_body(doc, "Clinical metrics phải lưu cấu hình retrieval đã sinh report, vì thay đổi lambda làm thay đổi thứ hạng support/non-support, evidence state và cuối cùng là văn bản LLM. Cấu hình mặc định của các Structured pipeline hiện là lambda_global = 0.9 và lambda_local = 0.1.")
    add_code(doc, "retrieval:\n  lambda_global: 0.9\n  lambda_local: 0.1\n  score: lambda_global * global + (1 - lambda_global) * local")
    add_body(doc, "Nếu chạy lambda_global = 0.7, phải ghi output vào một thư mục experiment riêng, chạy lại evidence và report liên quan, rồi đánh giá như một cấu hình độc lập. Không thay thế hoặc trộn các report lambda = 0.7 với kết quả lambda = 0.9 đã freeze.")

    heading(doc, "3. Kiến trúc triển khai", 1)
    add_code(doc, "summary_*.json (4 pipelines)\n        -> prepare_clinical_text_pairs.py\n        -> 4 clinical_text_pairs.csv\n        -> validate_common_studies.py\n        -> evaluate_clinical_metrics.py\n             |-> CheXbert labels -> F1\n             |-> RadGraph graphs -> Entity/Relation/Complete F1\n             |-> paired bootstrap -> CI\n        -> metrics.json + per_sample.csv + final_table.csv")
    add_body(doc, "Không đưa các metric này vào run_full_pipeline.py. Chúng là hậu đánh giá, chạy sau khi bốn luồng đã hoàn tất để tránh làm pipeline sinh report phức tạp hoặc phụ thuộc vào checkpoint evaluator.")

    heading(doc, "4. Chuẩn bị môi trường và dependency", 1)
    heading(doc, "4.1 Tách môi trường evaluator", 2)
    add_body(doc, "Tạo môi trường Conda riêng cho clinical metrics để không xung đột torch/transformers của pipeline ảnh. Dùng Python 3.10 là lựa chọn bảo thủ; lưu phiên bản thực tế sau khi cài vào requirements-clinical-metrics.lock.txt.")
    add_code(doc, "conda create -n cxr_metrics python=3.10 -y\nconda activate cxr_metrics\npython -m pip install --upgrade pip\npip install pandas numpy scipy scikit-learn nltk rouge-score bert-score torch transformers")
    heading(doc, "4.2 CheXbert", 2)
    add_body(doc, "Clone mã nguồn CheXbert chính thức vào thư mục ngoài code pipeline, ví dụ third_party/CheXbert, sau đó tải checkpoint chính thức. Wrapper sẽ tạo CSV tạm có cột Report Impression theo yêu cầu CLI của CheXbert và lưu raw label output để kiểm toán.")
    add_code(doc, "git clone https://github.com/stanfordmlgroup/CheXbert.git third_party/CheXbert\n# Tải checkpoint theo hướng dẫn chính thức, đặt tại models/chexbert/<checkpoint>.pth\npython third_party/CheXbert/label.py -d=<reports.csv> -o=<output_dir> -c=<checkpoint>")
    heading(doc, "4.3 RadGraph", 2)
    add_body(doc, "Cài repository/package RadGraph chính thức và lựa chọn đúng một model_type cho toàn bộ thử nghiệm. Không đổi model_type giữa các pipeline. Nếu mục tiêu là tái lập Entity/Relation/Complete F1 lịch sử, cần dùng đúng implementation scoring đã chọn, không chỉ inference extractor.")
    add_code(doc, "git clone https://github.com/Stanford-AIMI/radgraph.git third_party/radgraph\npip install -e third_party/radgraph\n# Pin commit hash và lưu model_type trong config/clinical_metrics.yaml")
    add_callout(doc, "Checkpoint và license.", "CheXbert checkpoint phải được tải từ nguồn công bố; một số implementation/metric RadGraph có điều kiện truy cập checkpoint. Không commit checkpoint hoặc dữ liệu report vào Git. Lưu đường dẫn, SHA-256 và license trong manifest.", CAUTION)

    heading(doc, "5. Cấu trúc file đề xuất", 1)
    add_code(doc, "config/clinical_metrics.yaml\nrequirements-clinical-metrics.txt\nscripts/prepare_clinical_text_pairs.py              # đã có\nscripts/evaluate_clinical_metrics.py                # cần tạo\nscripts/bootstrap_clinical_metrics.py               # có thể tích hợp vào script trên\noutputs/comparison_4pipelines/\n  clinical_pairs/\n  chexbert_raw/\n  radgraph_raw/\n  clinical_metrics_4pipelines.json\n  clinical_per_sample_4pipelines.csv\n  clinical_bootstrap_ci.json\n  clinical_final_table.csv\n  clinical_metrics_manifest.json")
    add_body(doc, "Không đặt artifact chạy trong src/. Mọi output phải nằm dưới outputs/comparison_4pipelines để tách source code, checkpoint và kết quả nghiên cứu.")

    heading(doc, "6. Các bước triển khai wrapper", 1)
    heading(doc, "Bước 1 - Xuất cặp văn bản của bốn pipeline", 2)
    add_code(doc, "python scripts/prepare_clinical_text_pairs.py `\n  --pipeline structured_rule=outputs/leave_one_out_206/llm `\n  --pipeline structured_learned_gate=src/learned_gate_oof_206/reports `\n  --pipeline baseline_a=outputs/baselines/classifier_vision_leave_one_out_206/llm `\n  --pipeline baseline_b=outputs/baselines/classifier_retriever_vision_leave_one_out_206/llm `\n  --output-dir outputs/comparison_4pipelines/clinical_pairs")
    add_body(doc, "Acceptance check: clinical_text_pairs_manifest.json phải báo rows_exported = 206 cho cả bốn pipeline và missing_reference_study_ids rỗng.")
    heading(doc, "Bước 2 - Xác nhận tập giao và tạo canonical input", 2)
    add_body(doc, "Wrapper đọc bốn CSV, kiểm tra study_id không trùng trong từng pipeline, lấy intersection, sắp xếp theo study_id và ghi canonical_pairs.parquet/csv. Nếu intersection khác 206, wrapper dừng với lỗi thay vì tự im lặng loại mẫu.")
    heading(doc, "Bước 3 - Chạy CheXbert", 2)
    add_body(doc, "Tạo hai batch report, generated và reference, với cùng thứ tự canonical study_id. Chạy CheXbert độc lập cho mỗi batch, join label output lại theo study_id, sau đó tính TP/FP/FN và F1. Nên báo cáo tập quan sát đầy đủ 14 nhãn và, nếu cần headline theo luận văn, bảng phụ chỉ cho sáu pathology đã chốt.")
    heading(doc, "Bước 4 - Chạy RadGraph và tính score", 2)
    add_body(doc, "Chạy inference RadGraph cho generated/reference, lưu raw graph JSON tách theo pipeline. Áp dụng cùng hàm matching cho mọi cặp. Entity F1 phản ánh thực thể lâm sàng, Relation F1 phản ánh liên hệ như located_at/suggestive_of, Complete F1 yêu cầu entity và relation phù hợp theo evaluator đã chọn.")
    heading(doc, "Bước 5 - Bootstrap paired CI", 2)
    add_body(doc, "Với mỗi so sánh A-B, lấy mẫu lại 206 study có hoàn lại, tính delta metric trên cùng các study đã được lấy. Dùng ít nhất 2,000 lần lặp, seed cố định. Lưu mean delta, percentile CI 2.5%-97.5%, và cờ CI có chứa 0 hay không.")
    heading(doc, "Bước 6 - Sinh artifact cuối", 2)
    add_body(doc, "Ghi JSON tổng hợp, CSV từng study và bảng cuối. JSON phải chứa config, nguồn input, số sample, skipped IDs, metric aggregate, CI và đường dẫn raw outputs. Không chỉ giữ số đã làm tròn trong bảng Word.")

    heading(doc, "7. Thiết kế evaluate_clinical_metrics.py", 1)
    add_table(doc, ["Hàm", "Trách nhiệm", "Lỗi phải chặn"], [
        ("load_and_validate", "Đọc bốn CSV, kiểm tra schema và lấy tập giao study.", "Thiếu pipeline, duplicate ID, intersection != 206."),
        ("build_report_text", "Chuẩn hoá Findings + Impression và tạo canonical batch.", "Reference rỗng hoặc report không phải string."),
        ("run_chexbert", "Gọi CLI/API, parse nhãn raw và map về canonical label order.", "Thiếu checkpoint, không khớp số dòng, nhãn lạ."),
        ("run_radgraph", "Sinh graph prediction/reference và gọi scorer thống nhất.", "Model type khác config, output thiếu entity/relation."),
        ("aggregate_and_bootstrap", "Tính metric từng pipeline và delta paired CI.", "NaN, số mẫu không bằng canonical set."),
        ("write_artifacts", "Ghi JSON/CSV/manifest có version và hashes.", "Ghi đè output không có --overwrite."),
    ], [2050, 4510, 2800])
    add_code(doc, "python scripts/evaluate_clinical_metrics.py `\n  --config config/clinical_metrics.yaml `\n  --pairs-dir outputs/comparison_4pipelines/clinical_pairs `\n  --output-dir outputs/comparison_4pipelines `\n  --bootstrap-samples 2000 --seed 20260904")

    heading(doc, "8. Mẫu cấu hình cần lưu", 1)
    add_code(doc, "study_set: data/evaluations/leave_one_out_206_manifest.json\nreport_join: findings_then_impression\nrequired_study_count: 206\nretrieval:\n  lambda_global: 0.9\n  lambda_local: 0.1\n  score: lambda_global * global + (1 - lambda_global) * local\nchexbert:\n  repo_commit: <commit-hash>\n  checkpoint: models/chexbert/<checkpoint>.pth\n  uncertain_policy: keep_as_uncertain\n  headline_labels: [Atelectasis, Cardiomegaly, Edema, Pleural Effusion, Pneumonia, Pneumothorax]\nradgraph:\n  repo_commit: <commit-hash>\n  model_type: <single-fixed-model-type>\n  score_levels: [entity, relation, complete]\nbootstrap:\n  samples: 2000\n  seed: 20260904\n  confidence_level: 0.95")
    add_body(doc, "uncertain_policy chỉ là ví dụ placeholder. Cần chọn và ghi rõ chính sách cuối cùng trước khi chạy. Không được thay đổi chính sách sau khi đã xem kết quả.")

    heading(doc, "9. Kiểm thử và tiêu chí hoàn thành", 1)
    add_table(doc, ["Kiểm tra", "Tiêu chí pass"], [
        ("Data alignment", "Bốn pipeline có chính xác cùng 206 study_id, không duplicate và không missing reference."),
        ("Determinism", "Chạy lại cùng config/seed cho số aggregate và CI không thay đổi ngoài sai số không đáng kể."),
        ("CheXbert raw output", "Số dòng generated/reference bằng 206 và map nhãn hợp lệ."),
        ("RadGraph raw output", "Có 206 graph mỗi phía/pipeline; model_type và commit đồng nhất."),
        ("Statistical output", "Có per-sample metric, aggregate, delta, 95% CI và ghi rõ CI chứa 0 hay không."),
        ("Research table", "Không gọi sample F1 là macro F1; tách text overlap metric khỏi clinical metric."),
    ], [2520, 6840])

    heading(doc, "10. Trình tự thực hiện khuyến nghị", 1)
    add_body(doc, "Giai đoạn 1: cài môi trường riêng, tải checkpoint và ghi config. Giai đoạn 2: chạy thử 5 study để kiểm tra schema, label mapping và runtime. Giai đoạn 3: chạy đủ 206 study cho cả bốn pipeline. Giai đoạn 4: kiểm toán raw output, chạy bootstrap và tạo bảng. Giai đoạn 5: chỉ sau khi các artifact được kiểm tra mới freeze kết quả và dùng trong Abstract/Conclusion.")
    add_callout(doc, "Điểm dừng hợp lệ.", "Nếu CheXbert/RadGraph checkpoint hoặc implementation chưa được chốt, dừng trước khi chạy full 206. Chạy thử với implementation tạm rồi đưa vào bảng cuối sẽ làm mất khả năng tái lập.", CAUTION)

    heading(doc, "Phụ lục A. Nguồn implementation", 1)
    add_body(doc, "CheXbert official repository: https://github.com/stanfordmlgroup/CheXbert. Repository mô tả CLI label report, checkpoint và quy ước nhãn Positive/Negative/Uncertain/Blank.")
    add_body(doc, "RadGraph official repository: https://github.com/Stanford-AIMI/radgraph. Repository mô tả inference entity/relation, model_type và dependencies của package.")
    add_body(doc, f"Tài liệu được tạo ngày {date.today().isoformat()} từ hiện trạng artifacts của dự án CXR-RAG.")

    doc.core_properties.title = "Clinical Metrics Implementation Guide for Four CXR-RAG Pipelines"
    doc.core_properties.subject = "CheXbert and RadGraph implementation protocol"
    doc.core_properties.author = "CXR-RAG project"
    doc.save(OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()
