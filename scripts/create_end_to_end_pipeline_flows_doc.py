"""Create a detailed Word guide for the three image-to-report CXR-RAG flows."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from docx import Document
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "mo_ta_3_luong_end_to_end_cxr_rag.docx"

BLUE = "2E74B5"
DARK = "1F4D78"
INK = "0B2545"
LIGHT_BLUE = "E8EEF5"
LIGHT_GRAY = "F2F4F7"
CALLOUT = "F4F6F9"
MUTED = "666666"


def set_run(run, *, size=None, bold=None, color=None, italic=None):
    run.font.name = "Calibri"
    run._element.rPr.rFonts.set(qn("w:ascii"), "Calibri")
    run._element.rPr.rFonts.set(qn("w:hAnsi"), "Calibri")
    if size is not None:
        run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    if color is not None:
        run.font.color.rgb = RGBColor.from_string(color)
    if italic is not None:
        run.italic = italic


def shade(cell, fill: str):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), fill)
    tc_pr.append(shd)


def cell_margin(cell, top=80, start=120, bottom=80, end=120):
    tc_pr = cell._tc.get_or_add_tcPr()
    tc_mar = tc_pr.first_child_found_in("w:tcMar")
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for side, value in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        item = tc_mar.find(qn(f"w:{side}"))
        if item is None:
            item = OxmlElement(f"w:{side}")
            tc_mar.append(item)
        item.set(qn("w:w"), str(value))
        item.set(qn("w:type"), "dxa")


def set_table_geometry(table, widths: list[int]):
    table.alignment = WD_TABLE_ALIGNMENT.LEFT
    table.autofit = False
    tbl_pr = table._tbl.tblPr
    tbl_w = tbl_pr.first_child_found_in("w:tblW")
    if tbl_w is None:
        tbl_w = OxmlElement("w:tblW")
        tbl_pr.append(tbl_w)
    tbl_w.set(qn("w:w"), str(sum(widths)))
    tbl_w.set(qn("w:type"), "dxa")
    indent = tbl_pr.first_child_found_in("w:tblInd")
    if indent is None:
        indent = OxmlElement("w:tblInd")
        tbl_pr.append(indent)
    indent.set(qn("w:w"), "120")
    indent.set(qn("w:type"), "dxa")
    for col, width in zip(table._tbl.tblGrid.gridCol_lst, widths):
        col.set(qn("w:w"), str(width))
    for row in table.rows:
        for cell, width in zip(row.cells, widths):
            tc_pr = cell._tc.get_or_add_tcPr()
            tc_w = tc_pr.find(qn("w:tcW"))
            if tc_w is None:
                tc_w = OxmlElement("w:tcW")
                tc_pr.append(tc_w)
            tc_w.set(qn("w:w"), str(width))
            tc_w.set(qn("w:type"), "dxa")
            cell_margin(cell)
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER


def repeat_header(row):
    tr_pr = row._tr.get_or_add_trPr()
    node = OxmlElement("w:tblHeader")
    node.set(qn("w:val"), "true")
    tr_pr.append(node)


def write_cell(cell, value: str, *, bold=False, color=None, align=WD_ALIGN_PARAGRAPH.LEFT, size=9):
    p = cell.paragraphs[0]
    p.alignment = align
    p.paragraph_format.space_after = Pt(0)
    p.paragraph_format.line_spacing = 1.0
    run = p.add_run(value)
    set_run(run, size=size, bold=bold, color=color)


def add_table(doc, headers: list[str], rows: list[list[str]], widths: list[int], *, body_size=9):
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = "Table Grid"
    set_table_geometry(table, widths)
    repeat_header(table.rows[0])
    for cell, label in zip(table.rows[0].cells, headers):
        shade(cell, LIGHT_BLUE)
        write_cell(cell, label, bold=True, color=INK, align=WD_ALIGN_PARAGRAPH.CENTER, size=9)
    for values in rows:
        cells = table.add_row().cells
        for index, (cell, value) in enumerate(zip(cells, values)):
            write_cell(cell, value, align=WD_ALIGN_PARAGRAPH.LEFT if index else WD_ALIGN_PARAGRAPH.CENTER, size=body_size)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


def para(doc, text: str, *, bold_prefix: str | None = None, italic=False, color=None, size=11, after=6):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(after)
    p.paragraph_format.line_spacing = 1.1
    if bold_prefix and text.startswith(bold_prefix):
        run = p.add_run(bold_prefix)
        set_run(run, size=size, bold=True, color=color)
        run = p.add_run(text[len(bold_prefix):])
        set_run(run, size=size, italic=italic, color=color)
    else:
        run = p.add_run(text)
        set_run(run, size=size, italic=italic, color=color)
    return p


def callout(doc, label: str, text: str):
    table = doc.add_table(rows=1, cols=1)
    table.style = "Table Grid"
    set_table_geometry(table, [9360])
    # A one-row callout has a label-led first row; mark it as a header for accessibility audits.
    repeat_header(table.rows[0])
    cell = table.cell(0, 0)
    shade(cell, CALLOUT)
    p = cell.paragraphs[0]
    p.paragraph_format.space_after = Pt(0)
    p.paragraph_format.line_spacing = 1.1
    run = p.add_run(label + " ")
    set_run(run, size=10, bold=True, color=DARK)
    run = p.add_run(text)
    set_run(run, size=10)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


def section_heading(doc, text: str, level: int = 1):
    return doc.add_heading(text, level=level)


def setup_document() -> Document:
    doc = Document()
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
    for name, size, color, before, after in (
        ("Heading 1", 16, BLUE, 16, 8),
        ("Heading 2", 13, BLUE, 12, 6),
        ("Heading 3", 12, DARK, 8, 4),
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

    header = section.header.paragraphs[0]
    header.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    run = header.add_run("CXR-RAG | Mo ta luong end-to-end")
    set_run(run, size=9, color=MUTED)
    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = footer.add_run("Tai lieu ky thuat noi bo | 31/08/2026")
    set_run(run, size=9, color=MUTED)
    return doc


def main():
    doc = setup_document()

    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(3)
    run = p.add_run("TAI LIEU KY THUAT")
    set_run(run, size=10, bold=True, color=DARK)
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(5)
    run = p.add_run("Ba luong end-to-end: tu anh X-quang den bao cao LLM")
    set_run(run, size=22, bold=True, color=INK)
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(14)
    run = p.add_run("Pipeline goc co structured evidence va hai baseline A/B trong thu nghiem leave-one-study-out")
    set_run(run, size=12, color=MUTED)

    add_table(doc, ["Muc", "Noi dung"], [
        ["Dau vao chung", "Mot anh X-quang nguc da xu ly va study ID de loai query khoi gallery."],
        ["Classifier", "DenseNet121 relation-aware CheXpert, xuat xac suat 6 nhan dich."],
        ["LLM", "gpt-5.6-luna; output JSON bat buoc: problems, findings, impression."],
        ["Dich danh gia", "Atelectasis, Cardiomegaly, Edema, Pleural Effusion, Pneumonia, Pneumothorax va normal."],
        ["Muc dich", "So sanh gia tri cua structured evidence va retrieval trong sinh bao cao."],
    ], [2600, 6760])

    section_heading(doc, "1. Tong quan 3 luong", 1)
    add_table(doc, ["Luong", "Thong tin LLM nhan", "Anh goc", "Diem khac biet"], [
        ["Pipeline goc", "Minimal JSON: classifier + common/label retrieval evidence + status S/M/D/I", "Chi dinh theo route: gan anh khi evidence M/I/D va anh co san", "Co buoc structured evidence, route toi uu chi phi."],
        ["Baseline A", "6 xac suat classifier va nhan vuot threshold", "Gan trong moi request", "Khong retriever, khong evidence structure."],
        ["Baseline B", "6 xac suat classifier + 5 ca gan nhat (similarity/tom tat retrieval)", "Gan trong moi request", "Co retriever, nhung khong phan loai evidence S/M/D/I."],
    ], [1500, 3300, 1350, 3210])
    callout(doc, "Nguyen tac so sanh.", "Ba luong dung cung DenseNet classifier, cung adaptive threshold, cung ontology output va cung manifest query. Vi vay khac biet chinh nam o evidence dua vao LLM.")

    section_heading(doc, "2. Thanh phan dung chung", 1)
    section_heading(doc, "2.1 Anh, classifier va chon nhan", 2)
    para(doc, "Anh da xu ly duoc dua vao DenseNet121 relation-aware. Classifier xuat raw probability cho 6 nhan. File config/densenet_adaptive_thresholds.json cung cap threshold rieng tung nhan; chi nhan co probability >= threshold moi duoc chon de route retrieval va gioi han enum problems cua LLM.")
    add_table(doc, ["Nhom", "Gia tri/hanh vi"], [
        ["6 nhan", "Atelectasis; Cardiomegaly; Edema; Pleural Effusion; Pneumonia; Pneumothorax"],
        ["Neu khong co nhan qua threshold", "LLM van co the tra [normal]; khong tu dong chen nhan benh."],
        ["Output classifier", "Raw probabilities duoc luu de truy vet; selected_labels dung cho route va schema."],
    ], [3100, 6260])
    section_heading(doc, "2.2 Quy tac output chung cua LLM", 2)
    para(doc, "Tat ca luong deu yeu cau mot JSON object duy nhat. problems chi chua cac nhan classifier da chon hoac [normal]; normal khong duoc di kem benh. findings gom 1-3 quan sat ngan gon, impression gom mot ket luan ngan gon. Validator local kiem tra JSON, nhan trung va quy tac normal truoc khi luu prediction.")
    section_heading(doc, "2.3 Gallery retrieval", 2)
    para(doc, "Gallery la index FAISS tu embedding anh pathology-aware. Query study bi loai khoi gallery trong leave-one-study-out de tranh leakage. Retriever co the dung rerank feature-map DenseNet tren tap candidate nho; ket qua sau do duoc dong goi khac nhau o pipeline goc va Baseline B.")

    section_heading(doc, "3. Pipeline goc: Classifier -> Retriever -> Structured Evidence -> LLM", 1)
    callout(doc, "Muc tieu.", "Dua evidence da chuan hoa, bao gom bang chung ung ho va phan bien, vao LLM voi kich thuoc JSON gon de giam token va kiem soat suy luan.")
    add_table(doc, ["Buoc", "Xu ly", "San pham trung gian"], [
        ["1", "Doc anh; DenseNet tinh 6 raw probabilities; adaptive threshold tao selected_labels.", "Classifier probabilities va selected_labels."],
        ["2", "Retriever tim common evidence va label-specific evidence; query study da bi exclude khoi gallery.", "Cac ca gan, similarity, support/non-support va thong ke margin."],
        ["3", "Xay structured evidence. Moi nhan duoc gan trang thai: S (supported), M (mixed), D (non-support-dominant), I (insufficient).", "classifier_retriever_evidence JSON."],
        ["4", "generate_minimal_json tinh gon evidence: f row chua [label, probability, status, support findings]; nhan M/D/I bo sung contrasting findings. g chua fallback hits.", "minimal JSON (schema v4)."],
        ["5", "Chon route chi phi thap nhat. Evidence M/I/D va co anh -> vision_escalation. Evidence du S -> text_default; evidence bat dinh nhung thieu anh -> text_fallback.", "Route, model va co/khong attach anh."],
        ["6", "Prompt radiology + minimal JSON -> LLM -> parse/validate JSON -> luu output.", "response TXT, summary JSON, minimal_data JSON."],
    ], [520, 5050, 3790], body_size=8.5)
    section_heading(doc, "3.1 Logic structured evidence", 2)
    para(doc, "Support case la ca truy hoi co nhan phu hop voi nhan dang xet; non-support case la ca tuong dong ve anh/feature nhung khong co nhan do. Retriever so sanh hai nhom de tinh margin. Prompt xem S la bang chung uu tien; contrasting findings cua M/D/I chi dung de hieu chinh do chac chan, khong duoc coi la finding xac nhan cua query.")
    section_heading(doc, "3.2 Route va chi phi", 2)
    add_table(doc, ["Dieu kien evidence", "Route", "Noi dung gui LLM"], [
        ["Co M, I hoac D va co file anh", "vision_escalation", "System prompt + minimal JSON + anh goc (detail high)."],
        ["Co M, I hoac D nhung khong co anh", "text_fallback", "System prompt + minimal JSON; khong anh."],
        ["Chi co S/khong co status bat dinh", "text_default", "System prompt + minimal JSON; khong anh."],
    ], [2700, 2050, 4610])
    section_heading(doc, "3.3 Tep dau ra", 2)
    para(doc, "Trong output cua moi request, pipeline luu minimal_<timestamp>.json (bieu dien evidence), minimal_data_<timestamp>.json (ban gui/luu), response_<timestamp>.txt (raw LLM text) va summary_<timestamp>.json (prediction da parse, source, route, metadata API). Batch 206 ca con co evidence/, llm/, logs/, manifest.json va batch_summary.json.")

    section_heading(doc, "4. Baseline A: Classifier -> Vision LLM", 1)
    callout(doc, "Muc tieu.", "Danh gia muc dong gop cua classifier va anh khi khong dua retrieval hoac structured evidence vao LLM.")
    add_table(doc, ["Buoc", "Xu ly", "Du lieu gui/luu"], [
        ["1", "Doc anh va chay cung DenseNet classifier.", "6 raw probabilities."],
        ["2", "Ap dung adaptive threshold chung; tao classifier candidates.", "selected_labels dung cho schema problems."],
        ["3", "Tao prompt baseline ngan: xac suat classifier + chi dan report.", "Khong co retrieval hit, khong co S/M/D/I, khong minimal JSON."],
        ["4", "Gui prompt kem anh goc toi Luna; response format bi ep boi JSON schema.", "Original image luon duoc attach trong baseline A."],
        ["5", "Parse/validate va luu response, summary, prediction va API metadata.", "Summary JSON co source study/image de danh gia offline."],
    ], [520, 5050, 3790], body_size=8.5)
    para(doc, "Baseline A co cung diagnostic head va output contract voi luong goc, nhung LLM tu suy luan tu anh va probability. Vi vay no la moc de kiem tra retrieval/evidence co thuc su cai thien bao cao hay khong.")

    section_heading(doc, "5. Baseline B: Classifier -> Retriever -> Vision LLM", 1)
    callout(doc, "Muc tieu.", "Danh gia retrieval truc tiep: LLM nhan cac ca tuong dong, nhung khong nhan evidence structure, status, support/non-support contrast hoac logic margin.")
    add_table(doc, ["Buoc", "Xu ly", "Du lieu gui/luu"], [
        ["1", "Doc anh; DenseNet tinh raw probabilities; threshold tao selected_labels.", "Cung classifier va threshold voi A va pipeline goc."],
        ["2", "Retriever xep hang gallery, exclude query study, lay 5 ca gan nhat.", "Similarity va tom tat ca; co the da duoc local rerank feature map DenseNet."],
        ["3", "Tao prompt baseline: xac suat classifier + danh sach 5 case + instructions sinh report.", "Khong tao classifier_retriever_evidence va khong generate minimal JSON."],
        ["4", "Gui prompt kem anh goc toi Luna, parse theo schema problems/findings/impression.", "Anh goc luon attach trong baseline B."],
        ["5", "Luu raw response va summary JSON theo request.", "Dung cung cau truc summary de co the danh gia bang mot script chung."],
    ], [520, 5050, 3790], body_size=8.5)
    para(doc, "Khac voi pipeline goc, Baseline B khong tach case thanh support va non-support. LLM tu dien giai similarity va tom tat ca retrieve, do do ket qua cho biet loi ich cua retrieval truoc khi them structured evidence.")

    section_heading(doc, "6. So sanh dau vao, output va chi phi", 1)
    add_table(doc, ["Tieu chi", "Pipeline goc", "Baseline A", "Baseline B"], [
        ["Classifier", "Co", "Co", "Co"],
        ["Retriever", "Co; evidence theo nhan", "Khong", "Co; top 5 case"],
        ["Structured evidence", "Co: support/non-support, S/M/D/I, margin", "Khong", "Khong"],
        ["Minimal JSON", "Co", "Khong", "Khong"],
        ["Anh gui LLM", "Co dieu kien theo route", "Moi request", "Moi request"],
        ["Chi phi prompt", "Thap hon khi route text_default; tang khi escalation", "On dinh, co image token moi request", "Cao hon A do them 5 case retrieval + image"],
        ["Output schema", "Giong nhau", "Giong nhau", "Giong nhau"],
    ], [2050, 2430, 2430, 2450])
    callout(doc, "Luu y chi phi.", "Structured evidence khong tu dong re hon Baseline A/B. Loi ich chi phi cua pipeline goc chi xuat hien khi evidence du ro de route text_default khong gan anh. Trong thu nghiem baseline, A va B co anh trong moi request.")

    section_heading(doc, "7. Batch, danh gia va kha nang so sanh", 1)
    para(doc, "scripts/run_leave_one_out_batch.py chay pipeline goc tren manifest co dinh. scripts/run_baseline_leave_one_out_batch.py chay A hoac B tren dung manifest do, nhung ghi sang cac output root rieng. Moi request deu giu study ID va image name, nen scripts/evaluate_llm_outputs.py co the join voi indiana_reports.csv va indiana_projections.csv.")
    add_table(doc, ["Nhom chi so", "Cach tinh"], [
        ["Problems", "TP, FP, FN, Precision/Recall/F1 micro va macro tren ontology 6 nhan; normal bao cao rieng."],
        ["Text", "BLEU-4, ROUGE-L va BERTScore cho Findings va Impression khi gold text khong rong."],
        ["Hallucination proxy", "Target label unmatched rate: nhan dich khong doi chieu duoc voi Problems gold sau chuan hoa; khong phai claim-level clinical hallucination."],
    ], [2600, 6760])

    section_heading(doc, "8. Script va artifact lien quan", 1)
    add_table(doc, ["Vai tro", "File chinh"], [
        ["Pipeline goc LLM", "scripts/run_full_pipeline.py"],
        ["Classifier -> retriever -> evidence", "scripts/test_classifier_to_retriever.py"],
        ["Rut gon evidence", "scripts/generate_minimal_json_for_llm.py"],
        ["Baseline A", "scripts/run_baseline_classifier_vision.py"],
        ["Baseline B", "scripts/run_baseline_classifier_retriever_vision.py"],
        ["Batch 206 ca", "scripts/run_leave_one_out_batch.py; scripts/run_baseline_leave_one_out_batch.py"],
        ["Prompt va schema", "src/prompts.py"],
        ["Threshold chung", "config/densenet_adaptive_thresholds.json; src/classifier_thresholds.py"],
        ["Danh gia offline", "scripts/evaluate_llm_outputs.py"],
    ], [3200, 6160])

    section_heading(doc, "9. Ket luan", 1)
    para(doc, "Ba luong co chung classifier, threshold va output contract. Baseline A tra loi cau hoi: anh va classifier co du khong? Baseline B tra loi: retrieval truc tiep co them gia tri khong? Pipeline goc tra loi: viec chuan hoa support/non-support, trang thai evidence va route co dieu kien co cai thien chat luong/chi phi so voi retrieval thuan hay khong?")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(OUT)
    print(OUT)


if __name__ == "__main__":
    main()
