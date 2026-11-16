"""Create a Word report comparing the three 206-case LLM pipeline evaluations."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION_START
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "bao_cao_danh_gia_3_pipeline_206_ca.docx"

SOURCES = [
    ("Pipeline goc (Classifier + Retriever + Structured Evidence + Vision)", ROOT / "outputs/leave_one_out_206/nlp_metrics_target6.json"),
    ("Baseline A (Classifier + Vision)", ROOT / "outputs/baselines/classifier_vision_leave_one_out_206/nlp_metrics_target6.json"),
    ("Baseline B (Classifier + Retriever + Vision)", ROOT / "outputs/baselines/classifier_retriever_vision_leave_one_out_206/nlp_metrics_target6.json"),
]

BLUE = "2E74B5"
DARK_BLUE = "1F4D78"
LIGHT_BLUE = "E8EEF5"
LIGHT_GRAY = "F2F4F7"
INK = "0B2545"
CAUTION = "7A5A00"


def set_cell_shading(cell, fill: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), fill)
    tc_pr.append(shd)


def set_cell_margin(cell, top=80, start=120, bottom=80, end=120) -> None:
    tc = cell._tc
    tc_pr = tc.get_or_add_tcPr()
    tc_mar = tc_pr.first_child_found_in("w:tcMar")
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for side, value in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        node = tc_mar.find(qn(f"w:{side}"))
        if node is None:
            node = OxmlElement(f"w:{side}")
            tc_mar.append(node)
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")


def set_cell_width(cell, width_dxa: int) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    tc_w = tc_pr.find(qn("w:tcW"))
    if tc_w is None:
        tc_w = OxmlElement("w:tcW")
        tc_pr.append(tc_w)
    tc_w.set(qn("w:w"), str(width_dxa))
    tc_w.set(qn("w:type"), "dxa")


def set_table_geometry(table, widths_dxa: list[int]) -> None:
    table.alignment = WD_TABLE_ALIGNMENT.LEFT
    table.autofit = False
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
    tbl_ind.set(qn("w:w"), "120")
    tbl_ind.set(qn("w:type"), "dxa")
    grid = table._tbl.tblGrid
    for col, width in zip(grid.gridCol_lst, widths_dxa):
        col.set(qn("w:w"), str(width))
    for row in table.rows:
        for cell, width in zip(row.cells, widths_dxa):
            set_cell_width(cell, width)
            set_cell_margin(cell)
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER


def set_repeat_table_header(row) -> None:
    tr_pr = row._tr.get_or_add_trPr()
    tbl_header = OxmlElement("w:tblHeader")
    tbl_header.set(qn("w:val"), "true")
    tr_pr.append(tbl_header)


def set_run(run, *, bold=None, size=None, color=None, italic=None) -> None:
    run.font.name = "Calibri"
    run._element.rPr.rFonts.set(qn("w:ascii"), "Calibri")
    run._element.rPr.rFonts.set(qn("w:hAnsi"), "Calibri")
    if bold is not None:
        run.bold = bold
    if size is not None:
        run.font.size = Pt(size)
    if color is not None:
        run.font.color.rgb = RGBColor.from_string(color)
    if italic is not None:
        run.italic = italic


def add_cell_text(cell, text: str, *, bold=False, color=None, size=9, align=WD_ALIGN_PARAGRAPH.CENTER) -> None:
    p = cell.paragraphs[0]
    p.alignment = align
    p.paragraph_format.space_after = Pt(0)
    p.paragraph_format.line_spacing = 1.0
    run = p.add_run(str(text))
    set_run(run, bold=bold, size=size, color=color)


def add_table(doc: Document, headers: list[str], rows: list[list[str]], widths_dxa: list[int]) -> None:
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = "Table Grid"
    set_table_geometry(table, widths_dxa)
    head = table.rows[0]
    set_repeat_table_header(head)
    for cell, label in zip(head.cells, headers):
        set_cell_shading(cell, LIGHT_BLUE)
        add_cell_text(cell, label, bold=True, color=INK)
    for row in rows:
        cells = table.add_row().cells
        for idx, (cell, value) in enumerate(zip(cells, row)):
            add_cell_text(cell, value, align=WD_ALIGN_PARAGRAPH.LEFT if idx == 0 else WD_ALIGN_PARAGRAPH.CENTER)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


def add_bullet(doc: Document, text: str) -> None:
    paragraph = doc.add_paragraph(style="List Bullet")
    paragraph.paragraph_format.space_after = Pt(4)
    paragraph.paragraph_format.line_spacing = 1.17
    run = paragraph.add_run(text)
    set_run(run, size=11)


def pct(value) -> str:
    return "-" if value is None else f"{float(value) * 100:.1f}%"


def num(value) -> str:
    return "-" if value is None else f"{float(value):.3f}"


def main() -> None:
    reports = [(name, json.loads(path.read_text(encoding="utf-8"))) for name, path in SOURCES]
    doc = Document()
    section = doc.sections[0]
    section.top_margin = Inches(1)
    section.bottom_margin = Inches(1)
    section.left_margin = Inches(1)
    section.right_margin = Inches(1)
    section.header_distance = Inches(0.492)
    section.footer_distance = Inches(0.492)

    styles = doc.styles
    normal = styles["Normal"]
    normal.font.name = "Calibri"
    normal._element.rPr.rFonts.set(qn("w:ascii"), "Calibri")
    normal._element.rPr.rFonts.set(qn("w:hAnsi"), "Calibri")
    normal.font.size = Pt(11)
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.line_spacing = 1.1
    for style_name, size, color, before, after in (("Heading 1", 16, BLUE, 12, 8), ("Heading 2", 13, BLUE, 10, 6), ("Heading 3", 12, DARK_BLUE, 8, 4)):
        style = styles[style_name]
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
    header_run = header.add_run("CXR-RAG | Bao cao danh gia thuc nghiem")
    set_run(header_run, size=9, color=DARK_BLUE)
    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    footer_run = footer.add_run("Noi bo | 206 ca leave-one-study-out | 31/08/2026")
    set_run(footer_run, size=9, color="666666")

    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(3)
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    run = p.add_run("BAO CAO DANH GIA")
    set_run(run, bold=True, size=10, color=DARK_BLUE)
    title = doc.add_paragraph()
    title.paragraph_format.space_after = Pt(6)
    run = title.add_run("So sanh 3 pipeline sinh bao cao X-quang nguc")
    set_run(run, bold=True, size=22, color=INK)
    subtitle = doc.add_paragraph()
    subtitle.paragraph_format.space_after = Pt(12)
    run = subtitle.add_run("Danh gia tren 206 study co dinh, leave-one-study-out")
    set_run(run, size=12, color="666666")

    add_table(doc, ["Phan muc", "Gia tri"], [
        ["Ngay lap bao cao", date.today().strftime("%d/%m/%Y")],
        ["Quy mo", "206 truy van/pipeline; 618 dau ra tong cong"],
        ["Model sinh", "gpt-5.6-luna voi anh X-quang"],
        ["Nhãn dich", "Atelectasis, Cardiomegaly, Edema, Pleural Effusion, Pneumonia, Pneumothorax; normal"],
        ["Nguon tham chieu", "IU X-Ray reports: Problems, Findings va Impression"],
    ], [2700, 6660])

    doc.add_heading("Ket luan chinh", level=1)
    original = reports[0][1]["metrics"]["target_problems"]
    baseline_a = reports[1][1]["metrics"]["target_problems"]
    baseline_b = reports[2][1]["metrics"]["target_problems"]
    add_bullet(doc, f"Baseline B (classifier + retriever + vision) cao nhat: F1 vi mo {num(baseline_b['micro_f1'])}, F1 macro {num(baseline_b['macro_f1'])} va F1 quyet dinh normal {num(baseline_b['normal_decision']['f1'])}.")
    add_bullet(doc, f"So voi pipeline goc, Baseline B tang {baseline_b['micro_f1'] - original['micro_f1']:+.3f} F1 vi mo va {baseline_b['macro_f1'] - original['macro_f1']:+.3f} F1 macro tren cung 206 ca.")
    add_bullet(doc, "Cac chi so van ban cao hon cac chi so nhan benh: BERTScore Impression dat khoang 0.875-0.878, trong khi BLEU-4 thap (0.019-0.027). Day la dac diem thuong gap khi bao cao dong nghia/khac cau truc.")
    add_bullet(doc, "Cac nhan Edema (4 ca gold) va Pneumonia (8 ca gold) co support rat nho; khong nen suy dien xep hang rieng tung nhan tu cac gia tri nay.")

    doc.add_heading("1. Thiet ke va pham vi danh gia", level=1)
    doc.add_paragraph("Ba pipeline dung cung manifest 206 study, cung anh truy van, cung classifier DenseNet va cung ontology dich 6 nhan. Pipeline goc bo sung structured evidence; Baseline A dua xac suat classifier va anh truc tiep vao LLM; Baseline B dua xac suat classifier, tom tat 5 ca truy hoi gan nhat va anh vao LLM.")
    add_table(doc, ["Ma", "Cau hinh dau vao LLM"], [
        ["Pipeline goc", "Structured evidence tu classifier + retriever, sau do LLM Vision"],
        ["Baseline A", "Xac suat classifier + anh; khong structured evidence, khong retriever"],
        ["Baseline B", "Xac suat classifier + 5 ca retriever (similarity/feature map/tom tat) + anh; khong structured evidence"],
    ], [2100, 7260])
    doc.add_paragraph("Evaluation doc tat ca 206 summary JSON cua moi pipeline. Khong co mau bi bo qua. Problems duoc chuan hoa ve ontology co dinh; Findings va Impression duoc doi chieu voi report CSV theo study ID.")

    doc.add_heading("2. Ket qua nhan benh (Problems)", level=1)
    headers = ["Pipeline", "TP", "FP", "FN", "Precision", "Recall", "F1 micro", "F1 macro", "Grounded", "Unmatched"]
    rows = []
    for name, report in reports:
        p = report["metrics"]["target_problems"]
        rows.append([name.replace("Pipeline goc ", "Goc ").replace("Baseline ", ""), str(p["tp"]), str(p["fp"]), str(p["fn"]), num(p["micro_precision"]), num(p["micro_recall"]), num(p["micro_f1"]), num(p["macro_f1"]), pct(p["target_label_groundedness"]), pct(p["target_label_unmatched_rate"])])
    add_table(doc, headers, rows, [2100, 500, 500, 500, 760, 700, 760, 760, 850, 930])
    note = doc.add_paragraph()
    note.paragraph_format.space_after = Pt(6)
    run = note.add_run("Ghi chu: ")
    set_run(run, bold=True, size=9, color=CAUTION)
    run = note.add_run("Grounded = 1 - unmatched rate, la ty le nhan dich duoc doi chieu voi Problems trong report. Day la proxy hallucination o cap nhan, khong phai kiem tra clinical claim o cap cau.")
    set_run(run, size=9, color=CAUTION)

    doc.add_heading("3. Quyết dinh normal", level=1)
    rows = []
    for name, report in reports:
        p = report["metrics"]["target_problems"]["normal_decision"]
        rows.append([name.replace("Pipeline goc ", "Goc ").replace("Baseline ", ""), str(p["tp"]), str(p["fp"]), str(p["fn"]), str(p["tn"]), num(p["precision"]), num(p["recall"]), num(p["f1"]), num(p["accuracy"])])
    add_table(doc, ["Pipeline", "TP", "FP", "FN", "TN", "Precision", "Recall", "F1", "Accuracy"], rows, [2700, 600, 600, 600, 600, 1000, 900, 700, 1260])
    doc.add_paragraph("Normal dung khi report gold khong co nhan dich va model khong du doan bat ky nhan dich nao. Do do, F1 normal duoc bao cao rieng, khong tron vao F1 cua 6 benh.")

    doc.add_heading("4. Chi so van ban", level=1)
    headers = ["Pipeline", "Findings: BLEU-4", "Findings: ROUGE-L", "Findings: BERTScore F1", "Impression: BLEU-4", "Impression: ROUGE-L", "Impression: BERTScore F1"]
    rows = []
    for name, report in reports:
        m = report["metrics"]
        f = m["findings"]["metrics_on_nonempty_reference"]
        i = m["impression"]["metrics_on_nonempty_reference"]
        rows.append([name.replace("Pipeline goc ", "Goc ").replace("Baseline ", ""), num(f["bleu4"]), num(f["rouge_l_f1"]), num(f["bertscore_f1"]), num(i["bleu4"]), num(i["rouge_l_f1"]), num(i["bertscore_f1"])])
    add_table(doc, headers, rows, [2100, 1150, 1250, 1450, 1150, 1250, 1010])
    doc.add_paragraph("Findings chi tinh tren 163/206 report co truong tham chieu khong rong (79.1%). Impression tinh tren 204/206 report co tham chieu khong rong (99.0%). Mau tham chieu rong duoc loai khoi trung binh, khong bi gan diem 0.")

    doc.add_heading("5. F1 theo tung nhan dich", level=1)
    labels = reports[0][1]["ontology"]["target_labels"]
    rows = []
    for label in labels:
        row = [label]
        for _, report in reports:
            item = report["metrics"]["target_problems"]["per_target_label"][label]
            row.extend([str(item["support"]), num(item["f1"])])
        rows.append(row)
    add_table(doc, ["Nhan", "Support", "Goc F1", "Support", "A F1", "Support", "B F1"], rows, [2400, 850, 1100, 850, 1100, 850, 1100])

    doc.add_heading("6. Cach dien giai va gioi han", level=1)
    add_bullet(doc, "F1 micro tong hop TP/FP/FN cua 6 nhan; F1 macro lay trung binh F1 cua cac nhan va nhay cam voi lop co support nho.")
    add_bullet(doc, "BLEU-4 va ROUGE-L do lap tu; BERTScore (roberta-large) do tuong dong ngu nghia. Chung khong the thay the danh gia chuyen mon X-quang.")
    add_bullet(doc, "Unmatched rate la proxy hallucination nhan: nhan dich model tra ra nhung khong xuat hien trong Problems gold sau chuan hoa. No khong ket luan rang toan bo cau mo ta la hallucination.")
    add_bullet(doc, "Ket qua la so sanh tren manifest co dinh 206 ca, khong phai ket qua tong quat hoa tren tap test doc lap. Nen bo sung bootstrap confidence interval va danh gia bac si cho ket luan cuoi cung.")

    doc.add_heading("Phu luc: Tep ket qua", level=1)
    for name, path in SOURCES:
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(3)
        run = p.add_run(f"{name}: ")
        set_run(run, bold=True, size=9)
        run = p.add_run(str(path.relative_to(ROOT)))
        set_run(run, size=9, color="555555")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(OUT)
    print(OUT)


if __name__ == "__main__":
    main()
