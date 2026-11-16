"""Append a verified project-status section to the supplied thesis checklist DOCX."""

from __future__ import annotations

import shutil
from pathlib import Path

from docx import Document
from docx.enum.text import WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt


ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(r"E:\Downloads\NoiDungCapThietCanBoSung_KhoaLuan_CXR_RAG.docx")
DESTINATION = ROOT / "outputs" / "thesis" / "NoiDungCapThietCanBoSung_KhoaLuan_CXR_RAG_cap_nhat.docx"


def add_bullet(document: Document, text: str) -> None:
    paragraph = document.add_paragraph(style="List Bullet")
    paragraph.add_run(text)


def set_cell_shading(cell, color: str) -> None:
    properties = cell._tc.get_or_add_tcPr()
    shading = OxmlElement("w:shd")
    shading.set(qn("w:fill"), color)
    properties.append(shading)


def add_status_table(document: Document) -> None:
    table = document.add_table(rows=1, cols=3)
    table.style = "Table Grid"
    header = table.rows[0].cells
    for cell, text in zip(header, ("Hạng mục", "Trạng thái xác nhận", "Lưu ý khi viết khóa luận")):
        cell.text = text
        set_cell_shading(cell, "D9EAF7")
        for run in cell.paragraphs[0].runs:
            run.font.bold = True
    rows = (
        ("Structured reasoning + output JSON", "Đã triển khai", "Có schema problems/findings/impression và prompt ràng buộc dùng evidence được cung cấp."),
        ("Classifier + retrieval + evidence", "Đã triển khai và chạy demo", "Classifier 6 nhãn; retrieval FAISS, support/non-support theo nhãn, khử trùng study và local rerank DenseNet."),
        ("Vision/text routing", "Đã triển khai", "Mặc định thử nghiệm hiện tại dùng Luna; ca evidence bất định có thể đính kèm ảnh, ca rõ ràng chỉ dùng text."),
        ("Đánh giá tự động", "Đã có script", "Có BLEU-4, ROUGE-L, BERTScore và hallucination theo nhãn; cần chạy batch để có bảng kết quả cuối cùng."),
        ("Leave-one-study-out", "Đã cấu hình", "Manifest 206 query unique study: 30 cho sáu nhóm, 26 Pneumothorax; query study chỉ bị loại tạm thời khỏi gallery."),
        ("Grad-CAM / claim-level hallucination / ablation", "Chưa hoàn tất", "Không ghi là đã có kết quả thực nghiệm; nên giữ ở phần công việc còn lại hoặc hướng phát triển."),
    )
    for values in rows:
        cells = table.add_row().cells
        for cell, value in zip(cells, values):
            cell.text = value
            for paragraph in cell.paragraphs:
                for run in paragraph.runs:
                    run.font.size = Pt(9)


def build_document() -> Path:
    if not SOURCE.is_file():
        raise FileNotFoundError(f"Source DOCX not found: {SOURCE}")
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SOURCE, DESTINATION)
    document = Document(DESTINATION)

    page_break = document.add_paragraph()
    page_break.add_run().add_break(WD_BREAK.PAGE)
    document.add_heading("7. Cập nhật các hạng mục đã hoàn thành trong project", level=1)
    lead = document.add_paragraph()
    lead.add_run("Nguyên tắc cập nhật. ").bold = True
    lead.add_run(
        "Danh sách dưới đây chỉ ghi nhận hạng mục có mã nguồn hoặc artifact đã kiểm tra trong project. "
        "Các chỉ số đánh giá, ablation và kết luận hiệu quả chỉ nên đưa vào Chương 4 sau khi chạy batch và tổng hợp kết quả."
    )

    document.add_heading("7.1. Pipeline end-to-end đã triển khai", level=2)
    add_bullet(document, "Luồng xử lý đã kết nối: ảnh X-quang -> DenseNet121 classifier 6 nhãn -> RAD-DINO/FAISS retrieval -> evidence theo nhãn -> JSON tối giản v4 -> LLM -> JSON prediction.")
    add_bullet(document, "Pipeline lưu độc lập evidence, JSON tối giản, response thô, summary có prediction và metadata usage của API; đã chạy demo end-to-end thành công với gpt-5.6-luna.")
    add_bullet(document, "Output LLM được kiểm soát bằng JSON schema bắt buộc gồm problems, findings và impression; không chấp nhận Markdown hoặc khóa ngoài schema.")

    document.add_heading("7.2. Retrieval và structured evidence đã triển khai", level=2)
    add_bullet(document, "Retriever dùng FAISS trên embedding RAD-DINO pathology-aware; gallery có metadata study_id, MeSH/Problems và vector support cho 6 nhãn CheXpert.")
    add_bullet(document, "Với từng nhãn, hệ thống tách supporting evidence và matched non-support evidence, báo cáo margin M@3/M@5/M@10/M@20, rồi gán trạng thái SUPPORTED, MIXED, NON_SUPPORT_DOMINANT hoặc INSUFFICIENT.")
    add_bullet(document, "DenseNet local feature-map reranker đã có trong pipeline; vì vậy không nên mô tả reranking là thành phần hoàn toàn chưa triển khai. Cần báo cáo thực nghiệm riêng trước khi khẳng định mức cải thiện của nó.")
    add_bullet(document, "JSON v4 giữ label, xác suất classifier, trạng thái evidence, finding hỗ trợ và finding đối chiếu; ID/rank/path dư thừa được loại khỏi prompt để giảm token.")

    document.add_heading("7.3. Chiến lược LLM/VLM và kiểm soát chi phí đã triển khai", level=2)
    add_bullet(document, "Mặc định thực nghiệm hiện tại dùng gpt-5.6-luna cho mọi request. Evidence rõ ràng đi theo text-only; evidence MIXED, INSUFFICIENT hoặc NON_SUPPORT_DOMINANT có thể đính kèm ảnh gốc cho Luna.")
    add_bullet(document, "Prompt yêu cầu chỉ dùng finding được evidence hỗ trợ, không xem contrasting evidence là finding xác nhận, không đưa thuốc/phác đồ và xuất đúng định dạng machine-readable.")
    add_bullet(document, "Route, trạng thái evidence, ảnh có/không đính kèm và token usage được ghi vào summary của từng request để audit chi phí và hành vi hệ thống.")

    document.add_heading("7.4. Thiết lập thực nghiệm leave-one-study-out", level=2)
    add_bullet(document, "Đã tạo manifest tái lập được với seed cố định gồm 206 query study không trùng nhau: 30 ca cho Atelectasis, Cardiomegaly, Edema, Pleural Effusion, Pneumonia và normal; 26 ca Pneumothorax do gallery hiện chỉ có 26 study support > 0.")
    add_bullet(document, "Trong từng request, toàn bộ ảnh của study query bị loại tạm thời khỏi global gallery, positive bank và non-support bank. Index FAISS không bị sửa, nên study lại khả dụng cho request tiếp theo.")
    add_bullet(document, "Batch runner lưu manifest snapshot, evidence, output LLM, log stdout/stderr và batch summary trong một thư mục output riêng để phục vụ truy vết và đánh giá sau này.")

    document.add_heading("7.5. Đánh giá đã chuẩn bị", level=2)
    add_bullet(document, "Đã có script join prediction với indiana_reports.csv thông qua study_id hoặc tên ảnh; script xuất JSON tổng hợp và CSV theo từng ca.")
    add_bullet(document, "Các metric đã tích hợp: BLEU-4, ROUGE-L F1, BERTScore precision/recall/F1, precision/recall/F1 cho problems và tỷ lệ hallucination theo nhãn (nhãn LLM nêu nhưng không có trong Problems tham chiếu).")
    add_bullet(document, "Cần ghi rõ giới hạn: hallucination hiện mới đo ở mức nhãn, chưa phải claim-level đầy đủ cho từng câu trong findings/impression.")

    document.add_heading("7.6. Những hạng mục chưa nên ghi là đã hoàn thành", level=2)
    add_bullet(document, "Chưa có kết quả batch 206 ca, bảng metric tổng hợp, latency end-to-end và phân tích lỗi hoàn chỉnh.")
    add_bullet(document, "Chưa triển khai/đánh giá Grad-CAM như image evidence trong output reasoning.")
    add_bullet(document, "Chưa có ablation A/B/C hoàn chỉnh và chưa có kiểm định claim-level cho groundedness/hallucination.")
    add_bullet(document, "Các giá trị Δ_path, diễn giải DenseNet/ResNet và các chi tiết số liệu trong bản khóa luận vẫn cần đối chiếu log thực nghiệm trước khi chuẩn hóa.")

    document.add_heading("7.7. Bảng tóm tắt trạng thái", level=2)
    add_status_table(document)
    document.save(DESTINATION)
    return DESTINATION


if __name__ == "__main__":
    print(build_document())
