"""Append a verified project-compliance matrix to the supplied thesis summary DOCX."""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

from docx import Document
from docx.enum.text import WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt


ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(r"E:\Downloads\TomTat_NoiDungCanHoanThien_KhoaLuan_CXR_RAG.docx")
DESTINATION = ROOT / "outputs" / "thesis" / "TomTat_NoiDungCanHoanThien_KhoaLuan_CXR_RAG_doi_chieu_project.docx"
EXPERIMENT_ROOT = ROOT / "outputs" / "leave_one_out_206"


def shade(cell, color: str) -> None:
    properties = cell._tc.get_or_add_tcPr()
    node = OxmlElement("w:shd")
    node.set(qn("w:fill"), color)
    properties.append(node)


def add_bullet(document: Document, text: str) -> None:
    paragraph = document.add_paragraph(style="List Bullet")
    paragraph.add_run(text)


def batch_snapshot() -> dict[str, int | bool]:
    evidence = len(list((EXPERIMENT_ROOT / "evidence").glob("*.json"))) if (EXPERIMENT_ROOT / "evidence").exists() else 0
    summaries = len(list((EXPERIMENT_ROOT / "llm").glob("*/summary_*.json"))) if (EXPERIMENT_ROOT / "llm").exists() else 0
    return {
        "evidence": evidence,
        "summaries": summaries,
        "finished": (EXPERIMENT_ROOT / "batch_summary.json").is_file(),
    }


def add_matrix(document: Document, snapshot: dict[str, int | bool]) -> None:
    table = document.add_table(rows=1, cols=4)
    table.style = "Table Grid"
    headers = ("Yêu cầu trong tài liệu", "Trạng thái", "Bằng chứng từ project", "Cách ghi an toàn trong khóa luận")
    for cell, value in zip(table.rows[0].cells, headers):
        cell.text = value
        shade(cell, "D9EAF7")
        for run in cell.paragraphs[0].runs:
            run.font.bold = True
    rows = (
        ("Pipeline end-to-end", "ĐÁP ỨNG", "run_full_pipeline.py; artifact evidence/minimal/summary; demo Luna đã chạy.", "Mô tả pipeline đã triển khai; chỉ trích metric sau khi batch hoàn tất."),
        ("Structured evidence, status và JSON schema", "ĐÁP ỨNG", "label evidence support/non-support, status SUPPORTED/MIXED/NON_SUPPORT_DOMINANT/INSUFFICIENT; prompts.py strict schema.", "Có thể mô tả phương pháp và controlled output; traceability hiện ở cấp evidence/candidate, chưa gắn ID cho từng claim LLM."),
        ("Local reranking", "ĐÁP ỨNG (mã nguồn)", "densenet_local_reranker.py được gọi trong classifier-to-retriever pipeline.", "Không khẳng định cải thiện cho đến khi có ablation hoặc metric reranker."),
        ("Text/vision routing và token audit", "ĐÁP ỨNG", "select_llm_route, Luna text/vision; summary lưu route, image_attached và usage.", "Nêu đây là cơ chế kiểm soát chi phí; báo cáo chi phí batch từ log sau khi hoàn tất."),
        ("Freeze protocol và leave-one-study-out", "ĐÁP ỨNG MỘT PHẦN", "Manifest seed 20260830, 206 study unique; retriever tạm exclude study_id khỏi mọi bank.", "Có thể khẳng định self-retrieval leakage được ngăn; chưa khẳng định sạch với dữ liệu fine-tune embedding."),
        ("Data leakage với fine-tune RAD-DINO/projection", "CHƯA XÁC MINH", "Chưa có artifact mapping 206 study với tập train của projection checkpoint.", "Giữ là việc bắt buộc trước khi kết luận protocol hoàn toàn study-disjoint."),
        ("BLEU/ROUGE-L/BERTScore/Problems F1/hallucination", "ĐÁP ỨNG (script)", "evaluate_llm_outputs.py có BLEU-4, ROUGE-L, BERTScore, Problems F1 và hallucination nhãn.", "Không báo cáo số cuối cùng nếu batch chưa hoàn tất; hallucination hiện là label-level, không phải claim-level."),
        ("Batch 206", "ĐANG CHẠY" if not snapshot["finished"] else "ĐÃ KẾT THÚC", f"Snapshot khi tạo tài liệu: {snapshot['evidence']} evidence, {snapshot['summaries']} LLM summary; batch_summary={'có' if snapshot['finished'] else 'chưa có'}.", "Chỉ đưa bảng metric/latency/cost khi batch_summary và evaluation aggregate đã được tạo."),
        ("Ablation A/B/C, Grad-CAM, claim-level groundedness", "CHƯA HOÀN TẤT", "Không có artifact kết quả A/B/C, Grad-CAM hoặc claim annotation hiện tại.", "Không mô tả là kết quả đạt được; giữ ở kế hoạch thực nghiệm/hướng phát triển."),
        ("Ground truth, uncertainty và reproducibility", "ĐÁP ỨNG MỘT PHẦN", "Metadata có Problems/MeSH/support vector; threshold, seed, checkpoint và manifest có trong config/script.", "Bổ sung mô tả Normal/6 nhãn, hardware/package version và công thức confidence trước khi freeze báo cáo."),
    )
    colors = {"ĐÁP ỨNG": "E2F0D9", "ĐÁP ỨNG (mã nguồn)": "E2F0D9", "ĐÁP ỨNG (script)": "E2F0D9", "ĐÁP ỨNG MỘT PHẦN": "FFF2CC", "ĐANG CHẠY": "FFF2CC", "ĐÃ KẾT THÚC": "E2F0D9", "CHƯA XÁC MINH": "FCE4D6", "CHƯA HOÀN TẤT": "FCE4D6"}
    for row in rows:
        cells = table.add_row().cells
        for cell, value in zip(cells, row):
            cell.text = value
            for paragraph in cell.paragraphs:
                for run in paragraph.runs:
                    run.font.size = Pt(8.5)
        shade(cells[1], colors[row[1]])


def build_document() -> Path:
    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)
    snapshot = batch_snapshot()
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SOURCE, DESTINATION)
    document = Document(DESTINATION)
    document.add_paragraph().add_run().add_break(WD_BREAK.PAGE)
    document.add_heading("9. Đối chiếu yêu cầu với trạng thái project hiện tại", level=1)
    document.add_paragraph(
        "Cập nhật ngày " + datetime.now().strftime("%d/%m/%Y %H:%M") +
        ". Trạng thái dưới đây được suy ra từ mã nguồn, manifest và artifact hiện có; "
        "không coi một script chưa chạy hoặc một batch đang chạy là kết quả thực nghiệm cuối cùng."
    )
    document.add_heading("9.1. Các nội dung có thể đưa vào phần đã triển khai", level=2)
    add_bullet(document, "Hệ thống end-to-end đã liên thông classifier DenseNet121 sáu nhãn, RAD-DINO/FAISS retrieval, construction of support/non-support evidence, JSON tối giản v4 và LLM output có cấu trúc.")
    add_bullet(document, "Structured reasoning có state machine evidence, controlled prompt và strict JSON output. Prompt giới hạn finding theo evidence hỗ trợ và tách contrasting evidence khỏi finding xác nhận.")
    add_bullet(document, "Local DenseNet feature-map reranker, route Luna text/vision, audit token usage và logging artifact đã tồn tại trong pipeline.")
    add_bullet(document, "Protocol 206 leave-one-study-out đã freeze bằng manifest seed cố định; toàn bộ ảnh cùng study với query bị lọc khỏi global/support/non-support gallery trong chính lượt đó.")
    document.add_heading("9.2. Ma trận đối chiếu", level=2)
    add_matrix(document, snapshot)
    document.add_heading("9.3. Nội dung nên bổ sung ngay vào báo cáo", level=2)
    add_bullet(document, "Mô tả rõ support vector sáu nhãn từ MeSH/Problems/findings, quy tắc trạng thái evidence và dữ liệu được phép vào LLM.")
    add_bullet(document, "Ghi protocol 206: 30 study cho sáu nhóm và 26 Pneumothorax do chỉ có 26 study support > 0; seed, manifest, checkpoint, threshold và route phải được đưa vào phụ lục/tái lập.")
    add_bullet(document, "Nêu rõ self-retrieval leakage đã được chặn theo study_id, đồng thời ghi rằng leakage với dữ liệu train projection vẫn phải kiểm tra riêng trước khi chốt kết luận.")
    add_bullet(document, "Trong Chương 4, chỉ điền BLEU-4/ROUGE-L/BERTScore/Problems F1/hallucination, latency và cost sau khi batch và script đánh giá tạo metric aggregate cuối cùng.")
    document.save(DESTINATION)
    return DESTINATION


if __name__ == "__main__":
    print(build_document())
