"""Build the Word report for vector database construction and Top-K retrieval results."""

from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt

from create_embedding_vector_report import (
    BLUE,
    DARK_BLUE,
    INK,
    MUTED,
    add_body,
    add_label_paragraph,
    add_table,
    configure_styles,
    set_header_footer,
    set_run_font,
)


OUT_PATH = Path("docs/Bao_cao_Ket_qua_Vector_Database_TopK.docx")


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
    run = p.add_run("BÁO CÁO KẾT QUẢ")
    set_run_font(run, size=10, bold=True, color=BLUE)

    title = doc.add_paragraph(style="Title")
    title.add_run("Kết quả xây dựng vector database và đánh giá độ liên quan truy xuất Top-K")
    subtitle = doc.add_paragraph(style="Subtitle")
    subtitle.add_run("IU-Xray image retrieval với RAD-DINO pathology-aware embedding")

    add_table(
        doc,
        ["Nội dung", "Kết quả"],
        [
            ("Vector database", "FAISS IndexFlatIP trên embedding 512-D L2-normalized"),
            ("Quy mô gallery", "3.818 ảnh Frontal, 3.689 study"),
            ("Checkpoint", "pathology_projection_rad_dino_pretrained.pt, fine-tune 10 epoch"),
            ("Query đánh giá", "1.246 query có support bệnh lý hợp lệ"),
            ("Độ liên quan", "Weighted Jaccard(support query, support gallery) >= 0,5"),
            ("Loại trừ leakage", "Loại cùng study và khử trùng theo dataset + study ID"),
        ],
        [2700, 6660],
    )

    doc.add_heading("1. Mục tiêu đánh giá", level=1)
    add_body(
        doc,
        "Đánh giá nhằm kiểm tra liệu vector database có trả về các study có mức hỗ trợ bệnh lý tương đồng với ảnh truy vấn hay không. Bài toán được đo ở Top-5 và Top-10 nhằm phản ánh hai tình huống sử dụng: danh sách ngắn để xem nhanh và danh sách dài hơn để tham khảo ca tương tự.",
    )
    add_body(
        doc,
        "Kết quả dùng relevance suy ra từ metadata support của IU-Xray. Đây là proxy có cấu trúc cho mức tương đồng bệnh lý, không phải ground truth chẩn đoán lâm sàng độc lập.",
    )

    doc.add_heading("2. Kết quả xây dựng vector database", level=1)
    add_body(
        doc,
        "Vector database cuối cùng nằm trong data/embeddings_pathology_pretrained. Mỗi vector được sinh bằng RAD-DINO MAIRA-2 đã fine-tune hai transformer block cuối, fusion CLS/local patch token và projection head 512-D. Sau L2-normalization, FAISS IndexFlatIP thực hiện exact nearest-neighbor search bằng inner product, tương đương cosine similarity.",
    )
    add_table(
        doc,
        ["Artefact", "Kích thước / số lượng", "Vai trò"],
        [
            ("embeddings.npy", "3.818 x 512 float32; 7,82 MB", "Ma trận embedding của gallery"),
            ("faiss.index", "512-D; 3.818 vector; 7,82 MB", "Exact Top-K similarity search"),
            ("ids.txt", "3.818 image ID", "Ánh xạ thứ tự FAISS sang ảnh"),
            ("metadata.jsonl", "3.818 dòng; 1,14 MB", "Study ID, view, MeSH, problems, support"),
            ("embedding_manifest.json", "Model và checkpoint metadata", "Xác thực tương thích khi retriever nạp index"),
        ],
        [2200, 3100, 4060],
    )
    add_label_paragraph(doc, "Tính toàn vẹn metadata. ", "Sau đồng bộ, 3.818/3.818 dòng metadata có MeSH và Problems. Việc đồng bộ chỉ cập nhật metadata, không thay đổi vector hoặc FAISS index.")
    add_label_paragraph(doc, "Tính toàn vẹn retriever. ", "Khi nạp index, hệ thống kiểm tra dimension, số vector, model revision và checkpoint projection trong manifest trước khi tìm kiếm.")

    doc.add_heading("3. Thiết kế đánh giá Top-K", level=1)
    add_body(
        doc,
        "Gallery và query cùng lấy từ 3.818 ảnh, nhưng mỗi query loại các ảnh thuộc cùng dataset và study ID. Kết quả cũng được khử trùng study-level để một study không xuất hiện nhiều lần trong Top-K. Có 1.247 ảnh có support khác 0; một query không còn ứng viên hợp lệ sau loại trừ nên 1.246 query được đánh giá.",
    )
    add_table(
        doc,
        ["Metric", "Ý nghĩa trong thí nghiệm"],
        [
            ("Precision@K", "Tỷ lệ study liên quan trong K kết quả đầu."),
            ("mAP@K", "Độ chính xác trung bình có xét vị trí của các kết quả liên quan."),
            ("nDCG@K", "Chất lượng xếp hạng có xét relevance graded và thứ hạng."),
            ("Delta_path", "Mức cải thiện pathology-support so với baseline vector hình ảnh."),
        ],
        [2500, 6860],
    )
    add_label_paragraph(doc, "Ngưỡng relevance. ", "Một study gallery được xem là liên quan khi weighted Jaccard giữa hai vector support đạt từ 0,5. Support bằng 0 biểu thị không quan sát được target support trong metadata, không được hiểu là âm tính lâm sàng.")

    doc.add_heading("4. Kết quả định lượng", level=1)
    add_body(
        doc,
        "Bảng dưới so sánh các artifact đã được đánh giá với cùng protocol, cùng 1.246 query và cùng ngưỡng relevance. Cấu hình hiện dùng là RAD-DINO pathology-aware pretrained 512-D.",
    )
    add_table(
        doc,
        ["Embedding", "P@5", "P@10", "mAP@5", "mAP@10", "nDCG@5", "nDCG@10", "Delta_path"],
        [
            ("RAD-DINO raw 768-D", "9,98%", "9,93%", "5,67%", "4,15%", "11,69%", "11,60%", "0,012"),
            ("DenseNet pathology 512-D", "15,10%", "14,76%", "9,89%", "7,90%", "16,57%", "16,44%", "0,033"),
            ("RAD-DINO projection 512-D", "18,44%", "18,27%", "11,90%", "9,77%", "18,80%", "18,77%", "0,054"),
            ("RAD-DINO pretrained 512-D", "45,01%", "44,28%", "35,93%", "32,66%", "43,33%", "43,01%", "0,307"),
        ],
        [2050, 1045, 1045, 1045, 1045, 1045, 1045, 1040],
    )
    add_label_paragraph(doc, "So với RAD-DINO raw. ", "Cấu hình final tăng Precision@5 từ 9,98% lên 45,01% (+35,02 điểm phần trăm), mAP@5 từ 5,67% lên 35,93% (+30,27 điểm phần trăm) và nDCG@5 từ 11,69% lên 43,33% (+31,64 điểm phần trăm).")
    add_label_paragraph(doc, "So với projection 512-D trước fine-tuning. ", "Precision@5 tăng 26,57 điểm phần trăm và nDCG@5 tăng 24,53 điểm phần trăm. Điều này cho thấy fine-tuning và CLS/local-patch fusion đóng góp đáng kể trong protocol hiện tại.")

    doc.add_heading("5. Phân tích chất lượng Top-K", level=1)
    add_body(
        doc,
        "Với cấu hình final, Precision@5 đạt 45,01% và Precision@10 đạt 44,28%. Kết quả Top-5 tốt hơn nhẹ vì các vị trí đầu có độ liên quan cao hơn; khi mở rộng sang 10 kết quả, các ứng viên bổ sung làm precision giảm 0,73 điểm phần trăm. Xu hướng tương tự xuất hiện ở mAP và nDCG, là hành vi phù hợp với xếp hạng theo độ tương đồng.",
    )
    add_body(
        doc,
        "Median Precision@5 là 40%, median nDCG@5 là 39,13% và median Delta_path là 0,318. Có 985/1.246 query (79,05%) có ít nhất một kết quả liên quan trong Top-5; tỷ lệ này tăng lên 1.082/1.246 (86,84%) trong Top-10. Các con số cho thấy chất lượng không đồng đều giữa các query, do đó hệ thống nên hiển thị nhiều kết quả kèm metadata thay vì chỉ dựa vào Top-1.",
    )
    add_body(
        doc,
        "Delta_path trung bình 0,307 cao hơn các artifact baseline, phù hợp với mục tiêu làm embedding nhạy hơn với support bệnh lý. Tuy nhiên metric này vẫn phụ thuộc vào quy tắc gán support từ metadata, vì vậy cần xem các ví dụ retrieval cụ thể và đánh giá chuyên gia trước khi suy rộng sang sử dụng lâm sàng.",
    )

    doc.add_heading("6. Kết luận", level=1)
    add_body(
        doc,
        "Vector database FAISS 512-D đã được xây dựng hoàn chỉnh, có kiểm tra tương thích checkpoint và metadata đủ MeSH/Problems. Trong protocol nội bộ, RAD-DINO pathology-aware pretrained đạt kết quả Top-K tốt nhất trong các artifact đang có, với P@5 45,01%, mAP@5 35,93% và nDCG@5 43,33%.",
    )
    add_body(
        doc,
        "Bước tiếp theo phù hợp là đánh giá theo split độc lập ở mức patient/study, kiểm tra cross-domain, phân tích lỗi theo từng nhãn bệnh và thực hiện review lâm sàng đối với các ca Top-K. Các kết quả hiện tại nên được trình bày là đánh giá retrieval theo pathology-support proxy trong IU-Xray.",
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    doc.save(OUT_PATH)
    print(OUT_PATH.resolve())


if __name__ == "__main__":
    build_document()
