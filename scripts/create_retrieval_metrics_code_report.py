"""Build a Word note explaining the retrieval evaluation metric code."""

from pathlib import Path

from docx import Document
from docx.enum.style import WD_STYLE_TYPE
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

from create_embedding_vector_report import (
    BLUE,
    DARK_BLUE,
    INK,
    LIGHT_GRAY,
    MUTED,
    add_body,
    add_label_paragraph,
    add_table,
    configure_styles,
    set_header_footer,
    set_run_font,
)


OUT_PATH = Path("docs/Huong_dan_Code_Danh_gia_Retrieval_TopK.docx")


def shade_paragraph(paragraph, fill=LIGHT_GRAY):
    p_pr = paragraph._p.get_or_add_pPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), fill)
    p_pr.append(shd)


def add_code(doc, code):
    paragraph = doc.add_paragraph(style="Code Block")
    paragraph.paragraph_format.space_before = Pt(3)
    paragraph.paragraph_format.space_after = Pt(8)
    paragraph.paragraph_format.left_indent = Inches(0.08)
    paragraph.paragraph_format.right_indent = Inches(0.08)
    shade_paragraph(paragraph)
    run = paragraph.add_run(code.strip())
    run.font.name = "Courier New"
    run._element.rPr.rFonts.set(qn("w:ascii"), "Courier New")
    run._element.rPr.rFonts.set(qn("w:hAnsi"), "Courier New")
    run.font.size = Pt(8.5)
    return paragraph


def configure_code_style(doc):
    if "Code Block" not in doc.styles:
        style = doc.styles.add_style("Code Block", WD_STYLE_TYPE.PARAGRAPH)
    else:
        style = doc.styles["Code Block"]
    style.font.name = "Courier New"
    style._element.rPr.rFonts.set(qn("w:ascii"), "Courier New")
    style._element.rPr.rFonts.set(qn("w:hAnsi"), "Courier New")
    style.font.size = Pt(8.5)
    style.paragraph_format.space_after = Pt(8)
    style.paragraph_format.line_spacing = 1.0


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
    configure_code_style(doc)
    set_header_footer(section)

    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(16)
    p.paragraph_format.space_after = Pt(4)
    run = p.add_run("PHỤ LỤC MÃ NGUỒN")
    set_run_font(run, size=10, bold=True, color=BLUE)
    title = doc.add_paragraph(style="Title")
    title.add_run("Code đánh giá độ liên quan truy xuất Top-K")
    subtitle = doc.add_paragraph(style="Subtitle")
    subtitle.add_run("Trích và chuẩn hóa logic từ scripts/evaluate_pathology_retrieval.py")

    doc.add_heading("Phạm vi", level=1)
    add_body(
        doc,
        "Tài liệu chỉ trình bày các hàm tạo relevance và tính Precision@K, mAP@K, nDCG@K, Delta_path, cùng đoạn gọi hàm đánh giá. Phần đọc file embedding, argparse và ghi CSV/JSON của project được lược bỏ.",
    )
    add_label_paragraph(doc, "Lưu ý. ", "Trong script production, Precision, nDCG và Delta_path được viết inline trong vòng lặp. Ở đây chúng được tách thành hàm để dễ đưa vào báo cáo, nhưng giữ nguyên công thức và điều kiện lọc.")

    doc.add_heading("1. Tạo pathology relevance từ support", level=1)
    add_body(
        doc,
        "Mỗi ảnh có vector support sáu phần tử. Relevance giữa query và từng ảnh gallery là weighted Jaccard: tổng min theo nhãn chia cho tổng max theo nhãn. Giá trị thuộc [0, 1]. Một kết quả được xem là relevant cho Precision/mAP khi relevance >= 0,5.",
    )
    add_code(doc, '''
import numpy as np

def weighted_jaccard(query_support, gallery_support):
    """gallery_support has shape [N, 6]."""
    numerator = np.minimum(query_support, gallery_support).sum(axis=1)
    denominator = np.maximum(query_support, gallery_support).sum(axis=1)
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 0,
    )

def binary_relevance_from_support(query_support, gallery_support,
                                  threshold=0.5):
    graded_relevance = weighted_jaccard(query_support, gallery_support)
    binary_relevance = (graded_relevance >= threshold).astype(np.float32)
    return graded_relevance, binary_relevance
''')

    doc.add_heading("2. Các hàm metric", level=1)
    add_body(
        doc,
        "Các hàm sau nhận danh sách relevance đã được sắp xếp giảm dần theo cosine similarity. binary_relevance dùng cho Precision và AP; graded_relevance dùng cho nDCG và Delta_path.",
    )
    add_code(doc, '''
def precision_at_k(binary_relevance, k):
    top_k = binary_relevance[:k]
    return float(top_k.mean()) if len(top_k) else float("nan")

def average_precision_at_k(binary_relevance, total_relevant, k):
    if total_relevant == 0:
        return float("nan")
    top_k = binary_relevance[:k]
    hits = np.cumsum(top_k)
    precision_at_rank = hits / np.arange(1, len(top_k) + 1)
    return float(
        (precision_at_rank * top_k).sum() / min(total_relevant, k)
    )

def map_at_k(ap_scores):
    """Mean AP@K over all valid queries."""
    return float(np.nanmean(ap_scores)) if ap_scores else float("nan")
''')
    add_code(doc, '''
def dcg(graded_relevance):
    if not len(graded_relevance):
        return 0.0
    discounts = np.log2(np.arange(2, len(graded_relevance) + 2))
    return float((graded_relevance / discounts).sum())

def ndcg_at_k(graded_relevance, k):
    actual = graded_relevance[:k]
    ideal = np.sort(graded_relevance)[::-1][:k]
    ideal_dcg = dcg(ideal)
    return dcg(actual) / ideal_dcg if ideal_dcg else float("nan")

def pathology_delta(graded_relevance, ranked_similarity, threshold=0.5):
    same = ranked_similarity[graded_relevance >= threshold]
    different = ranked_similarity[graded_relevance == 0]
    if not len(same) or not len(different):
        return float("nan")
    return float(same.mean() - different.mean())
''')

    doc.add_heading("3. Lọc leakage và xếp hạng study-level", level=1)
    add_body(
        doc,
        "Trước khi tính metric, evaluator loại ảnh cùng study với query. Sau đó, với mỗi study còn lại chỉ giữ ứng viên có cosine similarity cao nhất. Điều này tránh hai ảnh của cùng một study chiếm nhiều vị trí trong Top-K.",
    )
    add_code(doc, '''
def study_key(row):
    dataset = str(row.get("dataset", "") or "").strip()
    study_id = str(row.get("study_id", row.get("uid", "")) or "").strip()
    return (dataset, study_id or f"image:{row.get('id', '')}")

def rank_unique_studies(similarities, gallery_rows, query_row):
    order = np.argsort(-similarities, kind="stable")
    query_study = study_key(query_row)
    seen_studies = set()
    ranked_indices = []

    for candidate_index in order:
        candidate_study = study_key(gallery_rows[candidate_index])
        if candidate_study == query_study:
            continue
        if candidate_study in seen_studies:
            continue
        seen_studies.add(candidate_study)
        ranked_indices.append(candidate_index)
    return np.asarray(ranked_indices, dtype=np.int64)
''')

    doc.add_heading("4. Đoạn evaluate gọi các hàm", level=1)
    add_body(
        doc,
        "Đoạn dưới là phần cốt lõi của evaluate. gallery_embeddings và query_embeddings phải được L2-normalize trước đó; vì thế phép nhân ma trận tạo cosine similarity. metrics_by_k tích lũy score theo query để cuối cùng lấy trung bình.",
    )
    add_code(doc, '''
def evaluate(query_embeddings, query_rows, query_support,
             gallery_embeddings, gallery_rows, gallery_support,
             ks=(5, 10), threshold=0.5):
    metrics_by_k = {k: {"precision": [], "ap": [], "ndcg": []}
                    for k in ks}
    delta_scores = []

    eligible_queries = np.flatnonzero(query_support.sum(axis=1) > 0)
    for query_index in eligible_queries:
        similarities = gallery_embeddings @ query_embeddings[query_index]
        relevance, _ = binary_relevance_from_support(
            query_support[query_index], gallery_support, threshold
        )

        ranked_indices = rank_unique_studies(
            similarities, gallery_rows, query_rows[query_index]
        )
        ranked_relevance = relevance[ranked_indices]
        ranked_similarity = similarities[ranked_indices]
        binary = (ranked_relevance >= threshold).astype(np.float32)
        total_relevant = int(binary.sum())
        if total_relevant == 0:
            continue

        for k in ks:
            metrics_by_k[k]["precision"].append(precision_at_k(binary, k))
            metrics_by_k[k]["ap"].append(
                average_precision_at_k(binary, total_relevant, k)
            )
            metrics_by_k[k]["ndcg"].append(ndcg_at_k(ranked_relevance, k))

        delta = pathology_delta(ranked_relevance, ranked_similarity, threshold)
        if not np.isnan(delta):
            delta_scores.append(delta)

    return {
        **{f"Precision@{k}": float(np.nanmean(metrics_by_k[k]["precision"]))
           for k in ks},
        **{f"mAP@{k}": map_at_k(metrics_by_k[k]["ap"]) for k in ks},
        **{f"nDCG@{k}": float(np.nanmean(metrics_by_k[k]["ndcg"]))
           for k in ks},
        "Delta_path": float(np.nanmean(delta_scores)),
    }
''')

    doc.add_heading("5. Lệnh gọi trong project", level=1)
    add_code(doc, '''
D:\\Miniconda3\\python.exe scripts\\evaluate_pathology_retrieval.py ^
  --gallery_dir "data\\embeddings_pathology_pretrained" ^
  --output_dir "data\\evaluation_pretrained" ^
  --k 5 10 ^
  --relevance_threshold 0.5
''')
    add_body(
        doc,
        "Lệnh trên sinh summary.json chứa mean metric toàn tập query và per_query.csv chứa metric từng query. Trong PowerShell, có thể thay ký tự ^ bằng backtick (`) khi viết lệnh nhiều dòng.",
    )

    doc.add_heading("6. Diễn giải nhanh", level=1)
    add_table(
        doc,
        ["Metric", "Giá trị cao biểu thị"],
        [
            ("Precision@5, Precision@10", "Nhiều study relevant trong danh sách đầu."),
            ("mAP@5, mAP@10", "Study relevant xuất hiện sớm trong xếp hạng."),
            ("nDCG@5, nDCG@10", "Các mức overlap pathology cao được xếp ở vị trí tốt."),
            ("Delta_path", "Cặp có pathology overlap có cosine similarity cao hơn cặp không overlap."),
        ],
        [3000, 6360],
    )
    add_label_paragraph(doc, "Giới hạn. ", "Metric dựa trên support metadata của IU-Xray. support = 0 là không quan sát được target support, không khẳng định âm tính; vì vậy các metric này không thay thế đánh giá lâm sàng.")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    doc.save(OUT_PATH)
    print(OUT_PATH.resolve())


if __name__ == "__main__":
    build_document()
