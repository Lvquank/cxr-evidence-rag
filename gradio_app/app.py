"""Native Gradio workspace over the existing CXR-RAG research pipelines."""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.environ.setdefault("GRADIO_TEMP_DIR", str(ROOT / "tmp/gradio"))
sys.path.insert(0, str(ROOT))

import gradio as gr
from gradio_app import backend

METHODS = backend.METHODS
METHOD_CHOICES = [(name, key) for key, name in METHODS.items()]
MODE_COMPARE = "So sánh cả 4"
STATUSES = {"queued": "Đang chuẩn bị", "running": "Đang xử lý", "complete": "Hoàn tất", "partial": "Có lỗi một phần", "failed": "Thất bại", "interrupted": "Đã gián đoạn"}


def safe_text(value):
    return re.sub(r"([\\`*_{}\[\]()#+.!<>|])", r"\\\1", str(value or ""))


def health_view():
    health = backend.probe_backend()
    label = "CUDA đã sẵn sàng" if health.get("ready") else health.get("message", "Chưa sẵn sàng")
    return gr.Markdown(value=label, elem_classes=["environment-ready" if health.get("ready") else "environment-error"]), health


def upload_view(files):
    files = files or []
    return gr.Gallery(value=[(str(p), Path(p).name) for p in files], visible=bool(files)), gr.Button(interactive=bool(files)), f"{len(files)} ảnh đã chọn · từng ảnh được phân tích độc lập" if files else "Chọn ảnh để bắt đầu"


def use_sample(study_id):
    if not study_id:
        raise gr.Error("Hãy chọn một ca mẫu IU-Xray.")
    return [backend.sample_image(study_id)], study_id


def analysis_mode_view(mode):
    comparing = mode == MODE_COMPARE
    return (
        gr.Radio(visible=not comparing),
        gr.CheckboxGroup(value=list(METHODS), visible=comparing),
        gr.Button(value="So sánh 4 pipeline" if comparing else "Phân tích X-quang"),
    )


def result_choices(state):
    return [(f"Ảnh {r['image_index'] + 1} · {METHODS[r['method']]}", f"{r['image_index']}|{r['method']}") for r in (state or {}).get("results", [])]


def selected_result(state, selection):
    return next((r for r in (state or {}).get("results", []) if f"{r['image_index']}|{r['method']}" == selection), None)


def status_view(state):
    historical = state.get("historical", False)
    prefix = "**Báo cáo nghiên cứu đã lưu · 01/11/2026**" if historical else f"**{safe_text(state.get('case_name'))}**"
    status = STATUSES.get(state.get("status"), state.get("status", ""))
    detail = state.get("stage", "")
    errors = [safe_text(x.get("error", "")) for x in state.get("errors", [])]
    if state.get("error"):
        errors.insert(0, safe_text(state["error"]))
    return prefix + f"  \n{status} · {state.get('progress', 100 if historical else 0)}% · {safe_text(detail)}" + ("\n\n" + "\n\n".join(errors) if errors else "")


def run_analysis(files, name, model, mode, method, effort, study_id, classifier):
    try:
        methods = list(METHODS) if mode == MODE_COMPARE else [method]
        state = backend.create_case(files or [], name, model, methods, effort, study_id, classifier_model=classifier)
        yield state, status_view(state), gr.Dropdown(choices=[], value=None), gr.Column(visible=True), gr.Tabs(selected="analysis")
        for state in backend.run_case(state["id"]):
            choices = result_choices(state)
            yield state, status_view(state), gr.Dropdown(choices=choices, value=choices[-1][1] if choices else None), gr.Column(visible=True), gr.skip()
    except (ValueError, RuntimeError, OSError) as error:
        raise gr.Error(str(error)) from error


def render_result(state, selection):
    result = selected_result(state, selection)
    if not result:
        return None, "Chưa có báo cáo. Kết quả sẽ xuất hiện khi từng nhánh hoàn tất.", [], "", gr.Dropdown(choices=[], value=None), {}, gr.File(value=None, visible=False), gr.File(value=None, visible=False)
    prediction = result.get("prediction") or {}
    report = "### Problems\n\n" + safe_text(", ".join(prediction.get("problems", [])))
    for section in ("findings", "impression"):
        report += f"\n\n### {section.title()}\n\n" + safe_text(prediction.get(section, ""))
    probabilities, thresholds = result.get("probabilities", {}), result.get("thresholds", {})
    labels = result.get("selected_labels", [])
    scores = [[label, round(float(value), 4), round(float(thresholds[label]), 4) if label in thresholds else None, "Có" if label in labels else "Không"] for label, value in probabilities.items()]
    api = result.get("api", {})
    usage = api.get("usage", {})
    actual_model = api.get("model") or result.get("model") or state.get("model", "")
    route = result.get("route", {})
    classifier = result.get("classifier_model") or state.get("classifier_model", "densenet121")
    meta = f"**{safe_text(classifier)}** · **{safe_text(actual_model)}** · reasoning **{safe_text(result.get('reasoning_effort', state.get('reasoning_effort', 'high')))}** · {safe_text(route.get('name'))}"
    meta += f"  \nInput: {usage.get('input_tokens', '—')} · Output (gồm reasoning): {usage.get('output_tokens', '—')} · Reasoning: {usage.get('output_tokens_details', {}).get('reasoning_tokens', '—')} tokens"
    if result.get("elapsed_s") is not None:
        meta += f"  \nThời gian nhánh sau bước chuẩn bị: {result['elapsed_s']:.2f} s"
    meta += f"  \nResponse ID: {safe_text(api.get('response_id'))}"
    evidence = result.get("evidence", [])
    choices = [row["label"] for row in evidence]
    provenance = {key: result.get(key) for key in ("model", "classifier_model", "classifier_provenance", "reasoning_effort", "route", "api", "gate", "summary_path")}
    image = state.get("images", [])[result["image_index"]]
    path = image.get("path")
    summary_file = result.get("summary_path")
    export = str(backend.RUNTIME / state["id"] / "run.json") if not state.get("historical") else None
    summary_file = summary_file if summary_file and Path(summary_file).is_file() else None
    return path, report, scores, meta, gr.Dropdown(choices=choices, value=choices[0] if choices else None), provenance, gr.File(value=export, visible=bool(export)), gr.File(value=summary_file, visible=bool(summary_file))


def evidence_view(state, selection, label):
    result = selected_result(state, selection) or {}
    row = next((r for r in result.get("evidence", []) if r["label"] == label), None)
    if row is None:
        neighbors = result.get("neighbors", [])
        if neighbors:
            text = "### Top-5 ca truy hồi của Baseline B"
            images = []
            for n in neighbors:
                sid = n.get("reference_study_id", "—")
                rank = n.get("global_rank", "—")
                text += f"\n\n**Hạng {rank} · Study {safe_text(sid)}**  \n" + safe_text(", ".join(n.get("reference_problems", [])))
                path = backend.reference_image(str(n.get("reference_id", "")))
                if path:
                    images.append((path, f"Hạng {rank} · Study {sid}"))
            return images, [], text
        else:
            text = "Nhánh này không có bằng chứng theo bệnh. Baseline A chỉ dùng classifier và ảnh gốc."
        return [], [], text
    def gallery(items):
        values = []
        for item in items:
            path = item.get("image_path") or backend.reference_image(str(item.get("reference_id", item.get("id", ""))))
            if path:
                score = item.get("final_score", item.get("score"))
                caption = f"Study {item.get('study_id', '—')}" + (f" · {float(score):.3f}" if isinstance(score, (int, float)) else "")
                values.append((path, caption))
        return values
    note = f"**{safe_text(label)}** · {safe_text(row.get('status'))}"
    if row.get("margin") is not None:
        note += f" · M@5: {row['margin']:.4f}"
    for title, key in (("Support", "support"), ("Non-support", "non_support")):
        for candidate in row.get(key, [])[:5]:
            text = candidate.get("findings") or candidate.get("report") or candidate.get("report_text") or candidate.get("caption")
            if text:
                note += f"\n\n**{title} · Study {safe_text(candidate.get('study_id', '—'))}**  \n{safe_text(text)}"
    return gallery(row.get("support", [])), gallery(row.get("non_support", [])), note


def open_state(state):
    choices = result_choices(state)
    return state, status_view(state), gr.Dropdown(choices=choices, value=choices[0][1] if choices else None), gr.Column(visible=True), gr.Tabs(selected="analysis")


def open_historical(study_id):
    try:
        return open_state(backend.historical_case(study_id))
    except (ValueError, OSError) as error:
        raise gr.Error(str(error)) from error


def history_view():
    cases = backend.list_cases()
    rows, choices = [], []
    for case in cases:
        stamp = case.get("created_at", "")
        if stamp:
            stamp = datetime.fromisoformat(stamp).astimezone(ZoneInfo("Asia/Ho_Chi_Minh")).strftime("%d/%m/%Y %H:%M")
        rows.append([case.get("case_name"), stamp, STATUSES.get(case.get("status"), case.get("status")), len(case.get("images", [])), case.get("model"), case.get("classifier_model", "densenet121"), case.get("result_count", 0)])
        choices.append((f"{case.get('case_name')} · {stamp}", case["id"]))
    return rows, gr.Dropdown(choices=choices, value=choices[0][1] if choices else None), "Chưa có ca mới. Tải ảnh trong mục Phân tích để bắt đầu." if not cases else f"{len(cases)} ca lưu trên máy."


def open_history(run_id):
    if not run_id:
        raise gr.Error("Chưa có lượt chạy để mở.")
    return open_state(backend.load_case(run_id))


def research_data():
    tables = backend.research_tables()
    def table(name, fields):
        return [[METHODS.get(row["pipeline"], row["pipeline"])] + [f"{float(row[key]):.3f} [{float(row[key + '_ci95_lower']):.3f}; {float(row[key + '_ci95_upper']):.3f}]" for key in fields] for row in tables[name]]
    t5 = table("table5", ["precision", "recall", "micro_f1", "sample_f1", "normal_f1"])
    t6 = table("table6", ["chexbert_all_micro_f1", "chexbert_headline_micro_f1", "radgraph_entity_f1", "radgraph_relation_f1", "radgraph_complete_f1"])
    text_path = ROOT / "outputs/text_similarity_4branches_high_20261113/table_4_13_full_precision.json"
    text = []
    if text_path.is_file():
        for key, row in json.loads(text_path.read_text(encoding="utf-8"))["results"].items():
            text.append([METHODS[key]] + [f"{row['metrics'][s]['metrics_on_nonempty_reference'][m]:.3f}" for s in ("findings", "impression") for m in ("bleu4", "rouge_l_f1", "bertscore_f1")])
    return t5, t6, text


def build_app():
    samples = backend.sample_choices()
    default_sample = "1614" if any(sid == "1614" for _, sid in samples) else samples[0][1]
    t5, t6, text = research_data()
    with gr.Blocks(title="CXR Evidence · Gradio", analytics_enabled=False) as demo:
        current = gr.State({})
        gr.HTML('<div class="brand"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" aria-hidden="true"><path d="M8 3H5a2 2 0 0 0-2 2v3m13-5h3a2 2 0 0 1 2 2v3M3 16v3a2 2 0 0 0 2 2h3m8 0h3a2 2 0 0 0 2-2v-3M12 7v10M7 12h10"/></svg><span>CXR<strong>Evidence</strong></span></div>', elem_id="brand")
        with gr.Tabs(selected="analysis", elem_id="workspace") as tabs:
            with gr.Tab("Phân tích", id="analysis"):
                with gr.Row(elem_id="title-row"):
                    gr.Markdown("# Phân tích **X-quang**", elem_id="page-title")
                    health_text = gr.Markdown("Đang kiểm tra CUDA…", elem_id="health-status")
                with gr.Row(equal_height=False, elem_id="analysis-layout"):
                    with gr.Column(scale=165, min_width=320, elem_id="upload-panel"):
                        gr.Markdown("## Tải ảnh", elem_classes="panel-title")
                        files = gr.File(file_count="multiple", file_types=[".png", ".jpg", ".jpeg"], type="filepath", label="Kéo thả ảnh X-quang hoặc chọn từ máy", show_label=False, height=282, elem_id="upload")
                        gr.Markdown("PNG / JPG · Tối đa 10 ảnh · 10 MB / ảnh", elem_classes="small-note")
                        previews = gr.Gallery(visible=False, columns=3, height=230, label="Ảnh đã chọn", interactive=False, object_fit="contain", buttons=[], elem_id="previews")
                        sample_button = gr.Button("Dùng ảnh mẫu IU-Xray", size="sm", elem_id="sample-button")
                        case_name = gr.Textbox(label="Tên ca (tùy chọn)", placeholder="Nhập tên ca", max_lines=1)
                        with gr.Accordion("Tùy chọn ca", open=False, elem_id="case-options"):
                            study_id = gr.Textbox(label="Study ID", placeholder="ID của ca trong kho truy hồi", info="Loại chính study này khỏi truy hồi. Ảnh gốc trong kho được nhận diện tự động.")
                            sample = gr.Dropdown(choices=samples, value=default_sample, label="Ca mẫu IU-Xray", filterable=True)
                            gr.Markdown("Ưu tiên ảnh ngực thẳng. Các ảnh trong cùng ca được phân tích độc lập.", elem_classes="small-note")
                    with gr.Column(scale=100, min_width=340, elem_id="config-panel"):
                        gr.Markdown("## Thiết lập phân tích", elem_id="config-title")
                        with gr.Column(elem_id="config-body"):
                            classifier = gr.Dropdown(choices=[("DenseNet121 (mặc định)", "densenet121"), ("ResNet50", "resnet50")], value="densenet121", label="Model phân loại X-quang")
                            classifier_note = gr.Markdown("ResNet50 dùng ngưỡng và gate DenseNet đóng băng, **chưa có hiệu chuẩn riêng**. Truy hồi vẫn dùng bộ trích xuất DenseNet của nghiên cứu.", visible=False, elem_classes="classifier-note")
                            model = gr.Dropdown(choices=["gpt-5.6-luna", "gpt-5.6-terra", "gpt-6-astra"], value="gpt-5.6-luna", allow_custom_value=True, label="Model GPT")
                            mode = gr.Radio(["Chọn một", MODE_COMPARE], value="Chọn một", label="Chế độ phân tích", elem_id="mode-switch")
                            method = gr.Radio(METHOD_CHOICES, value="structured_rule", label="Phương pháp", elem_id="method-options")
                            compare_methods = gr.CheckboxGroup(METHOD_CHOICES, value=list(METHODS), label="Phương pháp sẽ chạy", info="Cả 4 phương pháp sẽ chạy lần lượt trên mỗi ảnh.", interactive=False, visible=False, elem_id="compare-methods")
                            with gr.Row():
                                effort = gr.Dropdown(choices=[("Cao (high)", "high"), ("Vừa (medium)", "medium"), ("Thấp (low)", "low"), ("Tối thiểu (minimal)", "minimal"), ("Tắt (none)", "none"), ("Rất cao (xhigh)", "xhigh"), ("Tối đa (max)", "max")], value="high", label="Reasoning", min_width=160)
                                gr.Markdown("Giới hạn  \n**8.192 tokens**", elem_id="token-limit")
                            analyze = gr.Button("Phân tích X-quang", variant="primary", interactive=False, elem_id="analyze")
                            selection_note = gr.Markdown("Chọn ảnh để bắt đầu", elem_classes="small-note")
                            with gr.Accordion("Chi tiết phương pháp", open=False, elem_id="method-details"):
                                gr.Markdown("**Baseline A:** classifier + ảnh gốc.\n\n**Baseline B:** classifier + top-5 ca truy hồi + ảnh gốc.\n\n**Structured rule:** bằng chứng theo bệnh + rule gate.\n\n**Learned gate:** bằng chứng theo bệnh + learned gate đóng băng.\n\nCUDA · Cache phản hồi tắt. Sinh báo cáo gọi API đã cấu hình và có thể phát sinh phí.", elem_classes="small-note")
                status = gr.Markdown("", elem_id="run-status")
                with gr.Column(visible=False, elem_id="results") as results:
                    gr.Markdown("## Kết quả phân tích")
                    result_pick = gr.Dropdown(choices=[], label="Ảnh và phương pháp", interactive=True)
                    with gr.Row():
                        with gr.Column(scale=1, min_width=235):
                            result_image = gr.Image(label="Ảnh X-quang", interactive=False, height=320, buttons=["fullscreen", "download"])
                            export = gr.File(label="Tải JSON ca và provenance", interactive=False, visible=False)
                            summary_export = gr.File(label="Tải report gốc", interactive=False, visible=False)
                        with gr.Column(scale=3, min_width=320):
                            with gr.Tabs():
                                with gr.Tab("Báo cáo"):
                                    report = gr.Markdown("", buttons=["copy"], elem_id="report-content", line_breaks=True)
                                    meta = gr.Markdown("", elem_classes="small-note", line_breaks=True)
                                    gr.Markdown("Báo cáo phục vụ nghiên cứu, cần người có chuyên môn kiểm tra trước khi sử dụng.", elem_classes="small-note")
                                with gr.Tab("Classifier"):
                                    scores = gr.Dataframe(headers=["Nhãn", "Xác suất", "Ngưỡng", "Được chọn"], datatype=["str", "number", "number", "str"], interactive=False, show_label=False, type="array", wrap=True)
                                with gr.Tab("Bằng chứng"):
                                    evidence_label = gr.Dropdown(choices=[], label="Bệnh", interactive=True)
                                    with gr.Row():
                                        support = gr.Gallery(label="Support", interactive=False, columns=2, height=260, object_fit="contain", buttons=[])
                                        non_support = gr.Gallery(label="Non-support", interactive=False, columns=2, height=260, object_fit="contain", buttons=[])
                                    evidence_note = gr.Markdown("", line_breaks=True)
                                with gr.Tab("Provenance"):
                                    provenance = gr.JSON(label="Route, gate và actual API usage")
            with gr.Tab("Nghiên cứu", id="research") as research_tab:
                gr.Markdown("# Kết quả **nghiên cứu**")
                gr.Markdown("Kết quả đã lưu · **01/11/2026** · gpt-5.6-luna · reasoning high · 8.192 tokens · DenseNet121. Các bảng này không thay đổi theo ảnh mới tải lên.", elem_classes="research-note")
                gr.Markdown("## Nhãn bệnh — 206 study")
                gr.Dataframe(t5, headers=["Pipeline", "Precision [95% CI]", "Recall [95% CI]", "Micro F1 [95% CI]", "Sample F1 [95% CI]", "Normal F1 [95% CI]"], interactive=False, type="array", wrap=False, label="Table 5", show_label=False)
                gr.Markdown("## Đồng thuận lâm sàng — 205 study")
                gr.Dataframe(t6, headers=["Pipeline", "CheXbert 14 [95% CI]", "CheXbert 6 [95% CI]", "RadGraph entity [95% CI]", "RadGraph relation [95% CI]", "RadGraph complete [95% CI]"], interactive=False, type="array", wrap=False, label="Table 6", show_label=False)
                gr.Markdown("Cohort chung 205 study; loại study 1615 vì cả Findings và Impression tham chiếu rỗng. CI dùng 2.000 paired study resamples. Không tạo reference thay thế.", elem_classes="small-note")
                gr.Markdown("## Độ tương đồng văn bản — đủ 4 pipeline")
                gr.Dataframe(text, headers=["Pipeline", "Findings BLEU-4", "Findings ROUGE-L", "Findings BERTScore F1", "Impression BLEU-4", "Impression ROUGE-L", "Impression BERTScore F1"], interactive=False, type="array", wrap=False, show_label=False)
                gr.Markdown("Đánh giá lại ngày 13/11/2026 từ cùng report high. Findings: 163 reference; Impression: 204 reference. BERTScore: roberta-large. Loại reference rỗng riêng theo trường.", elem_classes="small-note")
                with gr.Row(elem_classes="historical-controls"):
                    historical_pick = gr.Dropdown(choices=samples, value=default_sample, label="Xem report gốc của study", scale=4)
                    historical_button = gr.Button("Mở 4 báo cáo đã lưu", scale=1)
                gr.Markdown("Chỉ đọc report theo provenance; mở báo cáo đã lưu không gọi API.", elem_classes="small-note")
            with gr.Tab("Lịch sử", id="history") as history_tab:
                gr.Markdown("# Lịch sử **trên máy**")
                history_note = gr.Markdown("Chưa có ca mới.", elem_classes="small-note")
                history_table = gr.Dataframe(headers=["Tên ca", "Thời gian", "Trạng thái", "Ảnh", "Model GPT", "Classifier", "Báo cáo"], interactive=False, type="array", wrap=True, show_label=False)
                with gr.Row():
                    history_pick = gr.Dropdown(choices=[], label="Ca đã lưu", scale=4)
                    open_history_button = gr.Button("Mở kết quả", scale=1)
                    refresh_history = gr.Button("Làm mới", scale=1)
        with gr.Accordion("Kết nối và môi trường", open=False, elem_id="connection"):
            health_json = gr.JSON(label="Trạng thái pipeline")
            check = gr.Button("Kiểm tra lại môi trường", size="sm")
            gr.Markdown("API được đọc ở backend từ cấu hình hiện tại của dự án. Khởi động và xem lịch sử không tự sinh report.", elem_classes="small-note")
        files.change(upload_view, files, [previews, analyze, selection_note], queue=False)
        sample_button.click(use_sample, sample, [files, study_id], queue=False)
        classifier.change(lambda name: gr.Markdown(visible=name == "resnet50"), classifier, classifier_note, queue=False)
        mode.change(analysis_mode_view, mode, [method, compare_methods, analyze], queue=False)
        main_outputs = [current, status, result_pick, results, tabs]
        analyze.click(run_analysis, [files, case_name, model, mode, method, effort, study_id, classifier], main_outputs, concurrency_id="cuda", concurrency_limit=1, trigger_mode="once")
        result_pick.change(render_result, [current, result_pick], [result_image, report, scores, meta, evidence_label, provenance, export, summary_export], queue=False)
        evidence_label.change(evidence_view, [current, result_pick, evidence_label], [support, non_support, evidence_note], queue=False)
        historical_button.click(open_historical, historical_pick, main_outputs, queue=False)
        open_history_button.click(open_history, history_pick, main_outputs, queue=False)
        history_tab.select(history_view, outputs=[history_table, history_pick, history_note], queue=False)
        refresh_history.click(history_view, outputs=[history_table, history_pick, history_note], queue=False)
        check.click(health_view, outputs=[health_text, health_json])
        demo.load(health_view, outputs=[health_text, health_json])
    return demo.queue(default_concurrency_limit=1, max_size=8)


def main():
    parser = argparse.ArgumentParser(description="CXR Evidence local Gradio workspace")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--browser", action="store_true")
    args = parser.parse_args()
    demo = build_app()
    theme = gr.themes.Base(primary_hue="cyan", neutral_hue="slate", font=["Segoe UI", "Arial", "sans-serif"]).set(body_background_fill="#f5f8f9", body_background_fill_dark="#f5f8f9", body_text_color="#35424e", body_text_color_dark="#35424e", block_background_fill="#ffffff", block_background_fill_dark="#ffffff", input_background_fill="#ffffff", input_border_color="#dce6e9", button_primary_background_fill="#007f91", button_primary_background_fill_dark="#007f91", button_primary_text_color="#ffffff", button_primary_text_color_dark="#ffffff", block_radius="4px", input_radius="3px", button_large_radius="3px", button_small_radius="3px", button_large_text_size="14px")
    values = theme.to_dict()["theme"]
    theme.set(**{key: values[key[:-5]] for key in values if key.endswith("_dark") and key[:-5] in values})
    translations = {"upload_text.drop_file": "Kéo thả ảnh X-quang", "upload_text.click_to_upload": "Chọn ảnh từ máy", "common.or": "hoặc", "common.upload": "Tải ảnh", "common.complete": "Hoàn tất", "common.clear": "Xóa ảnh", "common.loading": "Đang tải", "common.error": "Lỗi", "common.download": "Tải xuống", "common.remove": "Bỏ ảnh"}
    demo.launch(server_name="127.0.0.1", server_port=args.port, share=False, inbrowser=args.browser, theme=theme, css_paths=APP_DIR / "styles.css", i18n=gr.I18n(en=translations, vi=translations), footer_links=[], max_file_size="10mb", allowed_paths=[str(ROOT / "data/reprocessed/images"), str(backend.RUNTIME), str(ROOT / "outputs/high_effort_4branches_20261031")], blocked_paths=[str(ROOT / ".env"), str(ROOT / "model"), str(ROOT / "src/learned_gate_oof_206/models")], show_error=True)


if __name__ == "__main__":
    main()
