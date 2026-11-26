"""Write the thesis tables index and interpretation from audited recomputation results."""


def cell(row):
    return f"{row['estimate']:.4f} [{row['ci95_low']:.4f}, {row['ci95_high']:.4f}]"


def evidence(row):
    if row['ci95_low'] > 0:
        return 'CI nằm trên 0'
    if row['ci95_high'] < 0:
        return 'CI nằm dưới 0'
    return 'CI chứa 0'


def write_thesis_reports(out, audit, checks, retrieval, retrieval_delta, normal_contrasts, targets, target_delta, gate, gate_delta):
    rows = {(r['configuration'], r['k'], r['metric'], r['cohort']): r for r in retrieval}
    paired = {(r['left'], r['right'], r['k'], r['metric'], r['cohort']): r for r in retrieval_delta}
    contrast = {(r['k'], r['metric']): r for r in normal_contrasts}
    high, latency = audit['report_sources']
    controls = {(r['run_id'], r['pipeline'], r['metric']): r for r in targets}
    heading = '# Kết quả đưa vào luận văn\n'
    text = [heading,
        f"Các bảng truy xuất và đối chứng nhãn dùng cùng {checks['n_studies']} study IU X-Ray từ manifest quota, gồm 30 ca nhóm Normal và 176 ca còn lại. Mỗi study có một ảnh truy vấn cố định. Gallery có 3.818 ảnh thuộc 3.689 study. Sau loại toàn bộ ảnh cùng study và khử trùng, mỗi truy vấn có 3.688 study hợp lệ. Cohort quota phục vụ so sánh trên cùng mẫu. Tỷ lệ nhóm trong cohort phụ thuộc cách tuyển.\n",
        '[Bảng 4.11 Embedding](table411.md) · [Bảng 4.12 Top-K](table412.md) · [Bảng 4.13 Sweep λ](table413.md) · [Bản đồ cập nhật luận văn](THESIS_UPDATE_MAP.md)\n',
        '## Truy xuất toàn cục và ảnh hưởng của nhóm Normal\n',
        '| Embedding | P@5 | mAP@5 | nDCG@5 | Δ_path toàn gallery |', '| --- | --- | --- | --- | --- |',
    ]
    names = {'raw': 'RAD-DINO raw 768-D', 'densenet': 'DenseNet pathology 512-D', 'adapted_1.0': 'CheXpert-adapted RAD-DINO 512-D'}
    for name, label in names.items():
        vals = [cell(rows[name, 5, metric, 'all']) for metric in ('precision', 'map', 'ndcg')]
        vals.append(cell(rows[name, 'global', 'delta_path_global', 'all']))
        text.append('| ' + ' | '.join([label, *vals]) + ' |')
    text += ['', 'Chênh lệch P@5 cho từng cặp theo bootstrap ghép cặp 2.000 lần, CI percentile 95%.\n']
    for left, right in [('adapted_1.0', 'raw'), ('densenet', 'raw'), ('adapted_1.0', 'densenet')]:
        row = paired[left, right, 5, 'precision', 'all']
        text.append(f'- {names[left]} trừ {names[right]} đạt {cell(row)} ({evidence(row)}).')
    raw_normal = rows['raw', 5, 'precision', 'normal']
    norm = rows['adapted_1.0', 5, 'precision', 'normal']
    other = rows['adapted_1.0', 5, 'precision', 'other']
    c = contrast[5, 'precision']
    text += ['',
        f"Trên 30 ca Normal theo manifest, P@5 của raw là {cell(raw_normal)} và adapted là {cell(norm)}. Trên 176 ca còn lại, adapted đạt {cell(other)}. P@5 adapted trừ raw lần lượt là {cell(paired['adapted_1.0','raw',5,'precision','normal'])} và {cell(paired['adapted_1.0','raw',5,'precision','other'])}. Với trọng số nhóm 30/206 và 176/206, hai nhóm đóng góp {c['normal_weighted_contribution']*100:.2f} và {c['other_weighted_contribution']*100:.2f} điểm phần trăm vào chênh lệch toàn cohort. Chênh lệch giữa hai mức tăng là {cell(c)} ({evidence(c)}).\n",
        f"Trong cohort này có {audit['cohort']['zero_support']} vector support toàn 0 và {audit['cohort']['gold_normal']} gold Normal theo Problems. Ba cách nhận diện trùng nhau trong 206 study này. Quy tắc truy xuất zero-support là so khớp vector MeSH. Theo Problems có {audit['cohort']['target_positive']} study mang ít nhất một trong sáu nhãn mục tiêu. [Đối chiếu theo nhóm](normal_subgroup_metrics.csv)\n",
        f"Study {', '.join(checks['r_zero'])} có R=0 nên AP bằng 0. Các độ liên quan phân mức dương của study này vẫn được dùng cho nDCG vì IDCG dương. Số study có IDCG=0 là {len(checks['idcg_zero'])}. Δ_path toàn gallery có {rows['raw','global','delta_path_global','all']['n']} study đủ cả nhóm relevant và nhóm độ liên quan bằng 0. Đơn vị là chênh lệch cosine.\n",
        '## Xếp hạng lại bằng đặc trưng DenseNet cục bộ\n',
        '| Độ đo tại K=5 | λ=1,0 | λ=0,9 | λ=0,9 trừ λ=1,0 |', '| --- | --- | --- | --- |',
    ]
    for metric, label in [('weighted_jaccard', 'Weighted Jaccard'), ('precision', 'P@5'), ('map', 'mAP@5'), ('ndcg', 'nDCG@5')]:
        text.append(f"| {label} | {cell(rows['adapted_1.0',5,metric,'all'])} | {cell(rows['adapted_0.9',5,metric,'all'])} | {cell(paired['adapted_0.9','adapted_1.0',5,metric,'all'])} |")
    text += ['', 'Ở K=5, các điểm ước lượng lớn nhất trong sweep xuất hiện ở các mức λ sau đây.\n']
    for metric, label in [('precision', 'P@5'), ('map', 'mAP@5'), ('ndcg', 'nDCG@5')]:
        best = max((r for r in retrieval if r['encoder'] == 'adapted' and r['cohort'] == 'all' and r['k'] == 5
                    and r['metric'] == metric), key=lambda r: r['estimate'])
        label_lambda = best['configuration']
        diff = paired[label_lambda, 'adapted_1.0', 5, metric, 'all']
        text.append(f"- {label} đạt {best['estimate']:.4f} tại λ={best['lambda_global']:.1f}. Chênh lệch với λ=1,0 là {cell(diff)} ({evidence(diff)}).")
    text += ['',
        'Sweep là phân tích độ nhạy hậu nghiệm trên cùng Top-50 study do adapted global chọn. λ=0 xếp hạng theo local trong Top-50 này. Cấu hình end-to-end lưu từ trước dùng λ=0,9. Tất cả CI là ước lượng thăm dò, chưa hiệu chỉnh đa so sánh. [Chênh lệch ghép cặp](retrieval_paired_ci.csv)\n',
        f"Các ô chung giữa ba bảng khớp đến sai số tối đa {max(r['max_absolute_error'] for r in checks['shared_table_cells']):.1e}. Đối chiếu riêng thứ hạng toàn gallery với cả ba K cho adapted λ=1,0 có {checks['independent_adapted_global_vs_sweep_study_k']} phép so sánh cấp study. Cache cũ có ngưỡng và pool 50 phù hợp, nhưng chỉ lưu IDCG@5. Hàng nDCG@10 từng bị để trống khi chạy sweep từ cache đó. Bảng mới tính R và IDCG từ cả gallery hợp lệ với MeSH hiện hành.\n",
        '## Kiểm tra kết quả lịch sử\n',
        'Summary 1.246 truy vấn ghi 3.818 hàng ảnh, 1.247 ảnh có support và bỏ truy vấn không còn relevant. Summary `pathology_512` trỏ tới projection RAD-DINO cũ. Run adapted hiện tại được nhận diện bằng manifest CheXpert, checkpoint và đối chiếu global score từng cặp. Artifact cũ có nhắc 904 ảnh có support và mẫu số 903 ở đánh giá adapted. Thiếu danh sách query gốc của 903 nên chưa xác định tập giao với 176 study ngoài nhóm Normal. Đối chiếu ảnh hưởng Normal dùng chênh lệch ghép cặp trong cùng cohort ở trên. [Kiểm toán đầu vào](input_audit.json)\n',
        f"So với cache cũ, R thay đổi ở {audit['historical_relevant_counts_changed']['adapted']} query adapted. Quy đổi support thập phân sang đơn vị nguyên tương đương trước khi cộng giữ chính xác các tỷ số đúng 0,5. Quy tắc này cùng cách đưa AP=0 vào trung bình khi R=0 có thể làm các ô cũ và mới chênh nhau. [Đối chiếu cache](cache_reconciliation.csv)\n",
        '| λ | P@5 cũ / mới | mAP@5 cũ / mới | nDCG@5 cũ / mới |', '| --- | --- | --- | --- |',
    ]
    for row in audit['historical_adapted_k5']:
        config = f"adapted_{row['lambda_global']:.1f}"
        metrics = ' | '.join(f"{row[old]:.4f} / {rows[config, 5, new, 'all']['estimate']:.4f}"
                             for old, new in [('P@5', 'precision'), ('AP@5', 'map'), ('nDCG@5', 'ndcg')])
        text.append(f"| {row['lambda_global']:.1f} | {metrics} |")
    text += ['',
        '## Đối chứng end-to-end theo sáu nhãn\n',
        f"Các bảng chính lấy 824 report đã lưu từ `{high}`. Bốn nhánh lưu `gpt-5.6-luna`, effort `high`, tối đa 8.192 token đầu ra. Run `{latency}` cũng có 824 report đã xác minh và được tính riêng. Run latency dùng effort `none` và giới hạn 160 token. [Provenance từng report](target_report_provenance.csv)\n",
        '| Pipeline | Precision | Recall | Micro F1 | Sample F1 | Normal F1 |', '| --- | --- | --- | --- | --- | --- |',
    ]
    for branch in ('C0', 'K0', 'C0_and_K0', 'baseline_a', 'baseline_b', 'structured_rule', 'structured_learned_gate'):
        text.append('| ' + ' | '.join([branch, *(cell(controls[high, branch, m]) for m in ('precision', 'recall', 'micro_f1', 'sample_f1', 'normal_f1'))]) + ' |')
    text += ['', 'Chênh lệch Micro F1 trên đúng cặp study của run high-effort.\n']
    for row in target_delta:
        if row['run_id'] == high and row['metric'] == 'micro_f1':
            text.append(f"- {row['left']} trừ {row['right']} đạt {cell(row)} ({evidence(row)}).")
    text += ['',
        f"C0 trùng `selected_labels` ở toàn bộ 206 study của từng run. Sáu ngưỡng và version trong evidence khớp cấu hình checkout theo giá trị. SHA-256 của file cấu hình checkout khác SHA lịch sử ghi trong evidence. C0 sử dụng trực tiếp ngưỡng đã lưu theo từng ca. K0 dùng ít nhất 3 trong 5 study có MeSH support dương từ Top-5 lưu trong run cấu hình λ=0,9. Pipeline đã bỏ qua local rerank ở {len(audit['global_only_rule_evidence_studies'])} ca không có nhãn classifier vượt ngưỡng ({', '.join(audit['global_only_rule_evidence_studies'])}). K0 dùng Top-5 global đã lưu cho các ca đó.\n",
        'A trừ C0 gộp mô hình sinh, ảnh và chính sách đầu ra. Learned trừ A gộp truy xuất, gate, prompt và routing. Các CI này mô tả chênh lệch giữa pipeline. Sample F1 bằng 1 khi hai tập nhãn mục tiêu cùng rỗng. Normal được đánh giá riêng. [Toàn bộ đối chứng và CI](target_label_controls.md) · [Chênh lệch ghép cặp](target_label_paired_ci.csv)\n',
        'Đánh giá nhãn dùng 206 study. Mask tham chiếu chung cho clinical/text giữ 205 study, loại study 1615 vì cả Findings và Impression rỗng. Khi tính theo một mục tham chiếu, Findings có 163 study và Impression có 204. Báo cáo tham chiếu được giữ nguyên. [Mask tham chiếu](clinical_reference_mask.csv)\n',
        '## Đối chứng gate trước mô hình sinh\n',
    ]
    if gate:
        gate_points = {(r['model'], r['metric']): r for r in gate}
        a = gate_points['probability_and_onehot', 'micro_f1']
        b = gate_points['full_evidence', 'micro_f1']
        diff = next(r for r in gate_delta if r['metric'] == 'micro_f1')
        text += [f"Gate probability + label one-hot đạt Micro F1 {cell(a)}. Gate dùng thêm toàn bộ feature evidence đạt {cell(b)}. Chênh lệch ghép cặp là {cell(diff)} ({evidence(diff)}). Hai mô hình fit riêng theo cùng 5 fold study, StandardScaler trong mỗi fold và cùng ngưỡng 0,5. Tập feature gồm 1.236 cặp study–nhãn. Log rule chỉ ghi 525 cặp đã được classifier chọn. Kết quả này đo quyết định OOF trước LLM. [Đối chứng gate](gate_ablation_metrics.md) · [CI ghép cặp](gate_ablation_paired_ci.csv)\n"]
    else:
        text += ['Chưa tính được đối chứng gate vì thiếu artifact đầu vào. [Danh sách nguồn cần bổ sung](gate_ablation_missing.json)\n']
    text += ['## Điều tác giả cần xác nhận\n',
             'Nguồn dữ liệu tác giả thực sự dùng khi chọn ngưỡng heuristic, quy tắc 3/5 và λ cần được xác nhận từ ghi chép thí nghiệm. Kết quả này xác nhận cấu hình đã lưu và phân tích độ nhạy, chưa xác lập lịch sử lựa chọn.\n']
    (out / 'RESULTS_FOR_THESIS.md').write_text('\n'.join(text), encoding='utf-8')

    thesis = audit['thesis']
    availability = ('Đã xác định bản Word đích từ đường dẫn trong bàn giao.' if thesis['exact_present'] else
                    'Bản Word ghi trong bàn giao vắng ở đường dẫn đích. Dùng tiêu đề đã đối chiếu trong bản 15/11 hiện có và số bảng mới theo bàn giao. Cần kiểm tra lại số bảng và vị trí đoạn khi mở đúng bản 22/11.')
    update = f"""# Bản đồ cập nhật luận văn

{availability}

- Bản đích `{thesis['expected']}`.
- Bản đối chiếu tiêu đề `{thesis['caption_reference']}`.

| Vị trí theo tên | Nội dung cần cập nhật | Số liệu và nguồn |
| --- | --- | --- |
| Tóm tắt Tiếng Việt và Abstract | Dùng 206 study gồm 30 Normal, chọn một ảnh theo manifest. Đồng bộ kết quả retriever chính và cách diễn giải các pipeline. Ghi 205 ca tham chiếu khi nêu CheXbert hoặc RadGraph. | [Kết quả](RESULTS_FOR_THESIS.md) và [mask](clinical_reference_mask.csv) |
| 4.2 Thiết lập thực nghiệm | Ghi seed 20260830, quota, 206 study, 3.818 ảnh thuộc 3.689 study. Phân biệt MeSH support, Problems gold và tham chiếu văn bản. Nêu bootstrap ghép cặp theo study, 2.000 lần, seed 20260904. | [Giao thức](protocol.json) và [cohort](cohort.csv) |
| Kết quả xây dựng vector và đánh giá độ liên quan truy xuất Top-K | Thay đoạn mô tả bỏ query không còn relevant. Nêu leave-one-study-out, ảnh đại diện theo global cosine, công thức gain tuyến tính và mẫu số AP/IDCG toàn gallery. Nêu quy ước zero-support và đóng góp nhóm Normal. | [Giao thức](protocol.json) và [đối chiếu nhóm](normal_subgroup_contrasts.csv) |
| So sánh chất lượng truy xuất của các cấu hình embedding tại K=5 và K=10, Bảng 4.11 | Thay ba hàng global và đoạn sau bảng. Ghi Δ_path trên 3.688 study hợp lệ mỗi query, có N hợp lệ riêng. Dùng CI ghép cặp cho adapted, raw và DenseNet. | [Bảng 4.11](table411.md) và [CI](retrieval_paired_ci.csv) |
| Chỉ số CheXpert-adapted RAD-DINO theo độ sâu Top-K, Bảng 4.12 | Dùng K=3, 5, 10 cho adapted + local λ=0,9. Có thể trình bày kèm global-only λ=1,0 vì cùng protocol và cùng cohort. | [Bảng 4.12](table412.md) |
| Phân tích độ nhạy local reranking theo λ, Bảng 4.13 | Thay sweep cũ bằng 11 mức λ trong cùng Top-50 adapted. Nếu bảng in chỉ để K=5, giữ K=3,10 ở phụ lục. Dẫn các delta λ trừ global và λ trừ 0,9. Giữ λ chính bằng 0,9 và nêu tính hậu nghiệm. | [Bảng 4.13](table413.md) và [CI](retrieval_paired_ci.csv) |
| Kết quả end-to-end của bốn pipeline theo sáu nhãn mục tiêu, và bảng chênh lệch paired bootstrap | Đối chiếu điểm ước lượng với đúng run high-effort. Thay CI theo cùng ma trận bootstrap manifest nếu trình bày lại. Tách kết quả latency thành một đối chiếu nguồn. | [Đối chứng](target_label_controls.md) và [CI](target_label_paired_ci.csv) |
| Đối chứng không dùng mô hình sinh và nhánh learned trên 206 study, Bảng 4.18 theo bàn giao | Đây là Bảng 4.19 trong bản đối chiếu 15/11. Ghi C0, K0, C0 ∧ K0, thêm A để có A−C0 và Learned−A. Diễn giải chênh lệch cấp pipeline. | [Đối chứng](target_label_controls.md) và [evidence Top-5](classifier_top5_controls.csv) |
| Phân bố trạng thái bằng chứng, Bảng 4.22 theo bàn giao | Dùng đếm trên 525 cặp nhãn đã chọn từ 206 evidence rule. Mỗi hàng là một cặp study–nhãn. Bảng trạng thái cần ghi rõ run và mẫu số. | [Đếm trạng thái](rule_evidence_status_counts.csv) |
| RQ2 trong tổng hợp câu hỏi nghiên cứu | Dùng so sánh embedding cùng 206 study và delta ghép cặp. Kiểm tra đóng góp Normal trong cohort hiện tại. | [Bảng 4.11](table411.md), [CI](retrieval_paired_ci.csv), [nhóm Normal](normal_subgroup_contrasts.csv) |
| RQ3 trong tổng hợp câu hỏi nghiên cứu | Ghi điểm ước lượng và CI λ=0,9 so với 1,0. Nếu nêu các λ khác, lấy từ đúng sweep và ghi đây là so sánh thăm dò. | [Bảng 4.13](table413.md) và [CI](retrieval_paired_ci.csv) |
| RQ4 trong tổng hợp câu hỏi nghiên cứu | Ghi các chênh lệch A−C0, Learned−A, Learned−Rule. Trình bày thay đổi Precision, Recall, FP, FN theo đúng run. Dẫn đối chứng feature gate ở phần trước LLM. | [Kết quả](RESULTS_FOR_THESIS.md), [gate OOF](gate_ablation_metrics.csv) |
| 5.1 Tóm tắt bài toán nghiên cứu, phương pháp tiếp cận và kết quả đạt được | Thay các điểm ước lượng cũ cho retrieval, sweep và CI đối chứng. Giữ mẫu số 206 cho target labels, 205 cho clinical/text gộp. | [Kết quả](RESULTS_FOR_THESIS.md) |
| 5.2 Đóng góp của đề tài, Đ2 | Trình bày đối chứng logistic regression probability + one-hot và full evidence trên cùng 5 fold theo study, ngưỡng 0,5. Ghi rõ đánh giá trước LLM và CI ghép cặp. | [Gate OOF](gate_ablation_metrics.csv) |
| 5.2 Đóng góp của đề tài, Đ4 theo bàn giao | Đối chiếu đoạn đóng góp về đánh giá nhiều tầng và kiểm chứng. Ghi đúng chỉ số, cohort và nguồn report. Rà soát số mục Đ4 vì bản 15/11 gộp đánh giá nhiều tầng ở đóng góp thứ ba. | [Giao thức](protocol.json) và [đối chứng](target_label_controls.csv) |

Kiểm tra các đoạn lặp lại số 0,3462, 0,3796, 0,3233, 0,3953 và +0,1845 trước khi thay. Dùng giá trị mới từ đúng cột, cùng N, đúng chiều delta và cùng quy tắc làm tròn. Tác giả xác nhận lịch sử sử dụng dữ liệu khi chọn tham số heuristic trước khi hoàn thiện phần phương pháp.
"""
    (out / 'THESIS_UPDATE_MAP.md').write_text(update, encoding='utf-8')
