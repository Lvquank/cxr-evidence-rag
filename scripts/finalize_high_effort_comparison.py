"""Audit the completed high run and compare it with the earlier none run on CUDA."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.recompute_latency_tables import read_csv, read_json, save_json, save_csv, tensor, div, require_cuda, T5, T6, LABELS14
from scripts.validate_common_studies import sha256
import numpy as np
import pandas as pd
import torch

RUN=ROOT/"outputs/high_effort_4branches_20261031"
HIGH=RUN/"metrics"
NONE=ROOT/"outputs/recompute_table5_table6_20261031"
BRANCHES={"structured_rule":"structured_rule","structured_learned_gate":"structured_learned_gate","baseline_a":"baselines/baseline_a","baseline_b":"baselines/baseline_b"}
OLD={"structured_rule":"outputs/rule_latency_206_20261029","structured_learned_gate":"outputs/learned_gate_latency_206_20261029","baseline_a":"outputs/ab_latency_206_20261030/baseline_a","baseline_b":"outputs/ab_latency_206_20261030/baseline_b"}


def sufficient(root,table,name,ids):
    if table=="table5":
        rows=read_csv(root/"per_study"/f"{name}_target_labels.csv")
    else:
        rows=[r for r in read_csv(root/"table6_per_study.csv") if r["pipeline"]==name]
    assert len(rows)==len(ids) and len({r["study_id"] for r in rows})==len(ids)
    by_id={r["study_id"]:r for r in rows}
    data=[]
    for sid in ids:
        r=by_id[sid]
        if table=="table5":
            gold,pred=r["gold_normal"]=="True",r["predicted_normal"]=="True"
            data.append([*[float(r[k]) for k in ["problem_tp","problem_fp","problem_fn","problem_f1"]],gold and pred,not gold and pred,gold and not pred])
        else:
            data.append([float(r[k]) for k in ["chexbert_all_tp","chexbert_all_fp","chexbert_all_fn","chexbert_headline_tp","chexbert_headline_fp","chexbert_headline_fn",*T6[2:]]])
    return tensor(data)


def scores(v,index,table):
    stats=torch.cat([v.sum(0,keepdim=True),v[index].sum(1)],dim=0)
    n=len(v)
    if table=="table5":
        tp,fp,fn,sf,ntp,nfp,nfn=stats.unbind(1)
        return torch.stack([div(tp,tp+fp),div(tp,tp+fn),div(2*tp,2*tp+fp+fn),sf/n,div(tp,tp+fp,1.),div(fp,tp+fp),div(2*ntp,2*ntp+nfp+nfn)],dim=1)
    atp,afp,afn,htp,hfp,hfn,ent,rel,complete=stats.unbind(1)
    return torch.stack([div(2*atp,2*atp+afp+afn),div(2*htp,2*htp+hfp+hfn),ent/n,rel/n,complete/n],dim=1)


def referenced_ids(node):
    if isinstance(node,dict):
        for key,value in node.items():
            if key=="study_id": yield str(value)
            elif isinstance(value,(dict,list)): yield from referenced_ids(value)
    elif isinstance(node,list):
        for value in node: yield from referenced_ids(value)


def main():
    hardware=require_cuda()
    assert read_json(RUN/"execution_status.json")["status"]=="complete"
    assert read_json(HIGH/"final_audit.json")["passed"]
    # Validate every pre-existing metrics artifact before adding derived reports.
    old_checks=read_json(HIGH/"output_checksums.json")
    assert all(sha256(HIGH/p)==digest for p,digest in old_checks.items())
    ref_old=pd.read_csv(NONE/"chexbert_raw/labeled_reports.csv").iloc[:205]
    ref_new=pd.read_csv(HIGH/"chexbert_raw/labeled_reports.csv").iloc[:205]
    assert ref_old["Report Impression"].tolist()==ref_new["Report Impression"].tolist()
    assert np.array_equal(ref_old[LABELS14].to_numpy(),ref_new[LABELS14].to_numpy(),equal_nan=True)
    for stage,key in [("chexbert_raw","checkpoint_sha256"),("radgraph_raw","weights_sha256"),("radgraph_raw","config_sha256")]:
        assert read_json(NONE/stage/"inference_manifest.json")[key]==read_json(HIGH/stage/"inference_manifest.json")[key]
    comparison=[]
    for table,names in [("table5",T5),("table6",T6)]:
        new=read_json(HIGH/f"{table}_results.json")
        old=read_json(NONE/f"{table}_results.json")
        ids=new["study_ids"]
        assert ids==old["study_ids"]
        raw_index=np.load(HIGH/f"{table}_bootstrap_indices.npy")
        np.testing.assert_array_equal(raw_index,np.load(NONE/f"{table}_bootstrap_indices.npy"))
        index=torch.as_tensor(raw_index,dtype=torch.long,device="cuda:0")
        for name in BRANCHES:
            before=scores(sufficient(NONE,table,name,ids),index,table)
            after=scores(sufficient(HIGH,table,name,ids),index,table)
            for j,metric in enumerate(names):
                assert abs(float(before[0,j])-old["absolute"][name][metric]["estimate"])<1e-12
                assert abs(float(after[0,j])-new["absolute"][name][metric]["estimate"])<1e-12
                delta=after[:,j]-before[:,j]
                lo,hi=torch.quantile(delta[1:],tensor([.025,.975])).tolist()
                comparison.append({"table":table,"pipeline":name,"metric":metric,"study_count":len(ids),"old_none_160":float(before[0,j]),"new_high_8192":float(after[0,j]),"delta_high_minus_none":float(delta[0]),"ci95_lower":lo,"ci95_upper":hi,"ci_contains_zero":lo<=0<=hi})
    save_csv(HIGH/"none_vs_high_paired.csv",comparison)
    save_json(HIGH/"none_vs_high_protocol.json",{"old_metrics_dir":str(NONE),"new_metrics_dir":str(HIGH),"delta_direction":"high minus none within the same pipeline","samples":2000,"seed":20260904,"resampling_unit":"study_id","arithmetic_device":"cuda:0","indices":"identical to both runs' saved paired bootstrap indices","identical_reference_text_and_chexbert_labels":True,"identical_clinical_checkpoint_hashes":True,"interpretation":"Describes jointly changed reasoning effort, output cap and newly sampled reports, not an isolated causal effort effect. The high client also uses a longer request timeout (120 s versus the former 30 s)."})
    parity,latency=[],[]
    seen=set()
    for name,relative in BRANCHES.items():
        path=RUN/relative
        source=ROOT/OLD[name]
        current=read_json(path/"latency_summary.json")
        prior=read_json(source/"latency_summary.json")
        assert current["complete"] and current["succeeded"]==206
        assert [r["study_id"] for r in current["results"]]==[r["study_id"] for r in prior["results"]]
        prior_by_id={r["study_id"]:r for r in prior["results"]}
        max_diff=0.;selected_diff=0;leakage=0; image_mismatch=0; gate_max_diff=0.
        for row in current["results"]:
            sid=row["study_id"]
            report=read_json(ROOT/row["summary"])
            before=read_json(ROOT/prior_by_id[sid]["summary"])
            assert report["api"]["model"]=="gpt-5.6-luna" and report["reasoning_effort"]=="high"
            assert report["api"]["response_id"] not in seen and not report["api"]["cached"]
            seen.add(report["api"]["response_id"])
            image_mismatch+=report["source"]["image_name"]!=before["source"]["image_name"]
            if name.startswith("structured"):
                raw=read_json(path/"raw_evidence"/(row["request_id"]+".json"))
                old_raw=read_json(source/"raw_evidence"/(row["request_id"]+".json"))
                assert raw["retriever"]["excluded_study_ids"]==[sid]
                a,b=raw["classifier"],old_raw["classifier"]
                probs,old_probs=a["raw_probabilities"],b["raw_probabilities"]
                selected_diff+=a["selected_labels"]!=b["selected_labels"]
                leakage+=sid in set(referenced_ids(raw["retriever"]))
                if name=="structured_learned_gate":
                    gate=read_json(path/"evidence"/(row["request_id"]+".json"))["learned_gate"]
                    old_gate=read_json(source/"evidence"/(row["request_id"]+".json"))["learned_gate"]
                    assert gate["learned_selected_labels"]==old_gate["learned_selected_labels"]
                    gate_max_diff=max(gate_max_diff,max(abs(x["probability"]-y["probability"]) for x,y in zip(gate["decisions"],old_gate["decisions"],strict=True)))
            else:
                probs,old_probs=report["classifier_probabilities"],before["classifier_probabilities"]
                selected_diff+=report["classifier_selected_labels"]!=before["classifier_selected_labels"]
                if name=="baseline_b":
                    refs=[r["reference_study_id"] for r in report["five_nearest_cases"]]
                    assert len(refs)==len(set(refs))==5
                    leakage+=sid in refs
            max_diff=max(max_diff,max(abs(probs[k]-old_probs[k]) for k in probs))
        assert max_diff<1e-6 and selected_diff==leakage==image_mismatch==0 and gate_max_diff<1e-12
        parity.append({"pipeline":name,"studies":206,"classifier_probability_max_abs_diff_vs_none":max_diff,"classifier_selected_label_disagreements_vs_none":selected_diff,"learned_gate_probability_max_abs_diff_vs_none":gate_max_diff if name=="structured_learned_gate" else None,"query_study_leakage_count":leakage,"image_mismatch_count":image_mismatch})
        times=current["latency"]["end_to_end_s"]
        failed=[read_json(p) for p in path.glob("failed_attempts/*/*/attempt.json")]
        latency.append({"pipeline":name,"study_count":206,"mean_s":times["mean_s"],"median_s":times["median_s"],"p95_s":times["p95_s"],"successful_attempt_total_s":times["total_s"],"active_batch_wall_s":current["batch_wall_s"],"archived_failed_attempts":len(failed),"archived_failed_attempt_total_s":sum(f["failed_attempt_s"] for f in failed),"old_none_mean_s":prior["latency"]["end_to_end_s"]["mean_s"],"mean_high_minus_none_s":times["mean_s"]-prior["latency"]["end_to_end_s"]["mean_s"]})
    save_csv(RUN/"upstream_parity_vs_none.csv",parity)
    save_csv(RUN/"latency_4branches.csv",latency)
    lines=["# Latency bốn run mới: Luna high, CUDA", "", "206 study mỗi nhánh; inference lại từ ảnh, mới tiến trình/model và response API cho mỗi ca. Các lượt chạy tuần tự, cùng model/effort/cap, nhưng prompt/evidence/routing riêng được giữ nguyên.", "", "| Nhánh | Mean (s) | Median (s) | P95 (s) | Tổng các lần thử thành công (phút) |", "|---|---:|---:|---:|---:|"]
    lines += [f"| {r['pipeline']} | {r['mean_s']:.4f} | {r['median_s']:.4f} | {r['p95_s']:.4f} | {r['successful_attempt_total_s']/60:.3f} |" for r in latency]
    lines += ["", "Rule có một lần thử đầy đủ không hợp lệ ở study 1445/request loo_158_pleural_effusion, sau 157 ca thành công; đã archive rồi chạy lại thành công với cùng high/8192/CUDA. Latency từng ca trong bảng là lần thử thành công; 22.9408494 s của lần thử bị loại được lưu riêng và nằm trong active batch wall time. Không bỏ qua hoặc ước lượng study nào. Response/API metadata của lần thất bại ban đầu không được pipeline cũ lưu nên không khôi phục được; lỗi/log/provenance được giữ trong resume_note_case158.json và failed_attempts/.", "", "Lượt smoke ban đầu còn gặp DNS trong sandbox trước khi có quyền mạng; smoke không thuộc 824 report/latency chính. Sau đó user đã cho phép gửi ảnh/evidence tới OpenAI Responses API.", "", "So sánh latency với none là quan sát giữa các lượt chạy khác thời điểm, không khống chế tải API/GPU/OS. Effort và cap đã đổi từ none/160 sang high/8192, timeout từ 30 lên 120 giây. Không quy toàn bộ chênh lệch cho riêng gate hay reasoning effort.", "", "upstream_parity_vs_none.csv xác nhận cùng ảnh/study, 0 sai khác nhãn classifier, xác suất khớp trong tolerance đã ghi; learned gate giữ nguyên labels/xác suất OOF; không query-study leakage. Rule/learned vẫn dùng các route 197/9 và 153/53; A/B đều 206 vision."]
    (RUN/"LATENCY_REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    report=HIGH/"REPORT.md"
    addition="\n## Đối chiếu trực tiếp với bốn run none ngày 29–30/10\n\nnone_vs_high_paired.csv lưu 48 chênh lệch high−none và 95% CI, dùng chung 2.000 study resamples (seed 20260904), cùng ontology/cohort/evaluator. Đã kiểm tra reference text và toàn bộ CheXbert reference labels khớp chính xác; checkpoint/hash và bootstrap indices khớp. Đây là so sánh hai cấu hình none/160 và high/8192 cùng các response mới, không phải tác động nhân quả tách riêng effort. Không trộn các điểm CheXbert lịch sử ~0.9 vào phép so sánh này.\n\nLatency và retry được ghi trong ../LATENCY_REPORT.md; parity đầu vào trong ../upstream_parity_vs_none.csv. Một lần thử rule ca 158 bị loại vì response không hợp lệ, đã archive rồi chạy lại cùng cấu hình; 824 report được chọn đều hợp lệ.\n"
    report.write_text(report.read_text(encoding="utf-8")+addition,encoding="utf-8")
    snapshot=HIGH/"code_snapshot/scripts"/Path(__file__).name
    snapshot.parent.mkdir(parents=True,exist_ok=True)
    snapshot.write_bytes(Path(__file__).read_bytes())
    save_json(RUN/"final_execution_audit.json",{"passed":True,"finished_at":datetime.now(timezone.utc).isoformat(),"full_reports":824,"unique_fresh_full_response_ids":len(seen),"model":"gpt-5.6-luna","reasoning_effort":"high","max_output_tokens":8192,"hardware":hardware,"table5_studies":206,"table6_studies":205,"excluded_clinical_study_ids":["1615"],"old_none_reference_labels_identical":True,"old_none_checkpoint_hashes_identical":True,"comparison_arithmetic_device":"cuda:0","none_vs_high_comparison_rows":len(comparison),"upstream_parity_passed":True,"metrics_existing_checksums_verified":len(old_checks),"known_full_failed_attempts":sum(r["archived_failed_attempts"] for r in latency),"posterior_script_sha256":sha256(Path(__file__))})
    save_json(HIGH/"output_checksums.json",{str(p.relative_to(HIGH)):sha256(p) for p in sorted(HIGH.rglob("*")) if p.is_file() and p.name!="output_checksums.json"})
    print(json.dumps({"passed":True,"comparison_rows":len(comparison),"upstream_parity":parity,"latency":latency},indent=2),flush=True)


if __name__=="__main__":
    main()
