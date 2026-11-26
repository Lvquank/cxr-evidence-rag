"""Reproduce Table 4.11 and Table 4.12 with paired bootstrap confidence intervals.

Evaluates three embedding configurations on IU X-Ray with current support vectors:
1. RAD-DINO raw 768-D (data/embeddings)
2. DenseNet pathology 512-D (data/embeddings_densenet_pathology)
3. CheXpert-adapted RAD-DINO 512-D (data/embeddings_chexpert_projection)

Protocol:
- Gallery: 3,818 frontal images / 3,689 studies
- Query set: all frontal images with support.sum() > 0 (904 images / 869 studies)
- Leave-one-study-out: candidate images from the query's study are excluded
- Study-level deduplication: keep best similarity per candidate study
- Binary relevance threshold: 0.5
- Exclude queries without any relevant candidate outside their study
- Paired bootstrap: 2,000 iterations resampled by unique query study ID
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
GALLERY_META_PATH = ROOT / "data" / "reprocessed" / "metadata.jsonl"

CONFIGS = {
    "RAD-DINO raw 768-D": ROOT / "data" / "embeddings",
    "DenseNet pathology 512-D": ROOT / "data" / "embeddings_densenet_pathology",
    "CheXpert-adapted RAD-DINO": ROOT / "data" / "embeddings_chexpert_projection",
}


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def weighted_jaccard(q_supp: np.ndarray, g_supp: np.ndarray) -> np.ndarray:
    num = np.minimum(q_supp, g_supp).sum(axis=1)
    den = np.maximum(q_supp, g_supp).sum(axis=1)
    return np.divide(num, den, out=np.zeros_like(num, dtype=np.float32), where=den > 0)


def dcg(rel: np.ndarray) -> float:
    if len(rel) == 0:
        return 0.0
    disc = np.log2(np.arange(2, len(rel) + 2))
    return float((rel / disc).sum())


def ap_at_k(bin_rel: np.ndarray, total_rel: int, k: int) -> float:
    if total_rel == 0:
        return float("nan")
    bin_k = bin_rel[:k]
    hits = np.cumsum(bin_k)
    prec = hits / np.arange(1, len(bin_k) + 1)
    return float((prec * bin_k).sum() / min(total_rel, k))


def evaluate_queries(emb_dir: Path, meta_rows: list[dict], k_values=(3, 5, 10)):
    embeddings = np.load(emb_dir / "embeddings.npy").astype(np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.maximum(norms, 1e-8)

    support = np.asarray([r.get("support", []) for r in meta_rows], dtype=np.float32)
    studies = [str(r.get("study_id", r.get("uid", ""))).strip() for r in meta_rows]
    image_ids = [str(r.get("id", "")).strip() for r in meta_rows]

    eligible_queries = np.flatnonzero(support.sum(axis=1) > 0)

    results_by_study = {}
    query_details = []

    for q_idx in eligible_queries:
        q_sid = studies[q_idx]
        q_img_id = image_ids[q_idx]
        sims = embeddings @ embeddings[q_idx]
        rel = weighted_jaccard(support[q_idx], support)

        order = np.argsort(-sims, kind="stable")

        ranked_indices = []
        seen_studies = set()
        for c_idx in order:
            c_sid = studies[c_idx]
            if c_sid == q_sid or c_sid in seen_studies:
                continue
            seen_studies.add(c_sid)
            ranked_indices.append(c_idx)

        ranked_indices = np.asarray(ranked_indices, dtype=np.int64)
        ranked_rel = rel[ranked_indices]
        ranked_sim = sims[ranked_indices]
        bin_rel = (ranked_rel >= 0.5).astype(np.float32)
        total_rel = int(bin_rel.sum())

        if total_rel == 0:
            continue

        q_metrics = {
            "query_id": q_img_id,
            "study_id": q_sid,
            "relevant_studies": total_rel,
        }

        for k in k_values:
            p_k = float(bin_rel[:k].mean()) if len(bin_rel) >= k else float(bin_rel.mean())
            wj_k = float(ranked_rel[:k].mean()) if len(ranked_rel) >= k else float(ranked_rel.mean())
            ap_k = ap_at_k(bin_rel, total_rel, k)
            idcg_k = dcg(np.sort(ranked_rel)[::-1][:k])
            ndcg_k = dcg(ranked_rel[:k]) / idcg_k if idcg_k > 0 else float("nan")

            q_metrics[f"P@{k}"] = p_k
            q_metrics[f"WJ@{k}"] = wj_k
            q_metrics[f"mAP@{k}"] = ap_k
            q_metrics[f"nDCG@{k}"] = ndcg_k

        rel_sims = ranked_sim[ranked_rel >= 0.5]
        non_rel_sims = ranked_sim[ranked_rel == 0]
        if len(rel_sims) and len(non_rel_sims):
            delta = float(rel_sims.mean() - non_rel_sims.mean())
        else:
            delta = float("nan")
        q_metrics["Delta_path"] = delta

        query_details.append(q_metrics)

        if q_sid not in results_by_study:
            results_by_study[q_sid] = []
        results_by_study[q_sid].append(q_metrics)

    study_aggregated = {}
    metric_keys = [k for k in query_details[0].keys() if k not in ("query_id", "study_id", "relevant_studies")]
    for sid, q_list in results_by_study.items():
        study_aggregated[sid] = {
            m: float(np.nanmean([item[m] for item in q_list]))
            for m in metric_keys
        }

    return query_details, study_aggregated


def run_bootstrap_analysis(study_data_dict, iterations=2000, seed=42):
    rng = np.random.default_rng(seed)

    first_config = next(iter(study_data_dict.keys()))
    study_ids = sorted(study_data_dict[first_config].keys())
    n_studies = len(study_ids)

    metric_names = list(next(iter(study_data_dict[first_config].values())).keys())

    arrays = {}
    for cname, sdata in study_data_dict.items():
        arr_dict = {}
        for m in metric_names:
            arr_dict[m] = np.array([sdata[sid][m] for sid in study_ids], dtype=np.float32)
        arrays[cname] = arr_dict

    boot_indices = rng.integers(0, n_studies, size=(iterations, n_studies))

    summary = {}
    for cname in study_data_dict.keys():
        c_summary = {}
        for m in metric_names:
            orig = arrays[cname][m]
            point = float(np.nanmean(orig))
            boot_means = np.nanmean(orig[boot_indices], axis=1)
            ci_low = float(np.percentile(boot_means, 2.5))
            ci_high = float(np.percentile(boot_means, 97.5))
            c_summary[m] = {
                "point": point,
                "ci_lower": ci_low,
                "ci_upper": ci_high,
                "formatted": f"{point*100:.2f}% [{ci_low*100:.2f}%; {ci_high*100:.2f}%]" if m != "Delta_path" else f"{point:.4f} [{ci_low:.4f}; {ci_high:.4f}]"
            }
        summary[cname] = c_summary

    diff_summary = {}
    adapted_name = "CheXpert-adapted RAD-DINO"
    for comp_name in ["RAD-DINO raw 768-D", "DenseNet pathology 512-D"]:
        pair_key = f"{adapted_name} - {comp_name}"
        pair_dict = {}
        for m in metric_names:
            diff_orig = arrays[adapted_name][m] - arrays[comp_name][m]
            point = float(np.nanmean(diff_orig))
            diff_boots = np.nanmean(diff_orig[boot_indices], axis=1)
            ci_low = float(np.percentile(diff_boots, 2.5))
            ci_high = float(np.percentile(diff_boots, 97.5))
            pair_dict[m] = {
                "point": point,
                "ci_lower": ci_low,
                "ci_upper": ci_high,
                "formatted": f"{point*100:+.2f}% [{ci_low*100:+.2f}%; {ci_high*100:+.2f}%]" if m != "Delta_path" else f"{point:+.4f} [{ci_low:+.4f}; {ci_high:+.4f}]"
            }
        diff_summary[pair_key] = pair_dict

    return summary, diff_summary, n_studies


def main():
    print("Loading metadata...")
    meta_rows = read_jsonl(GALLERY_META_PATH)
    print(f"Total metadata rows: {len(meta_rows)}")

    study_data = {}
    query_details_all = {}

    for name, path in CONFIGS.items():
        print(f"Evaluating {name} from {path}...")
        q_det, s_agg = evaluate_queries(path, meta_rows)
        study_data[name] = s_agg
        query_details_all[name] = q_det
        print(f"  Evaluated {len(q_det)} query images across {len(s_agg)} studies.")

    print("\nRunning 2,000 paired bootstrap iterations...")
    summary, diff_summary, n_studies = run_bootstrap_analysis(study_data, iterations=2000, seed=42)

    out_dir = ROOT / "outputs" / "table_4_11_4_12_rerun"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "table_4_11_summary.json", "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "differences": diff_summary, "n_studies": n_studies}, f, indent=2)

    for name, q_det in query_details_all.items():
        clean_name = name.lower().replace(" ", "_").replace("-", "_")
        df_q = pd.DataFrame(q_det)
        df_q.to_csv(out_dir / f"per_query_{clean_name}.csv", index=False)

    print("\n" + "="*80)
    print("BẢNG 4.11: SO SÁNH CHẤT LƯỢNG TRUY XUẤT CÁC CẤU HÌNH EMBEDDING (K=5, K=10)")
    print("="*80)
    header = f"{'Cấu hình':<28} | {'P@5':<24} | {'P@10':<24} | {'mAP@5':<24} | {'mAP@10':<24} | {'nDCG@5':<24} | {'nDCG@10':<24} | {'Delta_path':<22}"
    print(header)
    print("-" * len(header))
    for name in CONFIGS.keys():
        s = summary[name]
        print(f"{name:<28} | {s['P@5']['formatted']:<24} | {s['P@10']['formatted']:<24} | {s['mAP@5']['formatted']:<24} | {s['mAP@10']['formatted']:<24} | {s['nDCG@5']['formatted']:<24} | {s['nDCG@10']['formatted']:<24} | {s['Delta_path']['formatted']:<22}")

    print("\n" + "="*80)
    print("CHÊNH LỆCH GHÉP CẶP (PAIRED BOOTSTRAP CI 95%):")
    print("="*80)
    for pair_name, diffs in diff_summary.items():
        print(f"\n--- {pair_name} ---")
        for m in ["P@5", "P@10", "mAP@5", "mAP@10", "nDCG@5", "nDCG@10", "Delta_path"]:
            print(f"  {m:<12}: {diffs[m]['formatted']}")

    print("\n" + "="*80)
    print("BẢNG 4.12: CHỈ SỐ CheXpert-adapted RAD-DINO THEO ĐỘ SÂU TOP-K")
    print("="*80)
    adapted_s = summary["CheXpert-adapted RAD-DINO"]
    print(f"{'Chỉ số':<20} | {'@3':<25} | {'@5':<25} | {'@10':<25}")
    print("-" * 75)
    for metric_prefix, label in [("WJ", "WeightedJaccard"), ("P", "Precision"), ("mAP", "mAP"), ("nDCG", "nDCG")]:
        row_str = f"{label:<20} | {adapted_s[f'{metric_prefix}@3']['formatted']:<25} | {adapted_s[f'{metric_prefix}@5']['formatted']:<25} | {adapted_s[f'{metric_prefix}@10']['formatted']:<25}"
        print(row_str)

    print("\nKết quả đã được lưu tại:", out_dir)


if __name__ == "__main__":
    main()
