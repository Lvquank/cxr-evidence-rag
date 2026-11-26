"""Recompute the unified IU X-Ray 206-study retrieval tables and saved-report controls."""

import argparse
import csv
import gzip
import hashlib
import json
import logging
import os
import platform
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from importlib import metadata as packages
from itertools import product
from pathlib import Path

import numpy as np

from evaluate_llm_outputs import load_ontology, normalize_target_problems, split_problems
from unified_206_metrics import (KS, RANK_METRICS, THRESHOLD, delta_path, fused_order, ideal,
                                 interval, label_metrics, mean_samples, rank_metrics,
                                 relevance, representatives, self_check)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.retriever.pathology_support import CHEXPERT_LABELS, support_from_metadata

SEED, DRAWS = 20260904, 2000
LAMBDAS = [i / 10 for i in reversed(range(11))]
LOCAL_PARITY_ATOL = 1e-4
ENCODERS = {
    'raw': ('embeddings', 768, 'microsoft/rad-dino-maira-2', 'rad_dino_768'),
    'densenet': ('embeddings_densenet_pathology', 512, 'local/chexpert-relation-aware-densenet121', 'densenet_512'),
    'adapted': ('embeddings_chexpert_projection', 512, 'microsoft/rad-dino-maira-2', 'pathology_512'),
}
MANIFEST = ROOT / 'data/evaluations/leave_one_out_206_manifest.json'
META = ROOT / 'data/embeddings/metadata.jsonl'
REFS = ROOT / 'src/data/indiana_reports.csv'
ONTOLOGY = ROOT / 'config/evaluation_ontology.json'
THRESHOLDS = ROOT / 'config/densenet_adaptive_thresholds.json'
FLOW = ROOT / 'outputs/flow1'
HIGH = ROOT / 'outputs/high_effort_4branches_20261031'
LATENCY = ROOT / 'outputs/recompute_table5_table6_20261031'
LATENCY_RULE = ROOT / 'outputs/rule_latency_206_20261029'
GATE = ROOT / 'src/learned_gate_oof_206'
CHECKPOINT = ROOT / 'model/best_E2_relation_aware_weighted_bce.pt'
THESIS = ROOT / 'docs/59_KLTN_CaoHuynhVoThanh_LeVinhQuang_15-11_GopY_GVHD_M7_M8_M9_20261122.docx'
EARLIER_THESIS = ROOT / 'docs/59_KLTN_CaoHuynhVoThanh_LeVinhQuang_15112026.docx'
TZ = timezone(timedelta(hours=7))
LOG = logging.getLogger('unified206')


def check(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def str_sha(value):
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def write_json(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    temp.replace(path)


def write_csv(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def display(value):
    return 'N/A' if value is None else f'{value:.4f}'


def write_table(out, name, title, rows, columns):
    write_csv(out / f'{name}.csv', rows)
    headings = [*columns, 'Estimate', '95% CI', 'N study', 'Đơn vị']
    body = [f'# {title}', '', '| ' + ' | '.join(headings) + ' |', '| ' + ' | '.join(['---'] * len(headings)) + ' |']
    for row in rows:
        ci = f"[{display(row['ci95_low'])}, {display(row['ci95_high'])}]"
        items = [*(str(row[x]) for x in columns), display(row['estimate']), ci, str(row['n']), row['unit']]
        body.append('| ' + ' | '.join(items) + ' |')
    (out / f'{name}.md').write_text('\n'.join(body) + '\n', encoding='utf-8')


class Audit:
    def __init__(self):
        self.files = {}

    def add(self, path, role, run_id='', studies='', images='', checkpoint='', metric='', labels='', note=''):
        key = str(path.resolve())
        if key not in self.files:
            self.files[key] = dict(path=key, sha256=sha(path) if path.is_file() else '', role=role,
                                   run_id=run_id, study_count=studies, image_count=images,
                                   checkpoint=checkpoint, formula_or_protocol=metric, label_source=labels, note=note)
        return self.files[key]['sha256']


def required_keyed(rows, key, expected):
    indexed = {row[key]: row for row in rows}
    check(len(rows) == len(indexed) and set(indexed) == set(expected), f'Duplicate or mismatched {key}')
    return indexed


def load_gallery(audit):
    meta = read_jsonl(META)
    ids = np.array([row['id'] for row in meta])
    studies = np.array([row['study_id'] for row in meta])
    support = np.array([row['support'] for row in meta], dtype=float)
    check(len(ids) == len(set(ids)) == 3818 and len(set(studies)) == 3689, 'Gallery count or duplicate image IDs')
    check(set(row['dataset'] for row in meta) == {'IU-Xray'} and all(studies), 'Unexpected gallery study namespace')
    check(support.shape == (3818, 6) and np.isfinite(support).all(), 'Invalid support vectors')
    by_study = {}
    for row in meta:
        expected, _ = support_from_metadata({'mesh': row['mesh']})
        check(row['support'] == list(expected.values()), f"MeSH support mismatch for {row['id']}")
        check(by_study.setdefault(row['study_id'], row['support']) == row['support'],
              f"Support varies by image for study {row['study_id']}")
    audit.add(META, 'Canonical MeSH support and gallery', 'unified_206', 3689, 3818, labels='MeSH only')
    return meta, ids, studies, support


def load_vectors(audit, meta, ids):
    vectors, details = {}, {}
    canonical = {row['id']: row for row in meta}
    dense_sha = audit.add(CHECKPOINT, 'DenseNet121 classifier and local reranker checkpoint')
    for name, (directory, dim, model, old_dir) in ENCODERS.items():
        base = ROOT / 'data' / directory
        manifest = read_json(base / 'embedding_manifest.json')
        check((manifest['embedding_model'], manifest['embedding_dimension'], manifest['normalization']) ==
              (model, dim, 'L2'), f'Model manifest mismatch for {name}')
        revision = manifest['model_revision']
        if name == 'densenet':
            check(revision == 'sha256:' + dense_sha, 'DenseNet backbone checkpoint mismatch')
        else:
            check(revision == 'a6132efb9268859545e9f68be18a76a6473f174a', f'RAD-DINO revision mismatch for {name}')
        projection = manifest.get('projection')
        projection_sha = ''
        backbone_sha = dense_sha
        if projection:
            basename = projection['path'].replace('\\', '/').rsplit('/', 1)[-1]
            projection_sha = audit.add(ROOT / 'model' / basename, f'{name} projection checkpoint', name)
            check(projection_sha and projection['output_dim'] == dim, f'Projection checkpoint missing for {name}')
        if name != 'densenet':
            snapshot = Path.home() / '.cache/huggingface/hub/models--microsoft--rad-dino-maira-2' / 'snapshots' / revision
            backbone_sha = audit.add(snapshot / 'model.safetensors', 'Cached RAD-DINO upstream model', revision)
        if name == 'adapted':
            check(projection['pipeline'] == 'rad_dino_chexpert_pre_adapted_7cohort'
                  and projection['training']['dataset'] == 'CheXpert frontal', 'Wrong adapted projection')
        rows = read_jsonl(base / 'metadata.jsonl')
        ordered = (base / 'ids.txt').read_text(encoding='utf-8').splitlines()
        check(ordered == [row['id'] for row in rows] and len(ordered) == len(set(ordered))
              and set(ordered) == set(ids) and all(row == canonical[row['id']] for row in rows),
              f'Ids, metadata or support differ for {name}')
        raw = np.load(base / 'embeddings.npy', allow_pickle=False).astype(np.float32)
        check(raw.shape == (len(ids), dim) and np.isfinite(raw).all(), f'Invalid vector array for {name}')
        norm = np.linalg.norm(raw, axis=1)
        np.testing.assert_allclose(norm, 1, atol=1e-5, rtol=0)
        lookup = {image: i for i, image in enumerate(ordered)}
        vectors[name] = (raw / norm[:, None])[[lookup[image] for image in ids]]
        for file in ('metadata.jsonl', 'ids.txt', 'embeddings.npy', 'embedding_manifest.json'):
            audit.add(base / file, f'{name} {file}', name, 3689, 3818,
                      projection_sha or backbone_sha, 'L2-normalized global cosine', 'MeSH support via canonical ID')
        old_path = ROOT / 'data/evaluations' / old_dir / 'summary.json'
        old = read_json(old_path)
        audit.add(old_path, 'Historical 1246-query benchmark', old_dir, metric=old['relevance'],
                  images=old['evaluated_queries'], note='Historical pathology-only image-query denominator')
        details[name] = dict(manifest=manifest, path=str(base), backbone_checkpoint_sha256=backbone_sha,
                             projection_checkpoint_sha256=projection_sha, norms=[float(norm.min()), float(norm.max())],
                             id_order_sha256=str_sha('\n'.join(ordered)), source_metadata_sha256=sha(base / 'metadata.jsonl'),
                             historical_summary=old, historical_gallery_is_selected_variant=name != 'adapted')
    return vectors, details


def load_cohort(audit, meta, ids, support):
    manifest = read_json(MANIFEST)
    requests = manifest['requests']
    all_ids = [req['study_id'] for req in requests]
    check(manifest['seed'] == 20260830 and len(all_ids) == len(set(all_ids)) == 206, 'Cohort size or seed changed')
    for path, role in ((MANIFEST, 'Frozen query manifest'), (ONTOLOGY, 'Target-label ontology'), (REFS, 'Reference labels/text')):
        audit.add(path, role, studies=206, labels='Problems for clinical target labels' if path == REFS else '')
    ontology = load_ontology(ONTOLOGY)
    check(tuple(ontology['target_labels']) == CHEXPERT_LABELS, 'Label order mismatch')
    reference_rows = read_csv(REFS)
    refs = {row['uid']: row for row in reference_rows}
    check(len(refs) == len(reference_rows), 'Duplicate report reference UID')
    image_index = {name: i for i, name in enumerate(ids)}
    cohort = []
    for i, req in enumerate(requests):
        sid, image = req['study_id'], req['image_id']
        idx = image_index[image]
        check(meta[idx]['study_id'] == sid, f'Manifest query image mismatch for {sid}')
        ref = refs[sid]
        gold = normalize_target_problems(split_problems(ref['Problems']), ontology)
        pathology = gold['target_problems']
        cohort.append(dict(order=i, study_id=sid, query_image_id=image, request_id=req['request_id'],
                           recruitment_group=req['case_label'], support=json.dumps(support[idx].tolist()),
                           gold_target_labels=';'.join(pathology), target_positive=bool(pathology),
                           manifest_normal=req['case_label'].lower() == 'normal', zero_support=not bool(support[idx].any()),
                           gold_normal=gold['normal'] and not pathology, gallery_index=idx))
    check(sum(x['manifest_normal'] for x in cohort) == 30, 'Expected 30 manifest Normal studies')
    return cohort, refs, ontology


def archived_path(value):
    source = Path(value)
    if source.is_file():
        return source
    parts = value.replace('\\', '/').split('/')
    return ROOT.joinpath(*parts[parts.index('image-rag') + 1:])


def load_run(audit, directory, cohort, refs, ontology):
    manifest_path = directory / 'input_manifest.json'
    run = read_json(manifest_path)
    run_id = directory.parent.name if directory == HIGH / 'metrics' else directory.name
    audit.add(manifest_path, 'Report input manifest', run_id, 206, 206)
    ids = {x['study_id'] for x in cohort}
    check(run['source_cohort_count'] == 206 and set(run['source_study_ids']) == ids, f'{run_id} cohort differs')
    for source, expected_hash in run['source_files'].items():
        path = archived_path(source)
        check(audit.add(path, 'Frozen report run input', run_id) == expected_hash, f'Archived source hash changed: {path}')
    prov_path = directory / 'report_provenance.csv'
    prov = read_csv(prov_path)
    audit.add(prov_path, 'Archived summary path, hash and response ID', run_id, 206, 206, labels='Problems')
    responses = set()
    results, original = {}, {}
    report_rows = []
    for branch in ('baseline_a', 'baseline_b', 'structured_rule', 'structured_learned_gate'):
        config = run['branches'][branch]['run_config']
        check(config['manifest_sha256'] == sha(MANIFEST), f'{run_id}/{branch} manifest hash')
        per_study = directory / 'per_study' / f'{branch}_target_labels.csv'
        audit.add(per_study, f'{branch} archived predictions', run_id, 206, 206, labels='Problems')
        archived = required_keyed(read_csv(per_study), 'study_id', ids)
        rows = required_keyed([row for row in prov if row['pipeline'] == branch], 'study_id', ids)
        results[branch] = []
        for entry in cohort:
            sid = entry['study_id']
            row = rows[sid]
            check(row['request_id'] == entry['request_id'], f'{run_id}/{branch}/{sid} request ID')
            path = archived_path(row['summary_path'])
            check(audit.add(path, 'Saved report summary', f'{run_id}/{branch}', 1, 1, labels='Problems') == row['summary_sha256'],
                  f'{run_id}/{branch}/{sid} summary hash')
            summary = read_json(path)
            prediction = summary['prediction']
            found = set(normalize_target_problems(split_problems(prediction['problems']), ontology)['target_problems'])
            expected = archived[sid]
            api = summary['api']
            effort = summary.get('reasoning_effort') or 'none'
            check(api['response_id'] == row['response_id'] and api['response_id'] not in responses
                  and not api.get('cached', False) and summary['model'] == config['model']
                  and effort == config['reasoning_effort'], f'{run_id}/{branch}/{sid} API provenance')
            responses.add(api['response_id'])
            check(str(summary['source']['study_id']) == sid and Path(summary['source']['image_name']).stem == entry['query_image_id'],
                  f'{run_id}/{branch}/{sid} report study/image')
            check(str_sha(prediction['findings'].strip() + '\n' + prediction['impression'].strip()) == row['prediction_text_sha256']
                  and str_sha(refs[sid]['findings'].strip() + '\n' + refs[sid]['impression'].strip()) == row['reference_text_sha256'],
                  f'{run_id}/{branch}/{sid} report text hashes')
            check(set(split_problems(expected['predicted_target_problems'])) == found
                  and expected['gold_target_problems'] == entry['gold_target_labels']
                  and (expected['gold_normal'] == 'True') == entry['gold_normal']
                  and (expected['predicted_normal'] == 'True') == (not found), f'{run_id}/{branch}/{sid} target labels')
            results[branch].append(found)
            usage = api['usage']
            report_rows.append(dict(run_id=run_id, **row, model=config['model'], reasoning_effort=effort,
                                    max_output_tokens=config['max_output_tokens'], input_tokens=usage.get('input_tokens'),
                                    output_tokens=usage.get('output_tokens'),
                                    reasoning_tokens=(usage.get('output_tokens_details') or {}).get('reasoning_tokens')))
    old_table = directory / 'table5.csv'
    audit.add(old_table, 'Published Table 5 cells for exact parity test', run_id, 206)
    original = {row['pipeline']: row for row in read_csv(old_table)}
    return results, original, report_rows, run


def load_controls(audit, cohort, image_to_row, branch, run_id):
    thresholds_hash = audit.add(THRESHOLDS, 'Frozen production classifier thresholds', run_id)
    checkpoint_hash = audit.add(CHECKPOINT, 'DenseNet121 classifier and local reranker checkpoint')
    config = read_json(THRESHOLDS)
    thresholds = {label: payload['threshold'] for label, payload in config['labels'].items()}
    predictions = {'C0': [], 'K0': [], 'C0_and_K0': []}
    controls, status = [], []
    recorded_threshold_hashes = set()
    raw_dir = branch / 'raw_evidence'
    check({p.stem for p in raw_dir.glob('*.json')} == {req['request_id'] for req in cohort}, 'Raw evidence coverage')
    for req in cohort:
        sid = req['study_id']
        path = raw_dir / f"{req['request_id']}.json"
        evidence_hash = audit.add(path, 'Frozen classifier and rerank evidence', run_id, 1, 1, checkpoint_hash,
                                  'C0 prob >= threshold; K0 >=3 study support>0', 'MeSH retrieval support')
        entry = read_json(path)
        classifier, retriever = entry['classifier'], entry['retriever']
        check(Path(entry['image_path']).stem == req['query_image_id'] and retriever['excluded_study_ids'] == [sid],
              f'{run_id}/{sid} evidence identity')
        frozen = classifier['thresholds']
        recorded_threshold_hashes.add(classifier['thresholds_sha256'])
        check(frozen == thresholds and classifier['threshold_version'] == config['version'], f'{run_id}/{sid} threshold values')
        probabilities = classifier['raw_probabilities']
        check(set(probabilities) == set(CHEXPERT_LABELS) and all(0 <= p <= 1 for p in probabilities.values()),
              f'{run_id}/{sid} probabilities')
        c0 = {label for label in CHEXPERT_LABELS if probabilities[label] >= frozen[label]}
        check(c0 == set(classifier['selected_labels']), f'{run_id}/{sid} C0 differs from selected_labels')
        check(retriever['local_rerank_enabled'] == bool(c0), f'{run_id}/{sid} unexpected local routing')
        top = retriever['common_evidence']
        check([item['rank'] for item in top] == list(range(1, 6))
              and len({x['study_id'] for x in top}) == 5 and sid not in [x['study_id'] for x in top], f'{run_id}/{sid} top5')
        supports = []
        for hit in top:
            meta = image_to_row[hit['id']]
            check(hit['study_id'] == meta['study_id'] and hit['mesh'] == meta['mesh'], f'{run_id}/{sid} stale Top-5 MeSH')
            support, _ = support_from_metadata({'mesh': hit['mesh']})
            supports.append(list(support.values()))
        votes = (np.asarray(supports) > 0).sum(axis=0)
        k0 = {label for label, hits in zip(CHEXPERT_LABELS, votes) if hits >= 3}
        for name, labels in zip(predictions, (c0, k0, c0 & k0)):
            predictions[name].append(labels)
        for label, hits in zip(CHEXPERT_LABELS, votes):
            controls.append(dict(run_id=run_id, study_id=sid, pathology=label, probability=probabilities[label],
                                 threshold=frozen[label], threshold_source_sha256=classifier['thresholds_sha256'],
                                 current_threshold_file_sha256=thresholds_hash, local_rerank_enabled=retriever['local_rerank_enabled'],
                                 top5_study_ids=';'.join(x['study_id'] for x in top),
                                 top5_image_ids=';'.join(x['id'] for x in top), top5_support_count=int(hits),
                                 c0=int(label in c0), k0=int(label in k0), c0_and_k0=int(label in c0 & k0),
                                 evidence_path=str(path), evidence_sha256=evidence_hash))
        all_status = retriever['label_evidence']
        check(len(all_status) == len(c0) and {item['label'] for item in all_status} == c0, f'{run_id}/{sid} status scope')
        for item in all_status:
            status.append(dict(run_id=run_id, study_id=sid, pathology=item['label'],
                               evidence_status=item['evidence_status'], evidence_sha256=evidence_hash))
    check(len(recorded_threshold_hashes) == 1, f'Inconsistent threshold snapshot for {run_id}')
    if recorded_threshold_hashes != {thresholds_hash}:
        LOG.info('%s has a different threshold file SHA than the checkout; frozen per-case values and version agree.', run_id)
    return predictions, controls, status


def load_local_cache(audit, path, cohort, meta):
    cache = read_json(path)
    config = cache['protocol']
    check(config['cache_candidates'] == config['local_rerank_candidates'] == 50
          and config['same_study_excluded'] and config['deduplicate_by'] == 'dataset + study_id'
          and config['production_equivalent_global_weight'] == .9, f'Unexpected cache protocol: {path}')
    queries = required_keyed(cache['queries'], 'query_study_id', [row['study_id'] for row in cohort])
    images = {x['id']: x['study_id'] for x in meta}
    for row in cohort:
        q = queries[row['study_id']]
        c = q['candidates']
        check(q['query_id'] == row['query_image_id'] and len(c) == 50 and len({x['study_id'] for x in c}) == 50
              and all(x['study_id'] != row['study_id'] and x['study_id'] == images[x['id']] for x in c)
              and [x['global_rank'] for x in c] == list(range(1, 51)), f"Unexpected cache identity: {q['query_id']}")
    audit.add(path, 'Cached global Top-50 and DenseNet spatial scores', path.parent.name, 206, 206,
              metric='Frozen study pool and symmetric patch maximum cosine', labels='Original MeSH relevance audited separately')
    return queries, config


def compute_local(out, audit, queries, meta, ids, pools, global_scores, cohort):
    local = np.full((len(cohort), 50), np.nan)
    missing = []
    for i, row in enumerate(cohort):
        cache = {x['id']: x for x in queries[row['study_id']]['candidates']}
        for rank, index in enumerate(pools[i]):
            entry = cache.get(ids[index], {})
            value = entry.get('local_similarity')
            if value is None or not np.isfinite(value):
                missing.append((i, rank, index))
            else:
                local[i, rank] = value
    repairs = []
    if missing:
        import torch
        from predict_chexpert_6labels import load_model
        from src.retriever.densenet_local_reranker import DenseNetLocalFeatureReranker

        check(torch.cuda.is_available(), 'Missing local scores require CUDA. No CPU fallback.')
        accelerator = torch.cuda.get_device_name(0)
        device = torch.device('cuda:0')
        LOG.info('Local cache gaps: %s pairs. CUDA device: %s', len(missing), accelerator)
        classifier, _ = load_model(CHECKPOINT, device)
        reranker = DenseNetLocalFeatureReranker(classifier.backbone, device)
        all_images = {row['id']: row for row in meta}
        for q_index in sorted({i for i, _, _ in missing}):
            row = cohort[q_index]
            candidates = queries[row['study_id']]['candidates']
            known = [x for x in candidates if x.get('local_similarity') is not None and np.isfinite(x['local_similarity'])]
            gaps = [(rank, idx) for i, rank, idx in missing if i == q_index]
            image_ids = [x['id'] for x in known] + [str(ids[idx]) for rank, idx in gaps]
            paths = {image: str(ROOT / 'data/reprocessed' / all_images[image]['path'])
                     for image in [row['query_image_id'], *image_ids]}
            for path in paths.values():
                check(audit.add(Path(path), 'CUDA local gap image or cache calibration image', 'unified_206', images=1),
                      f'CUDA image missing: {path}')
            scores = reranker.score_paths(paths[row['query_image_id']], [paths[image] for image in image_ids])
            max_error = max(abs(scores[paths[x['id']]] - x['local_similarity']) for x in known)
            check(max_error < LOCAL_PARITY_ATOL, f"Saved local scores fail CUDA calibration for {row['study_id']}: {max_error}")
            for rank, idx in gaps:
                score = scores[paths[ids[idx]]]
                local[q_index, rank] = score
                repairs.append(dict(study_id=row['study_id'], query_id=row['query_image_id'], image_id=str(ids[idx]),
                                    study_candidate=meta[idx]['study_id'], global_rank=rank + 1, local_score=score,
                                    cached_calibration_pairs=len(known), calibration_max_abs_error=max_error,
                                    calibration_atol=LOCAL_PARITY_ATOL, top10_matches_fully_rescored_pool=True,
                                    device='cuda:0', gpu=accelerator, torch=torch.__version__, cuda=torch.version.cuda))
            pool_ids = ids[pools[q_index]]
            rescored = np.array([scores[paths[image]] for image in pool_ids])
            for weight in LAMBDAS:
                score = global_scores[q_index, :len(pool_ids)].astype(float)
                hybrid_rank, _ = fused_order(score, local[q_index], pool_ids, weight)
                rescored_rank, _ = fused_order(score, rescored, pool_ids, weight)
                np.testing.assert_equal(hybrid_rank[:max(KS)], rescored_rank[:max(KS)])
            reranker.clear_cache()
    write_json(out / 'local_score_repairs.json', repairs)
    return local, repairs


def get_global_orders(vectors, ids, studies, cohort):
    orders, ranked_scores = {}, {}
    for encoder, emb in vectors.items():
        rows, scores = [], []
        for row in cohort:
            cosine = emb @ emb[row['gallery_index']]
            rank = representatives(cosine, ids, studies, row['study_id'])
            check(len(rank) == 3688, f"Missing study in {encoder}/{row['study_id']}")
            rows.append(rank)
            scores.append(cosine[rank])
        orders[encoder], ranked_scores[encoder] = np.array(rows, dtype=np.int32), np.array(scores)
    return orders, ranked_scores


def calculate_retrieval(out, cohort, ids, support, orders, scores, local, adapted_cache, raw_cache):
    metadata = {'raw': ('raw', 1.), 'densenet': ('densenet', 1.)}
    metadata.update({f'adapted_{weight:.1f}': ('adapted', weight) for weight in LAMBDAS})
    group = np.array([row['manifest_normal'] for row in cohort])
    rows, values, independent, cache_diff = [], {}, {}, []
    gallery_relevance = defaultdict(list)
    ticks = np.rint(support * 100).astype(np.int64)
    np.testing.assert_array_equal(ticks / 100, support)
    with gzip.open(out / 'retrieval_rankings.jsonl.gz', 'wt', encoding='utf-8') as ranking_stream:
        for i, row in enumerate(cohort):
            graded = relevance(ticks[row['gallery_index']], ticks)
            by_encoder = {name: graded[index[i]] for name, index in orders.items()}
            r, idcg = ideal(by_encoder['adapted'])
            for encoder, ranked in by_encoder.items():
                check(ideal(ranked) == (r, idcg), f"IDCG/R changed with encoder for {row['study_id']}")
                gallery_relevance[encoder].append(ranked)
            for encoder, queries in [('raw', raw_cache), ('adapted', adapted_cache)]:
                old = queries[row['study_id']]
                hits = old['candidates']
                previous = {x['id']: x for x in hits}
                current = ids[orders[encoder][i, :50]].tolist()
                score_errors = [abs(float(score) - previous[image]['global_similarity'])
                                for image, score in zip(current, scores[encoder][i]) if image in previous]
                check(max(score_errors) < 1e-5, f'{encoder} cache does not match frozen embedding scores')
                cache_diff.append(dict(encoder=encoder, study_id=row['study_id'], ordered_top50_matches=current == list(previous),
                                       max_global_score_error=max(score_errors), new_ids=';'.join(sorted(set(current) - previous.keys())),
                                       removed_ids=';'.join(sorted(previous.keys() - set(current))),
                                       old_r=old['total_relevant_in_gallery'], r=r, old_idcg5=old['ideal_dcg']['5'], idcg5=idcg[5],
                                       max_shared_relevance_error=max(abs(graded[idx] - previous[image]['relevance'])
                                                                      for image, idx in zip(current, orders[encoder][i]) if image in previous)))
            for config, (encoder, weight) in metadata.items():
                all_indices = orders[encoder][i]
                indices = all_indices[:50] if encoder == 'adapted' else all_indices
                global_scores = scores[encoder][i, :len(indices)].astype(float)
                if encoder == 'adapted':
                    order, fused = fused_order(global_scores, local[i], ids[indices], weight)
                else:
                    order, fused = np.arange(len(indices)), global_scores
                top = order[:50]
                ranking_stream.write(json.dumps(dict(configuration=config, study_id=row['study_id'],
                    indices=indices[top].tolist(), image_ids=ids[indices[top]].tolist(),
                    relevance=graded[indices[top]].tolist(), global_score=global_scores[top].tolist(),
                    local_score=local[i, top].tolist() if encoder == 'adapted' else None,
                    ranking_score=fused[top].tolist()), separators=(',', ':')) + '\n')
                global_delta = delta_path(by_encoder[encoder], scores[encoder][i]) if weight == 1. else None
                values.setdefault((config, 'delta_path_global', 0), []).append(global_delta)
                for k in KS:
                    ranking = indices[order[:k]]
                    metrics = rank_metrics(graded[ranking], k, r, idcg[k])
                    top_delta = delta_path(graded[ranking], global_scores[order[:k]])
                    for metric, value in {**metrics, 'delta_path_topk': top_delta}.items():
                        values.setdefault((config, metric, k), []).append(value)
                    rows.append(dict(configuration=config, encoder=encoder, lambda_global=weight, k=k,
                                     study_id=row['study_id'], query_image_id=row['query_image_id'],
                                     recruitment_group=row['recruitment_group'], r=r, idcg=idcg[k],
                                     r_zero=r == 0, idcg_zero=idcg[k] == 0, eligible_gallery_studies=3688,
                                     candidate_count=len(indices), **metrics, delta_path_global=global_delta,
                                     delta_path_topk=top_delta,
                                     delta_global_na=('reranker' if weight != 1 else 'missing score group') if global_delta is None else '',
                                     delta_topk_na='missing score group' if top_delta is None else ''))
                    if config == 'adapted_1.0':
                        full = rank_metrics(by_encoder['adapted'], k, r, idcg[k])
                        independent[(row['study_id'], k)] = full
                        np.testing.assert_allclose(list(full.values()), list(metrics.values()), atol=1e-10, rtol=0)
    arrays = {key: np.array(series, dtype=float) for key, series in values.items()}
    np.savez_compressed(out / 'full_gallery_rankings.npz', **{
        f'{encoder}_{kind}': data for encoder in orders for kind, data in (
            ('gallery_indices', orders[encoder]), ('cosine', scores[encoder]), ('relevance', gallery_relevance[encoder]))})
    write_csv(out / 'retrieval_per_study.csv', rows)
    write_csv(out / 'cache_reconciliation.csv', cache_diff)
    return arrays, metadata, group, rows, cache_diff, len(independent)


def bootstrap(out, group):
    rng = np.random.default_rng(SEED)
    groups = {'all': np.arange(len(group)), 'normal': np.flatnonzero(group), 'other': np.flatnonzero(~group)}
    indices = {name: ids[rng.integers(0, len(ids), size=(DRAWS, len(ids)), dtype=np.int32)]
               for name, ids in groups.items()}
    np.save(out / 'bootstrap_indices.npy', indices['all'])
    np.savez_compressed(out / 'subgroup_bootstrap_indices.npz', normal=indices['normal'], other=indices['other'])
    return groups, indices


def summarize_retrieval(out, arrays, configs, groups, indices):
    rows, paired, subgroup_deltas = [], [], []
    for (config, metric, k), values in arrays.items():
        encoder, weight = configs[config]
        unit = 'cosine' if metric.startswith('delta_') else 'fraction'
        points = {}
        for group, ids in groups.items():
            points[group] = mean_samples(values, ids)
            rows.append(dict(configuration=config, encoder=encoder, lambda_global=weight, k=k or 'global',
                             metric=metric, cohort=group, cohort_n=len(ids),
                             **interval(points[group], mean_samples(values, indices[group]), np.isfinite(values[ids]).sum(), unit)))
        valid_norm = np.isfinite(values[groups['normal']]).sum()
        valid_other = np.isfinite(values[groups['other']]).sum()
        if valid_norm and valid_other:
            np.testing.assert_allclose(points['all'], (valid_norm * points['normal'] + valid_other * points['other']) /
                                       (valid_norm + valid_other), atol=1e-12, rtol=0)
    pairs = [('adapted_1.0', 'raw'), ('densenet', 'raw'), ('adapted_1.0', 'densenet')]
    pairs += [(f'adapted_{weight:.1f}', f'adapted_{ref:.1f}') for ref in (1., .9) for weight in LAMBDAS]
    metric_ranges = [*product(RANK_METRICS, KS), ('delta_path_global', 0), *product(['delta_path_topk'], KS)]
    for left, right in pairs:
        for metric, k in metric_ranges:
            a, b = arrays[left, metric, k], arrays[right, metric, k]
            # A cosine margin comparison requires both margins on the same study.
            values = a - b
            for group, ids in groups.items():
                valid = int(np.isfinite(values[ids]).sum())
                if not valid:
                    continue
                prefix = dict(left=left, right=right, metric=metric, k=k or 'global', cohort=group)
                unit = 'cosine difference' if metric.startswith('delta_') else 'fraction difference'
                paired.append(dict(**prefix, **interval(mean_samples(values, ids), mean_samples(values, indices[group]), valid, unit)))
            if (left, right) == ('adapted_1.0', 'raw') and metric in RANK_METRICS:
                norm = mean_samples(values, indices['normal'])
                other = mean_samples(values, indices['other'])
                n = len(groups['normal'])
                norm_point = mean_samples(values, groups['normal'])
                other_point = mean_samples(values, groups['other'])
                w = n / len(values)
                subgroup_deltas.append(dict(metric=metric, k=k, normal_n=n, other_n=len(values)-n,
                    normal_gain=norm_point, other_gain=other_point, normal_weighted_contribution=w * norm_point,
                    other_weighted_contribution=(1 - w) * other_point,
                    contrast='gain_normal_minus_gain_other',
                    **interval(norm_point-other_point, norm-other, len(values), 'fraction difference')))
                np.testing.assert_allclose(w*norm_point+(1-w)*other_point, values.mean(), atol=1e-12, rtol=0)
    base = [row for row in rows if row['cohort'] == 'all']
    table411 = [row for row in base if row['lambda_global'] == 1 and
                ((row['k'] in (5, 10) and row['metric'] in ('precision', 'map', 'ndcg')) or row['metric'] == 'delta_path_global')]
    table412 = [row for row in base if row['encoder'] == 'adapted' and row['lambda_global'] in (1., .9) and row['metric'] in RANK_METRICS]
    table413 = [row for row in base if row['encoder'] == 'adapted' and row['metric'] in RANK_METRICS]
    columns = ('encoder', 'lambda_global', 'k', 'metric')
    write_table(out, 'table411', 'Bảng 4.11 So sánh embedding bằng cosine toàn cục', table411, columns)
    write_table(out, 'table412', 'Bảng 4.12 CheXpert-adapted RAD-DINO Top-K với λ bằng 0,9 và 1,0', table412, columns)
    write_table(out, 'table413', 'Bảng 4.13 Phân tích độ nhạy λ trong Top-50 study toàn cục', table413, columns)
    write_csv(out / 'normal_subgroup_metrics.csv', rows)
    write_csv(out / 'normal_subgroup_contrasts.csv', subgroup_deltas)
    write_csv(out / 'retrieval_paired_ci.csv', paired)
    common = {(row['configuration'], row['k'], row['metric']): row for row in table413}
    cross_checks = []
    for name, table in [('table411', table411), ('table412', table412)]:
        for row in table:
            key = row['configuration'], row['k'], row['metric']
            if key in common:
                other = common[key]
                fields = ('estimate', 'ci95_low', 'ci95_high')
                error = max(abs(row[field] - other[field]) for field in fields)
                check(error <= 1e-10 and row['n'] == other['n'] == 206, f'{name}/table413 mismatch {key}')
                cross_checks.append(dict(table=name, matches='table413', configuration=key[0], k=key[1], metric=key[2],
                                         fields=list(fields), max_absolute_error=error, n=row['n']))
    return rows, paired, subgroup_deltas, cross_checks


def label_counts(gold, predictions):
    return np.stack((gold & predictions, ~gold & predictions, gold & ~predictions), axis=-1).astype(int)


def summarize_targets(out, runs, cohort, indices):
    all_idx = np.arange(len(cohort))
    gold = np.array([[label in split_problems(req['gold_target_labels']) for label in CHEXPERT_LABELS] for req in cohort])
    gold_normal = np.array([req['gold_normal'] for req in cohort])
    aggregate, per_study, per_label, paired, original_checks = [], [], [], [], []
    for run_id, (predictions, original) in runs.items():
        estimates, samples = {}, {}
        for branch, labels in predictions.items():
            predicted = np.array([[label in row for label in CHEXPERT_LABELS] for row in labels])
            conflicts = label_counts(gold, predicted)
            normal = label_counts(gold_normal, ~predicted.any(axis=1))
            pooled = conflicts.sum(axis=1)
            estimates[branch] = label_metrics(pooled, all_idx)
            samples[branch] = label_metrics(pooled, indices)
            for summary, idx in [(estimates, all_idx), (samples, indices)]:
                normal_metrics = label_metrics(normal, idx)
                for metric in ('precision', 'recall', 'micro_f1'):
                    summary[branch]['normal_' + ('f1' if metric == 'micro_f1' else metric)] = normal_metrics[metric]
            for metric, estimate in estimates[branch].items():
                if branch in original and metric in original[branch]:
                    err = abs(float(original[branch][metric]) - estimate)
                    check(err < 1e-12, f'Archived target metric changed for {run_id}/{branch}/{metric}')
                    original_checks.append(dict(run_id=run_id, branch=branch, metric=metric, abs_error=float(err)))
                tp, fp, fn = normal.sum(axis=0) if metric.startswith('normal_') else pooled.sum(axis=0)
                aggregate.append(dict(run_id=run_id, pipeline=branch, metric=metric, tp=int(tp), fp=int(fp), fn=int(fn),
                                      **interval(estimate, samples[branch][metric], len(cohort))))
            for j, label in enumerate(CHEXPERT_LABELS):
                values = conflicts[:, j]
                tp, fp, fn = values.sum(axis=0)
                points, draws = label_metrics(values, all_idx), label_metrics(values, indices)
                for metric in ('precision', 'recall', 'micro_f1'):
                    per_label.append(dict(run_id=run_id, pipeline=branch, pathology=label, metric=metric,
                                          tp=int(tp), fp=int(fp), fn=int(fn), **interval(points[metric], draws[metric], len(cohort))))
            for i, req in enumerate(cohort):
                tp, fp, fn = pooled[i]
                per_study.append(dict(run_id=run_id, pipeline=branch, study_id=req['study_id'],
                                      query_image_id=req['query_image_id'], gold_target_labels=req['gold_target_labels'],
                                      predicted_target_labels=';'.join(label for label in CHEXPERT_LABELS if label in labels[i]),
                                      gold_normal=bool(gold_normal[i]), predicted_normal=not labels[i], tp=int(tp), fp=int(fp),
                                      fn=int(fn), sample_f1=float(2 * tp / (2 * tp + fp + fn)) if tp + fp + fn else 1.0))
        pairs = [('baseline_a', 'C0'), ('structured_learned_gate', 'baseline_a'), ('baseline_b', 'baseline_a'),
                 ('structured_rule', 'baseline_a'), ('structured_learned_gate', 'structured_rule'),
                 ('structured_learned_gate', 'C0')]
        for left, right in pairs:
            for metric in estimates[left]:
                paired.append(dict(run_id=run_id, left=left, right=right, metric=metric,
                                   **interval(estimates[left][metric] - estimates[right][metric],
                                              samples[left][metric] - samples[right][metric], len(cohort), 'fraction difference')))
    write_csv(out / 'target_label_per_study.csv', per_study)
    write_table(out, 'target_label_controls', 'Đối chứng sáu nhãn và quyết định Normal', aggregate,
                ('run_id', 'pipeline', 'metric', 'tp', 'fp', 'fn'))
    write_csv(out / 'target_label_per_label.csv', per_label)
    write_csv(out / 'target_label_paired_ci.csv', paired)
    return aggregate, paired, original_checks, gold


def gate_ablation(out, cohort, indices, audit):
    dataset_path, folds_path = GATE / 'gate_dataset.csv', GATE / 'folds.json'
    for path, role in [(dataset_path, 'Frozen gate dataset'), (folds_path, 'Frozen study-level OOF folds')]:
        audit.add(path, role, GATE.name, 206, 206, metric='Fixed-threshold study-level OOF', labels='Problems')
    if not dataset_path.exists() or not folds_path.exists():
        missing = [str(path) for path in (dataset_path, folds_path) if not path.is_file()]
        write_json(out / 'gate_ablation_missing.json', {'missing': missing, 'required': '1,236 study–label feature rows and five OOF folds'})
        return [], []

    import pandas as pd
    from run_nested_gate_threshold_oof import NUMERIC_FEATURES, build_model, make_feature_matrix

    frame = pd.read_csv(dataset_path, dtype={'study_id': str}).set_index(['study_id', 'pathology'])
    ids = [row['study_id'] for row in cohort]
    key = pd.MultiIndex.from_product([ids, CHEXPERT_LABELS], names=frame.index.names)
    check(len(frame) == 1236 and frame.index.is_unique and set(frame.index) == set(key), 'Gate does not cover all 1236 study–label pairs')
    frame = frame.loc[key].reset_index()
    fold_info = read_json(folds_path)
    fold_map = fold_info['study_to_fold']
    check(set(fold_map) == set(ids) and set(fold_map.values()) == set(range(5)), 'Unexpected gate OOF folds')
    fold = frame['study_id'].map(fold_map).to_numpy()
    x = make_feature_matrix(frame)
    y = frame['gold'].to_numpy().reshape(-1, 6)
    check(np.isfinite(x).all() and np.isin(y, [0, 1]).all() and
          y.tolist() == [[int(label in split_problems(row['gold_target_labels'])) for label in CHEXPERT_LABELS] for row in cohort],
          'Gate features or target-label provenance mismatch')
    oof, metrics, draws = [], [], {}
    all_idx = np.arange(len(cohort))
    for name, features in [('probability_and_onehot', x[:, [0, *range(len(NUMERIC_FEATURES), x.shape[1])]]),
                           ('full_evidence', x)]:
        probability = np.full(len(frame), np.nan)
        for f in range(5):
            test = fold == f
            model = build_model(fold_info['seed']).fit(features[~test], y.ravel()[~test])
            probability[test] = model.predict_proba(features[test])[:, 1]
        check(np.isfinite(probability).all(), f'OOF gap for {name}')
        decisions = probability >= .5
        for row, f, p, d in zip(frame.to_dict('records'), fold, probability, decisions):
            oof.append(dict(model=name, study_id=row['study_id'], request_id=row['request_id'], pathology=row['pathology'],
                            fold=int(f), gold=int(row['gold']), probability=float(p), decision=int(d),
                            threshold=.5, raw_prob=row['raw_prob'], m5=row['m5'], support_mean=row['support_mean'],
                            non_support_mean=row['non_support_mean'], support_count=row['support_count'],
                            non_support_count=row['non_support_count']))
        counts = label_counts(y.astype(bool), decisions.reshape(-1, 6)).sum(axis=1)
        point = label_metrics(counts, all_idx)
        draws[name] = label_metrics(counts, indices)
        tp, fp, fn = (int(n) for n in counts.sum(axis=0))
        for metric, est in point.items():
            metrics.append(dict(model=name, metric=metric, tp=tp, fp=fp, fn=fn,
                                **interval(est, draws[name][metric], len(cohort))))
    points = {(r['model'], r['metric']): r['estimate'] for r in metrics}
    differences = [dict(left='full_evidence', right='probability_and_onehot', metric=m,
                        **interval(points['full_evidence', m] - points['probability_and_onehot', m],
                                   draws['full_evidence'][m] - draws['probability_and_onehot'][m], 206, 'fraction difference'))
                   for m in draws['full_evidence']]
    write_csv(out / 'gate_ablation_oof.csv', oof)
    write_table(out, 'gate_ablation_metrics', 'Đối chứng logistic regression OOF trước LLM', metrics, ('model', 'metric', 'tp', 'fp', 'fn'))
    write_csv(out / 'gate_ablation_paired_ci.csv', differences)
    return metrics, differences


def execution_protocol(audit, out, cache_config, study_ids):
    metric_file, runner_file = Path(__file__).with_name('unified_206_metrics.py'), Path(__file__)
    for path in (metric_file, runner_file, ROOT / 'scripts/evaluate_llm_outputs.py', ROOT / 'src/retriever/pathology_support.py',
                 ROOT / 'src/retriever/densenet_local_reranker.py', ROOT / 'scripts/predict_chexpert_6labels.py',
                 ROOT / 'scripts/run_nested_gate_threshold_oof.py', Path(__file__).with_name('unified_206_thesis.py')):
        audit.add(path, 'Offline code source')
    return dict(run_id=out.name, manifest_sha256=sha(MANIFEST), study_order=study_ids, study_count=206,
                study_order_sha256=str_sha('\n'.join(study_ids)), metadata_sha256=sha(META),
                metric_code_sha256=sha(metric_file), runner_sha256=sha(runner_file), gallery_images=3818,
                gallery_studies=3689, exclude='Remove every image of the query study',
                dedup='Within IU-Xray retain highest global float32 cosine per study, then ascending image UUID for equal scores',
                normalization='Validate saved L2, renormalize each float32 row, dot product, exact (no ANN)',
                k=KS, binary_relevance_threshold=THRESHOLD, top_n=cache_config['cache_candidates'], lambda_values=LAMBDAS,
                main_lambda=.9, ranking='λ × S_global + (1 − λ) × S_local, within the same adapted Top-50. Equal scores break by ascending image UUID.',
                local_scores=f'Reused by exact image pair. Missing pairs rescored on CUDA. Cached peer score tolerance {LOCAL_PARITY_ATOL:g}; require identical Top-10 for all λ on the repaired query versus rescoring its full pool.',
                relevance='MeSH six-dimensional support. Equal per-label weights. Sum min / sum max. Scale saved weights by 100 exactly before summing so exact r=0.5 stays binary relevant. Zero-support queries match zero-support studies.',
                denominators='R counts all eligible deduplicated studies at r>=0.5. IDCG takes the highest graded relevances in that same gallery.',
                formulas={'precision': 'sum(hits)/K', 'map': 'sum(P@j for hits in first K)/min(R,K), then average by study',
                          'ndcg': 'sum(r_j / log2(j+1))/IDCG@K, linear gain', 'weighted_jaccard': 'mean ranked graded relevance',
                          'delta_path_global': 'Mean cosine for r>=0.5 minus mean cosine for r=0, all 3,688 candidates',
                          'delta_path_topk': 'Same score and definition within retrieved Top-K; N/A if either group absent'},
                zero_division='AP=0 when R=0. nDCG=0 when IDCG=0. Keep every study. A positive IDCG remains valid when R=0.',
                bootstrap=dict(seed=SEED, count=DRAWS, interval='2.5/97.5 percentile, study resampling, unadjusted exploratory CIs',
                               groups='Same 206-index matrix across analyses. Fixed group-size secondary matrices shared across subgroup analyses.'),
                targets='Canonical six Problems labels. P/R/Micro F1 denominators zero resolve to 0. Sample F1 both empty=1. Normal is separate.',
                gate='Frozen dataset and 5 study folds. StandardScaler + balanced liblinear LogisticRegression(C=1, max_iter=2000, tol=1e-4). threshold=0.5; probabilities and one-hot label in both.',
                devices='CPU for metrics/bootstrap/gate; CUDA for DenseNet local cache gaps', api_calls=0)


def version(name):
    try:
        return packages.version(name)
    except packages.PackageNotFoundError:
        return 'unavailable'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'outputs/unified_206_retrieval_controls_20261125')
    parser.add_argument('--resume', action='store_true', help='Resume exact output directory with the same input and code hashes.')
    parser.add_argument('--self-check', action='store_true', help='Run synthetic metric tests and exit.')
    args = parser.parse_args()
    self_check()
    if args.self_check:
        print('Shared metric self-check passed.')
        return
    started = datetime.now(TZ)
    out = args.output_dir.resolve()
    if out.exists() and not args.resume:
        out = out / ('run_' + started.strftime('%Y%m%dT%H%M%S%f%z'))
    out.mkdir(parents=True, exist_ok=args.resume)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        handlers=[logging.StreamHandler(), logging.FileHandler(out / 'run.log', encoding='utf-8')])
    log_format = logging.Formatter('%(asctime)s ICT %(levelname)s %(message)s')
    log_format.converter = lambda seconds: datetime.fromtimestamp(seconds, TZ).timetuple()
    for handler in logging.root.handlers:
        handler.setFormatter(log_format)
    LOG.info('Output: %s. Starting source audit.', out)
    audit = Audit()
    meta, ids, studies, support = load_gallery(audit)
    vectors, embedding_details = load_vectors(audit, meta, ids)
    cohort, refs, ontology = load_cohort(audit, meta, ids, support)
    image_to_row = {row['id']: row for row in meta}
    study_ids = [req['study_id'] for req in cohort]
    raw_cache, _ = load_local_cache(audit, FLOW / 'raw_rad_dino_densenet/retrieval_similarity_cache.json', cohort, meta)
    adapted_cache, cache_config = load_local_cache(audit, FLOW / 'retrieval_similarity_cache.json', cohort, meta)
    high, original_high, prov_high, high_input = load_run(audit, HIGH / 'metrics', cohort, refs, ontology)
    low, original_low, prov_low, low_input = load_run(audit, LATENCY, cohort, refs, ontology)
    high_rule = HIGH / 'structured_rule'
    check(read_json(high_rule / 'run_config.json')['lambda_global'] == .9
          and read_json(LATENCY_RULE / 'run_config.json')['lambda_global'] == .9, 'Evidence source not λ=0.9')
    high_c, high_controls, high_status = load_controls(audit, cohort, image_to_row, high_rule, HIGH.name)
    low_c, low_controls, low_status = load_controls(audit, cohort, image_to_row, LATENCY_RULE, LATENCY.name)
    check(len(high_status) == len(low_status) == 525, 'Expected 525 selected rule label pairs')
    history = [FLOW / 'retrieval_cache_build.json', FLOW / 'lambda_sweep_results.json',
               ROOT / 'outputs/table_4_11_4_12_rerun/table_4_11_summary.json',
               ROOT / 'outputs/table_4_13_rerun/lambda_sweep_k5_k10_results.json']
    for path in history:
        audit.add(path, 'Historical audit only', path.parent.name)
    for filename, role in [(THESIS, 'Exact thesis revision requested'), (EARLIER_THESIS, 'Available earlier thesis for captions'),
                           (HIGH / 'api_usage_totals.json', 'Archived high-effort API token totals'),
                           (GATE / 'gate_dataset.csv', 'Frozen gate dataset'), (GATE / 'folds.json', 'Frozen study-level OOF folds')]:
        audit.add(filename, role)
    protocol = execution_protocol(audit, out, cache_config, study_ids)
    protocol['support_sha256'] = str_sha(json.dumps([(row['id'], row['support']) for row in meta], separators=(',', ':')))
    initial = {path: record['sha256'] for path, record in audit.files.items()}
    input_state = out / 'resume_inputs.json'
    if args.resume:
        check(input_state.is_file() and read_json(input_state) == initial, 'Resume inputs differ, start a new output run')
        final = out / 'output_checksums.json'
        if final.is_file():
            check(all(sha(out / name) == value for name, value in read_json(final).items()), 'Saved output checksum mismatch')
            for row in read_csv(out / 'provenance.csv'):
                path = Path(row['path'])
                check((sha(path) if path.is_file() else '') == row['sha256'], f'Source checksum mismatch: {path}')
            LOG.info('Completed run verified by SHA-256.')
            return
    write_json(input_state, initial)
    write_json(out / 'protocol.json', protocol)
    notes = dict(cohort=dict(studies=len(cohort), recruitment_groups=dict(Counter(x['recruitment_group'] for x in cohort)),
                            zero_support=sum(x['zero_support'] for x in cohort), gold_normal=sum(x['gold_normal'] for x in cohort),
                            target_positive=sum(x['target_positive'] for x in cohort),
                            normal_vs_zero_support_disagree=[x['study_id'] for x in cohort if x['manifest_normal'] != x['zero_support']],
                            normal_vs_gold_disagree=[x['study_id'] for x in cohort if x['manifest_normal'] != x['gold_normal']]),
                 gallery=dict(images=len(meta), studies=len(set(studies)), support_matches_mesh=len(meta),
                              nonzero_support_image_rows=int(support.any(axis=1).sum()),
                              study_support_consistency=True, metadata_sha256=sha(META)), embedding=embedding_details,
                 historical=dict(full_gallery_1246='3818 image rows; 1247 image queries with support; 1246 reported after skipping R=0. The 512-D RAD-DINO variant in pathology_512 is not the CheXpert-adapted model.',
                                 adapted_hybrid=read_json(history[0]), old_sweep=read_json(history[1])['protocol'],
                                 historical_source_caption='Earlier thesis combines 1246-query raw and DenseNet with a 903-query adapted benchmark.',
                                 legacy_903='206 cache-build summary records 904 image rows with support. Original 903 cohort manifest and ranking were not located.',
                                 old_topk_rerun=read_json(history[3])['protocol']),
                 thesis=dict(expected=str(THESIS), exact_present=THESIS.is_file(), caption_reference=str(THESIS if THESIS.exists() else EARLIER_THESIS)),
                 report_sources={HIGH.name: dict(selection='Matches the handoff A and Learned points; all 824 summaries validated',
                         historical_api_usage=read_json(HIGH / 'api_usage_totals.json'), configs=high_input['branches']),
                                 LATENCY.name: dict(selection='Named source audited and recomputed separately; points differ and source reports/settings are different', configs=low_input['branches'])},
                 api_calls_this_run=0)
    write_json(out / 'input_audit.json', notes)
    write_csv(out / 'provenance.csv', list(audit.files.values()))
    LOG.info('Source audit passed. %s reports, %s patients/studies, %s gallery images.', len(prov_high)+len(prov_low), len(cohort), len(meta))

    order, scores = get_global_orders(vectors, ids, studies, cohort)
    local, repairs = compute_local(out, audit, adapted_cache, meta, ids, order['adapted'][:, :50], scores['adapted'], cohort)
    arrays, configurations, group, per_query, cache_diff, equality_n = calculate_retrieval(
        out, cohort, ids, support, order, scores, local, adapted_cache, raw_cache)
    groups, indices = bootstrap(out, group)
    retrieval, retrieval_delta, normal_contrasts, overlaps = summarize_retrieval(out, arrays, configurations, groups, indices)
    LOG.info('Retrieval completed. %s shared table cells match. Building target-label and gate controls.', len(overlaps))
    targets, target_delta, report_checks, _ = summarize_targets(out, {
        HIGH.name: ({**high, **high_c}, original_high), LATENCY.name: ({**low, **low_c}, original_low),
    }, cohort, indices['all'])
    metrics_gate, delta_gate = gate_ablation(out, cohort, indices['all'], audit)
    write_csv(out / 'cohort.csv', [{k: v for k, v in row.items() if k != 'gallery_index'} for row in cohort])
    write_csv(out / 'normal_definition_crosswalk.csv', [dict(manifest_normal=k[0], zero_support=k[1], gold_normal=k[2],
            target_positive=k[3], studies=v) for k, v in Counter((r['manifest_normal'], r['zero_support'], r['gold_normal'], r['target_positive'])
                                                               for r in cohort).items()])
    write_csv(out / 'gallery_images.csv', [dict(row_index=i, image_id=row['id'], study_id=row['study_id'], dataset=row['dataset'],
                                               support=json.dumps(row['support'])) for i, row in enumerate(meta)])
    write_csv(out / 'target_report_provenance.csv', prov_high + prov_low)
    write_csv(out / 'classifier_top5_controls.csv', high_controls + low_controls)
    write_csv(out / 'rule_evidence_status_per_label.csv', high_status + low_status)
    status_counts = Counter((r['run_id'], r['pathology'], r['evidence_status']) for r in high_status + low_status)
    status_counts.update((r['run_id'], 'ALL', r['evidence_status']) for r in high_status + low_status)
    write_csv(out / 'rule_evidence_status_counts.csv', [dict(run_id=run, pathology=label, status=status, count=n)
                                                      for (run, label, status), n in sorted(status_counts.items())])
    text_mask = []
    for row in cohort:
        sid = row['study_id']
        findings, impression = (bool(refs[sid][section].strip()) for section in ('findings', 'impression'))
        text_mask.append(dict(study_id=sid, in_clinical=bool(findings or impression), in_findings=findings, in_impression=impression,
                              reason='' if findings or impression else 'Both reference Findings and Impression blank',
                              findings_exclusion='' if findings else 'Reference Findings blank',
                              impression_exclusion='' if impression else 'Reference Impression blank'))
    check([r['study_id'] for r in text_mask if not r['in_clinical']] == ['1615'], 'Unexpected common clinical mask')
    write_csv(out / 'clinical_reference_mask.csv', text_mask)
    for path, record in audit.files.items():
        check((sha(Path(path)) if Path(path).is_file() else '') == record['sha256'], f'Source changed during run: {path}')
    notes.update(cache_repairs=repairs, cache_changed_pool=[r for r in cache_diff if not r['ordered_top50_matches']],
                 historical_relevant_counts_changed={encoder: sum(r['old_r'] != r['r'] and r['encoder'] == encoder for r in cache_diff)
                                                     for encoder in ('raw', 'adapted')},
                 historical_adapted_k5=[row for row in read_json(history[1])['summary'] if row['lambda_global'] in (1., .9)],
                 clinical_reference_counts={key: sum(row[key] for row in text_mask) for key in ('in_clinical', 'in_findings', 'in_impression')},
                 paper_comparators={run: {row['pipeline']: row['estimate'] for row in targets if row['metric'] == 'micro_f1' and row['run_id'] == run}
                                    for run in (HIGH.name, LATENCY.name)},
                 status_selected_label_pairs={HIGH.name: len(high_status), LATENCY.name: len(low_status)},
                 global_only_rule_evidence_studies=sorted({row['study_id'] for row in high_controls if not row['local_rerank_enabled']}),
                 thresholds=dict(current_file_sha256=high_controls[0]['current_threshold_file_sha256'],
                                 evidence_recorded_sha256={HIGH.name: high_controls[0]['threshold_source_sha256'],
                                                           LATENCY.name: low_controls[0]['threshold_source_sha256']},
                                 all_values_and_version_equal=True, all_predictions_equal_selected_labels=True),
                 source_files_checked=len(audit.files))
    write_json(out / 'input_audit.json', notes)
    write_csv(out / 'provenance.csv', list(audit.files.values()))

    r_zero = sorted({r['study_id'] for r in per_query if r['r_zero']})
    idcg_zero = sorted({r['study_id'] for r in per_query if r['idcg_zero']})
    checks = dict(passed=True, retrieval_records=len(per_query), independent_adapted_global_vs_sweep_study_k=equality_n,
                  shared_table_cells=overlaps, n_studies=206, n_normal_manifest=30, n_other=176, r_zero=r_zero, idcg_zero=idcg_zero,
                  cohort_sha256=sha(out / 'cohort.csv'), manifest_sha256=protocol['manifest_sha256'], support_sha256=protocol['support_sha256'],
                  k=protocol['k'], threshold=THRESHOLD, normalization=protocol['normalization'],
                  metric_code_sha256=protocol['metric_code_sha256'], bootstrap_shape=list(indices['all'].shape),
                  bootstrap_indices_sha256=sha(out / 'bootstrap_indices.npy'), weighted_subgroup_means_match=True,
                  report_parity=report_checks, historical_control_predictions_match=high_c == low_c,
                  cache_local_complete=bool(np.isfinite(local).all()), source_hashes_unchanged=True,
                  gate_predictions=len(read_csv(out / 'gate_ablation_oof.csv')) if metrics_gate else 0)
    write_json(out / 'consistency_checks.json', checks)
    from unified_206_thesis import write_thesis_reports
    write_thesis_reports(out, notes, checks, retrieval, retrieval_delta, normal_contrasts, targets, target_delta, metrics_gate, delta_gate)
    elapsed = (datetime.now(TZ) - started).total_seconds()
    gpu = ', '.join(sorted({r['gpu'] for r in repairs})) or 'Không cần suy luận'
    commands = [subprocess.list2cmdline([sys.executable, *sys.argv])]
    dep_versions = ', '.join(f'{name} {version(name)}' for name in ('numpy', 'pandas', 'scikit-learn', 'scipy', 'torch', 'torchvision', 'Pillow'))
    (out / 'RUN_COMMANDS.md').write_text(
        '# Lệnh tái lập\n\n```powershell\n' + '\n'.join(commands) + '\n```\n\n'
        f'- Thư mục làm việc `{Path.cwd()}`.\n- Python {platform.python_version()} từ `{sys.executable}`.\n'
        f'- Bắt đầu {started.isoformat(timespec="seconds")}. Tổng thời gian {elapsed:.2f} giây.\n'
        f'- Thiết bị CPU cho thống kê và gate, CUDA cho bổ sung cache ({gpu}).\n- Phiên bản {dep_versions}.\n'
        '- API calls 0. Sử dụng báo cáo đã lưu, có SHA-256, response ID và token usage trong `target_report_provenance.csv`.\n'
        '- Dùng `--resume --output-dir <đường dẫn run>` để kiểm tra hoặc tiếp tục cùng bộ hash. Bỏ `--resume` để tạo thư mục mới.\n', encoding='utf-8')
    required = ('table411.csv', 'table412.csv', 'table413.csv', 'retrieval_per_study.csv', 'retrieval_paired_ci.csv',
                'target_label_per_study.csv', 'target_label_paired_ci.csv', 'RESULTS_FOR_THESIS.md', 'THESIS_UPDATE_MAP.md')
    check(all((out / path).stat().st_size > 0 for path in required), 'Required artifact missing or empty')
    LOG.info('Completed in %.2fs. Source and output provenance checked.', elapsed)
    write_json(out / 'output_checksums.json', {path.name: sha(path) for path in sorted(out.iterdir())
                                               if path.is_file() and path.name not in ('run.log', 'output_checksums.json')})


if __name__ == '__main__':
    main()
