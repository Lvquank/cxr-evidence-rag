"""Shared retrieval and study-bootstrap metrics for the frozen 206-study audit."""

import numpy as np


THRESHOLD = 0.5
KS = (3, 5, 10)
RANK_METRICS = ("weighted_jaccard", "precision", "map", "ndcg")


def relevance(query, gallery):
    """Calculate Jaccard using integer support ticks so exact half-overlap stays 0.5."""
    if not query.any():
        return (~gallery.any(axis=1)).astype(float)
    return np.minimum(query, gallery).sum(axis=1) / np.maximum(query, gallery).sum(axis=1)


def representatives(scores, image_ids, studies, query_study):
    order = np.lexsort((image_ids, -scores))
    order = order[studies[order] != query_study]
    _, first = np.unique(studies[order], return_index=True)
    return order[np.sort(first)]


def dcg(graded):
    return float(graded @ (1 / np.log2(np.arange(2, len(graded) + 2))))


def ideal(eligible):
    best = np.sort(eligible)[::-1][:max(KS)]
    return int(np.count_nonzero(eligible >= THRESHOLD)), {k: dcg(best[:k]) for k in KS}


def rank_metrics(ranked, k, relevant_studies, idcg):
    graded = ranked[:k]
    if len(graded) != k:
        raise ValueError(f"Expected at least {k} ranked studies")
    hits = graded >= THRESHOLD
    precision = hits.cumsum() / np.arange(1, k + 1)
    return {
        "weighted_jaccard": float(graded.mean()),
        "precision": float(precision[-1]),
        "map": float(precision[hits].sum() / min(relevant_studies, k)) if relevant_studies else 0.0,
        "ndcg": dcg(graded) / idcg if idcg else 0.0,
    }


def delta_path(graded, global_scores):
    positive = global_scores[graded >= THRESHOLD]
    zero = global_scores[graded == 0]
    if not len(positive) or not len(zero):
        return None
    return float(positive.mean(dtype=float) - zero.mean(dtype=float))


def fused_order(global_scores, local_scores, image_ids, weight):
    if not np.isfinite(local_scores).all():
        raise ValueError("Missing local score in frozen candidate pool")
    scores = weight * global_scores + (1 - weight) * local_scores
    return np.lexsort((image_ids, -scores)), scores


def divide(numerator, denominator, empty=0.0):
    return np.divide(numerator, denominator, out=np.full_like(denominator, empty, dtype=float), where=denominator != 0)


def label_metrics(counts, indices):
    """Counts are [study, (TP, FP, FN)]. Indices may be one or many resamples."""
    sampled = counts[indices]
    tp, fp, fn = np.moveaxis(sampled, -1, 0)
    sample_f1 = divide(2 * tp, 2 * tp + fp + fn, empty=1).mean(axis=-1)
    tp, fp, fn = (column.sum(axis=-1) for column in (tp, fp, fn))
    return {
        "precision": divide(tp, tp + fp),
        "recall": divide(tp, tp + fn),
        "micro_f1": divide(2 * tp, 2 * tp + fp + fn),
        "sample_f1": sample_f1,
    }


def interval(estimate, samples, n, unit="fraction"):
    finite = np.asarray(samples)[np.isfinite(samples)]
    low, high = np.quantile(finite, [0.025, 0.975]) if len(finite) else (None, None)
    return {
        "estimate": float(estimate) if np.isfinite(estimate) else None,
        "ci95_low": float(low) if low is not None else None,
        "ci95_high": float(high) if high is not None else None,
        "n": int(n),
        "bootstrap_valid": len(finite),
        "unit": unit,
    }


def mean_samples(values, indices):
    sampled = values[indices]
    valid = np.isfinite(sampled)
    return divide(np.where(valid, sampled, 0).sum(axis=-1), valid.sum(axis=-1), empty=np.nan)


def self_check():
    gallery = np.array([[100, 0], [40, 0], [50, 50], [0, 0]])
    graded = relevance(gallery[0], gallery)
    np.testing.assert_allclose(graded, [1, .4, 1/3, 0])
    np.testing.assert_equal(relevance(gallery[-1], gallery), [0, 0, 0, 1])
    np.testing.assert_equal(relevance(np.array([100, 20, 100, 20]), np.array([[100, 20, 0, 0]])), [.5])
    full = np.array([1., 1., 1., .5, .25, 0.])
    r, best = ideal(full)
    values = rank_metrics(np.array([0., 1., .25]), 3, r, best[3])
    np.testing.assert_allclose([values['precision'], values['map'], values['ndcg']],
                               [1/3, 1/6, (1/np.log2(3) + .125) / (1 + 1/np.log2(3) + .5)])
    np.testing.assert_allclose(rank_metrics(np.array([.25, 0., 0.]), 3, 0, .25)['ndcg'], 1)
    assert rank_metrics(np.zeros(3), 3, 0, 0)['ndcg'] == 0
    assert rank_metrics(np.zeros(3), 3, 0, 0)['map'] == 0
    np.testing.assert_equal(representatives(np.array([1, 1, 1, 1, 0]), np.array(list('edcba')),
                                           np.array([1, 1, 2, 2, 3]), 1), [3, 4])
    try:
        fused_order(np.array([.8]), np.array([np.nan]), np.array(['a']), .9)
    except ValueError:
        pass
    else:
        raise AssertionError('Expected missing local score to fail')
    np.testing.assert_allclose(label_metrics(np.array([[0, 0, 0], [1, 1, 0]]), [0, 1])['sample_f1'], 5/6)
