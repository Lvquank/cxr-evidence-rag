"""Apply an opt-in classifier refinement to matching frozen image features."""

import numpy as np


def predict_refinement(bundle, raw_probabilities, image_features):
    """Return a boolean study-by-label matrix; scores keep their source threshold."""
    names = bundle["labels"]
    raw = np.asarray(raw_probabilities, dtype=float)
    if raw.ndim != 2 or raw.shape[1] != len(names) or not np.isfinite(raw).all() or ((raw < 0) | (raw > 1)).any():
        raise ValueError("Expected a finite study-by-label probability matrix in [0, 1]")
    logits = np.log(np.clip(raw,1e-6,1-1e-6)/np.clip(1-raw,1e-6,1-1e-6))
    baseline_thresholds = bundle["baseline_thresholds"]["labels"]
    prediction = raw >= np.array([baseline_thresholds[n]["threshold"] for n in names])
    details = {}
    for j,name in enumerate(names):
        entry = bundle["models"].get(name)
        if entry is None:
            details[name] = dict(method="baseline",threshold=baseline_thresholds[name]["threshold"],scores=raw[:,j].tolist())
            continue
        method = entry["method"]
        if method in {"threshold_tuned", "platt"}:
            x=logits[:,[j]]
        elif method in {"score_fusion", "score_boost"}:
            x=logits
        elif method == "hybrid_probe":
            x=np.column_stack([image_features["dense"],image_features["rad"],logits])
        else:
            x=image_features[{"densenet_probe":"dense", "rad_dino_probe":"rad",
                              "chexpert_rad_probe":"chexpert_rad"}[method]]
        if len(x) != len(raw) or not np.isfinite(x).all():
            raise ValueError("Image features do not match classifier inputs")
        model=entry["model"]
        values=x[:,0] if model is None else model.predict_proba(x)[:,1]
        threshold=entry["threshold"]
        refined=np.zeros(len(raw),dtype=bool) if threshold is None else values >= threshold
        if bundle.get("mode") == "add_only":
            prediction[:,j] |= refined
        else:
            prediction[:,j] = refined
        details[name]=dict(method=method,threshold=threshold,scores=values.tolist())
    return prediction,details


def refinement_scores(bundle, raw_probabilities, image_features):
    """Return selection scores/thresholds in [0, 1], retaining their classifier source."""
    prediction, details = predict_refinement(bundle, raw_probabilities, image_features)
    raw = np.asarray(raw_probabilities, dtype=float)
    scores = raw.copy()
    thresholds, sources = {}, {}
    for j, name in enumerate(bundle["labels"]):
        entry = details[name]
        if entry["method"] == "threshold_tuned":
            thresholds[name] = float(bundle["models"][name]["raw_threshold"])
            sources[name] = "densenet"
        else:
            scores[:, j] = entry["scores"]
            thresholds[name] = float(entry["threshold"])
            sources[name] = "densenet" if entry["method"] == "baseline" else entry["method"]
    if not np.array_equal(prediction, scores >= np.array([thresholds[n] for n in bundle["labels"]])):
        raise ValueError("Refined selection scores differ from the candidate decision.")
    return prediction, scores, thresholds, sources
