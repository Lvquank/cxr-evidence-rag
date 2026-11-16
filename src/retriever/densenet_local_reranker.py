"""DenseNet feature-map reranking for a small set of retrieval candidates."""

from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


CLASSIFIER_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((320, 320)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


class DenseNetLocalFeatureReranker:
    """Compare DenseNet spatial feature maps with symmetric patch matching.

    The model is only applied after ANN retrieval, so it evaluates a bounded
    candidate set rather than replacing the global RAD-DINO index.
    """

    def __init__(self, densenet_backbone, device: torch.device, batch_size: int = 16):
        self.features = densenet_backbone.features.to(device).eval()
        self.device = device
        self.batch_size = batch_size
        self._feature_cache: dict[str, torch.Tensor] = {}

    @staticmethod
    def _load_image(path: str) -> torch.Tensor:
        with Image.open(path) as image:
            return CLASSIFIER_TRANSFORM(image.convert("RGB"))

    def _encode_paths(self, paths: list[str]) -> None:
        missing = [path for path in dict.fromkeys(paths) if path not in self._feature_cache]
        for start in range(0, len(missing), self.batch_size):
            batch_paths = missing[start:start + self.batch_size]
            valid_paths = []
            tensors = []
            for path in batch_paths:
                try:
                    tensors.append(self._load_image(path))
                    valid_paths.append(path)
                except (FileNotFoundError, OSError):
                    continue
            if not tensors:
                continue
            with torch.inference_mode():
                feature_map = F.relu(self.features(torch.stack(tensors).to(self.device)), inplace=False)
                tokens = F.normalize(feature_map.flatten(2).transpose(1, 2), p=2, dim=-1).cpu()
            for path, token_map in zip(valid_paths, tokens):
                self._feature_cache[path] = token_map

    def score_paths(self, query_path: str, candidate_paths: list[str]) -> dict[str, float]:
        paths = [query_path, *candidate_paths]
        self._encode_paths(paths)
        if query_path not in self._feature_cache:
            return {}
        valid_paths = [path for path in dict.fromkeys(candidate_paths) if path in self._feature_cache]
        if not valid_paths:
            return {}

        query = self._feature_cache[query_path].to(self.device)
        candidates = torch.stack([self._feature_cache[path] for path in valid_paths]).to(self.device)
        with torch.inference_mode():
            similarities = torch.einsum("pc,nqc->npq", query, candidates)
            query_to_candidate = similarities.max(dim=2).values.mean(dim=1)
            candidate_to_query = similarities.max(dim=1).values.mean(dim=1)
            scores = 0.5 * (query_to_candidate + candidate_to_query)
        return {path: float(score) for path, score in zip(valid_paths, scores.cpu())}

    def clear_cache(self) -> None:
        """Release cached maps when evaluating many independent queries."""
        self._feature_cache.clear()

    def rerank_candidates(
        self,
        query_path: str,
        candidates: list[dict],
        base_score_key: str,
        candidate_limit: int,
        lambda_global: float,
    ) -> list[dict]:
        if not candidates or candidate_limit < 1:
            return candidates
        selected = candidates[:candidate_limit]
        paths = [str(candidate.get("processed_image_path", "")) for candidate in selected]
        local_scores = self.score_paths(query_path, paths)
        reranked = []
        for candidate in selected:
            item = dict(candidate)
            path = str(item.get("processed_image_path", ""))
            base_score = float(item[base_score_key])
            local_score = local_scores.get(path)
            if local_score is None:
                item["densenet_local_rerank_applied"] = False
                item["final_rank_score"] = base_score
            else:
                item["densenet_local_rerank_applied"] = True
                item["densenet_local_similarity"] = local_score
                item["final_rank_score"] = (
                    lambda_global * base_score + (1.0 - lambda_global) * local_score
                )
            reranked.append(item)
        reranked.sort(key=lambda item: item["final_rank_score"], reverse=True)
        return reranked
