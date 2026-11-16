import json
from pathlib import Path
from typing import Collection, Optional
import numpy as np
import faiss
import torch
from PIL import Image

from src.embeddings.encoder import DEFAULT_MODEL_NAME, ImagePreprocessor, load_encoder
from src.retriever.pathology_support import CHEXPERT_LABELS, pathology_support_score


class FAISSImageRetriever:
    def __init__(
        self,
        index_path: str,
        metadata_path: str,
        processed_dir: str = None,
        embedding_dim: Optional[int] = None,
        device: str = 'cpu',
        model_name: str = DEFAULT_MODEL_NAME,
        projection_checkpoint: str | None = None,
        densenet_checkpoint: str | None = None,
        minimum_label_support: float = 0.0,
    ):
        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)
        self.processed_dir = Path(processed_dir) if processed_dir else None
        self.device = device
        if not 0.0 <= minimum_label_support <= 1.0:
            raise ValueError("minimum_label_support must be within [0, 1].")
        self.minimum_label_support = minimum_label_support
        self.model = load_encoder(
            device=device,
            model_name=model_name,
            projection_checkpoint=projection_checkpoint,
            densenet_checkpoint=densenet_checkpoint,
        )
        self.embedding_dim = embedding_dim or self.model.embedding_dimension
        self.preprocessor = ImagePreprocessor(model_name=self.model.model_name)
        self.manifest = self._load_and_validate_manifest()
        self.index = self._load_or_build_index()
        self.metadata = self._load_metadata()
        self._validate_collection()
        self._build_label_banks()

    def _load_or_build_index(self):
        if self.index_path.exists():
            return faiss.read_index(str(self.index_path))

        npy_path = self.index_path.with_name('embeddings.npy')
        if not npy_path.exists():
            raise FileNotFoundError(
                f"No FAISS index found at '{self.index_path}' and no embeddings.npy found at '{npy_path}'. "
                "Please run the embedding script first with --vector_db faiss."
            )

        embeddings = np.load(str(npy_path)).astype(np.float32)
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        embeddings = embeddings / norms

        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        faiss.write_index(index, str(self.index_path))
        return index

    def _load_metadata(self):
        if self.metadata_path.suffix.lower() == ".jsonl":
            with self.metadata_path.open("r", encoding="utf-8") as handle:
                return [json.loads(line) for line in handle if line.strip()]
        import csv
        rows = []
        with self.metadata_path.open('r', encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
        return rows

    def _load_and_validate_manifest(self):
        manifest_path = self.index_path.with_name("embedding_manifest.json")
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Embedding manifest was not found: {manifest_path}. "
                "Rebuild embeddings with scripts/compute_image_embeddings.py before searching."
            )
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)

        if manifest.get("embedding_dimension") != self.embedding_dim:
            raise ValueError("Embedding manifest dimension does not match the retriever. Rebuild embeddings.")
        if manifest.get("embedding_model") != self.model.model_name:
            raise ValueError("Embedding model differs from the retriever. Rebuild embeddings.")
        if manifest.get("preprocessing") and manifest["preprocessing"] != self.preprocessor.signature():
            raise ValueError("Gallery preprocessing differs from query preprocessing. Rebuild embeddings.")
        manifest_revision = manifest.get("model_revision")
        if manifest_revision and self.model.model_revision and manifest_revision != self.model.model_revision:
            raise ValueError("Embedding model revision differs from the retriever. Rebuild embeddings.")
        manifest_projection = manifest.get("projection")
        if bool(manifest_projection) != bool(self.model.projection_metadata):
            raise ValueError("Projection checkpoint differs from the retriever. Rebuild embeddings.")
        if manifest_projection:
            expected_projection = self.model.projection_metadata
            identity_fields = (
                "output_dim",
                "pipeline",
                "requires_image_encoder",
                "feature_fusion",
            )
            if any(
                manifest_projection.get(field) != expected_projection.get(field)
                for field in identity_fields
            ):
                raise ValueError("Projection checkpoint differs from the retriever. Rebuild embeddings.")

            manifest_hash = manifest_projection.get("sha256")
            retriever_hash = expected_projection.get("sha256")
            if manifest_hash and retriever_hash:
                if manifest_hash != retriever_hash:
                    raise ValueError("Projection checkpoint differs from the retriever. Rebuild embeddings.")
            else:
                # Older manifests stored only an absolute build-machine path. A matching
                # filename plus projection semantics supports moving the exact artifact.
                manifest_name = manifest_projection.get("file_name") or Path(
                    manifest_projection.get("path", "")
                ).name
                retriever_name = expected_projection.get("file_name") or Path(
                    expected_projection.get("path", "")
                ).name
                if not manifest_name or manifest_name != retriever_name:
                    raise ValueError("Projection checkpoint differs from the retriever. Rebuild embeddings.")
        return manifest

    def _validate_collection(self):
        if self.index.d != self.embedding_dim:
            raise ValueError(
                f"FAISS index dimension is {self.index.d}, expected {self.embedding_dim}. Rebuild embeddings."
            )
        if self.index.ntotal != len(self.metadata):
            raise ValueError(
                "FAISS index and metadata row counts differ. Rebuild embeddings and the index together."
            )

    def _build_label_banks(self):
        """Create logical FAISS selectors without duplicating the embedding matrix."""
        supports = [self._support_vector_from_row(row) for row in self.metadata]
        self.label_banks = {}
        for label in CHEXPERT_LABELS:
            positive = [index for index, support in enumerate(supports)
                        if support[label] > 0.0 and support[label] >= self.minimum_label_support]
            non_support = [index for index, support in enumerate(supports) if support[label] == 0.0]
            self.label_banks[label] = {
                "positive": np.asarray(positive, dtype=np.int64),
                "non_support": np.asarray(non_support, dtype=np.int64),
            }

    def _search_indices(self, embedding: np.ndarray, top_k: int, allowed_indices=None):
        if allowed_indices is not None and len(allowed_indices) == 0:
            return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.int64)
        search_k = min(self.index.ntotal if allowed_indices is None else len(allowed_indices), top_k)
        query = np.expand_dims(embedding, axis=0).astype("float32")
        if allowed_indices is None:
            distances, indices = self.index.search(query, search_k)
        else:
            params = faiss.SearchParameters()
            params.sel = faiss.IDSelectorBatch(np.asarray(allowed_indices, dtype=np.int64))
            distances, indices = self.index.search(query, search_k, params=params)
        valid = indices[0] >= 0
        return distances[0][valid], indices[0][valid]

    def _candidate_from_index(self, cosine_similarity: float, index: int):
        row = self.metadata[int(index)] if int(index) < len(self.metadata) else {}
        return {
            "index_id": int(index),
            "id": row.get("id", str(index)),
            "study_id": row.get("study_id", row.get("uid", "")),
            "image_path": row.get("path", row.get("image_path", "")),
            "processed_image_path": self._resolve_processed_image_path(row),
            "cosine_similarity": float(cosine_similarity),
            "support": self._support_vector_from_row(row),
            "metadata": row,
        }

    def _search_candidates(self, embedding: np.ndarray, top_k: int, allowed_indices=None):
        distances, indices = self._search_indices(embedding, top_k, allowed_indices)
        return [
            self._candidate_from_index(distance, index)
            for distance, index in zip(distances, indices)
        ]

    def _exclude_studies_from_indices(
        self, indices: np.ndarray, excluded_study_ids: Collection[str]
    ) -> np.ndarray:
        """Filter a logical FAISS bank without mutating the shared gallery/index."""
        if not excluded_study_ids:
            return indices
        excluded = {str(study_id).strip() for study_id in excluded_study_ids if str(study_id).strip()}
        return np.asarray(
            [
                int(index)
                for index in indices
                if str(self.metadata[int(index)].get("study_id", self.metadata[int(index)].get("uid", ""))).strip()
                not in excluded
            ],
            dtype=np.int64,
        )

    def _apply_crop(self, img: Image.Image, bbox=None):
        if bbox:
            if isinstance(bbox, str):
                parts = [float(x) for x in bbox.split(',')]
                if len(parts) == 4:
                    x1, y1, x2, y2 = parts
                    try:
                        return img.crop((x1, y1, x2, y2))
                    except Exception:
                        pass
            else:
                try:
                    x1, y1, x2, y2 = bbox
                    return img.crop((x1, y1, x2, y2))
                except Exception:
                    pass

        # Keep the complete field of view. The model processor preserves aspect
        # ratio and pads to a patch-compatible tensor without center cropping.
        return img

    def embed_image(self, image_path: str):
        img = Image.open(image_path).convert('RGB')
        img = self._apply_crop(img)
        tensor = self.preprocessor(img).unsqueeze(0)
        with torch.inference_mode():
            emb = self.model(tensor.to(self.device)).cpu().numpy()[0]
        emb = emb.astype('float32')
        emb = emb / (np.linalg.norm(emb) + 1e-8)
        return emb

    def _resolve_processed_image_path(self, row):
        raw_path = row.get('path', row.get('image_path', ''))
        if not raw_path:
            return ''

        candidate = Path(raw_path)
        if not candidate.is_absolute() and self.processed_dir is not None:
            candidate = self.processed_dir / candidate
        return str(candidate.resolve())

    @staticmethod
    def _support_vector_from_row(row):
        support = row.get("support")
        if isinstance(support, list) and len(support) == len(CHEXPERT_LABELS):
            return dict(zip(CHEXPERT_LABELS, (float(value) for value in support)))
        raw_vector = row.get("support_vector", "")
        try:
            parsed = json.loads(raw_vector)
            if isinstance(parsed, dict):
                return {label: float(parsed.get(label, 0.0)) for label in CHEXPERT_LABELS}
        except (TypeError, ValueError, json.JSONDecodeError):
            pass
        return {label: float(row.get(f"support_{label.lower().replace(' ', '_')}", 0.0) or 0.0) for label in CHEXPERT_LABELS}

    def search(
        self,
        image_path: str,
        top_k: int = 10,
        query_support=None,
        alpha: float = 0.0,
        candidate_k: int = 200,
        required_label: str | None = None,
        exclude_study_ids: Collection[str] = (),
    ):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be between 0.0 and 1.0.")
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")
        if required_label is not None and required_label not in CHEXPERT_LABELS:
            raise ValueError(f"required_label must be one of {CHEXPERT_LABELS}.")

        q = self.embed_image(image_path)
        excluded = {str(study_id).strip() for study_id in exclude_study_ids if str(study_id).strip()}
        allowed_indices = None
        if excluded:
            allowed_indices = self._exclude_studies_from_indices(
                np.arange(self.index.ntotal, dtype=np.int64), excluded
            )
        # Search beyond Top-K because multiple images can belong to one study.
        # Deduplication happens after optional pathology-aware reranking.
        search_k = min(
            self.index.ntotal if allowed_indices is None else len(allowed_indices),
            max(top_k, candidate_k or 0),
        )
        distances, indices = self._search_indices(q, search_k, allowed_indices)
        candidates = []
        for dist, idx in zip(distances, indices):
            row = self.metadata[int(idx)] if int(idx) < len(self.metadata) else {}
            resolved_path = self._resolve_processed_image_path(row)
            candidate_support = self._support_vector_from_row(row)
            if required_label is not None and candidate_support.get(required_label, 0.0) <= 0.0:
                continue
            support_score = pathology_support_score(query_support, candidate_support) if query_support else None
            final_score = float(dist) if support_score is None else alpha * float(dist) + (1.0 - alpha) * support_score
            candidates.append({
                'distance': float(dist),
                'cosine_similarity': float(dist),
                'pathology_support': support_score,
                'final_score': final_score,
                'id': row.get('id', str(idx)),
                'image_path': row.get('path', row.get('image_path', '')),
                'processed_image_path': resolved_path,
                'report_text': row.get('report_text', ''),
                'projection': row.get('view', row.get('projection', '')),
                'uid': row.get('study_id', row.get('uid', '')),
                'study_id': row.get('study_id', row.get('uid', '')),
                'metadata': row,
            })
        if query_support:
            candidates.sort(key=lambda candidate: candidate['final_score'], reverse=True)

        results = []
        seen_studies = set()
        for candidate in candidates:
            study_id = str(candidate['study_id'] or '').strip()
            # Missing study IDs must not collapse unrelated images into one result.
            dedupe_key = f"study:{study_id}" if study_id else f"image:{candidate['id']}"
            if dedupe_key in seen_studies:
                continue
            seen_studies.add(dedupe_key)
            results.append(candidate)
            if len(results) == top_k:
                break
        for rank, result in enumerate(results, start=1):
            result['rank'] = rank
        return results

    def search_label_evidence(
        self,
        image_path: str,
        raw_probabilities: dict[str, float],
        labels: tuple[str, ...] | None = None,
        top_k: int = 3,
        common_top_k: int = 5,
        candidate_k: int = 200,
        bank_candidate_k: int = 50,
        statistics_k: int = 20,
        support_similarity_threshold: float = 0.10,
        quality_margin_threshold: float = 0.02,
        margin_10_reversal_ratio: float = 0.5,
        neighborhood_temperature: float = 0.07,
        local_reranker=None,
        local_rerank_k: int = 10,
        lambda_global: float = 0.9,
        exclude_study_ids: Collection[str] = (),
    ) -> dict:
        """Retrieve global, supporting and matched non-support evidence.

        The classifier probability is kept for label routing and output only. It
        is intentionally excluded from ranking references for a fixed query.
        """
        if top_k < 1 or common_top_k < 1:
            raise ValueError("top_k and common_top_k must be at least 1.")
        if candidate_k < max(top_k, common_top_k):
            raise ValueError("candidate_k must be at least top_k and common_top_k.")
        if bank_candidate_k < 1 or statistics_k < 20:
            raise ValueError("bank_candidate_k must be positive and statistics_k must be at least 20.")
        if not 0.0 <= support_similarity_threshold <= 1.0:
            raise ValueError("support_similarity_threshold must be between 0.0 and 1.0.")
        if margin_10_reversal_ratio < 0.0:
            raise ValueError("margin_10_reversal_ratio must be non-negative.")
        if local_rerank_k < 1:
            raise ValueError("local_rerank_k must be at least 1.")
        if not 0.0 <= lambda_global <= 1.0:
            raise ValueError("lambda_global must be between 0.0 and 1.0.")
        if quality_margin_threshold < 0.0 or neighborhood_temperature <= 0.0:
            raise ValueError("quality_margin_threshold must be non-negative and temperature positive.")
        missing_labels = set(CHEXPERT_LABELS).difference(raw_probabilities)
        if missing_labels:
            raise ValueError(f"raw_probabilities is missing labels: {sorted(missing_labels)}")

        query_embedding = self.embed_image(image_path)
        excluded = {str(study_id).strip() for study_id in exclude_study_ids if str(study_id).strip()}
        gallery_indices = None
        if excluded:
            gallery_indices = self._exclude_studies_from_indices(
                np.arange(self.index.ntotal, dtype=np.int64), excluded
            )
        global_candidates = self._search_candidates(query_embedding, candidate_k, gallery_indices)
        selected_labels = CHEXPERT_LABELS if labels is None else labels
        unknown_labels = set(selected_labels).difference(CHEXPERT_LABELS)
        if unknown_labels:
            raise ValueError(f"labels contains unsupported values: {sorted(unknown_labels)}")

        def unique_studies(candidates: list[dict], limit: int) -> list[dict]:
            results = []
            seen_studies = set()
            for candidate in candidates:
                study_id = str(candidate["study_id"] or "").strip()
                dedupe_key = f"study:{study_id}" if study_id else f"image:{candidate['id']}"
                if dedupe_key in seen_studies:
                    continue
                seen_studies.add(dedupe_key)
                results.append(dict(candidate))
                if len(results) == limit:
                    break
            for rank, candidate in enumerate(results, start=1):
                candidate["rank"] = rank
            return results

        def public_candidate(candidate: dict, label: str | None = None, score_name: str | None = None):
            result = {
                "id": candidate["id"],
                "study_id": candidate["study_id"],
                "image_path": candidate["image_path"],
                "processed_image_path": candidate["processed_image_path"],
                "cosine_similarity": candidate["cosine_similarity"],
                "metadata": candidate["metadata"],
            }
            if label is not None:
                result["label_support"] = float(candidate["support"][label])
            if score_name is not None:
                result[score_name] = float(candidate[score_name])
            for key in (
                "base_rank_score",
                "densenet_local_similarity",
                "final_rank_score",
                "densenet_local_rerank_applied",
            ):
                if key in candidate and key not in result:
                    result[key] = candidate[key]
            return result

        def similarity_statistics(candidates: list[dict]):
            values = np.asarray([candidate["cosine_similarity"] for candidate in candidates], dtype=np.float32)
            if not len(values):
                return {"count": 0, "mean": None, "median": None, "max": None, "std": None}
            return {
                "count": int(len(values)),
                "mean": float(values.mean()),
                "median": float(np.median(values)),
                "max": float(values.max()),
                "std": float(values.std()),
            }

        def margin_at_k(support_candidates: list[dict], non_support_candidates: list[dict], k: int):
            support = similarity_statistics(support_candidates[:k])
            non_support = similarity_statistics(non_support_candidates[:k])
            margin = (
                support["mean"] - non_support["mean"]
                if support["mean"] is not None and non_support["mean"] is not None
                else None
            )
            return {"support_mean": support["mean"], "non_support_mean": non_support["mean"], "margin": margin}

        def same_nonzero_sign(left: float | None, right: float | None) -> bool:
            return left is not None and right is not None and left * right > 0.0

        common_evidence = unique_studies(
            [
                public_candidate(candidate)
                for candidate in global_candidates
            ],
            common_top_k,
        )

        label_evidence = []
        for label in selected_labels:
            probability = float(raw_probabilities[label])
            positive_bank = self._search_candidates(
                query_embedding,
                bank_candidate_k,
                self._exclude_studies_from_indices(self.label_banks[label]["positive"], excluded),
            )
            non_support_bank = self._search_candidates(
                query_embedding,
                bank_candidate_k,
                self._exclude_studies_from_indices(self.label_banks[label]["non_support"], excluded),
            )
            union_by_index = {}
            for candidate in global_candidates + positive_bank + non_support_bank:
                union_by_index.setdefault(candidate["index_id"], candidate)
            union_candidates = list(union_by_index.values())

            supported_candidates = [
                candidate
                for candidate in union_candidates
                if float(candidate["support"][label]) > 0.0
                and float(candidate["support"][label]) >= self.minimum_label_support
                and candidate["cosine_similarity"] >= support_similarity_threshold
            ]
            non_support_candidates = [
                candidate
                for candidate in union_candidates
                if float(candidate["support"][label]) == 0.0
            ]
            supporting_reranked = []
            for candidate in supported_candidates:
                ranked = dict(candidate)
                ranked["support_rank_score"] = candidate["cosine_similarity"]
                ranked["base_rank_score"] = ranked["support_rank_score"]
                ranked["final_rank_score"] = ranked["base_rank_score"]
                ranked["densenet_local_rerank_applied"] = False
                supporting_reranked.append(ranked)
            supporting_reranked.sort(key=lambda item: item["support_rank_score"], reverse=True)
            if local_reranker is not None:
                supporting_reranked = local_reranker.rerank_candidates(
                    image_path,
                    supporting_reranked,
                    base_score_key="base_rank_score",
                    candidate_limit=local_rerank_k,
                    lambda_global=lambda_global,
                )
            supporting_evidence = unique_studies(
                [public_candidate(candidate, label, "final_rank_score") for candidate in supporting_reranked], top_k
            )

            non_support_candidates.sort(key=lambda item: item["cosine_similarity"], reverse=True)
            non_support_reranked = []
            for candidate in non_support_candidates:
                ranked = dict(candidate)
                ranked["non_support_rank_score"] = ranked["cosine_similarity"]
                ranked["base_rank_score"] = ranked["non_support_rank_score"]
                ranked["final_rank_score"] = ranked["base_rank_score"]
                ranked["densenet_local_rerank_applied"] = False
                non_support_reranked.append(ranked)
            if local_reranker is not None:
                non_support_reranked = local_reranker.rerank_candidates(
                    image_path,
                    non_support_reranked,
                    base_score_key="base_rank_score",
                    candidate_limit=local_rerank_k,
                    lambda_global=lambda_global,
                )
            matched_non_support_evidence = unique_studies(
                [
                    {
                        **public_candidate(candidate, label),
                        "non_support_rank_score": candidate["final_rank_score"],
                        "non_support_basis": "no_positive_label_support",
                    }
                    for candidate in non_support_reranked
                ],
                top_k,
            )

            support_stats_candidates = unique_studies(positive_bank, statistics_k)
            non_support_stats_candidates = unique_studies(non_support_bank, statistics_k)
            support_stats = similarity_statistics(support_stats_candidates)
            non_support_stats = similarity_statistics(non_support_stats_candidates)
            margins = {
                f"M@{k}": margin_at_k(support_stats_candidates, non_support_stats_candidates, k)
                for k in (3, 5, 10, 20)
            }
            primary_margin = margins["M@5"]["margin"]
            margin_3_consistent = same_nonzero_sign(margins["M@3"]["margin"], primary_margin)
            margin_10 = margins["M@10"]["margin"]
            margin_10_strong_reversal = bool(
                primary_margin is not None
                and margin_10 is not None
                and primary_margin * margin_10 < 0.0
                and abs(margin_10) >= margin_10_reversal_ratio * abs(primary_margin)
            )
            margin_20 = margins["M@20"]["margin"]
            margin_20_warning = bool(
                primary_margin is not None
                and margin_20 is not None
                and primary_margin * margin_20 < 0.0
            )
            neighborhood = unique_studies(global_candidates, statistics_k)
            if neighborhood:
                similarities = np.asarray([candidate["cosine_similarity"] for candidate in neighborhood])
                weights = np.exp((similarities - similarities.max()) / neighborhood_temperature)
                global_support_density = float(
                    np.dot(weights, [candidate["support"][label] for candidate in neighborhood]) / weights.sum()
                )
            else:
                global_support_density = None
            valid_support_count = sum(
                candidate["cosine_similarity"] >= support_similarity_threshold
                for candidate in support_stats_candidates
            )
            support_coverage = valid_support_count / max(1, min(statistics_k, len(support_stats_candidates)))
            best_support = support_stats["max"]
            absolute_quality = (
                max(0.0, min(1.0, (best_support - support_similarity_threshold) / (1.0 - support_similarity_threshold)))
                if best_support is not None else 0.0
            )
            separation_quality = (
                max(0.0, min(1.0, abs(primary_margin) / max(quality_margin_threshold, 1e-6)))
                if primary_margin is not None else 0.0
            )
            quality_score = 0.50 * absolute_quality + 0.25 * support_coverage + 0.25 * separation_quality
            quality_usable = bool(best_support is not None and best_support >= support_similarity_threshold and quality_score >= 0.50)
            quality_level = "HIGH" if quality_score >= 0.75 else "MEDIUM" if quality_score >= 0.50 else "LOW"
            if not supporting_evidence or not quality_usable:
                evidence_status = "INSUFFICIENT"
            elif not margin_3_consistent or margin_10_strong_reversal:
                evidence_status = "MIXED"
            elif primary_margin is not None and primary_margin >= quality_margin_threshold:
                evidence_status = "SUPPORTED"
            elif primary_margin is not None and primary_margin <= -quality_margin_threshold:
                evidence_status = "NON_SUPPORT_DOMINANT"
            else:
                evidence_status = "MIXED"

            label_evidence.append(
                {
                    "label": label,
                    "classifier_probability": probability,
                    "evidence_status": evidence_status,
                    "evidence_available": bool(supporting_evidence),
                    "candidate_counts": {
                        "global": len(global_candidates),
                        "positive_bank": len(positive_bank),
                        "non_support_bank": len(non_support_bank),
                        "union": len(union_candidates),
                    },
                    "retrieval_statistics": {
                        "top_k": min(statistics_k, bank_candidate_k),
                        "primary_metric": "M@5",
                        "support": support_stats,
                        "non_support": non_support_stats,
                        "margins": margins,
                        "global_weighted_support_density": global_support_density,
                        "support_coverage": float(support_coverage),
                    },
                    "retrieval_quality": {
                        "score": float(quality_score),
                        "level": quality_level,
                        "usable": quality_usable,
                        "support_similarity_threshold": support_similarity_threshold,
                        "margin_threshold": quality_margin_threshold,
                        "margin_3_same_sign_as_margin_5": margin_3_consistent,
                        "margin_10_strong_reversal": margin_10_strong_reversal,
                        "margin_10_reversal_ratio": margin_10_reversal_ratio,
                        "margin_20_warning": margin_20_warning,
                    },
                    "supporting_evidence": supporting_evidence,
                    "matched_non_support_evidence": matched_non_support_evidence,
                }
            )

        label_evidence.sort(
            key=lambda item: item["classifier_probability"],
            reverse=True,
        )
        return {
            "selected_labels": list(selected_labels),
            "global_retrieval_executed": True,
            "densenet_local_reranker_applied": local_reranker is not None,
            "common_evidence": common_evidence,
            "label_evidence": label_evidence,
        }
