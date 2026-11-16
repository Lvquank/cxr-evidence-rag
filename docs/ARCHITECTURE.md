# Architecture

## Evidence pipeline

1. `src/data/processor.py` prepares radiographs and normalizes their metadata.
2. `src/embeddings/encoder.py` extracts RAD-DINO representations. Pathology projection is implemented in `src/embeddings/pathology_projection.py`.
3. `src/embeddings/densenet_chexpert.py` produces six-label probabilities and spatial feature maps. Frozen selection thresholds are read by `src/classifier_thresholds.py`.
4. `src/retriever/faiss_retriever.py` searches the gallery and constructs label-specific support and non-support banks. `densenet_local_reranker.py` combines local spatial similarity with global similarity.
5. `src/retriever/pathology_support.py` describes evidence strength and consistency. Classifier probabilities determine label selection and are not included in reference ranking.
6. `scripts/generate_minimal_json_for_llm.py` creates compact evidence while retaining selected labels, thresholds, and supporting/non-supporting references.
7. `scripts/run_full_pipeline.py` chooses a report route and passes evidence to `src/llm_client.py` and `src/prompts.py`. Responses are validated against the target problem ontology.

The compact evidence contract supports up to six selected labels. Retrieval summaries describe agreement in the gallery; report polarity and classifier selection remain separate signals.

## Learned gate and comparison runners

`scripts/profile_learned_gate_latency.py` applies frozen scikit-learn gate artifacts. Its model files and fold mapping are external research assets. Do not retrain or substitute them when reproducing the frozen comparison.

`scripts/run_high_effort_4branches.py` coordinates structured rule, structured learned gate, classifier-plus-vision, and classifier-plus-retrieval-plus-vision runs. This driver also expects prior local provenance artifacts and is not a standalone fresh-clone demo.

`scripts/validate_common_studies.py` checks common study identities and report provenance. Clinical evaluation and bootstrap scripts are separate from generation. Missing reference reports are excluded explicitly; no synthetic reference is inserted.

## Interface boundary

`gradio_app/backend.py` manages case records and subprocesses. `worker.py` adapts the research pipeline, checks CUDA and required artifacts, and records model settings, hashes, routing, gate decisions, and API usage. `presentation.py` formats the results.

The interface uses a separate Python environment from the research worker. API keys remain in the worker environment. New cases are written under `outputs/gradio_cases/` rather than into frozen experiment directories.

## Reproduction boundaries

This repository preserves source code and small configuration artifacts. It intentionally excludes medical images, report CSVs, generated study reports, frozen model binaries, retrieval galleries, and local credentials. Historical result views and data-dependent regression tests require the original external assets.

No report generation, neural inference, or metric recomputation is needed to inspect the implementation or run the focused mocked tests in the main README.
