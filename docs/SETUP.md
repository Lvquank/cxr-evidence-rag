# Setup and external artifacts

## Environment

Use Python 3.11 or newer. Install `requirements.txt` into a research environment, then install a CUDA-enabled PyTorch/torchvision build appropriate for the machine. The generic requirements and legacy Conda file do not guarantee a CUDA build.

```powershell
python -c "import torch; assert torch.cuda.is_available(), 'CUDA is required'; print(torch.__version__, torch.cuda.get_device_name(0))"
```

Stop if CUDA is unavailable. Local neural inference and clinical evaluators must not fall back to CPU. FAISS CPU indexing is separate from neural-model inference.

Copy `.env.example` to `.env` and set a private API key. The Responses API client supports `LLM_API_KEY` or `OPENAI_API_KEY`; endpoint overrides are available through the existing client and worker. Verify model availability on your endpoint before generating reports. Keep the experiment's model identifier and actual returned model in provenance.

The YAML files contain historical workstation paths and some legacy generation defaults. Adapt data paths to your machine and explicitly pass CUDA/high/8192/`--no-cache` to current command-line workflows. Do not change the frozen retrieval protocol or learned gate as part of environment setup.

## Assets excluded from Git

Obtain datasets from their original sources under the applicable access and usage terms. Restore trained artifacts from the original research environment; this repository does not provide public checkpoint downloads or replacement weights.

| Asset | Expected location |
| --- | --- |
| IU-Xray report and projection CSVs | `src/data/indiana_reports.csv`, `src/data/indiana_projections.csv` |
| Processed gallery radiographs | `data/reprocessed/images/` |
| Projected gallery artifacts | `data/embeddings_chexpert_projection/` containing `faiss.index`, `metadata.jsonl`, embedding manifest, and associated arrays/IDs |
| DenseNet checkpoint | `model/best_E2_relation_aware_weighted_bce.pt` |
| Pathology projection checkpoint | `model/rad_dino_chexpert_projection_best.pt` |
| Frozen learned gate | `src/learned_gate_oof_206/folds.json` and `models/fold_0.pkl` through `models/fold_4.pkl` |
| Evaluation manifest | `data/evaluations/leave_one_out_206_manifest.json` |
| Optional ResNet50 checkpoint | `gradio_app/models/best_resnet50_E2_relation_aware_weighted_bce.pt` |
| Historical results and report provenance | Original `outputs/` experiment folders |
| Clinical evaluator dependencies and weights | Local CheXbert/RadGraph checkouts and trusted checkpoints required by their evaluator scripts |

Only load serialized model and scikit-learn artifacts from trusted sources. The source release does not establish redistribution rights for external data or model files.

## Ingestion and gallery construction

The following commands illustrate preparing a new local gallery. They do not reproduce the frozen evaluation gallery unless its original inputs, settings, and trained artifacts are supplied unchanged.

```powershell
python scripts/ingest_iuxray.py --raw_dir path/to/IU-Xray --out_dir data/reprocessed --projections_csv src/data/indiana_projections.csv --reports_csv src/data/indiana_reports.csv
python scripts/compute_image_embeddings.py --processed_dir data/reprocessed --out_dir data/embeddings_chexpert_projection --projection_checkpoint model/rad_dino_chexpert_projection_best.pt --device cuda --vector_db faiss
```

Review `config/config.yaml` and the script's `--help` before running. Preserve existing data directories and use new directories when exploring different gallery settings. Do not mix embeddings, metadata, indexes, or projection checkpoints from different runs.

## Gradio

The research interpreter is selected by `CXR_RESEARCH_PYTHON`; without it, the Windows launcher looks for `.venv-bertscore/Scripts/python.exe`. The launcher installs UI dependencies separately into `gradio_app/.venv`.

```powershell
$env:CXR_RESEARCH_PYTHON = (Get-Command python).Source
.\StartGradio.ps1 -Port 7860
```

The research tab needs the original high-effort and text-similarity result folders. Those artifacts are excluded from Git, so historical views are unavailable in a code-only clone.

## Tests and experiments

The README's focused unit tests exercise synthetic evidence, mocked API responses, and report validation. Full test discovery also includes tests tied to local research artifacts; it is not an asset-free reproduction test.

For new report-generation experiments, use `gpt-5.6-luna`, high reasoning effort, 8192 output tokens, CUDA inference, and disabled local response cache. Use identical effort and token caps for all comparison branches. Preserve existing outputs and record the actual request settings and returned usage.

Before recomputing metrics, validate all 206 studies and report provenance. Use the common 205-study clinical cohort when both reference sections are required; do not invent text for the empty reference in study 1615. Section-specific text metrics use only the nonempty section references.
