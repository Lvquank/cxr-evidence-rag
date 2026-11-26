# CXR Evidence Gradio interface

Prepare the research environment and restore the external artifacts before launching the interface. [Setup and external artifacts](../docs/SETUP.md)

## Launch on Windows

From the repository root, run `StartGradio.bat` or use PowerShell.

```powershell
$env:CXR_RESEARCH_PYTHON = (Get-Command python).Source
.\StartGradio.ps1
```

Set `CXR_RESEARCH_PYTHON` while your CUDA-enabled research environment is active. Without the variable, the launcher looks for `.venv-bertscore/Scripts/python.exe`. The launcher installs the UI dependencies, including Gradio 6.32.0, into `gradio_app/.venv`.

Open <http://127.0.0.1:7860>. Pass `-Port 7861` to change the port, `-NoBrowser` to skip opening the browser, or `-SetupOnly` to install the UI dependencies and exit. Press Ctrl+C to stop the server.

## Interface

- **Analysis** accepts 1–10 PNG or JPEG images and runs each independently. You can choose DenseNet121 or ResNet50, a model, a reasoning effort, and one or all four comparison methods. The defaults are `gpt-5.6-luna`, high reasoning effort, an 8192-token output cap, CUDA, and a disabled local response cache.
- **Research** displays Tables 5 and 6 from the high-effort run completed on November 1, 2026, and the text evaluation updated on November 13, 2026. Select a study to view the four saved reports. This tab requires the historical result artifacts.
- **History** shows reports, classifier scores, supporting and matched non-supporting evidence, routing and gate decisions, API response IDs, and actual token usage. You can download the original JSON for a case.

Each new case is saved in `outputs/gradio_cases/<uuid>/`. Starting the interface and viewing saved reports use local data. New analyses call the API endpoint configured in `.env` through the research worker and can incur charges. The API key stays with the worker.

## Models and evaluation

DenseNet121 is the default classifier. The optional ResNet50 checkpoint belongs in `gradio_app/models/`. Its SHA-256 is `a304a6c73a2a0543314cace0fdd31666e1e1534584d0ae41c3a7bdf7e92f08a0`. ResNet50 uses the frozen DenseNet thresholds and gate settings and still requires separate calibration and validation. Local reranking uses the original DenseNet feature extractor for both classifiers.

The worker matches original IU X-Ray files by SHA-256 so it can exclude the query study from retrieval. Enter a known study ID for a re-encoded image. The learned gate uses the held-out fold recorded in `folds.json` for known studies. External cases use frozen fold 0 as the deployment fallback. Only known studies with held-out folds have out-of-fold predictions.

Table 5 covers 206 studies. Table 6 uses the common 205-study clinical cohort because study 1615 has empty Findings and Impression references. The section-specific text evaluation uses 163 Findings and 204 Impression references. Reference text remains empty where missing. Unlabeled uploads have no accuracy score.

## Backend tests

Run from the repository root in your research environment. These tests use local fixtures and require no API calls.

```powershell
python -m unittest discover -s tests -p test_gradio_backend.py -v
```

`app.py` defines the Gradio components and callbacks. `styles.css` controls presentation, `backend.py` manages case state and subprocesses, and `worker.py` runs the research pipeline.
