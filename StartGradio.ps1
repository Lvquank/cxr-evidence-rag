param([int]$Port = 7860, [switch]$NoBrowser, [switch]$SetupOnly)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$uiPython = Join-Path $PSScriptRoot 'gradio_app\.venv\Scripts\python.exe'
$researchPython = if ($env:CXR_RESEARCH_PYTHON) { $env:CXR_RESEARCH_PYTHON } else { Join-Path $PSScriptRoot '.venv-bertscore\Scripts\python.exe' }
if (-not (Test-Path -LiteralPath $researchPython)) { throw 'CUDA Python environment not found. See gradio_app/README.md.' }
if (-not (Test-Path -LiteralPath $uiPython)) {
    & $researchPython -m venv (Join-Path $PSScriptRoot 'gradio_app\.venv')
    if ($LASTEXITCODE -ne 0) { throw 'Could not create the Gradio environment.' }
}
& $uiPython -c "import importlib.util as u, sys; from importlib.metadata import version; sys.exit(0 if u.find_spec('gradio') and u.find_spec('PIL') and version('gradio') == '6.32.0' else 1)"
if ($LASTEXITCODE -ne 0) {
    & $uiPython -m ensurepip --upgrade
    & $uiPython -m pip install -r (Join-Path $PSScriptRoot 'gradio_app\requirements.txt')
    if ($LASTEXITCODE -ne 0) { throw 'Could not install the Gradio dependencies.' }
}
if ($SetupOnly) { exit 0 }
$env:PYTHONIOENCODING = 'utf-8'
$env:GRADIO_ANALYTICS_ENABLED = 'False'
$appArgs = @('-m', 'gradio_app.app', '--port', "$Port")
if (-not $NoBrowser) { $appArgs += '--browser' }
& $uiPython @appArgs
exit $LASTEXITCODE
