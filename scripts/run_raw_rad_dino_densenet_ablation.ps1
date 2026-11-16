<#
Run the retrieval-only ablation:
  raw RAD-DINO global retrieval vs. raw RAD-DINO + DenseNet-local reranking.

No LLM request is made. The script creates a separate output folder, then runs:
  1. cache global/local similarities for the frozen 206-study manifest;
  2. lambda-global sweep from that cache;
  3. paired study-level bootstrap confidence intervals versus global-only.
#>

[CmdletBinding()]
param(
    [string]$PythonPath = 'D:\envs\image_search\python.exe',
    [ValidateSet('cpu', 'cuda', 'cuda:0')]
    [string]$Device = 'cpu',
    [string]$OutputRoot = 'outputs\flow1\raw_rad_dino_densenet',
    [int]$CacheCandidates = 50,
    [double[]]$LambdaGlobal = @(1.0, 0.9, 0.7, 0.5, 0.3),
    [int]$BootstrapIterations = 2000,
    [int]$Seed = 20260904,
    [switch]$AllowDuplicateOpenMP,
    [switch]$RebuildCache
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

if (-not (Test-Path -LiteralPath $PythonPath -PathType Leaf)) {
    throw "Python executable was not found: $PythonPath"
}
if ($CacheCandidates -lt 5) {
    throw 'CacheCandidates must be at least 5 for the configured @5 metrics.'
}
if ($BootstrapIterations -lt 100) {
    throw 'BootstrapIterations must be at least 100.'
}
if ($LambdaGlobal | Where-Object { $_ -lt 0.0 -or $_ -gt 1.0 }) {
    throw 'Every LambdaGlobal value must be within [0, 1].'
}

$RawEmbeddingDir = Join-Path $ProjectRoot 'data\embeddings'
$ProcessedDir = Join-Path $ProjectRoot 'data\reprocessed'
$Manifest = Join-Path $ProjectRoot 'data\evaluations\leave_one_out_206_manifest.json'
foreach ($required in @(
    (Join-Path $RawEmbeddingDir 'embeddings.npy'),
    (Join-Path $RawEmbeddingDir 'metadata.jsonl'),
    (Join-Path $RawEmbeddingDir 'faiss.index'),
    $ProcessedDir,
    $Manifest
)) {
    if (-not (Test-Path -LiteralPath $required)) {
        throw "Required raw-RAD-DINO artifact is missing: $required"
    }
}

$OutputPath = [System.IO.Path]::GetFullPath((Join-Path $ProjectRoot $OutputRoot))
New-Item -ItemType Directory -Force -Path $OutputPath | Out-Null
$CacheBuild = Join-Path $OutputPath 'retrieval_cache_build.json'
$Cache = Join-Path $OutputPath 'retrieval_similarity_cache.json'
$SweepJson = Join-Path $OutputPath 'lambda_sweep_results.json'
$SweepCsv = Join-Path $OutputPath 'lambda_sweep_per_query.csv'
$BootstrapJson = Join-Path $OutputPath 'lambda_sweep_bootstrap_ci.json'

# This affects only child commands launched by this PowerShell process.
$PreviousOpenMP = $env:KMP_DUPLICATE_LIB_OK
if ($AllowDuplicateOpenMP) {
    Write-Warning 'KMP_DUPLICATE_LIB_OK=TRUE is an OpenMP compatibility workaround; use it only for this experimental CPU run.'
    $env:KMP_DUPLICATE_LIB_OK = 'TRUE'
}

try {
    if ($RebuildCache -or -not (Test-Path -LiteralPath $Cache)) {
        Write-Host '[1/3] Caching raw RAD-DINO global and DenseNet-local scores for 206 studies...' -ForegroundColor Cyan
        & $PythonPath scripts\evaluate_hybrid_cxr_retrieval.py `
            --gallery_dir $RawEmbeddingDir `
            --processed_dir $ProcessedDir `
            --query-manifest $Manifest `
            --local_rerank_candidates $CacheCandidates `
            --cache_candidates $CacheCandidates `
            --k 5 `
            --device $Device `
            --output_json $CacheBuild `
            --similarity_cache_json $Cache
        if ($LASTEXITCODE -ne 0) { throw "Cache build failed with exit code $LASTEXITCODE." }
    }
    else {
        Write-Host "[1/3] Reusing existing cache: $Cache" -ForegroundColor Yellow
    }

    Write-Host '[2/3] Sweeping lambda_global without model inference or LLM calls...' -ForegroundColor Cyan
    & $PythonPath scripts\sweep_retrieval_lambda.py `
        --cache-json $Cache `
        --output-json $SweepJson `
        --output-csv $SweepCsv `
        --lambda-values $LambdaGlobal `
        --k 5
    if ($LASTEXITCODE -ne 0) { throw "Lambda sweep failed with exit code $LASTEXITCODE." }

    Write-Host '[3/3] Computing paired study-level 95% bootstrap confidence intervals...' -ForegroundColor Cyan
    & $PythonPath scripts\bootstrap_retrieval_lambda_ci.py `
        --per-query-csv $SweepCsv `
        --output-json $BootstrapJson `
        --iterations $BootstrapIterations `
        --seed $Seed
    if ($LASTEXITCODE -ne 0) { throw "Bootstrap CI failed with exit code $LASTEXITCODE." }
}
finally {
    if ($null -eq $PreviousOpenMP) {
        Remove-Item Env:\KMP_DUPLICATE_LIB_OK -ErrorAction SilentlyContinue
    }
    else {
        $env:KMP_DUPLICATE_LIB_OK = $PreviousOpenMP
    }
}

Write-Host "Completed. Results: $OutputPath" -ForegroundColor Green
Write-Host "- $SweepJson"
Write-Host "- $BootstrapJson"
