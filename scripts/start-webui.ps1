[CmdletBinding()]
param([switch]$CheckOnly)

$ErrorActionPreference = 'Stop'
$indexTtsProject = Split-Path -Parent $PSScriptRoot
$indexTtsModelDir = Join-Path $indexTtsProject 'checkpoints'
$indexTtsVirtualEnv = Join-Path $indexTtsProject '.venv'
$indexTtsEnvironmentScript = $null
$indexTtsLocalConfig = Join-Path $indexTtsProject '.index-tts.local.ps1'
if (Test-Path -LiteralPath $indexTtsLocalConfig) { . $indexTtsLocalConfig }
if ($indexTtsEnvironmentScript) { . $indexTtsEnvironmentScript }

if (-not (Test-Path -LiteralPath (Join-Path $indexTtsVirtualEnv 'Scripts/python.exe'))) {
    throw 'Python environment not found. Run uv sync --extra webui or configure .index-tts.local.ps1.'
}
if (-not (Test-Path -LiteralPath (Join-Path $indexTtsModelDir 'config.yaml'))) {
    throw 'Model configuration not found. Download the model or configure .index-tts.local.ps1.'
}
if (-not $CheckOnly -and (Get-NetTCPConnection -LocalPort 17860 -State Listen -ErrorAction SilentlyContinue)) {
    throw 'Port 17860 is already in use. Check http://127.0.0.1:17860.'
}

$indexTtsUv = (Get-Command uv -ErrorAction Stop).Source
$env:UV_PROJECT_ENVIRONMENT = $indexTtsVirtualEnv
$env:PYTHONUNBUFFERED = '1'
$env:NO_PROXY = (@($env:NO_PROXY, '127.0.0.1', 'localhost', '::1') | Where-Object { $_ }) -join ','
Push-Location $indexTtsProject
try {
    # Keep the verified environment unchanged; dependency updates are deliberate.
    if ($CheckOnly) {
        & $indexTtsUv run --no-sync --extra webui webui.py --help
    } else {
        Write-Host 'Starting IndexTTS at http://127.0.0.1:17860. Press Ctrl+C to stop.'
        & $indexTtsUv run --no-sync --extra webui webui.py --model_dir $indexTtsModelDir --version 2.5 --host 127.0.0.1 --port 17860 --fp16
    }
    if ($LASTEXITCODE -ne 0) { throw "IndexTTS exited with code $LASTEXITCODE." }
} finally {
    Pop-Location
}
