param()

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    throw "Python 3.11+ is required and was not found on PATH."
}

if (-not (Test-Path ".venv")) {
    python -m venv .venv
}

$PythonExe = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

& $PythonExe -m pip install --upgrade pip
& $PythonExe -m pip install -e .

Write-Host ""
Write-Host "Bootstrap complete."
Write-Host "Project root: $ProjectRoot"
Write-Host "Python env:  $PythonExe"
