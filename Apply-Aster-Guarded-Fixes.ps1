param(
    [string]$Repo = ".",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$py = Join-Path $scriptDir "apply_aster_guarded_fixes.py"

if (-not (Test-Path $py)) {
    Write-Error "Missing helper script: $py"
    exit 1
}

$pythonCmd = Get-Command python -ErrorAction SilentlyContinue
if ($pythonCmd) {
    $runner = @("python")
} else {
    $pyCmd = Get-Command py -ErrorAction SilentlyContinue
    if (-not $pyCmd) {
        Write-Error "Python was not found on PATH."
        exit 1
    }
    $runner = @("py", "-3")
}

$args = @($py, $Repo)
if ($DryRun) {
    $args += "--dry-run"
}

Write-Host ""
Write-Host "=========================================="
Write-Host "  Aster Guarded Fixes"
Write-Host "=========================================="
Write-Host ""

& $runner[0] @($runner[1..($runner.Length-1)] + $args)
exit $LASTEXITCODE
