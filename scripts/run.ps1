param(
    [switch]$Console
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot
$DataDir = Join-Path $ProjectRoot ".aster"
New-Item -ItemType Directory -Force -Path $DataDir | Out-Null
$StdoutLog = Join-Path $DataDir "last_launch_stdout.log"
$StderrLog = Join-Path $DataDir "last_launch_stderr.log"
Set-Content -Path $StdoutLog -Value "" -Encoding utf8
Set-Content -Path $StderrLog -Value "" -Encoding utf8

$PythonExe = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $PythonExe)) {
    Write-Host "Virtual environment not found. Bootstrapping..."
    & powershell.exe -NoProfile -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot "bootstrap.ps1")
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path $PythonExe)) {
        throw "Bootstrap failed. Check the PowerShell output and try again."
    }
}
$PythonwExe = Join-Path $ProjectRoot ".venv\Scripts\pythonw.exe"

$RequiredModules = @("requests")

$MissingModules = & $PythonExe -c "import importlib.util, sys; missing = [name for name in sys.argv[1:] if importlib.util.find_spec(name) is None]; print('|'.join(missing))" @RequiredModules
if ($LASTEXITCODE -ne 0) {
    throw "Failed to verify runtime dependencies."
}

if ($MissingModules) {
    Write-Host "Installing missing runtime dependencies: $MissingModules"
    & $PythonExe -m pip install -e .
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to install runtime dependencies."
    }
}

if ($Console) {
    & $PythonExe .\run_assistant.py ui 1>> $StdoutLog 2>> $StderrLog
} else {
    if (-not (Test-Path $PythonwExe)) {
        Start-Process -FilePath $PythonExe -ArgumentList ".\run_assistant.py","ui" -WorkingDirectory $ProjectRoot | Out-Null
    } else {
        Start-Process -FilePath $PythonwExe -ArgumentList ".\run_assistant.py","ui" -WorkingDirectory $ProjectRoot | Out-Null
    }
}
