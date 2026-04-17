param(
    [switch]$Console
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot
$PythonExe = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$ExistingUi = Get-CimInstance Win32_Process -ErrorAction SilentlyContinue | Where-Object {
    $_.ExecutablePath -eq $PythonExe -and $_.CommandLine -match 'run_assistant\.py"\s+ui|run_assistant\.py\s+ui'
} | Select-Object -First 1
if ($ExistingUi) {
    Write-Host "Aster UI is already running."
    return
}

$DataDir = Join-Path $ProjectRoot ".aster"
New-Item -ItemType Directory -Force -Path $DataDir | Out-Null
$StdoutLog = Join-Path $DataDir "last_launch_stdout.log"
$StderrLog = Join-Path $DataDir "last_launch_stderr.log"
try {
    Set-Content -Path $StdoutLog -Value "" -Encoding utf8 -ErrorAction Stop
} catch {
    Write-Warning "Could not reset $StdoutLog because it is in use. Appending to the existing file instead."
}
try {
    Set-Content -Path $StderrLog -Value "" -Encoding utf8 -ErrorAction Stop
} catch {
    Write-Warning "Could not reset $StderrLog because it is in use. Appending to the existing file instead."
}

if (-not (Test-Path $PythonExe)) {
    Write-Host "Virtual environment not found. Bootstrapping..."
    & powershell.exe -NoProfile -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot "bootstrap.ps1")
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path $PythonExe)) {
        throw "Bootstrap failed. Check the PowerShell output and try again."
    }
}
$RequiredModules = @(
    "requests",
    "pyautogui",
    "pywinauto",
    "pyperclip",
    "screen_reader.automation",
    "screen_reader.capture",
    "screen_reader.ocr"
)

$DependencyCheckScript = @'
import importlib.util
import sys
from pathlib import Path

sibling = Path(r"C:/Screen Reader")
if sibling.exists() and str(sibling) not in sys.path:
    sys.path.insert(0, str(sibling))

missing = [name for name in sys.argv[1:] if importlib.util.find_spec(name) is None]
print("|".join(missing))
'@

$MissingModules = $DependencyCheckScript | & $PythonExe - @RequiredModules
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

$DoctorOutput = & $PythonExe .\run_assistant.py doctor --project-root . 2>&1
$DoctorExit = $LASTEXITCODE
if ($DoctorOutput) {
    Add-Content -Path $StdoutLog -Value (($DoctorOutput | Out-String).TrimEnd()) -Encoding utf8
}
if ($DoctorExit -ne 0) {
    throw "Startup preflight failed. Review .aster\\last_launch_stdout.log for details."
}

$SyncOutput = & $PythonExe .\run_assistant.py sync-runtime --project-root . --message "Aster runtime sync: launcher start" 2>&1
$SyncExit = $LASTEXITCODE
if ($SyncOutput) {
    Add-Content -Path $StdoutLog -Value (($SyncOutput | Out-String).TrimEnd()) -Encoding utf8
}
if ($SyncExit -ne 0) {
    Add-Content -Path $StderrLog -Value "Runtime sync returned exit code $SyncExit." -Encoding utf8
}

if ($Console) {
    & $PythonExe .\run_assistant.py ui 1>> $StdoutLog 2>> $StderrLog
} else {
    Start-Process `
        -FilePath $PythonExe `
        -ArgumentList ".\run_assistant.py","ui" `
        -WorkingDirectory $ProjectRoot `
        -WindowStyle Hidden `
        -RedirectStandardOutput $StdoutLog `
        -RedirectStandardError $StderrLog | Out-Null
}
