# Aster

Aster is a local desktop coding orchestrator. You describe the change in plain English, Aster gathers project context from your machine, sends a structured request to ChatGPT in your browser by default, and returns a deterministic patch plan with file-level operations.

## What it does

- Accepts natural-language coding requests.
- Detects or lets you choose a project root.
- Collects a filtered directory tree, manifests, configs, source files, logs, and readme notes.
- Redacts common secrets before sending context externally.
- Requires strict machine-parseable operations from the model.
- Previews diffs before applying changes locally.
- Creates backups before edits and can create a git checkpoint.
- Stores session history for iterative follow-up requests.
- Can sync with GitHub so the local repo stays current before planning and auto-commit plus auto-push after approved applies.

## Architecture

```text
aster/
  config/
  context_collector/
  prompt_builder/
  transport_api/
  transport_browser/
  response_parser/
  patch_executor/
  diff_preview/
  safety_guard/
  ui_or_cli/
  audit_logger.py
  git_sync.py
  orchestrator.py
  session.py
```

## Modes

- Browser mode: default. Uses the Screen Reader automation stack to attach to a logged-in ChatGPT browser tab, or opens `https://chatgpt.com/` if no ChatGPT window is open yet.
- API mode: optional. Disabled by default and only available if you explicitly turn it on in `aster.config.json`.

## GitHub sync

If your project is connected to GitHub, Aster can:

- `git fetch` and `git pull --ff-only` before building context.
- `git add -A`, `git commit`, and `git push` after approved changes are applied.
- initialize a local repo and set the remote if needed.

Default remote URL for this project:

```text
https://github.com/Code4life69/AI-aster.git
```

## Setup

1. Create the virtual environment and install dependencies:

```powershell
.\scripts\bootstrap.ps1
```

2. Optionally copy the example config:

```powershell
Copy-Item .\aster.config.example.json .\aster.config.json
```

3. Optionally connect the local folder to GitHub:

```powershell
.\.venv\Scripts\python.exe .\run_assistant.py connect-github --url https://github.com/Code4life69/AI-aster.git
```

## Run

Desktop UI:

```powershell
.\scripts\run.ps1
```

Double-click launch:

- `Start Aster.vbs`: opens Aster with no console window.
- `Start Aster.cmd`: launches through PowerShell and auto-bootstraps the virtual environment if needed.
- `Start Aster Console.cmd`: same launcher, but keeps the console visible for troubleshooting.

CLI planning only:

```powershell
.\.venv\Scripts\python.exe .\run_assistant.py plan "build a tkinter todo app" --project-root .
```

CLI apply:

```powershell
.\.venv\Scripts\python.exe .\run_assistant.py apply "refactor the parser" --project-root . --yes
```

## Safety rules

- Secret-like values are redacted before prompts are sent.
- Paths outside the project root are rejected.
- Delete, move, and rename operations are flagged for confirmation.
- Backups are created before edits.
- Prompts, plans, and actions are logged under `.aster/`.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest
```

## Current status

Browser mode is the intended free path. API mode still exists behind configuration, but the normal workflow is browser-first and will open ChatGPT in your browser if needed.
By default, approved applied changes are committed and pushed to the configured GitHub remote automatically.
