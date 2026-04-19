# Aster

Aster is a local desktop coding orchestrator. You describe the change in plain English, Aster gathers project context from your machine, sends a structured request to ChatGPT in your browser by default, and returns a deterministic patch plan with file-level operations.

## What it does

- Accepts natural-language coding requests.
- Detects or lets you choose a project root.
- Collects a filtered directory tree, manifests, configs, source files, logs, and readme notes.
- Redacts common secrets before sending context externally.
- Requires strict machine-parseable operations from the model.
- Previews diffs before applying changes locally.
- Creates backups before edits and can create a git checkpoint for the files about to change.
- Stores session history for iterative follow-up requests.
- Can sync with GitHub so the local repo stays current before planning, pushes tracked runtime logs after planning runs, and pushes approved code changes plus logs after apply runs.

## Architecture

```text
aster/
  browser_core/
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

- Browser mode: default. Uses the Screen Reader automation stack to attach to a logged-in ChatGPT browser tab, or opens `https://chatgpt.com/` if no ChatGPT window is open yet. The browser architecture is being split into `patch_runner` and `conversation_operator` strategies; this pass adds the `browser_core` scaffolding and keeps current runtime behavior on the existing patch-runner flow.
- API mode: optional. Disabled by default and only available if you explicitly turn it on in `aster.config.json`.

## GitHub sync

If your project is connected to GitHub, Aster can:

- `git fetch` and `git pull --ff-only` before building context.
- `git add -A -- .aster/audit.log.jsonl .aster/last_launch_stdout.log .aster/last_launch_stderr.log`, `git commit`, and `git push` after planning runs so only tracked runtime logs are published.
- `git add -A`, `git commit`, and `git push` after approved changes are applied.
- publish `.aster/audit.log.jsonl`, `.aster/last_launch_stdout.log`, and `.aster/last_launch_stderr.log` to GitHub so remote diagnostics stay current.
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

4. Run the startup preflight:

```powershell
.\.venv\Scripts\python.exe .\run_assistant.py doctor --project-root .
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
- The launcher now runs a preflight check before opening the UI, writes its result into `.aster/last_launch_stdout.log`, and then triggers a runtime Git sync so launch logs are pushed too.
- The UI now shows the patch preview, raw model response, and a live tail of `.aster` logs on the right side.

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
- Command and dependency operations are validated before apply and require explicit review.
- Command execution uses argv-based allowlisted commands instead of shell execution.
- Backups are created before edits.
- Prompts, plans, and actions are logged under `.aster/`.
- The tracked `.aster` logs are committed and pushed so GitHub reflects the latest runtime state after each run.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest
```

## Current status

Browser mode is the intended free path. API mode still exists behind configuration, but the normal workflow is browser-first and will open ChatGPT in your browser if needed. By default, Aster now commits and pushes tracked runtime logs after planning and launcher runs, commits and pushes approved code changes after apply runs, and keeps the current branch name instead of renaming it to `main`.
