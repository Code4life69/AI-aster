from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class AsterConfig:
    project_root: Path
    ignore_patterns: list[str] = field(default_factory=lambda: [
        ".git/",
        ".aster/",
        ".venv/",
        "node_modules/",
        "__pycache__/",
        ".pytest_cache/",
        ".mypy_cache/",
        ".idea/",
        ".vscode/",
        "assistant_app/",
        "data/",
        "haven_assistant.egg-info/",
        ".env",
        "*.env",
        "*.sqlite",
        "*.db",
        "*.db-*",
        "*.png",
        "*.jpg",
        "*.jpeg",
        "*.gif",
        "*.pdf",
        "*.onnx",
        "*.bin",
    ])
    max_file_bytes: int = 48_000
    max_total_prompt_bytes: int = 220_000
    preferred_model: str = "gpt-4o-mini"
    openai_api_key_env: str = "OPENAI_API_KEY"
    default_mode: str = "browser"
    browser_mode_enabled: bool = True
    api_mode_enabled: bool = False
    chatgpt_url: str = "https://chatgpt.com/"
    browser_launch_timeout_seconds: int = 45
    auto_apply: bool = False
    git_integration: bool = True
    auto_commit_and_push: bool = True
    auto_commit_message_prefix: str = "Aster update"
    git_remote_name: str = "origin"
    sync_with_remote: bool = True
    dry_run: bool = True
    approval_required_for_destructive: bool = True
    history_limit: int = 8
    log_dir: Path | None = None
