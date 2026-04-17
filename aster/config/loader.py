from __future__ import annotations

import json
from pathlib import Path

from .models import AsterConfig


def load_config(project_root: Path, config_path: Path | None = None) -> AsterConfig:
    root = project_root.resolve()
    target = config_path or root / "aster.config.json"
    config = AsterConfig(project_root=root, log_dir=root / ".aster")
    if not target.exists():
        return config
    data = json.loads(target.read_text(encoding="utf-8"))
    return AsterConfig(
        project_root=root,
        ignore_patterns=list(data.get("ignore_patterns", config.ignore_patterns)),
        max_file_bytes=int(data.get("max_file_bytes", config.max_file_bytes)),
        max_total_prompt_bytes=int(data.get("max_total_prompt_bytes", config.max_total_prompt_bytes)),
        preferred_model=str(data.get("preferred_model", config.preferred_model)),
        openai_api_key_env=str(data.get("openai_api_key_env", config.openai_api_key_env)),
        default_mode=str(data.get("default_mode", config.default_mode)),
        browser_mode_enabled=bool(data.get("browser_mode_enabled", config.browser_mode_enabled)),
        api_mode_enabled=bool(data.get("api_mode_enabled", config.api_mode_enabled)),
        chatgpt_url=str(data.get("chatgpt_url", config.chatgpt_url)),
        browser_launch_timeout_seconds=int(
            data.get("browser_launch_timeout_seconds", config.browser_launch_timeout_seconds)
        ),
        auto_apply=bool(data.get("auto_apply", config.auto_apply)),
        git_integration=bool(data.get("git_integration", config.git_integration)),
        auto_commit_and_push=bool(data.get("auto_commit_and_push", config.auto_commit_and_push)),
        push_runtime_logs_after_plan=bool(
            data.get("push_runtime_logs_after_plan", config.push_runtime_logs_after_plan)
        ),
        auto_commit_message_prefix=str(
            data.get("auto_commit_message_prefix", config.auto_commit_message_prefix)
        ),
        git_remote_name=str(data.get("git_remote_name", config.git_remote_name)),
        sync_with_remote=bool(data.get("sync_with_remote", config.sync_with_remote)),
        dry_run=bool(data.get("dry_run", config.dry_run)),
        approval_required_for_destructive=bool(
            data.get("approval_required_for_destructive", config.approval_required_for_destructive)
        ),
        approval_required_for_commands=bool(
            data.get("approval_required_for_commands", config.approval_required_for_commands)
        ),
        history_limit=int(data.get("history_limit", config.history_limit)),
        log_dir=root / str(data.get("log_dir", ".aster")),
    )
