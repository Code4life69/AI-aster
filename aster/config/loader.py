from __future__ import annotations

import json
from pathlib import Path

from aster.browser_core.models import BrowserStrategy

from .models import AsterConfig


def load_config(project_root: Path, config_path: Path | None = None) -> AsterConfig:
    root = project_root.resolve()
    target = config_path or root / "aster.config.json"
    config = AsterConfig(
        project_root=root,
        log_dir=root / ".aster",
        thread_registry_path=root / ".aster" / "thread_registry.json",
        visual_action_trace_dir=root / ".aster" / "visual_trace",
        visual_action_memory_path=root / ".aster" / "visual_region_memory.json",
    )
    if not target.exists():
        return config
    data = json.loads(target.read_text(encoding="utf-8"))
    thread_registry_raw = Path(str(data.get("thread_registry_path", config.thread_registry_path)))
    if not thread_registry_raw.is_absolute():
        thread_registry_raw = root / thread_registry_raw
    visual_trace_dir_raw = Path(str(data.get("visual_action_trace_dir", config.visual_action_trace_dir)))
    if not visual_trace_dir_raw.is_absolute():
        visual_trace_dir_raw = root / visual_trace_dir_raw
    visual_memory_path_raw = Path(str(data.get("visual_action_memory_path", config.visual_action_memory_path)))
    if not visual_memory_path_raw.is_absolute():
        visual_memory_path_raw = root / visual_memory_path_raw
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
        browser_strategy=_parse_browser_strategy(data.get("browser_strategy", config.browser_strategy)),
        thread_reuse_enabled=bool(data.get("thread_reuse_enabled", config.thread_reuse_enabled)),
        thread_registry_path=thread_registry_raw,
        verification_level=str(data.get("verification_level", config.verification_level)),
        log_screenshots=bool(data.get("log_screenshots", config.log_screenshots)),
        max_recovery_attempts=int(data.get("max_recovery_attempts", config.max_recovery_attempts)),
        visual_action_trace_enabled=bool(
            data.get("visual_action_trace_enabled", config.visual_action_trace_enabled)
        ),
        visual_action_trace_dir=visual_trace_dir_raw,
        visual_action_memory_enabled=bool(
            data.get("visual_action_memory_enabled", config.visual_action_memory_enabled)
        ),
        visual_action_memory_path=visual_memory_path_raw,
        visual_action_trace_checkpoint_interval_seconds=int(
            data.get(
                "visual_action_trace_checkpoint_interval_seconds",
                config.visual_action_trace_checkpoint_interval_seconds,
            )
        ),
        visual_action_trace_max_checkpoints_per_key=int(
            data.get(
                "visual_action_trace_max_checkpoints_per_key",
                config.visual_action_trace_max_checkpoints_per_key,
            )
        ),
        runtime_log_heartbeat_push_enabled=bool(
            data.get("runtime_log_heartbeat_push_enabled", config.runtime_log_heartbeat_push_enabled)
        ),
        runtime_log_heartbeat_interval_seconds=int(
            data.get(
                "runtime_log_heartbeat_interval_seconds",
                config.runtime_log_heartbeat_interval_seconds,
            )
        ),
        runtime_log_heartbeat_branch_only=bool(
            data.get("runtime_log_heartbeat_branch_only", config.runtime_log_heartbeat_branch_only)
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


def _parse_browser_strategy(value: object) -> BrowserStrategy:
    try:
        return BrowserStrategy(str(value))
    except ValueError:
        return BrowserStrategy.PATCH_RUNNER
