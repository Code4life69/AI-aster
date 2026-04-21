from aster.browser_core.models import BrowserStrategy
from pathlib import Path

from aster.config import load_config


def test_default_config_uses_browser_mode(tmp_path: Path) -> None:
    config = load_config(tmp_path)
    assert config.default_mode == "browser"
    assert config.browser_mode_enabled is True
    assert config.api_mode_enabled is False
    assert config.browser_strategy == BrowserStrategy.PATCH_RUNNER
    assert config.thread_reuse_enabled is False
    assert config.thread_registry_path == tmp_path / ".aster" / "thread_registry.json"
    assert config.verification_level == "basic"
    assert config.log_screenshots is False
    assert config.max_recovery_attempts == 3
    assert config.visual_action_trace_enabled is False
    assert config.visual_action_trace_dir == tmp_path / ".aster" / "visual_trace"
    assert config.visual_action_memory_enabled is True
    assert config.visual_action_memory_path == tmp_path / ".aster" / "visual_region_memory.json"
    assert config.visual_action_trace_checkpoint_interval_seconds == 15
    assert config.visual_action_trace_max_checkpoints_per_key == 2
    assert config.runtime_log_heartbeat_push_enabled is False
    assert config.runtime_log_heartbeat_interval_seconds == 30
    assert config.runtime_log_heartbeat_branch_only is True
    assert config.auto_commit_and_push is True
    assert config.approval_required_for_commands is True
    assert config.push_runtime_logs_after_plan is True
    assert ".aster/" in config.ignore_patterns


def test_config_loader_reads_new_browser_fields(tmp_path: Path) -> None:
    (tmp_path / "aster.config.json").write_text(
        """
{
  "browser_strategy": "conversation_operator",
  "thread_reuse_enabled": true,
  "thread_registry_path": ".aster/custom_threads.json",
  "verification_level": "strict",
  "log_screenshots": true,
  "max_recovery_attempts": 5,
  "visual_action_trace_enabled": true,
  "visual_action_trace_dir": ".aster/debug_trace",
  "visual_action_memory_enabled": false,
  "visual_action_memory_path": ".aster/custom_region_memory.json",
  "visual_action_trace_checkpoint_interval_seconds": 21,
  "visual_action_trace_max_checkpoints_per_key": 4,
  "runtime_log_heartbeat_push_enabled": true,
  "runtime_log_heartbeat_interval_seconds": 45,
  "runtime_log_heartbeat_branch_only": false
}
""".strip(),
        encoding="utf-8",
    )

    config = load_config(tmp_path)

    assert config.browser_strategy == BrowserStrategy.CONVERSATION_OPERATOR
    assert config.thread_reuse_enabled is True
    assert config.thread_registry_path == tmp_path / ".aster" / "custom_threads.json"
    assert config.verification_level == "strict"
    assert config.log_screenshots is True
    assert config.max_recovery_attempts == 5
    assert config.visual_action_trace_enabled is True
    assert config.visual_action_trace_dir == tmp_path / ".aster" / "debug_trace"
    assert config.visual_action_memory_enabled is False
    assert config.visual_action_memory_path == tmp_path / ".aster" / "custom_region_memory.json"
    assert config.visual_action_trace_checkpoint_interval_seconds == 21
    assert config.visual_action_trace_max_checkpoints_per_key == 4
    assert config.runtime_log_heartbeat_push_enabled is True
    assert config.runtime_log_heartbeat_interval_seconds == 45
    assert config.runtime_log_heartbeat_branch_only is False
