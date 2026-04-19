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
  "max_recovery_attempts": 5
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
