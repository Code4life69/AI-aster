from pathlib import Path

from aster.config import load_config


def test_default_config_uses_browser_mode(tmp_path: Path) -> None:
    config = load_config(tmp_path)
    assert config.default_mode == "browser"
    assert config.browser_mode_enabled is True
    assert config.api_mode_enabled is False
    assert config.approval_required_for_commands is True
    assert config.push_runtime_logs_after_plan is True
    assert ".aster/" in config.ignore_patterns
