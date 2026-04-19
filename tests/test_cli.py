from pathlib import Path

from aster.ui_or_cli import cli
from aster.session import SessionStore


class _FakeDoctorReport:
    def __init__(self, ok: bool, payload: str) -> None:
        self.ok = ok
        self._payload = payload

    def to_json(self) -> str:
        return self._payload


def test_sync_runtime_command_uses_runtime_log_sync(monkeypatch, capsys, tmp_path: Path) -> None:
    calls: dict[str, object] = {}

    class _FakeOrchestrator:
        def __init__(self, config) -> None:
            calls["config"] = config

        def sync_runtime_logs(self, message: str) -> list[str]:
            calls["message"] = message
            return ["commit runtime", "push runtime"]

    sentinel_config = object()
    monkeypatch.setattr(cli, "load_config", lambda project_root: sentinel_config)
    monkeypatch.setattr(cli, "AsterOrchestrator", _FakeOrchestrator)

    exit_code = cli.main(
        [
            "sync-runtime",
            "--project-root",
            str(tmp_path),
            "--message",
            "runtime sync",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert calls["config"] is sentinel_config
    assert calls["message"] == "runtime sync"
    assert "commit runtime" in captured.out
    assert "push runtime" in captured.out


def test_doctor_json_prints_report_and_returns_nonzero_for_failures(monkeypatch, capsys, tmp_path: Path) -> None:
    report = _FakeDoctorReport(ok=False, payload='{"ok": false}')
    monkeypatch.setattr(cli, "run_doctor", lambda project_root: report)

    exit_code = cli.main(["doctor", "--project-root", str(tmp_path), "--json"])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out.strip() == '{"ok": false}'


def test_session_store_ignores_corrupt_history_file(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / ".aster")
    store.file.write_text("{not valid json", encoding="utf-8")

    assert store.load() == []
