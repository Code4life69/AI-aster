from pathlib import Path
import subprocess

import pytest

from aster.audit_logger import AuditLogger
from aster.patch_executor import PatchApplier
from aster.response_parser import ParsedPlan, PatchOperation


def test_patch_applier_writes_file(tmp_path: Path) -> None:
    logger = AuditLogger(tmp_path / ".aster")
    applier = PatchApplier(tmp_path, logger, git_integration=False)
    plan = ParsedPlan(
        summary="Create file",
        notes=[],
        operations=[
            PatchOperation(
                type="CREATE FILE",
                path="src/main.py",
                reason="bootstrap",
                content="print('ok')",
            )
        ],
    )

    results = applier.apply(plan, dry_run=False)

    assert (tmp_path / "src" / "main.py").read_text(encoding="utf-8") == "print('ok')"
    assert any("Wrote src/main.py" in item for item in results)


def test_patch_applier_rejects_shell_operators(tmp_path: Path) -> None:
    logger = AuditLogger(tmp_path / ".aster")
    applier = PatchApplier(tmp_path, logger, git_integration=False)

    with pytest.raises(ValueError, match="Shell operators are not allowed"):
        applier._run_commands(["pytest && git status"])


def test_patch_applier_rejects_inline_python_execution(tmp_path: Path) -> None:
    logger = AuditLogger(tmp_path / ".aster")
    applier = PatchApplier(tmp_path, logger, git_integration=False)

    with pytest.raises(ValueError, match="restricted to approved module execution"):
        applier._run_commands(['python -c "print(1)"'])


def test_patch_applier_allows_read_only_git_status(tmp_path: Path) -> None:
    logger = AuditLogger(tmp_path / ".aster")
    applier = PatchApplier(tmp_path, logger, git_integration=False)

    assert applier._parse_command("git status --short") == ["git", "status", "--short"]


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
    )


def test_patch_applier_checkpoint_only_stages_selected_files(tmp_path: Path) -> None:
    _git(tmp_path, "init")
    _git(tmp_path, "config", "user.name", "Aster Test")
    _git(tmp_path, "config", "user.email", "aster@example.com")
    (tmp_path / "app.py").write_text("print('v1')\n", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("keep me dirty\n", encoding="utf-8")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-m", "initial")

    (tmp_path / "app.py").write_text("print('pending pre-apply state')\n", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("still unrelated\n", encoding="utf-8")

    logger = AuditLogger(tmp_path / ".aster")
    applier = PatchApplier(tmp_path, logger, git_integration=True)
    plan = ParsedPlan(
        summary="Edit app only",
        notes=[],
        operations=[
            PatchOperation(
                type="EDIT FILE",
                path="app.py",
                reason="fix",
                content="print('applied')\n",
            )
        ],
    )

    applier.apply(plan, dry_run=False)

    head = _git(tmp_path, "show", "--name-only", "--pretty=format:", "HEAD")
    assert "app.py" in head.stdout
    assert "notes.txt" not in head.stdout
    status = _git(tmp_path, "status", "--porcelain")
    assert " M app.py" in status.stdout
    assert " M notes.txt" in status.stdout
