from pathlib import Path

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
