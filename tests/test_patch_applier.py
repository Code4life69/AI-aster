from pathlib import Path

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
