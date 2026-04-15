from __future__ import annotations

import shutil
import subprocess
from datetime import datetime
from pathlib import Path

from aster.audit_logger import AuditLogger
from aster.response_parser import ParsedPlan, PatchOperation


class PatchApplier:
    def __init__(self, project_root: Path, logger: AuditLogger, git_integration: bool = True) -> None:
        self.project_root = project_root.resolve()
        self.logger = logger
        self.git_integration = git_integration

    def apply(self, plan: ParsedPlan, selected_indices: list[int] | None = None, dry_run: bool = True) -> list[str]:
        indices = selected_indices or list(range(1, len(plan.operations) + 1))
        selected = [plan.operations[i - 1] for i in indices]
        backup_root = self._create_backup(selected, dry_run=dry_run)
        results = [f"Backup: {backup_root}"]
        if not dry_run:
            self._maybe_git_checkpoint()
        for op in selected:
            results.append(self._apply_operation(op, dry_run=dry_run))
        return results

    def _create_backup(self, operations: list[PatchOperation], dry_run: bool) -> Path:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        backup_root = self.project_root / ".aster" / "backups" / stamp
        if dry_run:
            return backup_root
        backup_root.mkdir(parents=True, exist_ok=True)
        for op in operations:
            source = self.project_root / op.path
            if source.exists() and source.is_file():
                target = backup_root / op.path
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
        self.logger.log("backup_created", {"backup_root": backup_root, "count": len(operations)})
        return backup_root

    def _maybe_git_checkpoint(self) -> None:
        if not self.git_integration or not (self.project_root / ".git").exists():
            return
        subprocess.run(["git", "add", "-A"], cwd=self.project_root, check=False, capture_output=True, text=True)
        subprocess.run(
            ["git", "commit", "-m", "Aster checkpoint before apply"],
            cwd=self.project_root,
            check=False,
            capture_output=True,
            text=True,
        )

    def _apply_operation(self, op: PatchOperation, dry_run: bool) -> str:
        self.logger.log("apply_operation", {"type": op.type, "path": op.path, "dry_run": dry_run})
        if dry_run:
            return f"DRY RUN: {op.type} {op.path}"
        path = self.project_root / op.path
        if op.type in {"CREATE FILE", "REPLACE FILE", "EDIT FILE"}:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(op.content, encoding="utf-8")
            return f"Wrote {op.path}"
        if op.type in {"RENAME FILE", "MOVE FILE"}:
            target = self.project_root / op.new_path
            target.parent.mkdir(parents=True, exist_ok=True)
            path.replace(target)
            return f"Moved {op.path} -> {op.new_path}"
        if op.type == "DELETE FILE":
            if path.exists():
                path.unlink()
            return f"Deleted {op.path}"
        if op.type == "INSTALL DEPENDENCIES":
            return self._run_commands(op.packages)
        if op.type == "RUN COMMANDS":
            return self._run_commands(op.commands)
        if op.type == "NEED THESE FILES FIRST":
            return f"Needs more files: {op.path}"
        raise ValueError(f"Unsupported operation type: {op.type}")

    def _run_commands(self, commands: list[str]) -> str:
        results = []
        for command in commands:
            completed = subprocess.run(
                command,
                cwd=self.project_root,
                shell=True,
                check=False,
                capture_output=True,
                text=True,
            )
            results.append(f"{command} -> {completed.returncode}")
        return "\n".join(results)
