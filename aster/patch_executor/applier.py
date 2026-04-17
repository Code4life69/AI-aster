from __future__ import annotations

import shlex
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from aster.audit_logger import AuditLogger
from aster.response_parser import ParsedPlan, PatchOperation


ALLOWED_RUN_COMMANDS = {
    "git": {
        "status",
        "diff",
        "rev-parse",
        "branch",
        "log",
    },
    "pytest": None,
    "pytest.exe": None,
    "ruff": {"check"},
    "ruff.exe": {"check"},
    "mypy": None,
    "mypy.exe": None,
}
DISALLOWED_EXECUTABLES = {
    "bash",
    "bash.exe",
    "cmd",
    "cmd.exe",
    "powershell",
    "powershell.exe",
    "pwsh",
    "pwsh.exe",
    "sh",
    "sh.exe",
}
SHELL_META_TOKENS = {"&&", "||", ";", "|", ">", ">>", "<", "2>", "1>", "&"}


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
            return self._install_dependencies(op.packages)
        if op.type == "RUN COMMANDS":
            return self._run_commands(op.commands)
        if op.type == "NEED THESE FILES FIRST":
            return f"Needs more files: {op.path}"
        raise ValueError(f"Unsupported operation type: {op.type}")

    def _install_dependencies(self, packages: list[str]) -> str:
        cleaned = [package.strip() for package in packages if package.strip()]
        if not cleaned:
            raise ValueError("INSTALL DEPENDENCIES requires at least one package")
        completed = subprocess.run(
            [sys.executable, "-m", "pip", "install", *cleaned],
            cwd=self.project_root,
            check=False,
            capture_output=True,
            text=True,
        )
        return self._format_completed_process("python -m pip install", completed)

    def _run_commands(self, commands: list[str]) -> str:
        results = []
        for command in commands:
            argv = self._parse_command(command)
            completed = subprocess.run(
                argv,
                cwd=self.project_root,
                check=False,
                capture_output=True,
                text=True,
            )
            results.append(self._format_completed_process(" ".join(argv), completed))
        return "\n".join(results)

    def _parse_command(self, command: str) -> list[str]:
        raw = command.strip()
        if not raw:
            raise ValueError("RUN COMMANDS contains an empty command")
        if any(token in raw for token in SHELL_META_TOKENS):
            raise ValueError(f"Shell operators are not allowed in command operations: {command}")
        argv = shlex.split(raw, posix=False)
        if not argv:
            raise ValueError(f"Unable to parse command: {command}")
        executable = Path(argv[0].strip('"')).name.lower()
        if executable in DISALLOWED_EXECUTABLES:
            raise ValueError(f"Interactive shells are not allowed in command operations: {command}")
        if executable in {"python", "python.exe", "py"}:
            return self._validate_python_command(argv, original=command)
        allowed_subcommands = ALLOWED_RUN_COMMANDS.get(executable)
        if allowed_subcommands is None and executable not in ALLOWED_RUN_COMMANDS:
            raise ValueError(f"Command executable is not allowed: {command}")
        if allowed_subcommands is not None:
            if len(argv) < 2:
                raise ValueError(f"Command requires an allowed subcommand: {command}")
            if argv[1] not in allowed_subcommands:
                raise ValueError(f"Command subcommand is not allowed: {command}")
        return argv

    @staticmethod
    def _validate_python_command(argv: list[str], *, original: str) -> list[str]:
        if len(argv) < 3 or argv[1] != "-m":
            raise ValueError(f"Python command is restricted to approved module execution: {original}")
        module = argv[2]
        if module not in {"pytest", "unittest", "ruff", "mypy"}:
            raise ValueError(f"Python module is not allowed for command execution: {original}")
        if module == "ruff" and len(argv) >= 4 and argv[3] != "check":
            raise ValueError(f"ruff is restricted to the check subcommand: {original}")
        return argv

    @staticmethod
    def _format_completed_process(command_label: str, completed: subprocess.CompletedProcess[str]) -> str:
        output = (completed.stdout or completed.stderr or "").strip()
        preview = output[:240]
        if preview:
            return f"{command_label} -> {completed.returncode}: {preview}"
        return f"{command_label} -> {completed.returncode}"
