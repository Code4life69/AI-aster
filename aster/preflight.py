from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from aster.git_sync import GitSync


BROWSER_RUNTIME_MODULES = (
    "requests",
    "pyautogui",
    "pywinauto",
    "pyperclip",
    "screen_reader.automation",
    "screen_reader.capture",
    "screen_reader.ocr",
)


@dataclass(slots=True)
class DoctorCheck:
    name: str
    status: str
    detail: str

    @property
    def ok(self) -> bool:
        return self.status != "error"


@dataclass(slots=True)
class DoctorReport:
    project_root: str
    checks: list[DoctorCheck]

    @property
    def ok(self) -> bool:
        return all(item.ok for item in self.checks)

    def to_json(self) -> str:
        return json.dumps(
            {
                "project_root": self.project_root,
                "ok": self.ok,
                "checks": [asdict(item) for item in self.checks],
            },
            indent=2,
        )


def run_doctor(project_root: Path) -> DoctorReport:
    root = project_root.resolve()
    checks = [
        _python_check(),
        _git_check(),
        _project_root_check(root),
        _write_access_check(root),
        _browser_runtime_check(),
        _repo_state_check(root),
    ]
    return DoctorReport(project_root=str(root), checks=checks)


def _python_check() -> DoctorCheck:
    version = sys.version_info
    if version < (3, 11) or version >= (3, 13):
        return DoctorCheck(
            "python",
            "error",
            f"Unsupported Python version {version.major}.{version.minor}.{version.micro}; expected >=3.11,<3.13.",
        )
    return DoctorCheck("python", "ok", f"Python {version.major}.{version.minor}.{version.micro}")


def _git_check() -> DoctorCheck:
    git = shutil.which("git")
    if git is None:
        return DoctorCheck("git", "error", "Git was not found on PATH.")
    completed = subprocess.run(
        ["git", "--version"],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return DoctorCheck("git", "error", completed.stderr.strip() or "Git CLI is installed but not runnable.")
    return DoctorCheck("git", "ok", completed.stdout.strip())


def _project_root_check(root: Path) -> DoctorCheck:
    if not root.exists():
        return DoctorCheck("project_root", "error", f"Project root does not exist: {root}")
    if not root.is_dir():
        return DoctorCheck("project_root", "error", f"Project root is not a directory: {root}")
    return DoctorCheck("project_root", "ok", f"Project root is available: {root}")


def _write_access_check(root: Path) -> DoctorCheck:
    probe_dir = root / ".aster"
    probe_dir.mkdir(parents=True, exist_ok=True)
    probe_file = probe_dir / ".doctor-write-test"
    try:
        probe_file.write_text("ok", encoding="utf-8")
        probe_file.unlink()
    except OSError as exc:
        return DoctorCheck("write_access", "error", f"Unable to write under {probe_dir}: {exc}")
    return DoctorCheck("write_access", "ok", f"Write access confirmed under {probe_dir}")


def _browser_runtime_check() -> DoctorCheck:
    missing = _find_missing_modules(BROWSER_RUNTIME_MODULES)
    if missing:
        return DoctorCheck(
            "browser_runtime",
            "error",
            "Missing browser/runtime modules: " + ", ".join(missing),
        )
    return DoctorCheck("browser_runtime", "ok", "Browser runtime modules and Screen Reader imports are available.")


def _repo_state_check(root: Path) -> DoctorCheck:
    git = GitSync(root)
    if not git.is_repo():
        return DoctorCheck("repo_state", "warning", "Git repository is not initialized in this project yet.")
    branch = git.current_branch()
    if not branch:
        return DoctorCheck("repo_state", "warning", "Git repository exists but no active branch is checked out.")
    return DoctorCheck("repo_state", "ok", f"Repository is on branch {branch}.")


def _find_missing_modules(module_names: tuple[str, ...]) -> list[str]:
    sibling = Path("C:/Screen Reader")
    inserted = False
    if sibling.exists() and str(sibling) not in sys.path:
        sys.path.insert(0, str(sibling))
        inserted = True
    try:
        return [name for name in module_names if importlib.util.find_spec(name) is None]
    finally:
        if inserted:
            try:
                sys.path.remove(str(sibling))
            except ValueError:
                pass
