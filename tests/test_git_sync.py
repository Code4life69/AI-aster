from pathlib import Path
import subprocess

from aster.git_sync import GitSync


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
    )


def test_commit_all_if_needed_creates_commit(tmp_path: Path) -> None:
    _git(tmp_path, "init")
    _git(tmp_path, "config", "user.name", "Aster Test")
    _git(tmp_path, "config", "user.email", "aster@example.com")
    (tmp_path / "file.txt").write_text("hello", encoding="utf-8")

    sync = GitSync(tmp_path)
    results = sync.commit_all_if_needed("test commit")

    assert any("git add -A -> 0" in item for item in results)
    assert any("git commit -m test commit -> 0" in item for item in results)
    log = _git(tmp_path, "log", "--oneline")
    assert "test commit" in log.stdout


def test_has_changes_detects_clean_and_dirty_repo(tmp_path: Path) -> None:
    _git(tmp_path, "init")
    _git(tmp_path, "config", "user.name", "Aster Test")
    _git(tmp_path, "config", "user.email", "aster@example.com")
    sync = GitSync(tmp_path)

    assert sync.has_changes() is False

    (tmp_path / "new.txt").write_text("x", encoding="utf-8")
    assert sync.has_changes() is True


def test_ensure_branch_does_not_rename_current_branch(tmp_path: Path) -> None:
    _git(tmp_path, "init")
    _git(tmp_path, "checkout", "-b", "feature/demo")
    sync = GitSync(tmp_path)

    message = sync.ensure_branch("main")

    assert "Git branch preserved" in message
    assert sync.current_branch() == "feature/demo"


def test_commit_paths_if_needed_only_commits_selected_paths(tmp_path: Path) -> None:
    _git(tmp_path, "init")
    _git(tmp_path, "config", "user.name", "Aster Test")
    _git(tmp_path, "config", "user.email", "aster@example.com")
    audit_dir = tmp_path / ".aster"
    audit_dir.mkdir()
    (audit_dir / "audit.log.jsonl").write_text('{"kind":"seed"}\n', encoding="utf-8")
    (tmp_path / "app.py").write_text("print('v1')\n", encoding="utf-8")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-m", "initial")

    (audit_dir / "audit.log.jsonl").write_text('{"kind":"next"}\n', encoding="utf-8")
    (tmp_path / "app.py").write_text("print('v2')\n", encoding="utf-8")

    sync = GitSync(tmp_path)
    results = sync.commit_paths_if_needed(
        "runtime only",
        [
            ".aster/audit.log.jsonl",
            ".aster/last_launch_stdout.log",
            ".aster/last_launch_stderr.log",
        ],
    )

    assert any("git add -A -- .aster/audit.log.jsonl" in item for item in results)
    assert any("git commit -m runtime only -> 0" in item for item in results)
    head_files = _git(tmp_path, "show", "--name-only", "--pretty=format:", "HEAD")
    assert ".aster/audit.log.jsonl" in head_files.stdout
    assert "app.py" not in head_files.stdout
    status = _git(tmp_path, "status", "--porcelain")
    assert " M app.py" in status.stdout
