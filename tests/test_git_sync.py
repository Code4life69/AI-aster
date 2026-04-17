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
