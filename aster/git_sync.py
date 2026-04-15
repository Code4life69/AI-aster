from __future__ import annotations

import subprocess
from pathlib import Path


class GitSync:
    def __init__(self, project_root: Path, remote_name: str = "origin") -> None:
        self.project_root = project_root
        self.remote_name = remote_name

    def is_repo(self) -> bool:
        return (self.project_root / ".git").exists()

    def sync_pull(self) -> list[str]:
        if not self.is_repo():
            return ["Git sync skipped: repository is not initialized locally."]
        branch = self.current_branch()
        if not branch:
            return ["Git sync skipped: no active branch yet."]
        return [
            self._run(["git", "fetch", self.remote_name]),
            self._run(["git", "pull", "--ff-only", self.remote_name, branch]),
        ]

    def sync_push(self) -> list[str]:
        if not self.is_repo():
            return ["Git push skipped: repository is not initialized locally."]
        branch = self.current_branch()
        if not branch:
            return ["Git push skipped: no active branch yet."]
        upstream = self._completed(["git", "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"])
        if upstream.returncode != 0:
            return [self._run(["git", "push", "-u", self.remote_name, branch])]
        return [self._run(["git", "push", self.remote_name, branch])]

    def set_remote(self, url: str) -> str:
        if not self.is_repo():
            self._run(["git", "init"])
        remotes = self._run(["git", "remote"]).splitlines()
        if self.remote_name in remotes:
            return self._run(["git", "remote", "set-url", self.remote_name, url])
        return self._run(["git", "remote", "add", self.remote_name, url])

    def current_branch(self) -> str:
        if not self.is_repo():
            return ""
        result = self._completed(["git", "branch", "--show-current"])
        if result.returncode != 0:
            return ""
        return result.stdout.strip()

    def commit_all_if_needed(self, message: str) -> list[str]:
        if not self.is_repo():
            return ["Git commit skipped: repository is not initialized locally."]
        if not self.has_changes():
            return ["Git commit skipped: no tracked or untracked changes detected."]
        results = [
            self._run(["git", "add", "-A"]),
            self._run(["git", "commit", "-m", message]),
        ]
        return results

    def has_changes(self) -> bool:
        if not self.is_repo():
            return False
        result = self._completed(["git", "status", "--porcelain"])
        if result.returncode != 0:
            return False
        return bool(result.stdout.strip())

    def ensure_branch(self, branch: str) -> str:
        if not self.is_repo():
            return "Git branch skipped: repository is not initialized locally."
        current = self.current_branch()
        if current == branch:
            return f"Git branch ready: {branch}"
        return self._run(["git", "branch", "-M", branch])

    def _run(self, command: list[str]) -> str:
        completed = self._completed(command)
        output = (completed.stdout or completed.stderr).strip()
        return f"{' '.join(command)} -> {completed.returncode}: {output}"

    def _completed(self, command: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            command,
            cwd=self.project_root,
            check=False,
            capture_output=True,
            text=True,
        )
