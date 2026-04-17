from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .ignore_rules import should_ignore
from .secret_filter import redact_secrets, seems_binary


MANIFEST_FILES = {
    "pyproject.toml",
    "package.json",
    "package-lock.json",
    "requirements.txt",
    "Cargo.toml",
    "go.mod",
    "pom.xml",
    "build.gradle",
    "README.md",
    "README.txt",
    ".gitignore",
    "Dockerfile",
    "docker-compose.yml",
    "Makefile",
}
CONFIG_SUFFIXES = {".json", ".toml", ".yaml", ".yml", ".ini", ".cfg"}
SOURCE_SUFFIXES = {
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".java",
    ".cs",
    ".cpp",
    ".c",
    ".h",
    ".hpp",
    ".rs",
    ".go",
    ".rb",
    ".php",
    ".html",
    ".css",
    ".scss",
    ".md",
    ".txt",
    ".sql",
    ".sh",
    ".ps1",
}
LOG_HINTS = ("error", "trace", "stack", "stderr", "stdout", "log")
DEBUG_GOAL_HINTS = ("fail", "fails", "failure", "error", "trace", "crash", "debug", "exception", "bug", "broken")
FEATURE_GOAL_HINTS = ("build", "create", "add", "implement", "feature", "refactor", "rename", "move", "edit", "update")


@dataclass(slots=True)
class ContextFile:
    path: str
    reason: str
    content: str


@dataclass(slots=True)
class CollectedContext:
    project_root: Path
    project_summary: str
    file_tree: str
    relevant_files: list[ContextFile]
    skipped_files: list[str]


class ContextCollector:
    def __init__(self, ignore_patterns: list[str], max_file_bytes: int, max_total_prompt_bytes: int) -> None:
        self.ignore_patterns = ignore_patterns
        self.max_file_bytes = max_file_bytes
        self.max_total_prompt_bytes = max_total_prompt_bytes

    def collect(self, project_root: Path, user_goal: str) -> CollectedContext:
        root = project_root.resolve()
        all_files = sorted(path for path in root.rglob("*") if path.is_file())
        visible_files = [path for path in all_files if not should_ignore(path, root, self.ignore_patterns)]
        tree = self._build_tree(root, visible_files)
        scored = sorted(
            ((self._score_file(path, user_goal), path) for path in visible_files),
            key=lambda item: (-item[0], str(item[1])),
        )
        included: list[ContextFile] = []
        skipped: list[str] = []
        total = len(tree.encode("utf-8"))
        for score, path in scored:
            if score <= 0:
                continue
            raw = self._read_text(path)
            if raw is None:
                skipped.append(_root_relative(path, root))
                continue
            redacted = redact_secrets(raw)
            size = len(redacted.encode("utf-8"))
            if size > self.max_file_bytes:
                redacted = redacted[: self.max_file_bytes] + "\n...[TRUNCATED]..."
                size = len(redacted.encode("utf-8"))
            if total + size > self.max_total_prompt_bytes:
                skipped.append(_root_relative(path, root))
                continue
            included.append(ContextFile(path=_root_relative(path, root), reason=self._reason(path), content=redacted))
            total += size
        return CollectedContext(
            project_root=root,
            project_summary=self._summarize_project(root, visible_files),
            file_tree=tree,
            relevant_files=included,
            skipped_files=skipped,
        )

    def _build_tree(self, root: Path, files: list[Path]) -> str:
        lines = [root.name + "/"]
        for path in files:
            rel = path.relative_to(root)
            indent = "  " * (len(rel.parts) - 1)
            lines.append(f"{indent}- {rel.as_posix()}")
        return "\n".join(lines)

    def _score_file(self, path: Path, user_goal: str) -> int:
        name = path.name.lower()
        suffix = path.suffix.lower()
        score = 0
        goal_text = user_goal.lower()
        words = {word for word in goal_text.replace("-", " ").split() if len(word) > 2}
        debug_goal = any(hint in goal_text for hint in DEBUG_GOAL_HINTS)
        feature_goal = any(hint in goal_text for hint in FEATURE_GOAL_HINTS)
        rel_text = path.as_posix().lower()
        if name in MANIFEST_FILES:
            score += 120
        if suffix in CONFIG_SUFFIXES:
            score += 40
        if suffix in SOURCE_SUFFIXES:
            score += 60
            if feature_goal:
                score += 30
        if any(hint in name for hint in LOG_HINTS):
            score += 70 if debug_goal else 8
        if "test" in rel_text:
            score += 30 if feature_goal else 20
        score += sum(25 for word in words if word in rel_text)
        return score

    def _read_text(self, path: Path) -> str | None:
        try:
            data = path.read_bytes()
        except OSError:
            return None
        if seems_binary(data):
            return None
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError:
            return data.decode("utf-8", errors="ignore")

    def _summarize_project(self, root: Path, files: list[Path]) -> str:
        suffix_counts: dict[str, int] = {}
        for path in files:
            suffix = path.suffix.lower() or "[no_ext]"
            suffix_counts[suffix] = suffix_counts.get(suffix, 0) + 1
        top = ", ".join(f"{suffix}:{count}" for suffix, count in sorted(suffix_counts.items(), key=lambda item: (-item[1], item[0]))[:8])
        return f"Project root: {root}. Visible files: {len(files)}. Dominant file types: {top}."

    def _reason(self, path: Path) -> str:
        name = path.name
        if name in MANIFEST_FILES:
            return "manifest_or_readme"
        if any(token in name.lower() for token in LOG_HINTS):
            return "logs_or_runtime_output"
        if path.suffix.lower() in CONFIG_SUFFIXES:
            return "configuration"
        return "source_or_related_file"


def _root_relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()
