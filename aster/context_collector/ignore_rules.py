from __future__ import annotations

from fnmatch import fnmatch
from pathlib import Path


def should_ignore(path: Path, root: Path, patterns: list[str]) -> bool:
    rel = path.relative_to(root).as_posix()
    parts = rel.split("/")
    parent_parts = parts[:-1]
    for pattern in patterns:
        normalized = pattern.replace("\\", "/")
        if normalized.endswith("/"):
            directory_pattern = normalized[:-1]
            if directory_pattern in parent_parts:
                return True
            prefixes = ["/".join(parts[: index + 1]) for index in range(len(parent_parts))]
            if any(fnmatch(prefix, directory_pattern) for prefix in prefixes):
                return True
        if fnmatch(rel, normalized) or fnmatch(path.name, normalized):
            return True
    return False
