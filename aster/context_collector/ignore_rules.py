from __future__ import annotations

from fnmatch import fnmatch
from pathlib import Path


def should_ignore(path: Path, root: Path, patterns: list[str]) -> bool:
    rel = path.relative_to(root).as_posix()
    for pattern in patterns:
        normalized = pattern.replace("\\", "/")
        if normalized.endswith("/") and rel.startswith(normalized[:-1]):
            return True
        if fnmatch(rel, normalized) or fnmatch(path.name, normalized):
            return True
    return False
