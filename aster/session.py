from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class SessionTurn:
    role: str
    content: str


class SessionStore:
    def __init__(self, root: Path, limit: int = 8) -> None:
        self.file = root / "session_history.json"
        self.limit = limit
        root.mkdir(parents=True, exist_ok=True)

    def load(self) -> list[SessionTurn]:
        if not self.file.exists():
            return []
        data = json.loads(self.file.read_text(encoding="utf-8"))
        return [SessionTurn(role=item["role"], content=item["content"]) for item in data[-self.limit :]]

    def append(self, role: str, content: str) -> None:
        items = self.load()
        items.append(SessionTurn(role=role, content=content))
        payload = [{"role": item.role, "content": item.content} for item in items[-self.limit :]]
        self.file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
