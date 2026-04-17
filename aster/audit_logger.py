from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class AuditLogger:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.file = self.root / "audit.log.jsonl"

    def log(self, kind: str, payload: dict[str, Any]) -> None:
        event = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "kind": kind,
            "payload": _normalize(payload),
        }
        with self.file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=True) + "\n")

    def activity(
        self,
        step: str,
        message: str,
        why: str = "",
        *,
        status: str = "info",
        details: dict[str, Any] | None = None,
    ) -> None:
        payload: dict[str, Any] = {
            "step": step,
            "message": message,
            "why": why,
            "status": status,
        }
        if details:
            payload["details"] = details
        self.log("activity", payload)


def _normalize(value: Any) -> Any:
    if is_dataclass(value):
        return {key: _normalize(item) for key, item in asdict(value).items()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _normalize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalize(item) for item in value]
    return value
