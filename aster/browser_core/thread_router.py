from __future__ import annotations

import json
from pathlib import Path

from .models import ThreadRecord


class ThreadRegistry:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._threads: dict[str, ThreadRecord] = {}

    def load(self) -> dict[str, ThreadRecord]:
        if not self.path.exists():
            self._threads = {}
            return self._threads
        data = json.loads(self.path.read_text(encoding="utf-8"))
        threads = data.get("threads", [])
        records: dict[str, ThreadRecord] = {}
        for item in threads:
            if not isinstance(item, dict):
                continue
            thread_id = str(item.get("thread_id", "")).strip()
            if not thread_id:
                continue
            records[thread_id] = ThreadRecord(
                thread_id=thread_id,
                title=str(item.get("title", "")),
                url=str(item.get("url", "")),
                strategy_hint=str(item.get("strategy_hint", "")),
                last_used_ts=str(item.get("last_used_ts", "")),
                metadata={str(key): str(value) for key, value in dict(item.get("metadata", {})).items()},
            )
        self._threads = records
        return self._threads

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "threads": [
                {
                    "thread_id": record.thread_id,
                    "title": record.title,
                    "url": record.url,
                    "strategy_hint": record.strategy_hint,
                    "last_used_ts": record.last_used_ts,
                    "metadata": record.metadata,
                }
                for record in sorted(self._threads.values(), key=lambda item: item.thread_id)
            ]
        }
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def get_thread(self, thread_id: str) -> ThreadRecord | None:
        return self._threads.get(thread_id)

    def upsert_thread(self, record: ThreadRecord) -> ThreadRecord:
        self._threads[record.thread_id] = record
        return record

    def choose_strategy(self, *, reuse_enabled: bool, thread_id: str | None = None) -> str:
        if not reuse_enabled:
            return "create_new"
        if thread_id and thread_id in self._threads:
            return "reuse_existing"
        return "unknown"
