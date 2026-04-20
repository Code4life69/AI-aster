from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable


def _is_protected_branch(branch: str) -> bool:
    return branch.strip().lower() in {"", "main", "master"}


@dataclass(slots=True)
class RuntimeLogHeartbeatPusher:
    tracked_paths: tuple[str, ...]
    enabled: bool
    interval_seconds: int
    branch_only: bool
    sync_runtime_logs: Callable[[str], list[str]]
    current_branch: Callable[[], str]
    log_event: Callable[[str, dict[str, Any]], None]
    monotonic: Callable[[], float] = time.monotonic
    commit_message: str = "debug: sync runtime logs during active browser run"
    _run_active: bool = field(init=False, default=False)
    _last_sync_at: float | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        self.interval_seconds = max(1, int(self.interval_seconds))

    def start_run(self) -> None:
        self._run_active = True
        self._last_sync_at = self.monotonic()

    def stop_run(self) -> None:
        self._run_active = False
        self._last_sync_at = None

    def tick(self, *, reason: str) -> list[str]:
        if not self.enabled or not self._run_active:
            return []
        branch = self.current_branch().strip()
        if self.branch_only and _is_protected_branch(branch):
            return []
        now = self.monotonic()
        if self._last_sync_at is None:
            self._last_sync_at = now
            return []
        elapsed = now - self._last_sync_at
        if elapsed < self.interval_seconds:
            return []
        payload = {
            "reason": reason,
            "elapsed_seconds": round(elapsed, 1),
            "interval_seconds": self.interval_seconds,
            "branch": branch,
            "tracked_paths": list(self.tracked_paths),
        }
        self.log_event("runtime_log_heartbeat_due", payload)
        self._last_sync_at = now
        self.log_event("runtime_log_heartbeat_sync_start", payload)
        try:
            results = self.sync_runtime_logs(self.commit_message)
        except Exception as exc:
            self.log_event(
                "runtime_log_heartbeat_sync_failed",
                {
                    **payload,
                    "error": str(exc),
                },
            )
            return []
        self.log_event(
            "runtime_log_heartbeat_sync_success",
            {
                **payload,
                "result_count": len(results),
                "result_preview": list(results[:3]),
            },
        )
        return results
