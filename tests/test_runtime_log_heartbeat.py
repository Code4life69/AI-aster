from aster.runtime_log_heartbeat import RuntimeLogHeartbeatPusher


class _Clock:
    def __init__(self) -> None:
        self.value = 0.0

    def monotonic(self) -> float:
        return self.value


def test_runtime_log_heartbeat_noops_when_disabled() -> None:
    clock = _Clock()
    calls: list[str] = []
    events: list[tuple[str, dict[str, object]]] = []
    heartbeat = RuntimeLogHeartbeatPusher(
        tracked_paths=(".aster/audit.log.jsonl",),
        enabled=False,
        interval_seconds=30,
        branch_only=True,
        sync_runtime_logs=lambda message: calls.append(message) or ["ok"],
        current_branch=lambda: "feature/debug",
        log_event=lambda event, payload: events.append((event, payload)),
        monotonic=clock.monotonic,
    )

    heartbeat.start_run()
    clock.value = 31.0
    results = heartbeat.tick(reason="wait_for_chatgpt_ready")

    assert results == []
    assert calls == []
    assert events == []


def test_runtime_log_heartbeat_noops_before_interval() -> None:
    clock = _Clock()
    calls: list[str] = []
    heartbeat = RuntimeLogHeartbeatPusher(
        tracked_paths=(".aster/audit.log.jsonl",),
        enabled=True,
        interval_seconds=30,
        branch_only=True,
        sync_runtime_logs=lambda message: calls.append(message) or ["ok"],
        current_branch=lambda: "feature/debug",
        log_event=lambda event, payload: None,
        monotonic=clock.monotonic,
    )

    heartbeat.start_run()
    clock.value = 12.0
    results = heartbeat.tick(reason="send_loop")

    assert results == []
    assert calls == []


def test_runtime_log_heartbeat_calls_runtime_sync_when_due() -> None:
    clock = _Clock()
    calls: list[str] = []
    events: list[tuple[str, dict[str, object]]] = []
    heartbeat = RuntimeLogHeartbeatPusher(
        tracked_paths=(".aster/audit.log.jsonl", ".aster/last_launch_stdout.log"),
        enabled=True,
        interval_seconds=30,
        branch_only=True,
        sync_runtime_logs=lambda message: calls.append(message) or ["commit", "push"],
        current_branch=lambda: "feature/debug",
        log_event=lambda event, payload: events.append((event, payload)),
        monotonic=clock.monotonic,
    )

    heartbeat.start_run()
    clock.value = 31.0
    results = heartbeat.tick(reason="capture_reply_text")

    assert calls == ["debug: sync runtime logs during active browser run"]
    assert results == ["commit", "push"]
    assert [item[0] for item in events] == [
        "runtime_log_heartbeat_due",
        "runtime_log_heartbeat_sync_start",
        "runtime_log_heartbeat_sync_success",
    ]


def test_runtime_log_heartbeat_respects_branch_only_protection() -> None:
    clock = _Clock()
    calls: list[str] = []
    heartbeat = RuntimeLogHeartbeatPusher(
        tracked_paths=(".aster/audit.log.jsonl",),
        enabled=True,
        interval_seconds=30,
        branch_only=True,
        sync_runtime_logs=lambda message: calls.append(message) or ["commit"],
        current_branch=lambda: "main",
        log_event=lambda event, payload: None,
        monotonic=clock.monotonic,
    )

    heartbeat.start_run()
    clock.value = 45.0
    results = heartbeat.tick(reason="wait_for_reply_start")

    assert results == []
    assert calls == []


def test_runtime_log_heartbeat_failure_does_not_crash_active_run() -> None:
    clock = _Clock()
    events: list[tuple[str, dict[str, object]]] = []

    def _boom(message: str) -> list[str]:
        raise RuntimeError(f"bad sync: {message}")

    heartbeat = RuntimeLogHeartbeatPusher(
        tracked_paths=(".aster/audit.log.jsonl",),
        enabled=True,
        interval_seconds=30,
        branch_only=False,
        sync_runtime_logs=_boom,
        current_branch=lambda: "feature/debug",
        log_event=lambda event, payload: events.append((event, payload)),
        monotonic=clock.monotonic,
    )

    heartbeat.start_run()
    clock.value = 31.0
    results = heartbeat.tick(reason="wait_for_prompt_inserted")

    assert results == []
    assert [item[0] for item in events][-1] == "runtime_log_heartbeat_sync_failed"
