import subprocess

import pytest

from aster.browser_core.reply_tracker import (
    clean_captured_segment_for_policy,
    extract_structured_block,
    looks_like_patch_plan_json,
    merge_text_segments,
    reply_looks_incomplete,
    score_candidate_for_policy,
    segment_looks_like_prompt_echo_for_policy,
    should_ignore_candidate_for_policy,
)
from aster.transport_browser.browser_transport import (
    BrowserChatGPTTransport,
    REPLY_TRACKER_POLICY,
    _AutomationNotice,
    summarize_controls_near_composer_area,
    summarize_edit_candidates,
    summarize_named_button_candidates,
)
from aster.browser_core.turn_anchor import build_turn_anchor


def test_extract_structured_block_prefers_fenced_json() -> None:
    raw = """
    some OCR noise
    ```json
    {"summary":"ok","notes":[],"operations":[{"type":"NEED THESE FILES FIRST","path":"README.md","reason":"need context"}]}
    ```
    trailing text
    """
    parsed = extract_structured_block(raw)
    assert parsed.startswith("{")
    assert '"summary":"ok"' in parsed


def test_extract_structured_block_falls_back_to_outer_json() -> None:
    raw = 'noise {"summary":"ok","notes":[],"operations":[{"type":"RUN COMMANDS","path":".","reason":"verify","commands":["pytest"]}]} end'
    parsed = extract_structured_block(raw)
    assert parsed.startswith("{")
    assert parsed.endswith("}")


def test_extract_structured_block_prefers_aster_markers() -> None:
    raw = """
    random intro
    ASTER_PATCH_BEGIN
    {"summary":"ok","notes":[],"operations":[{"type":"CREATE FILE","path":"main.py","reason":"bootstrap","content":"print('ok')"}]}
    ASTER_PATCH_END
    random footer
    """
    parsed = extract_structured_block(raw)
    assert parsed.startswith("{")
    assert '"CREATE FILE"' in parsed


def test_summarize_controls_near_composer_area_limits_and_compacts() -> None:
    summary = summarize_controls_near_composer_area(
        [
            {"control_type": "Text", "name": "Helper text", "rect": (40, 620, 240, 650), "enabled": True},
            {"control_type": "Edit", "name": "Ask anything", "rect": (120, 680, 960, 760), "enabled": True},
            {"control_type": "Button", "name": "Send prompt", "rect": (980, 700, 1030, 744), "enabled": True},
        ],
        limit=2,
    )

    assert len(summary) == 2
    assert summary[0]["control_type"] == "Edit"
    assert summary[0]["rect"]["top"] == 680
    assert summary[1]["control_type"] == "Button"


def test_summarize_named_button_candidates_flags_send_like_variant() -> None:
    summary = summarize_named_button_candidates(
        [
            {
                "name": "Submit message",
                "rect": (980, 700, 1030, 744),
                "enabled": True,
                "match_score": 88.0,
                "exact_match": False,
                "looks_send_like": True,
            },
            {
                "name": "Show in text field",
                "rect": (860, 700, 970, 744),
                "enabled": True,
                "match_score": 12.0,
                "exact_match": False,
                "looks_send_like": False,
            },
        ]
    )

    assert summary["send_like_button_detected"] is True
    assert summary["send_like_button_with_different_label"] is True
    assert "Submit message" in summary["send_like_button_names"]


def test_summarize_edit_candidates_flags_below_threshold() -> None:
    summary = summarize_edit_candidates(
        [
            {
                "name": "Search address bar",
                "rect": (140, 80, 840, 120),
                "enabled": True,
                "score": -35.0,
            }
        ]
    )

    assert summary["composer_candidate_below_threshold"] is True
    assert summary["top_candidates"][0]["above_threshold"] is False


def test_read_visible_reply_text_survives_malformed_descendant_iteration(monkeypatch) -> None:
    logger = _FakeAuditLogger()
    transport = BrowserChatGPTTransport(logger=logger)

    class _BrokenWindow:
        def descendants(self):
            yield _FakeReplyControl(
                "Text",
                "Assistant reply",
                _FakeRect(280, 180, 980, 240),
            )
            raise KeyError(None)

    class _FakeDesktop:
        def __init__(self, backend=None) -> None:
            assert backend == "uia"

        def window(self, handle=None):
            assert handle == 123
            return _BrokenWindow()

    class _Target:
        handle = 123
        title = "ChatGPT - Google Chrome"
        left = 0
        top = 0
        width = 1200
        height = 900

    monkeypatch.setattr("aster.transport_browser.browser_transport.Desktop", _FakeDesktop)

    reply_text = transport._read_visible_reply_text(_Target())

    assert reply_text == "Assistant reply"
    failure = next(event for event in logger.events if event["event"] == "uia_reply_read_failure")
    assert failure["uia_read_failure_reason"] == "KeyError"
    assert failure["descendant_enumeration_guard_triggered"] is True
    assert failure["stage"] == "descendant_iteration"


class _FakeAuditLogger:
    def __init__(self) -> None:
        self.events: list[dict[str, object]] = []

    def log(self, namespace: str, payload: dict[str, object]) -> None:
        assert namespace == "browser_transport"
        self.events.append(payload)

    def activity(self, *args, **kwargs) -> None:
        return None


class _FakeRect:
    def __init__(self, left: int, top: int, right: int, bottom: int) -> None:
        self.left = left
        self.top = top
        self.right = right
        self.bottom = bottom


class _FakeElementInfo:
    def __init__(self, control_type: str) -> None:
        self.control_type = control_type


class _FakeReplyControl:
    def __init__(self, control_type: str, text: str, rect: _FakeRect) -> None:
        self.element_info = _FakeElementInfo(control_type)
        self._text = text
        self._rect = rect

    def window_text(self) -> str:
        return self._text

    def rectangle(self) -> _FakeRect:
        return self._rect


class _FakeNoticeProcess:
    _next_pid = 5000

    def __init__(
        self,
        *,
        running: bool = True,
        terminate_error: Exception | None = None,
        wait_timeout: bool = False,
    ) -> None:
        type(self)._next_pid += 1
        self.pid = type(self)._next_pid
        self.returncode = None if running else 0
        self.terminate_error = terminate_error
        self.wait_timeout = wait_timeout
        self.terminate_calls = 0
        self.kill_calls = 0

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self.terminate_calls += 1
        if self.terminate_error is not None:
            raise self.terminate_error
        if not self.wait_timeout:
            self.returncode = 0

    def wait(self, timeout: float | None = None) -> int | None:
        if self.wait_timeout and self.returncode is None:
            raise subprocess.TimeoutExpired("notice", timeout)
        return self.returncode

    def kill(self) -> None:
        self.kill_calls += 1
        self.returncode = -9


def _reset_notice_state() -> None:
    _AutomationNotice._active_process = None
    _AutomationNotice._active_owner_id = None


def _notice_events(logger: _FakeAuditLogger, event: str) -> list[dict[str, object]]:
    return [payload for payload in logger.events if payload.get("event") == event]


def test_automation_notice_script_watches_parent_pid() -> None:
    script = _AutomationNotice._build_script(4321)

    assert "PARENT_PID = 4321" in script
    assert "def _poll_parent()" in script
    assert "root.after(1000, _poll_parent)" in script


def test_automation_notice_show_cleans_stale_dead_process_before_launch(monkeypatch) -> None:
    _reset_notice_state()
    logger = _FakeAuditLogger()
    stale_process = _FakeNoticeProcess(running=False)
    launched_process = _FakeNoticeProcess(running=True)
    _AutomationNotice._active_process = stale_process
    _AutomationNotice._active_owner_id = 111
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: launched_process)
    monkeypatch.setattr("aster.transport_browser.browser_transport.time.sleep", lambda *_args, **_kwargs: None)

    notice = _AutomationNotice(logger)
    notice.show()

    assert _AutomationNotice._active_process is launched_process
    assert _notice_events(logger, "automation_notice_stale_detected")
    assert _notice_events(logger, "automation_notice_stale_cleaned_up")
    assert _notice_events(logger, "automation_notice_show")
    _reset_notice_state()


def test_automation_notice_show_replaces_running_notice_from_other_owner(monkeypatch) -> None:
    _reset_notice_state()
    logger = _FakeAuditLogger()
    original_process = _FakeNoticeProcess(running=True)
    replacement_process = _FakeNoticeProcess(running=True)
    first_notice = _AutomationNotice(logger)
    second_notice = _AutomationNotice(logger)
    _AutomationNotice._active_process = original_process
    _AutomationNotice._active_owner_id = first_notice._owner_id
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: replacement_process)
    monkeypatch.setattr("aster.transport_browser.browser_transport.time.sleep", lambda *_args, **_kwargs: None)

    second_notice.show()

    assert original_process.terminate_calls == 1
    assert _AutomationNotice._active_process is replacement_process
    assert _AutomationNotice._active_owner_id == second_notice._owner_id
    assert _notice_events(logger, "automation_notice_stale_detected")
    assert _notice_events(logger, "automation_notice_stale_cleaned_up")
    _reset_notice_state()


def test_automation_notice_show_is_noop_for_same_owner(monkeypatch) -> None:
    _reset_notice_state()
    logger = _FakeAuditLogger()
    active_process = _FakeNoticeProcess(running=True)
    notice = _AutomationNotice(logger)
    _AutomationNotice._active_process = active_process
    _AutomationNotice._active_owner_id = notice._owner_id

    def _unexpected_popen(*args, **kwargs):
        raise AssertionError("show() should not relaunch an already-visible notice for the same owner")

    monkeypatch.setattr(subprocess, "Popen", _unexpected_popen)
    notice.show()

    assert active_process.terminate_calls == 0
    assert _AutomationNotice._active_process is active_process
    assert not _notice_events(logger, "automation_notice_show")
    _reset_notice_state()


def test_automation_notice_hide_cleans_dead_process_handle() -> None:
    _reset_notice_state()
    logger = _FakeAuditLogger()
    notice = _AutomationNotice(logger)
    dead_process = _FakeNoticeProcess(running=False)
    _AutomationNotice._active_process = dead_process
    _AutomationNotice._active_owner_id = notice._owner_id

    notice.hide()

    assert _AutomationNotice._active_process is None
    assert _notice_events(logger, "automation_notice_stale_detected")
    assert _notice_events(logger, "automation_notice_stale_cleaned_up")
    _reset_notice_state()


def test_automation_notice_show_logs_launch_failure(monkeypatch) -> None:
    _reset_notice_state()
    logger = _FakeAuditLogger()
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("launch failed")))
    monkeypatch.setattr("aster.transport_browser.browser_transport.time.sleep", lambda *_args, **_kwargs: None)

    notice = _AutomationNotice(logger)
    notice.show()

    assert _notice_events(logger, "automation_notice_launch_failure")
    assert _notice_events(logger, "automation_notice_unavailable")
    assert _AutomationNotice._active_process is None
    _reset_notice_state()


def test_automation_notice_hide_logs_terminate_failure() -> None:
    _reset_notice_state()
    logger = _FakeAuditLogger()
    notice = _AutomationNotice(logger)
    stuck_process = _FakeNoticeProcess(running=True, terminate_error=PermissionError("access denied"))
    _AutomationNotice._active_process = stuck_process
    _AutomationNotice._active_owner_id = notice._owner_id

    notice.hide()

    assert _notice_events(logger, "automation_notice_terminate_failure")
    assert _AutomationNotice._active_process is stuck_process
    _reset_notice_state()


class _FakeExecutor:
    def __init__(self, reply: str) -> None:
        self.reply = reply

    def read_chatgpt_browser_reply(self, *args, **kwargs) -> str:
        return self.reply


class _FakeCaptureService:
    def capture_region(self, *args, **kwargs):
        return object()


class _FakeOCR:
    def extract(self, image):
        return []


class _MonotonicClock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        self.value += 1.0
        return self.value


class _Line:
    def __init__(self, text: str) -> None:
        self.text = text
        self.bbox = (0, 150, 400, 180)
        self.center = (200, 165)


def test_reply_detection_is_blocked_while_show_in_text_field_is_present() -> None:
    transport = BrowserChatGPTTransport()
    transport._executor = _FakeExecutor("Good to see you, Justin.")
    transport._capture = object()
    transport._ocr = object()
    transport._logger = None

    blocked = transport._reply_started(
        before_lines=[],
        after_lines=[],
        target=object(),
        ui_state={
            "show_in_text_field_present": True,
            "send_prompt_present": True,
            "send_prompt_enabled": False,
        },
    )

    assert blocked is False


def test_reply_detection_accepts_reply_when_composer_is_not_waiting_to_send() -> None:
    transport = BrowserChatGPTTransport()
    transport._executor = _FakeExecutor('{"summary":"ok","notes":[],"operations":[{"type":"NEED THESE FILES FIRST","path":"README.md","reason":"need context"}]}')
    transport._capture = object()
    transport._ocr = object()
    transport._logger = None

    started = transport._reply_started(
        before_lines=[],
        after_lines=[],
        target=object(),
        ui_state={
            "show_in_text_field_present": False,
            "send_prompt_present": False,
            "send_prompt_enabled": None,
        },
    )

    assert started is True


def test_reply_detection_accepts_code_like_reply_when_send_button_is_gone() -> None:
    transport = BrowserChatGPTTransport()
    transport._executor = _FakeExecutor(
        "def build_ui(self) -> None:\n"
        "    frame = tk.Frame(self.root)\n"
        "    frame.pack(fill='both', expand=True)\n"
        "    self.display = tk.Entry(frame)\n"
        "    self.display.grid(row=0, column=0, columnspan=4)\n"
    )
    transport._capture = object()
    transport._ocr = object()
    transport._logger = None

    started = transport._reply_started(
        before_lines=[],
        after_lines=[],
        target=object(),
        ui_state={
            "show_in_text_field_present": False,
            "send_prompt_present": False,
            "send_prompt_enabled": None,
            "stop_streaming_present": True,
        },
    )

    assert started is True


def test_reply_detection_accepts_streaming_state_even_without_text_candidate() -> None:
    transport = BrowserChatGPTTransport()
    transport._executor = _FakeExecutor("")
    transport._capture = object()
    transport._ocr = object()
    transport._logger = None

    started = transport._reply_started(
        before_lines=[],
        after_lines=[],
        target=object(),
        ui_state={
            "show_in_text_field_present": False,
            "send_prompt_present": False,
            "send_prompt_enabled": None,
            "stop_streaming_present": True,
        },
    )

    assert started is True


def test_reply_detection_accepts_idle_composer_reset_after_send_attempt_with_low_signal_candidate() -> None:
    transport = BrowserChatGPTTransport()
    transport._executor = _FakeExecutor(
        'f"type":"CREATE FILE","path":"calculator_gui/README.md","reason":"Document how to run the calculator"}'
    )
    transport._capture = object()
    transport._ocr = object()
    transport._logger = None

    started = transport._reply_started(
        before_lines=[_Line("Ask anything")],
        after_lines=[_Line("Calculator GUI Project"), _Line("Ask anything")],
        target=object(),
        ui_state={
            "show_in_text_field_present": False,
            "send_prompt_present": False,
            "send_prompt_enabled": None,
            "stop_streaming_present": False,
            "composer_edit_length": 12,
            "composer_edit_preview": "Ask anything",
        },
        send_attempted=True,
    )

    assert started is True


def test_reply_detection_does_not_accept_idle_composer_reset_without_send_attempt() -> None:
    transport = BrowserChatGPTTransport()
    transport._executor = _FakeExecutor(
        'f"type":"CREATE FILE","path":"calculator_gui/README.md","reason":"Document how to run the calculator"}'
    )
    transport._capture = object()
    transport._ocr = object()
    transport._logger = None

    started = transport._reply_started(
        before_lines=[_Line("Ask anything")],
        after_lines=[_Line("Calculator GUI Project"), _Line("Ask anything")],
        target=object(),
        ui_state={
            "show_in_text_field_present": False,
            "send_prompt_present": False,
            "send_prompt_enabled": None,
            "stop_streaming_present": False,
            "composer_edit_length": 12,
            "composer_edit_preview": "Ask anything",
        },
        send_attempted=False,
    )

    assert started is False


def test_capture_reply_text_timeout_preserves_best_structured_candidate(monkeypatch) -> None:
    transport = BrowserChatGPTTransport()
    transport._executor = _FakeExecutor("")
    transport._capture = _FakeCaptureService()
    transport._ocr = _FakeOCR()
    transport._logger = None
    transport._ui_state = lambda target: {
        "show_in_text_field_present": False,
        "send_prompt_present": False,
        "send_prompt_enabled": None,
        "stop_streaming_present": False,
    }
    transport._raise_for_browser_error = lambda lines, ui_state, stage: None
    replies = iter(
        [
            (
                "",
                "ASTER_PATCH_BEGIN\n"
                '{"summary":"ok","notes":[],"operations":[{"type":"CREATE FILE","path":"calculator.py","reason":"add","content":"print(1)"}]}\n'
                "ASTER_PATCH_END",
            ),
            (
                "self.root.resizable(False, False)\n"
                "self.display_var = tk.StringVar(value='0')\n"
                "for label in ('7', '8', '9', '/'):\n"
                "    ttk.Button(frame, text=label).grid(sticky='nsew')\n",
                "",
            ),
        ]
    )
    transport._capture_visible_reply_sources = lambda target, before_lines, prompt, lines=None: next(replies)
    monkeypatch.setattr("aster.transport_browser.browser_transport.time.sleep", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("aster.transport_browser.browser_transport.time.monotonic", _MonotonicClock())

    class _Target:
        left = 0
        top = 0
        width = 1200
        height = 900

    reply = transport._capture_reply_text(_Target(), before_lines=[], prompt="create calculator", timeout_sec=4.0)

    assert looks_like_patch_plan_json(reply) is True
    assert '"operations"' in extract_structured_block(reply)
    assert "calculator.py" in reply


def test_capture_reply_text_can_reject_selected_raw_code_candidate(monkeypatch) -> None:
    logger = _FakeAuditLogger()
    transport = BrowserChatGPTTransport(logger=logger)
    transport._executor = _FakeExecutor("")
    transport._capture = _FakeCaptureService()
    transport._ocr = _FakeOCR()
    transport._ui_state = lambda target: {
        "show_in_text_field_present": False,
        "send_prompt_present": False,
        "send_prompt_enabled": None,
        "stop_streaming_present": False,
    }
    transport._raise_for_browser_error = lambda lines, ui_state, stage: None
    replies = iter(
        [
            (
                "self.root.resizable(False, False)\n"
                "self.display_var = tk.StringVar(value='0')\n"
                "for label in ('7', '8', '9', '/'):\n"
                "    ttk.Button(frame, text=label).grid(sticky='nsew')\n",
                "",
            ),
            ("", ""),
        ]
    )
    transport._capture_visible_reply_sources = lambda target, before_lines, prompt, lines=None: next(replies)
    monkeypatch.setattr("aster.transport_browser.browser_transport.time.sleep", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("aster.transport_browser.browser_transport.time.monotonic", _MonotonicClock())

    class _Target:
        left = 0
        top = 0
        width = 1200
        height = 900

    reply = transport._capture_reply_text(_Target(), before_lines=[], prompt="create calculator", timeout_sec=4.0)

    assert reply == ""
    assert _notice_events(logger, "reply_candidate_selected")
    timeout_events = _notice_events(logger, "reply_wait_timeout")
    assert timeout_events
    assert timeout_events[-1]["acceptance_tier"] == "blocked_or_ambiguous"
    assert "Raw code appeared" in str(timeout_events[-1]["rejection_reason"])


def test_finalize_captured_reply_salvages_best_previous_structured_candidate() -> None:
    transport = BrowserChatGPTTransport()
    snapshot = transport._last_reply_capture_snapshot
    transport._remember_reply_capture_candidate(
        snapshot,
        source="uia",
        text=(
            "ASTER_PATCH_BEGIN\n"
            '{"summary":"ok","notes":[],"operations":[{"type":"CREATE FILE","path":"saved.py","reason":"add","content":"print(1)"}]}\n'
            "ASTER_PATCH_END"
        ),
        score=350.0,
        salvage_allowed=True,
        observation_counts={},
    )

    parsed, diagnostics = transport._finalize_captured_reply("")

    assert looks_like_patch_plan_json(parsed) is True
    assert '"saved.py"' in parsed
    assert diagnostics["salvage_attempted"] is True
    assert diagnostics["salvage_succeeded"] is True
    assert diagnostics["salvage_source"] == "uia"
    assert diagnostics["retry_seed_valid"] is True
    assert diagnostics["retry_seed_validity_reason"] == "parseable_structured_json"


def test_finalize_captured_reply_does_not_salvage_untrusted_scrolled_candidate() -> None:
    transport = BrowserChatGPTTransport()
    snapshot = transport._last_reply_capture_snapshot
    transport._remember_reply_capture_candidate(
        snapshot,
        source="scrolled",
        text='{"summary":"ok","notes":[],"operations":[{"type":"CREATE FILE","path":"unsafe.py","reason":"add","content":"print(1)"}]}',
        score=300.0,
        salvage_allowed=False,
        observation_counts={},
    )

    parsed, diagnostics = transport._finalize_captured_reply("")

    assert parsed == ""
    assert diagnostics["salvage_attempted"] is True
    assert diagnostics["salvage_succeeded"] is False
    assert diagnostics["final_capture_failure_reason"] == "structured_block_seen_but_not_salvageable"
    assert diagnostics["retry_seed_valid"] is False


def test_finalize_captured_reply_rejects_low_region_trust_candidate_for_salvage() -> None:
    transport = BrowserChatGPTTransport()
    snapshot = transport._last_reply_capture_snapshot
    transport._remember_reply_capture_candidate(
        snapshot,
        source="uia",
        text='{"summary":"ok","notes":[],"operations":[{"type":"CREATE FILE","path":"unsafe.py","reason":"add","content":"print(1)"}]}',
        score=320.0,
        salvage_allowed=True,
        observation_counts={},
        region_trusted=False,
        region_confidence=0.18,
        region_reason="low_confidence_visual_reply_region",
    )

    parsed, diagnostics = transport._finalize_captured_reply("")

    assert parsed == ""
    assert diagnostics["salvage_attempted"] is True
    assert diagnostics["salvage_succeeded"] is False
    assert diagnostics["best_structured_candidate_region_trusted"] is False
    assert diagnostics["final_capture_failure_reason"] == "structured_block_seen_but_region_trust_too_low"
    assert diagnostics["retry_seed_valid"] is False


def test_finalize_captured_reply_prefers_trusted_candidate_for_salvage() -> None:
    transport = BrowserChatGPTTransport()
    snapshot = transport._last_reply_capture_snapshot
    observation_counts: dict[str, int] = {}
    transport._remember_reply_capture_candidate(
        snapshot,
        source="uia",
        text='{"summary":"ok","notes":[],"operations":[{"type":"CREATE FILE","path":"unsafe.py","reason":"add","content":"print(1)"}]}',
        score=390.0,
        salvage_allowed=True,
        observation_counts=observation_counts,
        region_trusted=False,
        region_confidence=0.12,
        region_reason="weak_visual_reply_region_support",
    )
    transport._remember_reply_capture_candidate(
        snapshot,
        source="ocr",
        text='{"summary":"ok","notes":[],"operations":[{"type":"CREATE FILE","path":"safe.py","reason":"add","content":"print(2)"}]}',
        score=280.0,
        salvage_allowed=True,
        observation_counts=observation_counts,
        region_trusted=True,
        region_confidence=0.82,
        region_reason="confirmed_reply_region_capture",
    )

    parsed, diagnostics = transport._finalize_captured_reply("")

    assert '"safe.py"' in parsed
    assert diagnostics["salvage_succeeded"] is True
    assert diagnostics["salvage_source"] == "ocr"
    assert diagnostics["best_salvageable_candidate_region_trusted"] is True
    assert diagnostics["best_retry_safe_candidate_source"] == "ocr"


def test_finalize_captured_reply_preserves_partial_structured_candidate_for_diagnostics_only() -> None:
    transport = BrowserChatGPTTransport()
    snapshot = transport._last_reply_capture_snapshot
    transport._remember_reply_capture_candidate(
        snapshot,
        source="uia",
        text=(
            "ASTER_PATCH_BEGIN\n"
            '{"summary":"ok","notes":[],"operations":[{"type":"CREATE FILE","path":"saved.py","reason":"add","content":"print(1)"}]\n'
        ),
        score=350.0,
        salvage_allowed=True,
        observation_counts={},
    )

    parsed, diagnostics = transport._finalize_captured_reply("")

    assert parsed == ""
    assert diagnostics["salvage_attempted"] is True
    assert diagnostics["salvage_succeeded"] is False
    assert diagnostics["salvage_preserved_for_diagnostics_only"] is True
    assert diagnostics["retry_seed_valid"] is False
    assert diagnostics["retry_seed_validity_reason"] == "unbalanced_structure"
    assert diagnostics["final_capture_failure_reason"] == "structured_block_seen_but_not_retry_safe"


def test_finalize_captured_reply_retry_attempt_recovers_parseable_block_with_prose() -> None:
    transport = BrowserChatGPTTransport()

    parsed, diagnostics = transport._finalize_captured_reply(
        (
            "Here is the corrected result.\n"
            "ASTER_PATCH_BEGIN\n"
            '{"summary":"ok","notes":[],"operations":[{"type":"CREATE FILE","path":"saved.py","reason":"add","content":"print(1)"}]}\n'
            "ASTER_PATCH_END\n"
            "No other commentary is needed."
        ),
        retry_attempt=True,
    )

    assert parsed == '{"summary":"ok","notes":[],"operations":[{"type":"CREATE FILE","path":"saved.py","reason":"add","content":"print(1)"}]}'
    assert diagnostics["retry_attempt_capture_mode"] == "structured_block_first"
    assert diagnostics["retry_attempt_acceptance_tier"] == "retry_structured_embedded_block"
    assert diagnostics["retry_attempt_parseable"] is True
    assert diagnostics["retry_attempt_prose_contamination"] is True
    assert diagnostics["retry_attempt_wrapper_only"] is False
    assert diagnostics["retry_attempt_exact_block_only"] is False
    assert diagnostics["retry_attempt_extra_text_detected"] is True
    assert diagnostics["retry_attempt_failure_reason"] == ""
    assert diagnostics["retry_attempt_selected_block_index"] == 1
    assert diagnostics["retry_attempt_block_selection_reason"] == "single_parseable_embedded_retry_block"
    assert diagnostics["retry_attempt_prose_recovery_attempted"] is True
    assert diagnostics["retry_attempt_prose_recovery_succeeded"] is True
    assert diagnostics["retry_attempt_prose_recovery_reason"] == "single_parseable_embedded_block"


def test_finalize_captured_reply_retry_attempt_recovers_single_block_with_safe_json_cleanup() -> None:
    transport = BrowserChatGPTTransport()

    parsed, diagnostics = transport._finalize_captured_reply(
        (
            "ASTER_PATCH_BEGIN\n"
            '{\u201csummary\u201d:\u201cok\u201d,\u201cnotes\u201d:[],\u201coperations\u201d:[{\u201ctype\u201d:\u201cCREATE FILE\u201d,\u201cpath\u201d:\u201csaved.py\u201d,\u201creason\u201d:\u201cadd\u201d,\u201ccontent\u201d:\u201cprint(1)\u201d}],}\n'
            "ASTER_PATCH_END"
        ),
        retry_attempt=True,
    )

    assert parsed == '{"summary":"ok","notes":[],"operations":[{"type":"CREATE FILE","path":"saved.py","reason":"add","content":"print(1)"}]}'
    assert diagnostics["retry_attempt_failure_reason"] == ""
    assert diagnostics["retry_attempt_acceptance_tier"] == "retry_structured_json_cleanup"
    assert diagnostics["retry_attempt_json_cleanup_attempted"] is True
    assert diagnostics["retry_attempt_json_cleanup_succeeded"] is True
    assert diagnostics["retry_attempt_json_cleanup_changed"] is True
    assert "normalized_smart_quotes" in diagnostics["retry_attempt_json_cleanup_reason"]
    assert "removed_trailing_commas" in diagnostics["retry_attempt_json_cleanup_reason"]


def test_finalize_captured_reply_retry_attempt_recovers_safely_closable_single_block() -> None:
    transport = BrowserChatGPTTransport()

    parsed, diagnostics = transport._finalize_captured_reply(
        (
            "ASTER_PATCH_BEGIN\n"
            '{"summary":"ok","notes":[],"operations":[{"type":"CREATE FILE","path":"saved.py","reason":"add","content":"print(1)"}]'
        ),
        retry_attempt=True,
    )

    assert parsed == '{"summary":"ok","notes":[],"operations":[{"type":"CREATE FILE","path":"saved.py","reason":"add","content":"print(1)"}]}'
    assert diagnostics["retry_attempt_failure_reason"] == ""
    assert diagnostics["retry_attempt_acceptance_tier"] == "retry_structured_single_block_completion"
    assert diagnostics["retry_attempt_single_block_completion_attempted"] is True
    assert diagnostics["retry_attempt_single_block_completion_succeeded"] is True
    assert diagnostics["retry_attempt_single_block_completion_reason"] == "balanced_structural_closure"
    assert "missing_closing_bracket" in diagnostics["retry_attempt_single_block_completion_defect_types"]
    assert "missing_closing_brace" in diagnostics["retry_attempt_single_block_completion_defect_types"]
    assert "missing_end_marker" in diagnostics["retry_attempt_single_block_completion_defect_types"]
    assert diagnostics["retry_attempt_single_block_completion_closure_added"] == "]}\nASTER_PATCH_END"
    assert diagnostics["retry_attempt_single_block_semantically_incomplete"] is False
    assert diagnostics["retry_attempt_single_block_parseable_after_completion"] is True


def test_finalize_captured_reply_retry_attempt_fails_when_json_cleanup_is_not_enough() -> None:
    transport = BrowserChatGPTTransport()

    parsed, diagnostics = transport._finalize_captured_reply(
        (
            "ASTER_PATCH_BEGIN\n"
            '{"summary":"ok","notes":[],"operations":[{"type":"CREATE FILE","path":"saved.py","reason":"add","content":"print(1)",,}]}\n'
            "ASTER_PATCH_END"
        ),
        retry_attempt=True,
    )

    assert parsed == ""
    assert diagnostics["retry_attempt_failure_reason"] == "retry_json_cleanup_failed"
    assert diagnostics["retry_attempt_json_cleanup_attempted"] is True
    assert diagnostics["retry_attempt_json_cleanup_succeeded"] is False
    assert diagnostics["retry_attempt_json_cleanup_reason"] == "cleanup_candidate_not_parseable"


def test_finalize_captured_reply_retry_attempt_rejects_semantically_incomplete_single_block() -> None:
    transport = BrowserChatGPTTransport()

    parsed, diagnostics = transport._finalize_captured_reply(
        (
            "ASTER_PATCH_BEGIN\n"
            '{"summary":"ok","notes":[],"operations": Stop answering, Enter'
        ),
        retry_attempt=True,
    )

    assert parsed == ""
    assert diagnostics["retry_attempt_failure_reason"] == "retry_block_semantically_incomplete"
    assert diagnostics["retry_attempt_single_block_completion_attempted"] is True
    assert diagnostics["retry_attempt_single_block_completion_succeeded"] is False
    assert diagnostics["retry_attempt_single_block_completion_reason"] == "invalid_bareword_value_tail"
    assert diagnostics["retry_attempt_single_block_semantically_incomplete"] is True
    assert diagnostics["retry_attempt_single_block_parseable_after_completion"] is False


def test_finalize_captured_reply_retry_attempt_rejects_mid_content_single_block_truncation() -> None:
    transport = BrowserChatGPTTransport()

    parsed, diagnostics = transport._finalize_captured_reply(
        (
            "ASTER_PATCH_BEGIN\n"
            '{"summary":"ok","notes":[],"operations":[{"type":"CREATE FILE","path":"saved.py","reason":"add","content":"prin'
        ),
        retry_attempt=True,
    )

    assert parsed == ""
    assert diagnostics["retry_attempt_failure_reason"] == "retry_block_semantically_incomplete"
    assert diagnostics["retry_attempt_single_block_completion_attempted"] is True
    assert diagnostics["retry_attempt_single_block_completion_succeeded"] is False
    assert diagnostics["retry_attempt_single_block_completion_reason"] == "unterminated_string"
    assert diagnostics["retry_attempt_single_block_semantically_incomplete"] is True
    assert diagnostics["retry_attempt_single_block_parseable_after_completion"] is False


def test_finalize_captured_reply_retry_attempt_keeps_plain_prose_as_prose_contamination() -> None:
    transport = BrowserChatGPTTransport()

    parsed, diagnostics = transport._finalize_captured_reply(
        "Here is the corrected result. Please return only the ASTER block next time.",
        retry_attempt=True,
    )

    assert parsed == ""
    assert diagnostics["retry_attempt_failure_reason"] == "prose_contaminated_retry_response"
    assert diagnostics["retry_attempt_prose_recovery_attempted"] is True
    assert diagnostics["retry_attempt_prose_recovery_succeeded"] is False
    assert diagnostics["retry_attempt_prose_recovery_reason"] == "no_single_parseable_embedded_block"


def test_finalize_captured_reply_retry_attempt_classifies_wrapper_only() -> None:
    transport = BrowserChatGPTTransport()

    parsed, diagnostics = transport._finalize_captured_reply(
        "ASTER_PATCH_BEGIN\nASTER_PATCH_END",
        retry_attempt=True,
    )

    assert parsed == ""
    assert diagnostics["retry_attempt_failure_reason"] == "wrapper_without_valid_json"
    assert diagnostics["retry_attempt_wrapper_only"] is True
    assert diagnostics["retry_attempt_structured_block_found"] is True


def test_finalize_captured_reply_retry_attempt_rejects_multiple_blocks() -> None:
    transport = BrowserChatGPTTransport()

    parsed, diagnostics = transport._finalize_captured_reply(
        (
            "ASTER_PATCH_BEGIN\n"
            '{"summary":"first","notes":[],"operations":[{"type":"CREATE FILE","path":"first.txt","reason":"add","content":"one"}]}\n'
            "ASTER_PATCH_END\n"
            "ASTER_PATCH_BEGIN\n"
            '{"summary":"second","notes":[],"operations":[{"type":"CREATE FILE","path":"second.txt","reason":"add","content":"two"}]}\n'
            "ASTER_PATCH_END"
        ),
        retry_attempt=True,
    )

    assert parsed == ""
    assert diagnostics["retry_attempt_block_count"] == 2
    assert diagnostics["retry_attempt_failure_reason"] == "ambiguous_multiple_retry_blocks"
    assert diagnostics["retry_attempt_multiple_blocks_ambiguous"] is True
    assert diagnostics["retry_attempt_multiple_blocks_recovered"] is False
    assert diagnostics["retry_attempt_selected_block_index"] is None
    assert diagnostics["retry_attempt_block_selection_reason"] == "multiple_distinct_parseable_retry_blocks"
    assert diagnostics["retry_attempt_block_relationship"] == "distinct_parseable_blocks"
    assert len(diagnostics["retry_attempt_block_forensics"]) == 2


def test_finalize_captured_reply_retry_attempt_recovers_single_parseable_block() -> None:
    transport = BrowserChatGPTTransport()

    parsed, diagnostics = transport._finalize_captured_reply(
        (
            "ASTER_PATCH_BEGIN\n"
            '{"summary":"bad"\n'
            "ASTER_PATCH_END\n"
            "ASTER_PATCH_BEGIN\n"
            '{"summary":"second","notes":[],"operations":[{"type":"CREATE FILE","path":"ok.txt","reason":"add","content":"ok"}]}\n'
            "ASTER_PATCH_END"
        ),
        retry_attempt=True,
    )

    assert parsed == '{"summary":"second","notes":[],"operations":[{"type":"CREATE FILE","path":"ok.txt","reason":"add","content":"ok"}]}'
    assert diagnostics["retry_attempt_failure_reason"] == ""
    assert diagnostics["retry_attempt_acceptance_tier"] == "retry_structured_recovered_single_block"
    assert diagnostics["retry_attempt_multiple_blocks_recovered"] is True
    assert diagnostics["retry_attempt_multiple_blocks_ambiguous"] is False
    assert diagnostics["retry_attempt_selected_block_index"] == 2
    assert diagnostics["retry_attempt_block_selection_reason"] == "single_parseable_retry_block"
    assert diagnostics["retry_attempt_block_relationship"] == "single_parseable_plus_fragments"


def test_finalize_captured_reply_retry_attempt_recovers_duplicate_parseable_blocks() -> None:
    transport = BrowserChatGPTTransport()
    block = (
        "ASTER_PATCH_BEGIN\n"
        '{"summary":"same","notes":[],"operations":[{"type":"CREATE FILE","path":"dup.txt","reason":"add","content":"ok"}]}\n'
        "ASTER_PATCH_END"
    )

    parsed, diagnostics = transport._finalize_captured_reply(
        f"{block}\n{block}",
        retry_attempt=True,
    )

    assert parsed == '{"summary":"same","notes":[],"operations":[{"type":"CREATE FILE","path":"dup.txt","reason":"add","content":"ok"}]}'
    assert diagnostics["retry_attempt_failure_reason"] == ""
    assert diagnostics["retry_attempt_multiple_blocks_recovered"] is True
    assert diagnostics["retry_attempt_selected_block_index"] == 2
    assert diagnostics["retry_attempt_block_selection_reason"] == "duplicate_parseable_retry_blocks"
    assert diagnostics["retry_attempt_block_relationship"] == "duplicate_parseable_blocks"


def test_finalize_captured_reply_retry_attempt_ambiguates_multiple_malformed_blocks() -> None:
    transport = BrowserChatGPTTransport()

    parsed, diagnostics = transport._finalize_captured_reply(
        (
            "ASTER_PATCH_BEGIN\n"
            '{"summary":"first"\n'
            "ASTER_PATCH_END\n"
            "ASTER_PATCH_BEGIN\n"
            '{"summary":"second"\n'
            "ASTER_PATCH_END"
        ),
        retry_attempt=True,
    )

    assert parsed == ""
    assert diagnostics["retry_attempt_failure_reason"] == "ambiguous_multiple_retry_blocks"
    assert diagnostics["retry_attempt_multiple_blocks_ambiguous"] is True
    assert diagnostics["retry_attempt_multiple_blocks_recovered"] is False
    assert diagnostics["retry_attempt_block_selection_reason"] == "multiple_malformed_retry_blocks_tied"
    assert diagnostics["retry_attempt_block_relationship"] == "malformed_blocks_tied"


def test_finalize_captured_reply_retry_attempt_classifies_fragmented_multi_block_output() -> None:
    transport = BrowserChatGPTTransport()

    parsed, diagnostics = transport._finalize_captured_reply(
        (
            "ASTER_PATCH_BEGIN\nASTER_PATCH_END\n"
            'ASTER_PATCH_BEGIN\n"summary":"first","notes":[],"operations":[{"type":"CREATE FILE","path":"one.txt","reason":"add","content":"one"}],,"extra":"bad"'
        ),
        retry_attempt=True,
    )

    assert parsed == ""
    assert diagnostics["retry_attempt_block_count"] == 2
    assert diagnostics["retry_attempt_failure_reason"] == "fragment_repair_not_parseable"
    assert diagnostics["retry_attempt_multiple_blocks_recovered"] is False
    assert diagnostics["retry_attempt_multiple_blocks_ambiguous"] is False
    assert diagnostics["retry_attempt_block_selection_reason"] == "fragment_repair_not_parseable"
    assert diagnostics["retry_attempt_block_relationship"] == "wrapper_only_plus_fragmented_block"
    assert diagnostics["retry_attempt_block_forensics"][0]["block_kind"] == "wrapper_only_block"
    assert diagnostics["retry_attempt_block_forensics"][1]["has_end_marker"] is False
    assert diagnostics["retry_attempt_block_forensics"][1]["payload_state"] == "partial_payload"
    assert diagnostics["retry_attempt_fragment_repair_pattern_matched"] is True
    assert diagnostics["retry_attempt_fragment_repair_attempted"] is True
    assert diagnostics["retry_attempt_fragment_repair_succeeded"] is False
    assert diagnostics["retry_attempt_fragment_repair_reason"] == "repair_candidate_not_parseable"
    assert diagnostics["retry_attempt_repaired_from_block_index"] == 2
    assert diagnostics["retry_attempt_discarded_wrapper_only_block_index"] == 1


def test_finalize_captured_reply_retry_attempt_classifies_wrapper_only_multi_block_output() -> None:
    transport = BrowserChatGPTTransport()

    parsed, diagnostics = transport._finalize_captured_reply(
        (
            "ASTER_PATCH_BEGIN\nASTER_PATCH_END\n"
            "ASTER_PATCH_BEGIN\nASTER_PATCH_END"
        ),
        retry_attempt=True,
    )

    assert parsed == ""
    assert diagnostics["retry_attempt_block_count"] == 2
    assert diagnostics["retry_attempt_failure_reason"] == "wrapper_only_multi_block_retry_output"
    assert diagnostics["retry_attempt_multiple_blocks_recovered"] is False
    assert diagnostics["retry_attempt_multiple_blocks_ambiguous"] is False
    assert diagnostics["retry_attempt_block_selection_reason"] == "wrapper_only_multi_block_fragments"
    assert diagnostics["retry_attempt_block_relationship"] == "wrapper_only_blocks"
    assert diagnostics["retry_attempt_wrapper_only_block_count"] == 2
    assert diagnostics["retry_attempt_wrapper_only_payload_lengths"] == [0, 0]
    assert diagnostics["retry_attempt_wrapper_only_has_internal_text"] is False
    assert diagnostics["retry_attempt_wrapper_only_noise_detected"] is False
    assert diagnostics["retry_attempt_wrapper_only_boundary_suspected"] is False
    assert diagnostics["retry_wrapper_recheck_attempted"] is False
    assert diagnostics["retry_wrapper_recheck_found_payload"] is False
    assert diagnostics["retry_wrapper_recheck_reason"] == "no_boundary_signal"
    assert all(item["block_kind"] == "wrapper_only_block" for item in diagnostics["retry_attempt_block_forensics"])


def test_finalize_captured_reply_retry_attempt_characterizes_wrapper_only_internal_noise() -> None:
    transport = BrowserChatGPTTransport()

    parsed, diagnostics = transport._finalize_captured_reply(
        (
            "ASTER_PATCH_BEGIN\nhello from a bad wrapper\nASTER_PATCH_END\n"
            "ASTER_PATCH_BEGIN\nASTER_PATCH_END"
        ),
        retry_attempt=True,
    )

    assert parsed == ""
    assert diagnostics["retry_attempt_failure_reason"] == "wrapper_only_multi_block_retry_output"
    assert diagnostics["retry_attempt_wrapper_only_block_count"] == 2
    assert diagnostics["retry_attempt_wrapper_only_payload_lengths"] == [24, 0]
    assert diagnostics["retry_attempt_wrapper_only_has_internal_text"] is True
    assert diagnostics["retry_attempt_wrapper_only_noise_detected"] is True
    assert diagnostics["retry_attempt_wrapper_only_boundary_suspected"] is True
    assert diagnostics["retry_wrapper_recheck_attempted"] is True
    assert diagnostics["retry_wrapper_recheck_found_payload"] is False
    assert diagnostics["retry_wrapper_recheck_reason"] == "boundary_suspected_but_no_payload"


def test_wrapper_only_retry_recheck_recovers_hidden_payload_when_same_text_contains_it() -> None:
    blocks = [
        {
            "index": 1,
            "raw_block": (
                "ASTER_PATCH_BEGIN\n"
                '{"summary":"ok","notes":[],"operations":[{"type":"CREATE FILE","path":"ok.txt","reason":"add","content":"ok"}]}\n'
                "ASTER_PATCH_END"
            ),
            "block_kind": "wrapper_only_block",
            "payload_state": "empty_payload",
            "has_end_marker": True,
        }
    ]

    selection = BrowserChatGPTTransport._attempt_wrapper_only_retry_recheck(
        blocks[0]["raw_block"],
        blocks,
        outside_text="",
    )

    assert selection["recovered"] is True
    assert selection["selection_reason"] == "wrapper_only_recheck_hidden_payload"
    assert selection["wrapper_recheck_attempted"] is True
    assert selection["wrapper_recheck_found_payload"] is True
    assert selection["wrapper_recheck_reason"] == "inner_payload_parseable"
    assert selection["selected_text"] == '{"summary":"ok","notes":[],"operations":[{"type":"CREATE FILE","path":"ok.txt","reason":"add","content":"ok"}]}'


def test_finalize_captured_reply_retry_attempt_repairs_wrapper_plus_bare_object_fragment() -> None:
    transport = BrowserChatGPTTransport()

    parsed, diagnostics = transport._finalize_captured_reply(
        (
            "ASTER_PATCH_BEGIN\nASTER_PATCH_END\n"
            'ASTER_PATCH_BEGIN\n"summary":"ok","notes":[],"operations":[{"type":"CREATE FILE","path":"ok.txt","reason":"add","content":"ok"}]'
        ),
        retry_attempt=True,
    )

    assert parsed == '{"summary":"ok","notes":[],"operations":[{"type":"CREATE FILE","path":"ok.txt","reason":"add","content":"ok"}]}'
    assert diagnostics["retry_attempt_failure_reason"] == ""
    assert diagnostics["retry_attempt_acceptance_tier"] == "retry_structured_repaired_fragment"
    assert diagnostics["retry_attempt_multiple_blocks_recovered"] is True
    assert diagnostics["retry_attempt_multiple_blocks_ambiguous"] is False
    assert diagnostics["retry_attempt_selected_block_index"] == 2
    assert diagnostics["retry_attempt_block_selection_reason"] == "repaired_fragmented_retry_block"
    assert diagnostics["retry_attempt_block_relationship"] == "wrapper_only_plus_repaired_fragment"
    assert diagnostics["retry_attempt_fragment_repair_pattern_matched"] is True
    assert diagnostics["retry_attempt_fragment_repair_attempted"] is True
    assert diagnostics["retry_attempt_fragment_repair_succeeded"] is True
    assert diagnostics["retry_attempt_fragment_repair_reason"] == "wrapped_object_body_and_appended_end_marker"
    assert diagnostics["retry_attempt_repaired_from_block_index"] == 2
    assert diagnostics["retry_attempt_discarded_wrapper_only_block_index"] == 1


def test_finalize_captured_reply_retry_attempt_does_not_repair_weak_fragment() -> None:
    transport = BrowserChatGPTTransport()

    parsed, diagnostics = transport._finalize_captured_reply(
        (
            "ASTER_PATCH_BEGIN\nASTER_PATCH_END\n"
            'ASTER_PATCH_BEGIN\n"summary":"weak fragment only"'
        ),
        retry_attempt=True,
    )

    assert parsed == ""
    assert diagnostics["retry_attempt_failure_reason"] == "fragment_repair_insufficient_structure"
    assert diagnostics["retry_attempt_multiple_blocks_recovered"] is False
    assert diagnostics["retry_attempt_fragment_repair_pattern_matched"] is True
    assert diagnostics["retry_attempt_fragment_repair_attempted"] is False
    assert diagnostics["retry_attempt_fragment_repair_succeeded"] is False
    assert diagnostics["retry_attempt_fragment_repair_reason"] == "insufficient_schema_hits"


def test_retry_attempt_prefers_uia_fragment_over_noisier_ocr_fragment() -> None:
    transport = BrowserChatGPTTransport()

    candidate, meta = transport._choose_retry_attempt_candidate_sources(
        ocr_text=(
            "ASTER_PATCH_BEGIN\n"
            '{"summary":"ok","notes":[],"operations":[{"type":"CREATE FILE","path":"ok.txt","reason":"add","content":"ok"}'
            "\nAdditional OCR junk line that should not outrank the cleaner UIA fragment.\n"
        ),
        uia_text=(
            "ASTER_PATCH_BEGIN\n"
            '{"summary":"ok","notes":[],"operations":[{"type":"CREATE FILE","path":"ok.txt","reason":"add","content":"ok"}]'
        ),
        prompt_anchor=None,
    )

    assert candidate is not None
    assert candidate[0] == "uia"
    assert meta["fragment_source_preference"] == "prefer_uia_fragment_over_ocr"
    assert meta["fragment_source_chosen"] == "uia"
    assert meta["fragment_uia_available"] is True
    assert meta["fragment_ocr_available"] is True


def test_retry_attempt_assessment_repairs_ocr_corrupted_fragment_when_cleanup_is_sufficient() -> None:
    transport = BrowserChatGPTTransport()

    assessment = transport._assess_retry_attempt_response(
        (
            "ASTER_PATCH_BEGIN\nASTER_PATCH_END\n"
            'ASTER_PATCH_BEGIN\n{\u201csummary\u201d:\u201cok\u201d,\u201cn0tes\u201d:[],\u201c0perati0ns\u201d:[{\u201ctype\u201d:\u201cCREATE FILE\u201d,\u201cpath\u201d:\u201cok.txt\u201d,\u201creason\u201d:\u201cadd\u201d,\u201ccontent\u201d:\u201cok\u201d}]'
        ),
        candidate_source="ocr",
    )

    assert assessment["failure_reason"] == ""
    assert assessment["selected_text"] == '{"summary":"ok","notes":[],"operations":[{"type":"CREATE FILE","path":"ok.txt","reason":"add","content":"ok"}]}'
    assert assessment["fragment_repair_succeeded"] is True
    assert assessment["fragment_repair_parseable_after_cleanup"] is True
    assert assessment["ocr_cleanup_attempted"] is True
    assert assessment["ocr_cleanup_succeeded"] is True
    assert assessment["ocr_cleanup_reason"] == "ocr_cleanup_and_balanced_closure"
    assert "smart_quotes" in assessment["ocr_defect_types"]
    assert "schema_key_ocr" in assessment["ocr_defect_types"]


def test_retry_attempt_assessment_rejects_deeply_corrupted_ocr_fragment() -> None:
    transport = BrowserChatGPTTransport()

    assessment = transport._assess_retry_attempt_response(
        (
            "ASTER_PATCH_BEGIN\nASTER_PATCH_END\n"
            'ASTER_PATCH_BEGIN\n{\u201csummary\u201d:\u201cok\u201d,\u201cn0tes\u201d:[],\u201c0perati0ns\u201d:[{\u201ctype\u201d:\u201cCREATE FILE\u201d,\u201cpath\u201d:\u201cok.txt\u201d,\u201creason\u201d:\u201cadd\u201d,\u201ccontent\u201d:\u201cok\u201d,\u201cbadvalue\u201d:::\u201d???'
        ),
        candidate_source="ocr",
    )

    assert assessment["failure_reason"] == "fragment_repair_not_parseable"
    assert assessment["selected_text"] == ""
    assert assessment["fragment_repair_succeeded"] is False
    assert assessment["fragment_repair_parseable_after_cleanup"] is False
    assert assessment["ocr_cleanup_attempted"] is True
    assert assessment["ocr_cleanup_succeeded"] is False
    assert assessment["ocr_cleanup_reason"] == "cleanup_candidate_not_parseable"
    assert assessment["block_relationship"] == "wrapper_only_plus_fragmented_block"


def test_capture_reply_text_does_not_require_anchor_when_thread_was_not_reused(monkeypatch) -> None:
    transport = BrowserChatGPTTransport(thread_reuse_enabled=True)
    transport._executor = _FakeExecutor("")
    transport._capture = _FakeCaptureService()
    transport._ocr = _FakeOCR()
    transport._ui_state = lambda target: {
        "show_in_text_field_present": False,
        "send_prompt_present": False,
        "send_prompt_enabled": None,
        "stop_streaming_present": False,
    }
    transport._raise_for_browser_error = lambda lines, ui_state, stage: None
    reply_text = '{"summary":"ok","notes":[],"operations":[{"type":"CREATE FILE","path":"weather.py","reason":"add","content":"print(1)"}]}'
    replies = iter([("", reply_text), ("", reply_text)])
    transport._capture_visible_reply_sources = lambda target, before_lines, prompt, lines=None: next(replies)
    monkeypatch.setattr("aster.transport_browser.browser_transport.time.sleep", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("aster.transport_browser.browser_transport.time.monotonic", _MonotonicClock())

    class _Target:
        left = 0
        top = 0
        width = 1200
        height = 900

    reply = transport._capture_reply_text(
        _Target(),
        before_lines=[],
        prompt="create a hello world python file in live_browser_test",
        timeout_sec=4.0,
        prompt_anchor=build_turn_anchor("create a hello world python file in live_browser_test"),
        thread_reused_for_capture=False,
    )

    assert looks_like_patch_plan_json(reply) is True
    assert "weather.py" in reply


def test_capture_reply_text_requires_anchor_only_when_thread_was_reused(monkeypatch) -> None:
    logger = _FakeAuditLogger()
    transport = BrowserChatGPTTransport(logger=logger, thread_reuse_enabled=True)
    transport._executor = _FakeExecutor("")
    transport._capture = _FakeCaptureService()
    transport._ocr = _FakeOCR()
    transport._ui_state = lambda target: {
        "show_in_text_field_present": False,
        "send_prompt_present": False,
        "send_prompt_enabled": None,
        "stop_streaming_present": False,
    }
    transport._raise_for_browser_error = lambda lines, ui_state, stage: None
    reply_text = '{"summary":"ok","notes":[],"operations":[{"type":"CREATE FILE","path":"weather.py","reason":"add","content":"print(1)"}]}'
    replies = iter([("", reply_text), ("", reply_text)])
    transport._capture_visible_reply_sources = lambda target, before_lines, prompt, lines=None: next(replies)
    monkeypatch.setattr("aster.transport_browser.browser_transport.time.sleep", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("aster.transport_browser.browser_transport.time.monotonic", _MonotonicClock())

    class _Target:
        left = 0
        top = 0
        width = 1200
        height = 900

    reply = transport._capture_reply_text(
        _Target(),
        before_lines=[],
        prompt="create a hello world python file in live_browser_test",
        timeout_sec=4.0,
        prompt_anchor=build_turn_anchor("create a hello world python file in live_browser_test"),
        thread_reused_for_capture=True,
    )

    assert reply == ""
    timeout_events = _notice_events(logger, "reply_wait_timeout")
    assert timeout_events
    assert timeout_events[-1]["acceptance_tier"] == "blocked_or_ambiguous"
    assert "Anchor confidence is too low" in str(timeout_events[-1]["rejection_reason"])


def test_ensure_fresh_chat_thread_reports_fresh_capture_state() -> None:
    transport = BrowserChatGPTTransport(thread_reuse_enabled=True)
    transport._window_title_suggests_existing_thread = lambda title: False
    transport._reset_to_new_chat_if_possible = lambda target, timeout_sec: True

    class _Target:
        title = "Untitled - Google Chrome"

    assert transport._ensure_fresh_chat_thread(_Target(), "https://chatgpt.com/", 5.0) is False


def test_capture_reply_text_blocks_one_shot_scrolled_structured_candidate(monkeypatch) -> None:
    logger = _FakeAuditLogger()
    transport = BrowserChatGPTTransport(logger=logger)
    transport._executor = _FakeExecutor("")
    transport._capture = _FakeCaptureService()
    transport._ocr = _FakeOCR()
    transport._ui_state = lambda target: {
        "show_in_text_field_present": False,
        "send_prompt_present": False,
        "send_prompt_enabled": None,
        "stop_streaming_present": False,
    }
    transport._raise_for_browser_error = lambda lines, ui_state, stage: None
    replies = iter(
        [
            (
                "",
                'ASTER_PATCH_BEGIN\n{"summary":"ok","notes":[],"operations":[',
            ),
            ("", ""),
        ]
    )
    transport._capture_visible_reply_sources = lambda target, before_lines, prompt, lines=None: next(replies)
    transport._capture_reply_text_by_scrolling = lambda target, before_lines, prompt, max_steps, **_kwargs: (
        '{"summary":"ok","notes":[],"operations":[{"type":"CREATE FILE","path":"app.py","reason":"add","content":"print(1)"}]}'
    )
    monkeypatch.setattr("aster.transport_browser.browser_transport.time.sleep", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("aster.transport_browser.browser_transport.time.monotonic", _MonotonicClock())

    class _Target:
        left = 0
        top = 0
        width = 1200
        height = 900

    reply = transport._capture_reply_text(_Target(), before_lines=[], prompt="create app", timeout_sec=4.0)

    assert reply == ""
    scroll_events = _notice_events(logger, "reply_scrolled_capture")
    assert scroll_events
    assert scroll_events[-1]["acceptance_tier"] == "scrolled_structured"
    assert scroll_events[-1]["accepted"] is False


def test_capture_reply_text_retry_attempt_prefers_exact_structured_block(monkeypatch) -> None:
    transport = BrowserChatGPTTransport()
    transport._executor = _FakeExecutor("")
    transport._capture = _FakeCaptureService()
    transport._ocr = _FakeOCR()
    transport._ui_state = lambda target: {
        "show_in_text_field_present": False,
        "send_prompt_present": False,
        "send_prompt_enabled": None,
        "stop_streaming_present": False,
    }
    transport._raise_for_browser_error = lambda lines, ui_state, stage: None
    reply_text = (
        "ASTER_PATCH_BEGIN\n"
        '{"summary":"ok","notes":[],"operations":[{"type":"CREATE FILE","path":"app.py","reason":"add","content":"print(1)"}]}\n'
        "ASTER_PATCH_END"
    )
    transport._capture_visible_reply_sources = lambda target, before_lines, prompt, lines=None: ("", reply_text)
    monkeypatch.setattr("aster.transport_browser.browser_transport.time.sleep", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("aster.transport_browser.browser_transport.time.monotonic", _MonotonicClock())

    class _Target:
        left = 0
        top = 0
        width = 1200
        height = 900

    reply = transport._capture_reply_text(
        _Target(),
        before_lines=[],
        prompt="retry create app",
        timeout_sec=4.0,
        retry_attempt=True,
    )

    assert looks_like_patch_plan_json(reply) is True
    assert reply.strip().startswith("{")
    assert "ASTER_PATCH_BEGIN" not in reply


def test_capture_reply_text_retry_attempt_recovers_prose_wrapped_block(monkeypatch) -> None:
    logger = _FakeAuditLogger()
    transport = BrowserChatGPTTransport(logger=logger)
    transport._executor = _FakeExecutor("")
    transport._capture = _FakeCaptureService()
    transport._ocr = _FakeOCR()
    transport._ui_state = lambda target: {
        "show_in_text_field_present": False,
        "send_prompt_present": False,
        "send_prompt_enabled": None,
        "stop_streaming_present": False,
    }
    transport._raise_for_browser_error = lambda lines, ui_state, stage: None
    reply_text = (
        "Sure, here is the final answer.\n"
        "ASTER_PATCH_BEGIN\n"
        '{"summary":"ok","notes":[],"operations":[{"type":"CREATE FILE","path":"app.py","reason":"add","content":"print(1)"}]}\n'
        "ASTER_PATCH_END\n"
        "Done."
    )
    transport._capture_visible_reply_sources = lambda target, before_lines, prompt, lines=None: ("", reply_text)
    monkeypatch.setattr("aster.transport_browser.browser_transport.time.sleep", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("aster.transport_browser.browser_transport.time.monotonic", _MonotonicClock())

    class _Target:
        left = 0
        top = 0
        width = 1200
        height = 900

    reply = transport._capture_reply_text(
        _Target(),
        before_lines=[],
        prompt="retry create app",
        timeout_sec=4.0,
        retry_attempt=True,
    )

    assert reply == '{"summary":"ok","notes":[],"operations":[{"type":"CREATE FILE","path":"app.py","reason":"add","content":"print(1)"}]}'
    accepted_events = _notice_events(logger, "reply_candidate_accepted")
    assert accepted_events
    assert accepted_events[-1]["acceptance_tier"] == "retry_structured_embedded_block"
    assert accepted_events[-1]["retry_attempt_failure_reason"] == ""


def test_generate_salvages_final_structured_block_from_trusted_capture(monkeypatch) -> None:
    transport = BrowserChatGPTTransport()
    transport._logger = None
    transport._ensure_runtime = lambda: None
    transport._show_automation_notice = lambda: None
    transport._hide_automation_notice = lambda: None
    transport._activity = lambda *_args, **_kwargs: None
    transport._populate_prompt_directly = lambda target, prompt, prompt_anchor=None: True
    transport._stabilize_and_send = lambda target, before_lines, prompt: None
    transport._prepare_chatgpt_window = lambda target, chatgpt_url, timeout_sec: False
    transport._capture = _FakeCaptureService()
    transport._ocr = _FakeOCR()

    class _Target:
        title = "ChatGPT - Google Chrome"
        left = 0
        top = 0
        width = 1200
        height = 900

    transport._ensure_chatgpt_window = lambda chatgpt_url, launch_timeout_sec: _Target()
    transport._ui_state = lambda target: {
        "show_in_text_field_present": False,
        "send_prompt_present": False,
        "send_prompt_enabled": None,
        "stop_streaming_present": False,
    }
    transport._capture_visible_text_lines = lambda target: []

    def _capture_reply_text(*_args, **_kwargs):
        snapshot = transport._last_reply_capture_snapshot
        transport._remember_reply_capture_candidate(
            snapshot,
            source="uia",
            text='{"summary":"ok","notes":[],"operations":[{"type":"CREATE FILE","path":"saved.py","reason":"add","content":"print(1)"}]}',
            score=360.0,
            salvage_allowed=True,
            observation_counts={},
        )
        transport._last_reply_capture_snapshot = snapshot
        return ""

    transport._capture_reply_text = _capture_reply_text

    result = transport.generate("create saved.py", timeout_sec=4.0, launch_timeout_sec=1.0)

    assert looks_like_patch_plan_json(result.raw_text) is True
    assert '"saved.py"' in result.raw_text
    assert result.metadata["retry_seed_valid"] is True


def test_generate_returns_empty_reply_when_only_diagnostic_salvage_exists(monkeypatch) -> None:
    transport = BrowserChatGPTTransport()
    transport._logger = None
    transport._ensure_runtime = lambda: None
    transport._show_automation_notice = lambda: None
    transport._hide_automation_notice = lambda: None
    transport._activity = lambda *_args, **_kwargs: None
    transport._populate_prompt_directly = lambda target, prompt, prompt_anchor=None: True
    transport._stabilize_and_send = lambda target, before_lines, prompt: None
    transport._prepare_chatgpt_window = lambda target, chatgpt_url, timeout_sec: False
    transport._capture = _FakeCaptureService()
    transport._ocr = _FakeOCR()

    class _Target:
        title = "ChatGPT - Google Chrome"
        left = 0
        top = 0
        width = 1200
        height = 900

    transport._ensure_chatgpt_window = lambda chatgpt_url, launch_timeout_sec: _Target()
    transport._ui_state = lambda target: {
        "show_in_text_field_present": False,
        "send_prompt_present": False,
        "send_prompt_enabled": None,
        "stop_streaming_present": False,
    }
    transport._capture_visible_text_lines = lambda target: []

    def _capture_reply_text(*_args, **_kwargs):
        snapshot = transport._last_reply_capture_snapshot
        transport._remember_reply_capture_candidate(
            snapshot,
            source="uia",
            text=(
                "ASTER_PATCH_BEGIN\n"
                '{"summary":"ok","notes":[],"operations":[{"type":"CREATE FILE","path":"saved.py","reason":"add","content":"print(1)"}]\n'
            ),
            score=360.0,
            salvage_allowed=True,
            observation_counts={},
        )
        transport._last_reply_capture_snapshot = snapshot
        return ""

    transport._capture_reply_text = _capture_reply_text

    result = transport.generate("create saved.py", timeout_sec=4.0, launch_timeout_sec=1.0)

    assert result.raw_text == ""
    assert result.metadata["retry_seed_valid"] is False
    assert result.metadata["retry_seed_validity_reason"] == "unbalanced_structure"
    assert result.metadata["retry_seed_validity_reason_source"] == "best_salvageable_candidate"
    assert result.metadata["final_capture_failure_reason"] == "structured_block_seen_but_not_retry_safe"


def test_generate_raises_final_capture_failure_when_no_salvageable_candidate_exists(monkeypatch) -> None:
    transport = BrowserChatGPTTransport()
    transport._logger = None
    transport._ensure_runtime = lambda: None
    transport._show_automation_notice = lambda: None
    transport._hide_automation_notice = lambda: None
    transport._activity = lambda *_args, **_kwargs: None
    transport._populate_prompt_directly = lambda target, prompt, prompt_anchor=None: True
    transport._stabilize_and_send = lambda target, before_lines, prompt: None
    transport._prepare_chatgpt_window = lambda target, chatgpt_url, timeout_sec: False
    transport._capture = _FakeCaptureService()
    transport._ocr = _FakeOCR()

    class _Target:
        title = "ChatGPT - Google Chrome"
        left = 0
        top = 0
        width = 1200
        height = 900

    transport._ensure_chatgpt_window = lambda chatgpt_url, launch_timeout_sec: _Target()
    transport._ui_state = lambda target: {
        "show_in_text_field_present": False,
        "send_prompt_present": False,
        "send_prompt_enabled": None,
        "stop_streaming_present": False,
    }
    transport._capture_visible_text_lines = lambda target: []
    transport._capture_reply_text = lambda *_args, **_kwargs: ""

    with pytest.raises(RuntimeError, match="no structured patch block was seen during reply capture"):
        transport.generate("create saved.py", timeout_sec=4.0, launch_timeout_sec=1.0)


def test_capture_reply_text_by_scrolling_keeps_structured_anchor_over_prompt_contamination(monkeypatch) -> None:
    transport = BrowserChatGPTTransport()
    transport._logger = None
    transport._tick_runtime_log_heartbeat = lambda *_args, **_kwargs: None
    transport._activity = lambda *_args, **_kwargs: None
    transport._scroll_reply_to_bottom = lambda target: None
    transport._scroll_reply_up = lambda target: None
    segments = iter(
        [
            "SYSTEM:\nYou are a coding orchestrator backend. Return JSON only.\nUSER:\nUser goal: Build app\n",
            'from tkinter import ttk\\nprint(\\"ok\\")"}]}\nASTER_PATCH_END',
            "",
        ]
    )
    transport._capture_visible_reply_segment = lambda target, before_lines, prompt: next(segments)
    monkeypatch.setattr("aster.transport_browser.browser_transport.time.sleep", lambda *_args, **_kwargs: None)

    class _Target:
        left = 0
        top = 0
        width = 1200
        height = 900

    reply = transport._capture_reply_text_by_scrolling(
        _Target(),
        before_lines=[],
        prompt="create app",
        max_steps=3,
        structured_anchor_text=(
            "ASTER_PATCH_BEGIN\n"
            '{"summary":"ok","notes":[],"operations":[{"type":"CREATE FILE","path":"app.py","reason":"add","content":"import tkinter as tk\\n'
        ),
    )

    assert "SYSTEM:" not in reply
    assert "USER:" not in reply
    assert "ASTER_PATCH_END" in reply


def test_capture_reply_text_by_scrolling_rejects_unrelated_page_drift_with_region_diagnostics(monkeypatch) -> None:
    logger = _FakeAuditLogger()
    transport = BrowserChatGPTTransport(logger=logger)
    transport._tick_runtime_log_heartbeat = lambda *_args, **_kwargs: None
    transport._activity = lambda *_args, **_kwargs: None
    transport._scroll_reply_to_bottom = lambda target: None
    transport._scroll_reply_up = lambda target: None
    segments = iter(
        [
            "tests/test_calc.py\ncollected 4 items\nshort test summary info\n",
            'from tkinter import ttk\\nprint(\\"ok\\")"}]}\nASTER_PATCH_END',
            "",
        ]
    )
    transport._capture_visible_reply_segment = lambda target, before_lines, prompt: next(segments)
    monkeypatch.setattr("aster.transport_browser.browser_transport.time.sleep", lambda *_args, **_kwargs: None)

    class _Target:
        left = 0
        top = 0
        width = 1200
        height = 900

    reply = transport._capture_reply_text_by_scrolling(
        _Target(),
        before_lines=[],
        prompt="create app",
        max_steps=3,
        structured_anchor_text=(
            "ASTER_PATCH_BEGIN\n"
            '{"summary":"ok","notes":[],"operations":[{"type":"CREATE FILE","path":"app.py","reason":"add","content":"import tkinter as tk\\n'
        ),
    )

    assert looks_like_patch_plan_json(reply) is True
    skipped = _notice_events(logger, "reply_scroll_segment_skipped")
    assert skipped
    assert skipped[0]["skip_reason"] in {
        "unrelated_page_content",
        "reply_region_drift",
        "low_region_integrity",
        "missing_seed_or_trusted_lineage",
        "current_blob_only_continuity",
    }
    assert skipped[0]["unrelated_page_content"] is True
    assert skipped[0]["region_integrity_score"] <= 0
    assert skipped[0]["drift_detected"] is True or skipped[0]["unrelated_page_hits"]
    assert "trusted_lineage_score" in skipped[0]
    assert skipped[0]["drift_containment_active"] is True


def test_capture_reply_text_by_scrolling_assembles_ordered_segments_into_parseable_json(monkeypatch) -> None:
    logger = _FakeAuditLogger()
    transport = BrowserChatGPTTransport(logger=logger)
    transport._tick_runtime_log_heartbeat = lambda *_args, **_kwargs: None
    transport._activity = lambda *_args, **_kwargs: None
    transport._scroll_reply_to_bottom = lambda target: None
    transport._scroll_reply_up = lambda target: None
    segments = iter(
        [
            'from tkinter import ttk\\nprint(\\"ok\\")"}',
            'from tkinter import ttk\\nprint(\\"ok\\")"}',
            ']}\nASTER_PATCH_END',
            "",
        ]
    )
    transport._capture_visible_reply_segment = lambda target, before_lines, prompt: next(segments)
    monkeypatch.setattr("aster.transport_browser.browser_transport.time.sleep", lambda *_args, **_kwargs: None)

    class _Target:
        left = 0
        top = 0
        width = 1200
        height = 900

    reply = transport._capture_reply_text_by_scrolling(
        _Target(),
        before_lines=[],
        prompt="create app",
        max_steps=4,
        structured_anchor_text=(
            "ASTER_PATCH_BEGIN\n"
            '{"summary":"ok","notes":[],"operations":[{"type":"CREATE FILE","path":"app.py","reason":"add","content":"import tkinter as tk\\n'
        ),
    )

    assert looks_like_patch_plan_json(reply) is True
    scroll_events = _notice_events(logger, "reply_scroll_segment")
    assert scroll_events
    assert "region_integrity_score" in scroll_events[-1]
    assert "continuity_against_seed" in scroll_events[-1]
    assert "trusted_lineage_score" in scroll_events[-1]
    assert "matched_trusted_lineage" in scroll_events[-1]
    skipped = _notice_events(logger, "reply_scroll_segment_skipped")
    assert skipped
    assert skipped[0]["skip_reason"] in {"duplicate_overlap", "no_new_structured_content"}


def test_capture_reply_text_by_scrolling_contains_later_weak_segment_after_drift(monkeypatch) -> None:
    logger = _FakeAuditLogger()
    transport = BrowserChatGPTTransport(logger=logger)
    transport._tick_runtime_log_heartbeat = lambda *_args, **_kwargs: None
    transport._activity = lambda *_args, **_kwargs: None
    transport._scroll_reply_to_bottom = lambda target: None
    transport._scroll_reply_up = lambda target: None
    segments = iter(
        [
            "tests/test_calc.py\ncollected 4 items\nshort test summary info\n",
            'config.runtime_log_heartbeat_branch_only is True\\nassert config.thread_reuse_enabled is False\\n',
            'from tkinter import ttk\\nprint(\\"ok\\")"}]}\nASTER_PATCH_END',
            "",
        ]
    )
    transport._capture_visible_reply_segment = lambda target, before_lines, prompt: next(segments)
    monkeypatch.setattr("aster.transport_browser.browser_transport.time.sleep", lambda *_args, **_kwargs: None)

    class _Target:
        left = 0
        top = 0
        width = 1200
        height = 900

    reply = transport._capture_reply_text_by_scrolling(
        _Target(),
        before_lines=[],
        prompt="create app",
        max_steps=4,
        structured_anchor_text=(
            "ASTER_PATCH_BEGIN\n"
            '{"summary":"ok","notes":[],"operations":[{"type":"CREATE FILE","path":"app.py","reason":"add","content":"import tkinter as tk\\n'
        ),
    )

    assert looks_like_patch_plan_json(reply) is True
    skipped = _notice_events(logger, "reply_scroll_segment_skipped")
    assert len(skipped) >= 2
    assert skipped[0]["drift_containment_active"] is True
    assert skipped[1]["skip_reason"] in {"drift_containment_active", "missing_seed_or_trusted_lineage"}
    scroll_events = _notice_events(logger, "reply_scroll_segment")
    assert scroll_events[-1]["trusted_lineage_extended"] is True


def test_capture_reply_text_by_scrolling_uses_bounded_continuation_window_for_structured_progress(monkeypatch) -> None:
    logger = _FakeAuditLogger()
    transport = BrowserChatGPTTransport(logger=logger)
    transport._tick_runtime_log_heartbeat = lambda *_args, **_kwargs: None
    transport._activity = lambda *_args, **_kwargs: None
    transport._scroll_reply_to_bottom = lambda target: None
    transport._scroll_reply_up = lambda target: None
    segments = iter(
        [
            '{"type":"CREATE FILE","path":"app.py","reason":"add","content":"print(1)"},',
            '{"type":"RUN COMMANDS","path":".","reason":"verify","commands":["pytest"]}]}\nASTER_PATCH_END',
            "",
        ]
    )
    transport._capture_visible_reply_segment = lambda target, before_lines, prompt: next(segments)
    monkeypatch.setattr("aster.transport_browser.browser_transport.time.sleep", lambda *_args, **_kwargs: None)

    class _Target:
        left = 0
        top = 0
        width = 1200
        height = 900

    reply = transport._capture_reply_text_by_scrolling(
        _Target(),
        before_lines=[],
        prompt="create app",
        max_steps=1,
        structured_anchor_text='ASTER_PATCH_BEGIN\n{"summary":"ok","notes":[],"operations":[',
    )

    assert looks_like_patch_plan_json(extract_structured_block(reply)) is True
    continuation_events = _notice_events(logger, "reply_scroll_continuation_window")
    assert continuation_events
    assert continuation_events[-1]["reason"] == "structured_completion_progress"


def test_reply_detection_rejects_code_like_reply_while_send_button_is_still_visible() -> None:
    transport = BrowserChatGPTTransport()
    transport._executor = _FakeExecutor(
        "def build_ui(self) -> None:\n"
        "    frame = tk.Frame(self.root)\n"
        "    frame.pack(fill='both', expand=True)\n"
        "    self.display = tk.Entry(frame)\n"
        "    self.display.grid(row=0, column=0, columnspan=4)\n"
    )
    transport._capture = object()
    transport._ocr = object()
    transport._logger = None

    started = transport._reply_started(
        before_lines=[],
        after_lines=[],
        target=object(),
        ui_state={
            "show_in_text_field_present": False,
            "send_prompt_present": True,
            "send_prompt_enabled": False,
            "stop_streaming_present": True,
        },
    )

    assert started is False


def test_score_reply_candidate_prefers_patch_json_over_page_greeting() -> None:
    greeting = "Good to see you, Justin.\nCompany knowledge"
    patch = '{"summary":"ok","notes":[],"operations":[{"type":"EDIT FILE","path":"app.py","reason":"fix","content":"print(1)"}]}'
    assert score_candidate_for_policy(patch, policy=REPLY_TRACKER_POLICY) > score_candidate_for_policy(
        greeting,
        policy=REPLY_TRACKER_POLICY,
    )


def test_merge_text_segments_stitches_overlapping_reply_slices() -> None:
    merged = merge_text_segments(
        [
            "ASTER_PATCH_BEGIN\n{\n  \"summary\": \"ok\",",
            "{\n  \"summary\": \"ok\",\n  \"notes\": [],",
            "  \"notes\": [],\n  \"operations\": [{\"type\": \"RUN COMMANDS\", \"path\": \".\", \"reason\": \"verify\"}]\n}\nASTER_PATCH_END",
        ]
    )
    assert "ASTER_PATCH_BEGIN" in merged
    assert "ASTER_PATCH_END" in merged
    assert merged.count('"summary": "ok"') == 1


def test_reply_looks_incomplete_for_partial_marker_block() -> None:
    partial = 'ASTER_PATCH_BEGIN\n{"summary":"ok","operations":['
    assert reply_looks_incomplete(partial) is True


def test_segment_looks_like_prompt_echo_for_context_block() -> None:
    segment = "User goal:\nBuild app\n\nRelevant file contents:\nPATH: app.py\nREASON: source_or_related_file"
    assert segment_looks_like_prompt_echo_for_policy(segment, policy=REPLY_TRACKER_POLICY) is True


def test_reply_candidate_ignore_rejects_retry_instruction_echo() -> None:
    candidate = """
    ASTER_PATCH_BEGIN on its own line
    Do not wrap the JSON in markdown fences.
    If you are missing context, return NEED THESE FILES FIRST.
    Previous response was rejected because it was vague.
    ASTER_PATCH_END on its own line
    """
    assert should_ignore_candidate_for_policy(candidate, policy=REPLY_TRACKER_POLICY) is True


def test_reply_candidate_ignore_keeps_real_patch_json() -> None:
    candidate = '{"summary":"ok","notes":[],"operations":[{"type":"CREATE FILE","path":"app.py","reason":"add","content":"print(1)"}]}'
    assert should_ignore_candidate_for_policy(candidate, policy=REPLY_TRACKER_POLICY) is False


def test_score_reply_candidate_penalizes_input_too_large_error() -> None:
    error_text = "Input too large\nRetry"
    patch = '{"summary":"ok","notes":[],"operations":[{"type":"CREATE FILE","path":"app.py","reason":"add","content":"print(1)"}]}'
    assert score_candidate_for_policy(patch, policy=REPLY_TRACKER_POLICY) > score_candidate_for_policy(
        error_text,
        policy=REPLY_TRACKER_POLICY,
    )


def test_reply_tracker_policy_detects_patch_json() -> None:
    assert looks_like_patch_plan_json(
        '{"summary":"ok","notes":[],"operations":[{"type":"CREATE FILE","path":"app.py","reason":"add","content":"print(1)"}]}'
    ) is True


def test_clean_captured_segment_for_policy_removes_transport_noise() -> None:
    cleaned = clean_captured_segment_for_policy(
        "User goal:\nBuild app\nStop streaming\n{\"summary\":\"ok\"}",
        policy=REPLY_TRACKER_POLICY,
    )

    assert "User goal" not in cleaned
    assert "Stop streaming" not in cleaned
    assert '{"summary":"ok"}' in cleaned


def test_chatgpt_page_detection_accepts_chatgpt_hints() -> None:
    lines = [_Line("Ask anything"), _Line("Search chats")]
    ui_state = {
        "send_prompt_present": False,
        "show_in_text_field_present": False,
        "stop_streaming_present": False,
        "window_title": "ChatGPT - Google Chrome",
        "composer_edit_preview": "",
    }

    assert BrowserChatGPTTransport._looks_like_chatgpt_page(lines, ui_state) is True


def test_chatgpt_page_detection_accepts_title_plus_composer_placeholder_when_ocr_is_blank() -> None:
    lines = []
    ui_state = {
        "send_prompt_present": False,
        "show_in_text_field_present": False,
        "stop_streaming_present": False,
        "window_title": "ChatGPT - Google Chrome",
        "composer_edit_preview": "Ask anything",
    }

    assert BrowserChatGPTTransport._looks_like_chatgpt_page(lines, ui_state) is True


def test_chatgpt_page_detection_rejects_title_only_without_composer_hint() -> None:
    lines = []
    ui_state = {
        "send_prompt_present": False,
        "show_in_text_field_present": False,
        "stop_streaming_present": False,
        "window_title": "ChatGPT - Google Chrome",
        "composer_edit_preview": "Email address",
    }

    assert BrowserChatGPTTransport._looks_like_chatgpt_page(lines, ui_state) is False


def test_chatgpt_page_detection_rejects_unrelated_browser_content() -> None:
    lines = [_Line("Github"), _Line("Two Cops"), _Line("Ask Gemini")]
    ui_state = {
        "send_prompt_present": False,
        "show_in_text_field_present": False,
        "stop_streaming_present": False,
        "window_title": "Other Site - Google Chrome",
        "composer_edit_preview": "",
    }

    assert BrowserChatGPTTransport._looks_like_chatgpt_page(lines, ui_state) is False


def test_window_title_suggests_existing_thread_for_named_chat() -> None:
    assert BrowserChatGPTTransport._window_title_suggests_existing_thread("Create Hello World File - Google Chrome") is False
    assert BrowserChatGPTTransport._window_title_suggests_existing_thread("Create Hello World File - ChatGPT - Google Chrome") is True


def test_window_title_suggests_existing_thread_rejects_generic_chatgpt_title() -> None:
    assert BrowserChatGPTTransport._window_title_suggests_existing_thread("ChatGPT - Google Chrome") is False


def test_browser_url_text_detection_flags_search_urls() -> None:
    assert BrowserChatGPTTransport._looks_like_browser_url_text("https://google.com/search?q=hello") is True
    assert BrowserChatGPTTransport._looks_like_browser_url_text("Ask anything") is False


def test_analyze_screen_marks_chatgpt_ready_from_composer_and_title() -> None:
    transport = BrowserChatGPTTransport()

    class _Target:
        title = "ChatGPT - Google Chrome"

    analysis = transport._analyze_screen(
        _Target(),
        lines=[_Line("Ask anything"), _Line("Search chats")],
        ui_state={
            "window_title": "ChatGPT - Google Chrome",
            "send_prompt_present": False,
            "send_prompt_enabled": None,
            "show_in_text_field_present": False,
            "stop_streaming_present": False,
            "composer_edit_length": 12,
            "composer_edit_preview": "Ask anything",
        },
    )

    assert analysis.page_kind == "chatgpt"
    assert analysis.ready_score >= 55.0
    assert analysis.composer_ready is True
    assert analysis.score_components["chatgpt_label_bonus"] == 25.0
    assert analysis.score_components["chatgpt_surface_bonus"] == 10.0
    assert analysis.score_components["composer_hint_bonus"] == 35.0
    assert "chatgpt" in analysis.chatgpt_hint_hits
    assert "search chats" in analysis.chatgpt_surface_hits
    assert "ask anything" in analysis.composer_hint_hits


def test_analyze_screen_marks_wrong_page_when_non_chatgpt_signals_dominate() -> None:
    transport = BrowserChatGPTTransport()

    class _Target:
        title = "Other Site - Google Chrome"

    analysis = transport._analyze_screen(
        _Target(),
        lines=[_Line("Ask Gemini"), _Line("YouTube"), _Line("Amazon")],
        ui_state={
            "window_title": "Other Site - Google Chrome",
            "send_prompt_present": False,
            "send_prompt_enabled": None,
            "show_in_text_field_present": False,
            "stop_streaming_present": False,
            "composer_edit_length": None,
            "composer_edit_preview": "",
        },
    )

    assert analysis.page_kind == "wrong_page"
    assert analysis.likely_wrong_page is True
    assert analysis.ready_score < 0
    assert analysis.score_components["wrong_page_penalty"] < 0
    assert "visible_text:ask gemini:-10" in analysis.wrong_page_penalties


def test_page_readiness_failure_classifies_wrong_page() -> None:
    transport = BrowserChatGPTTransport()

    class _Target:
        title = "Other Site - Google Chrome"

    analysis = transport._analyze_screen(
        _Target(),
        lines=[_Line("Ask Gemini"), _Line("YouTube"), _Line("Pull request")],
        ui_state={
            "window_title": "Other Site - Google Chrome",
            "send_prompt_present": False,
            "send_prompt_enabled": None,
            "show_in_text_field_present": False,
            "stop_streaming_present": False,
            "composer_edit_length": None,
            "composer_edit_preview": "",
        },
    )

    failure = transport._page_readiness_failure_details(analysis)

    assert failure["code"] == "wrong_page_detected"
    assert "wrong-page" in failure["reason"].lower()
    assert "ask gemini" in failure["wrong_page_hits"]


def test_page_readiness_failure_classifies_composer_missing() -> None:
    transport = BrowserChatGPTTransport()

    class _Target:
        title = "ChatGPT - Google Chrome"

    analysis = transport._analyze_screen(
        _Target(),
        lines=[_Line("ChatGPT"), _Line("New chat")],
        ui_state={
            "window_title": "ChatGPT - Google Chrome",
            "send_prompt_present": False,
            "send_prompt_enabled": None,
            "show_in_text_field_present": False,
            "stop_streaming_present": False,
            "composer_edit_length": None,
            "composer_edit_preview": "",
        },
    )

    failure = transport._page_readiness_failure_details(analysis)

    assert failure["code"] == "composer_missing"
    assert "composer" in failure["reason"].lower()


def test_page_readiness_payload_includes_uia_control_diagnostics_when_present() -> None:
    transport = BrowserChatGPTTransport()

    class _Target:
        title = "ChatGPT - Google Chrome"

    analysis = transport._analyze_screen(
        _Target(),
        lines=[_Line("Ask anything")],
        ui_state={
            "window_title": "ChatGPT - Google Chrome",
            "send_prompt_present": False,
            "send_prompt_enabled": None,
            "show_in_text_field_present": False,
            "stop_streaming_present": False,
            "composer_edit_length": 12,
            "composer_edit_preview": "Ask anything",
            "uia_control_diagnostics": {
                "nearby_controls": [{"control_type": "Edit", "name": "Ask anything"}],
                "button_candidates": {"top_candidates": []},
                "edit_candidates": {"top_candidates": []},
            },
        },
    )

    payload = transport._page_readiness_log_payload(analysis)

    assert "uia_control_diagnostics" in payload
    assert payload["uia_control_diagnostics"]["nearby_controls"][0]["control_type"] == "Edit"


def test_page_readiness_failure_classifies_still_loading() -> None:
    transport = BrowserChatGPTTransport()

    class _Target:
        title = "ChatGPT - Google Chrome"

    analysis = transport._analyze_screen(
        _Target(),
        lines=[_Line("Loading"), _Line("Please wait")],
        ui_state={
            "window_title": "ChatGPT - Google Chrome",
            "send_prompt_present": False,
            "send_prompt_enabled": None,
            "show_in_text_field_present": False,
            "stop_streaming_present": False,
            "composer_edit_length": None,
            "composer_edit_preview": "",
        },
    )

    failure = transport._page_readiness_failure_details(analysis)

    assert analysis.loading_detected is True
    assert failure["code"] == "still_loading"
    assert analysis.score_components["loading_penalty"] == -15.0
    assert analysis.loading_penalties


def test_page_readiness_failure_classifies_likely_wrong_window() -> None:
    transport = BrowserChatGPTTransport()

    class _Target:
        title = "Downloads - Google Chrome"

    analysis = transport._analyze_screen(
        _Target(),
        lines=[_Line("Downloads"), _Line("Recent files")],
        ui_state={
            "window_title": "Downloads - Google Chrome",
            "send_prompt_present": False,
            "send_prompt_enabled": None,
            "show_in_text_field_present": False,
            "stop_streaming_present": False,
            "composer_edit_length": None,
            "composer_edit_preview": "",
        },
    )

    failure = transport._page_readiness_failure_details(analysis)

    assert failure["code"] == "likely_wrong_window"
    assert "window" in failure["reason"].lower()


def test_page_readiness_failure_falls_back_to_low_readiness_score() -> None:
    transport = BrowserChatGPTTransport()

    class _Target:
        title = "ChatGPT - Google Chrome"

    analysis = transport._analyze_screen(
        _Target(),
        lines=[],
        ui_state={
            "window_title": "ChatGPT - Google Chrome",
            "send_prompt_present": False,
            "send_prompt_enabled": None,
            "show_in_text_field_present": False,
            "stop_streaming_present": True,
            "composer_edit_length": None,
            "composer_edit_preview": "",
        },
    )

    failure = transport._page_readiness_failure_details(analysis)

    assert failure["code"] == "low_readiness_score"
    assert "threshold" in failure["reason"].lower()
    assert "composer hints (+35)" in failure["missing_signals"]
    assert "ChatGPT surface hints (+20)" in failure["missing_signals"]
    assert "send button (+20)" in failure["missing_signals"]


def test_page_readiness_log_payload_includes_failure_reason_and_loading_state() -> None:
    transport = BrowserChatGPTTransport()

    class _Target:
        title = "ChatGPT - Google Chrome"

    analysis = transport._analyze_screen(
        _Target(),
        lines=[_Line("Loading"), _Line("Please wait")],
        ui_state={
            "window_title": "ChatGPT - Google Chrome",
            "send_prompt_present": False,
            "send_prompt_enabled": None,
            "show_in_text_field_present": False,
            "stop_streaming_present": False,
            "composer_edit_length": None,
            "composer_edit_preview": "",
        },
    )

    payload = transport._page_readiness_log_payload(analysis)

    assert payload["window_title"] == "ChatGPT - Google Chrome"
    assert payload["failure_code"] == "still_loading"
    assert payload["loading_detected"] is True
    assert payload["score_components"]["loading_penalty"] == -15.0
    assert payload["loading_penalties"]
    assert payload["page_classification"]["loading_detected"] is True


def test_page_readiness_log_payload_includes_score_breakdown_and_missing_signals() -> None:
    transport = BrowserChatGPTTransport()

    class _Target:
        title = "ChatGPT - Google Chrome"

    analysis = transport._analyze_screen(
        _Target(),
        lines=[],
        ui_state={
            "window_title": "ChatGPT - Google Chrome",
            "send_prompt_present": True,
            "send_prompt_enabled": False,
            "show_in_text_field_present": False,
            "stop_streaming_present": False,
            "composer_edit_length": None,
            "composer_edit_preview": "",
        },
    )

    payload = transport._page_readiness_log_payload(analysis)

    assert payload["failure_code"] == "low_readiness_score"
    assert payload["score_components"]["chatgpt_label_bonus"] == 25.0
    assert payload["score_components"]["send_button_bonus"] == 20.0
    assert "composer hints (+35)" in payload["missing_readiness_signals"]
    assert "chatgpt" in payload["chatgpt_hint_hits"]
    assert payload["ui_state_bonuses"] == ["send_button_present"]


def test_idle_chatgpt_composer_without_send_button_is_still_ready() -> None:
    transport = BrowserChatGPTTransport()

    class _Target:
        title = "ChatGPT - Google Chrome"

    analysis = transport._analyze_screen(
        _Target(),
        lines=[],
        ui_state={
            "window_title": "ChatGPT - Google Chrome",
            "send_prompt_present": False,
            "send_prompt_enabled": None,
            "show_in_text_field_present": False,
            "stop_streaming_present": False,
            "composer_edit_length": 12,
            "composer_edit_preview": "Ask anything",
        },
    )

    assert analysis.looks_like_chatgpt is True
    assert analysis.likely_idle_composer is True
    assert analysis.idle_composer_source == "composer_preview"
    assert analysis.ready_score >= 55.0
    assert analysis.score_components["idle_composer_bonus"] == 10.0
    assert analysis.score_components["chatgpt_surface_bonus"] == 0.0
    assert "idle composer state (+10)" not in analysis.missing_readiness_signals


def test_idle_chatgpt_composer_with_ocr_placeholder_and_empty_preview_is_still_ready() -> None:
    transport = BrowserChatGPTTransport()

    class _Target:
        title = "ChatGPT - Google Chrome"

    analysis = transport._analyze_screen(
        _Target(),
        lines=[_Line("Ask anything")],
        ui_state={
            "window_title": "ChatGPT - Google Chrome",
            "send_prompt_present": False,
            "send_prompt_enabled": None,
            "show_in_text_field_present": False,
            "stop_streaming_present": False,
            "composer_edit_length": None,
            "composer_edit_preview": "",
        },
    )

    assert analysis.looks_like_chatgpt is True
    assert analysis.likely_idle_composer is True
    assert analysis.idle_composer_source == "ocr_visible_text"
    assert analysis.ready_score >= 55.0
    assert analysis.score_components["idle_composer_bonus"] == 10.0


def test_idle_chatgpt_composer_payload_explains_uia_gap() -> None:
    transport = BrowserChatGPTTransport()

    class _Target:
        title = "ChatGPT - Google Chrome"

    analysis = transport._analyze_screen(
        _Target(),
        lines=[],
        ui_state={
            "window_title": "ChatGPT - Google Chrome",
            "send_prompt_present": False,
            "send_prompt_enabled": None,
            "show_in_text_field_present": False,
            "stop_streaming_present": False,
            "composer_edit_length": 12,
            "composer_edit_preview": "Ask anything",
        },
    )
    payload = transport._screen_analysis_payload(analysis)

    assert payload["likely_idle_composer"] is True
    assert payload["idle_composer_source"] == "composer_preview"
    assert payload["send_button_absence_reason"]
    assert "UIA control-detection gap" in payload["send_button_absence_reason"]


def test_composer_visible_without_controls_gets_absence_reason_and_gaps() -> None:
    transport = BrowserChatGPTTransport()

    class _Target:
        title = "ChatGPT - Google Chrome"

    analysis = transport._analyze_screen(
        _Target(),
        lines=[_Line("Ask anything"), _Line("Search chats")],
        ui_state={
            "window_title": "ChatGPT - Google Chrome",
            "send_prompt_present": False,
            "send_prompt_enabled": None,
            "show_in_text_field_present": False,
            "stop_streaming_present": False,
            "composer_edit_length": 12,
            "composer_edit_preview": "Ask anything",
        },
    )

    assert analysis.send_button_absence_reason
    assert "send button" in analysis.send_button_absence_reason.lower()
    assert "send_button_missing" in analysis.actionable_control_gaps
    assert "no_actionable_composer_controls" in analysis.actionable_control_gaps

    payload = transport._page_readiness_log_payload(analysis)

    assert payload["send_button_absence_reason"] == analysis.send_button_absence_reason
    assert "send_button_missing" in payload["actionable_control_gaps"]
    assert payload["likely_idle_composer"] is True
    assert payload["idle_composer_source"] == "both"


def test_chatgpt_surface_hints_can_make_valid_page_ready_without_send_button() -> None:
    transport = BrowserChatGPTTransport()

    class _Target:
        title = "New chat - Google Chrome"

    analysis = transport._analyze_screen(
        _Target(),
        lines=[_Line("Ask anything"), _Line("Search chats"), _Line("New chat")],
        ui_state={
            "window_title": "New chat - Google Chrome",
            "send_prompt_present": False,
            "send_prompt_enabled": None,
            "show_in_text_field_present": False,
            "stop_streaming_present": False,
            "composer_edit_length": 12,
            "composer_edit_preview": "Ask anything",
        },
    )

    assert analysis.looks_like_chatgpt is True
    assert analysis.ready_score >= 55.0
    assert analysis.score_components["chatgpt_surface_bonus"] == 20.0


def test_stray_wrong_page_text_does_not_over_penalize_valid_chatgpt_page() -> None:
    transport = BrowserChatGPTTransport()

    class _Target:
        title = "ChatGPT - Google Chrome"

    analysis = transport._analyze_screen(
        _Target(),
        lines=[_Line("Ask anything"), _Line("Search chats"), _Line("GitHub")],
        ui_state={
            "window_title": "ChatGPT - Google Chrome",
            "send_prompt_present": False,
            "send_prompt_enabled": None,
            "show_in_text_field_present": False,
            "stop_streaming_present": False,
            "composer_edit_length": 12,
            "composer_edit_preview": "Ask anything",
        },
    )

    assert analysis.looks_like_chatgpt is True
    assert analysis.score_components["wrong_page_penalty"] == -4.0
    assert "visible_text:github:-4" in analysis.wrong_page_penalties
    assert analysis.ready_score >= 55.0


def test_wrong_page_with_placeholder_text_and_non_chatgpt_title_still_fails() -> None:
    transport = BrowserChatGPTTransport()

    class _Target:
        title = "Other Site - Google Chrome"

    analysis = transport._analyze_screen(
        _Target(),
        lines=[_Line("GitHub")],
        ui_state={
            "window_title": "Other Site - Google Chrome",
            "send_prompt_present": False,
            "send_prompt_enabled": None,
            "show_in_text_field_present": False,
            "stop_streaming_present": False,
            "composer_edit_length": 12,
            "composer_edit_preview": "Ask anything",
        },
    )
    failure = transport._page_readiness_failure_details(analysis)

    assert analysis.likely_idle_composer is False
    assert analysis.looks_like_chatgpt is False
    assert failure["code"] == "wrong_page_detected"


def test_wrong_page_with_ocr_placeholder_but_no_chatgpt_title_still_fails_readiness() -> None:
    transport = BrowserChatGPTTransport()

    class _Target:
        title = "Other Site - Google Chrome"

    analysis = transport._analyze_screen(
        _Target(),
        lines=[_Line("Ask anything"), _Line("GitHub")],
        ui_state={
            "window_title": "Other Site - Google Chrome",
            "send_prompt_present": False,
            "send_prompt_enabled": None,
            "show_in_text_field_present": False,
            "stop_streaming_present": False,
            "composer_edit_length": None,
            "composer_edit_preview": "",
        },
    )
    failure = transport._page_readiness_failure_details(analysis)

    assert analysis.likely_idle_composer is False
    assert analysis.idle_composer_source == ""
    assert analysis.ready_score < 55.0
    assert failure["code"] in {"wrong_page_detected", "low_readiness_score"}
