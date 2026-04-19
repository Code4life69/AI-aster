from pathlib import Path

from aster.browser_core.models import BrowserStrategy, ThreadRecord
from aster.browser_core.page_classifier import classify_page
from aster.browser_core.recovery_engine import (
    build_recovery_handlers,
    decide_and_execute_recovery,
    decide_recovery,
    decision_payload,
    execute_recovery,
)
from aster.browser_core.reply_tracker import (
    ReplyTrackerPolicy,
    choose_best_reply_candidate,
    choose_best_reply_candidate_for_policy,
    clean_captured_segment_for_policy,
    extract_structured_block,
    extract_reply_from_ocr_lines,
    extract_reply_from_ocr_lines_for_policy,
    looks_like_patch_plan_json,
    looks_like_code_reply_candidate,
    looks_like_reply_started_candidate,
    looks_like_reply_started_candidate_for_policy,
    looks_like_substantive_reply_candidate,
    looks_like_substantive_reply_candidate_for_policy,
    merge_reply_segment_sources,
    reply_detection_blocked,
    reply_matches_anchor,
    reply_looks_incomplete,
    score_candidate,
    score_candidate_for_policy,
    segment_looks_like_prompt_echo,
    segment_looks_like_prompt_echo_for_policy,
    should_ignore_candidate,
    should_ignore_candidate_for_policy,
    build_reply_capture_result,
)
from aster.browser_core.thread_router import ThreadRegistry
from aster.browser_core.turn_anchor import (
    anchor_match_confidence,
    anchor_matches,
    build_turn_anchor,
    deserialize_anchor,
    serialize_anchor,
)


class _Line:
    def __init__(self, text: str) -> None:
        self.text = text
        self.bbox = (200, 150, 520, 180)
        self.center = (260, 165)


TEST_REPLY_POLICY = ReplyTrackerPolicy(
    browser_reply_noise=("ask anything", "search chats", "company knowledge"),
    input_too_large_hints=("input too large", "message too long"),
    prompt_echo_markers=(
        "user goal:",
        "relevant file contents:",
        "path:",
        "do not wrap the json",
        "if you are missing context",
        "previous response was rejected",
        "aster patch begin on its own line",
        "aster patch end on its own line",
    ),
    stop_streaming_hints=("stop generating", "stop streaming"),
    penalty_markers=("good to see you", "company knowledge", "input too large"),
    operation_markers=("CREATE FILE", "EDIT FILE", "RUN COMMANDS", "NEED THESE FILES FIRST"),
    extra_bad_markers=("def _normalize", 'self.log("activity"'),
)


def test_page_classifier_detects_fresh_chat() -> None:
    classification = classify_page(
        [_Line("Ask anything"), _Line("Search chats")],
        {
            "window_title": "ChatGPT - Google Chrome",
            "send_prompt_present": False,
            "send_prompt_enabled": None,
            "show_in_text_field_present": False,
            "stop_streaming_present": False,
            "composer_edit_length": 12,
            "composer_edit_preview": "Ask anything",
        },
    )

    assert classification.looks_like_chatgpt is True
    assert classification.composer_visible is True
    assert classification.likely_fresh_chat is True
    assert classification.likely_existing_chat is False


def test_page_classifier_detects_existing_chat_from_title() -> None:
    classification = classify_page(
        [_Line("ChatGPT"), _Line("New chat")],
        {
            "window_title": "Fix Parser - ChatGPT - Google Chrome",
            "send_prompt_present": True,
            "send_prompt_enabled": True,
            "show_in_text_field_present": False,
            "stop_streaming_present": False,
            "composer_edit_length": 64,
            "composer_edit_preview": "SYSTEM: Return JSON only",
        },
    )

    assert classification.looks_like_chatgpt is True
    assert classification.likely_existing_chat is True
    assert classification.send_button_present is True


def test_turn_anchor_round_trip_and_match() -> None:
    anchor = build_turn_anchor("Create a hello world python file inside a live_browser_test folder.")
    payload = serialize_anchor(anchor)
    restored = deserialize_anchor(payload)

    confidence = anchor_match_confidence(restored, "Please create a hello world python file in live_browser_test.")

    assert restored.signature == anchor.signature
    assert confidence > 0.5
    assert anchor_matches(restored, "Please create a hello world python file in live_browser_test.") is True
    assert anchor_matches(restored, "Open the weather page in the browser.") is False


def test_thread_registry_load_save_upsert_and_choose_strategy(tmp_path: Path) -> None:
    path = tmp_path / ".aster" / "thread_registry.json"
    registry = ThreadRegistry(path)
    registry.upsert_thread(
        ThreadRecord(
            thread_id="thread-1",
            title="ChatGPT",
            url="https://chatgpt.com/c/thread-1",
            strategy_hint=BrowserStrategy.PATCH_RUNNER.value,
        )
    )
    registry.save()

    loaded = ThreadRegistry(path)
    loaded.load()

    record = loaded.get_thread("thread-1")
    assert record is not None
    assert record.url.endswith("thread-1")
    assert loaded.choose_strategy(reuse_enabled=False) == "create_new"
    assert loaded.choose_strategy(reuse_enabled=True, thread_id="thread-1") == "reuse_existing"
    assert loaded.choose_strategy(reuse_enabled=True, thread_id="missing") == "unknown"


def test_reply_tracker_prefers_patch_candidate_over_prompt_echo() -> None:
    sources = {
        "ocr": "User goal:\nBuild app\nRelevant file contents:\nPATH: app.py",
        "uia": '{"summary":"ok","notes":[],"operations":[{"type":"CREATE FILE","path":"app.py","reason":"add","content":"print(1)"}]}',
    }

    best = choose_best_reply_candidate(
        sources,
        extract_structured_block=lambda text: text,
        is_patch_json=lambda text: '"operations"' in text and text.startswith("{"),
        should_ignore_candidate=lambda text: segment_looks_like_prompt_echo(
            text,
            prompt_echo_markers=("user goal:", "relevant file contents:", "path:"),
        ),
        score_candidate=lambda text: score_candidate(
            text,
            is_patch_json=lambda candidate: '"operations"' in candidate and candidate.startswith("{"),
            prompt_echo_markers=("user goal:", "relevant file contents:", "path:"),
            stop_streaming_hints=("stop generating",),
            penalty_markers=("user goal:",),
            operation_markers=("CREATE FILE",),
        ),
    )

    assert best is not None
    assert best[0] == "uia"


def test_reply_tracker_policy_prefers_patch_candidate_over_prompt_echo() -> None:
    best = choose_best_reply_candidate_for_policy(
        {
            "ocr": "User goal:\nBuild app\nRelevant file contents:\nPATH: app.py",
            "uia": '{"summary":"ok","notes":[],"operations":[{"type":"CREATE FILE","path":"app.py","reason":"add","content":"print(1)"}]}',
        },
        policy=TEST_REPLY_POLICY,
    )

    assert best is not None
    assert best[0] == "uia"


def test_reply_tracker_extracts_structured_block_from_markers() -> None:
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


def test_reply_tracker_patch_json_detection_requires_operations() -> None:
    assert looks_like_patch_plan_json('{"summary":"ok","notes":[],"operations":[{"type":"RUN COMMANDS","path":".","reason":"verify"}]}') is True
    assert looks_like_patch_plan_json('{"summary":"ok","notes":[],"operations":[]}') is False


def test_reply_tracker_rejects_anchor_echo_candidate() -> None:
    anchor = build_turn_anchor("create a hello world python file in live_browser_test")
    candidate = "Please create a hello world python file in live_browser_test."
    capture = build_reply_capture_result(candidate, "ocr", 10.0, looks_complete=False, anchor=anchor)

    assert reply_matches_anchor(capture, anchor, min_confidence=0.45) is True


def test_reply_started_candidate_uses_extracted_helpers() -> None:
    started = looks_like_reply_started_candidate(
        (
            "def build_ui(self) -> None:\n"
            "    frame = tk.Frame(self.root)\n"
            "    frame.pack(fill='both', expand=True)\n"
            "    self._status_label = tk.Label(frame, text='Ready to run browser capture')\n"
            "    self._status_label.pack()\n"
            "    return frame\n"
        ),
        ui_state={
            "send_prompt_present": False,
            "send_prompt_enabled": None,
            "show_in_text_field_present": False,
            "stop_streaming_present": True,
        },
        score_candidate=lambda text: score_candidate(
            text,
            is_patch_json=lambda candidate: False,
            prompt_echo_markers=("user goal:",),
            stop_streaming_hints=("stop streaming",),
            penalty_markers=(),
            operation_markers=("CREATE FILE",),
        ),
        looks_like_substantive_candidate=lambda text: looks_like_substantive_reply_candidate(
            text,
            browser_reply_noise=("ask anything",),
            input_too_large_hints=("input too large",),
            prompt_echo_markers=("user goal:",),
            is_patch_json=lambda candidate: False,
        ),
        looks_like_code_candidate=looks_like_code_reply_candidate,
    )

    assert reply_detection_blocked(
        {"show_in_text_field_present": False, "send_prompt_enabled": None}
    ) is False
    assert started is True


def test_reply_started_candidate_for_policy_uses_policy_rules() -> None:
    started = looks_like_reply_started_candidate_for_policy(
        (
            "def build_ui(self) -> None:\n"
            "    frame = tk.Frame(self.root)\n"
            "    frame.pack(fill='both', expand=True)\n"
            "    self._status_label = tk.Label(frame, text='Ready to run browser capture')\n"
            "    self._status_label.pack()\n"
            "    return frame\n"
        ),
        ui_state={
            "send_prompt_present": False,
            "send_prompt_enabled": None,
            "show_in_text_field_present": False,
            "stop_streaming_present": True,
        },
        policy=TEST_REPLY_POLICY,
    )

    assert started is True


def test_reply_tracker_should_ignore_prompt_echo_candidate() -> None:
    ignored = should_ignore_candidate(
        "User goal:\nBuild app\nRelevant file contents:\nPATH: app.py",
        is_patch_json=lambda candidate: False,
        prompt_echo_markers=("user goal:", "relevant file contents:", "path:"),
    )

    assert ignored is True


def test_reply_tracker_policy_ignores_retry_instruction_echo() -> None:
    ignored = should_ignore_candidate_for_policy(
        """
        ASTER_PATCH_BEGIN on its own line
        Do not wrap the JSON in markdown fences.
        If you are missing context, return NEED THESE FILES FIRST.
        Previous response was rejected because it was vague.
        ASTER_PATCH_END on its own line
        """,
        policy=TEST_REPLY_POLICY,
    )

    assert ignored is True


def test_reply_tracker_policy_scores_patch_above_error_text() -> None:
    patch = '{"summary":"ok","notes":[],"operations":[{"type":"CREATE FILE","path":"app.py","reason":"add","content":"print(1)"}]}'
    error_text = "Input too large\nRetry"

    assert score_candidate_for_policy(patch, policy=TEST_REPLY_POLICY) > score_candidate_for_policy(
        error_text,
        policy=TEST_REPLY_POLICY,
    )


def test_reply_tracker_policy_detects_substantive_patch_reply() -> None:
    assert looks_like_substantive_reply_candidate_for_policy(
        '{"summary":"ok","notes":[],"operations":[{"type":"CREATE FILE","path":"app.py","reason":"add","content":"print(1)"}]}',
        policy=TEST_REPLY_POLICY,
    ) is True


def test_reply_tracker_extracts_ocr_reply_region_without_sidebar_noise() -> None:
    before_lines = [_Line("Ask anything")]
    sidebar_line = _Line("Search chats")
    sidebar_line.center = (20, 165)
    reply_line = _Line('{"summary":"ok","operations":[{"type":"CREATE FILE","path":"app.py"}]}')

    extracted = extract_reply_from_ocr_lines(
        before_lines,
        [sidebar_line, reply_line],
        target_width=1200,
        target_height=900,
        prompt="create app.py",
        browser_reply_noise=("search chats",),
        prompt_echo_markers=("user goal:",),
    )

    assert extracted.startswith("{")
    assert "Search chats" not in extracted


def test_reply_tracker_policy_extracts_ocr_reply_region_without_sidebar_noise() -> None:
    before_lines = [_Line("Ask anything")]
    sidebar_line = _Line("Search chats")
    sidebar_line.center = (20, 165)
    reply_line = _Line('{"summary":"ok","operations":[{"type":"CREATE FILE","path":"app.py"}]}')

    extracted = extract_reply_from_ocr_lines_for_policy(
        before_lines,
        [sidebar_line, reply_line],
        target_width=1200,
        target_height=900,
        prompt="create app.py",
        policy=TEST_REPLY_POLICY,
    )

    assert extracted.startswith("{")
    assert "Search chats" not in extracted


def test_reply_tracker_policy_cleans_and_merges_reply_segments() -> None:
    merged = merge_reply_segment_sources(
        "User goal:\nBuild app\nShare",
        '{"summary":"ok","notes":[],"operations":[{"type":"CREATE FILE","path":"app.py","reason":"add"}]}',
        policy=TEST_REPLY_POLICY,
    )

    assert "User goal" not in merged
    assert "Share" not in merged
    assert '"operations"' in merged


def test_reply_tracker_policy_detects_prompt_echo_and_incomplete_reply() -> None:
    segment = "User goal:\nBuild app\nRelevant file contents:\nPATH: app.py"
    partial = 'ASTER_PATCH_BEGIN\n{"summary":"ok","operations":['

    assert segment_looks_like_prompt_echo_for_policy(segment, policy=TEST_REPLY_POLICY) is True
    assert reply_looks_incomplete(partial) is True


def test_reply_tracker_policy_cleans_prompt_echo_lines() -> None:
    cleaned = clean_captured_segment_for_policy(
        "User goal:\nBuild app\nStop streaming\n{\"summary\":\"ok\"}",
        policy=TEST_REPLY_POLICY,
    )

    assert "User goal" not in cleaned
    assert "Stop streaming" not in cleaned
    assert '{"summary":"ok"}' in cleaned


def test_recovery_engine_executes_mapped_handler() -> None:
    classification = classify_page(
        [_Line("Ask Gemini")],
        {
            "window_title": "Other Site - Google Chrome",
            "send_prompt_present": False,
            "send_prompt_enabled": None,
            "show_in_text_field_present": False,
            "stop_streaming_present": False,
            "composer_edit_length": None,
            "composer_edit_preview": "",
        },
    )
    decision = decide_recovery(classification, attempts_used=0, max_attempts=3)
    invoked: list[str] = []

    executed = execute_recovery(
        decision,
        handlers={decision.action: lambda: invoked.append(decision.action.value)},
    )

    assert decision_payload(decision)["action"] == decision.action.value
    assert executed is True
    assert invoked == [decision.action.value]


def test_recovery_engine_decide_and_execute_uses_handler_map() -> None:
    invoked: list[str] = []

    decision = decide_and_execute_recovery(
        None,
        attempts_used=0,
        max_attempts=3,
        handlers=build_recovery_handlers(rescan=lambda: invoked.append("rescan")),
    )

    assert decision.action.value == "rescan"
    assert invoked == ["rescan"]
