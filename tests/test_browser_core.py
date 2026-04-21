from pathlib import Path

from aster.browser_core.composer_controller import (
    DEFAULT_COMPOSER_VERIFICATION_POLICY,
    build_composer_state,
    composer_has_user_text,
    looks_like_browser_url_text,
    prompt_confirmation_words,
    prompt_insertion_confirmed,
)
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
    assess_scrolled_segment_addition,
    choose_best_reply_candidate,
    choose_best_reply_candidate_for_policy,
    clean_captured_segment_for_policy,
    evaluate_reply_acceptance,
    extract_structured_block,
    extract_reply_from_ocr_lines,
    extract_reply_from_ocr_lines_for_policy,
    looks_like_patch_plan_json,
    looks_like_code_reply_candidate,
    looks_like_reply_started_candidate,
    looks_like_reply_started_candidate_for_policy,
    looks_like_substantive_reply_candidate,
    looks_like_substantive_reply_candidate_for_policy,
    merge_scrolled_reply_segments,
    merge_scrolled_reply_with_anchor,
    merge_reply_segment_sources,
    reply_detection_blocked,
    reply_matches_anchor,
    reply_looks_incomplete,
    scrolled_segment_looks_contaminated_for_policy,
    score_candidate,
    score_candidate_for_policy,
    select_preferred_reply_candidate,
    segment_looks_like_prompt_echo,
    segment_looks_like_prompt_echo_for_policy,
    should_ignore_candidate,
    should_ignore_candidate_for_policy,
    should_extend_structured_scroll_window,
    structured_completion_score,
    structured_completion_progress,
    summarize_reply_wait_iteration,
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


def test_build_composer_state_shapes_preview_and_button_flags() -> None:
    state = build_composer_state(
        {
            "composer_edit_length": 40,
            "composer_edit_preview": "Create a hello world file",
            "send_prompt_present": True,
            "send_prompt_enabled": True,
            "show_in_text_field_present": False,
        }
    )

    assert state.visible is True
    assert state.preview_text == "Create a hello world file"
    assert state.text_length == 40
    assert state.send_button_present is True
    assert state.send_button_enabled is True
    assert state.show_in_text_field_present is False


def test_composer_has_user_text_requires_real_non_url_text() -> None:
    assert composer_has_user_text(
        {
            "composer_edit_length": 40,
            "composer_edit_preview": "Create a hello world file",
        }
    ) is True
    assert composer_has_user_text(
        {
            "composer_edit_length": 12,
            "composer_edit_preview": "Ask anything",
        }
    ) is False
    assert composer_has_user_text(
        {
            "composer_edit_length": 120,
            "composer_edit_preview": "https://google.com/search?q=create+hello+world",
        }
    ) is False


def test_prompt_confirmation_words_prefers_specific_terms() -> None:
    words = prompt_confirmation_words(
        "SYSTEM: Return JSON only. User goal: create a hello world file named live_browser_test",
        policy=DEFAULT_COMPOSER_VERIFICATION_POLICY,
    )

    assert "hello" in words
    assert "world" in words
    assert "live" in words
    assert "test" in words
    assert "json" not in words


def test_prompt_insertion_confirmed_requires_real_composer_text_not_just_send_button() -> None:
    inserted = prompt_insertion_confirmed(
        lines=[],
        prompt="create hello world",
        ui_state={
            "show_in_text_field_present": False,
            "send_prompt_present": True,
            "send_prompt_enabled": True,
            "composer_edit_length": 12,
            "composer_edit_preview": "Ask anything",
        },
        policy=DEFAULT_COMPOSER_VERIFICATION_POLICY,
    )

    assert inserted is False


def test_prompt_insertion_confirmed_accepts_real_composer_text() -> None:
    inserted = prompt_insertion_confirmed(
        lines=[],
        prompt="create hello world",
        ui_state={
            "show_in_text_field_present": False,
            "send_prompt_present": True,
            "send_prompt_enabled": True,
            "composer_edit_length": 40,
            "composer_edit_preview": "Create a hello world file",
        },
        policy=DEFAULT_COMPOSER_VERIFICATION_POLICY,
    )

    assert inserted is True


def test_prompt_insertion_confirmed_rejects_browser_url_text_in_composer_preview() -> None:
    inserted = prompt_insertion_confirmed(
        lines=[],
        prompt="create hello world",
        ui_state={
            "show_in_text_field_present": False,
            "send_prompt_present": True,
            "send_prompt_enabled": True,
            "composer_edit_length": 120,
            "composer_edit_preview": "https://google.com/search?q=create+hello+world",
        },
        policy=DEFAULT_COMPOSER_VERIFICATION_POLICY,
    )

    assert inserted is False


def test_prompt_insertion_confirmed_ignores_generic_template_words_in_ocr() -> None:
    inserted = prompt_insertion_confirmed(
        lines=[_Line("ChatGPT"), _Line("Return JSON only"), _Line("Project summary")],
        prompt="SYSTEM: Return JSON only. User goal: create a hello world file named live_browser_test",
        ui_state={
            "show_in_text_field_present": False,
            "send_prompt_present": True,
            "send_prompt_enabled": True,
            "composer_edit_length": None,
            "composer_edit_preview": "",
        },
        policy=DEFAULT_COMPOSER_VERIFICATION_POLICY,
    )

    assert inserted is False


def test_prompt_insertion_confirmed_accepts_anchor_match_when_preview_is_partial() -> None:
    anchor = build_turn_anchor("create a hello world file named live_browser_test")
    inserted = prompt_insertion_confirmed(
        lines=[],
        prompt="create a hello world file named live_browser_test",
        ui_state={
            "show_in_text_field_present": False,
            "send_prompt_present": True,
            "send_prompt_enabled": True,
            "composer_edit_length": 21,
            "composer_edit_preview": "hello world live test",
        },
        prompt_anchor=anchor,
        policy=DEFAULT_COMPOSER_VERIFICATION_POLICY,
    )

    assert inserted is True


def test_browser_url_text_detection_flags_search_urls() -> None:
    assert looks_like_browser_url_text("https://google.com/search?q=hello") is True
    assert looks_like_browser_url_text("Ask anything") is False


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


def test_select_preferred_reply_candidate_preserves_structured_candidate_over_code_blob() -> None:
    structured = (
        "uia",
        '{"summary":"ok","notes":[],"operations":[{"type":"CREATE FILE","path":"app.py","reason":"add","content":"print(1)"}]}',
        305.0,
    )
    code_blob = (
        "ocr",
        "self.root.resizable(False, False)\n"
        "self.display_var = tk.StringVar(value='0')\n"
        "for label in ('7', '8', '9', '/'):\n"
        "    ttk.Button(frame, text=label).grid(sticky='nsew')\n"
        "return frame\n",
        420.0,
    )

    preferred = select_preferred_reply_candidate(
        structured,
        code_blob,
        extract_structured_block=extract_structured_block,
        is_patch_json=looks_like_patch_plan_json,
    )

    assert preferred == structured


def test_select_preferred_reply_candidate_allows_stronger_later_structured_candidate() -> None:
    initial = (
        "uia",
        '{"summary":"ok","notes":[],"operations":[{"type":"CREATE FILE","path":"app.py","reason":"add","content":"print(1)"}]}',
        300.0,
    )
    stronger = (
        "ocr",
        '{"summary":"ok","notes":["verified"],"operations":[{"type":"CREATE FILE","path":"app.py","reason":"add","content":"print(1)"},{"type":"RUN COMMANDS","path":".","reason":"verify","commands":["pytest"]}]}',
        330.0,
    )

    preferred = select_preferred_reply_candidate(
        initial,
        stronger,
        extract_structured_block=extract_structured_block,
        is_patch_json=looks_like_patch_plan_json,
    )

    assert preferred == stronger


def test_select_preferred_reply_candidate_rejects_malformed_patch_wrapper_against_parseable_json() -> None:
    structured = (
        "uia",
        '{"summary":"ok","notes":[],"operations":[{"type":"CREATE FILE","path":"app.py","reason":"add","content":"print(1)"}]}',
        280.0,
    )
    malformed = (
        "ocr",
        "ASTER_PATCH_BEGIN on its own line\n"
        "Create a standalone GUI calculator program with buttons.\n"
        "Operations should include CREATE FILE and RUN COMMANDS.\n",
        430.0,
    )

    preferred = select_preferred_reply_candidate(
        structured,
        malformed,
        extract_structured_block=extract_structured_block,
        is_patch_json=looks_like_patch_plan_json,
    )

    assert preferred == structured


def test_evaluate_reply_acceptance_accepts_short_visible_structured_reply() -> None:
    acceptance = evaluate_reply_acceptance(
        (
            "uia",
            '{"summary":"ok","notes":[],"operations":[{"type":"CREATE FILE","path":"app.py","reason":"add","content":"print(1)"}]}',
            305.0,
        ),
        ui_state={
            "show_in_text_field_present": False,
            "send_prompt_present": False,
            "send_prompt_enabled": None,
            "stop_streaming_present": False,
        },
        policy=TEST_REPLY_POLICY,
        stable_structured_hits=1,
    )

    assert acceptance.accepted is True
    assert acceptance.acceptance_tier == "visible_structured_short"


def test_evaluate_reply_acceptance_requires_more_observation_for_long_visible_structured_reply() -> None:
    long_structured = (
        "uia",
        '{"summary":"ok","notes":[],"operations":[' + ",".join(
            '{"type":"CREATE FILE","path":"file%d.py","reason":"add","content":"print(%d)"}' % (index, index)
            for index in range(25)
        ) + "]}",
        420.0,
    )

    pending = evaluate_reply_acceptance(
        long_structured,
        ui_state={
            "show_in_text_field_present": False,
            "send_prompt_present": False,
            "send_prompt_enabled": None,
            "stop_streaming_present": False,
        },
        policy=TEST_REPLY_POLICY,
        stable_structured_hits=0,
    )
    accepted = evaluate_reply_acceptance(
        long_structured,
        ui_state={
            "show_in_text_field_present": False,
            "send_prompt_present": False,
            "send_prompt_enabled": None,
            "stop_streaming_present": False,
        },
        policy=TEST_REPLY_POLICY,
        stable_structured_hits=1,
    )

    assert pending.accepted is False
    assert pending.acceptance_tier == "visible_structured_long"
    assert pending.requires_more_observation is True
    assert accepted.accepted is True
    assert accepted.acceptance_tier == "visible_structured_long"


def test_evaluate_reply_acceptance_requires_more_observation_for_scrolled_structured_reply() -> None:
    candidate = (
        "scrolled",
        '{"summary":"ok","notes":[],"operations":[{"type":"CREATE FILE","path":"app.py","reason":"add","content":"print(1)"}]}',
        330.0,
    )

    pending = evaluate_reply_acceptance(
        candidate,
        ui_state={
            "show_in_text_field_present": False,
            "send_prompt_present": False,
            "send_prompt_enabled": None,
            "stop_streaming_present": False,
        },
        policy=TEST_REPLY_POLICY,
        stable_structured_hits=0,
        scrolling_attempted=True,
    )
    accepted = evaluate_reply_acceptance(
        candidate,
        ui_state={
            "show_in_text_field_present": False,
            "send_prompt_present": False,
            "send_prompt_enabled": None,
            "stop_streaming_present": False,
        },
        policy=TEST_REPLY_POLICY,
        stable_structured_hits=1,
        scrolling_attempted=True,
    )

    assert pending.accepted is False
    assert pending.acceptance_tier == "scrolled_structured"
    assert pending.requires_more_observation is True
    assert accepted.accepted is True
    assert accepted.acceptance_tier == "scrolled_structured"


def test_evaluate_reply_acceptance_accepts_anchored_structured_reply() -> None:
    anchor = build_turn_anchor("create a hello world python file in live_browser_test")
    acceptance = evaluate_reply_acceptance(
        (
            "uia",
            '{"summary":"ok","notes":[],"operations":[{"type":"CREATE FILE","path":"live_browser_test/app.py","reason":"add","content":"print(\\"hello world\\")"}]}',
            312.0,
        ),
        ui_state={
            "show_in_text_field_present": False,
            "send_prompt_present": False,
            "send_prompt_enabled": None,
            "stop_streaming_present": False,
        },
        policy=TEST_REPLY_POLICY,
        prompt_anchor=anchor,
        require_anchor=True,
        stable_structured_hits=1,
    )

    assert acceptance.accepted is True
    assert acceptance.acceptance_tier == "anchored_structured"


def test_evaluate_reply_acceptance_does_not_require_anchor_for_fresh_chat_capture() -> None:
    anchor = build_turn_anchor("create a hello world python file in live_browser_test")
    acceptance = evaluate_reply_acceptance(
        (
            "uia",
            '{"summary":"ok","notes":[],"operations":[{"type":"CREATE FILE","path":"weather.py","reason":"add","content":"print(\\"rain\\")"}]}',
            300.0,
        ),
        ui_state={
            "show_in_text_field_present": False,
            "send_prompt_present": False,
            "send_prompt_enabled": None,
            "stop_streaming_present": False,
        },
        policy=TEST_REPLY_POLICY,
        prompt_anchor=anchor,
        require_anchor=False,
        stable_structured_hits=1,
    )

    assert acceptance.accepted is True
    assert acceptance.acceptance_tier == "visible_structured_short"


def test_evaluate_reply_acceptance_blocks_live_send_state() -> None:
    acceptance = evaluate_reply_acceptance(
        (
            "uia",
            '{"summary":"ok","notes":[],"operations":[{"type":"CREATE FILE","path":"app.py","reason":"add","content":"print(1)"}]}',
            305.0,
        ),
        ui_state={
            "show_in_text_field_present": False,
            "send_prompt_present": True,
            "send_prompt_enabled": True,
            "stop_streaming_present": False,
        },
        policy=TEST_REPLY_POLICY,
        stable_structured_hits=1,
    )

    assert acceptance.accepted is False
    assert acceptance.acceptance_tier == "blocked_or_ambiguous"
    assert "Composer/send controls" in acceptance.rejection_reason


def test_evaluate_reply_acceptance_blocks_prompt_echo() -> None:
    acceptance = evaluate_reply_acceptance(
        (
            "ocr",
            "User goal:\nBuild app\nRelevant file contents:\nPATH: app.py",
            250.0,
        ),
        ui_state={
            "show_in_text_field_present": False,
            "send_prompt_present": False,
            "send_prompt_enabled": None,
            "stop_streaming_present": False,
        },
        policy=TEST_REPLY_POLICY,
    )

    assert acceptance.accepted is False
    assert "Prompt-echo markers" in acceptance.rejection_reason


def test_evaluate_reply_acceptance_blocks_raw_code_without_schema_keys() -> None:
    acceptance = evaluate_reply_acceptance(
        (
            "ocr",
            "self.root.resizable(False, False)\n"
            "self.display_var = tk.StringVar(value='0')\n"
            "for label in ('7', '8', '9', '/'):\n"
            "    ttk.Button(frame, text=label).grid(sticky='nsew')\n",
            410.0,
        ),
        ui_state={
            "show_in_text_field_present": False,
            "send_prompt_present": False,
            "send_prompt_enabled": None,
            "stop_streaming_present": False,
        },
        policy=TEST_REPLY_POLICY,
    )

    assert acceptance.accepted is False
    assert "Raw code appeared" in acceptance.rejection_reason


def test_evaluate_reply_acceptance_classifies_partial_patch_block_as_incomplete_structured() -> None:
    acceptance = evaluate_reply_acceptance(
        (
            "uia",
            'ASTER_PATCH_BEGIN\n{"summary":"ok","notes":[],"operations":[{"type":"CREATE FILE","path":"app.py"',
            333.0,
        ),
        ui_state={
            "show_in_text_field_present": False,
            "send_prompt_present": False,
            "send_prompt_enabled": None,
            "stop_streaming_present": False,
        },
        policy=TEST_REPLY_POLICY,
        scrolling_attempted=False,
    )

    assert acceptance.accepted is False
    assert acceptance.acceptance_tier == "partial_structured_reply"
    assert "incomplete structured patch reply" in acceptance.rejection_reason
    assert acceptance.should_scroll is True


def test_evaluate_reply_acceptance_keeps_scroll_intent_for_longer_partial_structured_candidate() -> None:
    acceptance = evaluate_reply_acceptance(
        (
            "uia",
            "ASTER_PATCH_BEGIN\n"
            '{"summary":"ok","notes":["partial"],"operations":[{"type":"CREATE FILE","path":"calculator_app.py","reason":"add","content":"import tkinter as tk\\n'
            "from tkinter import ttk\\n"
            "class CalculatorApp:\\n"
            "    def __init__(self) -> None:\\n"
            "        self.root = tk.Tk()\\n"
            "        self.root.resizable(False, False)\\n"
            '        self.display_var = tk.StringVar(value=\\"0\\")\\n'
            '        ttk.Button(self.root, text=\\"7\\").grid(row=0, column=0)\\n',
            368.0,
        ),
        ui_state={
            "show_in_text_field_present": False,
            "send_prompt_present": False,
            "send_prompt_enabled": None,
            "stop_streaming_present": False,
        },
        policy=TEST_REPLY_POLICY,
        scrolling_attempted=False,
    )

    assert acceptance.accepted is False
    assert acceptance.acceptance_tier == "partial_structured_reply"
    assert "Raw code appeared" not in acceptance.rejection_reason
    assert acceptance.should_scroll is True


def test_evaluate_reply_acceptance_blocks_low_anchor_confidence_when_required() -> None:
    anchor = build_turn_anchor("create a hello world python file in live_browser_test")
    acceptance = evaluate_reply_acceptance(
        (
            "uia",
            '{"summary":"ok","notes":[],"operations":[{"type":"CREATE FILE","path":"weather.py","reason":"add","content":"print(\\"rain\\")"}]}',
            300.0,
        ),
        ui_state={
            "show_in_text_field_present": False,
            "send_prompt_present": False,
            "send_prompt_enabled": None,
            "stop_streaming_present": False,
        },
        policy=TEST_REPLY_POLICY,
        prompt_anchor=anchor,
        require_anchor=True,
        stable_structured_hits=1,
    )

    assert acceptance.accepted is False
    assert "Anchor confidence is too low" in acceptance.rejection_reason


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


def test_scrolled_segment_contamination_rejects_system_and_user_preamble() -> None:
    segment = (
        "SYSTEM:\nYou are a coding orchestrator backend. Return JSON only.\n"
        "USER:\nUser goal: Build app\nRelevant file contents:\nPATH: app.py\n"
    )

    assert scrolled_segment_looks_contaminated_for_policy(segment, policy=TEST_REPLY_POLICY) is True


def test_merge_scrolled_reply_with_anchor_preserves_existing_partial_over_unrelated_raw_code() -> None:
    anchor = (
        "ASTER_PATCH_BEGIN\n"
        '{"summary":"ok","notes":[],"operations":[{"type":"CREATE FILE","path":"app.py","reason":"add","content":"import tkinter as tk\\n'
    )
    raw_code = (
        "self.root.resizable(False, False)\n"
        "self.display_var = tk.StringVar(value='0')\n"
        "for label in ('7', '8', '9', '/'):\n"
        "    ttk.Button(frame, text=label).grid(sticky='nsew')\n"
    )

    merged = merge_scrolled_reply_with_anchor(anchor, raw_code, policy=TEST_REPLY_POLICY)

    assert merged == anchor


def test_merge_scrolled_reply_with_anchor_prefers_structured_extension() -> None:
    anchor = (
        "ASTER_PATCH_BEGIN\n"
        '{"summary":"ok","notes":[],"operations":[{"type":"CREATE FILE","path":"app.py","reason":"add","content":"import tkinter as tk\\n'
    )
    extension = (
        'from tkinter import ttk\\nprint(\\"ok\\")"}]}\n'
        "ASTER_PATCH_END"
    )

    merged = merge_scrolled_reply_with_anchor(anchor, extension, policy=TEST_REPLY_POLICY)

    assert "ASTER_PATCH_END" in merged
    assert '"operations"' in merged
    assert len(merged) > len(anchor)


def test_merge_scrolled_reply_segments_keeps_structured_alignment_over_page_junk() -> None:
    anchor = (
        "ASTER_PATCH_BEGIN\n"
        '{"summary":"ok","notes":[],"operations":[{"type":"CREATE FILE","path":"app.py","reason":"add","content":"import tkinter as tk\\n'
    )
    segments = [
        "SYSTEM:\nYou are a coding orchestrator backend. Return JSON only.\nUSER:\nUser goal: Build app\n",
        'from tkinter import ttk\\nprint(\\"ok\\")"}]}\nASTER_PATCH_END',
        "Projects\nSearch chats\nShare\n",
    ]

    merged = merge_scrolled_reply_segments(anchor, segments, policy=TEST_REPLY_POLICY)

    assert "SYSTEM:" not in merged
    assert "USER:" not in merged
    assert "Search chats" not in merged
    assert "ASTER_PATCH_END" in merged


def test_assess_scrolled_segment_addition_marks_completion_progress() -> None:
    anchor = (
        "ASTER_PATCH_BEGIN\n"
        '{"summary":"ok","notes":[],"operations":[{"type":"CREATE FILE","path":"app.py","reason":"add","content":"import tkinter as tk\\n'
    )
    tail = 'from tkinter import ttk\\nprint(\\"ok\\")"}]}\nASTER_PATCH_END'

    assessment = assess_scrolled_segment_addition(anchor, tail, policy=TEST_REPLY_POLICY)

    assert assessment["contributed"] is True
    assert assessment["completion_progressed"] is True
    assert assessment["after_progress"]["has_end_marker"] is True


def test_assess_scrolled_segment_addition_skips_duplicate_tail_segment() -> None:
    current = (
        "ASTER_PATCH_BEGIN\n"
        '{"summary":"ok","notes":[],"operations":[{"type":"CREATE FILE","path":"app.py","reason":"add","content":"import tkinter as tk\\n'
        'from tkinter import ttk\\nprint(\\"ok\\")"}]}\n'
        "ASTER_PATCH_END"
    )

    assessment = assess_scrolled_segment_addition(current, 'from tkinter import ttk\\nprint(\\"ok\\")"}]}\nASTER_PATCH_END', policy=TEST_REPLY_POLICY)

    assert assessment["contributed"] is False
    assert assessment["skip_reason"] in {"duplicate_overlap", "no_new_structured_content"}


def test_assess_scrolled_segment_addition_rejects_unrelated_page_content_during_scroll_merge() -> None:
    anchor = (
        "ASTER_PATCH_BEGIN\n"
        '{"summary":"ok","notes":[],"operations":[{"type":"CREATE FILE","path":"app.py","reason":"add","content":"import tkinter as tk\\n'
    )
    unrelated = (
        "tests/test_calc.py\n"
        "collected 4 items\n"
        "short test summary info\n"
        'ASTER_PATCH_BEGIN\n{"summary":"noise","notes":[],"operations":[{"type":"RUN COMMANDS","path":".","reason":"verify","commands":["pytest"]}]}\nASTER_PATCH_END'
    )

    assessment = assess_scrolled_segment_addition(anchor, unrelated, policy=TEST_REPLY_POLICY, seed_text=anchor)

    assert assessment["contributed"] is False
    assert assessment["skip_reason"] in {"unrelated_page_content", "reply_region_drift", "low_region_integrity"}
    assert assessment["unrelated_page_content"] is True
    assert "test_output" in assessment["unrelated_page_hits"] or "repo_listing" in assessment["unrelated_page_hits"]


def test_assess_scrolled_segment_addition_requires_region_integrity_not_just_completion_score() -> None:
    anchor = (
        "ASTER_PATCH_BEGIN\n"
        '{"summary":"ok","notes":[],"operations":[{"type":"CREATE FILE","path":"app.py","reason":"add","content":"import tkinter as tk\\n'
    )
    drift_segment = (
        "tests/test_calc.py\n"
        "collected 4 items\n"
        'ASTER_PATCH_BEGIN\n{"summary":"other","notes":[],"operations":[{"type":"CREATE FILE","path":"tests/test_calc.py","reason":"add","content":"def test_ok():\\n    assert 1 == 1"}]}\nASTER_PATCH_END'
    )

    assessment = assess_scrolled_segment_addition(anchor, drift_segment, policy=TEST_REPLY_POLICY, seed_text=anchor)

    assert assessment["after_progress"]["completion_score"] >= assessment["before_progress"]["completion_score"]
    assert assessment["contributed"] is False
    assert assessment["region_integrity_score"] <= 0
    assert assessment["skip_reason"] in {"unrelated_page_content", "reply_region_drift", "low_region_integrity"}


def test_merge_scrolled_reply_segments_combines_ordered_segments_into_parseable_json() -> None:
    anchor = (
        "ASTER_PATCH_BEGIN\n"
        '{"summary":"ok","notes":[],"operations":[{"type":"CREATE FILE","path":"app.py","reason":"add","content":"import tkinter as tk\\n'
    )
    segments = [
        'from tkinter import ttk\\nprint(\\"ok\\")"}',
        ']}\nASTER_PATCH_END',
    ]

    merged = merge_scrolled_reply_segments(anchor, segments, policy=TEST_REPLY_POLICY)

    assert looks_like_patch_plan_json(merged) is True


def test_merge_scrolled_reply_segments_preserves_structured_seed_over_unrelated_page_drift() -> None:
    anchor = (
        "ASTER_PATCH_BEGIN\n"
        '{"summary":"ok","notes":[],"operations":[{"type":"CREATE FILE","path":"app.py","reason":"add","content":"import tkinter as tk\\n'
    )
    segments = [
        "tests/test_calc.py\ncollected 4 items\nshort test summary info\n",
        'from tkinter import ttk\\nprint(\\"ok\\")"}]}\nASTER_PATCH_END',
    ]

    merged = merge_scrolled_reply_segments(anchor, segments, policy=TEST_REPLY_POLICY)

    assert looks_like_patch_plan_json(merged) is True
    assert "tests/test_calc.py" not in merged


def test_assess_scrolled_segment_addition_allows_completion_progress_with_small_fragment() -> None:
    anchor = (
        "ASTER_PATCH_BEGIN\n"
        '{"summary":"ok","notes":[],"operations":[{"type":"CREATE FILE","path":"app.py","reason":"add","content":"import tkinter as tk\\n'
    )
    closing_fragment = 'from tkinter import ttk\\nprint(\\"ok\\")"}]}\nASTER_PATCH_END'

    assessment = assess_scrolled_segment_addition(anchor, closing_fragment, policy=TEST_REPLY_POLICY, seed_text=anchor)

    assert assessment["contributed"] is True
    assert assessment["meaningful_completion_progress"] is True
    assert assessment["completion_score_delta"] > 0
    assert assessment["after_progress"]["parseable"] is True
    assert assessment["region_integrity_score"] > 0
    assert assessment["region_consistent"] is True


def test_structured_completion_progress_prefers_balanced_parseable_growth_over_raw_length() -> None:
    long_partial = (
        "ASTER_PATCH_BEGIN\n"
        '{"summary":"ok","notes":[],"operations":[{"type":"CREATE FILE","path":"app.py","reason":"add","content":"line 1\\n'
        "line 2\\nline 3\\nline 4\\n"
    )
    shorter_complete = (
        "ASTER_PATCH_BEGIN\n"
        '{"summary":"ok","notes":[],"operations":[{"type":"CREATE FILE","path":"app.py","reason":"add","content":"done"}]}\n'
        "ASTER_PATCH_END"
    )

    partial_progress = structured_completion_progress(long_partial)
    complete_progress = structured_completion_progress(shorter_complete)

    assert partial_progress["parseable"] is False
    assert complete_progress["parseable"] is True
    assert complete_progress["has_end_marker"] is True


def test_structured_completion_score_rewards_closure_more_than_raw_growth() -> None:
    longer_partial = (
        "ASTER_PATCH_BEGIN\n"
        '{"summary":"ok","notes":[],"operations":[{"type":"CREATE FILE","path":"app.py","reason":"add","content":"line 1\\n'
        "line 2\\nline 3\\nline 4\\nline 5\\nline 6\\n"
    )
    shorter_closer = (
        "ASTER_PATCH_BEGIN\n"
        '{"summary":"ok","notes":[],"operations":[{"type":"CREATE FILE","path":"app.py","reason":"add","content":"done"}]}\n'
    )

    assert structured_completion_score(shorter_closer) > structured_completion_score(longer_partial)


def test_should_extend_structured_scroll_window_requires_recent_structured_progress() -> None:
    partial = (
        "ASTER_PATCH_BEGIN\n"
        '{"summary":"ok","notes":[],"operations":[{"type":"CREATE FILE","path":"app.py","reason":"add","content":"done"}]\n'
    )

    assert should_extend_structured_scroll_window(
        partial,
        recent_completion_progress_steps=1,
        continuation_windows_used=0,
        step_index=1,
        step_limit=2,
        no_progress_steps=0,
    ) is True
    assert should_extend_structured_scroll_window(
        partial,
        recent_completion_progress_steps=0,
        continuation_windows_used=0,
        step_index=1,
        step_limit=2,
        no_progress_steps=0,
    ) is False


def test_should_extend_structured_scroll_window_rejects_nonstructured_junk() -> None:
    assert should_extend_structured_scroll_window(
        "def build_ui(self):\n    return frame",
        recent_completion_progress_steps=2,
        continuation_windows_used=0,
        step_index=1,
        step_limit=2,
        no_progress_steps=0,
    ) is False


def test_summarize_reply_wait_iteration_flags_stale_capture_without_candidate() -> None:
    summary = summarize_reply_wait_iteration(
        elapsed_sec=18.4,
        ui_state={
            "show_in_text_field_present": False,
            "send_prompt_present": False,
            "send_prompt_enabled": None,
            "stop_streaming_present": False,
        },
        ocr_text="",
        uia_text="",
        current_candidate=None,
        previous_best_text="",
        previous_ocr_text="",
        previous_uia_text="",
    )

    assert summary["wait_substate"] == "stale_capture_no_candidate"
    assert summary["best_candidate_source"] == ""
    assert summary["ocr_text_changed"] is False
    assert summary["uia_text_changed"] is False


def test_summarize_reply_wait_iteration_flags_streaming_without_growth() -> None:
    summary = summarize_reply_wait_iteration(
        elapsed_sec=22.0,
        ui_state={
            "show_in_text_field_present": False,
            "send_prompt_present": False,
            "send_prompt_enabled": None,
            "stop_streaming_present": True,
        },
        ocr_text="",
        uia_text="",
        current_candidate=None,
        previous_best_text="",
        previous_ocr_text="",
        previous_uia_text="",
    )

    assert summary["wait_substate"] == "streaming_without_candidate_growth"
    assert summary["stop_streaming_present"] is True


def test_summarize_reply_wait_iteration_flags_incomplete_reply_ready_for_scroll() -> None:
    summary = summarize_reply_wait_iteration(
        elapsed_sec=27.1,
        ui_state={
            "show_in_text_field_present": False,
            "send_prompt_present": False,
            "send_prompt_enabled": None,
            "stop_streaming_present": False,
        },
        ocr_text='{"summary":"ok","operations":[',
        uia_text="",
        current_candidate=("ocr", '{"summary":"ok","operations":[', 205.0),
        previous_best_text="",
        previous_ocr_text="",
        previous_uia_text="",
        scrolling_attempted=False,
    )

    assert summary["wait_substate"] == "incomplete_reply_ready_for_scroll"
    assert summary["best_candidate_source"] == "ocr"
    assert summary["candidate_looks_incomplete"] is True
    assert summary["best_text_changed"] is True


def test_summarize_reply_wait_iteration_flags_conflicting_send_and_stream_state() -> None:
    summary = summarize_reply_wait_iteration(
        elapsed_sec=31.8,
        ui_state={
            "show_in_text_field_present": False,
            "send_prompt_present": True,
            "send_prompt_enabled": True,
            "stop_streaming_present": True,
        },
        ocr_text="working",
        uia_text="working",
        current_candidate=("uia", "working", 15.0),
        previous_best_text="",
        previous_ocr_text="",
        previous_uia_text="",
    )

    assert summary["wait_substate"] == "conflicting_send_and_stream_state"
    assert summary["send_prompt_present"] is True
    assert summary["send_prompt_enabled"] is True


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
