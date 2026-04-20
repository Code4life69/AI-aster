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
    summarize_controls_near_composer_area,
    summarize_edit_candidates,
    summarize_named_button_candidates,
)


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


class _FakeExecutor:
    def __init__(self, reply: str) -> None:
        self.reply = reply

    def read_chatgpt_browser_reply(self, *args, **kwargs) -> str:
        return self.reply


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
