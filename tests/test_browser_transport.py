from aster.transport_browser.browser_transport import BrowserChatGPTTransport


def test_extract_structured_block_prefers_fenced_json() -> None:
    raw = """
    some OCR noise
    ```json
    {"summary":"ok","notes":[],"operations":[{"type":"NEED THESE FILES FIRST","path":"README.md","reason":"need context"}]}
    ```
    trailing text
    """
    parsed = BrowserChatGPTTransport.extract_structured_block(raw)
    assert parsed.startswith("{")
    assert '"summary":"ok"' in parsed


def test_extract_structured_block_falls_back_to_outer_json() -> None:
    raw = 'noise {"summary":"ok","notes":[],"operations":[{"type":"RUN COMMANDS","path":".","reason":"verify","commands":["pytest"]}]} end'
    parsed = BrowserChatGPTTransport.extract_structured_block(raw)
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
    parsed = BrowserChatGPTTransport.extract_structured_block(raw)
    assert parsed.startswith("{")
    assert '"CREATE FILE"' in parsed


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
    assert BrowserChatGPTTransport._score_reply_candidate(patch) > BrowserChatGPTTransport._score_reply_candidate(greeting)


def test_merge_text_segments_stitches_overlapping_reply_slices() -> None:
    merged = BrowserChatGPTTransport._merge_text_segments(
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
    assert BrowserChatGPTTransport._reply_looks_incomplete(partial) is True


def test_segment_looks_like_prompt_echo_for_context_block() -> None:
    segment = "User goal:\nBuild app\n\nRelevant file contents:\nPATH: app.py\nREASON: source_or_related_file"
    assert BrowserChatGPTTransport._segment_looks_like_prompt_echo(segment) is True


def test_reply_candidate_ignore_rejects_retry_instruction_echo() -> None:
    candidate = """
    ASTER_PATCH_BEGIN on its own line
    Do not wrap the JSON in markdown fences.
    If you are missing context, return NEED THESE FILES FIRST.
    Previous response was rejected because it was vague.
    ASTER_PATCH_END on its own line
    """
    assert BrowserChatGPTTransport._should_ignore_reply_candidate(candidate) is True


def test_reply_candidate_ignore_keeps_real_patch_json() -> None:
    candidate = '{"summary":"ok","notes":[],"operations":[{"type":"CREATE FILE","path":"app.py","reason":"add","content":"print(1)"}]}'
    assert BrowserChatGPTTransport._should_ignore_reply_candidate(candidate) is False


def test_score_reply_candidate_penalizes_input_too_large_error() -> None:
    error_text = "Input too large\nRetry"
    patch = '{"summary":"ok","notes":[],"operations":[{"type":"CREATE FILE","path":"app.py","reason":"add","content":"print(1)"}]}'
    assert BrowserChatGPTTransport._score_reply_candidate(patch) > BrowserChatGPTTransport._score_reply_candidate(error_text)


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


def test_prompt_inserted_requires_real_composer_text_not_just_send_button() -> None:
    transport = BrowserChatGPTTransport()
    inserted = transport._prompt_inserted(
        lines=[],
        prompt="create hello world",
        ui_state={
            "show_in_text_field_present": False,
            "send_prompt_present": True,
            "send_prompt_enabled": True,
            "composer_edit_length": 12,
            "composer_edit_preview": "Ask anything",
        },
    )

    assert inserted is False


def test_prompt_inserted_accepts_real_composer_text() -> None:
    transport = BrowserChatGPTTransport()
    inserted = transport._prompt_inserted(
        lines=[],
        prompt="create hello world",
        ui_state={
            "show_in_text_field_present": False,
            "send_prompt_present": True,
            "send_prompt_enabled": True,
            "composer_edit_length": 40,
            "composer_edit_preview": "Create a hello world file",
        },
    )

    assert inserted is True


def test_prompt_inserted_rejects_browser_url_text_in_composer_preview() -> None:
    transport = BrowserChatGPTTransport()
    inserted = transport._prompt_inserted(
        lines=[],
        prompt="create hello world",
        ui_state={
            "show_in_text_field_present": False,
            "send_prompt_present": True,
            "send_prompt_enabled": True,
            "composer_edit_length": 120,
            "composer_edit_preview": "https://google.com/search?q=create+hello+world",
        },
    )

    assert inserted is False


def test_prompt_inserted_ignores_generic_template_words_in_ocr() -> None:
    transport = BrowserChatGPTTransport()
    inserted = transport._prompt_inserted(
        lines=[_Line("ChatGPT"), _Line("Return JSON only"), _Line("Project summary")],
        prompt="SYSTEM: Return JSON only. User goal: create a hello world file named live_browser_test",
        ui_state={
            "show_in_text_field_present": False,
            "send_prompt_present": True,
            "send_prompt_enabled": True,
            "composer_edit_length": None,
            "composer_edit_preview": "",
        },
    )

    assert inserted is False


def test_prompt_confirmation_words_prefers_specific_terms() -> None:
    words = BrowserChatGPTTransport._prompt_confirmation_words(
        "SYSTEM: Return JSON only. User goal: create a hello world file named live_browser_test"
    )

    assert "hello" in words
    assert "world" in words
    assert "live" in words
    assert "test" in words
    assert "json" not in words


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
