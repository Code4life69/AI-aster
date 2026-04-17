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


def test_score_reply_candidate_penalizes_input_too_large_error() -> None:
    error_text = "Input too large\nRetry"
    patch = '{"summary":"ok","notes":[],"operations":[{"type":"CREATE FILE","path":"app.py","reason":"add","content":"print(1)"}]}'
    assert BrowserChatGPTTransport._score_reply_candidate(patch) > BrowserChatGPTTransport._score_reply_candidate(error_text)
