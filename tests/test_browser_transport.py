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
