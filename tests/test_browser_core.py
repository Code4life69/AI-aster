from pathlib import Path

from aster.browser_core.models import BrowserStrategy, ThreadRecord
from aster.browser_core.page_classifier import classify_page
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
