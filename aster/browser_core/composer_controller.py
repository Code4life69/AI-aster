from __future__ import annotations

from .models import ComposerState


BROWSER_READY_HINTS = (
    "ask anything",
    "message chatgpt",
    "what are you working on",
    "type a message",
    "send a message",
)


def build_composer_state(ui_state: dict[str, object]) -> ComposerState:
    preview = str(ui_state.get("composer_edit_preview", "") or "")
    raw_length = ui_state.get("composer_edit_length")
    length = raw_length if isinstance(raw_length, int) else None
    visible = bool(preview.strip()) or length is not None
    return ComposerState(
        visible=visible,
        preview_text=preview,
        text_length=length,
        send_button_present=bool(ui_state.get("send_prompt_present")),
        send_button_enabled=_coerce_optional_bool(ui_state.get("send_prompt_enabled")),
        show_in_text_field_present=bool(ui_state.get("show_in_text_field_present")),
    )


def looks_like_browser_url_text(text: str) -> bool:
    lowered = _normalize(text)
    return (
        lowered.startswith(("http://", "https://", "www."))
        or "google.com/search" in lowered
        or "search?q=" in lowered
        or ("chrome" in lowered and "://" in lowered)
    )


def composer_has_user_text(ui_state: dict[str, object]) -> bool:
    state = build_composer_state(ui_state)
    preview = _normalize(state.preview_text)
    if state.text_length is None or state.text_length < 24:
        return False
    if looks_like_browser_url_text(preview):
        return False
    return not any(hint in preview for hint in BROWSER_READY_HINTS)


def _coerce_optional_bool(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    return None


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())
