from __future__ import annotations

from .composer_controller import BROWSER_READY_HINTS
from .models import PageClassification


CHATGPT_PAGE_HINTS = BROWSER_READY_HINTS + (
    "search chats",
    "new chat",
    "projects",
    "gpts",
    "explore gpts",
    "chatgpt",
    "openai",
    "chatgpt can make mistakes",
)

WRONG_PAGE_HINTS = (
    "ask gemini",
    "github",
    "youtube",
    "amazon",
    "google search",
    "search?q=",
)


def classify_page(lines, ui_state: dict[str, object]) -> PageClassification:
    preview = tuple(_line_preview(lines, limit=8))
    visible_text = "\n".join(_line_preview(lines, limit=24)).lower()
    window_title = _normalize(str(ui_state.get("window_title", "")))
    composer_preview = _normalize(str(ui_state.get("composer_edit_preview", "")))
    send_present = bool(ui_state.get("send_prompt_present"))
    send_enabled = ui_state.get("send_prompt_enabled") if isinstance(ui_state.get("send_prompt_enabled"), bool) else None
    show_in_text = bool(ui_state.get("show_in_text_field_present"))
    stop_streaming = bool(ui_state.get("stop_streaming_present"))
    composer_visible = any(hint in visible_text for hint in BROWSER_READY_HINTS) or any(
        hint in composer_preview for hint in BROWSER_READY_HINTS
    )
    looks_like_chatgpt = any(hint in visible_text for hint in CHATGPT_PAGE_HINTS) or (
        "chatgpt" in window_title and composer_visible
    ) or any((send_present, show_in_text, stop_streaming))
    wrong_page_hits = [
        hint for hint in WRONG_PAGE_HINTS if hint in visible_text or hint in composer_preview or hint in window_title
    ]
    wrong_page_signals_present = bool(wrong_page_hits) and "chatgpt" not in window_title and "chatgpt" not in visible_text
    likely_existing_chat = window_title_suggests_existing_chat(str(ui_state.get("window_title", "")))
    likely_fresh_chat = looks_like_chatgpt and composer_visible and not likely_existing_chat

    ready_score = 0.0
    signals: list[str] = []
    if "chatgpt" in visible_text or "chatgpt" in window_title:
        ready_score += 25.0
        signals.append("chatgpt_label")
    if composer_visible:
        ready_score += 35.0
        signals.append("composer_visible")
    if send_present:
        ready_score += 20.0
        signals.append("send_button")
    if stop_streaming:
        ready_score += 18.0
        signals.append("stop_streaming")
    if show_in_text:
        ready_score += 10.0
        signals.append("show_in_text_field")
    if likely_existing_chat:
        signals.append("likely_existing_chat")
    if likely_fresh_chat:
        signals.append("likely_fresh_chat")
    if wrong_page_hits:
        ready_score -= min(35.0, 10.0 * len(wrong_page_hits))
        signals.extend(f"wrong_page:{hint}" for hint in wrong_page_hits[:3])

    page_kind = "unknown"
    if looks_like_chatgpt:
        page_kind = "chatgpt"
    elif wrong_page_signals_present:
        page_kind = "wrong_page"

    return PageClassification(
        looks_like_chatgpt=looks_like_chatgpt,
        composer_visible=composer_visible,
        send_button_present=send_present,
        send_button_enabled=send_enabled,
        show_in_text_field_present=show_in_text,
        stop_streaming_present=stop_streaming,
        wrong_page_signals_present=wrong_page_signals_present,
        likely_fresh_chat=likely_fresh_chat,
        likely_existing_chat=likely_existing_chat,
        composer_ready=composer_visible,
        likely_wrong_page=wrong_page_signals_present,
        ready_score=ready_score,
        page_kind=page_kind,
        visible_text=visible_text,
        signals=tuple(signals),
        ui_state=dict(ui_state),
        ocr_preview=preview,
    )


def looks_like_chatgpt_page(lines, ui_state: dict[str, object]) -> bool:
    return classify_page(lines, ui_state).looks_like_chatgpt


def window_title_suggests_existing_chat(title: str) -> bool:
    lowered = _normalize(title)
    if "chatgpt" not in lowered:
        return False
    page_title = lowered.split(" - ", 1)[0].strip()
    return bool(page_title and page_title not in {"chatgpt", "openai", "chatgpt.com"})


def _line_preview(lines, limit: int) -> list[str]:
    preview: list[str] = []
    for line in list(lines)[:limit]:
        text = getattr(line, "text", line)
        preview.append(str(text)[:120])
    return preview


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())
