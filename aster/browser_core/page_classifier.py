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

LOADING_HINTS = (
    "loading",
    "please wait",
    "just a moment",
    "one moment",
    "checking your browser",
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
    visible_chatgpt_hint_hits = tuple(_collect_hits(CHATGPT_PAGE_HINTS, visible_text))
    chatgpt_hint_hits = tuple(_collect_hits(CHATGPT_PAGE_HINTS, visible_text, window_title))
    composer_hint_hits = tuple(_collect_hits(BROWSER_READY_HINTS, visible_text, composer_preview))
    wrong_page_hits = tuple(_collect_hits(WRONG_PAGE_HINTS, visible_text, composer_preview, window_title))
    loading_hint_hits = tuple(_collect_hits(LOADING_HINTS, visible_text, composer_preview, window_title))
    composer_visible = bool(composer_hint_hits)
    looks_like_chatgpt = bool(visible_chatgpt_hint_hits) or (
        "chatgpt" in window_title and composer_visible
    ) or any((send_present, show_in_text, stop_streaming))
    loading_detected = bool(loading_hint_hits)
    wrong_page_signals_present = bool(wrong_page_hits) and "chatgpt" not in window_title and "chatgpt" not in visible_text
    likely_existing_chat = window_title_suggests_existing_chat(str(ui_state.get("window_title", "")))
    likely_fresh_chat = looks_like_chatgpt and composer_visible and not likely_existing_chat

    chatgpt_label_bonus = 25.0 if "chatgpt" in visible_text or "chatgpt" in window_title else 0.0
    composer_hint_bonus = 35.0 if composer_visible else 0.0
    send_button_bonus = 20.0 if send_present else 0.0
    stop_streaming_bonus = 18.0 if stop_streaming else 0.0
    show_in_text_bonus = 10.0 if show_in_text else 0.0
    wrong_page_penalty = -min(35.0, 10.0 * len(wrong_page_hits)) if wrong_page_hits else 0.0
    # Loading already blocks readiness in the transport, so this penalty is diagnostic rather than semantic.
    loading_penalty = -15.0 if loading_detected else 0.0
    score_components = {
        "chatgpt_label_bonus": chatgpt_label_bonus,
        "composer_hint_bonus": composer_hint_bonus,
        "send_button_bonus": send_button_bonus,
        "stop_streaming_bonus": stop_streaming_bonus,
        "show_in_text_field_bonus": show_in_text_bonus,
        "wrong_page_penalty": wrong_page_penalty,
        "loading_penalty": loading_penalty,
    }
    ready_score = sum(score_components.values())
    signals: list[str] = []
    if chatgpt_label_bonus > 0:
        signals.append("chatgpt_label")
    if composer_hint_bonus > 0:
        signals.append("composer_visible")
    if send_button_bonus > 0:
        signals.append("send_button")
    if stop_streaming_bonus > 0:
        signals.append("stop_streaming")
    if show_in_text_bonus > 0:
        signals.append("show_in_text_field")
    if likely_existing_chat:
        signals.append("likely_existing_chat")
    if likely_fresh_chat:
        signals.append("likely_fresh_chat")
    if wrong_page_hits:
        signals.extend(f"wrong_page:{hint}" for hint in wrong_page_hits[:3])
    if loading_hint_hits:
        signals.extend(f"loading:{hint}" for hint in loading_hint_hits[:2])

    ui_state_bonuses = tuple(
        name
        for name, enabled in (
            ("send_button_present", send_present),
            ("stop_streaming_present", stop_streaming),
            ("show_in_text_field_present", show_in_text),
        )
        if enabled
    )
    ui_state_penalties: tuple[str, ...] = ()
    wrong_page_penalties = tuple(f"{hint}:-10" for hint in wrong_page_hits[:3])
    loading_penalties = tuple(f"{hint}:-15" for hint in loading_hint_hits[:2])
    missing_readiness_signals = tuple(
        label
        for label, value in (
            ("composer hints (+35)", composer_hint_bonus),
            ("ChatGPT label/title (+25)", chatgpt_label_bonus),
            ("send button (+20)", send_button_bonus),
            ("stop streaming (+18)", stop_streaming_bonus),
            ("show in text field (+10)", show_in_text_bonus),
        )
        if value == 0.0
    )

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
        loading_detected=loading_detected,
        ready_score=ready_score,
        score_components=score_components,
        chatgpt_hint_hits=chatgpt_hint_hits,
        composer_hint_hits=composer_hint_hits,
        wrong_page_hits=wrong_page_hits,
        loading_hint_hits=loading_hint_hits,
        wrong_page_penalties=wrong_page_penalties,
        loading_penalties=loading_penalties,
        ui_state_bonuses=ui_state_bonuses,
        ui_state_penalties=ui_state_penalties,
        missing_readiness_signals=missing_readiness_signals,
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


def _collect_hits(hints: tuple[str, ...], *texts: str) -> list[str]:
    hits: list[str] = []
    for hint in hints:
        if any(hint in text for text in texts if text):
            hits.append(hint)
    return hits


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())
