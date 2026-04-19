from __future__ import annotations

import re
from dataclasses import dataclass

from .models import ComposerState
from .models import TurnAnchor
from .reply_tracker import build_reply_capture_result, reply_matches_anchor


BROWSER_READY_HINTS = (
    "ask anything",
    "message chatgpt",
    "what are you working on",
    "type a message",
    "send a message",
)

DEFAULT_PROMPT_CONFIRMATION_STOPWORDS = frozenset(
    {
        "system",
        "user",
        "goal",
        "project",
        "summary",
        "relevant",
        "file",
        "files",
        "contents",
        "content",
        "conversation",
        "history",
        "constraints",
        "output",
        "exact",
        "operations",
        "browser",
        "mode",
        "return",
        "json",
        "only",
        "path",
        "reason",
        "final",
        "rules",
        "chatgpt",
        "prompt",
        "package",
        "included_files",
        "omitted_files",
    }
)


@dataclass(frozen=True, slots=True)
class ComposerVerificationPolicy:
    prompt_confirmation_stopwords: frozenset[str] = DEFAULT_PROMPT_CONFIRMATION_STOPWORDS
    prompt_word_limit: int = 12
    matched_word_threshold: int = 2
    system_match_threshold: int = 1
    anchor_min_confidence: float = 0.45


DEFAULT_COMPOSER_VERIFICATION_POLICY = ComposerVerificationPolicy()


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


def prompt_confirmation_words(
    prompt: str,
    *,
    policy: ComposerVerificationPolicy = DEFAULT_COMPOSER_VERIFICATION_POLICY,
) -> list[str]:
    words: list[str] = []
    seen: set[str] = set()
    for word in re.findall(r"[a-z0-9]{4,}", prompt.lower()):
        if word in policy.prompt_confirmation_stopwords:
            continue
        if word in seen:
            continue
        seen.add(word)
        words.append(word)
        if len(words) >= policy.prompt_word_limit:
            break
    return words


def prompt_insertion_confirmed(
    lines,
    prompt: str,
    ui_state: dict[str, object],
    *,
    prompt_anchor: TurnAnchor | None = None,
    policy: ComposerVerificationPolicy = DEFAULT_COMPOSER_VERIFICATION_POLICY,
) -> bool:
    visible_text = "\n".join(str(getattr(line, "text", line)).lower() for line in lines)
    matched = sum(1 for word in prompt_confirmation_words(prompt, policy=policy) if word in visible_text)
    state = build_composer_state(ui_state)
    if state.show_in_text_field_present:
        return True
    if composer_has_user_text(ui_state):
        return True
    if prompt_anchor is not None:
        if _text_matches_anchor(state.preview_text, "composer", prompt_anchor, min_confidence=policy.anchor_min_confidence):
            return True
        if _text_matches_anchor(visible_text, "ocr", prompt_anchor, min_confidence=policy.anchor_min_confidence):
            return True
    if matched >= policy.matched_word_threshold:
        return True
    if "system" in visible_text and matched >= policy.system_match_threshold:
        return True
    return False


def _text_matches_anchor(text: str, source: str, anchor: TurnAnchor, *, min_confidence: float) -> bool:
    if not text.strip():
        return False
    return reply_matches_anchor(
        build_reply_capture_result(text, source, 0.0, looks_complete=False, anchor=anchor),
        anchor,
        min_confidence=min_confidence,
    )


def _coerce_optional_bool(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    return None


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())
