from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class BrowserStrategy(StrEnum):
    PATCH_RUNNER = "patch_runner"
    CONVERSATION_OPERATOR = "conversation_operator"


class RecoveryAction(StrEnum):
    RESCAN = "rescan"
    REFOCUS_COMPOSER = "refocus_composer"
    REOPEN_THREAD = "reopen_thread"
    RELOAD_PAGE = "reload_page"
    REOPEN_CHATGPT = "reopen_chatgpt"
    FAIL_SAFE_STOP = "fail_safe_stop"


@dataclass(slots=True)
class PageClassification:
    looks_like_chatgpt: bool
    composer_visible: bool
    send_button_present: bool
    send_button_enabled: bool | None
    show_in_text_field_present: bool
    stop_streaming_present: bool
    wrong_page_signals_present: bool
    likely_fresh_chat: bool
    likely_existing_chat: bool
    composer_ready: bool = False
    likely_wrong_page: bool = False
    loading_detected: bool = False
    ready_score: float = 0.0
    score_components: dict[str, float] = field(default_factory=dict)
    chatgpt_hint_hits: tuple[str, ...] = ()
    composer_hint_hits: tuple[str, ...] = ()
    wrong_page_hits: tuple[str, ...] = ()
    loading_hint_hits: tuple[str, ...] = ()
    wrong_page_penalties: tuple[str, ...] = ()
    loading_penalties: tuple[str, ...] = ()
    ui_state_bonuses: tuple[str, ...] = ()
    ui_state_penalties: tuple[str, ...] = ()
    missing_readiness_signals: tuple[str, ...] = ()
    page_kind: str = "unknown"
    visible_text: str = ""
    signals: tuple[str, ...] = ()
    ui_state: dict[str, object] = field(default_factory=dict)
    ocr_preview: tuple[str, ...] = ()


@dataclass(slots=True)
class ThreadRecord:
    thread_id: str
    title: str = ""
    url: str = ""
    strategy_hint: str = ""
    last_used_ts: str = ""
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class TurnAnchor:
    signature: str
    excerpt: str
    distinctive_tokens: tuple[str, ...]
    text_length: int


@dataclass(slots=True)
class ComposerState:
    visible: bool
    preview_text: str
    text_length: int | None
    send_button_present: bool
    send_button_enabled: bool | None
    show_in_text_field_present: bool


@dataclass(slots=True)
class ReplyCaptureResult:
    text: str
    source: str
    score: float
    looks_complete: bool
    anchor_confidence: float = 0.0


@dataclass(slots=True)
class RecoveryDecision:
    action: RecoveryAction
    reason: str
    attempts_used: int
    should_stop: bool = False
