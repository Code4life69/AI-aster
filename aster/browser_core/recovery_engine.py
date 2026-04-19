from __future__ import annotations

from .models import PageClassification, RecoveryAction, RecoveryDecision


def decide_recovery(
    classification: PageClassification | None,
    *,
    attempts_used: int,
    max_attempts: int,
) -> RecoveryDecision:
    if attempts_used >= max_attempts:
        return RecoveryDecision(
            action=RecoveryAction.FAIL_SAFE_STOP,
            reason="Maximum browser recovery attempts reached.",
            attempts_used=attempts_used,
            should_stop=True,
        )
    if classification is None:
        return RecoveryDecision(
            action=RecoveryAction.RESCAN,
            reason="No page classification is available yet.",
            attempts_used=attempts_used,
        )
    if classification.wrong_page_signals_present:
        return RecoveryDecision(
            action=RecoveryAction.REOPEN_CHATGPT,
            reason="Visible browser signals suggest the attached page is not ChatGPT.",
            attempts_used=attempts_used,
        )
    if not classification.looks_like_chatgpt:
        return RecoveryDecision(
            action=RecoveryAction.RELOAD_PAGE,
            reason="The attached window does not yet classify as a usable ChatGPT page.",
            attempts_used=attempts_used,
        )
    if not classification.composer_visible:
        return RecoveryDecision(
            action=RecoveryAction.REFOCUS_COMPOSER,
            reason="ChatGPT is visible but the composer does not look ready for input.",
            attempts_used=attempts_used,
        )
    return RecoveryDecision(
        action=RecoveryAction.RESCAN,
        reason="The page classification is incomplete; rescan before taking a stronger action.",
        attempts_used=attempts_used,
    )
