from __future__ import annotations

from .models import ReplyCaptureResult, TurnAnchor
from .turn_anchor import anchor_match_confidence, anchor_matches


def build_reply_capture_result(
    text: str,
    source: str,
    score: float,
    *,
    looks_complete: bool,
    anchor: TurnAnchor | None = None,
) -> ReplyCaptureResult:
    anchor_confidence = anchor_match_confidence(anchor, text) if anchor is not None else 0.0
    return ReplyCaptureResult(
        text=text,
        source=source,
        score=score,
        looks_complete=looks_complete,
        anchor_confidence=anchor_confidence,
    )


def reply_matches_anchor(result: ReplyCaptureResult, anchor: TurnAnchor | None, *, min_confidence: float = 0.72) -> bool:
    if anchor is None:
        return False
    if result.anchor_confidence >= min_confidence:
        return True
    return anchor_matches(anchor, result.text, min_confidence=min_confidence)
