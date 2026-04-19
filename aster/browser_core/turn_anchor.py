from __future__ import annotations

import hashlib
import re

from .models import TurnAnchor


ANCHOR_STOPWORDS = {
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
    "chatgpt",
    "prompt",
}


def build_turn_anchor(text: str, *, max_tokens: int = 12) -> TurnAnchor:
    normalized = _normalize(text)
    tokens: list[str] = []
    seen: set[str] = set()
    for token in re.findall(r"[a-z0-9]{4,}", normalized):
        if token in ANCHOR_STOPWORDS or token in seen:
            continue
        seen.add(token)
        tokens.append(token)
        if len(tokens) >= max_tokens:
            break
    signature = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return TurnAnchor(
        signature=signature,
        excerpt=normalized[:160],
        distinctive_tokens=tuple(tokens),
        text_length=len(normalized),
    )


def serialize_anchor(anchor: TurnAnchor) -> dict[str, object]:
    return {
        "signature": anchor.signature,
        "excerpt": anchor.excerpt,
        "distinctive_tokens": list(anchor.distinctive_tokens),
        "text_length": anchor.text_length,
    }


def deserialize_anchor(payload: dict[str, object]) -> TurnAnchor:
    return TurnAnchor(
        signature=str(payload.get("signature", "")),
        excerpt=str(payload.get("excerpt", "")),
        distinctive_tokens=tuple(str(item) for item in payload.get("distinctive_tokens", [])),
        text_length=int(payload.get("text_length", 0)),
    )


def anchor_match_confidence(anchor: TurnAnchor, candidate_text: str) -> float:
    candidate = _normalize(candidate_text)
    if not candidate or not anchor.distinctive_tokens:
        return 0.0
    if candidate == anchor.excerpt:
        return 1.0
    hits = sum(1 for token in anchor.distinctive_tokens if token in candidate)
    confidence = hits / max(1, len(anchor.distinctive_tokens))
    if anchor.excerpt and anchor.excerpt[:60] in candidate:
        confidence = max(confidence, 0.9)
    return round(confidence, 4)


def anchor_matches(anchor: TurnAnchor, candidate_text: str, *, min_confidence: float = 0.55) -> bool:
    confidence = anchor_match_confidence(anchor, candidate_text)
    hits = sum(1 for token in anchor.distinctive_tokens if token in _normalize(candidate_text))
    return confidence >= min_confidence and hits >= min(3, len(anchor.distinctive_tokens))


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())
