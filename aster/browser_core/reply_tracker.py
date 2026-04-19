from __future__ import annotations

import re
from typing import Callable

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


def choose_best_reply_candidate(
    sources: dict[str, str],
    *,
    extract_structured_block: Callable[[str], str],
    is_patch_json: Callable[[str], bool],
    should_ignore_candidate: Callable[[str], bool],
    score_candidate: Callable[[str], float],
    prompt_anchor: TurnAnchor | None = None,
) -> tuple[str, str, float] | None:
    best: tuple[str, str, float] | None = None
    for source, text in sources.items():
        raw = text.strip()
        if not raw:
            continue
        variants = [raw]
        block = extract_structured_block(raw)
        if block and block != raw:
            variants.insert(0, block)
        for candidate in variants:
            if should_ignore_candidate(candidate):
                continue
            score = score_candidate(candidate)
            capture = build_reply_capture_result(
                candidate,
                source,
                score,
                looks_complete=is_patch_json(candidate),
                anchor=prompt_anchor,
            )
            if prompt_anchor is not None and reply_matches_anchor(capture, prompt_anchor) and not capture.looks_complete:
                continue
            if best is None or score > best[2] or (score == best[2] and len(candidate) > len(best[1])):
                best = (source, candidate, score)
    return best


def extract_reply_from_ocr_lines(
    before_lines,
    after_lines,
    *,
    target_width: int,
    target_height: int,
    prompt: str,
    browser_reply_noise: tuple[str, ...],
    prompt_echo_markers: tuple[str, ...],
) -> str:
    before_seen = {_normalize(_line_text(line)) for line in before_lines}
    prompt_words = {word for word in re.findall(r"[a-z0-9]{4,}", prompt.lower())}
    selected: list[tuple[int, int, str]] = []
    seen: set[str] = set()

    for line in after_lines:
        text = " ".join(_line_text(line).split())
        lowered = _normalize(text)
        bbox = getattr(line, "bbox", (0, 0, 0, 0))
        center = getattr(line, "center", (0, 0))
        if not text:
            continue
        if bbox[1] < 110:
            continue
        if bbox[3] > max(0, target_height - 70):
            continue
        if center[0] < target_width * 0.18:
            continue
        if lowered in seen:
            continue
        if any(noise in lowered for noise in browser_reply_noise):
            continue
        if any(marker in lowered for marker in prompt_echo_markers):
            continue
        overlap = sum(1 for word in prompt_words if word in lowered)
        if prompt_words and overlap >= max(6, len(prompt_words) // 2) and "{" not in text and '"' not in text:
            continue
        if lowered in before_seen and "{" not in text and '"' not in text:
            continue
        selected.append((bbox[1], bbox[0], text))
        seen.add(lowered)

    if not selected:
        return ""
    selected.sort(key=lambda item: (item[0], item[1]))
    return "\n".join(text for _, _, text in selected)


def merge_text_segments(segments: list[str]) -> str:
    merged_lines: list[str] = []
    merged_norms: list[str] = []
    for segment in segments:
        lines = [line.strip() for line in segment.splitlines() if line.strip()]
        norms = [_normalize(line) for line in lines]
        if not lines:
            continue
        overlap = 0
        max_overlap = min(len(merged_norms), len(norms), 30)
        for count in range(max_overlap, 0, -1):
            if merged_norms[-count:] == norms[:count]:
                overlap = count
                break
        for line, norm in zip(lines[overlap:], norms[overlap:]):
            if merged_norms and norm == merged_norms[-1]:
                continue
            merged_lines.append(line)
            merged_norms.append(norm)
    return "\n".join(merged_lines)


def reply_looks_incomplete(text: str) -> bool:
    cleaned = text.strip()
    if not cleaned:
        return True
    lowered = _normalize(cleaned)
    if "aster patch begin" in lowered and "aster patch end" not in lowered:
        return True
    if '"operations"' in cleaned and not cleaned.rstrip().endswith("}"):
        return True
    if "thought for" in lowered:
        return True
    return False


def segment_looks_like_prompt_echo(text: str, *, prompt_echo_markers: tuple[str, ...]) -> bool:
    lowered = _normalize(text)
    if not lowered:
        return False
    hits = sum(1 for marker in prompt_echo_markers if marker in lowered)
    return hits >= 2 or ("your message" in lowered and "path:" in lowered)


def clean_captured_segment(text: str, *, prompt_echo_markers: tuple[str, ...]) -> str:
    kept: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        lowered = _normalize(line)
        if not line:
            continue
        if any(marker in lowered for marker in prompt_echo_markers):
            continue
        if lowered in {"share", "stop streaming", "stop generating"}:
            continue
        kept.append(line)
    return "\n".join(kept)


def looks_like_substantive_reply_candidate(
    text: str,
    *,
    browser_reply_noise: tuple[str, ...],
    input_too_large_hints: tuple[str, ...],
    prompt_echo_markers: tuple[str, ...],
    is_patch_json: Callable[[str], bool],
    extra_bad_markers: tuple[str, ...] = (),
) -> bool:
    cleaned = text.strip()
    if len(cleaned) < 80:
        return False
    lowered = _normalize(cleaned)
    if segment_looks_like_prompt_echo(cleaned, prompt_echo_markers=prompt_echo_markers):
        return False
    if any(noise in lowered for noise in browser_reply_noise):
        return False
    if any(hint in lowered for hint in input_too_large_hints):
        return False
    if any(marker in lowered for marker in extra_bad_markers):
        return False
    if "aster patch begin" in lowered or "aster_patch_begin" in lowered:
        return True
    if is_patch_json(cleaned):
        return True
    if '"operations"' in cleaned and "{" in cleaned and "}" in cleaned:
        return True
    return False


def looks_like_code_reply_candidate(text: str) -> bool:
    cleaned = text.strip()
    if len(cleaned) < 80:
        return False
    lines = [line.rstrip() for line in cleaned.splitlines() if line.strip()]
    if len(lines) < 3:
        return False
    signals = 0
    if sum(1 for line in lines if line.startswith(("    ", "\t"))) >= 2:
        signals += 1
    if re.search(r"\b(def|class|return|import|from|if|elif|else|for|while|try|except|with)\b", cleaned):
        signals += 1
    if re.search(r"\bself\.\w+|\w+\([^)]*\)", cleaned):
        signals += 1
    if sum(1 for line in lines if any(char in line for char in "()[]{}=:")) >= 2:
        signals += 1
    return signals >= 2


def should_ignore_candidate(
    text: str,
    *,
    is_patch_json: Callable[[str], bool],
    prompt_echo_markers: tuple[str, ...],
) -> bool:
    cleaned = text.strip()
    if not cleaned:
        return True
    if is_patch_json(cleaned):
        return False
    lowered = _normalize(cleaned)
    if segment_looks_like_prompt_echo(cleaned, prompt_echo_markers=prompt_echo_markers):
        return True
    if "aster patch begin" in lowered and not is_patch_json(cleaned):
        if "on its own line" in lowered or "do not wrap the json" in lowered:
            return True
    return False


def score_candidate(
    text: str,
    *,
    is_patch_json: Callable[[str], bool],
    prompt_echo_markers: tuple[str, ...],
    stop_streaming_hints: tuple[str, ...],
    penalty_markers: tuple[str, ...],
    operation_markers: tuple[str, ...],
) -> float:
    cleaned = text.strip()
    if not cleaned:
        return float("-inf")
    lowered = _normalize(cleaned)
    score = min(120.0, len(cleaned) / 30.0)
    if "aster patch begin" in lowered or "aster_patch_begin" in lowered:
        score += 180.0
    if '"operations"' in cleaned:
        score += 80.0
    if any(op in cleaned for op in operation_markers):
        score += 60.0
    if is_patch_json(cleaned):
        score += 220.0
    if segment_looks_like_prompt_echo(cleaned, prompt_echo_markers=prompt_echo_markers):
        score -= 260.0
    for marker in penalty_markers:
        if marker in lowered:
            score -= 180.0
    if "thought for" in lowered:
        score -= 25.0
    if any(hint in lowered for hint in stop_streaming_hints):
        score -= 40.0
    if not (
        "aster patch begin" in lowered
        or "aster_patch_begin" in lowered
        or is_patch_json(cleaned)
        or ('"operations"' in cleaned and "{" in cleaned and "}" in cleaned)
    ):
        score -= 200.0
    return score


def _line_text(line) -> str:
    return str(getattr(line, "text", line))


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())
