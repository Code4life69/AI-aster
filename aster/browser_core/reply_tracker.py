from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Callable

from .models import ReplyCaptureResult, TurnAnchor
from .turn_anchor import anchor_match_confidence, anchor_matches


@dataclass(frozen=True, slots=True)
class ReplyTrackerPolicy:
    browser_reply_noise: tuple[str, ...] = ()
    input_too_large_hints: tuple[str, ...] = ()
    prompt_echo_markers: tuple[str, ...] = ()
    stop_streaming_hints: tuple[str, ...] = ()
    penalty_markers: tuple[str, ...] = ()
    operation_markers: tuple[str, ...] = ()
    extra_bad_markers: tuple[str, ...] = ()


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


def reply_detection_blocked(ui_state: dict[str, object]) -> bool:
    if ui_state.get("show_in_text_field_present"):
        return True
    return ui_state.get("send_prompt_enabled") is True


def looks_like_reply_started_candidate(
    text: str,
    *,
    ui_state: dict[str, object],
    score_candidate: Callable[[str], float],
    looks_like_substantive_candidate: Callable[[str], bool],
    looks_like_code_candidate: Callable[[str], bool],
) -> bool:
    if score_candidate(text) >= 20.0:
        return True
    if looks_like_substantive_candidate(text):
        return True
    if ui_state.get("send_prompt_present") or not ui_state.get("stop_streaming_present"):
        return False
    return looks_like_code_candidate(text)


def choose_best_reply_candidate_for_policy(
    sources: dict[str, str],
    *,
    policy: ReplyTrackerPolicy,
    prompt_anchor: TurnAnchor | None = None,
) -> tuple[str, str, float] | None:
    return choose_best_reply_candidate(
        sources,
        extract_structured_block=extract_structured_block,
        is_patch_json=looks_like_patch_plan_json,
        should_ignore_candidate=lambda candidate: should_ignore_candidate_for_policy(candidate, policy=policy),
        score_candidate=lambda candidate: score_candidate_for_policy(candidate, policy=policy),
        prompt_anchor=prompt_anchor,
    )


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


def summarize_reply_wait_iteration(
    *,
    elapsed_sec: float,
    ui_state: dict[str, object],
    ocr_text: str,
    uia_text: str,
    current_candidate: tuple[str, str, float] | None,
    previous_best_text: str = "",
    previous_ocr_text: str = "",
    previous_uia_text: str = "",
    stable_structured_hits: int = 0,
    scrolling_attempted: bool = False,
) -> dict[str, object]:
    ui_state_summary = {
        "show_in_text_field_present": bool(ui_state.get("show_in_text_field_present")),
        "send_prompt_present": bool(ui_state.get("send_prompt_present")),
        "send_prompt_enabled": ui_state.get("send_prompt_enabled"),
        "stop_streaming_present": bool(ui_state.get("stop_streaming_present")),
    }
    best_source = ""
    best_text = ""
    best_score: float | None = None
    if current_candidate is not None:
        best_source, best_text, best_score = current_candidate
    best_text_changed = _normalize(best_text) != _normalize(previous_best_text)
    ocr_text_changed = _normalize(ocr_text) != _normalize(previous_ocr_text)
    uia_text_changed = _normalize(uia_text) != _normalize(previous_uia_text)
    candidate_looks_complete = looks_like_patch_plan_json(best_text) if best_text else False
    candidate_looks_incomplete = reply_looks_incomplete(best_text) if best_text else False
    return {
        "elapsed_sec": round(elapsed_sec, 1),
        "wait_substate": _classify_reply_wait_substate(
            ui_state_summary=ui_state_summary,
            best_text=best_text,
            best_text_changed=best_text_changed,
            ocr_text_changed=ocr_text_changed,
            uia_text_changed=uia_text_changed,
            candidate_looks_complete=candidate_looks_complete,
            candidate_looks_incomplete=candidate_looks_incomplete,
            stable_structured_hits=stable_structured_hits,
            scrolling_attempted=scrolling_attempted,
        ),
        "ui_state_summary": ui_state_summary,
        "stop_streaming_present": ui_state_summary["stop_streaming_present"],
        "send_prompt_present": ui_state_summary["send_prompt_present"],
        "send_prompt_enabled": ui_state_summary["send_prompt_enabled"],
        "best_candidate_source": best_source,
        "best_candidate_score": round(best_score, 1) if best_score is not None else None,
        "best_candidate_length": len(best_text),
        "best_text_changed": best_text_changed,
        "ocr_text_changed": ocr_text_changed,
        "uia_text_changed": uia_text_changed,
        "candidate_looks_incomplete": candidate_looks_incomplete,
        "candidate_looks_complete": candidate_looks_complete,
        "stable_structured_hits": stable_structured_hits,
        "scrolling_attempted": scrolling_attempted,
    }


def extract_reply_from_ocr_lines_for_policy(
    before_lines,
    after_lines,
    *,
    target_width: int,
    target_height: int,
    prompt: str,
    policy: ReplyTrackerPolicy,
) -> str:
    return extract_reply_from_ocr_lines(
        before_lines,
        after_lines,
        target_width=target_width,
        target_height=target_height,
        prompt=prompt,
        browser_reply_noise=policy.browser_reply_noise,
        prompt_echo_markers=policy.prompt_echo_markers,
    )


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


def merge_reply_segment_sources(uia_text: str, ocr_text: str, *, policy: ReplyTrackerPolicy) -> str:
    return clean_captured_segment_for_policy(
        merge_text_segments([uia_text, ocr_text]),
        policy=policy,
    )


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


def segment_looks_like_prompt_echo_for_policy(text: str, *, policy: ReplyTrackerPolicy) -> bool:
    return segment_looks_like_prompt_echo(text, prompt_echo_markers=policy.prompt_echo_markers)


def clean_captured_segment_for_policy(text: str, *, policy: ReplyTrackerPolicy) -> str:
    return clean_captured_segment(text, prompt_echo_markers=policy.prompt_echo_markers)


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


def looks_like_substantive_reply_candidate_for_policy(text: str, *, policy: ReplyTrackerPolicy) -> bool:
    return looks_like_substantive_reply_candidate(
        text,
        browser_reply_noise=policy.browser_reply_noise,
        input_too_large_hints=policy.input_too_large_hints,
        prompt_echo_markers=policy.prompt_echo_markers,
        is_patch_json=looks_like_patch_plan_json,
        extra_bad_markers=policy.extra_bad_markers,
    )


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


def should_ignore_candidate_for_policy(text: str, *, policy: ReplyTrackerPolicy) -> bool:
    return should_ignore_candidate(
        text,
        is_patch_json=looks_like_patch_plan_json,
        prompt_echo_markers=policy.prompt_echo_markers,
    )


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


def score_candidate_for_policy(text: str, *, policy: ReplyTrackerPolicy) -> float:
    return score_candidate(
        text,
        is_patch_json=looks_like_patch_plan_json,
        prompt_echo_markers=policy.prompt_echo_markers,
        stop_streaming_hints=policy.stop_streaming_hints,
        penalty_markers=policy.penalty_markers,
        operation_markers=policy.operation_markers,
    )


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


def looks_like_reply_started_candidate_for_policy(
    text: str,
    *,
    ui_state: dict[str, object],
    policy: ReplyTrackerPolicy,
) -> bool:
    return looks_like_reply_started_candidate(
        text,
        ui_state=ui_state,
        score_candidate=lambda candidate: score_candidate_for_policy(candidate, policy=policy),
        looks_like_substantive_candidate=lambda candidate: looks_like_substantive_reply_candidate_for_policy(
            candidate,
            policy=policy,
        ),
        looks_like_code_candidate=looks_like_code_reply_candidate,
    )


def extract_structured_block(raw_text: str) -> str:
    text = raw_text.strip()
    if not text:
        return ""
    marker_match = re.search(
        r"ASTER[_ ]PATCH[_ ]BEGIN\s*(\{.*?\})\s*ASTER[_ ]PATCH[_ ]END",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if marker_match:
        return marker_match.group(1).strip()
    fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fenced:
        return fenced[-1].strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1].strip()
    return text


def looks_like_patch_plan_json(text: str) -> bool:
    candidate = extract_structured_block(text)
    try:
        data = json.loads(candidate)
    except Exception:
        return False
    if not isinstance(data, dict):
        return False
    operations = data.get("operations")
    return isinstance(operations, list) and bool(operations)


def _line_text(line) -> str:
    return str(getattr(line, "text", line))


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


def _classify_reply_wait_substate(
    *,
    ui_state_summary: dict[str, object],
    best_text: str,
    best_text_changed: bool,
    ocr_text_changed: bool,
    uia_text_changed: bool,
    candidate_looks_complete: bool,
    candidate_looks_incomplete: bool,
    stable_structured_hits: int,
    scrolling_attempted: bool,
) -> str:
    show_in_text_field_present = bool(ui_state_summary.get("show_in_text_field_present"))
    send_prompt_enabled = ui_state_summary.get("send_prompt_enabled") is True
    stop_streaming_present = bool(ui_state_summary.get("stop_streaming_present"))
    any_growth = best_text_changed or ocr_text_changed or uia_text_changed

    if show_in_text_field_present:
        return "blocked_show_in_text_field"
    if stop_streaming_present and send_prompt_enabled:
        return "conflicting_send_and_stream_state"
    if send_prompt_enabled:
        return "candidate_seen_but_send_still_available" if best_text else "send_still_available"
    if not best_text:
        if stop_streaming_present:
            return "streaming_without_candidate_growth" if not any_growth else "waiting_for_first_candidate"
        return "stale_capture_no_candidate" if not any_growth else "waiting_for_first_candidate"
    if candidate_looks_complete:
        if stop_streaming_present:
            return "structured_reply_waiting_for_stream_end"
        if stable_structured_hits > 0:
            return "structured_reply_stable"
        return "structured_reply_detected"
    if candidate_looks_incomplete:
        if not stop_streaming_present and not scrolling_attempted:
            return "incomplete_reply_ready_for_scroll"
        if not any_growth:
            return "incomplete_reply_stalled"
        return "incomplete_reply_waiting_for_completion"
    if stop_streaming_present:
        return "reply_growing" if any_growth else "streaming_without_visible_growth"
    return "candidate_waiting_for_completion" if any_growth else "candidate_stale"
