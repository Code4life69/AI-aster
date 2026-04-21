from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Callable

from .models import ReplyAcceptanceResult, ReplyCaptureResult, TurnAnchor
from .turn_anchor import anchor_match_confidence, anchor_matches


VISIBLE_STRUCTURED_SHORT_MAX_CHARS = 1200
STRUCTURED_SCHEMA_HINTS = (
    '"summary"',
    '"notes"',
    '"operations"',
    '"type"',
    '"path"',
    '"reason"',
)
STRONG_SCROLLED_CONTAMINATION_MARKERS = (
    "browser mode has a smaller prompt budget",
    "context omitted for browser size safety",
    "included_files",
    "omitted_files",
    "return promptpackage",
    "coding orchestrator backend",
    "return json only",
    "relevant file tree:",
    "relevant file contents:",
    "conversation history:",
    "final output rules for browser mode:",
)
SCROLLED_PAGE_CHROME_MARKERS = (
    "search chats",
    "new chat",
    "projects",
    "explore gpts",
    "library",
    "chatgpt can make mistakes",
    "what are you working on",
)
SCROLLED_PAGE_NAVIGATION_LINES = {
    "code",
    "issues",
    "pull requests",
    "actions",
    "projects",
    "security",
    "insights",
    "share",
}
SCROLLED_TEST_OUTPUT_PATTERNS = (
    r"^={3,}",
    r"^platform .+ -- python ",
    r"^rootdir:",
    r"^plugins:",
    r"^collected \d+ items?",
    r"^short test summary info",
    r"^.+::.+\s+(passed|failed|error|skipped)\b",
    r"^(passed|failed|errors?) in \d",
)
SCROLLED_REPO_SOURCE_MARKERS = (
    "monkeypatch.setattr",
    "assert config.",
    'calls["',
    "calls['",
    "def test_",
    "class test",
    "from aster.",
    "import aster",
    "browserchatgpttransport",
    "replytrackerpolicy",
    "choose_best_reply_candidate",
    "build_turn_anchor",
)
VISUAL_REGION_LOW_CONFIDENCE = 0.45
VISUAL_REGION_STRONG_CONFIDENCE = 0.55


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
            best = select_preferred_reply_candidate(
                best,
                (source, candidate, score),
                extract_structured_block=extract_structured_block,
                is_patch_json=is_patch_json,
            )
    return best


def select_preferred_reply_candidate(
    current_best: tuple[str, str, float] | None,
    incoming_candidate: tuple[str, str, float] | None,
    *,
    extract_structured_block: Callable[[str], str],
    is_patch_json: Callable[[str], bool],
) -> tuple[str, str, float] | None:
    if incoming_candidate is None:
        return current_best
    if current_best is None:
        return incoming_candidate
    current_priority = _reply_candidate_priority(
        current_best[1],
        current_best[2],
        extract_structured_block=extract_structured_block,
        is_patch_json=is_patch_json,
    )
    incoming_priority = _reply_candidate_priority(
        incoming_candidate[1],
        incoming_candidate[2],
        extract_structured_block=extract_structured_block,
        is_patch_json=is_patch_json,
    )
    if incoming_priority > current_priority:
        return incoming_candidate
    return current_best


def evaluate_reply_acceptance(
    candidate: tuple[str, str, float] | None,
    *,
    ui_state: dict[str, object],
    policy: ReplyTrackerPolicy,
    prompt_anchor: TurnAnchor | None = None,
    require_anchor: bool = False,
    stable_structured_hits: int = 0,
    scrolling_attempted: bool = False,
    timed_out: bool = False,
) -> ReplyAcceptanceResult:
    if candidate is None:
        return ReplyAcceptanceResult(
            accepted=False,
            acceptance_tier="blocked_or_ambiguous",
            acceptance_reason="",
            rejection_reason="No reply candidate is available yet.",
            requires_more_observation=not timed_out,
            should_scroll=False,
        )

    source, text, score = candidate
    cleaned = text.strip()
    looks_complete = looks_like_patch_plan_json(cleaned)
    capture = build_reply_capture_result(
        cleaned,
        source,
        score,
        looks_complete=looks_complete,
        anchor=prompt_anchor,
    )
    structure_state = _reply_candidate_structure_state(
        cleaned,
        is_patch_json=looks_like_patch_plan_json,
    )

    if _ui_state_blocks_final_acceptance(ui_state):
        return ReplyAcceptanceResult(
            accepted=False,
            acceptance_tier="blocked_or_ambiguous",
            acceptance_reason="",
            rejection_reason="Composer/send controls still look live or ambiguous.",
            requires_more_observation=not timed_out,
            should_scroll=False,
        )
    if segment_looks_like_prompt_echo_for_policy(cleaned, policy=policy):
        return ReplyAcceptanceResult(
            accepted=False,
            acceptance_tier="blocked_or_ambiguous",
            acceptance_reason="",
            rejection_reason="Prompt-echo markers were detected in the candidate text.",
            requires_more_observation=not timed_out,
            should_scroll=False,
        )
    if require_anchor and prompt_anchor is not None and not reply_matches_anchor(capture, prompt_anchor, min_confidence=0.72):
        return ReplyAcceptanceResult(
            accepted=False,
            acceptance_tier="blocked_or_ambiguous",
            acceptance_reason="",
            rejection_reason="Anchor confidence is too low for a reused-thread capture.",
            requires_more_observation=not timed_out,
            should_scroll=False,
        )
    if structure_state == "partial_structured":
        return ReplyAcceptanceResult(
            accepted=False,
            acceptance_tier="partial_structured_reply",
            acceptance_reason="",
            rejection_reason="The selected candidate looks like an incomplete structured patch reply and needs more capture.",
            requires_more_observation=not timed_out,
            should_scroll=not scrolling_attempted and source != "scrolled",
        )
    if structure_state == "raw_code" and not looks_complete:
        return ReplyAcceptanceResult(
            accepted=False,
            acceptance_tier="blocked_or_ambiguous",
            acceptance_reason="",
            rejection_reason="Raw code appeared without the required structured schema keys.",
            requires_more_observation=not timed_out,
            should_scroll=False,
        )
    if not looks_complete:
        should_scroll = bool(
            reply_looks_incomplete(cleaned)
            and not scrolling_attempted
            and source != "scrolled"
        )
        return ReplyAcceptanceResult(
            accepted=False,
            acceptance_tier="blocked_or_ambiguous",
            acceptance_reason="",
            rejection_reason="The selected candidate is not yet a complete structured patch reply.",
            requires_more_observation=not timed_out,
            should_scroll=should_scroll,
        )

    tier = _reply_acceptance_tier(
        cleaned,
        source=source,
        prompt_anchor=prompt_anchor,
        require_anchor=require_anchor,
    )
    if tier == "scrolled_structured" and stable_structured_hits < 1:
        return ReplyAcceptanceResult(
            accepted=False,
            acceptance_tier=tier,
            acceptance_reason="",
            rejection_reason="Scrolled structured capture needs another confirming observation before it can be trusted.",
            requires_more_observation=not timed_out,
            should_scroll=False,
        )
    if tier == "visible_structured_long" and stable_structured_hits < 1:
        return ReplyAcceptanceResult(
            accepted=False,
            acceptance_tier=tier,
            acceptance_reason="",
            rejection_reason="Long structured replies need another matching observation before final acceptance.",
            requires_more_observation=not timed_out,
            should_scroll=False,
        )
    if tier in {"visible_structured_short", "anchored_structured"} and stable_structured_hits < 1 and not timed_out:
        return ReplyAcceptanceResult(
            accepted=False,
            acceptance_tier=tier,
            acceptance_reason="",
            rejection_reason="Structured reply needs one more matching observation before acceptance.",
            requires_more_observation=True,
            should_scroll=False,
        )

    reasons = {
        "visible_structured_short": "Short visible structured reply met the acceptance gate.",
        "visible_structured_long": "Long visible structured reply repeated and met the stronger acceptance gate.",
        "scrolled_structured": "Scrolled structured reply repeated and met the stronger acceptance gate.",
        "anchored_structured": "Structured reply matched the current turn anchor and met the acceptance gate.",
    }
    if timed_out and tier in {"visible_structured_short", "anchored_structured"} and stable_structured_hits < 1:
        reasons[tier] = "Best structured reply was still clean at timeout and is safe to accept."
    return ReplyAcceptanceResult(
        accepted=True,
        acceptance_tier=tier,
        acceptance_reason=reasons[tier],
        rejection_reason="",
        requires_more_observation=False,
        should_scroll=False,
    )


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


def merge_scrolled_reply_with_anchor(anchor_text: str, scrolled_segment: str, *, policy: ReplyTrackerPolicy) -> str:
    anchor_clean = clean_captured_segment_for_policy(anchor_text, policy=policy)
    segment_clean = clean_captured_segment_for_policy(scrolled_segment, policy=policy)
    if not segment_clean:
        return anchor_clean
    if scrolled_segment_looks_contaminated_for_policy(segment_clean, policy=policy):
        return anchor_clean
    if not anchor_clean:
        return segment_clean

    candidates = [
        anchor_clean,
        segment_clean,
        clean_captured_segment_for_policy(_merge_structured_text_by_overlap(segment_clean, anchor_clean), policy=policy),
        clean_captured_segment_for_policy(_merge_structured_text_by_overlap(anchor_clean, segment_clean), policy=policy),
    ]
    best = anchor_clean
    best_priority = _scrolled_merge_priority(best, anchor_text=anchor_clean, policy=policy)
    for candidate in candidates[1:]:
        priority = _scrolled_merge_priority(candidate, anchor_text=anchor_clean, policy=policy)
        if priority > best_priority:
            best = candidate
            best_priority = priority
    return best


def merge_scrolled_reply_segments(anchor_text: str, segments: list[str], *, policy: ReplyTrackerPolicy) -> str:
    merged = clean_captured_segment_for_policy(anchor_text, policy=policy)
    trusted_lineage = merged
    drift_containment_active = False
    for segment in segments:
        assessment = assess_scrolled_segment_addition(
            merged,
            segment,
            policy=policy,
            seed_text=anchor_text,
            trusted_lineage_text=trusted_lineage,
            drift_containment_active=drift_containment_active,
        )
        if assessment["contributed"]:
            merged = str(assessment["merged_text"])
            if assessment["trusted_lineage_extended"]:
                trusted_lineage = merged
            drift_containment_active = False
        elif assessment["activate_drift_containment"]:
            drift_containment_active = True
    return merged


def structured_completion_progress(text: str) -> dict[str, object]:
    cleaned = text.strip()
    normalized = _normalize(cleaned)
    progress = {
        "parseable": looks_like_patch_plan_json(cleaned),
        "has_begin_marker": "aster patch begin" in normalized or "aster_patch_begin" in normalized,
        "has_end_marker": "aster patch end" in normalized or "aster_patch_end" in normalized,
        "brace_balance": cleaned.count("{") - cleaned.count("}"),
        "bracket_balance": cleaned.count("[") - cleaned.count("]"),
        "schema_hits": _structured_schema_hit_count(cleaned),
        "operation_items": len(re.findall(r'"type"\s*:', cleaned)),
        "text_length": len(cleaned),
    }
    progress["completion_score"] = _completion_progress_numeric_score(progress)
    return progress


def structured_completion_score(progress_or_text: dict[str, object] | str) -> int:
    if isinstance(progress_or_text, str):
        return int(structured_completion_progress(progress_or_text)["completion_score"])
    return int(progress_or_text.get("completion_score", _completion_progress_numeric_score(progress_or_text)))


def should_extend_structured_scroll_window(
    text: str,
    *,
    recent_completion_progress_steps: int,
    continuation_windows_used: int,
    step_index: int,
    step_limit: int,
    no_progress_steps: int,
) -> bool:
    if continuation_windows_used >= 1:
        return False
    if step_index + 1 < step_limit:
        return False
    if no_progress_steps > 1:
        return False
    progress = structured_completion_progress(text)
    if progress["parseable"]:
        return False
    if not reply_looks_incomplete(text):
        return False
    if not progress["has_begin_marker"] and int(progress["schema_hits"]) < 2:
        return False
    if structured_completion_score(progress) < 24:
        return False
    return recent_completion_progress_steps > 0


def assess_scrolled_segment_addition(
    current_merged: str,
    segment: str,
    *,
    policy: ReplyTrackerPolicy,
    seed_text: str = "",
    trusted_lineage_text: str = "",
    drift_containment_active: bool = False,
    visual_region_evidence: dict[str, object] | None = None,
) -> dict[str, object]:
    current_clean = clean_captured_segment_for_policy(current_merged, policy=policy)
    before_progress = structured_completion_progress(current_clean)
    segment_clean = clean_captured_segment_for_policy(segment, policy=policy)
    result = {
        "contributed": False,
        "skip_reason": "",
        "segment_text": segment_clean,
        "merged_text": current_clean,
        "novelty_count": 0,
        "growth_chars": 0,
        "completion_progressed": False,
        "completion_score_delta": 0,
        "meaningful_completion_progress": False,
        "before_progress": before_progress,
        "after_progress": before_progress,
        "region_integrity_score": 0,
        "region_reasons": [],
        "continuity_against_seed": 0,
        "continuity_against_trusted_lineage": 0,
        "continuity_against_current": 0,
        "region_consistent": False,
        "unrelated_page_content": False,
        "unrelated_page_hits": [],
        "drift_detected": False,
        "trusted_lineage_score": 0,
        "trusted_lineage_reasons": [],
        "trusted_lineage_match": False,
        "matched_seed_lineage": False,
        "matched_trusted_lineage": False,
        "contextual_structured_extension": False,
        "contextual_extension_allowed_reason": "",
        "contextual_extension_denied_reason": "",
        "matched_current_blob_only": False,
        "trusted_lineage_extended": False,
        "activate_drift_containment": False,
        "drift_containment_active": drift_containment_active,
        "visual_region_confidence": None,
        "visual_region_used_remembered_region": False,
        "visual_region_confirmed_reply_region": False,
        "visual_region_supports_extension": False,
        "visual_region_supports_reply_region": False,
        "visual_region_low_confidence": False,
        "segment_structure_state": "empty",
    }
    if not segment_clean:
        result["skip_reason"] = "empty_segment"
        return result
    if scrolled_segment_looks_contaminated_for_policy(segment, policy=policy):
        result["skip_reason"] = "prompt_or_preamble_contamination"
        return result

    ordered_merge = clean_captured_segment_for_policy(
        _merge_structured_text_by_overlap(current_clean, segment_clean) if current_clean else segment_clean,
        policy=policy,
    )
    if not ordered_merge:
        result["skip_reason"] = "empty_after_cleaning"
        return result

    novelty_count = _ordered_segment_novelty_count(current_clean, ordered_merge)
    after_progress = structured_completion_progress(ordered_merge)
    before_rank = _completion_progress_score(before_progress)
    after_rank = _completion_progress_score(after_progress)
    completion_score_delta = structured_completion_score(after_progress) - structured_completion_score(before_progress)
    result["novelty_count"] = novelty_count
    result["growth_chars"] = max(0, len(ordered_merge) - len(current_clean))
    result["after_progress"] = after_progress
    result["completion_progressed"] = after_rank > before_rank
    result["completion_score_delta"] = completion_score_delta
    result["meaningful_completion_progress"] = bool(
        result["completion_progressed"]
        or completion_score_delta > 0
        or (
            not bool(before_progress["has_end_marker"])
            and bool(after_progress["has_end_marker"])
        )
        or (
            not bool(before_progress["parseable"])
            and bool(after_progress["parseable"])
        )
    )
    integrity = _assess_scrolled_region_integrity(
        current_text=current_clean,
        segment_text=segment_clean,
        ordered_merge=ordered_merge,
        seed_text=seed_text,
        trusted_lineage_text=trusted_lineage_text,
        before_progress=before_progress,
        after_progress=after_progress,
        drift_containment_active=drift_containment_active,
        visual_region_evidence=visual_region_evidence,
    )
    result["region_integrity_score"] = integrity["region_integrity_score"]
    result["region_reasons"] = integrity["region_reasons"]
    result["continuity_against_seed"] = integrity["continuity_against_seed"]
    result["continuity_against_trusted_lineage"] = integrity["continuity_against_trusted_lineage"]
    result["continuity_against_current"] = integrity["continuity_against_current"]
    result["region_consistent"] = integrity["region_consistent"]
    result["unrelated_page_content"] = integrity["unrelated_page_content"]
    result["unrelated_page_hits"] = integrity["unrelated_page_hits"]
    result["drift_detected"] = integrity["drift_detected"]
    result["trusted_lineage_score"] = integrity["trusted_lineage_score"]
    result["trusted_lineage_reasons"] = integrity["trusted_lineage_reasons"]
    result["trusted_lineage_match"] = integrity["trusted_lineage_match"]
    result["matched_seed_lineage"] = integrity["matched_seed_lineage"]
    result["matched_trusted_lineage"] = integrity["matched_trusted_lineage"]
    result["contextual_structured_extension"] = integrity["contextual_structured_extension"]
    result["contextual_extension_allowed_reason"] = integrity["contextual_extension_allowed_reason"]
    result["contextual_extension_denied_reason"] = integrity["contextual_extension_denied_reason"]
    result["matched_current_blob_only"] = integrity["matched_current_blob_only"]
    result["activate_drift_containment"] = integrity["activate_drift_containment"]
    result["visual_region_confidence"] = integrity["visual_region_confidence"]
    result["visual_region_used_remembered_region"] = integrity["visual_region_used_remembered_region"]
    result["visual_region_confirmed_reply_region"] = integrity["visual_region_confirmed_reply_region"]
    result["visual_region_supports_extension"] = integrity["visual_region_supports_extension"]
    result["visual_region_supports_reply_region"] = integrity["visual_region_supports_reply_region"]
    result["visual_region_low_confidence"] = integrity["visual_region_low_confidence"]
    result["segment_structure_state"] = integrity["segment_structure_state"]

    if _normalize(ordered_merge) == _normalize(current_clean):
        result["skip_reason"] = "duplicate_overlap"
        return result
    if current_clean and _scrolled_anchor_consistency(current_clean, ordered_merge) <= 0:
        result["skip_reason"] = "lost_structured_seed"
        result["activate_drift_containment"] = True
        return result
    if "duplicate_patch_start" in integrity["region_reasons"]:
        result["skip_reason"] = "reply_region_drift"
        result["activate_drift_containment"] = True
        return result
    if not integrity["trusted_lineage_match"] and not integrity["clear_structured_closure"]:
        result["skip_reason"] = (
            "current_blob_only_continuity"
            if integrity["matched_current_blob_only"]
            else "missing_seed_or_trusted_lineage"
        )
        result["activate_drift_containment"] = True
        return result
    if (
        integrity["segment_structure_state"] in {"raw_code", "unstructured"}
        and not integrity["contextual_structured_extension"]
        and not integrity["clear_structured_closure"]
    ):
        result["skip_reason"] = integrity["contextual_extension_denied_reason"] or "weak_contextual_extension"
        result["activate_drift_containment"] = True
        return result
    if drift_containment_active and integrity["trusted_lineage_score"] < 8 and not integrity["clear_structured_closure"]:
        result["skip_reason"] = "drift_containment_active"
        result["activate_drift_containment"] = True
        return result
    if integrity["unrelated_page_content"] and not integrity["region_consistent"]:
        result["skip_reason"] = "unrelated_page_content"
        result["activate_drift_containment"] = True
        return result
    if integrity["drift_detected"]:
        result["skip_reason"] = "reply_region_drift"
        result["activate_drift_containment"] = True
        return result
    if current_clean and int(integrity["region_integrity_score"]) <= 0:
        result["skip_reason"] = "low_region_integrity"
        result["activate_drift_containment"] = True
        return result
    if novelty_count <= 0 and result["growth_chars"] <= 0 and not result["meaningful_completion_progress"]:
        result["skip_reason"] = "no_new_structured_content"
        return result

    result["contributed"] = True
    result["skip_reason"] = ""
    result["merged_text"] = ordered_merge
    result["trusted_lineage_extended"] = (
        integrity["trusted_lineage_match"]
        or integrity["clear_structured_closure"]
        or integrity["contextual_structured_extension"]
    )
    return result


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


def _merge_structured_text_by_overlap(current_text: str, segment_text: str) -> str:
    current = current_text.strip()
    segment = segment_text.strip()
    if not current:
        return segment
    if not segment:
        return current
    if segment in current:
        return current
    max_overlap = min(len(current), len(segment), 800)
    overlap = 0
    for count in range(max_overlap, 0, -1):
        if current[-count:] == segment[:count]:
            overlap = count
            break
    if overlap:
        return current + segment[overlap:]
    if current.endswith(("\\n", "\\t", "\\r")):
        return current + segment
    if current.endswith(("\n", "\t", " ")):
        return current + segment
    return current + "\n" + segment


def reply_looks_incomplete(text: str) -> bool:
    cleaned = text.strip()
    if not cleaned:
        return True
    lowered = _normalize(cleaned)
    if "aster patch begin" in lowered and "aster patch end" not in lowered:
        return True
    if cleaned.count("{") > cleaned.count("}") or cleaned.count("[") > cleaned.count("]"):
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


def scrolled_segment_looks_contaminated(text: str, *, prompt_echo_markers: tuple[str, ...]) -> bool:
    cleaned = text.strip()
    if not cleaned:
        return False
    lowered = _normalize(cleaned)
    if segment_looks_like_prompt_echo(cleaned, prompt_echo_markers=prompt_echo_markers):
        return True
    if re.search(r"(^|\n)\s*(system|user|assistant)\s*:", cleaned, flags=re.IGNORECASE):
        return True
    return any(marker in lowered for marker in STRONG_SCROLLED_CONTAMINATION_MARKERS)


def scrolled_segment_looks_contaminated_for_policy(text: str, *, policy: ReplyTrackerPolicy) -> bool:
    if scrolled_segment_looks_contaminated(text, prompt_echo_markers=policy.prompt_echo_markers):
        return True
    lowered = _normalize(text.strip())
    if any(noise in lowered for noise in policy.browser_reply_noise) and not looks_like_substantive_reply_candidate_for_policy(
        text,
        policy=policy,
    ):
        return True
    return False


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


def _reply_candidate_priority(
    text: str,
    score: float,
    *,
    extract_structured_block: Callable[[str], str],
    is_patch_json: Callable[[str], bool],
) -> tuple[int, int, float, int, int, int]:
    structured_quality = _structured_reply_quality(
        text,
        extract_structured_block=extract_structured_block,
        is_patch_json=is_patch_json,
    )
    structured_block = extract_structured_block(text).strip()
    effective_text = structured_block if structured_quality >= 3 and structured_block else text.strip()
    complete_structured = 1 if structured_quality >= 3 else 0
    clean_structured_json = 1 if complete_structured and effective_text == text.strip() else 0
    return (
        structured_quality,
        clean_structured_json,
        score,
        len(effective_text),
        complete_structured,
        len(text.strip()),
    )


def _scrolled_merge_priority(
    text: str,
    *,
    anchor_text: str,
    policy: ReplyTrackerPolicy,
) -> tuple[int, int, int, int, int, int]:
    cleaned = text.strip()
    if not cleaned:
        return (-1, -1, -1, -1, -1, -1)
    if scrolled_segment_looks_contaminated_for_policy(cleaned, policy=policy):
        return (-1, -1, -1, -1, -1, -1)
    structure_state = _reply_candidate_structure_state(cleaned, is_patch_json=looks_like_patch_plan_json)
    state_rank = {
        "empty": 0,
        "unstructured": 1,
        "raw_code": 1,
        "partial_structured": 3,
        "complete_structured": 4,
    }[structure_state]
    normalized = _normalize(cleaned)
    has_end_marker = int("aster patch end" in normalized or "aster_patch_end" in normalized)
    schema_hits = _structured_schema_hit_count(cleaned)
    anchor_consistency = _scrolled_anchor_consistency(anchor_text, cleaned)
    starts_with_structured = int(
        normalized.startswith("aster patch begin")
        or normalized.startswith("aster_patch_begin")
        or normalized.startswith("{")
        or normalized.startswith('"summary"')
    )
    return (
        int(looks_like_patch_plan_json(cleaned)),
        state_rank,
        starts_with_structured,
        anchor_consistency,
        has_end_marker,
        schema_hits,
    )


def _completion_progress_score(progress: dict[str, object]) -> tuple[int, int, int, int, int, int]:
    brace_distance = abs(int(progress["brace_balance"]))
    bracket_distance = abs(int(progress["bracket_balance"]))
    return (
        int(bool(progress["parseable"])),
        int(bool(progress["has_end_marker"])),
        -brace_distance,
        -bracket_distance,
        int(progress["schema_hits"]),
        int(progress["operation_items"]),
    )


def _completion_progress_numeric_score(progress: dict[str, object]) -> int:
    brace_distance = abs(int(progress["brace_balance"]))
    bracket_distance = abs(int(progress["bracket_balance"]))
    score = 0
    if progress["has_begin_marker"]:
        score += 8
    if progress["has_end_marker"]:
        score += 16
    if progress["parseable"]:
        score += 80
    score += min(int(progress["schema_hits"]), 6) * 6
    score += min(int(progress["operation_items"]), 8) * 4
    score += max(0, 18 - brace_distance * 4)
    score += max(0, 16 - bracket_distance * 5)
    return score


def _scrolled_visual_region_support(visual_region_evidence: dict[str, object] | None) -> dict[str, object]:
    if not visual_region_evidence:
        return {
            "has_visual_region_evidence": False,
            "used_remembered_region": False,
            "visual_region_confidence": None,
            "confirmed_reply_region": False,
            "supports_reply_region": False,
            "strong_visual_region_support": False,
            "weak_visual_region_support": False,
        }
    confidence_raw = visual_region_evidence.get("visual_region_confidence")
    confidence = None if confidence_raw is None else float(confidence_raw)
    used_remembered_region = bool(visual_region_evidence.get("used_remembered_region"))
    confirmed_reply_region = bool(visual_region_evidence.get("confirmed_reply_region"))
    supports_reply_region = bool(
        visual_region_evidence.get("supports_reply_region")
        or confirmed_reply_region
        or (confidence is not None and confidence >= VISUAL_REGION_STRONG_CONFIDENCE)
    )
    weak_visual_region_support = bool(
        used_remembered_region
        and not confirmed_reply_region
        and confidence is not None
        and confidence < VISUAL_REGION_LOW_CONFIDENCE
    )
    strong_visual_region_support = bool(
        confirmed_reply_region
        or (confidence is not None and confidence >= VISUAL_REGION_STRONG_CONFIDENCE)
    )
    return {
        "has_visual_region_evidence": True,
        "used_remembered_region": used_remembered_region,
        "visual_region_confidence": confidence,
        "confirmed_reply_region": confirmed_reply_region,
        "supports_reply_region": supports_reply_region,
        "strong_visual_region_support": strong_visual_region_support,
        "weak_visual_region_support": weak_visual_region_support,
    }


def _assess_scrolled_region_integrity(
    *,
    current_text: str,
    segment_text: str,
    ordered_merge: str,
    seed_text: str,
    trusted_lineage_text: str,
    before_progress: dict[str, object],
    after_progress: dict[str, object],
    drift_containment_active: bool,
    visual_region_evidence: dict[str, object] | None,
) -> dict[str, object]:
    current_clean = current_text.strip()
    seed_clean = seed_text.strip() or current_clean
    trusted_lineage_clean = trusted_lineage_text.strip() or seed_clean or current_clean
    segment_clean = segment_text.strip()
    normalized_segment = _normalize(segment_clean)
    current_consistency = _scrolled_anchor_consistency(current_clean, segment_clean) if current_clean else 0
    seed_consistency = _scrolled_anchor_consistency(seed_clean, segment_clean) if seed_clean else 0
    trusted_lineage_consistency = (
        _scrolled_anchor_consistency(trusted_lineage_clean, segment_clean) if trusted_lineage_clean else 0
    )
    tail_overlap = _suffix_prefix_overlap(_normalize(trusted_lineage_clean or seed_clean or current_clean), normalized_segment)
    unrelated_page_hits = _scrolled_unrelated_page_hits(segment_clean)
    duplicate_patch_start = bool(before_progress["has_begin_marker"] and _has_begin_marker(segment_clean))
    segment_structure_state = _reply_candidate_structure_state(
        segment_clean,
        is_patch_json=looks_like_patch_plan_json,
    )
    completion_progress = bool(
        (not bool(before_progress["has_end_marker"]) and bool(after_progress["has_end_marker"]))
        or (not bool(before_progress["parseable"]) and bool(after_progress["parseable"]))
        or int(after_progress["schema_hits"]) > int(before_progress["schema_hits"])
        or int(after_progress["operation_items"]) > int(before_progress["operation_items"])
    )
    visual_region = _scrolled_visual_region_support(visual_region_evidence)
    lineage_match_without_context = bool(
        seed_consistency > 0
        or trusted_lineage_consistency > 0
        or tail_overlap >= 24
    )
    clear_structured_closure = bool(
        (not bool(before_progress["has_end_marker"]) and bool(after_progress["has_end_marker"]))
        or (not bool(before_progress["parseable"]) and bool(after_progress["parseable"]))
    )
    contextual_structured_extension = False
    contextual_extension_allowed_reason = ""
    contextual_extension_denied_reason = ""
    base_contextual_conditions = bool(
        not drift_containment_active
        and not unrelated_page_hits
        and not duplicate_patch_start
        and current_clean
        and _reply_candidate_structure_state(current_clean, is_patch_json=looks_like_patch_plan_json) == "partial_structured"
        and segment_structure_state in {"raw_code", "unstructured", "partial_structured"}
        and len(segment_clean) >= 24
    )
    if not bool(visual_region["has_visual_region_evidence"]):
        contextual_structured_extension = base_contextual_conditions
        if contextual_structured_extension:
            contextual_extension_allowed_reason = "no_visual_region_evidence"
    elif drift_containment_active:
        contextual_extension_denied_reason = "drift_containment_active"
    elif unrelated_page_hits:
        contextual_extension_denied_reason = "unrelated_page_content"
    elif duplicate_patch_start:
        contextual_extension_denied_reason = "duplicate_patch_start"
    elif not current_clean:
        contextual_extension_denied_reason = "missing_current_structured_seed"
    elif _reply_candidate_structure_state(current_clean, is_patch_json=looks_like_patch_plan_json) != "partial_structured":
        contextual_extension_denied_reason = "current_not_partial_structured"
    elif segment_structure_state not in {"raw_code", "unstructured", "partial_structured"}:
        contextual_extension_denied_reason = "segment_not_extension_candidate"
    elif len(segment_clean) < 24:
        contextual_extension_denied_reason = "segment_too_short"
    elif not lineage_match_without_context:
        contextual_extension_denied_reason = "missing_seed_or_trusted_lineage"
    elif (
        segment_structure_state in {"raw_code", "unstructured"}
        and bool(visual_region["used_remembered_region"])
        and not bool(visual_region["strong_visual_region_support"])
    ):
        contextual_extension_denied_reason = "visual_region_support_required"
    else:
        contextual_structured_extension = base_contextual_conditions
        if contextual_structured_extension:
            contextual_extension_allowed_reason = (
                "trusted_lineage_with_visual_region_support"
                if segment_structure_state in {"raw_code", "unstructured"}
                else "trusted_lineage_partial_structured_extension"
            )
    trusted_lineage_match = bool(
        lineage_match_without_context
        or contextual_structured_extension
    )
    current_blob_only_match = bool(
        current_consistency > 0 and not trusted_lineage_match
    )
    structured_continuation = bool(
        trusted_lineage_match
        or (completion_progress and not unrelated_page_hits)
        or clear_structured_closure
    )

    trusted_lineage_score = 0
    trusted_lineage_reasons: list[str] = []
    if seed_consistency > 0:
        trusted_lineage_score += seed_consistency * 5
        trusted_lineage_reasons.append("seed_reply_continuity")
    if trusted_lineage_consistency > 0:
        trusted_lineage_score += trusted_lineage_consistency * 4
        trusted_lineage_reasons.append("trusted_lineage_continuity")
    if tail_overlap >= 24:
        trusted_lineage_score += min(4, max(1, tail_overlap // 32))
        trusted_lineage_reasons.append("trusted_lineage_overlap")
    if clear_structured_closure and structured_continuation:
        trusted_lineage_score += 6
        trusted_lineage_reasons.append("clear_structured_closure")
    elif completion_progress and trusted_lineage_match:
        trusted_lineage_score += 3
        trusted_lineage_reasons.append("structured_completion_progress")
    if contextual_structured_extension:
        trusted_lineage_score += 2
        trusted_lineage_reasons.append("contextual_structured_extension")
    elif contextual_extension_denied_reason == "visual_region_support_required":
        trusted_lineage_score -= 5
        trusted_lineage_reasons.append("weak_visual_region_support")
    if current_blob_only_match:
        trusted_lineage_score -= 6
        trusted_lineage_reasons.append("current_blob_only_match")
    if unrelated_page_hits:
        penalty = 4 + min(len(unrelated_page_hits), 3) * 2
        if not trusted_lineage_match:
            penalty += 4
        trusted_lineage_score -= penalty
        trusted_lineage_reasons.append("unrelated_page_content")
    if duplicate_patch_start:
        trusted_lineage_score -= 12
        trusted_lineage_reasons.append("duplicate_patch_start")
    if drift_containment_active and not trusted_lineage_match and not clear_structured_closure:
        trusted_lineage_score -= 8
        trusted_lineage_reasons.append("drift_containment_active")

    score = 0
    reasons: list[str] = []
    if seed_consistency > 0:
        score += seed_consistency * 5
        reasons.append("seed_reply_continuity")
    if trusted_lineage_consistency > 0:
        score += trusted_lineage_consistency * 4
        reasons.append("trusted_lineage_continuity")
    if current_consistency > 0 and trusted_lineage_match:
        score += current_consistency * 2
        reasons.append("current_reply_continuity")
    if current_blob_only_match:
        score -= 4
        reasons.append("current_blob_only_match")
    if tail_overlap >= 24:
        score += min(4, max(1, tail_overlap // 32))
        reasons.append("trusted_lineage_overlap")
    if structured_continuation:
        score += 5
        reasons.append("structured_continuation")
    if contextual_structured_extension:
        score += 2
        reasons.append("contextual_structured_extension")
    elif contextual_extension_denied_reason == "visual_region_support_required":
        score -= 4
        reasons.append("weak_visual_region_support")
    if segment_structure_state in {"partial_structured", "complete_structured"} and structured_continuation:
        score += 3
        reasons.append("structured_segment")
    if clear_structured_closure:
        score += 4
        reasons.append("clear_structured_closure")
    if not clear_structured_closure and completion_progress and trusted_lineage_match:
        score += 2
        reasons.append("structured_completion_progress")
    if int(after_progress["schema_hits"]) > int(before_progress["schema_hits"]):
        score += 2
        reasons.append("adds_schema_keys")
    if int(after_progress["operation_items"]) > int(before_progress["operation_items"]):
        score += 2
        reasons.append("adds_operations")
    if unrelated_page_hits:
        penalty = 4 + min(len(unrelated_page_hits), 3) * 2
        if not trusted_lineage_match:
            penalty += 4
        score -= penalty
        reasons.append("unrelated_page_content")
    if duplicate_patch_start:
        score -= 12
        reasons.append("duplicate_patch_start")
    if current_clean and segment_structure_state in {"raw_code", "unstructured"} and not trusted_lineage_match:
        score -= 6
        reasons.append("weak_region_continuity")
    if drift_containment_active and not trusted_lineage_match and not clear_structured_closure:
        score -= 8
        reasons.append("drift_containment_active")

    drift_detected = bool(
        current_clean
        and (
            duplicate_patch_start
            or current_blob_only_match
            or (unrelated_page_hits and not trusted_lineage_match and not clear_structured_closure)
        )
    )
    if drift_detected:
        score -= 6
        reasons.append("reply_region_drift")

    deduped_reasons: list[str] = []
    for reason in reasons:
        if reason not in deduped_reasons:
            deduped_reasons.append(reason)
    deduped_lineage_reasons: list[str] = []
    for reason in trusted_lineage_reasons:
        if reason not in deduped_lineage_reasons:
            deduped_lineage_reasons.append(reason)

    return {
        "region_integrity_score": score,
        "region_reasons": deduped_reasons,
        "continuity_against_seed": seed_consistency,
        "continuity_against_trusted_lineage": trusted_lineage_consistency,
        "continuity_against_current": current_consistency,
        "region_consistent": score > 0 and not drift_detected and trusted_lineage_score > 0,
        "unrelated_page_content": bool(unrelated_page_hits),
        "unrelated_page_hits": unrelated_page_hits,
        "drift_detected": drift_detected,
        "trusted_lineage_score": trusted_lineage_score,
        "trusted_lineage_reasons": deduped_lineage_reasons,
        "trusted_lineage_match": trusted_lineage_match,
        "matched_seed_lineage": seed_consistency > 0,
        "matched_trusted_lineage": trusted_lineage_consistency > 0 or tail_overlap >= 24,
        "contextual_structured_extension": contextual_structured_extension,
        "contextual_extension_allowed_reason": contextual_extension_allowed_reason,
        "contextual_extension_denied_reason": contextual_extension_denied_reason,
        "matched_current_blob_only": current_blob_only_match,
        "clear_structured_closure": clear_structured_closure,
        "segment_structure_state": segment_structure_state,
        "visual_region_confidence": visual_region["visual_region_confidence"],
        "visual_region_used_remembered_region": visual_region["used_remembered_region"],
        "visual_region_confirmed_reply_region": visual_region["confirmed_reply_region"],
        "visual_region_supports_extension": visual_region["strong_visual_region_support"],
        "visual_region_supports_reply_region": visual_region["supports_reply_region"],
        "visual_region_low_confidence": visual_region["weak_visual_region_support"],
        "activate_drift_containment": drift_detected or current_blob_only_match or (drift_containment_active and not trusted_lineage_match),
    }


def _structured_reply_quality(
    text: str,
    *,
    extract_structured_block: Callable[[str], str],
    is_patch_json: Callable[[str], bool],
) -> int:
    cleaned = text.strip()
    if not cleaned:
        return 0
    if is_patch_json(cleaned):
        return 3
    lowered = _normalize(cleaned)
    has_begin_marker = "aster patch begin" in lowered or "aster_patch_begin" in lowered
    has_end_marker = "aster patch end" in lowered or "aster_patch_end" in lowered
    if has_begin_marker and has_end_marker:
        return 2
    if has_begin_marker:
        return 1
    if '"operations"' in cleaned and "{" in cleaned:
        return 1
    return 0


def _structured_schema_hit_count(text: str) -> int:
    return sum(1 for hint in STRUCTURED_SCHEMA_HINTS if hint in text)


def _ordered_segment_novelty_count(current_text: str, merged_text: str) -> int:
    current_lines = {_normalize(line) for line in current_text.splitlines() if line.strip()}
    merged_lines = [_normalize(line) for line in merged_text.splitlines() if line.strip()]
    return sum(1 for line in merged_lines if line and line not in current_lines)


def _scrolled_anchor_consistency(anchor_text: str, candidate_text: str) -> int:
    anchor_clean = anchor_text.strip()
    candidate_clean = candidate_text.strip()
    if not anchor_clean or not candidate_clean:
        return 0
    anchor_normalized = _normalize(anchor_clean)
    candidate_normalized = _normalize(candidate_clean)
    if anchor_normalized and anchor_normalized in candidate_normalized:
        return 3
    excerpt = anchor_normalized[:160]
    if excerpt and excerpt in candidate_normalized:
        return 2
    anchor_lines = [_normalize(line) for line in anchor_clean.splitlines() if line.strip()]
    candidate_lines = {_normalize(line) for line in candidate_clean.splitlines() if line.strip()}
    overlap = sum(1 for line in anchor_lines[:8] if line in candidate_lines)
    return min(overlap, 2)


def _suffix_prefix_overlap(current_text: str, segment_text: str, *, max_chars: int = 240) -> int:
    current = current_text.strip()
    segment = segment_text.strip()
    if not current or not segment:
        return 0
    max_overlap = min(len(current), len(segment), max_chars)
    for count in range(max_overlap, 0, -1):
        if current[-count:] == segment[:count]:
            return count
    return 0


def _has_begin_marker(text: str) -> bool:
    lowered = _normalize(text)
    return "aster patch begin" in lowered or "aster_patch_begin" in lowered


def _scrolled_unrelated_page_hits(text: str) -> list[str]:
    hits: list[str] = []
    lowered = _normalize(text)
    for marker in SCROLLED_PAGE_CHROME_MARKERS:
        if marker in lowered and marker not in hits:
            hits.append(marker)

    for raw_line in text.splitlines():
        line = raw_line.strip()
        normalized_line = _normalize(line)
        if not line:
            continue
        if normalized_line in SCROLLED_PAGE_NAVIGATION_LINES and normalized_line not in hits:
            hits.append(normalized_line)
        if _looks_like_repo_listing_line(line) and "repo_listing" not in hits:
            hits.append("repo_listing")
        if _looks_like_test_output_line(line) and "test_output" not in hits:
            hits.append("test_output")
        if _looks_like_repo_source_line(line) and "repo_source" not in hits:
            hits.append("repo_source")
    return hits


def _looks_like_repo_listing_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if any(char in stripped for char in '{}[]"=:,'):
        return False
    return bool(
        re.fullmatch(r"[\w./\\-]+\.(py|pyi|md|txt|json|toml|yaml|yml|ini|cfg)", stripped, flags=re.IGNORECASE)
        or re.fullmatch(r"[\w./\\-]+/", stripped)
        or re.fullmatch(r"(tests?|docs?|src|scripts?)([/\\][\w./\\-]+)?", stripped, flags=re.IGNORECASE)
    )


def _looks_like_test_output_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if any(char in stripped for char in '{}[]"') and '"type"' in stripped:
        return False
    for pattern in SCROLLED_TEST_OUTPUT_PATTERNS:
        if re.search(pattern, stripped, flags=re.IGNORECASE):
            return True
    return False


def _looks_like_repo_source_line(line: str) -> bool:
    lowered = _normalize(line.strip())
    if not lowered:
        return False
    return any(marker in lowered for marker in SCROLLED_REPO_SOURCE_MARKERS)


def _reply_candidate_structure_state(
    text: str,
    *,
    is_patch_json: Callable[[str], bool],
) -> str:
    cleaned = text.strip()
    if not cleaned:
        return "empty"
    if is_patch_json(cleaned):
        return "complete_structured"
    lowered = _normalize(cleaned)
    has_begin_marker = "aster patch begin" in lowered or "aster_patch_begin" in lowered
    has_end_marker = "aster patch end" in lowered or "aster_patch_end" in lowered
    schema_hits = sum(1 for hint in STRUCTURED_SCHEMA_HINTS if hint in cleaned)
    if has_begin_marker and (schema_hits > 0 or not has_end_marker or reply_looks_incomplete(cleaned)):
        return "partial_structured"
    if schema_hits >= 2 and "{" in cleaned:
        return "partial_structured"
    if '"operations"' in cleaned and "{" in cleaned:
        return "partial_structured"
    if looks_like_code_reply_candidate(cleaned):
        return "raw_code"
    return "unstructured"


def _reply_acceptance_tier(
    text: str,
    *,
    source: str,
    prompt_anchor: TurnAnchor | None,
    require_anchor: bool,
) -> str:
    if source == "scrolled":
        return "scrolled_structured"
    if require_anchor and prompt_anchor is not None:
        return "anchored_structured"
    if len(text.strip()) <= VISIBLE_STRUCTURED_SHORT_MAX_CHARS:
        return "visible_structured_short"
    return "visible_structured_long"


def _ui_state_blocks_final_acceptance(ui_state: dict[str, object]) -> bool:
    if ui_state.get("show_in_text_field_present"):
        return True
    if ui_state.get("send_prompt_present"):
        return True
    return ui_state.get("send_prompt_enabled") is True


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
