from __future__ import annotations

import json
import math
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


def _clamp(value: float, *, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


@dataclass(frozen=True, slots=True)
class NormalizedRegion:
    left: float
    top: float
    width: float
    height: float

    def clamped(self) -> "NormalizedRegion":
        left = _clamp(self.left)
        top = _clamp(self.top)
        width = _clamp(self.width, low=0.0)
        height = _clamp(self.height, low=0.0)
        if left + width > 1.0:
            width = max(0.0, 1.0 - left)
        if top + height > 1.0:
            height = max(0.0, 1.0 - top)
        return NormalizedRegion(left=left, top=top, width=width, height=height)

    def center(self) -> tuple[float, float]:
        return (self.left + self.width / 2.0, self.top + self.height / 2.0)


@dataclass(slots=True)
class VisualMemoryEntry:
    intent: str
    region: NormalizedRegion
    confidence: float
    window_title_pattern: str = ""
    page_state: str = ""
    uia_hints: tuple[str, ...] = ()
    ocr_hints: tuple[str, ...] = ()
    successful_action_types: tuple[str, ...] = ()
    success_count: int = 0
    failure_count: int = 0
    updated_at: float = 0.0

    def to_json(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["region"] = asdict(self.region)
        return payload

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> "VisualMemoryEntry":
        return cls(
            intent=str(payload["intent"]),
            region=NormalizedRegion(**payload["region"]).clamped(),
            confidence=float(payload["confidence"]),
            window_title_pattern=str(payload.get("window_title_pattern", "")),
            page_state=str(payload.get("page_state", "")),
            uia_hints=tuple(str(item) for item in payload.get("uia_hints", ())),
            ocr_hints=tuple(str(item) for item in payload.get("ocr_hints", ())),
            successful_action_types=tuple(str(item) for item in payload.get("successful_action_types", ())),
            success_count=int(payload.get("success_count", 0)),
            failure_count=int(payload.get("failure_count", 0)),
            updated_at=float(payload.get("updated_at", 0.0)),
        )


@dataclass(frozen=True, slots=True)
class VisualMemoryHint:
    intent: str
    region: NormalizedRegion
    confidence: float
    score: float
    title_match: bool
    page_state_match: bool
    matched_uia_hints: tuple[str, ...]
    matched_ocr_hints: tuple[str, ...]
    used_remembered_region: bool = True


@dataclass(frozen=True, slots=True)
class VisualTraceStep:
    run_id: str
    step_id: str
    pre_path: Path
    during_path: Path
    post_path: Path
    manifest_path: Path


def normalize_region(rect: tuple[int, int, int, int], window_rect: tuple[int, int, int, int]) -> NormalizedRegion:
    left, top, right, bottom = rect
    win_left, win_top, win_right, win_bottom = window_rect
    win_width = max(1, win_right - win_left)
    win_height = max(1, win_bottom - win_top)
    return NormalizedRegion(
        left=(left - win_left) / win_width,
        top=(top - win_top) / win_height,
        width=max(0.0, right - left) / win_width,
        height=max(0.0, bottom - top) / win_height,
    ).clamped()


def denormalize_region(region: NormalizedRegion, window_rect: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    normalized = region.clamped()
    win_left, win_top, win_right, win_bottom = window_rect
    win_width = max(1, win_right - win_left)
    win_height = max(1, win_bottom - win_top)
    left = int(round(win_left + normalized.left * win_width))
    top = int(round(win_top + normalized.top * win_height))
    right = int(round(left + normalized.width * win_width))
    bottom = int(round(top + normalized.height * win_height))
    return (left, top, right, bottom)


def decay_confidence(
    confidence: float,
    *,
    elapsed_seconds: float,
    half_life_seconds: float = 7 * 24 * 60 * 60,
) -> float:
    if confidence <= 0.0:
        return 0.0
    if elapsed_seconds <= 0.0:
        return _clamp(confidence)
    decay_factor = 0.5 ** (elapsed_seconds / max(1.0, half_life_seconds))
    return _clamp(confidence * decay_factor)


def merge_normalized_regions(
    current: NormalizedRegion,
    new: NormalizedRegion,
    *,
    current_weight: float,
    new_weight: float,
) -> NormalizedRegion:
    total = max(1e-6, current_weight + new_weight)
    return NormalizedRegion(
        left=(current.left * current_weight + new.left * new_weight) / total,
        top=(current.top * current_weight + new.top * new_weight) / total,
        width=(current.width * current_weight + new.width * new_weight) / total,
        height=(current.height * current_weight + new.height * new_weight) / total,
    ).clamped()


def build_visual_trace_paths(root_dir: Path, run_id: str, step_id: str) -> VisualTraceStep:
    run_dir = root_dir / run_id
    return VisualTraceStep(
        run_id=run_id,
        step_id=step_id,
        pre_path=run_dir / f"{step_id}_pre.png",
        during_path=run_dir / f"{step_id}_during.png",
        post_path=run_dir / f"{step_id}_post.png",
        manifest_path=run_dir / "trace_manifest.jsonl",
    )


def build_visual_manifest_entry(
    *,
    run_id: str,
    step_id: str,
    timestamp: str,
    stage: str,
    action_type: str,
    target_intent: str,
    window_title: str,
    cursor_before: tuple[int, int] | None,
    cursor_during: tuple[int, int] | None,
    cursor_after: tuple[int, int] | None,
    target_rect: tuple[int, int, int, int] | None,
    candidate_rects: list[tuple[int, int, int, int]] | None,
    ui_summary: dict[str, Any],
    ocr_summary: list[str],
    page_summary: dict[str, Any],
    screenshot_paths: dict[str, str | None],
    action_outcome: str,
    confidence_before: float | None,
    confidence_after: float | None,
    used_remembered_region: bool,
    updated_remembered_region: bool,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "step_id": step_id,
        "timestamp": timestamp,
        "stage": stage,
        "action_type": action_type,
        "target_intent": target_intent,
        "window_title": window_title,
        "cursor_position_before": cursor_before,
        "cursor_position_during": cursor_during,
        "cursor_position_after": cursor_after,
        "target_rect": target_rect,
        "candidate_rects": candidate_rects or [],
        "ui_summary": ui_summary,
        "ocr_summary": ocr_summary,
        "page_summary": page_summary,
        "screenshot_paths": screenshot_paths,
        "action_outcome": action_outcome,
        "confidence_before": confidence_before,
        "confidence_after": confidence_after,
        "used_remembered_region": used_remembered_region,
        "updated_remembered_region": updated_remembered_region,
    }


class VisualRegionMemoryStore:
    def __init__(self, path: Path, *, half_life_seconds: float = 7 * 24 * 60 * 60) -> None:
        self.path = path
        self.half_life_seconds = half_life_seconds
        self.entries: list[VisualMemoryEntry] = []

    def load(self) -> None:
        if not self.path.exists():
            self.entries = []
            return
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        self.entries = [VisualMemoryEntry.from_json(item) for item in payload]

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps([entry.to_json() for entry in self.entries], indent=2),
            encoding="utf-8",
        )

    def select_hint(
        self,
        *,
        intent: str,
        window_title: str,
        page_state: str,
        uia_hints: tuple[str, ...] = (),
        ocr_hints: tuple[str, ...] = (),
        now: float | None = None,
    ) -> VisualMemoryHint | None:
        if not self.entries:
            return None
        current_time = time.time() if now is None else now
        best: VisualMemoryHint | None = None
        for entry in self.entries:
            if entry.intent != intent:
                continue
            decayed_confidence = decay_confidence(
                entry.confidence,
                elapsed_seconds=max(0.0, current_time - entry.updated_at),
                half_life_seconds=self.half_life_seconds,
            )
            title_match = bool(entry.window_title_pattern and entry.window_title_pattern.lower() in window_title.lower())
            page_state_match = bool(entry.page_state and entry.page_state == page_state)
            matched_uia = tuple(sorted(set(entry.uia_hints).intersection(uia_hints)))
            matched_ocr = tuple(sorted(set(entry.ocr_hints).intersection(ocr_hints)))
            score = decayed_confidence
            if title_match:
                score += 0.25
            if page_state_match:
                score += 0.15
            score += len(matched_uia) * 0.05
            score += len(matched_ocr) * 0.05
            if decayed_confidence < 0.12 or score < 0.2:
                continue
            hint = VisualMemoryHint(
                intent=intent,
                region=entry.region,
                confidence=decayed_confidence,
                score=score,
                title_match=title_match,
                page_state_match=page_state_match,
                matched_uia_hints=matched_uia,
                matched_ocr_hints=matched_ocr,
            )
            if best is None or hint.score > best.score:
                best = hint
        return best

    def remember_success(
        self,
        *,
        intent: str,
        region: NormalizedRegion,
        window_title_pattern: str,
        page_state: str,
        action_type: str,
        uia_hints: tuple[str, ...] = (),
        ocr_hints: tuple[str, ...] = (),
        now: float | None = None,
    ) -> VisualMemoryEntry:
        current_time = time.time() if now is None else now
        match = self._find_entry(intent=intent, window_title_pattern=window_title_pattern, page_state=page_state)
        if match is None:
            match = VisualMemoryEntry(
                intent=intent,
                region=region.clamped(),
                confidence=0.55,
                window_title_pattern=window_title_pattern,
                page_state=page_state,
                uia_hints=tuple(sorted(set(uia_hints))),
                ocr_hints=tuple(sorted(set(ocr_hints))),
                successful_action_types=(action_type,),
                success_count=1,
                updated_at=current_time,
            )
            self.entries.append(match)
            return match
        match.region = merge_normalized_regions(
            match.region,
            region.clamped(),
            current_weight=max(0.25, match.confidence + match.success_count),
            new_weight=1.0,
        )
        match.confidence = min(1.0, match.confidence + 0.12)
        match.success_count += 1
        match.updated_at = current_time
        match.uia_hints = tuple(sorted(set(match.uia_hints).union(uia_hints)))
        match.ocr_hints = tuple(sorted(set(match.ocr_hints).union(ocr_hints)))
        match.successful_action_types = tuple(sorted(set(match.successful_action_types).union((action_type,))))
        return match

    def invalidate(
        self,
        *,
        intent: str,
        window_title_pattern: str,
        page_state: str,
        reason_weight: float = 0.25,
        now: float | None = None,
    ) -> VisualMemoryEntry | None:
        current_time = time.time() if now is None else now
        match = self._find_entry(intent=intent, window_title_pattern=window_title_pattern, page_state=page_state)
        if match is None:
            return None
        match.confidence = max(0.0, match.confidence - max(0.05, reason_weight))
        match.failure_count += 1
        match.updated_at = current_time
        return match

    def _find_entry(self, *, intent: str, window_title_pattern: str, page_state: str) -> VisualMemoryEntry | None:
        for entry in self.entries:
            if (
                entry.intent == intent
                and entry.window_title_pattern == window_title_pattern
                and entry.page_state == page_state
            ):
                return entry
        return None


class VisualActionDebugSession:
    def __init__(
        self,
        *,
        root_dir: Path,
        enabled: bool,
        checkpoint_interval_seconds: int = 15,
        max_checkpoints_per_key: int = 2,
    ) -> None:
        self.root_dir = root_dir
        self.enabled = enabled
        self.checkpoint_interval_seconds = checkpoint_interval_seconds
        self.max_checkpoints_per_key = max_checkpoints_per_key
        self.run_id = ""
        self._step_counter = 0
        self._checkpoint_state: dict[str, tuple[float, int]] = {}

    def start_run(self, *, run_id: str | None = None) -> str:
        self.run_id = run_id or f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{uuid.uuid4().hex[:8]}"
        self._step_counter = 0
        self._checkpoint_state = {}
        if self.enabled:
            (self.root_dir / self.run_id).mkdir(parents=True, exist_ok=True)
        return self.run_id

    def next_step(self, action_type: str) -> VisualTraceStep:
        self._step_counter += 1
        step_id = f"{self._step_counter:04d}_{action_type}"
        return build_visual_trace_paths(self.root_dir, self.run_id, step_id)

    def should_capture_checkpoint(self, key: str, *, now: float | None = None) -> bool:
        current_time = time.time() if now is None else now
        last_time, count = self._checkpoint_state.get(key, (0.0, 0))
        if count >= self.max_checkpoints_per_key:
            return False
        if current_time - last_time < self.checkpoint_interval_seconds:
            return False
        self._checkpoint_state[key] = (current_time, count + 1)
        return True

    def save_stage_image(self, step: VisualTraceStep, stage: str, image: Any) -> Path | None:
        if not self.enabled or not self.run_id or image is None:
            return None
        path = {
            "pre": step.pre_path,
            "during": step.during_path,
            "post": step.post_path,
        }.get(stage)
        if path is None:
            return None
        path.parent.mkdir(parents=True, exist_ok=True)
        if hasattr(image, "save"):
            image.save(path)
            return path
        if isinstance(image, (bytes, bytearray)):
            path.write_bytes(bytes(image))
            return path
        return None

    def append_manifest_entry(self, step: VisualTraceStep, entry: dict[str, Any]) -> None:
        if not self.enabled or not self.run_id:
            return
        step.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with step.manifest_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
