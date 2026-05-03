from __future__ import annotations

import atexit
import difflib
import hashlib
import json
import os
import re
import subprocess
import sys
import textwrap
import time
import traceback
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aster.audit_logger import AuditLogger
from aster.browser_core import (
    BrowserStrategy,
    DEFAULT_COMPOSER_VERIFICATION_POLICY,
    PageClassification,
    RecoveryAction,
    ReplyTrackerPolicy,
    ThreadRegistry,
    build_recovery_handlers,
    build_turn_anchor,
    choose_best_reply_candidate_for_policy,
    classify_page,
    evaluate_reply_acceptance,
    extract_structured_block,
    decide_and_execute_recovery,
    decision_payload,
    extract_reply_from_ocr_lines_for_policy,
    looks_like_browser_url_text,
    looks_like_chatgpt_page,
    looks_like_patch_plan_json,
    looks_like_reply_started_candidate_for_policy,
    assess_scrolled_segment_addition,
    merge_text_segments,
    merge_reply_segment_sources,
    merge_scrolled_reply_segments,
    reply_detection_blocked,
    reply_looks_incomplete,
    prompt_insertion_confirmed,
    score_candidate_for_policy,
    scrolled_segment_looks_contaminated_for_policy,
    select_preferred_reply_candidate,
    serialize_anchor,
    should_extend_structured_scroll_window,
    summarize_reply_wait_iteration,
    structured_completion_progress,
    structured_completion_score,
    window_title_suggests_existing_chat,
)
from aster.runtime_log_heartbeat import RuntimeLogHeartbeatPusher
from aster.visual_action_memory import (
    VisualActionDebugSession,
    VisualRegionMemoryStore,
    build_visual_manifest_entry,
    denormalize_region,
    normalize_region,
)


BROWSER_REPLY_NOISE = (
    "ask anything",
    "chatgpt can make mistakes",
    "search chats",
    "library",
    "deep research",
    "company knowledge",
    "show in text field",
    "send prompt",
    "new chat",
    "projects",
    "gpts",
    "explore gpts",
    "chatgpt",
    "openai",
    "ask gemini",
    "github",
    "pull request",
    "issues",
    "commit",
    "context omitted for browser size safety",
    "included_files",
    "omitted_files",
)

BROWSER_READY_HINTS = (
    "ask anything",
    "message chatgpt",
    "what are you working on",
    "type a message",
    "send a message",
)

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

STOP_STREAMING_HINTS = (
    "stop streaming",
    "stop generating",
)

INPUT_TOO_LARGE_HINTS = (
    "input too large",
    "message too long",
)

PROMPT_ECHO_MARKERS = (
    "user goal:",
    "project summary:",
    "relevant file tree:",
    "relevant file contents:",
    "conversation history:",
    "constraints:",
    "final output rules for browser mode:",
    "path:",
    "reason:",
    "content:",
    "your message",
    "context omitted for browser size safety",
    "included_files",
    "omitted_files",
    "return promptpackage",
    "self.log(""activity""",
    "def _normalize",
    "aster_patch_begin on its own line",
    "aster patch begin on its own line",
    "aster_patch_end on its own line",
    "aster patch end on its own line",
    "do not wrap the json",
    "if you are missing context",
    "previous response was rejected",
    "previous response:",
    "one or more exact operations",
    "deterministic. final output rules for browser mode",
)

REPLY_TRACKER_POLICY = ReplyTrackerPolicy(
    browser_reply_noise=BROWSER_REPLY_NOISE,
    input_too_large_hints=INPUT_TOO_LARGE_HINTS,
    prompt_echo_markers=PROMPT_ECHO_MARKERS,
    stop_streaming_hints=STOP_STREAMING_HINTS,
    penalty_markers=(
        "good to see you",
        "company knowledge",
        "show in text field",
        "ask anything",
        "what are you working on",
        "input too large",
        "message too long",
        "ask gemini",
        "github",
        "context omitted for browser size safety",
        "included_files",
        "omitted_files",
        "return promptpackage",
        "def _normalize",
        'self.log("activity"',
    ),
    operation_markers=("CREATE FILE", "EDIT FILE", "REPLACE FILE", "RUN COMMANDS", "NEED THESE FILES FIRST"),
    extra_bad_markers=(
        "ask gemini",
        "github",
        "context omitted for browser size safety",
        "included_files",
        "omitted_files",
        "return promptpackage",
        "def _normalize",
        'self.log("activity"',
    ),
)

pyautogui = None
Desktop = None

PAGE_READINESS_SCORE_THRESHOLD = 55.0
PAGE_READINESS_WRONG_PAGE_MARKERS = (
    "ask gemini",
    "github",
    "youtube",
    "pull request",
    "issues",
    "commit",
)


@dataclass(slots=True)
class BrowserResult:
    raw_text: str
    metadata: dict[str, Any]


@dataclass(slots=True)
class _StructuredReplyCandidate:
    source: str
    raw_text: str
    parsed_text: str
    score: float
    parseable: bool
    salvage_allowed: bool
    retry_safe: bool = False
    retry_seed_validity_reason: str = ""
    region_trusted: bool = True
    region_confidence: float | None = None
    region_reason: str = ""
    observation_count: int = 1


@dataclass(slots=True)
class _ReplyCaptureSnapshot:
    best_structured: _StructuredReplyCandidate | None = None
    best_salvageable: _StructuredReplyCandidate | None = None
    best_retry_safe: _StructuredReplyCandidate | None = None


class _AutomationNotice:
    _active_process: subprocess.Popen[str] | None = None
    _active_owner_id: int | None = None
    _exit_cleanup_registered = False

    def __init__(self, logger) -> None:
        self._logger = logger
        self._owner_id = id(self)
        if not _AutomationNotice._exit_cleanup_registered:
            atexit.register(_AutomationNotice._cleanup_at_exit)
            _AutomationNotice._exit_cleanup_registered = True

    def _log(self, event: str, payload: dict[str, Any] | None = None) -> None:
        if self._logger is None:
            return
        self._logger.log("browser_transport", {"event": event, **(payload or {})})

    @staticmethod
    def _process_payload(process: subprocess.Popen[str] | None) -> dict[str, Any]:
        if process is None:
            return {"pid": None, "returncode": None}
        return {
            "pid": getattr(process, "pid", None),
            "returncode": process.poll(),
        }

    @staticmethod
    def _process_running(process: subprocess.Popen[str] | None) -> bool:
        return process is not None and process.poll() is None

    @staticmethod
    def _build_script(parent_pid: int) -> str:
        return textwrap.dedent(
            f"""
            import ctypes
            import tkinter as tk

            PARENT_PID = {parent_pid}
            PROCESS_SYNCHRONIZE = 0x00100000
            WAIT_OBJECT_0 = 0x00000000
            WAIT_TIMEOUT = 0x00000102

            def _parent_alive(pid: int) -> bool:
                if pid <= 0:
                    return False
                kernel32 = ctypes.windll.kernel32
                handle = kernel32.OpenProcess(PROCESS_SYNCHRONIZE, False, pid)
                if not handle:
                    return False
                try:
                    status = kernel32.WaitForSingleObject(handle, 0)
                    return status == WAIT_TIMEOUT
                finally:
                    kernel32.CloseHandle(handle)

            root = tk.Tk()
            root.title("Aster Using Your PC")
            root.attributes("-topmost", True)
            root.resizable(False, False)
            root.configure(bg="#FFF3CD")
            root.protocol("WM_DELETE_WINDOW", lambda: None)
            frame = tk.Frame(root, bg="#FFF3CD", padx=18, pady=14)
            frame.pack(fill="both", expand=True)
            tk.Label(
                frame,
                text="Aster is using your keyboard and mouse right now.",
                font=("Segoe UI", 12, "bold"),
                bg="#FFF3CD",
                fg="#5C3B00",
                justify="left",
                wraplength=340,
            ).pack(anchor="w")
            tk.Label(
                frame,
                text="Please do not type, click, or move the mouse until this notice disappears.",
                font=("Segoe UI", 10),
                bg="#FFF3CD",
                fg="#5C3B00",
                justify="left",
                wraplength=340,
                pady=8,
            ).pack(anchor="w")
            root.update_idletasks()
            width = max(root.winfo_width(), 390)
            height = max(root.winfo_height(), 120)
            screen_width = root.winfo_screenwidth()
            x = max(20, screen_width - width - 30)
            y = 30
            root.geometry(f"{{width}}x{{height}}+{{x}}+{{y}}")

            def _poll_parent() -> None:
                if not _parent_alive(PARENT_PID):
                    root.destroy()
                    return
                root.after(1000, _poll_parent)

            root.after(1000, _poll_parent)
            root.mainloop()
            """
        )

    @classmethod
    def _clear_active_process(cls) -> None:
        cls._active_process = None
        cls._active_owner_id = None

    @classmethod
    def _cleanup_at_exit(cls) -> None:
        cls._terminate_active_process(reason="atexit", logger=None, owner_id=None)

    @classmethod
    def _terminate_active_process(
        cls,
        *,
        reason: str,
        logger,
        owner_id: int | None,
    ) -> bool:
        process = cls._active_process
        if process is None:
            return False
        if owner_id is not None and cls._active_owner_id not in (None, owner_id):
            return False
        payload = {"reason": reason, **cls._process_payload(process)}
        if process.poll() is not None:
            if logger is not None:
                logger.log("browser_transport", {"event": "automation_notice_stale_detected", **payload})
                logger.log("browser_transport", {"event": "automation_notice_stale_cleaned_up", **payload})
            cls._clear_active_process()
            return True
        if reason != "hide":
            payload["owner_id"] = cls._active_owner_id
            if logger is not None:
                logger.log("browser_transport", {"event": "automation_notice_stale_detected", **payload})
        try:
            process.terminate()
            process.wait(timeout=2.0)
        except subprocess.TimeoutExpired as exc:
            try:
                process.kill()
                process.wait(timeout=2.0)
            except Exception as kill_exc:
                if logger is not None:
                    logger.log(
                        "browser_transport",
                        {
                            "event": "automation_notice_terminate_failure",
                            "reason": reason,
                            "error": str(kill_exc),
                            "fallback_error": str(exc),
                            **cls._process_payload(process),
                        },
                    )
                return False
        except Exception as exc:
            if process.poll() is not None:
                if logger is not None:
                    logger.log("browser_transport", {"event": "automation_notice_stale_detected", **payload})
                    logger.log("browser_transport", {"event": "automation_notice_stale_cleaned_up", **payload})
                cls._clear_active_process()
                return True
            if logger is not None:
                logger.log(
                    "browser_transport",
                    {
                        "event": "automation_notice_terminate_failure",
                        "reason": reason,
                        "error": str(exc),
                        **cls._process_payload(process),
                    },
                )
            return False
        payload = {"reason": reason, **cls._process_payload(process)}
        if logger is not None:
            event = "automation_notice_stale_cleaned_up" if reason != "hide" else "automation_notice_hide"
            logger.log("browser_transport", {"event": event, **payload})
        cls._clear_active_process()
        return True

    def show(self) -> None:
        if self._process_running(type(self)._active_process) and type(self)._active_owner_id == self._owner_id:
            return
        type(self)._terminate_active_process(reason="show_before_launch", logger=self._logger, owner_id=None)
        script = self._build_script(os.getpid())
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            process = subprocess.Popen(
                [sys.executable, "-c", script],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=creationflags,
            )
            type(self)._active_process = process
            type(self)._active_owner_id = self._owner_id
            time.sleep(0.35)
            self._log("automation_notice_show", self._process_payload(process))
        except Exception as exc:
            type(self)._clear_active_process()
            self._log("automation_notice_launch_failure", {"error": str(exc)})
            if self._logger is not None:
                self._logger.log("browser_transport", {"event": "automation_notice_unavailable", "error": str(exc)})
            return

    def hide(self) -> None:
        type(self)._terminate_active_process(reason="hide", logger=self._logger, owner_id=self._owner_id)


class BrowserChatGPTTransport:
    """Experimental browser-mode adapter."""

    def __init__(
        self,
        logger: AuditLogger | None = None,
        *,
        strategy: BrowserStrategy | str = BrowserStrategy.PATCH_RUNNER,
        thread_reuse_enabled: bool = False,
        thread_registry_path: Path | None = None,
        verification_level: str = "basic",
        log_screenshots: bool = False,
        max_recovery_attempts: int = 3,
        runtime_log_heartbeat: RuntimeLogHeartbeatPusher | None = None,
        visual_action_debug: VisualActionDebugSession | None = None,
        visual_region_memory: VisualRegionMemoryStore | None = None,
        visual_action_memory_enabled: bool = True,
    ) -> None:
        self._executor = None
        self._capture = None
        self._ocr = None
        self._logger = logger
        self._last_window_state: str = ""
        self._automation_notice = _AutomationNotice(logger)
        self.strategy = BrowserStrategy(str(strategy))
        self.thread_reuse_enabled = thread_reuse_enabled
        self.thread_registry = ThreadRegistry(thread_registry_path or Path(".aster/thread_registry.json"))
        self.verification_level = verification_level
        self.log_screenshots = log_screenshots
        self.max_recovery_attempts = max_recovery_attempts
        self._runtime_log_heartbeat = runtime_log_heartbeat
        self._last_page_classification: PageClassification | None = None
        self._last_uia_control_diagnostics: dict[str, Any] | None = None
        self._last_uia_diagnostic_signature = ""
        self._last_reply_capture_snapshot = _ReplyCaptureSnapshot()
        self._last_reply_region_visual_evidence: dict[str, Any] = {}
        self._last_scrolled_capture_meta: dict[str, Any] = {}
        self._last_retry_fragment_source_meta: dict[str, Any] = self._retry_source_preference_defaults()
        self._last_uia_reply_read_diagnostics: dict[str, Any] = {
            "uia_read_failure_reason": "",
            "descendant_enumeration_guard_triggered": False,
        }
        self._visual_action_debug = visual_action_debug
        self._visual_region_memory = visual_region_memory
        self._visual_action_memory_enabled = visual_action_memory_enabled
        self._visual_action_run_id = ""

    def _log(self, event: str, payload: dict[str, Any]) -> None:
        if self._logger is None:
            return
        self._logger.log("browser_transport", {"event": event, **payload})

    def _activity(
        self,
        step: str,
        message: str,
        why: str,
        *,
        status: str = "info",
        details: dict[str, Any] | None = None,
    ) -> None:
        if self._logger is None:
            return
        self._logger.activity(step, message, why, status=status, details=details)

    def _start_runtime_log_heartbeat(self) -> None:
        if self._runtime_log_heartbeat is None:
            return
        self._runtime_log_heartbeat.start_run()

    def _stop_runtime_log_heartbeat(self) -> None:
        if self._runtime_log_heartbeat is None:
            return
        self._runtime_log_heartbeat.stop_run()

    def _tick_runtime_log_heartbeat(self, reason: str) -> None:
        if self._runtime_log_heartbeat is None:
            return
        self._runtime_log_heartbeat.tick(reason=reason)

    @staticmethod
    def _window_rect(target) -> tuple[int, int, int, int]:
        return (
            int(getattr(target, "left", 0)),
            int(getattr(target, "top", 0)),
            int(getattr(target, "left", 0) + getattr(target, "width", 0)),
            int(getattr(target, "top", 0) + getattr(target, "height", 0)),
        )

    @staticmethod
    def _rect_from_center(center_x: int, center_y: int, *, radius_x: int = 28, radius_y: int = 20) -> tuple[int, int, int, int]:
        return (
            int(center_x - radius_x),
            int(center_y - radius_y),
            int(center_x + radius_x),
            int(center_y + radius_y),
        )

    @staticmethod
    def _safe_cursor_position() -> tuple[int, int] | None:
        if pyautogui is None:
            return None
        try:
            pos = pyautogui.position()
            if hasattr(pos, "x") and hasattr(pos, "y"):
                return (int(pos.x), int(pos.y))
            if isinstance(pos, tuple) and len(pos) >= 2:
                return (int(pos[0]), int(pos[1]))
        except Exception:
            return None
        return None

    def _visual_page_state(self) -> str:
        analysis = self._last_page_classification
        if analysis is None:
            return ""
        if analysis.composer_visible and analysis.looks_like_chatgpt:
            return "chatgpt_composer_visible"
        if analysis.looks_like_chatgpt:
            return "chatgpt_visible"
        return "browser_unknown"

    def _visual_page_summary(self) -> dict[str, Any]:
        analysis = self._last_page_classification
        if analysis is None:
            return {}
        return {
            "ready_score": round(analysis.ready_score, 1),
            "looks_like_chatgpt": analysis.looks_like_chatgpt,
            "composer_visible": analysis.composer_visible,
            "send_button_present": analysis.send_button_present,
            "stop_streaming_present": analysis.stop_streaming_present,
            "wrong_page_signals_present": analysis.wrong_page_signals_present,
        }

    @staticmethod
    def _visual_ui_summary(ui_state: dict[str, Any] | None) -> dict[str, Any]:
        if not ui_state:
            return {}
        keys = (
            "send_prompt_present",
            "send_prompt_enabled",
            "show_in_text_field_present",
            "stop_streaming_present",
            "composer_edit_length",
            "composer_edit_preview",
            "window_title",
        )
        return {key: ui_state.get(key) for key in keys if key in ui_state}

    @staticmethod
    def _visual_ocr_summary(lines) -> list[str]:
        if not lines:
            return []
        return [str(getattr(line, "text", line))[:120] for line in list(lines)[:5]]

    def _start_visual_action_run(self) -> None:
        if self._visual_action_debug is None:
            self._visual_action_run_id = ""
            return
        self._visual_action_run_id = self._visual_action_debug.start_run()
        self._log(
            "visual_action_trace_started",
            {
                "run_id": self._visual_action_run_id,
                "trace_enabled": self._visual_action_debug.enabled,
                "memory_enabled": self._visual_action_memory_enabled,
            },
        )

    def _capture_visual_window(self, target):
        if self._capture is None:
            return None
        try:
            return self._capture.capture_region(target.left, target.top, target.width, target.height)
        except Exception:
            return None

    def _visual_memory_hint(
        self,
        *,
        intent: str,
        target,
        ui_state: dict[str, Any] | None = None,
        lines=None,
        log_events: bool = True,
    ):
        if not self._visual_action_memory_enabled or self._visual_region_memory is None:
            return None
        hint = self._visual_region_memory.select_hint(
            intent=intent,
            window_title=str(getattr(target, "title", "")),
            page_state=self._visual_page_state(),
            uia_hints=tuple(
                item for item in [
                    str((ui_state or {}).get("composer_edit_preview", "")).strip(),
                    "send" if (ui_state or {}).get("send_prompt_present") else "",
                    "streaming" if (ui_state or {}).get("stop_streaming_present") else "",
                ] if item
            ),
            ocr_hints=tuple(self._visual_ocr_summary(lines)[:3]),
        )
        if hint is None:
            if log_events:
                self._log("visual_memory_miss", {"intent": intent, "window_title": str(getattr(target, "title", ""))})
            return None
        if log_events:
            self._log(
                "visual_memory_hit",
                {
                    "intent": intent,
                    "confidence": round(hint.confidence, 3),
                    "score": round(hint.score, 3),
                    "title_match": hint.title_match,
                    "page_state_match": hint.page_state_match,
                },
            )
            confidence_event = (
                "action_region_confidence_confirmed"
                if hint.confidence >= 0.45 or hint.score >= 0.6
                else "action_region_confidence_low"
            )
            self._log(
                confidence_event,
                {
                    "intent": intent,
                    "confidence": round(hint.confidence, 3),
                    "score": round(hint.score, 3),
                },
            )
        return hint

    def _visual_memory_update(
        self,
        *,
        intent: str,
        target,
        action_type: str,
        rect: tuple[int, int, int, int],
        ui_state: dict[str, Any] | None = None,
        lines=None,
    ) -> None:
        if not self._visual_action_memory_enabled or self._visual_region_memory is None:
            return
        entry = self._visual_region_memory.remember_success(
            intent=intent,
            region=normalize_region(rect, self._window_rect(target)),
            window_title_pattern=str(getattr(target, "title", "")),
            page_state=self._visual_page_state(),
            action_type=action_type,
            uia_hints=tuple(
                item for item in [
                    str((ui_state or {}).get("composer_edit_preview", "")).strip(),
                    "send" if (ui_state or {}).get("send_prompt_present") else "",
                    "streaming" if (ui_state or {}).get("stop_streaming_present") else "",
                ] if item
            ),
            ocr_hints=tuple(self._visual_ocr_summary(lines)[:3]),
        )
        self._visual_region_memory.save()
        self._log(
            "visual_memory_updated",
            {
                "intent": intent,
                "confidence": round(entry.confidence, 3),
                "success_count": entry.success_count,
            },
        )

    def _visual_memory_invalidate(self, *, intent: str, target, reason_weight: float = 0.25) -> None:
        if not self._visual_action_memory_enabled or self._visual_region_memory is None:
            return
        entry = self._visual_region_memory.invalidate(
            intent=intent,
            window_title_pattern=str(getattr(target, "title", "")),
            page_state=self._visual_page_state(),
            reason_weight=reason_weight,
        )
        if entry is None:
            return
        self._visual_region_memory.save()
        self._log(
            "visual_memory_invalidated",
            {
                "intent": intent,
                "confidence": round(entry.confidence, 3),
                "failure_count": entry.failure_count,
            },
        )

    @staticmethod
    def _build_reply_region_visual_evidence(*, hint, merged_length: int) -> dict[str, Any]:
        confidence = None if hint is None else float(hint.confidence)
        used_remembered_region = hint is not None
        confirmed_reply_region = merged_length > 0
        supports_reply_region = bool(
            confirmed_reply_region
            or hint is None
            or confidence is None
            or confidence >= 0.55
        )
        return {
            "visual_region_confidence": confidence,
            "used_remembered_region": used_remembered_region,
            "confirmed_reply_region": confirmed_reply_region,
            "supports_reply_region": supports_reply_region,
            "low_confidence_region": bool(
                used_remembered_region and not confirmed_reply_region and confidence is not None and confidence < 0.45
            ),
        }

    def _reply_region_candidate_trusted(self) -> tuple[bool, str]:
        evidence = self._last_reply_region_visual_evidence
        if not evidence:
            return True, "no_visual_region_evidence"
        if evidence.get("low_confidence_region"):
            return False, "low_confidence_visual_reply_region"
        if evidence.get("confirmed_reply_region"):
            return True, "confirmed_reply_region_capture"
        if not evidence.get("used_remembered_region"):
            return True, "no_remembered_region_dependency"
        if evidence.get("supports_reply_region"):
            return True, "remembered_reply_region_supported"
        return False, "weak_visual_reply_region_support"

    def _begin_visual_action(
        self,
        *,
        target,
        action_type: str,
        target_intent: str,
        ui_state: dict[str, Any] | None = None,
        lines=None,
        target_rect: tuple[int, int, int, int] | None = None,
        candidate_rects: list[tuple[int, int, int, int]] | None = None,
        confidence_before: float | None = None,
        used_remembered_region: bool = False,
    ) -> dict[str, Any] | None:
        if self._visual_action_debug is None or not self._visual_action_debug.enabled:
            return None
        step = self._visual_action_debug.next_step(action_type)
        cursor_before = self._safe_cursor_position()
        image = self._capture_visual_window(target)
        pre_path = self._visual_action_debug.save_stage_image(step, "pre", image)
        context = {
            "step": step,
            "action_type": action_type,
            "target_intent": target_intent,
            "window_title": str(getattr(target, "title", "")),
            "cursor_before": cursor_before,
            "cursor_during": None,
            "cursor_after": None,
            "target_rect": target_rect,
            "candidate_rects": candidate_rects or [],
            "ui_summary": self._visual_ui_summary(ui_state),
            "ocr_summary": self._visual_ocr_summary(lines),
            "page_summary": self._visual_page_summary(),
            "confidence_before": confidence_before,
            "confidence_after": None,
            "used_remembered_region": used_remembered_region,
            "updated_remembered_region": False,
            "screenshot_paths": {
                "pre": str(pre_path) if pre_path is not None else None,
                "during": None,
                "post": None,
            },
        }
        self._log(
            "visual_action_trace_started",
            {
                "run_id": step.run_id,
                "step_id": step.step_id,
                "action_type": action_type,
                "target_intent": target_intent,
                "used_remembered_region": used_remembered_region,
            },
        )
        return context

    def _capture_visual_action_stage(
        self,
        context: dict[str, Any] | None,
        *,
        target,
        stage: str,
        ui_state: dict[str, Any] | None = None,
        lines=None,
        action_outcome: str = "",
        confidence_after: float | None = None,
        updated_remembered_region: bool = False,
    ) -> None:
        if context is None or self._visual_action_debug is None:
            return
        image = self._capture_visual_window(target)
        path = self._visual_action_debug.save_stage_image(context["step"], stage, image)
        if stage == "during":
            context["cursor_during"] = self._safe_cursor_position()
        if stage == "post":
            context["cursor_after"] = self._safe_cursor_position()
        context["ui_summary"] = self._visual_ui_summary(ui_state)
        context["ocr_summary"] = self._visual_ocr_summary(lines)
        context["page_summary"] = self._visual_page_summary()
        context["confidence_after"] = confidence_after
        context["updated_remembered_region"] = updated_remembered_region
        if path is not None:
            context["screenshot_paths"][stage] = str(path)
        entry = build_visual_manifest_entry(
            run_id=context["step"].run_id,
            step_id=context["step"].step_id,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            stage=stage,
            action_type=context["action_type"],
            target_intent=context["target_intent"],
            window_title=context["window_title"],
            cursor_before=context["cursor_before"],
            cursor_during=context["cursor_during"],
            cursor_after=context["cursor_after"],
            target_rect=context["target_rect"],
            candidate_rects=context["candidate_rects"],
            ui_summary=context["ui_summary"],
            ocr_summary=context["ocr_summary"],
            page_summary=context["page_summary"],
            screenshot_paths=context["screenshot_paths"],
            action_outcome=action_outcome,
            confidence_before=context["confidence_before"],
            confidence_after=context["confidence_after"],
            used_remembered_region=context["used_remembered_region"],
            updated_remembered_region=context["updated_remembered_region"],
        )
        self._visual_action_debug.append_manifest_entry(context["step"], entry)
        self._log(
            "visual_action_trace_captured",
            {
                "run_id": context["step"].run_id,
                "step_id": context["step"].step_id,
                "stage": stage,
                "action_type": context["action_type"],
                "target_intent": context["target_intent"],
                "screenshot_path": context["screenshot_paths"].get(stage),
                "action_outcome": action_outcome,
            },
        )

    @staticmethod
    def _rect_center(rect: tuple[int, int, int, int] | None) -> tuple[int, int] | None:
        if rect is None:
            return None
        left, top, right, bottom = rect
        return (int((left + right) / 2), int((top + bottom) / 2))

    @staticmethod
    def _rect_distance(
        left_rect: tuple[int, int, int, int] | None,
        right_rect: tuple[int, int, int, int] | None,
    ) -> float:
        left_center = BrowserChatGPTTransport._rect_center(left_rect)
        right_center = BrowserChatGPTTransport._rect_center(right_rect)
        if left_center is None or right_center is None:
            return float("inf")
        return abs(left_center[0] - right_center[0]) + abs(left_center[1] - right_center[1])

    def _hint_rect(self, hint, target) -> tuple[int, int, int, int] | None:
        if hint is None:
            return None
        return denormalize_region(hint.region, self._window_rect(target))

    @staticmethod
    def _visual_intent_from_phrase(phrase: str) -> str:
        normalized = _normalize(phrase)
        if normalized == "send prompt":
            return "send_button_area"
        if normalized == "new chat":
            return "new_chat_button"
        return f"named_control:{normalized.replace(' ', '_')}"

    @staticmethod
    def _default_composer_rect(target) -> tuple[int, int, int, int]:
        return (
            int(target.left + target.width * 0.24),
            int(target.top + target.height * 0.79),
            int(target.left + target.width * 0.91),
            int(target.top + target.height * 0.94),
        )

    @staticmethod
    def _default_send_button_rect(target) -> tuple[int, int, int, int]:
        center_x = int(target.left + target.width * 0.95)
        center_y = int(target.top + target.height * 0.93)
        return BrowserChatGPTTransport._rect_from_center(center_x, center_y, radius_x=26, radius_y=22)

    @staticmethod
    def _default_reply_region_rect(target) -> tuple[int, int, int, int]:
        return (
            int(target.left + target.width * 0.38),
            int(target.top + target.height * 0.20),
            int(target.left + target.width * 0.94),
            int(target.top + target.height * 0.74),
        )

    def _run_visual_recovery_action(
        self,
        *,
        target,
        action_name: str,
        target_intent: str,
        callback,
    ) -> None:
        ui_state = self._ui_state(target)
        if target_intent == "composer_area":
            target_rect = self._default_composer_rect(target)
        elif target_intent in {"reply_region", "page_rescan"}:
            target_rect = self._default_reply_region_rect(target)
        else:
            target_rect = self._default_send_button_rect(target)
        context = self._begin_visual_action(
            target=target,
            action_type="recovery_action",
            target_intent=target_intent,
            ui_state=ui_state,
            target_rect=target_rect,
        )
        self._capture_visual_action_stage(
            context,
            target=target,
            stage="during",
            ui_state=ui_state,
            action_outcome=f"recovery_started:{action_name}",
        )
        try:
            callback()
        except Exception as exc:
            self._capture_visual_action_stage(
                context,
                target=target,
                stage="post",
                ui_state=self._ui_state(target),
                action_outcome=f"recovery_failed:{action_name}:{exc}",
            )
            raise
        self._capture_visual_action_stage(
            context,
            target=target,
            stage="post",
            ui_state=self._ui_state(target),
            action_outcome=f"recovery_completed:{action_name}",
        )

    def _ensure_runtime(self) -> None:
        if self._executor is not None:
            return
        self._log("runtime_init_start", {})
        self._activity(
            "browser_runtime",
            "Loading the browser automation runtime.",
            "Aster needs the Screen Reader modules and UI automation hooks before it can control ChatGPT.",
        )
        try:
            ActionExecutor, ScreenCaptureService, ocr_class = self._load_runtime_classes()
            _load_gui_dependencies()
        except Exception as exc:
            sibling = Path("C:/Screen Reader")
            if sibling.exists() and str(sibling) not in sys.path:
                sys.path.insert(0, str(sibling))
                self._log("runtime_added_sibling_path", {"path": str(sibling)})
            try:
                ActionExecutor, ScreenCaptureService, ocr_class = self._load_runtime_classes()
                _load_gui_dependencies()
            except Exception as nested_exc:
                self._log(
                    "runtime_init_failed",
                    {
                        "initial_error": str(exc),
                        "fallback_error": str(nested_exc),
                    },
                )
                self._activity(
                    "browser_runtime_failed",
                    "Browser runtime initialization failed.",
                    "Without the automation runtime, Aster cannot operate the ChatGPT browser window.",
                    status="error",
                    details={"error": str(nested_exc)},
                )
                raise RuntimeError(
                    "Browser mode could not load the Screen Reader runtime. "
                    f"Initial error: {exc}. Fallback error: {nested_exc}."
                ) from nested_exc
        self._executor = ActionExecutor()
        self._capture = ScreenCaptureService()
        self._ocr = ocr_class()
        self._log(
            "runtime_init_complete",
            {
                "executor": type(self._executor).__name__,
                "capture": type(self._capture).__name__,
                "ocr": type(self._ocr).__name__,
            },
        )
        self._activity(
            "browser_runtime_ready",
            "Browser automation runtime is ready.",
            "Aster can now inspect the ChatGPT window, read the screen, and interact with controls.",
            status="success",
        )

    @staticmethod
    def _load_runtime_classes():
        from screen_reader.automation import ActionExecutor
        from screen_reader.capture import ScreenCaptureService
        from screen_reader.ocr import ScreenOCR

        return ActionExecutor, ScreenCaptureService, ScreenOCR

    def generate(
        self,
        prompt: str,
        timeout_sec: float = 90.0,
        chatgpt_url: str = "https://chatgpt.com/",
        launch_timeout_sec: float = 45.0,
        *,
        retry_attempt: bool = False,
        retry_prompt_mode: str = "",
        retry_reason: str = "",
    ) -> BrowserResult:
        self._log(
            "generate_start",
            {
                "prompt_length": len(prompt),
                "timeout_sec": timeout_sec,
                "launch_timeout_sec": launch_timeout_sec,
                "retry_attempt": retry_attempt,
                "retry_prompt_mode": retry_prompt_mode,
                "retry_reason": retry_reason,
            },
        )
        self._activity(
            "browser_generate_start",
            "Starting a browser ChatGPT run.",
            "Aster will attach to ChatGPT, place the structured prompt, send it, and wait for a parseable reply.",
            details={"prompt_length": len(prompt)},
        )
        prompt_anchor = build_turn_anchor(prompt)
        self._last_reply_capture_snapshot = _ReplyCaptureSnapshot()
        self._start_visual_action_run()
        self._start_runtime_log_heartbeat()
        try:
            self.thread_registry.load()
        except Exception as exc:
            self._log("thread_registry_load_failed", {"error": str(exc), "path": str(self.thread_registry.path)})
        thread_route = self.thread_registry.choose_strategy(reuse_enabled=self.thread_reuse_enabled)
        self._log(
            "browser_strategy_selected",
            {
                "strategy": self.strategy.value,
                "thread_route": thread_route,
                "verification_level": self.verification_level,
                "log_screenshots": self.log_screenshots,
                "turn_anchor": serialize_anchor(prompt_anchor),
            },
        )
        # TODO: route into thread-aware conversation reuse once thread discovery is implemented.
        try:
            self._ensure_runtime()
            target = self._ensure_chatgpt_window(chatgpt_url=chatgpt_url, launch_timeout_sec=launch_timeout_sec)
            self._show_automation_notice()
            try:
                thread_reused_for_capture = self._prepare_chatgpt_window(
                    target,
                    chatgpt_url=chatgpt_url,
                    timeout_sec=min(20.0, max(8.0, launch_timeout_sec)),
                )
                image = self._capture.capture_region(target.left, target.top, target.width, target.height)
                before_lines = self._ocr.extract(image)
                self._log(
                    "initial_window_state",
                    {
                        "target_title": target.title,
                        "ocr_lines": len(before_lines),
                        "ocr_preview": [line.text[:120] for line in before_lines[:5]],
                        "ui_state": self._ui_state(target),
                    },
                )
                self._activity(
                    "browser_window_ready",
                    f"Attached to the ChatGPT window: {target.title}.",
                    "Aster needs the active browser window before it can place or read the prompt.",
                    details={"window_title": target.title},
                )
                self._activity(
                    "browser_prompt_fill",
                    "Trying to place the prompt directly into the ChatGPT composer.",
                    "Direct entry is more reliable than a raw paste because it avoids attachment-style paste behavior.",
                )
                populated_directly = self._populate_prompt_directly(target, prompt, prompt_anchor=prompt_anchor)
                self._log("populate_prompt_result", {"direct_uia_write": populated_directly})
                if not populated_directly:
                    click_point = self._executor.choose_chatgpt_composer_point(target, before_lines)
                    self._log("fallback_clipboard_send_start", {"click_point": click_point})
                    self._activity(
                        "browser_prompt_fallback",
                        "Direct composer entry failed, so Aster is falling back to paste automation.",
                        "This keeps the run moving even when the browser blocks direct UI automation text entry.",
                    )
                    populated_directly = self._paste_prompt_with_click(target, prompt, click_point, prompt_anchor)
                if not populated_directly:
                    self._activity(
                        "browser_prompt_missing",
                        "Aster could not confirm that the prompt actually appeared in the ChatGPT composer.",
                        "It is safer to stop here than to pretend the prompt was sent when the composer stayed blank.",
                        status="error",
                    )
                    raise RuntimeError(
                        "Aster could not confirm that the prompt was inserted into the ChatGPT composer. "
                        "The page stayed in blank composer state."
                    )
                self._stabilize_and_send(target, before_lines, prompt)
                self._activity(
                    "browser_wait_reply",
                    "Waiting for ChatGPT to start and finish its reply.",
                    "Aster needs a completed response before it can extract a machine-readable patch block.",
                )
                # TODO: route reply capture through browser_core.reply_tracker once viewport anchoring is implemented.
                reply = self._capture_reply_text(
                    target,
                    before_lines,
                    prompt,
                    timeout_sec=timeout_sec,
                    prompt_anchor=prompt_anchor,
                    thread_reused_for_capture=thread_reused_for_capture,
                    retry_attempt=retry_attempt,
                )
                parsed, final_capture_diagnostics = self._finalize_captured_reply(reply, retry_attempt=retry_attempt)
                self._log(
                    "generate_reply_captured",
                    {
                        "reply_length": len(reply),
                        "parsed_length": len(parsed),
                        "reply_preview": reply[:300],
                        **final_capture_diagnostics,
                    },
                )
                result_metadata: dict[str, Any] = {
                    "window_title": target.title,
                    "retry_seed_valid": final_capture_diagnostics.get("retry_seed_valid", bool(parsed.strip())),
                    "retry_seed_validity_reason": final_capture_diagnostics.get("retry_seed_validity_reason", ""),
                    "retry_seed_validity_reason_source": final_capture_diagnostics.get(
                        "retry_seed_validity_reason_source",
                        "",
                    ),
                    "retry_output_strict_mode": retry_attempt,
                    "retry_attempt_capture_mode": final_capture_diagnostics.get("retry_attempt_capture_mode", ""),
                    "retry_attempt_acceptance_tier": final_capture_diagnostics.get("retry_attempt_acceptance_tier", ""),
                    "retry_attempt_failure_reason": final_capture_diagnostics.get("retry_attempt_failure_reason", ""),
                    "retry_attempt_structured_block_found": final_capture_diagnostics.get(
                        "retry_attempt_structured_block_found",
                        False,
                    ),
                    "retry_attempt_parseable": final_capture_diagnostics.get("retry_attempt_parseable", False),
                    "retry_attempt_prose_contamination": final_capture_diagnostics.get(
                        "retry_attempt_prose_contamination",
                        False,
                    ),
                    "retry_attempt_wrapper_only": final_capture_diagnostics.get("retry_attempt_wrapper_only", False),
                    "retry_attempt_block_count": final_capture_diagnostics.get("retry_attempt_block_count", 0),
                    "retry_attempt_exact_block_only": final_capture_diagnostics.get(
                        "retry_attempt_exact_block_only",
                        False,
                    ),
                    "retry_attempt_extra_text_detected": final_capture_diagnostics.get(
                        "retry_attempt_extra_text_detected",
                        False,
                    ),
                    "retry_attempt_json_object_count": final_capture_diagnostics.get(
                        "retry_attempt_json_object_count",
                        0,
                    ),
                    "retry_attempt_selected_block_index": final_capture_diagnostics.get(
                        "retry_attempt_selected_block_index",
                        None,
                    ),
                    "retry_attempt_block_selection_reason": final_capture_diagnostics.get(
                        "retry_attempt_block_selection_reason",
                        "",
                    ),
                    "retry_attempt_multiple_blocks_ambiguous": final_capture_diagnostics.get(
                        "retry_attempt_multiple_blocks_ambiguous",
                        False,
                    ),
                    "retry_attempt_multiple_blocks_recovered": final_capture_diagnostics.get(
                        "retry_attempt_multiple_blocks_recovered",
                        False,
                    ),
                    "retry_attempt_block_relationship": final_capture_diagnostics.get(
                        "retry_attempt_block_relationship",
                        "",
                    ),
                    "retry_attempt_block_forensics": final_capture_diagnostics.get(
                        "retry_attempt_block_forensics",
                        [],
                    ),
                    "retry_attempt_wrapper_only_block_count": final_capture_diagnostics.get(
                        "retry_attempt_wrapper_only_block_count",
                        0,
                    ),
                    "retry_attempt_wrapper_only_payload_lengths": final_capture_diagnostics.get(
                        "retry_attempt_wrapper_only_payload_lengths",
                        [],
                    ),
                    "retry_attempt_wrapper_only_has_internal_text": final_capture_diagnostics.get(
                        "retry_attempt_wrapper_only_has_internal_text",
                        False,
                    ),
                    "retry_attempt_wrapper_only_noise_detected": final_capture_diagnostics.get(
                        "retry_attempt_wrapper_only_noise_detected",
                        False,
                    ),
                    "retry_attempt_wrapper_only_boundary_suspected": final_capture_diagnostics.get(
                        "retry_attempt_wrapper_only_boundary_suspected",
                        False,
                    ),
                    "retry_wrapper_recheck_attempted": final_capture_diagnostics.get(
                        "retry_wrapper_recheck_attempted",
                        False,
                    ),
                    "retry_wrapper_recheck_found_payload": final_capture_diagnostics.get(
                        "retry_wrapper_recheck_found_payload",
                        False,
                    ),
                    "retry_wrapper_recheck_reason": final_capture_diagnostics.get(
                        "retry_wrapper_recheck_reason",
                        "",
                    ),
                    "retry_attempt_fragment_repair_pattern_matched": final_capture_diagnostics.get(
                        "retry_attempt_fragment_repair_pattern_matched",
                        False,
                    ),
                    "retry_attempt_fragment_repair_attempted": final_capture_diagnostics.get(
                        "retry_attempt_fragment_repair_attempted",
                        False,
                    ),
                    "retry_attempt_fragment_repair_succeeded": final_capture_diagnostics.get(
                        "retry_attempt_fragment_repair_succeeded",
                        False,
                    ),
                    "retry_attempt_fragment_repair_reason": final_capture_diagnostics.get(
                        "retry_attempt_fragment_repair_reason",
                        "",
                    ),
                    "retry_attempt_repaired_from_block_index": final_capture_diagnostics.get(
                        "retry_attempt_repaired_from_block_index",
                        None,
                    ),
                    "retry_attempt_discarded_wrapper_only_block_index": final_capture_diagnostics.get(
                        "retry_attempt_discarded_wrapper_only_block_index",
                        None,
                    ),
                    "retry_attempt_fragment_repair_parseable_after_cleanup": final_capture_diagnostics.get(
                        "retry_attempt_fragment_repair_parseable_after_cleanup",
                        False,
                    ),
                    "retry_attempt_json_cleanup_attempted": final_capture_diagnostics.get(
                        "retry_attempt_json_cleanup_attempted",
                        False,
                    ),
                    "retry_attempt_json_cleanup_succeeded": final_capture_diagnostics.get(
                        "retry_attempt_json_cleanup_succeeded",
                        False,
                    ),
                    "retry_attempt_json_cleanup_reason": final_capture_diagnostics.get(
                        "retry_attempt_json_cleanup_reason",
                        "",
                    ),
                    "retry_attempt_json_cleanup_changed": final_capture_diagnostics.get(
                        "retry_attempt_json_cleanup_changed",
                        False,
                    ),
                    "retry_attempt_prose_recovery_attempted": final_capture_diagnostics.get(
                        "retry_attempt_prose_recovery_attempted",
                        False,
                    ),
                    "retry_attempt_prose_recovery_succeeded": final_capture_diagnostics.get(
                        "retry_attempt_prose_recovery_succeeded",
                        False,
                    ),
                    "retry_attempt_prose_recovery_reason": final_capture_diagnostics.get(
                        "retry_attempt_prose_recovery_reason",
                        "",
                    ),
                    "retry_attempt_fragment_source_preference": final_capture_diagnostics.get(
                        "retry_attempt_fragment_source_preference",
                        "",
                    ),
                    "retry_attempt_fragment_source_chosen": final_capture_diagnostics.get(
                        "retry_attempt_fragment_source_chosen",
                        "",
                    ),
                    "retry_attempt_fragment_uia_available": final_capture_diagnostics.get(
                        "retry_attempt_fragment_uia_available",
                        False,
                    ),
                    "retry_attempt_fragment_ocr_available": final_capture_diagnostics.get(
                        "retry_attempt_fragment_ocr_available",
                        False,
                    ),
                    "retry_attempt_ocr_cleanup_attempted": final_capture_diagnostics.get(
                        "retry_attempt_ocr_cleanup_attempted",
                        False,
                    ),
                    "retry_attempt_ocr_cleanup_succeeded": final_capture_diagnostics.get(
                        "retry_attempt_ocr_cleanup_succeeded",
                        False,
                    ),
                    "retry_attempt_ocr_cleanup_reason": final_capture_diagnostics.get(
                        "retry_attempt_ocr_cleanup_reason",
                        "",
                    ),
                    "retry_attempt_ocr_defect_types": final_capture_diagnostics.get(
                        "retry_attempt_ocr_defect_types",
                        [],
                    ),
                    "retry_attempt_single_block_completion_attempted": final_capture_diagnostics.get(
                        "retry_attempt_single_block_completion_attempted",
                        False,
                    ),
                    "retry_attempt_single_block_completion_succeeded": final_capture_diagnostics.get(
                        "retry_attempt_single_block_completion_succeeded",
                        False,
                    ),
                    "retry_attempt_single_block_completion_reason": final_capture_diagnostics.get(
                        "retry_attempt_single_block_completion_reason",
                        "",
                    ),
                    "retry_attempt_single_block_completion_defect_types": final_capture_diagnostics.get(
                        "retry_attempt_single_block_completion_defect_types",
                        [],
                    ),
                    "retry_attempt_single_block_completion_closure_added": final_capture_diagnostics.get(
                        "retry_attempt_single_block_completion_closure_added",
                        "",
                    ),
                    "retry_attempt_single_block_semantically_incomplete": final_capture_diagnostics.get(
                        "retry_attempt_single_block_semantically_incomplete",
                        False,
                    ),
                    "retry_attempt_single_block_parseable_after_completion": final_capture_diagnostics.get(
                        "retry_attempt_single_block_parseable_after_completion",
                        False,
                    ),
                    "final_capture_failure_reason": final_capture_diagnostics.get("final_capture_failure_reason", ""),
                    "salvage_preserved_for_diagnostics_only": final_capture_diagnostics.get(
                        "salvage_preserved_for_diagnostics_only",
                        False,
                    ),
                }
                if not parsed.strip():
                    failure_reason = final_capture_diagnostics.get("final_capture_failure_reason", "no_structured_block_seen")
                    if failure_reason == "structured_block_seen_but_not_retry_safe":
                        self._activity(
                            "browser_reply_partial",
                            "Captured a partial structured reply, but it was not trustworthy enough to reuse as a retry seed.",
                            "Aster preserved the best structured fragment for diagnostics only and will let the retry path continue without reusing malformed JSON.",
                            status="warning",
                            details={
                                "best_structured_candidate_length": final_capture_diagnostics.get(
                                    "best_structured_candidate_length",
                                    0,
                                ),
                                "retry_seed_validity_reason": final_capture_diagnostics.get(
                                    "retry_seed_validity_reason",
                                    "",
                                ),
                            },
                        )
                        return BrowserResult(
                            raw_text="",
                            metadata=result_metadata,
                        )
                    reason_messages = {
                        "no_structured_block_seen": "no structured patch block was seen during reply capture",
                        "structured_block_seen_but_lost": "a structured patch block was seen earlier but final extraction lost it",
                        "structured_block_seen_but_not_parseable": "structured patch text was seen, but none of it became parseable JSON",
                        "structured_block_seen_but_not_salvageable": "a structured patch block was seen earlier, but it was not trusted enough to salvage",
                        "structured_block_seen_but_region_trust_too_low": "a structured patch block was seen earlier, but the reply-region evidence was too weak to trust it for salvage",
                        "structured_block_seen_but_not_retry_safe": "a structured patch block was seen earlier, but none of it was safe enough to reuse as a retry seed",
                    }
                    raise RuntimeError(
                        "Browser mode could not capture a final ChatGPT response: "
                        f"{reason_messages.get(failure_reason, failure_reason)}."
                    )
                self._activity(
                    "browser_reply_ready",
                    "Captured a reply from ChatGPT and extracted the structured block.",
                    "Aster can now parse the response into exact file operations.",
                    status="success",
                    details={"reply_length": len(reply), "parsed_length": len(parsed)},
                )
                return BrowserResult(
                    raw_text=parsed,
                    metadata=result_metadata,
                )
            finally:
                self._hide_automation_notice()
        except Exception as exc:
            self._log(
                "generate_failed",
                {
                    "error": str(exc),
                    "traceback": "".join(traceback.format_exception_only(type(exc), exc)).strip(),
                },
            )
            recovery = decide_and_execute_recovery(
                self._last_page_classification,
                attempts_used=1,
                max_attempts=self.max_recovery_attempts,
                handlers=build_recovery_handlers(rescan=lambda: None),
            )
            self._log("recovery_decision", decision_payload(recovery))
            self._activity(
                "browser_failed",
                "Browser mode failed before a valid patch block was captured.",
                "The ChatGPT page did not reach a clean sent-and-replied state that Aster could parse.",
                status="error",
                details={"error": str(exc)},
            )
            raise
        finally:
            self._stop_runtime_log_heartbeat()

    def _show_automation_notice(self) -> None:
        self._activity(
            "browser_takeover_notice",
            "Aster is about to use the keyboard and mouse.",
            "Avoiding user input during the automation steps keeps the browser run from being interrupted.",
        )
        self._automation_notice.show()

    def _hide_automation_notice(self) -> None:
        self._automation_notice.hide()

    def _stabilize_and_send(self, target, before_lines, prompt: str) -> None:
        deadline = time.monotonic() + 35.0
        showed_text = False
        send_attempt = 0
        while time.monotonic() < deadline:
            self._tick_runtime_log_heartbeat("send_loop")
            image = self._capture.capture_region(target.left, target.top, target.width, target.height)
            lines = self._ocr.extract(image)
            ui_state = self._ui_state(target)
            self._log_window_state(target, lines, prompt, ui_state=ui_state)
            if send_attempt > 0:
                self._raise_for_browser_error(lines, ui_state, stage="send_loop")

            show_line = self._find_line(lines, "show in text field")
            if show_line is not None and not showed_text:
                self._log("show_in_text_field_detected", {"line_text": show_line.text[:120]})
                self._activity(
                    "browser_show_in_text_field",
                    "ChatGPT turned the pasted prompt into an attachment chip, so Aster is moving it back into the text field.",
                    "The prompt has to be in the actual composer for the Send button flow to work reliably.",
                )
                if not self._click_named_button(target, "show in text field"):
                    self._click_line(target, show_line)
                self._focus_composer(target, lines)
                showed_text = True
                time.sleep(1.5)
                continue

            if send_attempt > 0 and self._reply_started(
                before_lines,
                lines,
                target,
                ui_state=ui_state,
                send_attempted=True,
            ):
                self._log("reply_started_before_send_retry", {"send_attempts": send_attempt})
                self._activity(
                    "browser_reply_started",
                    "ChatGPT has started responding.",
                    "Aster detected reply content before attempting another resend.",
                    status="success",
                )
                return

            if self._needs_send_retry(lines, prompt):
                self._log(
                    "send_retry_needed",
                    {
                        "attempt_index": send_attempt,
                        "ui_state": self._ui_state(target),
                    },
                )
                self._activity(
                    "browser_send_retry",
                    "The prompt still looks unsent, so Aster is trying to send it again.",
                    "The composer state has not transitioned into a reply state yet.",
                    details={"attempt_index": send_attempt},
                )
                self._focus_composer(target, lines)
                self._attempt_send(target, send_attempt)
                send_attempt += 1
                if self._wait_for_reply_start(target, before_lines, seconds=4.0):
                    self._log("reply_started_after_send_attempt", {"attempt_index": send_attempt - 1})
                    self._activity(
                        "browser_reply_started",
                        "ChatGPT has started responding.",
                        "The prompt appears to have been accepted and the page has moved out of composer state.",
                        status="success",
                    )
                    return
                continue

            if send_attempt > 0 and self._reply_started(
                before_lines,
                lines,
                target,
                ui_state=ui_state,
                send_attempted=True,
            ):
                self._log("reply_started_before_send_loop_exit", {"send_attempts": send_attempt})
                self._activity(
                    "browser_reply_started",
                    "ChatGPT has started responding.",
                    "The page changed in a way that indicates the prompt was sent successfully.",
                    status="success",
                )
                return

            time.sleep(1.0)

        self._log("send_loop_timeout", {"ui_state": self._ui_state(target)})
        raise RuntimeError(
            "Aster pasted the prompt into ChatGPT but could not confirm it was sent. "
            "The browser page did not transition from composer state to reply state."
        )

    @staticmethod
    def _composer_looks_idle_after_send(ui_state: dict[str, Any] | None) -> bool:
        state = ui_state or {}
        if state.get("send_prompt_present") or state.get("show_in_text_field_present"):
            return False
        preview = _normalize(str(state.get("composer_edit_preview", "")))
        if not preview:
            return False
        idle_hints = ("ask anything", "message chatgpt", "type a message", "send a message")
        if not any(hint in preview for hint in idle_hints):
            return False
        composer_length = state.get("composer_edit_length")
        if isinstance(composer_length, int):
            return composer_length <= 24
        return composer_length in {None, "unknown"}

    def _reply_started(
        self,
        before_lines,
        after_lines,
        target,
        ui_state: dict[str, Any] | None = None,
        *,
        send_attempted: bool = False,
    ) -> bool:
        state = ui_state or self._ui_state(target)
        if (
            state.get("stop_streaming_present")
            and not state.get("send_prompt_present")
            and not state.get("show_in_text_field_present")
        ):
            return True
        candidate = self._executor.read_chatgpt_browser_reply(
            target,
            self._capture,
            self._ocr,
            before_lines,
            "",
            timeout_sec=1.0,
        )
        if candidate.strip():
            if reply_detection_blocked(state):
                self._log(
                    "reply_detection_suppressed",
                    {
                        "reason": "unsent_composer_state",
                        "candidate_preview": candidate[:160],
                        "ui_state": state,
                    },
                )
                return False
            if not looks_like_reply_started_candidate_for_policy(
                candidate,
                ui_state=state,
                policy=REPLY_TRACKER_POLICY,
            ):
                self._log(
                    "reply_detection_suppressed",
                    {
                        "reason": "low_signal_candidate",
                        "candidate_preview": candidate[:160],
                        "score": score_candidate_for_policy(candidate, policy=REPLY_TRACKER_POLICY),
                        "ui_state": state,
                    },
                )
                if send_attempted and self._composer_looks_idle_after_send(state):
                    self._log(
                        "reply_started_after_composer_reset",
                        {
                            "reason": "low_signal_candidate_but_idle_composer_reset",
                            "candidate_preview": candidate[:160],
                            "ui_state": state,
                        },
                    )
                    return True
                return False
            return True
        before_text = {_normalize(line.text) for line in before_lines}
        for line in after_lines:
            lowered = _normalize(line.text)
            if len(lowered) < 12:
                continue
            if lowered in before_text:
                continue
            if "show in text field" in lowered:
                continue
            if "ask anything" in lowered:
                continue
            if reply_detection_blocked(state):
                self._log(
                    "reply_detection_suppressed",
                    {
                        "reason": "unsent_composer_state",
                        "line_preview": lowered[:160],
                        "ui_state": state,
                    },
                )
                return False
            if not looks_like_reply_started_candidate_for_policy(
                lowered,
                ui_state=state,
                policy=REPLY_TRACKER_POLICY,
            ):
                self._log(
                    "reply_detection_suppressed",
                    {
                        "reason": "low_signal_line",
                        "line_preview": lowered[:160],
                        "score": score_candidate_for_policy(lowered, policy=REPLY_TRACKER_POLICY),
                        "ui_state": state,
                    },
                )
                if send_attempted and self._composer_looks_idle_after_send(state):
                    self._log(
                        "reply_started_after_composer_reset",
                        {
                            "reason": "low_signal_line_but_idle_composer_reset",
                            "line_preview": lowered[:160],
                            "ui_state": state,
                        },
                    )
                    return True
                return False
            return True
        return False

    def _needs_send_retry(self, lines, prompt: str) -> bool:
        full = "\n".join(line.text.lower() for line in lines)
        prompt_words = [word for word in re.findall(r"[a-z0-9]{4,}", prompt.lower())[:8]]
        if "show in text field" in full:
            return True
        if any(word in full for word in prompt_words):
            return True
        if "system" in full and "ask anything" in full:
            return True
        return "ask anything" in full

    def _click_send_button(self, target) -> None:
        ui_state = self._ui_state(target)
        hint = self._visual_memory_hint(intent="send_button_area", target=target, ui_state=ui_state)
        hint_rect = self._hint_rect(hint, target)
        fallback_rect = hint_rect or self._default_send_button_rect(target)
        context = self._begin_visual_action(
            target=target,
            action_type="click_action",
            target_intent="send_button_area",
            ui_state=ui_state,
            target_rect=fallback_rect,
            candidate_rects=[fallback_rect] if fallback_rect is not None else [],
            confidence_before=hint.confidence if hint is not None else None,
            used_remembered_region=hint is not None,
        )
        if self._click_named_button(target, "send prompt"):
            self._capture_visual_action_stage(
                context,
                target=target,
                stage="during",
                ui_state=self._ui_state(target),
                action_outcome="delegated_named_send_click",
                confidence_after=hint.confidence if hint is not None else None,
            )
            self._capture_visual_action_stage(
                context,
                target=target,
                stage="post",
                ui_state=self._ui_state(target),
                action_outcome="send_click_completed",
                confidence_after=hint.confidence if hint is not None else None,
            )
            return
        self._focus_window(target)
        self._capture_visual_action_stage(
            context,
            target=target,
            stage="during",
            ui_state=ui_state,
            action_outcome="coordinate_send_click_start",
            confidence_after=hint.confidence if hint is not None else None,
        )
        click_center = self._rect_center(fallback_rect) or (
            int(target.left + target.width * 0.95),
            int(target.top + target.height * 0.93),
        )
        x, y = click_center
        self._log("send_button_coordinate_click", {"x": x, "y": y, "used_remembered_region": hint is not None})
        pyautogui.click(x, y)
        target_rect = self._rect_from_center(x, y, radius_x=26, radius_y=22)
        self._visual_memory_update(
            intent="send_button_area",
            target=target,
            action_type="click_action",
            rect=target_rect,
            ui_state=self._ui_state(target),
        )
        self._capture_visual_action_stage(
            context,
            target=target,
            stage="post",
            ui_state=self._ui_state(target),
            action_outcome="coordinate_send_click_completed",
            confidence_after=1.0 if hint is not None else None,
            updated_remembered_region=True,
        )

    def _click_line(self, target, line, *, target_intent: str = "click_target", action_type: str = "click_action") -> None:
        x = int(target.left + line.center[0])
        y = int(target.top + line.center[1])
        target_rect = self._rect_from_center(x, y)
        context = self._begin_visual_action(
            target=target,
            action_type=action_type,
            target_intent=target_intent,
            target_rect=target_rect,
            candidate_rects=[target_rect],
        )
        self._focus_window(target)
        self._capture_visual_action_stage(
            context,
            target=target,
            stage="during",
            ui_state=self._ui_state(target),
            action_outcome="click_line_start",
        )
        pyautogui.click(x, y)
        if target_intent in {"composer_area", "reply_region"}:
            self._visual_memory_update(
                intent=target_intent,
                target=target,
                action_type=action_type,
                rect=target_rect,
                ui_state=self._ui_state(target),
                lines=[line],
            )
        self._capture_visual_action_stage(
            context,
            target=target,
            stage="post",
            ui_state=self._ui_state(target),
            action_outcome="click_line_completed",
            updated_remembered_region=target_intent in {"composer_area", "reply_region"},
        )

    def _click_named_button(self, target, phrase: str) -> bool:
        return self._click_named_control(target, phrase, control_types=("Button",))

    def _log_uia_diagnostic(self, event: str, payload: dict[str, Any]) -> None:
        signature = f"{event}:{payload}"
        if signature == self._last_uia_diagnostic_signature:
            return
        self._last_uia_diagnostic_signature = signature
        self._log(event, payload)

    @staticmethod
    def _control_rect_tuple(rect) -> tuple[int, int, int, int]:
        return (int(rect.left), int(rect.top), int(rect.right), int(rect.bottom))

    @classmethod
    def _control_is_near_composer_area(cls, target, rect, control_type: str) -> bool:
        rel_top = rect.top - target.top
        rel_bottom = rect.bottom - target.top
        rel_left = rect.left - target.left
        if rel_bottom < target.height * 0.45:
            return False
        if rel_left < target.width * 0.12 and control_type not in {"Edit", "Button"}:
            return False
        return control_type in {"Edit", "Button", "Text", "Document", "Group"}

    @classmethod
    def _score_named_button_candidate(cls, target, rect, name: str, phrase: str) -> float:
        normalized = _normalize(name)
        phrase_normalized = _normalize(phrase)
        rel_left = rect.left - target.left
        rel_top = rect.top - target.top
        width = max(0, rect.right - rect.left)
        height = max(0, rect.bottom - rect.top)
        score = 0.0
        if phrase_normalized and phrase_normalized in normalized:
            score += 140.0
        if "send" in normalized:
            score += 70.0
        for token in re.findall(r"[a-z0-9]{3,}", phrase_normalized):
            if token in normalized:
                score += 30.0
        if rel_top >= target.height * 0.55:
            score += 25.0
        if rel_left >= target.width * 0.72:
            score += 25.0
        if 18 <= height <= 72:
            score += 10.0
        if 20 <= width <= target.width * 0.22:
            score += 10.0
        if normalized in {"share", "show in text field", "stop streaming", "stop generating"}:
            score -= 60.0
        return score

    def _build_uia_control_diagnostics(self, target, *, phrase: str = "send prompt") -> dict[str, Any]:
        diagnostics: dict[str, Any] = {
            "nearby_controls": [],
            "button_candidates": summarize_named_button_candidates([]),
            "edit_candidates": summarize_edit_candidates([]),
        }
        try:
            window = Desktop(backend="uia").window(handle=target.handle)
            nearby_controls: list[dict[str, Any]] = []
            button_candidates: list[dict[str, Any]] = []
            edit_candidates: list[dict[str, Any]] = []
            for ctrl in window.descendants():
                try:
                    control_type = str(ctrl.element_info.control_type or "")
                    rect = ctrl.rectangle()
                    name = " ".join((ctrl.window_text() or "").split())
                except Exception:
                    continue
                try:
                    enabled = ctrl.is_enabled()
                except Exception:
                    enabled = None
                rect_tuple = self._control_rect_tuple(rect)
                raw = {
                    "control_type": control_type,
                    "name": name,
                    "rect": rect_tuple,
                    "enabled": enabled,
                }
                if self._control_is_near_composer_area(target, rect, control_type):
                    nearby_controls.append(raw)
                if control_type == "Button":
                    match_score = self._score_named_button_candidate(target, rect, name, phrase)
                    button_candidates.append(
                        {
                            **raw,
                            "match_score": match_score,
                            "exact_match": bool(phrase and _normalize(phrase) in _normalize(name)),
                            "looks_send_like": match_score >= 55.0 and bool(name),
                        }
                    )
                if control_type == "Edit":
                    edit_candidates.append(
                        {
                            **raw,
                            "score": self._score_composer_edit_candidate(target, rect, name),
                        }
                    )
            diagnostics = {
                "nearby_controls": summarize_controls_near_composer_area(nearby_controls),
                "button_candidates": summarize_named_button_candidates(button_candidates),
                "edit_candidates": summarize_edit_candidates(edit_candidates),
            }
        except Exception as exc:
            diagnostics["inspection_error"] = str(exc)
        self._last_uia_control_diagnostics = diagnostics
        return diagnostics

    def _click_named_control(self, target, phrase: str, control_types: tuple[str, ...] | None = None) -> bool:
        ui_state = self._ui_state(target)
        memory_intent = self._visual_intent_from_phrase(phrase)
        hint = self._visual_memory_hint(intent=memory_intent, target=target, ui_state=ui_state)
        hint_rect = self._hint_rect(hint, target)
        candidate_rects: list[tuple[int, int, int, int]] = []
        context = self._begin_visual_action(
            target=target,
            action_type="named_control_click",
            target_intent=memory_intent,
            ui_state=ui_state,
            target_rect=hint_rect,
            candidate_rects=candidate_rects,
            confidence_before=hint.confidence if hint is not None else None,
            used_remembered_region=hint is not None,
        )
        try:
            window = Desktop(backend="uia").window(handle=target.handle)
            candidates: list[dict[str, Any]] = []
            for ctrl in window.descendants():
                name = (ctrl.window_text() or "").strip().lower()
                if control_types is not None:
                    try:
                        control_type = ctrl.element_info.control_type
                    except Exception:
                        continue
                    if control_type not in control_types:
                        continue
                if phrase in name:
                    try:
                        rect = self._control_rect_tuple(ctrl.rectangle())
                    except Exception:
                        rect = None
                    base_score = 0.0
                    if rect is not None and control_types == ("Button",):
                        rect_obj = ctrl.rectangle()
                        base_score = self._score_named_button_candidate(target, rect_obj, name, phrase)
                    distance = self._rect_distance(rect, hint_rect)
                    candidates.append(
                        {
                            "control": ctrl,
                            "name": name,
                            "rect": rect,
                            "base_score": base_score,
                            "hint_distance": distance,
                        }
                    )
        except Exception as exc:
            self._capture_visual_action_stage(
                context,
                target=target,
                stage="post",
                ui_state=self._ui_state(target),
                action_outcome=f"named_control_search_failed:{phrase}",
                confidence_after=hint.confidence if hint is not None else None,
            )
            self._log_uia_diagnostic(
                "named_control_search_failed",
                {
                    "phrase": phrase,
                    "control_types": list(control_types or ()),
                    "error": str(exc),
                    "uia_control_diagnostics": self._build_uia_control_diagnostics(target, phrase=phrase),
                },
            )
            return False
        if candidates:
            candidate_rects.extend(
                rect for rect in (item.get("rect") for item in candidates[:8]) if isinstance(rect, tuple)
            )
            if context is not None:
                context["candidate_rects"] = list(candidate_rects)
        ordered = sorted(
            candidates,
            key=lambda item: (
                0 if item["hint_distance"] != float("inf") else 1,
                item["hint_distance"],
                -float(item["base_score"]),
                item["name"],
            ),
        )
        if not ordered:
            if hint is not None:
                self._visual_memory_invalidate(intent=memory_intent, target=target, reason_weight=0.18)
            self._capture_visual_action_stage(
                context,
                target=target,
                stage="post",
                ui_state=self._ui_state(target),
                action_outcome=f"named_control_not_found:{phrase}",
                confidence_after=hint.confidence if hint is not None else None,
            )
            self._log_uia_diagnostic(
                "named_control_search_failed",
                {
                    "phrase": phrase,
                    "control_types": list(control_types or ()),
                    "uia_control_diagnostics": self._build_uia_control_diagnostics(target, phrase=phrase),
                },
            )
            return False
        last_error = None
        for item in ordered:
            ctrl = item["control"]
            rect = item.get("rect")
            if rect is not None and context is not None:
                context["target_rect"] = rect
            self._capture_visual_action_stage(
                context,
                target=target,
                stage="during",
                ui_state=self._ui_state(target),
                action_outcome=f"named_control_click_attempt:{phrase}",
                confidence_after=hint.confidence if hint is not None else None,
            )
            try:
                if self._invoke_button(ctrl):
                    self._log("button_invoke", {"phrase": phrase, "name": item["name"], "method": "invoke"})
                else:
                    ctrl.click_input()
                    self._log("button_invoke", {"phrase": phrase, "name": item["name"], "method": "click_input"})
                if rect is not None:
                    self._visual_memory_update(
                        intent=memory_intent,
                        target=target,
                        action_type="named_control_click",
                        rect=rect,
                        ui_state=self._ui_state(target),
                    )
                self._capture_visual_action_stage(
                    context,
                    target=target,
                    stage="post",
                    ui_state=self._ui_state(target),
                    action_outcome=f"named_control_click_completed:{phrase}",
                    confidence_after=1.0 if hint is not None else None,
                    updated_remembered_region=rect is not None,
                )
                return True
            except Exception as exc:
                last_error = exc
                self._log_uia_diagnostic(
                    "named_control_click_failed",
                    {
                        "phrase": phrase,
                        "name": item["name"],
                        "control_types": list(control_types or ()),
                        "error": str(exc),
                        "uia_control_diagnostics": self._build_uia_control_diagnostics(target, phrase=phrase),
                    },
                )
        if hint is not None:
            self._visual_memory_invalidate(intent=memory_intent, target=target, reason_weight=0.22)
        self._capture_visual_action_stage(
            context,
            target=target,
            stage="post",
            ui_state=self._ui_state(target),
            action_outcome=f"named_control_click_failed:{phrase}",
            confidence_after=hint.confidence if hint is not None else None,
        )
        self._log_uia_diagnostic(
            "named_control_search_failed",
            {
                "phrase": phrase,
                "control_types": list(control_types or ()),
                "error": str(last_error) if last_error is not None else None,
                "uia_control_diagnostics": self._build_uia_control_diagnostics(target, phrase=phrase),
            },
        )
        return False

    def _populate_prompt_directly(self, target, prompt: str, *, prompt_anchor=None) -> bool:
        ui_state = self._ui_state(target)
        hint = self._visual_memory_hint(intent="composer_area", target=target, ui_state=ui_state)
        hint_rect = self._hint_rect(hint, target)
        composer = self._find_composer_edit(target)
        composer_rect = None
        if composer is not None:
            try:
                composer_rect = self._control_rect_tuple(composer.rectangle())
            except Exception:
                composer_rect = None
        context = self._begin_visual_action(
            target=target,
            action_type="prompt_insertion",
            target_intent="composer_area",
            ui_state=ui_state,
            target_rect=composer_rect or hint_rect or self._default_composer_rect(target),
            candidate_rects=[rect for rect in [composer_rect, hint_rect] if rect is not None],
            confidence_before=hint.confidence if hint is not None else None,
            used_remembered_region=hint is not None,
        )
        if composer is None:
            self._capture_visual_action_stage(
                context,
                target=target,
                stage="post",
                ui_state=self._ui_state(target),
                action_outcome="composer_edit_not_found",
                confidence_after=hint.confidence if hint is not None else None,
            )
            self._log("composer_edit_not_found", {"ui_state": self._ui_state(target)})
            return False
        try:
            self._focus_window(target)
            wrapper = composer
            if hasattr(composer, "wrapper_object"):
                try:
                    wrapper = composer.wrapper_object()
                except Exception:
                    wrapper = composer
            try:
                wrapper.click_input()
            except Exception:
                pass
            try:
                wrapper.set_focus()
            except Exception:
                pass
            methods = (
                ("set_edit_text", lambda: wrapper.set_edit_text(prompt)),
                ("type_keys", lambda: wrapper.type_keys(prompt, with_spaces=True, pause=0.001, set_foreground=True)),
                ("clipboard_paste", lambda: self._paste_prompt_via_clipboard(target, prompt, click_point=None)),
            )
            for method_name, method in methods:
                try:
                    self._clear_composer(target)
                    self._capture_visual_action_stage(
                        context,
                        target=target,
                        stage="during",
                        ui_state=self._ui_state(target),
                        action_outcome=f"prompt_insertion_method:{method_name}",
                        confidence_after=hint.confidence if hint is not None else None,
                    )
                    method()
                    if self._wait_for_prompt_inserted(target, prompt, seconds=6.0, prompt_anchor=prompt_anchor):
                        update_rect = composer_rect
                        if update_rect is not None:
                            self._visual_memory_update(
                                intent="composer_area",
                                target=target,
                                action_type="prompt_insertion",
                                rect=update_rect,
                                ui_state=self._ui_state(target),
                            )
                        self._capture_visual_action_stage(
                            context,
                            target=target,
                            stage="post",
                            ui_state=self._ui_state(target),
                            action_outcome=f"prompt_insertion_completed:{method_name}",
                            confidence_after=1.0 if hint is not None else None,
                            updated_remembered_region=update_rect is not None,
                        )
                        self._log("composer_populated", {"method": method_name, "prompt_length": len(prompt)})
                        return True
                except Exception as exc:
                    self._log(
                        "composer_population_method_failed",
                        {"method": method_name, "error": str(exc), "ui_state": self._ui_state(target)},
                    )
        except Exception as exc:
            if hint is not None:
                self._visual_memory_invalidate(intent="composer_area", target=target, reason_weight=0.14)
            self._capture_visual_action_stage(
                context,
                target=target,
                stage="post",
                ui_state=self._ui_state(target),
                action_outcome=f"prompt_insertion_failed:{exc}",
                confidence_after=hint.confidence if hint is not None else None,
            )
            self._log("composer_population_failed", {"error": str(exc), "ui_state": self._ui_state(target)})
            return False
        if hint is not None:
            self._visual_memory_invalidate(intent="composer_area", target=target, reason_weight=0.1)
        self._capture_visual_action_stage(
            context,
            target=target,
            stage="post",
            ui_state=self._ui_state(target),
            action_outcome="prompt_insertion_unconfirmed",
            confidence_after=hint.confidence if hint is not None else None,
        )
        self._log("composer_population_failed", {"ui_state": self._ui_state(target)})
        return False

    def _paste_prompt_with_click(self, target, prompt: str, click_point: tuple[int, int], prompt_anchor=None) -> bool:
        target_rect = self._rect_from_center(click_point[0], click_point[1], radius_x=30, radius_y=24)
        context = self._begin_visual_action(
            target=target,
            action_type="paste_insertion",
            target_intent="composer_area",
            ui_state=self._ui_state(target),
            target_rect=target_rect,
            candidate_rects=[target_rect],
        )
        self._focus_window(target)
        self._capture_visual_action_stage(
            context,
            target=target,
            stage="during",
            ui_state=self._ui_state(target),
            action_outcome="paste_insertion_started",
        )
        pyautogui.click(click_point[0], click_point[1])
        time.sleep(0.25)
        self._clear_composer(target)
        self._paste_prompt_via_clipboard(target, prompt, click_point=click_point)
        inserted = self._wait_for_prompt_inserted(target, prompt, seconds=6.0, prompt_anchor=prompt_anchor)
        if inserted:
            self._visual_memory_update(
                intent="composer_area",
                target=target,
                action_type="paste_insertion",
                rect=target_rect,
                ui_state=self._ui_state(target),
            )
        else:
            self._visual_memory_invalidate(intent="composer_area", target=target, reason_weight=0.08)
        self._capture_visual_action_stage(
            context,
            target=target,
            stage="post",
            ui_state=self._ui_state(target),
            action_outcome="paste_insertion_confirmed" if inserted else "paste_insertion_unconfirmed",
            updated_remembered_region=inserted,
        )
        return inserted

    def _paste_prompt_via_clipboard(self, target, prompt: str, click_point: tuple[int, int] | None) -> None:
        previous_clipboard = ""
        try:
            import pyperclip

            previous_clipboard = pyperclip.paste()
            pyperclip.copy(prompt)
            time.sleep(0.15)
            self._focus_window(target)
            if click_point is not None:
                pyautogui.click(click_point[0], click_point[1])
                time.sleep(0.15)
            pyautogui.hotkey("ctrl", "v")
            time.sleep(0.35)
            pyperclip.copy(previous_clipboard)
        except Exception:
            try:
                pyautogui.write(prompt, interval=0.001)
            except Exception:
                pass

    def _clear_composer(self, target) -> None:
        self._focus_window(target)
        pyautogui.hotkey("ctrl", "a")
        time.sleep(0.05)
        pyautogui.press("backspace")
        time.sleep(0.1)

    def _wait_for_prompt_inserted(self, target, prompt: str, seconds: float, *, prompt_anchor=None) -> bool:
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            self._tick_runtime_log_heartbeat("wait_for_prompt_inserted")
            image = self._capture.capture_region(target.left, target.top, target.width, target.height)
            lines = self._ocr.extract(image)
            ui_state = self._ui_state(target)
            if prompt_insertion_confirmed(
                lines,
                prompt,
                ui_state,
                prompt_anchor=prompt_anchor,
                policy=DEFAULT_COMPOSER_VERIFICATION_POLICY,
            ):
                self._log(
                    "prompt_inserted_confirmed",
                    {"ui_state": ui_state, "ocr_preview": [line.text[:120] for line in lines[:6]]},
                )
                return True
            time.sleep(0.5)
        self._log("prompt_inserted_missing", {"ui_state": self._ui_state(target)})
        return False

    def _find_composer_edit(self, target):
        try:
            window = Desktop(backend="uia").window(handle=target.handle)
            best = None
            best_score = float("-inf")
            candidates: list[dict[str, Any]] = []
            for ctrl in window.descendants(control_type="Edit"):
                try:
                    name = (ctrl.window_text() or "").strip().lower()
                    rect = ctrl.rectangle()
                except Exception:
                    continue
                if "chatgpt.com" in name:
                    continue
                score = self._score_composer_edit_candidate(target, rect, name)
                candidates.append(
                    {
                        "name": name,
                        "rect": self._control_rect_tuple(rect),
                        "enabled": None,
                        "score": score,
                    }
                )
                if score > best_score:
                    best = ctrl
                    best_score = score
            if best_score < 0:
                self._log_uia_diagnostic(
                    "composer_edit_not_found",
                    {
                        "edit_candidates": summarize_edit_candidates(candidates),
                        "uia_control_diagnostics": self._build_uia_control_diagnostics(target),
                    },
                )
                return None
            return best
        except Exception:
            return None

    @classmethod
    def _score_composer_edit_candidate(cls, target, rect, text: str) -> float:
        width = max(0, rect.right - rect.left)
        height = max(0, rect.bottom - rect.top)
        rel_top = rect.top - target.top
        rel_bottom = rect.bottom - target.top
        score = 0.0
        if width >= target.width * 0.35:
            score += 40.0
        else:
            score -= 40.0
        if rel_top >= target.height * 0.45:
            score += 60.0
        else:
            score -= 80.0
        if rel_bottom >= target.height * 0.72:
            score += 35.0
        if height >= 24:
            score += 10.0
        normalized = _normalize(text)
        if looks_like_browser_url_text(normalized):
            score -= 200.0
        if any(hint in normalized for hint in BROWSER_READY_HINTS):
            score += 50.0
        if normalized.startswith("system:") or normalized.startswith("user goal:"):
            score += 25.0
        return score

    def _focus_composer(self, target, lines) -> None:
        ui_state = self._ui_state(target)
        hint = self._visual_memory_hint(intent="composer_area", target=target, ui_state=ui_state, lines=lines)
        hint_rect = self._hint_rect(hint, target)
        composer = self._find_composer_line(lines)
        target_rect = hint_rect or self._default_composer_rect(target)
        if composer is not None:
            x = int(target.left + composer.center[0])
            y = int(target.top + composer.center[1])
            target_rect = self._rect_from_center(x, y)
        context = self._begin_visual_action(
            target=target,
            action_type="composer_focus",
            target_intent="composer_area",
            ui_state=ui_state,
            lines=lines,
            target_rect=target_rect,
            candidate_rects=[rect for rect in [hint_rect, target_rect] if rect is not None],
            confidence_before=hint.confidence if hint is not None else None,
            used_remembered_region=hint is not None,
        )
        if composer is not None:
            self._capture_visual_action_stage(
                context,
                target=target,
                stage="during",
                ui_state=ui_state,
                lines=lines,
                action_outcome="composer_focus_line_click",
                confidence_after=hint.confidence if hint is not None else None,
            )
            self._click_line(target, composer, target_intent="composer_area", action_type="composer_focus")
            self._capture_visual_action_stage(
                context,
                target=target,
                stage="post",
                ui_state=self._ui_state(target),
                action_outcome="composer_focus_completed",
                confidence_after=hint.confidence if hint is not None else None,
            )
            time.sleep(0.2)
            return
        self._focus_window(target)
        self._capture_visual_action_stage(
            context,
            target=target,
            stage="during",
            ui_state=ui_state,
            lines=lines,
            action_outcome="composer_focus_area_click",
            confidence_after=hint.confidence if hint is not None else None,
        )
        click_rect = hint_rect or self._default_composer_rect(target)
        click_center = self._rect_center(click_rect) or (
            int(target.left + target.width * 0.55),
            int(target.top + target.height * 0.88),
        )
        pyautogui.click(click_center[0], click_center[1])
        if hint_rect is not None:
            self._visual_memory_update(
                intent="composer_area",
                target=target,
                action_type="composer_focus",
                rect=click_rect,
                ui_state=self._ui_state(target),
                lines=lines,
            )
        self._capture_visual_action_stage(
            context,
            target=target,
            stage="post",
            ui_state=self._ui_state(target),
            action_outcome="composer_focus_completed",
            confidence_after=hint.confidence if hint is not None else None,
            updated_remembered_region=hint_rect is not None,
        )
        time.sleep(0.2)

    def _attempt_send(self, target, attempt_index: int) -> None:
        strategy = attempt_index % 5
        ui_state = self._ui_state(target)
        hint = self._visual_memory_hint(intent="send_button_area", target=target, ui_state=ui_state)
        context = self._begin_visual_action(
            target=target,
            action_type="send_action",
            target_intent="send_button_area",
            ui_state=ui_state,
            target_rect=self._hint_rect(hint, target) or self._default_send_button_rect(target),
            confidence_before=hint.confidence if hint is not None else None,
            used_remembered_region=hint is not None,
        )
        self._focus_window(target)
        self._capture_visual_action_stage(
            context,
            target=target,
            stage="during",
            ui_state=ui_state,
            action_outcome=f"send_strategy_{strategy}_start",
            confidence_after=hint.confidence if hint is not None else None,
        )
        if strategy == 0:
            self._log("send_attempt", {"attempt_index": attempt_index, "strategy": "button_or_enter"})
            self._activity(
                "browser_send_attempt",
                "Trying the primary send path.",
                "Aster first attempts the real ChatGPT Send button and falls back to Enter if needed.",
                details={"attempt_index": attempt_index, "strategy": "button_or_enter"},
            )
            if not self._click_named_button(target, "send prompt"):
                pyautogui.press("enter")
        elif strategy == 1:
            self._log("send_attempt", {"attempt_index": attempt_index, "strategy": "button_or_ctrl_enter"})
            self._activity(
                "browser_send_attempt",
                "Trying an alternate send shortcut.",
                "Some browser states accept Ctrl+Enter more reliably than a simple Enter keypress.",
                details={"attempt_index": attempt_index, "strategy": "button_or_ctrl_enter"},
            )
            if not self._click_named_button(target, "send prompt"):
                pyautogui.hotkey("ctrl", "enter")
        elif strategy == 2:
            self._log("send_attempt", {"attempt_index": attempt_index, "strategy": "button_click"})
            self._activity(
                "browser_send_attempt",
                "Trying a direct Send button click.",
                "This targets the visible send control when keyboard shortcuts do not advance the page.",
                details={"attempt_index": attempt_index, "strategy": "button_click"},
            )
            self._click_send_button(target)
        elif strategy == 3:
            self._log("send_attempt", {"attempt_index": attempt_index, "strategy": "tab_then_enter"})
            self._activity(
                "browser_send_attempt",
                "Trying keyboard focus navigation to the Send button.",
                "If the button exists but is not focused, tab navigation can still trigger it.",
                details={"attempt_index": attempt_index, "strategy": "tab_then_enter"},
            )
            pyautogui.press("tab", presses=6, interval=0.08)
            pyautogui.press("enter")
        else:
            self._log("send_attempt", {"attempt_index": attempt_index, "strategy": "button_click_then_enter"})
            self._activity(
                "browser_send_attempt",
                "Trying a combined click-plus-Enter send fallback.",
                "This is a last-resort browser send strategy when earlier attempts did not move the page.",
                details={"attempt_index": attempt_index, "strategy": "button_click_then_enter"},
            )
            self._click_send_button(target)
            time.sleep(0.4)
            pyautogui.press("enter")
        self._capture_visual_action_stage(
            context,
            target=target,
            stage="post",
            ui_state=self._ui_state(target),
            action_outcome=f"send_strategy_{strategy}_completed",
            confidence_after=hint.confidence if hint is not None else None,
        )

    def _wait_for_reply_start(self, target, before_lines, seconds: float) -> bool:
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            self._tick_runtime_log_heartbeat("wait_for_reply_start")
            image = self._capture.capture_region(target.left, target.top, target.width, target.height)
            lines = self._ocr.extract(image)
            ui_state = self._ui_state(target)
            if self._reply_started(before_lines, lines, target, ui_state=ui_state, send_attempted=True):
                return True
            time.sleep(0.6)
        return False

    @staticmethod
    def _find_composer_line(lines):
        hints = ("ask anything", "message chatgpt", "type a message", "send a message")
        for hint in hints:
            for line in lines:
                if hint in line.text.lower():
                    return line
        return None

    @staticmethod
    def _find_line(lines, phrase: str):
        phrase_lower = phrase.lower()
        for line in lines:
            if phrase_lower in line.text.lower():
                return line
        return None

    @staticmethod
    def _invoke_button(ctrl) -> bool:
        try:
            ctrl.invoke()
            return True
        except Exception:
            try:
                ctrl.iface_invoke.Invoke()
                return True
            except Exception:
                return False


    def _ensure_chatgpt_window(self, chatgpt_url: str, launch_timeout_sec: float):
        self._log("chatgpt_window_missing_opening_browser", {"url": chatgpt_url})
        self._activity(
            "browser_window_open",
            "Opening a fresh ChatGPT browser window now.",
            "Reusing existing Chrome tabs has been unreliable, so Aster now starts from a fresh ChatGPT page.",
            details={"url": chatgpt_url},
        )
        webbrowser.open(chatgpt_url, new=2)
        deadline = time.monotonic() + max(5.0, launch_timeout_sec)
        last_error = None
        while time.monotonic() < deadline:
            self._tick_runtime_log_heartbeat("ensure_chatgpt_window")
            try:
                target = self._executor.find_chatgpt_browser_window()
                self._log("chatgpt_window_found_after_launch", {"title": target.title, "handle": target.handle})
                self._activity(
                    "browser_window_found",
                    "The newly opened ChatGPT browser window is ready.",
                    "Aster will only continue from a freshly opened ChatGPT page for browser reliability.",
                    status="success",
                    details={"window_title": target.title},
                )
                return target
            except Exception as exc:
                last_error = exc
                time.sleep(1.5)
        raise RuntimeError(
            "ChatGPT browser window was not found after launch. "
            "Open ChatGPT, sign in if needed, then try again."
        ) from last_error

    def _prepare_chatgpt_window(self, target, chatgpt_url: str, timeout_sec: float) -> bool:
        ready = self._wait_for_chatgpt_ready(target, timeout_sec=min(timeout_sec, 12.0))
        if not ready:
            def recovery_action(action_name: str, target_intent: str, callback):
                return lambda: self._run_visual_recovery_action(
                    target=target,
                    action_name=action_name,
                    target_intent=target_intent,
                    callback=callback,
                )

            recovery_handlers = build_recovery_handlers(
                rescan=recovery_action("rescan", "reply_region", lambda: None),
                refocus_composer=recovery_action("refocus_composer", "composer_area", lambda: self._focus_composer(target, [])),
                # Dedicated thread reopen/reattach is still pending, so conservative recovery
                # currently falls back to direct ChatGPT navigation inside the same window.
                reopen_thread=recovery_action(
                    "reopen_thread",
                    "new_chat_button",
                    lambda: self._navigate_browser_to_chatgpt(target, chatgpt_url),
                ),
                reload_page=recovery_action(
                    "reload_page",
                    "reply_region",
                    lambda: self._navigate_browser_to_chatgpt(target, chatgpt_url),
                ),
                reopen_chatgpt=recovery_action(
                    "reopen_chatgpt",
                    "reply_region",
                    lambda: self._navigate_browser_to_chatgpt(target, chatgpt_url),
                ),
            )
            recovery = decide_and_execute_recovery(
                self._last_page_classification,
                attempts_used=0,
                max_attempts=self.max_recovery_attempts,
                handlers=recovery_handlers,
            )
            self._log("recovery_decision", decision_payload(recovery))
            if recovery.action in {RecoveryAction.RELOAD_PAGE, RecoveryAction.REOPEN_CHATGPT}:
                self._activity(
                    "browser_window_reset",
                    "Resetting the attached browser tab to a fresh ChatGPT page.",
                    "The initial browser window did not settle into a usable ChatGPT state, so Aster is forcing a direct navigation retry.",
                    details={"url": chatgpt_url, "recovery_action": recovery.action.value},
                )
            ready = self._wait_for_chatgpt_ready(target, timeout_sec=timeout_sec)
        image = self._capture.capture_region(target.left, target.top, target.width, target.height)
        lines = self._ocr.extract(image)
        final_ui_state = self._ui_state(target, include_uia_diagnostics=not ready)
        analysis = self._analyze_screen(target, lines=lines, ui_state=final_ui_state)
        failure = self._page_readiness_failure_details(analysis)
        if failure["code"] == "wrong_page_detected":
            self._log("wrong_page_detected", self._page_readiness_log_payload(analysis, failure))
            raise RuntimeError(self._page_readiness_error_message(failure))
        if ready:
            return self._ensure_fresh_chat_thread(target, chatgpt_url=chatgpt_url, timeout_sec=min(10.0, timeout_sec))
        self._log("chatgpt_page_not_ready", self._page_readiness_log_payload(analysis, failure))
        raise RuntimeError(self._page_readiness_error_message(failure))

    def _navigate_browser_to_chatgpt(self, target, chatgpt_url: str) -> None:
        self._focus_window(target)
        pyautogui.hotkey("ctrl", "l")
        time.sleep(0.15)
        self._paste_prompt_via_clipboard(target, chatgpt_url, click_point=None)
        time.sleep(0.15)
        pyautogui.press("enter")
        time.sleep(0.8)

    def _ensure_fresh_chat_thread(self, target, chatgpt_url: str, timeout_sec: float) -> bool:
        title = str(getattr(target, "title", "") or "")
        if not self._window_title_suggests_existing_thread(title):
            self._reset_to_new_chat_if_possible(target, timeout_sec=timeout_sec)
            return False
        self._log("stale_thread_title_detected", {"title": title})
        self._activity(
            "browser_stale_thread",
            "Aster detected an existing ChatGPT thread title and is resetting to a fresh chat.",
            "Old thread context and old code snippets have been contaminating reply capture.",
            details={"window_title": title},
        )
        if self._reset_to_new_chat_if_possible(target, timeout_sec=timeout_sec):
            return False
        self._log("stale_thread_reset_via_navigation", {"title": title, "url": chatgpt_url})
        self._navigate_browser_to_chatgpt(target, chatgpt_url)
        self._wait_for_chatgpt_ready(target, timeout_sec=max(3.0, timeout_sec))
        return False

    def _reset_to_new_chat_if_possible(self, target, timeout_sec: float) -> bool:
        clicked = self._click_named_button(target, "new chat")
        if not clicked:
            clicked = self._click_named_control(
                target,
                "new chat",
                control_types=("Button", "Hyperlink", "ListItem", "Text", "MenuItem"),
            )
        if not clicked:
            return False
        self._log("new_chat_clicked", {"title": getattr(target, "title", "")})
        self._activity(
            "browser_new_chat",
            "Resetting the ChatGPT thread to a new chat before inserting the prompt.",
            "Aster gets more reliable structured replies from a fresh thread than from a reused conversation title.",
        )
        self._wait_for_chatgpt_ready(target, timeout_sec=max(3.0, timeout_sec))
        return True

    def _wait_for_chatgpt_ready(self, target, timeout_sec: float) -> bool:
        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline:
            self._tick_runtime_log_heartbeat("wait_for_chatgpt_ready")
            image = self._capture.capture_region(target.left, target.top, target.width, target.height)
            lines = self._ocr.extract(image)
            analysis = self._analyze_screen(target, lines=lines)
            if (
                analysis.looks_like_chatgpt
                and analysis.ready_score >= PAGE_READINESS_SCORE_THRESHOLD
                and not analysis.loading_detected
            ):
                self._log(
                    "window_ready_confirmed",
                    {
                        "ui_state": analysis.ui_state,
                        "ocr_preview": list(analysis.ocr_preview),
                        "page_classification": self._screen_analysis_payload(analysis),
                    },
                )
                return True
            time.sleep(0.6)
        timeout_ui_state = self._ui_state(target, include_uia_diagnostics=True)
        timeout_analysis = self._analyze_screen(target, ui_state=timeout_ui_state)
        failure = self._page_readiness_failure_details(timeout_analysis)
        self._log("window_ready_timeout", self._page_readiness_log_payload(timeout_analysis, failure))
        return False

    @staticmethod
    def _looks_like_chatgpt_page(lines, ui_state: dict[str, Any]) -> bool:
        return looks_like_chatgpt_page(lines, ui_state)

    def _capture_reply_text(
        self,
        target,
        before_lines,
        prompt: str,
        timeout_sec: float,
        *,
        prompt_anchor=None,
        thread_reused_for_capture: bool = False,
        retry_attempt: bool = False,
    ) -> str:
        started_at = time.monotonic()
        deadline = time.monotonic() + timeout_sec
        self._last_reply_region_visual_evidence = {}
        self._last_scrolled_capture_meta = {}
        self._last_retry_fragment_source_meta = self._retry_source_preference_defaults()
        best_text = ""
        best_source = ""
        best_score = float("-inf")
        last_structured = ""
        stable_structured_hits = 0
        attempt_index = 0
        scanned_with_scroll = False
        previous_candidate_text = ""
        previous_ocr_text = ""
        previous_uia_text = ""
        last_wait_diagnostics: dict[str, Any] | None = None
        last_ui_state: dict[str, Any] = {}
        snapshot = _ReplyCaptureSnapshot()
        structured_observation_counts: dict[str, int] = {}

        while time.monotonic() < deadline:
            self._tick_runtime_log_heartbeat("capture_reply_text")
            time.sleep(2.0 if attempt_index else 1.2)
            ui_state = self._ui_state(target)
            last_ui_state = ui_state
            image = self._capture.capture_region(target.left, target.top, target.width, target.height)
            lines = self._ocr.extract(image)
            self._raise_for_browser_error(lines, ui_state, stage="reply_capture")
            ocr_text, uia_text = self._capture_visible_reply_sources(target, before_lines, prompt, lines=lines)
            if retry_attempt:
                current_best, self._last_retry_fragment_source_meta = self._choose_retry_attempt_candidate_sources(
                    ocr_text=ocr_text,
                    uia_text=uia_text,
                    prompt_anchor=prompt_anchor,
                )
            else:
                self._last_retry_fragment_source_meta = self._retry_source_preference_defaults()
                current_best = choose_best_reply_candidate_for_policy(
                    {
                        "ocr": ocr_text,
                        "uia": uia_text,
                    },
                    policy=REPLY_TRACKER_POLICY,
                    prompt_anchor=prompt_anchor,
                )
            current_stable_hits = stable_structured_hits
            if current_best is not None and looks_like_patch_plan_json(current_best[1]):
                current_stable_hits = stable_structured_hits + 1 if current_best[1] == last_structured else 0
            acceptance = evaluate_reply_acceptance(
                current_best,
                ui_state=ui_state,
                policy=REPLY_TRACKER_POLICY,
                prompt_anchor=prompt_anchor,
                require_anchor=thread_reused_for_capture,
                stable_structured_hits=current_stable_hits,
                scrolling_attempted=scanned_with_scroll,
            )
            diagnostics = summarize_reply_wait_iteration(
                elapsed_sec=time.monotonic() - started_at,
                ui_state=ui_state,
                ocr_text=ocr_text,
                uia_text=uia_text,
                current_candidate=current_best,
                previous_best_text=previous_candidate_text,
                previous_ocr_text=previous_ocr_text,
                previous_uia_text=previous_uia_text,
                stable_structured_hits=current_stable_hits,
                scrolling_attempted=scanned_with_scroll,
            )
            diagnostics.update(self._reply_acceptance_log_payload(acceptance))
            self._log("reply_wait_heartbeat", diagnostics)
            last_wait_diagnostics = diagnostics
            if current_best is not None:
                source, candidate, score = current_best
                retry_assessment = (
                    self._assess_retry_attempt_response(candidate, candidate_source=source)
                    if retry_attempt
                    else None
                )
                region_trusted, region_reason = self._reply_region_candidate_trusted()
                self._remember_reply_capture_candidate(
                    snapshot,
                    source=source,
                    text=candidate,
                    score=score,
                    salvage_allowed=source in {"uia", "ocr"} and region_trusted,
                    region_trusted=region_trusted,
                    region_confidence=self._last_reply_region_visual_evidence.get("visual_region_confidence"),
                    region_reason=region_reason,
                    observation_counts=structured_observation_counts,
                )
                preferred = select_preferred_reply_candidate(
                    (source, best_text, best_score) if best_text else None,
                    current_best,
                    extract_structured_block=extract_structured_block,
                    is_patch_json=looks_like_patch_plan_json,
                )
                if preferred == current_best and (candidate != best_text or score != best_score):
                    best_text = candidate
                    best_source = source
                    best_score = score
                    self._log(
                        "reply_candidate_selected",
                        {
                            "source": source,
                            "score": score,
                            "length": len(candidate),
                            "preview": candidate[:240],
                        },
                    )
                if looks_like_patch_plan_json(candidate):
                    if candidate == last_structured:
                        stable_structured_hits += 1
                    else:
                        last_structured = candidate
                        stable_structured_hits = 0
                if retry_assessment is not None and retry_assessment["selected_text"]:
                    self._log(
                        "reply_candidate_accepted",
                        {
                            "source": source,
                            "score": score,
                            "length": len(str(retry_assessment["selected_text"])),
                            "accepted": True,
                            "acceptance_tier": retry_assessment["acceptance_tier"],
                            "acceptance_reason": "Retry-attempt capture isolated a machine-parseable structured block.",
                            "rejection_reason": "",
                            "requires_more_observation": False,
                            "should_scroll": False,
                            "retry_attempt_capture_mode": "structured_block_first",
                            "retry_attempt_structured_block_found": retry_assessment["structured_block_found"],
                            "retry_attempt_parseable": retry_assessment["parseable"],
                            "retry_attempt_prose_contamination": retry_assessment["prose_contamination"],
                            "retry_attempt_wrapper_only": retry_assessment["wrapper_only"],
                            "retry_attempt_block_count": retry_assessment["block_count"],
                            "retry_attempt_exact_block_only": retry_assessment["exact_block_only"],
                            "retry_attempt_extra_text_detected": retry_assessment["extra_text_detected"],
                            "retry_attempt_json_object_count": retry_assessment["json_object_count"],
                            "retry_attempt_failure_reason": retry_assessment["failure_reason"],
                        },
                    )
                    self._last_reply_capture_snapshot = snapshot
                    return str(retry_assessment["selected_text"])
                if acceptance.accepted and not retry_attempt:
                    self._log(
                        "reply_candidate_accepted",
                        {
                            "source": source,
                            "score": score,
                            "length": len(candidate),
                            **self._reply_acceptance_log_payload(acceptance),
                        },
                    )
                    self._last_reply_capture_snapshot = snapshot
                    return candidate
                if acceptance.should_scroll and not scanned_with_scroll and not self._response_still_streaming(ui_state):
                    self._log(
                        "reply_wait_scroll_requested",
                        {
                            **summarize_reply_wait_iteration(
                                elapsed_sec=time.monotonic() - started_at,
                                ui_state=ui_state,
                                ocr_text=ocr_text,
                                uia_text=uia_text,
                                current_candidate=current_best,
                                previous_best_text=previous_candidate_text,
                                previous_ocr_text=previous_ocr_text,
                                previous_uia_text=previous_uia_text,
                                stable_structured_hits=stable_structured_hits,
                                scrolling_attempted=True,
                            ),
                            **self._reply_acceptance_log_payload(acceptance),
                        },
                    )
                    scrolled = self._capture_reply_text_by_scrolling(
                        target,
                        before_lines,
                        prompt,
                        max_steps=4 if retry_attempt else 10,
                        structured_anchor_text=candidate if acceptance.acceptance_tier == "partial_structured_reply" else "",
                    )
                    if scrolled.strip():
                        scrolled_progress = structured_completion_progress(scrolled)
                        best_from_scroll = choose_best_reply_candidate_for_policy(
                            {"scrolled": scrolled},
                            policy=REPLY_TRACKER_POLICY,
                            prompt_anchor=prompt_anchor,
                        )
                        if best_from_scroll is not None:
                            _, candidate, score = best_from_scroll
                            retry_scroll_assessment = (
                                self._assess_retry_attempt_response(candidate, candidate_source="scrolled")
                                if retry_attempt
                                else None
                            )
                            scroll_stable_hits = stable_structured_hits
                            if looks_like_patch_plan_json(candidate):
                                scroll_stable_hits = stable_structured_hits + 1 if candidate == last_structured else 0
                            scroll_acceptance = evaluate_reply_acceptance(
                                best_from_scroll,
                                ui_state=ui_state,
                                policy=REPLY_TRACKER_POLICY,
                                prompt_anchor=prompt_anchor,
                                require_anchor=thread_reused_for_capture,
                                stable_structured_hits=scroll_stable_hits,
                                scrolling_attempted=True,
                            )
                            scroll_candidate = self._remember_reply_capture_candidate(
                                snapshot,
                                source="scrolled",
                                text=candidate,
                                score=score,
                                salvage_allowed=scroll_acceptance.accepted,
                                region_trusted=bool(
                                    self._last_scrolled_capture_meta.get("region_trusted", scroll_acceptance.accepted)
                                ),
                                region_confidence=self._last_scrolled_capture_meta.get("visual_region_confidence"),
                                region_reason=str(self._last_scrolled_capture_meta.get("region_reason", "")),
                                observation_counts=structured_observation_counts,
                            )
                            self._log(
                                "reply_scrolled_capture",
                                {
                                    "length": len(scrolled),
                                    "preview": scrolled[:240],
                                    "structured_completion_score": structured_completion_score(scrolled_progress),
                                    "structured_completion_progress": scrolled_progress,
                                    "best_structured_candidate_source": (
                                        snapshot.best_structured.source if snapshot.best_structured is not None else None
                                    ),
                                    "best_structured_candidate_length": (
                                        len(snapshot.best_structured.parsed_text) if snapshot.best_structured is not None else 0
                                    ),
                                    "best_structured_candidate_parseable": (
                                        snapshot.best_structured.parseable if snapshot.best_structured is not None else False
                                    ),
                                    "best_structured_candidate_salvage_allowed": (
                                        snapshot.best_salvageable is not None
                                    ),
                                    "scrolled_region_trusted": self._last_scrolled_capture_meta.get("region_trusted"),
                                    "scrolled_region_reason": self._last_scrolled_capture_meta.get("region_reason"),
                                    "scrolled_region_confidence": self._last_scrolled_capture_meta.get("visual_region_confidence"),
                                    "retry_attempt_capture_mode": "structured_block_first" if retry_attempt else "",
                                    "retry_attempt_structured_block_found": (
                                        retry_scroll_assessment["structured_block_found"]
                                        if retry_scroll_assessment is not None
                                        else False
                                    ),
                                    "retry_attempt_parseable": (
                                        retry_scroll_assessment["parseable"] if retry_scroll_assessment is not None else False
                                    ),
                                    "retry_attempt_prose_contamination": (
                                        retry_scroll_assessment["prose_contamination"]
                                        if retry_scroll_assessment is not None
                                        else False
                                    ),
                                    "retry_attempt_wrapper_only": (
                                        retry_scroll_assessment["wrapper_only"]
                                        if retry_scroll_assessment is not None
                                        else False
                                    ),
                                    "retry_attempt_block_count": (
                                        retry_scroll_assessment["block_count"]
                                        if retry_scroll_assessment is not None
                                        else 0
                                    ),
                                    "retry_attempt_exact_block_only": (
                                        retry_scroll_assessment["exact_block_only"]
                                        if retry_scroll_assessment is not None
                                        else False
                                    ),
                                    "retry_attempt_extra_text_detected": (
                                        retry_scroll_assessment["extra_text_detected"]
                                        if retry_scroll_assessment is not None
                                        else False
                                    ),
                                    "retry_attempt_json_object_count": (
                                        retry_scroll_assessment["json_object_count"]
                                        if retry_scroll_assessment is not None
                                        else 0
                                    ),
                                    "retry_attempt_failure_reason": (
                                        retry_scroll_assessment["failure_reason"]
                                        if retry_scroll_assessment is not None
                                        else ""
                                    ),
                                    **self._reply_acceptance_log_payload(scroll_acceptance),
                                },
                            )
                            preferred = select_preferred_reply_candidate(
                                (best_source or "best", best_text, best_score) if best_text else None,
                                best_from_scroll,
                                extract_structured_block=extract_structured_block,
                                is_patch_json=looks_like_patch_plan_json,
                            )
                            if preferred == best_from_scroll and (candidate != best_text or score != best_score):
                                best_text = candidate
                                best_source = "scrolled"
                                best_score = score
                            if looks_like_patch_plan_json(candidate):
                                if candidate == last_structured:
                                    stable_structured_hits += 1
                                else:
                                    last_structured = candidate
                                    stable_structured_hits = 0
                            if retry_scroll_assessment is not None and retry_scroll_assessment["selected_text"]:
                                self._log(
                                    "reply_candidate_accepted",
                                    {
                                        "source": "scrolled",
                                        "score": score,
                                        "length": len(str(retry_scroll_assessment["selected_text"])),
                                        "accepted": True,
                                        "acceptance_tier": retry_scroll_assessment["acceptance_tier"],
                                        "acceptance_reason": "Retry-attempt scroll capture isolated a machine-parseable structured block.",
                                        "rejection_reason": "",
                                        "requires_more_observation": False,
                                        "should_scroll": False,
                                        "retry_attempt_capture_mode": "structured_block_first",
                                        "retry_attempt_structured_block_found": retry_scroll_assessment["structured_block_found"],
                                        "retry_attempt_parseable": retry_scroll_assessment["parseable"],
                                        "retry_attempt_prose_contamination": retry_scroll_assessment["prose_contamination"],
                                        "retry_attempt_wrapper_only": retry_scroll_assessment["wrapper_only"],
                                        "retry_attempt_block_count": retry_scroll_assessment["block_count"],
                                        "retry_attempt_exact_block_only": retry_scroll_assessment["exact_block_only"],
                                        "retry_attempt_extra_text_detected": retry_scroll_assessment["extra_text_detected"],
                                        "retry_attempt_json_object_count": retry_scroll_assessment["json_object_count"],
                                        "retry_attempt_failure_reason": retry_scroll_assessment["failure_reason"],
                                    },
                                )
                                self._last_reply_capture_snapshot = snapshot
                                return str(retry_scroll_assessment["selected_text"])
                            if scroll_acceptance.accepted and not retry_attempt:
                                self._log(
                                    "reply_candidate_accepted",
                                    {
                                        "source": "scrolled",
                                        "score": score,
                                        "length": len(candidate),
                                        **self._reply_acceptance_log_payload(scroll_acceptance),
                                    },
                                )
                                self._last_reply_capture_snapshot = snapshot
                                return candidate
                    scanned_with_scroll = True
            previous_candidate_text = current_best[1] if current_best is not None else ""
            previous_ocr_text = ocr_text
            previous_uia_text = uia_text
            attempt_index += 1

        timeout_candidate = (best_source or "best", best_text, best_score) if best_text else None
        if timeout_candidate is not None:
            region_trusted, region_reason = self._reply_region_candidate_trusted()
            self._remember_reply_capture_candidate(
                snapshot,
                source=str(timeout_candidate[0]),
                text=str(timeout_candidate[1]),
                score=float(timeout_candidate[2]),
                salvage_allowed=str(timeout_candidate[0]) in {"uia", "ocr"} and region_trusted,
                region_trusted=region_trusted,
                region_confidence=self._last_reply_region_visual_evidence.get("visual_region_confidence"),
                region_reason=region_reason,
                observation_counts=structured_observation_counts,
            )
        timeout_acceptance = evaluate_reply_acceptance(
            timeout_candidate,
            ui_state=last_ui_state,
            policy=REPLY_TRACKER_POLICY,
            prompt_anchor=prompt_anchor,
            require_anchor=thread_reused_for_capture,
            stable_structured_hits=stable_structured_hits,
            scrolling_attempted=scanned_with_scroll,
            timed_out=True,
        )
        timeout_retry_assessment = (
            self._assess_retry_attempt_response(str(timeout_candidate[1]), candidate_source=str(timeout_candidate[0]))
            if retry_attempt and timeout_candidate is not None
            else None
        )
        if last_wait_diagnostics is not None:
            timeout_payload = {
                **last_wait_diagnostics,
                **self._reply_acceptance_log_payload(timeout_acceptance),
                "elapsed_sec": round(time.monotonic() - started_at, 1),
                "best_overall_source": best_source or None,
                "best_overall_score": round(best_score, 1) if best_text else None,
                "best_overall_length": len(best_text),
            }
            if timeout_retry_assessment is not None:
                timeout_payload.update(
                    {
                        "retry_attempt_capture_mode": "structured_block_first",
                        "retry_attempt_acceptance_tier": timeout_retry_assessment["acceptance_tier"],
                        "retry_attempt_failure_reason": timeout_retry_assessment["failure_reason"],
                        "retry_attempt_structured_block_found": timeout_retry_assessment["structured_block_found"],
                        "retry_attempt_parseable": timeout_retry_assessment["parseable"],
                        "retry_attempt_prose_contamination": timeout_retry_assessment["prose_contamination"],
                        "retry_attempt_wrapper_only": timeout_retry_assessment["wrapper_only"],
                        "retry_attempt_block_count": timeout_retry_assessment["block_count"],
                        "retry_attempt_exact_block_only": timeout_retry_assessment["exact_block_only"],
                        "retry_attempt_extra_text_detected": timeout_retry_assessment["extra_text_detected"],
                        "retry_attempt_json_object_count": timeout_retry_assessment["json_object_count"],
                    }
                )
            self._log(
                "reply_wait_timeout",
                timeout_payload,
            )
        if timeout_candidate is not None and timeout_acceptance.accepted and not retry_attempt:
            self._last_reply_capture_snapshot = snapshot
            return best_text

        fallback = self._executor.read_chatgpt_browser_reply(
            target,
            self._capture,
            self._ocr,
            before_lines,
            prompt,
            timeout_sec=min(8.0, max(2.0, timeout_sec / 4.0)),
        )
        fallback_block = extract_structured_block(fallback)
        fallback_candidate = None
        if looks_like_patch_plan_json(fallback_block):
            fallback_candidate = (
                "screen_reader_fallback",
                fallback_block,
                score_candidate_for_policy(fallback_block, policy=REPLY_TRACKER_POLICY),
            )
        fallback_acceptance = evaluate_reply_acceptance(
            fallback_candidate,
            ui_state=last_ui_state,
            policy=REPLY_TRACKER_POLICY,
            prompt_anchor=prompt_anchor,
            require_anchor=thread_reused_for_capture,
            stable_structured_hits=stable_structured_hits,
            scrolling_attempted=scanned_with_scroll,
            timed_out=True,
        )
        retry_fallback_assessment = (
            self._assess_retry_attempt_response(fallback, candidate_source="screen_reader_fallback")
            if retry_attempt
            else None
        )
        if fallback_block.strip():
            region_trusted, region_reason = self._reply_region_candidate_trusted()
            self._remember_reply_capture_candidate(
                snapshot,
                source="screen_reader_fallback",
                text=fallback,
                score=fallback_candidate[2] if fallback_candidate is not None else score_candidate_for_policy(fallback_block, policy=REPLY_TRACKER_POLICY),
                salvage_allowed=fallback_candidate is not None and fallback_acceptance.accepted and region_trusted,
                region_trusted=region_trusted,
                region_confidence=self._last_reply_region_visual_evidence.get("visual_region_confidence"),
                region_reason=region_reason,
                observation_counts=structured_observation_counts,
            )
        if retry_fallback_assessment is not None and retry_fallback_assessment["selected_text"]:
            self._log(
                "reply_candidate_selected",
                {
                    "source": "screen_reader_fallback",
                    "score": fallback_candidate[2] if fallback_candidate is not None else 0.0,
                    "length": len(str(retry_fallback_assessment["selected_text"])),
                    "preview": str(retry_fallback_assessment["selected_text"])[:240],
                    "accepted": True,
                    "acceptance_tier": retry_fallback_assessment["acceptance_tier"],
                    "acceptance_reason": "Retry-attempt fallback isolated a machine-parseable structured block.",
                    "rejection_reason": "",
                    "requires_more_observation": False,
                    "should_scroll": False,
                    "retry_attempt_capture_mode": "structured_block_first",
                    "retry_attempt_structured_block_found": retry_fallback_assessment["structured_block_found"],
                    "retry_attempt_parseable": retry_fallback_assessment["parseable"],
                    "retry_attempt_prose_contamination": retry_fallback_assessment["prose_contamination"],
                    "retry_attempt_wrapper_only": retry_fallback_assessment["wrapper_only"],
                    "retry_attempt_block_count": retry_fallback_assessment["block_count"],
                    "retry_attempt_exact_block_only": retry_fallback_assessment["exact_block_only"],
                    "retry_attempt_extra_text_detected": retry_fallback_assessment["extra_text_detected"],
                    "retry_attempt_json_object_count": retry_fallback_assessment["json_object_count"],
                    "retry_attempt_failure_reason": retry_fallback_assessment["failure_reason"],
                },
            )
            self._last_reply_capture_snapshot = snapshot
            return str(retry_fallback_assessment["selected_text"])
        if fallback_candidate is not None and fallback_acceptance.accepted and not retry_attempt:
            self._log(
                "reply_candidate_selected",
                {
                    "source": "screen_reader_fallback",
                    "score": fallback_candidate[2],
                    "length": len(fallback_block),
                    "preview": fallback_block[:240],
                    **self._reply_acceptance_log_payload(fallback_acceptance),
                },
            )
            self._last_reply_capture_snapshot = snapshot
            return fallback_block
        self._last_reply_capture_snapshot = snapshot
        return ""

    def _capture_visible_reply_sources(self, target, before_lines, prompt: str, lines=None) -> tuple[str, str]:
        after_lines = lines
        if after_lines is None:
            image = self._capture.capture_region(target.left, target.top, target.width, target.height)
            after_lines = self._ocr.extract(image)
        ui_state = self._ui_state(target)
        checkpoint_context = None
        hint = self._visual_memory_hint(
            intent="reply_region",
            target=target,
            ui_state=ui_state,
            lines=after_lines,
            log_events=False,
        )
        hint_rect = None
        if self._visual_action_debug is not None and self._visual_action_debug.should_capture_checkpoint("reply_capture_region"):
            hint_rect = self._hint_rect(hint, target)
            if hint is None and self._visual_action_debug.enabled:
                self._log("visual_memory_miss", {"intent": "reply_region", "window_title": str(getattr(target, "title", ""))})
            elif hint is not None:
                self._log(
                    "visual_memory_hit",
                    {
                        "intent": "reply_region",
                        "confidence": round(hint.confidence, 3),
                        "score": round(hint.score, 3),
                        "title_match": hint.title_match,
                        "page_state_match": hint.page_state_match,
                    },
                )
            checkpoint_context = self._begin_visual_action(
                target=target,
                action_type="reply_capture_region",
                target_intent="reply_region",
                ui_state=ui_state,
                lines=after_lines,
                target_rect=hint_rect or self._default_reply_region_rect(target),
                candidate_rects=[rect for rect in [hint_rect, self._default_reply_region_rect(target)] if rect is not None],
                confidence_before=hint.confidence if hint is not None else None,
                used_remembered_region=hint is not None,
            )
        ocr_text = extract_reply_from_ocr_lines_for_policy(
            before_lines,
            after_lines,
            target_width=target.width,
            target_height=target.height,
            prompt=prompt,
            policy=REPLY_TRACKER_POLICY,
        )
        self._capture_visual_action_stage(
            checkpoint_context,
            target=target,
            stage="during",
            ui_state=ui_state,
            lines=after_lines,
            action_outcome=f"reply_capture_ocr:{len(ocr_text)}",
            confidence_after=hint.confidence if hint is not None else None,
        )
        uia_text = self._read_visible_reply_text(target)
        merged_length = len(merge_reply_segment_sources(uia_text, ocr_text, policy=REPLY_TRACKER_POLICY))
        self._last_reply_region_visual_evidence = self._build_reply_region_visual_evidence(
            hint=hint,
            merged_length=merged_length,
        )
        if checkpoint_context is not None:
            target_rect = hint_rect or self._default_reply_region_rect(target)
            if merged_length > 0:
                self._visual_memory_update(
                    intent="reply_region",
                    target=target,
                    action_type="reply_capture_region",
                    rect=target_rect,
                    ui_state=ui_state,
                    lines=after_lines,
                )
            self._capture_visual_action_stage(
                checkpoint_context,
                target=target,
                stage="post",
                ui_state=self._ui_state(target),
                lines=after_lines,
                action_outcome=f"reply_capture_visible:{merged_length}",
                confidence_after=hint.confidence if hint is not None else None,
                updated_remembered_region=merged_length > 0,
            )
        return ocr_text, uia_text

    def _capture_reply_text_by_scrolling(
        self,
        target,
        before_lines,
        prompt: str,
        max_steps: int,
        *,
        structured_anchor_text: str = "",
    ) -> str:
        self._activity(
            "browser_scroll_read",
            "Scrolling through the ChatGPT thread to collect the full reply.",
            "Long patch plans may span multiple viewports, so Aster needs to read more than the currently visible slice.",
        )
        self._scroll_reply_to_bottom(target)
        segments: list[str] = []
        no_progress_steps = 0
        merged = structured_anchor_text.strip()
        trusted_lineage = merged
        step = 0
        step_limit = max_steps
        continuation_windows_used = 0
        recent_completion_progress_steps = 0
        drift_containment_active = False
        self._last_scrolled_capture_meta = {}

        while step < step_limit:
            self._tick_runtime_log_heartbeat("capture_reply_text_by_scrolling")
            segment = self._capture_visible_reply_segment(target, before_lines, prompt)
            assessment = assess_scrolled_segment_addition(
                merged,
                segment,
                policy=REPLY_TRACKER_POLICY,
                seed_text=structured_anchor_text or merged,
                trusted_lineage_text=trusted_lineage,
                drift_containment_active=drift_containment_active,
                visual_region_evidence=self._last_reply_region_visual_evidence,
            )
            self._last_scrolled_capture_meta = {
                "region_trusted": bool(
                    assessment["trusted_lineage_score"] >= 10
                    and not assessment["drift_detected"]
                    and not assessment["matched_current_blob_only"]
                    and (
                        assessment["matched_seed_lineage"]
                        or assessment["matched_trusted_lineage"]
                        or assessment["contextual_structured_extension"]
                    )
                ),
                "visual_region_confidence": assessment["visual_region_confidence"],
                "region_reason": (
                    assessment["contextual_extension_allowed_reason"]
                    or assessment["contextual_extension_denied_reason"]
                    or ",".join(str(reason) for reason in assessment["trusted_lineage_reasons"][:2])
                ),
            }
            if assessment["contributed"]:
                no_progress_steps = 0
                if assessment["meaningful_completion_progress"]:
                    recent_completion_progress_steps += 1
                else:
                    recent_completion_progress_steps = max(0, recent_completion_progress_steps - 1)
                segments.append(str(assessment["segment_text"]))
                merged = str(assessment["merged_text"])
                if assessment["trusted_lineage_extended"]:
                    trusted_lineage = merged
                drift_containment_active = False
                self._log(
                    "reply_scroll_segment",
                    {
                        "step": step,
                        "length": len(str(assessment["segment_text"])),
                        "preview": str(assessment["segment_text"])[:220],
                        "merged_length": len(merged),
                        "novelty_count": assessment["novelty_count"],
                        "growth_chars": assessment["growth_chars"],
                        "completion_progressed": assessment["completion_progressed"],
                        "meaningful_completion_progress": assessment["meaningful_completion_progress"],
                        "completion_score_delta": assessment["completion_score_delta"],
                        "completion_score": assessment["after_progress"]["completion_score"],
                        "region_integrity_score": assessment["region_integrity_score"],
                        "region_reasons": assessment["region_reasons"],
                        "region_consistent": assessment["region_consistent"],
                        "trusted_lineage_score": assessment["trusted_lineage_score"],
                        "trusted_lineage_reasons": assessment["trusted_lineage_reasons"],
                        "trusted_lineage_extended": assessment["trusted_lineage_extended"],
                        "matched_seed_lineage": assessment["matched_seed_lineage"],
                        "matched_trusted_lineage": assessment["matched_trusted_lineage"],
                        "contextual_structured_extension": assessment["contextual_structured_extension"],
                        "contextual_extension_allowed_reason": assessment["contextual_extension_allowed_reason"],
                        "contextual_extension_denied_reason": assessment["contextual_extension_denied_reason"],
                        "matched_current_blob_only": assessment["matched_current_blob_only"],
                        "continuity_against_seed": assessment["continuity_against_seed"],
                        "continuity_against_trusted_lineage": assessment["continuity_against_trusted_lineage"],
                        "continuity_against_current": assessment["continuity_against_current"],
                        "visual_region_confidence": assessment["visual_region_confidence"],
                        "visual_region_used_remembered_region": assessment["visual_region_used_remembered_region"],
                        "visual_region_confirmed_reply_region": assessment["visual_region_confirmed_reply_region"],
                        "visual_region_supports_extension": assessment["visual_region_supports_extension"],
                        "visual_region_supports_reply_region": assessment["visual_region_supports_reply_region"],
                        "visual_region_low_confidence": assessment["visual_region_low_confidence"],
                        "unrelated_page_content": assessment["unrelated_page_content"],
                        "unrelated_page_hits": assessment["unrelated_page_hits"],
                        "drift_detected": assessment["drift_detected"],
                        "drift_containment_active": drift_containment_active,
                        "before_progress": assessment["before_progress"],
                        "after_progress": assessment["after_progress"],
                    },
                )
            else:
                no_progress_steps += 1
                recent_completion_progress_steps = max(0, recent_completion_progress_steps - 1)
                if assessment["activate_drift_containment"]:
                    drift_containment_active = True
                event = (
                    "reply_scroll_prompt_boundary"
                    if assessment["skip_reason"] == "prompt_or_preamble_contamination"
                    else "reply_scroll_segment_skipped"
                )
                self._log(
                    event,
                    {
                        "step": step,
                        "skip_reason": assessment["skip_reason"],
                        "length": len(str(assessment["segment_text"])),
                        "preview": str(assessment["segment_text"])[:220],
                        "novelty_count": assessment["novelty_count"],
                        "growth_chars": assessment["growth_chars"],
                        "completion_progressed": assessment["completion_progressed"],
                        "meaningful_completion_progress": assessment["meaningful_completion_progress"],
                        "completion_score_delta": assessment["completion_score_delta"],
                        "completion_score": assessment["after_progress"]["completion_score"],
                        "region_integrity_score": assessment["region_integrity_score"],
                        "region_reasons": assessment["region_reasons"],
                        "region_consistent": assessment["region_consistent"],
                        "trusted_lineage_score": assessment["trusted_lineage_score"],
                        "trusted_lineage_reasons": assessment["trusted_lineage_reasons"],
                        "trusted_lineage_extended": assessment["trusted_lineage_extended"],
                        "matched_seed_lineage": assessment["matched_seed_lineage"],
                        "matched_trusted_lineage": assessment["matched_trusted_lineage"],
                        "contextual_structured_extension": assessment["contextual_structured_extension"],
                        "contextual_extension_allowed_reason": assessment["contextual_extension_allowed_reason"],
                        "contextual_extension_denied_reason": assessment["contextual_extension_denied_reason"],
                        "matched_current_blob_only": assessment["matched_current_blob_only"],
                        "continuity_against_seed": assessment["continuity_against_seed"],
                        "continuity_against_trusted_lineage": assessment["continuity_against_trusted_lineage"],
                        "continuity_against_current": assessment["continuity_against_current"],
                        "visual_region_confidence": assessment["visual_region_confidence"],
                        "visual_region_used_remembered_region": assessment["visual_region_used_remembered_region"],
                        "visual_region_confirmed_reply_region": assessment["visual_region_confirmed_reply_region"],
                        "visual_region_supports_extension": assessment["visual_region_supports_extension"],
                        "visual_region_supports_reply_region": assessment["visual_region_supports_reply_region"],
                        "visual_region_low_confidence": assessment["visual_region_low_confidence"],
                        "unrelated_page_content": assessment["unrelated_page_content"],
                        "unrelated_page_hits": assessment["unrelated_page_hits"],
                        "drift_detected": assessment["drift_detected"],
                        "drift_containment_active": drift_containment_active,
                        "before_progress": assessment["before_progress"],
                        "after_progress": assessment["after_progress"],
                    },
                )

            if looks_like_patch_plan_json(merged):
                self._scroll_reply_to_bottom(target)
                return merged
            if should_extend_structured_scroll_window(
                merged,
                recent_completion_progress_steps=recent_completion_progress_steps,
                continuation_windows_used=continuation_windows_used,
                step_index=step,
                step_limit=step_limit,
                no_progress_steps=no_progress_steps,
            ):
                previous_limit = step_limit
                continuation_windows_used += 1
                step_limit += 2
                self._log(
                    "reply_scroll_continuation_window",
                    {
                        "step": step,
                        "reason": "structured_completion_progress",
                        "previous_step_limit": previous_limit,
                        "new_step_limit": step_limit,
                        "recent_completion_progress_steps": recent_completion_progress_steps,
                        "completion_score": structured_completion_score(merged),
                        "progress": structured_completion_progress(merged),
                    },
                )
            if no_progress_steps >= 3:
                break
            self._scroll_reply_up(target)
            step += 1

        self._scroll_reply_to_bottom(target)
        if merged.strip():
            return merged
        if structured_anchor_text.strip():
            return merge_scrolled_reply_segments(structured_anchor_text, list(reversed(segments)), policy=REPLY_TRACKER_POLICY)
        return merge_text_segments(list(reversed(segments)))

    def _capture_visible_reply_segment(self, target, before_lines, prompt: str) -> str:
        ocr_text, uia_text = self._capture_visible_reply_sources(target, before_lines, prompt)
        return merge_reply_segment_sources(uia_text, ocr_text, policy=REPLY_TRACKER_POLICY)

    def _scroll_reply_to_bottom(self, target) -> None:
        context = self._begin_visual_action(
            target=target,
            action_type="scroll_action",
            target_intent="reply_region",
            ui_state=self._ui_state(target),
            target_rect=self._default_reply_region_rect(target),
        )
        self._capture_visual_action_stage(
            context,
            target=target,
            stage="during",
            ui_state=self._ui_state(target),
            action_outcome="scroll_to_bottom_start",
        )
        self._focus_reply_area(target)
        for _ in range(3):
            pyautogui.press("end")
            time.sleep(0.25)
            pyautogui.press("pagedown")
            time.sleep(0.25)
        self._capture_visual_action_stage(
            context,
            target=target,
            stage="post",
            ui_state=self._ui_state(target),
            action_outcome="scroll_to_bottom_completed",
        )

    def _scroll_reply_up(self, target) -> None:
        context = self._begin_visual_action(
            target=target,
            action_type="scroll_action",
            target_intent="reply_region",
            ui_state=self._ui_state(target),
            target_rect=self._default_reply_region_rect(target),
        )
        self._capture_visual_action_stage(
            context,
            target=target,
            stage="during",
            ui_state=self._ui_state(target),
            action_outcome="scroll_reply_up_start",
        )
        self._focus_reply_area(target)
        pyautogui.press("pageup")
        time.sleep(0.25)
        self._capture_visual_action_stage(
            context,
            target=target,
            stage="post",
            ui_state=self._ui_state(target),
            action_outcome="scroll_reply_up_completed",
        )

    def _focus_reply_area(self, target) -> None:
        ui_state = self._ui_state(target)
        hint = self._visual_memory_hint(intent="reply_region", target=target, ui_state=ui_state)
        hint_rect = self._hint_rect(hint, target)
        target_rect = hint_rect or self._default_reply_region_rect(target)
        context = self._begin_visual_action(
            target=target,
            action_type="reply_focus",
            target_intent="reply_region",
            ui_state=ui_state,
            target_rect=target_rect,
            candidate_rects=[target_rect],
            confidence_before=hint.confidence if hint is not None else None,
            used_remembered_region=hint is not None,
        )
        self._focus_window(target)
        self._capture_visual_action_stage(
            context,
            target=target,
            stage="during",
            ui_state=ui_state,
            action_outcome="reply_focus_start",
            confidence_after=hint.confidence if hint is not None else None,
        )
        click_center = self._rect_center(target_rect) or (
            int(target.left + target.width * 0.70),
            int(target.top + target.height * 0.42),
        )
        pyautogui.click(click_center[0], click_center[1])
        self._visual_memory_update(
            intent="reply_region",
            target=target,
            action_type="reply_focus",
            rect=target_rect,
            ui_state=self._ui_state(target),
        )
        self._capture_visual_action_stage(
            context,
            target=target,
            stage="post",
            ui_state=self._ui_state(target),
            action_outcome="reply_focus_completed",
            confidence_after=hint.confidence if hint is not None else None,
            updated_remembered_region=True,
        )
        time.sleep(0.2)

    def _focus_window(self, target) -> None:
        try:
            self._executor.focus_window_target(target)
            return
        except Exception as exc:
            error = str(exc)
            self._log(
                "window_focus_failed",
                {
                    "error": error,
                    "title": getattr(target, "title", ""),
                    "handle": getattr(target, "handle", None),
                },
            )
        try:
            x = int(target.left + min(target.width * 0.5, max(80, target.width * 0.5)))
            y = int(target.top + min(target.height * 0.08, max(30, min(80, target.height * 0.08))))
            pyautogui.click(x, y)
            time.sleep(0.35)
            self._log(
                "window_focus_fallback_click",
                {
                    "x": x,
                    "y": y,
                    "title": getattr(target, "title", ""),
                    "handle": getattr(target, "handle", None),
                },
            )
        except Exception as fallback_exc:
            raise RuntimeError(
                "Could not focus the ChatGPT browser window for automation. "
                f"Original focus error: {error}. Fallback error: {fallback_exc}."
            ) from fallback_exc

    @staticmethod
    def _response_still_streaming(ui_state: dict[str, Any]) -> bool:
        return bool(ui_state.get("stop_streaming_present"))

    @staticmethod
    def _reply_acceptance_log_payload(acceptance) -> dict[str, Any]:
        return {
            "accepted": acceptance.accepted,
            "acceptance_tier": acceptance.acceptance_tier,
            "acceptance_reason": acceptance.acceptance_reason,
            "rejection_reason": acceptance.rejection_reason,
            "requires_more_observation": acceptance.requires_more_observation,
            "should_scroll": acceptance.should_scroll,
        }

    @staticmethod
    def _structured_candidate_priority(candidate: _StructuredReplyCandidate) -> tuple[int, int, int, int, float, int, int]:
        return (
            int(candidate.parseable),
            int(candidate.retry_safe),
            int(candidate.salvage_allowed),
            int(candidate.region_trusted),
            candidate.observation_count,
            candidate.score,
            len(candidate.parsed_text),
            len(candidate.raw_text),
        )

    @staticmethod
    def _retry_seed_contaminated(text: str) -> bool:
        lowered = " ".join(text.lower().split())
        markers = (
            "system:",
            "user:",
            "coding orchestrator backend",
            "return json only",
            "browser mode has a smaller prompt budget",
            "context omitted for browser size safety",
            "relevant file tree:",
            "please do not type, click, or move the mouse",
        )
        return any(marker in lowered for marker in markers)

    def _classify_retry_seed_text(
        self,
        text: str,
        *,
        region_trusted: bool,
        salvage_allowed: bool,
    ) -> tuple[bool, str]:
        cleaned = text.strip()
        if not cleaned:
            return False, "empty_structured_text"
        if not salvage_allowed:
            return False, "candidate_not_salvageable"
        if not region_trusted:
            return False, "reply_region_not_trusted"
        if self._retry_seed_contaminated(cleaned):
            return False, "prompt_or_preamble_contamination"
        progress = structured_completion_progress(cleaned)
        if int(progress.get("schema_hits", 0)) < 3:
            return False, "missing_required_schema_keys"
        if int(progress.get("operation_items", 0)) < 1:
            return False, "operations_array_incomplete"
        if int(progress.get("brace_balance", 0)) != 0 or int(progress.get("bracket_balance", 0)) != 0:
            return False, "unbalanced_structure"
        if not bool(progress.get("parseable")):
            return False, "not_parseable_json"
        return True, "parseable_structured_json"

    @staticmethod
    def _retry_attempt_outside_block_text(raw_text: str, extracted: str) -> str:
        outside = raw_text.strip()
        if extracted and extracted in outside:
            outside = outside.replace(extracted, " ", 1)
        outside = outside.replace("ASTER_PATCH_BEGIN", " ").replace("ASTER_PATCH_END", " ")
        return " ".join(outside.split())

    @staticmethod
    def _count_retry_attempt_blocks(raw_text: str) -> int:
        normalized = raw_text.upper()
        return normalized.count("ASTER_PATCH_BEGIN")

    @staticmethod
    def _retry_attempt_block_hash(text: str) -> str:
        normalized = " ".join(text.split())
        return hashlib.sha1(normalized.encode("utf-8", errors="ignore")).hexdigest()[:12]

    @staticmethod
    def _retry_attempt_preview(text: str, limit: int = 160) -> str:
        preview = " ".join(text.split())
        if len(preview) <= limit:
            return preview
        return preview[:limit].rstrip() + "..."

    @staticmethod
    def _classify_retry_attempt_block_fragment(
        *,
        raw_block: str,
        extracted: str,
        parseable: bool,
        has_end_marker: bool,
    ) -> tuple[str, str]:
        lowered = _normalize(raw_block)
        wrapper_only = (
            "aster_patch_begin" in lowered or "aster patch begin" in lowered
        ) and not any(token in raw_block for token in ("{", '"summary"', '"operations"'))
        code_only = any(marker in lowered for marker in ("def ", "class ", "import ", "self.", "tkinter", "assert "))
        has_jsonish = any(token in raw_block for token in ("{", "}", '"summary"', '"operations"', '"notes"'))
        if parseable:
            return "parseable_json", "parseable_json_block"
        if wrapper_only:
            return "empty_payload", "wrapper_only_block"
        if not extracted.strip():
            if code_only:
                return "non_json_payload", "code_only_block"
            return "empty_payload", "marker_only_block"
        if not has_end_marker:
            if has_jsonish:
                return "partial_payload", "fragment_without_end_marker"
            return "partial_payload", "truncated_non_json_fragment"
        if has_jsonish:
            return "non_parseable_json", "malformed_json_block"
        if code_only:
            return "non_json_payload", "code_only_block"
        return "non_json_payload", "non_json_block"

    @staticmethod
    def _extract_retry_attempt_blocks(raw_text: str) -> list[dict[str, Any]]:
        if not raw_text.strip():
            return []
        begin_pattern = re.compile(r"ASTER[_ ]PATCH[_ ]BEGIN", flags=re.IGNORECASE)
        end_pattern = re.compile(r"ASTER[_ ]PATCH[_ ]END", flags=re.IGNORECASE)
        begins = list(begin_pattern.finditer(raw_text))
        if not begins:
            return []
        blocks: list[dict[str, Any]] = []
        for index, match in enumerate(begins, start=1):
            next_start = begins[index].start() if index < len(begins) else len(raw_text)
            search_region = raw_text[match.end() : next_start]
            end_match = end_pattern.search(search_region)
            if end_match is not None:
                block_end = match.end() + end_match.end()
                has_end_marker = True
            else:
                block_end = next_start
                has_end_marker = False
            raw_block = raw_text[match.start() : block_end].strip()
            extracted = extract_structured_block(raw_block).strip()
            parseable = bool(extracted) and looks_like_patch_plan_json(extracted)
            canonical_json = ""
            if parseable:
                try:
                    canonical_json = json.dumps(json.loads(extracted), sort_keys=True, separators=(",", ":"))
                except Exception:
                    canonical_json = extracted
            payload_state, block_kind = BrowserChatGPTTransport._classify_retry_attempt_block_fragment(
                raw_block=raw_block,
                extracted=extracted,
                parseable=parseable,
                has_end_marker=has_end_marker,
            )
            raw_schema_hits = sum(1 for token in ('"summary"', '"notes"', '"operations"') if token in raw_block)
            blocks.append(
                {
                    "index": index,
                    "raw_block": raw_block,
                    "extracted": extracted,
                    "parseable": parseable,
                    "canonical_json": canonical_json,
                    "has_end_marker": has_end_marker,
                    "schema_hits": sum(1 for token in ('"summary"', '"notes"', '"operations"') if token in extracted),
                    "raw_schema_hits": raw_schema_hits,
                    "brace_balance": extracted.count("{") - extracted.count("}"),
                    "bracket_balance": extracted.count("[") - extracted.count("]"),
                    "length": len(extracted),
                    "raw_length": len(raw_block),
                    "payload_state": payload_state,
                    "block_kind": block_kind,
                    "preview": BrowserChatGPTTransport._retry_attempt_preview(extracted or raw_block),
                    "block_hash": BrowserChatGPTTransport._retry_attempt_block_hash(extracted or raw_block),
                }
            )
        return blocks

    @staticmethod
    def _summarize_retry_attempt_blocks(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        summary: list[dict[str, Any]] = []
        for block in blocks[:4]:
            summary.append(
                {
                    "index": int(block["index"]),
                    "block_hash": str(block["block_hash"]),
                    "preview": str(block["preview"]),
                    "raw_length": int(block["raw_length"]),
                    "extracted_length": int(block["length"]),
                    "has_end_marker": bool(block["has_end_marker"]),
                    "parseable": bool(block["parseable"]),
                    "payload_state": str(block["payload_state"]),
                    "block_kind": str(block["block_kind"]),
                    "schema_hits": int(block["schema_hits"]),
                    "raw_schema_hits": int(block["raw_schema_hits"]),
                    "brace_balance": int(block["brace_balance"]),
                    "bracket_balance": int(block["bracket_balance"]),
                }
            )
        return summary

    @staticmethod
    def _default_retry_block_selection() -> dict[str, Any]:
        return {
            "recovered": False,
            "ambiguous": False,
            "selected_block_index": None,
            "selection_reason": "",
            "failure_reason": "",
            "selected_text": "",
            "relationship": "",
            "fragment_repair_pattern_matched": False,
            "fragment_repair_attempted": False,
            "fragment_repair_succeeded": False,
            "fragment_repair_reason": "",
            "fragment_repair_parseable_after_cleanup": False,
            "repaired_from_block_index": None,
            "discarded_wrapper_only_block_index": None,
            "wrapper_only_block_count": 0,
            "wrapper_only_payload_lengths": [],
            "wrapper_only_has_internal_text": False,
            "wrapper_only_noise_detected": False,
            "wrapper_only_boundary_suspected": False,
            "wrapper_recheck_attempted": False,
            "wrapper_recheck_found_payload": False,
            "wrapper_recheck_reason": "",
            "json_cleanup_attempted": False,
            "json_cleanup_succeeded": False,
            "json_cleanup_reason": "",
            "json_cleanup_changed": False,
            "prose_recovery_attempted": False,
            "prose_recovery_succeeded": False,
            "prose_recovery_reason": "",
            "ocr_cleanup_attempted": False,
            "ocr_cleanup_succeeded": False,
            "ocr_cleanup_reason": "",
            "ocr_defect_types": [],
            "single_block_completion_attempted": False,
            "single_block_completion_succeeded": False,
            "single_block_completion_reason": "",
            "single_block_completion_defect_types": [],
            "single_block_completion_closure_added": "",
            "single_block_semantically_incomplete": False,
            "single_block_parseable_after_completion": False,
        }

    @staticmethod
    def _retry_attempt_inner_payload(raw_block: str) -> str:
        text = raw_block.strip()
        text = re.sub(
            r"^\s*ASTER[_ ]PATCH[_ ]BEGIN\s*",
            "",
            text,
            count=1,
            flags=re.IGNORECASE,
        )
        text = re.sub(
            r"\s*ASTER[_ ]PATCH[_ ]END\s*$",
            "",
            text,
            count=1,
            flags=re.IGNORECASE,
        )
        return text

    @staticmethod
    def _retry_source_preference_defaults() -> dict[str, Any]:
        return {
            "fragment_source_preference": "",
            "fragment_source_chosen": "",
            "fragment_uia_available": False,
            "fragment_ocr_available": False,
        }

    @staticmethod
    def _retry_fragment_assessment_metrics(assessment: dict[str, Any]) -> tuple[int, int]:
        forensics = list(assessment.get("block_forensics", []))
        max_schema_hits = 0
        balance_penalty = 99
        for item in forensics:
            max_schema_hits = max(max_schema_hits, int(item.get("raw_schema_hits", item.get("schema_hits", 0))))
            balance_penalty = min(
                balance_penalty,
                abs(int(item.get("brace_balance", 0))) + abs(int(item.get("bracket_balance", 0))),
            )
        if balance_penalty == 99:
            balance_penalty = 0
        return max_schema_hits, balance_penalty

    @classmethod
    def _retry_fragment_assessment_strength(cls, assessment: dict[str, Any], *, source: str) -> tuple[int, int, int, int, int, int]:
        max_schema_hits, balance_penalty = cls._retry_fragment_assessment_metrics(assessment)
        return (
            int(bool(assessment.get("selected_text"))),
            int(bool(assessment.get("parseable"))),
            int(bool(assessment.get("structured_block_found"))),
            max_schema_hits,
            -balance_penalty,
            int(source == "uia"),
        )

    def _choose_retry_attempt_candidate_sources(
        self,
        *,
        ocr_text: str,
        uia_text: str,
        prompt_anchor,
    ) -> tuple[tuple[str, str, float] | None, dict[str, Any]]:
        meta = self._retry_source_preference_defaults()
        available: dict[str, tuple[str, str, float]] = {}
        assessments: dict[str, dict[str, Any]] = {}
        for source, text in (("uia", uia_text), ("ocr", ocr_text)):
            if not text.strip():
                continue
            candidate = choose_best_reply_candidate_for_policy(
                {source: text},
                policy=REPLY_TRACKER_POLICY,
                prompt_anchor=prompt_anchor,
            )
            candidate_score = (
                candidate[2]
                if candidate is not None
                else score_candidate_for_policy(text, policy=REPLY_TRACKER_POLICY)
            )
            # Retry-attempt assessment needs the raw captured source text so it can
            # reason about wrapper completeness, prose contamination, and malformed
            # multi-block fragments before any block extraction short-circuits that view.
            available[source] = (source, text, candidate_score)
            assessments[source] = self._assess_retry_attempt_response(text, candidate_source=source)
        meta["fragment_uia_available"] = "uia" in available
        meta["fragment_ocr_available"] = "ocr" in available
        if not available:
            return None, meta
        if "uia" not in available:
            meta["fragment_source_preference"] = "ocr_only_available"
            meta["fragment_source_chosen"] = "ocr"
            return available["ocr"], meta
        if "ocr" not in available:
            meta["fragment_source_preference"] = "uia_only_available"
            meta["fragment_source_chosen"] = "uia"
            return available["uia"], meta

        uia_assessment = assessments["uia"]
        ocr_assessment = assessments["ocr"]
        if self._retry_fragment_assessment_strength(ocr_assessment, source="ocr") > self._retry_fragment_assessment_strength(
            uia_assessment,
            source="uia",
        ):
            ocr_schema_hits, ocr_balance_penalty = self._retry_fragment_assessment_metrics(ocr_assessment)
            uia_schema_hits, uia_balance_penalty = self._retry_fragment_assessment_metrics(uia_assessment)
            ocr_clearly_better = (
                (bool(ocr_assessment.get("selected_text")) and not bool(uia_assessment.get("selected_text")))
                or (bool(ocr_assessment.get("parseable")) and not bool(uia_assessment.get("parseable")))
                or (
                    ocr_schema_hits >= uia_schema_hits + 2
                    and ocr_balance_penalty <= uia_balance_penalty
                    and bool(ocr_assessment.get("structured_block_found"))
                )
            )
            if ocr_clearly_better:
                meta["fragment_source_preference"] = "ocr_clearly_better_structured_signal"
                meta["fragment_source_chosen"] = "ocr"
                return available["ocr"], meta

        meta["fragment_source_preference"] = "prefer_uia_fragment_over_ocr"
        meta["fragment_source_chosen"] = "uia"
        return available["uia"], meta

    @staticmethod
    def _normalize_ocr_schema_key_candidate(token: str) -> str:
        normalized = token.strip().lower()
        normalized = normalized.translate(str.maketrans({"0": "o", "1": "i", "|": "l", "!": "i", "5": "s", "$": "s"}))
        normalized = re.sub(r"[^a-z_]", "", normalized)
        return normalized

    @classmethod
    def _cleanup_ocr_fragment_body(cls, fragment_text: str) -> tuple[str, list[str]]:
        known_keys = [
            "summary",
            "notes",
            "operations",
            "type",
            "path",
            "reason",
            "content",
            "commands",
            "packages",
            "new_path",
            "diff_hint",
        ]
        cleaned = fragment_text.strip()
        defects: list[str] = []
        normalized_quotes = (
            cleaned.replace("\u201c", '"')
            .replace("\u201d", '"')
            .replace("\u2018", "'")
            .replace("\u2019", "'")
        )
        if normalized_quotes != cleaned:
            cleaned = normalized_quotes
            defects.append("smart_quotes")
        without_markers = re.sub(
            r"ASTER[_ ]PATCH[_ ](?:BEGIN|END)\s*",
            "",
            cleaned,
            flags=re.IGNORECASE,
        ).strip()
        if without_markers != cleaned:
            cleaned = without_markers
            defects.append("inner_wrapper_markers")

        def _repair_key(match: re.Match[str]) -> str:
            key = match.group(1)
            normalized = cls._normalize_ocr_schema_key_candidate(key)
            if normalized in known_keys:
                repaired = normalized
            else:
                close = difflib.get_close_matches(normalized, known_keys, n=1, cutoff=0.75)
                repaired = close[0] if close else key
            if repaired != key:
                defects.append("schema_key_ocr")
            return f'"{repaired}":'

        repaired_keys = re.sub(r'"([^"\n]{1,40})"\s*:', _repair_key, cleaned)
        if repaired_keys != cleaned:
            cleaned = repaired_keys

        repaired_notes = re.sub(r'("notes"\s*:\s*)[lI|]\s*[:;]', r"\1[]", cleaned)
        if repaired_notes != cleaned:
            cleaned = repaired_notes
            defects.append("array_opening_punctuation")
        repaired_operations = re.sub(r'("operations"\s*:\s*)[lI|]\s*[:;]', r"\1[", cleaned)
        if repaired_operations != cleaned:
            cleaned = repaired_operations
            defects.append("array_opening_punctuation")
        repaired_commas = re.sub(r",\s*,+", ",", cleaned)
        if repaired_commas != cleaned:
            cleaned = repaired_commas
            defects.append("duplicated_commas")
        repaired_colons = re.sub(r":\s*:+", ":", cleaned)
        if repaired_colons != cleaned:
            cleaned = repaired_colons
            defects.append("duplicated_colons")
        repaired_braces = re.sub(r"{\s*{+", "{", cleaned)
        if repaired_braces != cleaned:
            cleaned = repaired_braces
            defects.append("duplicated_braces")
        deduped_defects = list(dict.fromkeys(defects))
        return cleaned.strip(), deduped_defects

    @staticmethod
    def _sanitize_retry_json_candidate(candidate_text: str) -> tuple[str, list[str]]:
        cleaned = candidate_text.strip()
        changes: list[str] = []
        stripped_wrappers = re.sub(
            r"ASTER[_ ]PATCH[_ ](?:BEGIN|END)\s*",
            "",
            cleaned,
            flags=re.IGNORECASE,
        ).strip()
        if stripped_wrappers != cleaned:
            cleaned = stripped_wrappers
            changes.append("removed_inner_wrapper_markers")
        normalized_quotes = (
            cleaned.replace("\u201c", '"')
            .replace("\u201d", '"')
            .replace("\u2018", "'")
            .replace("\u2019", "'")
        )
        if normalized_quotes != cleaned:
            cleaned = normalized_quotes
            changes.append("normalized_smart_quotes")
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            trimmed = cleaned[start : end + 1].strip()
            if trimmed != cleaned:
                cleaned = trimmed
                changes.append("trimmed_text_around_json_object")
        without_trailing_commas = re.sub(r",(\s*[}\]])", r"\1", cleaned)
        if without_trailing_commas != cleaned:
            cleaned = without_trailing_commas
            changes.append("removed_trailing_commas")
        return cleaned.strip(), changes

    @classmethod
    def _analyze_single_retry_block_completion_candidate(cls, candidate_text: str) -> dict[str, Any]:
        cleaned_candidate, cleanup_changes = cls._sanitize_retry_json_candidate(candidate_text)
        defect_types = list(cleanup_changes)
        stack: list[str] = []
        inside_string = False
        escaped = False
        last_significant = ""
        for char in cleaned_candidate:
            if inside_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    inside_string = False
                continue
            if char.isspace():
                continue
            last_significant = char
            if char == '"':
                inside_string = True
            elif char in "{[":
                stack.append(char)
            elif char in "}]":
                if not stack:
                    return {
                        "safe_to_complete": False,
                        "reason": "unexpected_closing_delimiter",
                        "semantically_incomplete": False,
                        "parseable_after_completion": False,
                        "defect_types": list(dict.fromkeys(defect_types + ["unexpected_closing_delimiter"])),
                        "completion_candidate": cleaned_candidate,
                        "closure_tokens": "",
                        "closure_added": "",
                    }
                expected = "{" if char == "}" else "["
                if stack[-1] != expected:
                    return {
                        "safe_to_complete": False,
                        "reason": "mismatched_closing_delimiter",
                        "semantically_incomplete": False,
                        "parseable_after_completion": False,
                        "defect_types": list(dict.fromkeys(defect_types + ["mismatched_closing_delimiter"])),
                        "completion_candidate": cleaned_candidate,
                        "closure_tokens": "",
                        "closure_added": "",
                    }
                stack.pop()

        if escaped:
            return {
                "safe_to_complete": False,
                "reason": "unterminated_escape",
                "semantically_incomplete": True,
                "parseable_after_completion": False,
                "defect_types": list(dict.fromkeys(defect_types + ["unterminated_escape"])),
                "completion_candidate": cleaned_candidate,
                "closure_tokens": "",
                "closure_added": "",
            }
        if inside_string:
            return {
                "safe_to_complete": False,
                "reason": "unterminated_string",
                "semantically_incomplete": True,
                "parseable_after_completion": False,
                "defect_types": list(dict.fromkeys(defect_types + ["unterminated_string"])),
                "completion_candidate": cleaned_candidate,
                "closure_tokens": "",
                "closure_added": "",
            }
        if not stack:
            return {
                "safe_to_complete": False,
                "reason": "no_missing_structural_closure",
                "semantically_incomplete": False,
                "parseable_after_completion": False,
                "defect_types": defect_types,
                "completion_candidate": cleaned_candidate,
                "closure_tokens": "",
                "closure_added": "",
            }
        if len(stack) > 4:
            return {
                "safe_to_complete": False,
                "reason": "too_many_missing_closures",
                "semantically_incomplete": False,
                "parseable_after_completion": False,
                "defect_types": list(dict.fromkeys(defect_types + ["too_many_missing_closures"])),
                "completion_candidate": cleaned_candidate,
                "closure_tokens": "",
                "closure_added": "",
            }
        if last_significant == ":":
            return {
                "safe_to_complete": False,
                "reason": "trailing_colon",
                "semantically_incomplete": True,
                "parseable_after_completion": False,
                "defect_types": list(dict.fromkeys(defect_types + ["trailing_colon"])),
                "completion_candidate": cleaned_candidate,
                "closure_tokens": "",
                "closure_added": "",
            }
        if last_significant == ",":
            return {
                "safe_to_complete": False,
                "reason": "trailing_comma",
                "semantically_incomplete": True,
                "parseable_after_completion": False,
                "defect_types": list(dict.fromkeys(defect_types + ["trailing_comma"])),
                "completion_candidate": cleaned_candidate,
                "closure_tokens": "",
                "closure_added": "",
            }
        if last_significant == "{":
            return {
                "safe_to_complete": False,
                "reason": "trailing_open_object",
                "semantically_incomplete": True,
                "parseable_after_completion": False,
                "defect_types": list(dict.fromkeys(defect_types + ["trailing_open_object"])),
                "completion_candidate": cleaned_candidate,
                "closure_tokens": "",
                "closure_added": "",
            }
        if last_significant == "[":
            return {
                "safe_to_complete": False,
                "reason": "trailing_open_array",
                "semantically_incomplete": True,
                "parseable_after_completion": False,
                "defect_types": list(dict.fromkeys(defect_types + ["trailing_open_array"])),
                "completion_candidate": cleaned_candidate,
                "closure_tokens": "",
                "closure_added": "",
            }

        tail = cleaned_candidate[-160:]
        invalid_bareword_patterns = (
            r':\s*[A-Za-z_][A-Za-z0-9 _.\-]*$',
            r',\s*[A-Za-z_][A-Za-z0-9 _.\-]*$',
        )
        if any(re.search(pattern, tail) for pattern in invalid_bareword_patterns) and not re.search(
            r':\s*(true|false|null)\s*$',
            tail,
            flags=re.IGNORECASE,
        ):
            return {
                "safe_to_complete": False,
                "reason": "invalid_bareword_value_tail",
                "semantically_incomplete": True,
                "parseable_after_completion": False,
                "defect_types": list(dict.fromkeys(defect_types + ["invalid_bareword_value_tail"])),
                "completion_candidate": cleaned_candidate,
                "closure_tokens": "",
                "closure_added": "",
            }

        closure_tokens = "".join("}" if opener == "{" else "]" for opener in reversed(stack))
        closure_defects = ["missing_closing_brace" if opener == "{" else "missing_closing_bracket" for opener in reversed(stack)]
        closure_added = closure_tokens + "\nASTER_PATCH_END"
        return {
            "safe_to_complete": True,
            "reason": "balanced_structural_closure",
            "semantically_incomplete": False,
            "parseable_after_completion": False,
            "defect_types": list(dict.fromkeys(defect_types + closure_defects + ["missing_end_marker"])),
            "completion_candidate": cleaned_candidate + closure_tokens,
            "closure_tokens": closure_tokens,
            "closure_added": closure_added,
        }

    @classmethod
    def _attempt_single_incomplete_retry_block_completion(
        cls,
        raw_text: str,
        blocks: list[dict[str, Any]],
        *,
        extracted: str,
    ) -> dict[str, Any]:
        result = cls._default_retry_block_selection()
        if len(blocks) != 1:
            return result
        block = blocks[0]
        if bool(block["has_end_marker"]):
            return result
        if str(block["block_kind"]) != "fragment_without_end_marker":
            return result

        candidate_text = str(block["extracted"]).strip() or cls._retry_attempt_inner_payload(str(block["raw_block"])).strip()
        if not candidate_text:
            result["single_block_completion_attempted"] = True
            result["single_block_completion_reason"] = "no_completion_candidate"
            result["failure_reason"] = "retry_block_not_safely_completable"
            return result

        result["single_block_completion_attempted"] = True
        schema_hits = max(
            int(block.get("raw_schema_hits", 0)),
            int(block.get("schema_hits", 0)),
            sum(1 for token in ('"summary"', '"notes"', '"operations"') if token in candidate_text),
        )
        if schema_hits < 2 or '"operations"' not in candidate_text:
            result["single_block_completion_reason"] = "insufficient_structure"
            result["failure_reason"] = "retry_block_not_safely_completable"
            return result

        analysis = cls._analyze_single_retry_block_completion_candidate(candidate_text)
        result["single_block_completion_defect_types"] = list(analysis["defect_types"])
        result["single_block_completion_closure_added"] = str(analysis["closure_added"])
        result["single_block_semantically_incomplete"] = bool(analysis["semantically_incomplete"])
        if not analysis["safe_to_complete"]:
            result["single_block_completion_reason"] = str(analysis["reason"])
            result["failure_reason"] = (
                "retry_block_semantically_incomplete"
                if analysis["semantically_incomplete"]
                else "retry_block_not_safely_completable"
            )
            return result

        try:
            parsed_candidate = json.loads(str(analysis["completion_candidate"]))
        except Exception:
            result["single_block_completion_reason"] = "completion_candidate_not_parseable"
            result["failure_reason"] = "retry_block_completion_not_parseable"
            return result

        canonical_candidate = json.dumps(parsed_candidate, separators=(",", ":"))
        if not looks_like_patch_plan_json(canonical_candidate):
            result["single_block_completion_reason"] = "completion_candidate_missing_required_schema"
            result["failure_reason"] = "retry_block_not_safely_completable"
            return result

        result.update(
            {
                "recovered": True,
                "selected_block_index": int(block["index"]),
                "selection_reason": "single_incomplete_retry_block_completed",
                "failure_reason": "",
                "selected_text": canonical_candidate,
                "relationship": "single_incomplete_retry_block",
                "single_block_completion_succeeded": True,
                "single_block_completion_reason": str(analysis["reason"]),
                "single_block_parseable_after_completion": True,
            }
        )
        return result

    @classmethod
    def _attempt_retry_json_cleanup(
        cls,
        raw_text: str,
        blocks: list[dict[str, Any]],
        *,
        extracted: str,
    ) -> dict[str, Any]:
        result = cls._default_retry_block_selection()
        candidate_text = ""
        selected_block_index: int | None = None
        if len(blocks) == 1:
            block = blocks[0]
            selected_block_index = int(block["index"])
            candidate_text = str(block["extracted"]).strip() or cls._retry_attempt_inner_payload(str(block["raw_block"])).strip()
        elif extracted.strip():
            candidate_text = extracted.strip()
        if not candidate_text:
            result["json_cleanup_reason"] = "no_extractable_json_candidate"
            return result

        result["json_cleanup_attempted"] = True
        cleaned_candidate, changes = cls._sanitize_retry_json_candidate(candidate_text)
        result["json_cleanup_changed"] = cleaned_candidate != candidate_text
        if "{" not in cleaned_candidate or (
            '"summary"' not in cleaned_candidate
            and '"operations"' not in cleaned_candidate
            and '"notes"' not in cleaned_candidate
        ):
            result["json_cleanup_reason"] = "insufficient_json_structure"
            result["failure_reason"] = "retry_json_not_safely_repairable"
            return result
        if not changes:
            result["json_cleanup_reason"] = "no_safe_cleanup_transform"
            result["failure_reason"] = "retry_json_not_safely_repairable"
            return result
        try:
            parsed_candidate = json.loads(cleaned_candidate)
        except Exception:
            result["json_cleanup_reason"] = "cleanup_candidate_not_parseable"
            result["failure_reason"] = "retry_json_cleanup_failed"
            return result
        canonical_candidate = json.dumps(parsed_candidate, separators=(",", ":"))
        if not looks_like_patch_plan_json(canonical_candidate):
            result["json_cleanup_reason"] = "cleanup_candidate_missing_required_schema"
            result["failure_reason"] = "retry_json_not_safely_repairable"
            return result
        result.update(
            {
                "recovered": True,
                "selected_block_index": selected_block_index,
                "selection_reason": "retry_json_cleanup_recovered_block",
                "failure_reason": "",
                "selected_text": canonical_candidate,
                "relationship": "single_block_json_cleanup",
                "json_cleanup_succeeded": True,
                "json_cleanup_reason": ",".join(changes),
            }
        )
        return result

    @classmethod
    def _attempt_retry_prose_embedded_block_recovery(
        cls,
        raw_text: str,
        blocks: list[dict[str, Any]],
        *,
        extracted: str,
        outside_text: str,
    ) -> dict[str, Any]:
        result = cls._default_retry_block_selection()
        if not outside_text.strip():
            if not blocks and raw_text.strip():
                result["prose_recovery_attempted"] = True
                result["prose_recovery_reason"] = "no_single_parseable_embedded_block"
                return result
            result["prose_recovery_reason"] = "no_outside_text"
            return result
        result["prose_recovery_attempted"] = True
        if len(blocks) == 1 and bool(blocks[0]["parseable"]):
            result.update(
                {
                    "recovered": True,
                    "selected_block_index": int(blocks[0]["index"]),
                    "selection_reason": "single_parseable_embedded_retry_block",
                    "selected_text": str(blocks[0]["extracted"]),
                    "relationship": "prose_wrapped_retry_block",
                    "prose_recovery_succeeded": True,
                    "prose_recovery_reason": "single_parseable_embedded_block",
                }
            )
            return result
        if not blocks and extracted.strip() and looks_like_patch_plan_json(extracted):
            result.update(
                {
                    "recovered": True,
                    "selected_block_index": None,
                    "selection_reason": "single_parseable_embedded_json_object",
                    "selected_text": extracted.strip(),
                    "relationship": "prose_wrapped_json_object",
                    "prose_recovery_succeeded": True,
                    "prose_recovery_reason": "single_parseable_embedded_json_object",
                }
            )
            return result
        result["prose_recovery_reason"] = "no_single_parseable_embedded_block"
        return result

    @classmethod
    def _analyze_wrapper_only_retry_blocks(
        cls,
        raw_text: str,
        blocks: list[dict[str, Any]],
        *,
        outside_text: str,
    ) -> dict[str, Any]:
        wrapper_like_blocks = [
            block
            for block in blocks
            if str(block["block_kind"]) in {"wrapper_only_block", "marker_only_block"}
        ]
        payloads = [cls._retry_attempt_inner_payload(str(block["raw_block"])) for block in wrapper_like_blocks]
        payload_lengths = [len(payload.strip()) for payload in payloads]
        has_internal_text = any(length > 0 for length in payload_lengths)
        raw_lowered = raw_text.lower()
        noise_detected = bool(outside_text.strip()) or any(payload.strip() for payload in payloads)
        boundary_suspected = has_internal_text or any(
            token in raw_lowered
            for token in ('"summary"', '"notes"', '"operations"', "{", "[")
        )
        return {
            "wrapper_only_block_count": len(wrapper_like_blocks),
            "wrapper_only_payload_lengths": payload_lengths,
            "wrapper_only_has_internal_text": has_internal_text,
            "wrapper_only_noise_detected": noise_detected,
            "wrapper_only_boundary_suspected": boundary_suspected,
        }

    @classmethod
    def _attempt_wrapper_only_retry_recheck(
        cls,
        raw_text: str,
        blocks: list[dict[str, Any]],
        *,
        outside_text: str,
    ) -> dict[str, Any]:
        analysis = cls._analyze_wrapper_only_retry_blocks(raw_text, blocks, outside_text=outside_text)
        result = {
            **cls._default_retry_block_selection(),
            **analysis,
        }
        if not analysis["wrapper_only_block_count"]:
            return result
        if not analysis["wrapper_only_boundary_suspected"]:
            result["wrapper_recheck_reason"] = "no_boundary_signal"
            return result

        result["wrapper_recheck_attempted"] = True
        for block in blocks:
            if str(block["block_kind"]) not in {"wrapper_only_block", "marker_only_block"}:
                continue
            inner_payload = cls._retry_attempt_inner_payload(str(block["raw_block"])).strip()
            if not inner_payload:
                continue
            extracted = extract_structured_block(inner_payload).strip()
            if extracted and looks_like_patch_plan_json(extracted):
                result.update(
                    {
                        "recovered": True,
                        "selected_block_index": int(block["index"]),
                        "selection_reason": "wrapper_only_recheck_hidden_payload",
                        "selected_text": extracted,
                        "relationship": "wrapper_only_block_with_hidden_payload",
                        "wrapper_recheck_found_payload": True,
                        "wrapper_recheck_reason": "inner_payload_parseable",
                    }
                )
                return result

        whole_text_extracted = extract_structured_block(raw_text).strip()
        if whole_text_extracted and looks_like_patch_plan_json(whole_text_extracted):
            result.update(
                {
                    "recovered": True,
                    "selected_block_index": None,
                    "selection_reason": "wrapper_only_recheck_full_text_payload",
                    "selected_text": whole_text_extracted,
                    "relationship": "wrapper_only_boundary_recheck_recovered",
                    "wrapper_recheck_found_payload": True,
                    "wrapper_recheck_reason": "full_text_parseable",
                }
            )
            return result

        result["wrapper_recheck_reason"] = "boundary_suspected_but_no_payload"
        return result

    @classmethod
    def _attempt_fragmented_retry_block_repair(
        cls,
        blocks: list[dict[str, Any]],
        *,
        candidate_source: str = "",
    ) -> dict[str, Any]:
        default = cls._default_retry_block_selection()
        if len(blocks) != 2:
            return default
        wrapper_blocks = [
            block
            for block in blocks
            if str(block["block_kind"]) == "wrapper_only_block"
            and str(block["payload_state"]) == "empty_payload"
            and bool(block["has_end_marker"])
        ]
        fragment_blocks = [
            block
            for block in blocks
            if str(block["block_kind"]) == "fragment_without_end_marker"
            and str(block["payload_state"]) == "partial_payload"
            and not bool(block["has_end_marker"])
        ]
        if len(wrapper_blocks) != 1 or len(fragment_blocks) != 1:
            return default
        wrapper = wrapper_blocks[0]
        fragment = fragment_blocks[0]
        result = {
            **default,
            "relationship": "wrapper_only_plus_fragmented_block",
            "fragment_repair_pattern_matched": True,
            "repaired_from_block_index": int(fragment["index"]),
            "discarded_wrapper_only_block_index": int(wrapper["index"]),
        }

        raw_fragment = str(fragment["raw_block"]).strip()
        fragment_body = re.sub(
            r"^\s*ASTER[_ ]PATCH[_ ]BEGIN\s*",
            "",
            raw_fragment,
            count=1,
            flags=re.IGNORECASE,
        ).strip()
        cleaned_fragment_body = ""
        ocr_defect_types: list[str] = []
        if candidate_source == "ocr":
            cleaned_fragment_body, ocr_defect_types = cls._cleanup_ocr_fragment_body(fragment_body or raw_fragment)
            result["ocr_cleanup_attempted"] = True
            result["ocr_defect_types"] = list(ocr_defect_types)
        raw_schema_hits = int(fragment.get("raw_schema_hits", 0))
        cleaned_schema_hits = sum(1 for token in ('"summary"', '"notes"', '"operations"') if token in cleaned_fragment_body)
        effective_schema_hits = max(raw_schema_hits, cleaned_schema_hits)
        if effective_schema_hits < 2:
            result["selection_reason"] = "fragment_repair_insufficient_structure"
            result["failure_reason"] = "fragment_repair_insufficient_structure"
            result["fragment_repair_reason"] = "insufficient_schema_hits"
            if candidate_source == "ocr":
                result["ocr_cleanup_reason"] = result["ocr_cleanup_reason"] or "insufficient_schema_hits_after_cleanup"
            return result

        repair_candidates: list[tuple[str, str]] = []
        repair_candidates.append(
            (
                raw_fragment + "\nASTER_PATCH_END",
                "appended_missing_end_marker",
            )
        )
        if (
            fragment_body
            and not fragment_body.startswith("{")
            and fragment_body.startswith('"')
        ):
            repair_candidates.append(
                (
                    "ASTER_PATCH_BEGIN\n{" + fragment_body.rstrip() + "}\nASTER_PATCH_END",
                    "wrapped_object_body_and_appended_end_marker",
                )
            )

        if candidate_source == "ocr":
            if cleaned_fragment_body and ocr_defect_types:
                ocr_candidate_body = cleaned_fragment_body
                if not ocr_candidate_body.startswith("{") and ocr_candidate_body.startswith('"'):
                    ocr_candidate_body = "{" + ocr_candidate_body
                missing_brackets = ocr_candidate_body.count("[") - ocr_candidate_body.count("]")
                missing_braces = ocr_candidate_body.count("{") - ocr_candidate_body.count("}")
                if 0 <= missing_brackets <= 2 and 0 <= missing_braces <= 2:
                    repaired_body = (
                        ocr_candidate_body
                        + ("]" * missing_brackets)
                        + ("}" * missing_braces)
                    )
                    repair_candidates.append(
                        (
                            "ASTER_PATCH_BEGIN\n" + repaired_body.rstrip() + "\nASTER_PATCH_END",
                            "ocr_cleanup_and_balanced_closure",
                        )
                    )
                repair_candidates.append(
                    (
                        "ASTER_PATCH_BEGIN\n" + ocr_candidate_body.rstrip() + "\nASTER_PATCH_END",
                        "ocr_cleanup_and_appended_end_marker",
                    )
                )
            else:
                result["ocr_cleanup_reason"] = "no_safe_ocr_cleanup_available"

        result["fragment_repair_attempted"] = True
        for candidate_text, reason in repair_candidates:
            extracted = extract_structured_block(candidate_text).strip()
            if extracted and looks_like_patch_plan_json(extracted):
                result.update(
                    {
                        "recovered": True,
                        "selected_block_index": int(fragment["index"]),
                        "selection_reason": "repaired_fragmented_retry_block",
                        "failure_reason": "",
                        "selected_text": extracted,
                        "relationship": "wrapper_only_plus_repaired_fragment",
                        "fragment_repair_succeeded": True,
                        "fragment_repair_reason": reason,
                        "fragment_repair_parseable_after_cleanup": True,
                    }
                )
                if reason.startswith("ocr_cleanup"):
                    result["ocr_cleanup_succeeded"] = True
                    result["ocr_cleanup_reason"] = reason
                return result

        result["selection_reason"] = "fragment_repair_not_parseable"
        result["failure_reason"] = "fragment_repair_not_parseable"
        result["fragment_repair_reason"] = "repair_candidate_not_parseable"
        result["fragment_repair_parseable_after_cleanup"] = False
        if candidate_source == "ocr":
            result["ocr_cleanup_reason"] = result["ocr_cleanup_reason"] or "cleanup_candidate_not_parseable"
        return result

    @staticmethod
    def _select_single_retry_attempt_block(
        raw_text: str,
        blocks: list[dict[str, Any]],
        *,
        outside_text: str,
        candidate_source: str = "",
    ) -> dict[str, Any]:
        if len(blocks) <= 1:
            return BrowserChatGPTTransport._default_retry_block_selection()
        parseable_blocks = [block for block in blocks if block["parseable"]]
        if len(parseable_blocks) == 1:
            selected = parseable_blocks[0]
            return {
                **BrowserChatGPTTransport._default_retry_block_selection(),
                "recovered": True,
                "selected_block_index": int(selected["index"]),
                "selection_reason": "single_parseable_retry_block",
                "selected_text": str(selected["extracted"]),
                "relationship": "single_parseable_plus_fragments",
            }
        if len(parseable_blocks) > 1:
            canonical = {str(block["canonical_json"]) for block in parseable_blocks if str(block["canonical_json"]).strip()}
            if len(canonical) == 1:
                selected = parseable_blocks[-1]
                return {
                    **BrowserChatGPTTransport._default_retry_block_selection(),
                    "recovered": True,
                    "selected_block_index": int(selected["index"]),
                    "selection_reason": "duplicate_parseable_retry_blocks",
                    "selected_text": str(selected["extracted"]),
                    "relationship": "duplicate_parseable_blocks",
                }
            return {
                **BrowserChatGPTTransport._default_retry_block_selection(),
                "ambiguous": True,
                "selection_reason": "multiple_distinct_parseable_retry_blocks",
                "failure_reason": "ambiguous_multiple_retry_blocks",
                "relationship": "distinct_parseable_blocks",
            }
        fragmented_repair = BrowserChatGPTTransport._attempt_fragmented_retry_block_repair(
            blocks,
            candidate_source=candidate_source,
        )
        if fragmented_repair["fragment_repair_pattern_matched"]:
            return fragmented_repair
        block_kinds = {str(block["block_kind"]) for block in blocks}
        wrapper_recheck = BrowserChatGPTTransport._attempt_wrapper_only_retry_recheck(
            raw_text,
            blocks,
            outside_text=outside_text,
        )
        if block_kinds and block_kinds <= {"wrapper_only_block", "marker_only_block"}:
            if wrapper_recheck["recovered"]:
                return wrapper_recheck
            return {
                **wrapper_recheck,
                "selection_reason": "wrapper_only_multi_block_fragments",
                "failure_reason": "wrapper_only_multi_block_retry_output",
                "relationship": "wrapper_only_blocks",
            }
        if any(
            str(block["block_kind"]) in {"fragment_without_end_marker", "truncated_non_json_fragment"}
            for block in blocks
        ):
            return {
                **BrowserChatGPTTransport._default_retry_block_selection(),
                "selection_reason": "fragmented_multi_block_fragments",
                "failure_reason": "fragmented_multi_block_retry_output",
                "relationship": "fragmented_blocks",
            }
        ranked = sorted(
            blocks,
            key=lambda block: (
                int(block["schema_hits"]),
                -abs(int(block["brace_balance"])),
                -abs(int(block["bracket_balance"])),
                int(block["length"]),
            ),
            reverse=True,
        )
        if len(ranked) >= 2:
            top = ranked[0]
            second = ranked[1]
            top_signature = (
                int(top["schema_hits"]),
                abs(int(top["brace_balance"])),
                abs(int(top["bracket_balance"])),
            )
            second_signature = (
                int(second["schema_hits"]),
                abs(int(second["brace_balance"])),
                abs(int(second["bracket_balance"])),
            )
            if top_signature == second_signature:
                return {
                    **BrowserChatGPTTransport._default_retry_block_selection(),
                    "ambiguous": True,
                    "selection_reason": "multiple_malformed_retry_blocks_tied",
                    "failure_reason": "ambiguous_multiple_retry_blocks",
                    "relationship": "malformed_blocks_tied",
                }
        return {
            **BrowserChatGPTTransport._default_retry_block_selection(),
            "selected_block_index": int(ranked[0]["index"]) if ranked else None,
            "selection_reason": "no_parseable_retry_block_selected",
            "failure_reason": "no_single_retry_block_selected",
            "relationship": "malformed_blocks_without_safe_winner",
        }

    def _assess_retry_attempt_response(self, raw_text: str, *, candidate_source: str) -> dict[str, Any]:
        cleaned = raw_text.strip()
        extracted = extract_structured_block(cleaned).strip() if cleaned else ""
        lowered = _normalize(cleaned)
        block_count = self._count_retry_attempt_blocks(cleaned) if cleaned else 0
        blocks = self._extract_retry_attempt_blocks(cleaned) if cleaned else []
        block_forensics = self._summarize_retry_attempt_blocks(blocks) if blocks else []
        outside_text = self._retry_attempt_outside_block_text(cleaned, extracted) if cleaned else ""
        wrapper_details = (
            self._analyze_wrapper_only_retry_blocks(cleaned, blocks, outside_text=outside_text)
            if blocks
            else self._default_retry_block_selection()
        )
        wrapper_recheck = (
            self._attempt_wrapper_only_retry_recheck(cleaned, blocks, outside_text=outside_text)
            if blocks
            else self._default_retry_block_selection()
        )
        prose_recovery = (
            self._attempt_retry_prose_embedded_block_recovery(
                cleaned,
                blocks,
                extracted=extracted,
                outside_text=outside_text,
            )
            if cleaned
            else self._default_retry_block_selection()
        )
        block_selection = (
            self._select_single_retry_attempt_block(
                cleaned,
                blocks,
                outside_text=outside_text,
                candidate_source=candidate_source,
            )
            if blocks
            else self._default_retry_block_selection()
        )
        parseable = bool(extracted) and looks_like_patch_plan_json(extracted)
        single_block_completion = (
            self._attempt_single_incomplete_retry_block_completion(cleaned, blocks, extracted=extracted)
            if cleaned and len(blocks) == 1 and not parseable
            else self._default_retry_block_selection()
        )
        exact_block_match = (
            re.fullmatch(r"\s*ASTER_PATCH_BEGIN\s*(.*?)\s*ASTER_PATCH_END\s*", cleaned, flags=re.DOTALL)
            if cleaned
            else None
        )
        exact_block_only = exact_block_match is not None and not outside_text
        recovered_single_block = bool(block_count > 1 and block_selection["recovered"])
        recovered_single_block_completion = bool(
            not recovered_single_block
            and single_block_completion["recovered"]
            and single_block_completion["selected_text"]
        )
        recovered_wrapper_payload = bool(
            not recovered_single_block
            and not recovered_single_block_completion
            and wrapper_recheck["recovered"]
            and wrapper_recheck["selected_text"]
            and block_count <= 1
        )
        json_cleanup = (
            self._attempt_retry_json_cleanup(cleaned, blocks, extracted=extracted)
            if (
                cleaned
                and block_count <= 1
                and not parseable
                and not single_block_completion["single_block_completion_attempted"]
                and bool(extracted)
                and "{" in extracted
            )
            else self._default_retry_block_selection()
        )
        recovered_json_cleanup = bool(
            not recovered_single_block
            and not recovered_single_block_completion
            and not recovered_wrapper_payload
            and json_cleanup["recovered"]
            and json_cleanup["selected_text"]
        )
        recovered_prose_block = bool(
            not recovered_single_block
            and not recovered_single_block_completion
            and not recovered_wrapper_payload
            and not recovered_json_cleanup
            and prose_recovery["recovered"]
            and prose_recovery["selected_text"]
        )
        effective_parseable = (
            parseable
            or recovered_single_block
            or recovered_single_block_completion
            or recovered_wrapper_payload
            or recovered_json_cleanup
            or recovered_prose_block
        )
        json_object_count = 1 if effective_parseable else 0
        prose_contamination = bool(outside_text) or self._retry_seed_contaminated(cleaned) or "retry mode for browser output" in lowered
        marker_present = bool(
            "aster_patch_begin" in lowered or "aster patch begin" in lowered or "aster_patch_end" in lowered or "aster patch end" in lowered
        )
        wrapper_only = bool(
            marker_present
            and not parseable
            and not any(token in cleaned for token in ("{", '"summary"', '"operations"'))
        )
        json_like_extracted = bool(extracted) and any(
            token in extracted for token in ("{", '"summary"', '"operations"', '"notes"')
        )
        code_only = (
            not bool(extracted)
            and any(marker in lowered for marker in ("def ", "class ", "import ", "self.", "tkinter", "assert "))
        )
        prose_like_text = bool(cleaned) and not block_count and not marker_present and not json_like_extracted and not code_only
        prose_contamination = prose_contamination or prose_like_text
        structured_block_found = bool(extracted) or any(bool(block.get("extracted", "").strip()) for block in blocks)
        if not cleaned:
            failure_reason = "empty_retry_response"
        elif block_count > 1 and block_selection["recovered"]:
            failure_reason = ""
        elif recovered_wrapper_payload:
            failure_reason = ""
        elif recovered_prose_block:
            failure_reason = ""
        elif recovered_json_cleanup:
            failure_reason = ""
        elif recovered_single_block_completion:
            failure_reason = ""
        elif block_count > 1:
            failure_reason = str(block_selection["failure_reason"] or "multiple_retry_blocks")
        elif single_block_completion["single_block_completion_attempted"]:
            failure_reason = str(single_block_completion["failure_reason"] or "retry_block_not_safely_completable")
        elif parseable and exact_block_only and json_object_count == 1:
            failure_reason = ""
        elif parseable and not exact_block_only:
            failure_reason = "prose_contaminated_retry_response"
        elif wrapper_only:
            failure_reason = "wrapper_without_valid_json"
        elif json_like_extracted:
            failure_reason = str(json_cleanup["failure_reason"] or "malformed_json_inside_retry_block")
        elif code_only:
            failure_reason = "code_only_retry_response"
        elif prose_contamination:
            failure_reason = "prose_contaminated_retry_response"
        else:
            failure_reason = "malformed_structured_retry_response"
        return {
            "candidate_source": candidate_source,
            "structured_block_found": structured_block_found,
            "parseable": effective_parseable,
            "prose_contamination": prose_contamination,
            "wrapper_only": wrapper_only,
            "block_count": block_count,
            "exact_block_only": exact_block_only,
            "extra_text_detected": bool(outside_text),
            "json_object_count": json_object_count,
            "failure_reason": failure_reason,
            "selected_text": (
                str(block_selection["selected_text"])
                if recovered_single_block
                else str(single_block_completion["selected_text"])
                if recovered_single_block_completion
                else str(wrapper_recheck["selected_text"])
                if recovered_wrapper_payload
                else str(prose_recovery["selected_text"])
                if recovered_prose_block
                else str(json_cleanup["selected_text"])
                if recovered_json_cleanup
                else extracted if parseable and exact_block_only and json_object_count == 1 else ""
            ),
            "acceptance_tier": (
                "retry_structured_wrapper_recheck"
                if recovered_wrapper_payload
                else
                "retry_structured_embedded_block"
                if recovered_prose_block
                else
                "retry_structured_json_cleanup"
                if recovered_json_cleanup
                else
                "retry_structured_single_block_completion"
                if recovered_single_block_completion
                else
                "retry_structured_repaired_fragment"
                if recovered_single_block and bool(block_selection["fragment_repair_succeeded"])
                else
                "retry_structured_recovered_single_block"
                if recovered_single_block
                else "retry_structured_exact"
                if parseable and exact_block_only and json_object_count == 1
                else "blocked_or_ambiguous"
            ),
            "selected_block_index": (
                prose_recovery["selected_block_index"]
                if recovered_prose_block
                else json_cleanup["selected_block_index"]
                if recovered_json_cleanup
                else wrapper_recheck["selected_block_index"]
                if recovered_wrapper_payload
                else block_selection["selected_block_index"]
            ),
            "block_selection_reason": (
                str(prose_recovery["selection_reason"])
                if recovered_prose_block
                else str(json_cleanup["selection_reason"])
                if recovered_json_cleanup
                else str(single_block_completion["selection_reason"])
                if recovered_single_block_completion
                else str(wrapper_recheck["selection_reason"])
                if recovered_wrapper_payload
                else str(block_selection["selection_reason"])
            ),
            "multiple_blocks_ambiguous": bool(block_count > 1 and block_selection["ambiguous"]),
            "multiple_blocks_recovered": recovered_single_block,
            "block_relationship": (
                str(prose_recovery["relationship"])
                if recovered_prose_block
                else str(json_cleanup["relationship"])
                if recovered_json_cleanup
                else str(single_block_completion["relationship"])
                if recovered_single_block_completion
                else str(wrapper_recheck["relationship"])
                if recovered_wrapper_payload
                else str(block_selection["relationship"])
            ),
            "block_forensics": block_forensics,
            "wrapper_only_block_count": int(wrapper_details["wrapper_only_block_count"]),
            "wrapper_only_payload_lengths": list(wrapper_details["wrapper_only_payload_lengths"]),
            "wrapper_only_has_internal_text": bool(wrapper_details["wrapper_only_has_internal_text"]),
            "wrapper_only_noise_detected": bool(wrapper_details["wrapper_only_noise_detected"]),
            "wrapper_only_boundary_suspected": bool(wrapper_details["wrapper_only_boundary_suspected"]),
            "wrapper_recheck_attempted": bool(wrapper_recheck["wrapper_recheck_attempted"]),
            "wrapper_recheck_found_payload": bool(wrapper_recheck["wrapper_recheck_found_payload"]),
            "wrapper_recheck_reason": str(wrapper_recheck["wrapper_recheck_reason"]),
            "fragment_repair_pattern_matched": bool(block_selection["fragment_repair_pattern_matched"]),
            "fragment_repair_attempted": bool(block_selection["fragment_repair_attempted"]),
            "fragment_repair_succeeded": bool(block_selection["fragment_repair_succeeded"]),
            "fragment_repair_reason": str(block_selection["fragment_repair_reason"]),
            "fragment_repair_parseable_after_cleanup": bool(block_selection["fragment_repair_parseable_after_cleanup"]),
            "repaired_from_block_index": block_selection["repaired_from_block_index"],
            "discarded_wrapper_only_block_index": block_selection["discarded_wrapper_only_block_index"],
            "json_cleanup_attempted": bool(json_cleanup["json_cleanup_attempted"]),
            "json_cleanup_succeeded": bool(json_cleanup["json_cleanup_succeeded"]),
            "json_cleanup_reason": str(json_cleanup["json_cleanup_reason"]),
            "json_cleanup_changed": bool(json_cleanup["json_cleanup_changed"]),
            "prose_recovery_attempted": bool(prose_recovery["prose_recovery_attempted"]),
            "prose_recovery_succeeded": bool(prose_recovery["prose_recovery_succeeded"]),
            "prose_recovery_reason": str(prose_recovery["prose_recovery_reason"]),
            "ocr_cleanup_attempted": bool(block_selection["ocr_cleanup_attempted"]),
            "ocr_cleanup_succeeded": bool(block_selection["ocr_cleanup_succeeded"]),
            "ocr_cleanup_reason": str(block_selection["ocr_cleanup_reason"]),
            "ocr_defect_types": list(block_selection["ocr_defect_types"]),
            "single_block_completion_attempted": bool(single_block_completion["single_block_completion_attempted"]),
            "single_block_completion_succeeded": bool(single_block_completion["single_block_completion_succeeded"]),
            "single_block_completion_reason": str(single_block_completion["single_block_completion_reason"]),
            "single_block_completion_defect_types": list(single_block_completion["single_block_completion_defect_types"]),
            "single_block_completion_closure_added": str(single_block_completion["single_block_completion_closure_added"]),
            "single_block_semantically_incomplete": bool(single_block_completion["single_block_semantically_incomplete"]),
            "single_block_parseable_after_completion": bool(single_block_completion["single_block_parseable_after_completion"]),
        }

    @staticmethod
    def _apply_retry_attempt_assessment(diagnostics: dict[str, Any], assessment: dict[str, Any]) -> None:
        diagnostics["retry_attempt_acceptance_tier"] = str(assessment["acceptance_tier"])
        diagnostics["retry_attempt_failure_reason"] = str(assessment["failure_reason"])
        diagnostics["retry_attempt_structured_block_found"] = bool(assessment["structured_block_found"])
        diagnostics["retry_attempt_parseable"] = bool(assessment["parseable"])
        diagnostics["retry_attempt_prose_contamination"] = bool(assessment["prose_contamination"])
        diagnostics["retry_attempt_wrapper_only"] = bool(assessment["wrapper_only"])
        diagnostics["retry_attempt_block_count"] = int(assessment["block_count"])
        diagnostics["retry_attempt_exact_block_only"] = bool(assessment["exact_block_only"])
        diagnostics["retry_attempt_extra_text_detected"] = bool(assessment["extra_text_detected"])
        diagnostics["retry_attempt_json_object_count"] = int(assessment["json_object_count"])
        diagnostics["retry_attempt_selected_block_index"] = assessment["selected_block_index"]
        diagnostics["retry_attempt_block_selection_reason"] = str(assessment["block_selection_reason"])
        diagnostics["retry_attempt_multiple_blocks_ambiguous"] = bool(assessment["multiple_blocks_ambiguous"])
        diagnostics["retry_attempt_multiple_blocks_recovered"] = bool(assessment["multiple_blocks_recovered"])
        diagnostics["retry_attempt_block_relationship"] = str(assessment["block_relationship"])
        diagnostics["retry_attempt_block_forensics"] = list(assessment["block_forensics"])
        diagnostics["retry_attempt_wrapper_only_block_count"] = int(assessment["wrapper_only_block_count"])
        diagnostics["retry_attempt_wrapper_only_payload_lengths"] = list(assessment["wrapper_only_payload_lengths"])
        diagnostics["retry_attempt_wrapper_only_has_internal_text"] = bool(assessment["wrapper_only_has_internal_text"])
        diagnostics["retry_attempt_wrapper_only_noise_detected"] = bool(assessment["wrapper_only_noise_detected"])
        diagnostics["retry_attempt_wrapper_only_boundary_suspected"] = bool(assessment["wrapper_only_boundary_suspected"])
        diagnostics["retry_wrapper_recheck_attempted"] = bool(assessment["wrapper_recheck_attempted"])
        diagnostics["retry_wrapper_recheck_found_payload"] = bool(assessment["wrapper_recheck_found_payload"])
        diagnostics["retry_wrapper_recheck_reason"] = str(assessment["wrapper_recheck_reason"])
        diagnostics["retry_attempt_fragment_repair_pattern_matched"] = bool(assessment["fragment_repair_pattern_matched"])
        diagnostics["retry_attempt_fragment_repair_attempted"] = bool(assessment["fragment_repair_attempted"])
        diagnostics["retry_attempt_fragment_repair_succeeded"] = bool(assessment["fragment_repair_succeeded"])
        diagnostics["retry_attempt_fragment_repair_reason"] = str(assessment["fragment_repair_reason"])
        diagnostics["retry_attempt_fragment_repair_parseable_after_cleanup"] = bool(
            assessment["fragment_repair_parseable_after_cleanup"]
        )
        diagnostics["retry_attempt_repaired_from_block_index"] = assessment["repaired_from_block_index"]
        diagnostics["retry_attempt_discarded_wrapper_only_block_index"] = assessment["discarded_wrapper_only_block_index"]
        diagnostics["retry_attempt_json_cleanup_attempted"] = bool(assessment["json_cleanup_attempted"])
        diagnostics["retry_attempt_json_cleanup_succeeded"] = bool(assessment["json_cleanup_succeeded"])
        diagnostics["retry_attempt_json_cleanup_reason"] = str(assessment["json_cleanup_reason"])
        diagnostics["retry_attempt_json_cleanup_changed"] = bool(assessment["json_cleanup_changed"])
        diagnostics["retry_attempt_prose_recovery_attempted"] = bool(assessment["prose_recovery_attempted"])
        diagnostics["retry_attempt_prose_recovery_succeeded"] = bool(assessment["prose_recovery_succeeded"])
        diagnostics["retry_attempt_prose_recovery_reason"] = str(assessment["prose_recovery_reason"])
        diagnostics["retry_attempt_ocr_cleanup_attempted"] = bool(assessment["ocr_cleanup_attempted"])
        diagnostics["retry_attempt_ocr_cleanup_succeeded"] = bool(assessment["ocr_cleanup_succeeded"])
        diagnostics["retry_attempt_ocr_cleanup_reason"] = str(assessment["ocr_cleanup_reason"])
        diagnostics["retry_attempt_ocr_defect_types"] = list(assessment["ocr_defect_types"])
        diagnostics["retry_attempt_single_block_completion_attempted"] = bool(
            assessment["single_block_completion_attempted"]
        )
        diagnostics["retry_attempt_single_block_completion_succeeded"] = bool(
            assessment["single_block_completion_succeeded"]
        )
        diagnostics["retry_attempt_single_block_completion_reason"] = str(
            assessment["single_block_completion_reason"]
        )
        diagnostics["retry_attempt_single_block_completion_defect_types"] = list(
            assessment["single_block_completion_defect_types"]
        )
        diagnostics["retry_attempt_single_block_completion_closure_added"] = str(
            assessment["single_block_completion_closure_added"]
        )
        diagnostics["retry_attempt_single_block_semantically_incomplete"] = bool(
            assessment["single_block_semantically_incomplete"]
        )
        diagnostics["retry_attempt_single_block_parseable_after_completion"] = bool(
            assessment["single_block_parseable_after_completion"]
        )

    def _build_structured_reply_candidate(
        self,
        *,
        source: str,
        text: str,
        score: float,
        salvage_allowed: bool,
        region_trusted: bool,
        region_confidence: float | None,
        region_reason: str,
        observation_count: int,
    ) -> _StructuredReplyCandidate | None:
        cleaned = text.strip()
        if not cleaned:
            return None
        parsed = extract_structured_block(cleaned)
        if not parsed.strip():
            return None
        normalized = " ".join(cleaned.lower().split())
        structured_hint = (
            looks_like_patch_plan_json(parsed)
            or "aster patch begin" in normalized
            or "aster_patch_begin" in normalized
            or ('"operations"' in parsed and "{" in parsed)
            or ('"summary"' in parsed and "{" in parsed)
        )
        if not structured_hint:
            return None
        retry_safe, retry_seed_validity_reason = self._classify_retry_seed_text(
            parsed,
            region_trusted=region_trusted,
            salvage_allowed=salvage_allowed,
        )
        return _StructuredReplyCandidate(
            source=source,
            raw_text=cleaned,
            parsed_text=parsed,
            score=score,
            parseable=looks_like_patch_plan_json(parsed),
            salvage_allowed=salvage_allowed,
            retry_safe=retry_safe,
            retry_seed_validity_reason=retry_seed_validity_reason,
            region_trusted=region_trusted,
            region_confidence=region_confidence,
            region_reason=region_reason,
            observation_count=observation_count,
        )

    def _remember_reply_capture_candidate(
        self,
        snapshot: _ReplyCaptureSnapshot,
        *,
        source: str,
        text: str,
        score: float,
        salvage_allowed: bool,
        observation_counts: dict[str, int],
        region_trusted: bool = True,
        region_confidence: float | None = None,
        region_reason: str = "",
    ) -> _StructuredReplyCandidate | None:
        parsed = extract_structured_block(text)
        if not parsed.strip():
            return None
        observation_count = observation_counts.get(parsed, 0) + 1
        observation_counts[parsed] = observation_count
        candidate = self._build_structured_reply_candidate(
            source=source,
            text=text,
            score=score,
            salvage_allowed=salvage_allowed,
            region_trusted=region_trusted,
            region_confidence=region_confidence,
            region_reason=region_reason,
            observation_count=observation_count,
        )
        if candidate is None:
            return None
        if snapshot.best_structured is None or self._structured_candidate_priority(candidate) > self._structured_candidate_priority(snapshot.best_structured):
            snapshot.best_structured = candidate
        if candidate.salvage_allowed and candidate.region_trusted and (
            snapshot.best_salvageable is None
            or self._structured_candidate_priority(candidate) > self._structured_candidate_priority(snapshot.best_salvageable)
        ):
            snapshot.best_salvageable = candidate
        if candidate.retry_safe and (
            snapshot.best_retry_safe is None
            or self._structured_candidate_priority(candidate) > self._structured_candidate_priority(snapshot.best_retry_safe)
        ):
            snapshot.best_retry_safe = candidate
        return candidate

    def _finalize_captured_reply(self, reply: str, *, retry_attempt: bool = False) -> tuple[str, dict[str, Any]]:
        parsed = extract_structured_block(reply)
        snapshot = self._last_reply_capture_snapshot
        best_structured = snapshot.best_structured
        best_salvageable = snapshot.best_salvageable
        best_retry_safe = snapshot.best_retry_safe
        retry_assessment = (
            self._assess_retry_attempt_response(reply, candidate_source="final_reply")
            if retry_attempt
            else None
        )
        final_reply_retry_safe, final_reply_retry_reason = self._classify_retry_seed_text(
            parsed,
            region_trusted=True,
            salvage_allowed=True,
        )
        retry_fragment_source_meta = dict(self._last_retry_fragment_source_meta)
        diagnostics: dict[str, Any] = {
            "best_structured_candidate_length": len(best_structured.parsed_text) if best_structured is not None else 0,
            "best_structured_candidate_source": best_structured.source if best_structured is not None else None,
            "best_structured_candidate_parseable": best_structured.parseable if best_structured is not None else False,
            "best_structured_candidate_observations": best_structured.observation_count if best_structured is not None else 0,
            "best_structured_candidate_region_trusted": best_structured.region_trusted if best_structured is not None else False,
            "best_structured_candidate_region_confidence": best_structured.region_confidence if best_structured is not None else None,
            "best_structured_candidate_region_reason": best_structured.region_reason if best_structured is not None else "",
            "best_structured_candidate_retry_safe": best_structured.retry_safe if best_structured is not None else False,
            "best_structured_candidate_retry_seed_validity_reason": (
                best_structured.retry_seed_validity_reason if best_structured is not None else ""
            ),
            "best_salvageable_candidate_length": len(best_salvageable.parsed_text) if best_salvageable is not None else 0,
            "best_salvageable_candidate_source": best_salvageable.source if best_salvageable is not None else None,
            "best_salvageable_candidate_region_trusted": best_salvageable.region_trusted if best_salvageable is not None else False,
            "best_salvageable_candidate_region_confidence": best_salvageable.region_confidence if best_salvageable is not None else None,
            "best_salvageable_candidate_region_reason": best_salvageable.region_reason if best_salvageable is not None else "",
            "best_salvageable_candidate_retry_safe": best_salvageable.retry_safe if best_salvageable is not None else False,
            "best_salvageable_candidate_retry_seed_validity_reason": (
                best_salvageable.retry_seed_validity_reason if best_salvageable is not None else ""
            ),
            "best_retry_safe_candidate_length": len(best_retry_safe.parsed_text) if best_retry_safe is not None else 0,
            "best_retry_safe_candidate_source": best_retry_safe.source if best_retry_safe is not None else None,
            "best_retry_safe_candidate_region_trusted": best_retry_safe.region_trusted if best_retry_safe is not None else False,
            "best_retry_safe_candidate_region_confidence": best_retry_safe.region_confidence if best_retry_safe is not None else None,
            "best_retry_safe_candidate_region_reason": best_retry_safe.region_reason if best_retry_safe is not None else "",
            "best_retry_safe_candidate_parseable": best_retry_safe.parseable if best_retry_safe is not None else False,
            "best_retry_safe_candidate_retry_seed_validity_reason": (
                best_retry_safe.retry_seed_validity_reason if best_retry_safe is not None else ""
            ),
            "retry_attempt_fragment_source_preference": retry_fragment_source_meta.get("fragment_source_preference", ""),
            "retry_attempt_fragment_source_chosen": retry_fragment_source_meta.get("fragment_source_chosen", ""),
            "retry_attempt_fragment_uia_available": retry_fragment_source_meta.get("fragment_uia_available", False),
            "retry_attempt_fragment_ocr_available": retry_fragment_source_meta.get("fragment_ocr_available", False),
            "salvage_attempted": False,
            "salvage_succeeded": False,
            "salvage_source": None,
            "salvage_preserved_for_diagnostics_only": False,
            "retry_seed_valid": False,
            "retry_seed_validity_reason": final_reply_retry_reason if parsed.strip() else "",
            "retry_seed_validity_reason_source": "final_reply" if parsed.strip() else "",
            "retry_attempt_capture_mode": "structured_block_first" if retry_attempt else "",
            "retry_attempt_acceptance_tier": (
                str(retry_assessment["acceptance_tier"]) if retry_assessment is not None else ""
            ),
            "retry_attempt_failure_reason": (
                str(retry_assessment["failure_reason"]) if retry_assessment is not None else ""
            ),
            "retry_attempt_structured_block_found": (
                bool(retry_assessment["structured_block_found"]) if retry_assessment is not None else False
            ),
            "retry_attempt_parseable": bool(retry_assessment["parseable"]) if retry_assessment is not None else False,
            "retry_attempt_prose_contamination": (
                bool(retry_assessment["prose_contamination"]) if retry_assessment is not None else False
            ),
            "retry_attempt_wrapper_only": bool(retry_assessment["wrapper_only"]) if retry_assessment is not None else False,
            "retry_attempt_block_count": int(retry_assessment["block_count"]) if retry_assessment is not None else 0,
            "retry_attempt_exact_block_only": (
                bool(retry_assessment["exact_block_only"]) if retry_assessment is not None else False
            ),
            "retry_attempt_extra_text_detected": (
                bool(retry_assessment["extra_text_detected"]) if retry_assessment is not None else False
            ),
            "retry_attempt_json_object_count": (
                int(retry_assessment["json_object_count"]) if retry_assessment is not None else 0
            ),
            "retry_attempt_selected_block_index": (
                retry_assessment["selected_block_index"] if retry_assessment is not None else None
            ),
            "retry_attempt_block_selection_reason": (
                str(retry_assessment["block_selection_reason"]) if retry_assessment is not None else ""
            ),
            "retry_attempt_multiple_blocks_ambiguous": (
                bool(retry_assessment["multiple_blocks_ambiguous"]) if retry_assessment is not None else False
            ),
            "retry_attempt_multiple_blocks_recovered": (
                bool(retry_assessment["multiple_blocks_recovered"]) if retry_assessment is not None else False
            ),
            "retry_attempt_block_relationship": (
                str(retry_assessment["block_relationship"]) if retry_assessment is not None else ""
            ),
            "retry_attempt_block_forensics": (
                list(retry_assessment["block_forensics"]) if retry_assessment is not None else []
            ),
            "retry_attempt_wrapper_only_block_count": (
                int(retry_assessment["wrapper_only_block_count"]) if retry_assessment is not None else 0
            ),
            "retry_attempt_wrapper_only_payload_lengths": (
                list(retry_assessment["wrapper_only_payload_lengths"]) if retry_assessment is not None else []
            ),
            "retry_attempt_wrapper_only_has_internal_text": (
                bool(retry_assessment["wrapper_only_has_internal_text"]) if retry_assessment is not None else False
            ),
            "retry_attempt_wrapper_only_noise_detected": (
                bool(retry_assessment["wrapper_only_noise_detected"]) if retry_assessment is not None else False
            ),
            "retry_attempt_wrapper_only_boundary_suspected": (
                bool(retry_assessment["wrapper_only_boundary_suspected"]) if retry_assessment is not None else False
            ),
            "retry_wrapper_recheck_attempted": (
                bool(retry_assessment["wrapper_recheck_attempted"]) if retry_assessment is not None else False
            ),
            "retry_wrapper_recheck_found_payload": (
                bool(retry_assessment["wrapper_recheck_found_payload"]) if retry_assessment is not None else False
            ),
            "retry_wrapper_recheck_reason": (
                str(retry_assessment["wrapper_recheck_reason"]) if retry_assessment is not None else ""
            ),
            "retry_attempt_fragment_repair_pattern_matched": (
                bool(retry_assessment["fragment_repair_pattern_matched"]) if retry_assessment is not None else False
            ),
            "retry_attempt_fragment_repair_attempted": (
                bool(retry_assessment["fragment_repair_attempted"]) if retry_assessment is not None else False
            ),
            "retry_attempt_fragment_repair_succeeded": (
                bool(retry_assessment["fragment_repair_succeeded"]) if retry_assessment is not None else False
            ),
            "retry_attempt_fragment_repair_reason": (
                str(retry_assessment["fragment_repair_reason"]) if retry_assessment is not None else ""
            ),
            "retry_attempt_repaired_from_block_index": (
                retry_assessment["repaired_from_block_index"] if retry_assessment is not None else None
            ),
            "retry_attempt_discarded_wrapper_only_block_index": (
                retry_assessment["discarded_wrapper_only_block_index"] if retry_assessment is not None else None
            ),
            "retry_attempt_json_cleanup_attempted": (
                bool(retry_assessment["json_cleanup_attempted"]) if retry_assessment is not None else False
            ),
            "retry_attempt_json_cleanup_succeeded": (
                bool(retry_assessment["json_cleanup_succeeded"]) if retry_assessment is not None else False
            ),
            "retry_attempt_json_cleanup_reason": (
                str(retry_assessment["json_cleanup_reason"]) if retry_assessment is not None else ""
            ),
            "retry_attempt_json_cleanup_changed": (
                bool(retry_assessment["json_cleanup_changed"]) if retry_assessment is not None else False
            ),
            "retry_attempt_prose_recovery_attempted": (
                bool(retry_assessment["prose_recovery_attempted"]) if retry_assessment is not None else False
            ),
            "retry_attempt_prose_recovery_succeeded": (
                bool(retry_assessment["prose_recovery_succeeded"]) if retry_assessment is not None else False
            ),
            "retry_attempt_prose_recovery_reason": (
                str(retry_assessment["prose_recovery_reason"]) if retry_assessment is not None else ""
            ),
            "retry_attempt_single_block_completion_attempted": (
                bool(retry_assessment["single_block_completion_attempted"]) if retry_assessment is not None else False
            ),
            "retry_attempt_single_block_completion_succeeded": (
                bool(retry_assessment["single_block_completion_succeeded"]) if retry_assessment is not None else False
            ),
            "retry_attempt_single_block_completion_reason": (
                str(retry_assessment["single_block_completion_reason"]) if retry_assessment is not None else ""
            ),
            "retry_attempt_single_block_completion_defect_types": (
                list(retry_assessment["single_block_completion_defect_types"]) if retry_assessment is not None else []
            ),
            "retry_attempt_single_block_completion_closure_added": (
                str(retry_assessment["single_block_completion_closure_added"]) if retry_assessment is not None else ""
            ),
            "retry_attempt_single_block_semantically_incomplete": (
                bool(retry_assessment["single_block_semantically_incomplete"]) if retry_assessment is not None else False
            ),
            "retry_attempt_single_block_parseable_after_completion": (
                bool(retry_assessment["single_block_parseable_after_completion"]) if retry_assessment is not None else False
            ),
            "final_capture_failure_reason": "",
        }
        retry_assessment_selected = bool(retry_assessment is not None and retry_assessment["selected_text"])
        if retry_assessment_selected:
            diagnostics["retry_seed_valid"] = True
            diagnostics["retry_seed_validity_reason"] = "parseable_structured_json"
            diagnostics["retry_seed_validity_reason_source"] = "final_reply"
            return str(retry_assessment["selected_text"]), diagnostics
        if parsed.strip() and final_reply_retry_safe and not retry_attempt:
            diagnostics["retry_seed_valid"] = True
            return parsed, diagnostics

        diagnostics["salvage_attempted"] = best_structured is not None
        retry_safe_assessment = (
            self._assess_retry_attempt_response(best_retry_safe.raw_text, candidate_source=best_retry_safe.source)
            if retry_attempt and best_retry_safe is not None and best_retry_safe.parsed_text.strip()
            else None
        )
        if best_retry_safe is not None and best_retry_safe.parsed_text.strip() and (
            not retry_attempt or (retry_safe_assessment is not None and retry_safe_assessment["selected_text"])
        ):
            if retry_safe_assessment is not None:
                self._apply_retry_attempt_assessment(diagnostics, retry_safe_assessment)
            diagnostics["salvage_succeeded"] = True
            diagnostics["salvage_source"] = best_retry_safe.source
            diagnostics["retry_seed_valid"] = True
            diagnostics["retry_seed_validity_reason"] = best_retry_safe.retry_seed_validity_reason
            diagnostics["retry_seed_validity_reason_source"] = "best_retry_safe_candidate"
            diagnostics["final_capture_failure_reason"] = "structured_block_seen_but_lost"
            if retry_safe_assessment is not None and retry_safe_assessment["selected_text"]:
                return str(retry_safe_assessment["selected_text"]), diagnostics
            return best_retry_safe.parsed_text, diagnostics
        if best_salvageable is not None and best_salvageable.parsed_text.strip():
            salvage_assessment = (
                self._assess_retry_attempt_response(best_salvageable.raw_text, candidate_source=best_salvageable.source)
                if retry_attempt
                else None
            )
            if salvage_assessment is not None:
                self._apply_retry_attempt_assessment(diagnostics, salvage_assessment)
            diagnostics["salvage_preserved_for_diagnostics_only"] = True
            diagnostics["retry_seed_validity_reason"] = best_salvageable.retry_seed_validity_reason
            diagnostics["retry_seed_validity_reason_source"] = "best_salvageable_candidate"
            diagnostics["final_capture_failure_reason"] = "structured_block_seen_but_not_retry_safe"
            return "", diagnostics

        if best_structured is None:
            diagnostics["final_capture_failure_reason"] = "no_structured_block_seen"
        elif not best_structured.parseable:
            diagnostics["final_capture_failure_reason"] = "structured_block_seen_but_not_parseable"
        elif best_salvageable is None:
            diagnostics["final_capture_failure_reason"] = (
                "structured_block_seen_but_region_trust_too_low"
                if not best_structured.region_trusted
                else "structured_block_seen_but_not_salvageable"
            )
        else:
            diagnostics["final_capture_failure_reason"] = "structured_block_seen_but_not_retry_safe"
        return "", diagnostics

    def _analyze_screen(self, target, lines=None, ui_state: dict[str, Any] | None = None) -> PageClassification:
        if lines is None:
            image = self._capture.capture_region(target.left, target.top, target.width, target.height)
            lines = self._ocr.extract(image)
        state = ui_state or self._ui_state(target)
        classification = classify_page(lines, state)
        self._last_page_classification = classification
        return classification

    @staticmethod
    def _screen_analysis_payload(analysis: PageClassification) -> dict[str, Any]:
        return {
            "ready_score": round(analysis.ready_score, 1),
            "score_components": {key: round(value, 1) for key, value in analysis.score_components.items()},
            "chatgpt_hint_hits": list(analysis.chatgpt_hint_hits),
            "chatgpt_surface_hits": list(analysis.chatgpt_surface_hits),
            "composer_hint_hits": list(analysis.composer_hint_hits),
            "wrong_page_penalties": list(analysis.wrong_page_penalties),
            "loading_penalties": list(analysis.loading_penalties),
            "ui_state_bonuses": list(analysis.ui_state_bonuses),
            "ui_state_penalties": list(analysis.ui_state_penalties),
            "actionable_control_gaps": list(analysis.actionable_control_gaps),
            "send_button_absence_reason": analysis.send_button_absence_reason,
            "idle_composer_source": analysis.idle_composer_source,
            "missing_readiness_signals": list(analysis.missing_readiness_signals[:5]),
            "page_kind": analysis.page_kind,
            "looks_like_chatgpt": analysis.looks_like_chatgpt,
            "composer_visible": analysis.composer_visible,
            "likely_idle_composer": analysis.likely_idle_composer,
            "send_button_present": analysis.send_button_present,
            "send_button_enabled": analysis.send_button_enabled,
            "show_in_text_field_present": analysis.show_in_text_field_present,
            "stop_streaming_present": analysis.stop_streaming_present,
            "wrong_page_signals_present": analysis.wrong_page_signals_present,
            "likely_fresh_chat": analysis.likely_fresh_chat,
            "likely_existing_chat": analysis.likely_existing_chat,
            "composer_ready": analysis.composer_ready,
            "likely_wrong_page": analysis.likely_wrong_page,
            "loading_detected": analysis.loading_detected,
            "signals": list(analysis.signals[:8]),
        }

    @staticmethod
    def _page_readiness_wrong_page_hits(analysis: PageClassification) -> list[str]:
        visible_text = analysis.visible_text
        window_title = _normalize(str(analysis.ui_state.get("window_title", "")))
        return [
            marker
            for marker in PAGE_READINESS_WRONG_PAGE_MARKERS
            if marker in visible_text or marker in window_title
        ]

    @classmethod
    def _page_readiness_failure_details(cls, analysis: PageClassification) -> dict[str, Any]:
        title = str(analysis.ui_state.get("window_title", "") or "")
        normalized_title = _normalize(title)
        wrong_page_hits = cls._page_readiness_wrong_page_hits(analysis)
        likely_wrong_window = (
            not analysis.looks_like_chatgpt
            and not analysis.composer_visible
            and not analysis.send_button_present
            and not analysis.stop_streaming_present
            and not analysis.show_in_text_field_present
            and "chatgpt" not in normalized_title
        )

        if wrong_page_hits and not analysis.looks_like_chatgpt:
            return {
                "code": "wrong_page_detected",
                "reason": f"Wrong-page markers detected: {', '.join(wrong_page_hits[:2])}.",
                "wrong_page_hits": wrong_page_hits,
                "loading_detected": analysis.loading_detected,
                "missing_signals": list(analysis.missing_readiness_signals[:5]),
            }
        if analysis.wrong_page_signals_present or analysis.likely_wrong_page:
            return {
                "code": "wrong_page_detected",
                "reason": "Wrong-page signals were stronger than ChatGPT readiness signals.",
                "wrong_page_hits": wrong_page_hits,
                "loading_detected": analysis.loading_detected,
                "missing_signals": list(analysis.missing_readiness_signals[:5]),
            }
        if analysis.loading_detected:
            return {
                "code": "still_loading",
                "reason": "The attached page still appears to be loading.",
                "wrong_page_hits": wrong_page_hits,
                "loading_detected": analysis.loading_detected,
                "missing_signals": list(analysis.missing_readiness_signals[:5]),
            }
        if likely_wrong_window:
            return {
                "code": "likely_wrong_window",
                "reason": "The attached window did not show ChatGPT labels or composer controls.",
                "wrong_page_hits": wrong_page_hits,
                "loading_detected": analysis.loading_detected,
                "missing_signals": list(analysis.missing_readiness_signals[:5]),
            }
        if (
            analysis.looks_like_chatgpt
            and not analysis.composer_visible
            and not analysis.send_button_present
            and not analysis.stop_streaming_present
        ):
            return {
                "code": "composer_missing",
                "reason": "ChatGPT signals were present, but the composer was not visible.",
                "wrong_page_hits": wrong_page_hits,
                "loading_detected": analysis.loading_detected,
                "missing_signals": list(analysis.missing_readiness_signals[:5]),
            }
        missing = list(analysis.missing_readiness_signals[:3])
        missing_suffix = f" Missing strongest signals: {', '.join(missing)}." if missing else ""
        control_suffix = (
            f" {analysis.send_button_absence_reason}"
            if analysis.send_button_absence_reason
            else ""
        )
        idle_suffix = (
            " The page still resembles an idle ChatGPT composer state, so this may reflect a UIA control-detection gap."
            if analysis.likely_idle_composer
            else ""
        )
        return {
            "code": "low_readiness_score",
            "reason": (
                "ChatGPT readiness stayed below the acceptance threshold "
                f"({round(analysis.ready_score, 1)} < {PAGE_READINESS_SCORE_THRESHOLD:.1f})."
                f"{missing_suffix}"
                f"{control_suffix}"
                f"{idle_suffix}"
            ),
            "wrong_page_hits": wrong_page_hits,
            "loading_detected": analysis.loading_detected,
            "missing_signals": list(analysis.missing_readiness_signals[:5]),
        }

    def _page_readiness_log_payload(
        self,
        analysis: PageClassification,
        failure: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        details = failure or self._page_readiness_failure_details(analysis)
        payload = {
            "failure_code": details["code"],
            "failure_reason": details["reason"],
            "window_title": str(analysis.ui_state.get("window_title", "") or ""),
            "ready_score": round(analysis.ready_score, 1),
            "score_components": {key: round(value, 1) for key, value in analysis.score_components.items()},
            "chatgpt_hint_hits": list(analysis.chatgpt_hint_hits),
            "chatgpt_surface_hits": list(analysis.chatgpt_surface_hits),
            "composer_hint_hits": list(analysis.composer_hint_hits),
            "composer_visible": analysis.composer_visible,
            "likely_idle_composer": analysis.likely_idle_composer,
            "send_button_present": analysis.send_button_present,
            "stop_streaming_present": analysis.stop_streaming_present,
            "wrong_page_signals_present": analysis.wrong_page_signals_present,
            "loading_detected": details["loading_detected"],
            "wrong_page_hits": list(details.get("wrong_page_hits", [])[:3]),
            "wrong_page_penalties": list(analysis.wrong_page_penalties),
            "loading_penalties": list(analysis.loading_penalties),
            "ui_state_bonuses": list(analysis.ui_state_bonuses),
            "ui_state_penalties": list(analysis.ui_state_penalties),
            "actionable_control_gaps": list(analysis.actionable_control_gaps),
            "send_button_absence_reason": analysis.send_button_absence_reason,
            "idle_composer_source": analysis.idle_composer_source,
            "missing_readiness_signals": list(details.get("missing_signals", [])[:5]),
            "ocr_preview": list(analysis.ocr_preview),
            "page_classification": self._screen_analysis_payload(analysis),
            "ui_state": analysis.ui_state,
        }
        uia_diagnostics = analysis.ui_state.get("uia_control_diagnostics")
        if isinstance(uia_diagnostics, dict):
            payload["uia_control_diagnostics"] = uia_diagnostics
        return payload

    @staticmethod
    def _page_readiness_error_message(failure: dict[str, Any]) -> str:
        code = failure["code"]
        if code == "wrong_page_detected":
            return (
                "Attached browser window appears to be the wrong page. "
                f"{failure['reason']} Open ChatGPT in the active browser window."
            )
        if code == "composer_missing":
            return (
                "Attached ChatGPT window did not expose a visible composer. "
                "Open a ready ChatGPT page where the composer is visible."
            )
        if code == "still_loading":
            return (
                "Attached ChatGPT window still appears to be loading. "
                "Wait for the page to finish loading, then try again."
            )
        if code == "likely_wrong_window":
            return (
                "Attached browser window may not be the ChatGPT tab. "
                "Focus the ChatGPT window before running Aster."
            )
        return (
            "Attached browser window did not reach a usable ChatGPT page. "
            f"{failure['reason']}"
        )

    def _read_visible_reply_text(self, target) -> str:
        self._last_uia_reply_read_diagnostics = {
            "uia_read_failure_reason": "",
            "descendant_enumeration_guard_triggered": False,
        }
        try:
            window = Desktop(backend="uia").window(handle=target.handle)
        except Exception as exc:
            self._record_uia_reply_read_failure(target, stage="window_attach", exc=exc)
            return ""

        selected: list[tuple[int, int, str]] = []
        seen: set[str] = set()
        min_left = target.left + int(target.width * 0.18)
        min_top = target.top + 110
        max_bottom = target.top + target.height - 70

        try:
            descendants = iter(window.descendants())
        except Exception as exc:
            self._record_uia_reply_read_failure(target, stage="descendant_enumeration", exc=exc)
            return ""

        while True:
            try:
                ctrl = next(descendants)
            except StopIteration:
                break
            except Exception as exc:
                self._record_uia_reply_read_failure(
                    target,
                    stage="descendant_iteration",
                    exc=exc,
                    partial_matches=len(selected),
                )
                break
            try:
                control_type = ctrl.element_info.control_type
                if control_type not in {"Document", "Text", "Group"}:
                    continue
                text = " ".join((ctrl.window_text() or "").split())
                rect = ctrl.rectangle()
            except Exception:
                continue
            lowered = _normalize(text)
            if not text or lowered in seen:
                continue
            if rect.left < min_left or rect.top < min_top or rect.bottom > max_bottom:
                continue
            if any(noise in lowered for noise in BROWSER_REPLY_NOISE):
                continue
            if any(marker in lowered for marker in PROMPT_ECHO_MARKERS):
                continue
            selected.append((rect.top - target.top, rect.left - target.left, text))
            seen.add(lowered)

        if not selected:
            return ""
        selected.sort(key=lambda item: (item[0], item[1]))
        return "\n".join(text for _, _, text in selected)

    def _record_uia_reply_read_failure(
        self,
        target,
        *,
        stage: str,
        exc: Exception,
        partial_matches: int = 0,
    ) -> None:
        message = str(exc).strip()
        reason = type(exc).__name__ if not message or message == "None" else f"{type(exc).__name__}: {message}"
        diagnostics = {
            "uia_read_failure_reason": reason,
            "descendant_enumeration_guard_triggered": stage != "window_attach",
            "stage": stage,
            "partial_matches": partial_matches,
        }
        self._last_uia_reply_read_diagnostics = diagnostics
        self._log(
            "uia_reply_read_failure",
            {
                "window_title": getattr(target, "title", ""),
                "stage": stage,
                "partial_matches": partial_matches,
                **diagnostics,
            },
        )

    @staticmethod
    def _window_title_suggests_existing_thread(title: str) -> bool:
        return window_title_suggests_existing_chat(title)

    def _raise_for_browser_error(self, lines, ui_state: dict[str, Any], *, stage: str) -> None:
        visible_text = "\n".join(line.text for line in lines[:30])
        lowered = _normalize(visible_text)
        if any(hint in lowered for hint in INPUT_TOO_LARGE_HINTS):
            self._log(
                "browser_page_error",
                {
                    "stage": stage,
                    "error": "input_too_large",
                    "ui_state": ui_state,
                    "ocr_preview": [line.text[:120] for line in lines[:8]],
                },
            )
            self._activity(
                "browser_input_too_large",
                "ChatGPT rejected the browser prompt as too large.",
                "The browser composer has a smaller input limit than the API, so Aster must send less context.",
                status="error",
                details={"stage": stage},
            )
            raise RuntimeError(
                "ChatGPT browser rejected the prompt as too large. "
                "Aster needs a smaller browser prompt for this run."
            )

    @staticmethod
    def _looks_like_browser_url_text(text: str) -> bool:
        return looks_like_browser_url_text(text)

    def _log_window_state(self, target, lines, prompt: str, ui_state: dict[str, Any] | None = None) -> None:
        ui_state = ui_state or self._ui_state(target)
        analysis = self._analyze_screen(target, lines=lines, ui_state=ui_state)
        visible_text = "\n".join(line.text.lower() for line in lines[:12])
        prompt_words = [word for word in re.findall(r"[a-z0-9]{4,}", prompt.lower())[:8]]
        payload = {
            "ui_state": ui_state,
            "ocr_preview": [line.text[:120] for line in lines[:5]],
            "page_classification": self._screen_analysis_payload(analysis),
            "ocr_line_count": len(lines),
            "show_in_text_field_visible": "show in text field" in visible_text,
            "prompt_words_visible": [word for word in prompt_words if word in visible_text],
        }
        signature = str(payload)
        if signature != self._last_window_state:
            self._last_window_state = signature
            self._log("window_state", payload)

    def _ui_state(self, target, *, include_uia_diagnostics: bool = False) -> dict[str, Any]:
        state: dict[str, Any] = {
            "window_title": getattr(target, "title", ""),
            "send_prompt_present": False,
            "send_prompt_enabled": None,
            "show_in_text_field_present": False,
            "stop_streaming_present": False,
            "composer_edit_length": None,
            "composer_edit_preview": "",
        }
        try:
            window = Desktop(backend="uia").window(handle=target.handle)
            for ctrl in window.descendants(control_type="Button"):
                name = (ctrl.window_text() or "").strip().lower()
                if "send prompt" in name:
                    state["send_prompt_present"] = True
                    try:
                        state["send_prompt_enabled"] = ctrl.is_enabled()
                    except Exception:
                        state["send_prompt_enabled"] = "unknown"
                if "show in text field" in name:
                    state["show_in_text_field_present"] = True
                if any(hint in name for hint in STOP_STREAMING_HINTS):
                    state["stop_streaming_present"] = True
            composer = self._find_composer_edit(target)
            if composer is not None:
                try:
                    composer_text = (composer.window_text() or "").strip()
                    state["composer_edit_length"] = len(composer_text)
                    state["composer_edit_preview"] = composer_text[:120]
                except Exception:
                    state["composer_edit_length"] = "unknown"
                    state["composer_edit_preview"] = "unknown"
            if include_uia_diagnostics:
                state["uia_control_diagnostics"] = self._build_uia_control_diagnostics(target)
        except Exception as exc:
            state["inspection_error"] = str(exc)
        return state


def summarize_controls_near_composer_area(
    controls: list[dict[str, Any]],
    *,
    limit: int = 8,
) -> list[dict[str, Any]]:
    priority = {"Edit": 0, "Button": 1, "Text": 2, "Document": 3, "Group": 4}

    def sort_key(item: dict[str, Any]) -> tuple[float, float, float]:
        rect = item.get("rect") or (0, 0, 0, 0)
        return (
            float(priority.get(str(item.get("control_type", "")), 9)),
            float(rect[1]),
            float(rect[0]),
        )

    summary: list[dict[str, Any]] = []
    for item in sorted(controls, key=sort_key)[:limit]:
        entry = {
            "control_type": str(item.get("control_type", "")),
            "name": str(item.get("name", ""))[:120],
            "rect": _compact_rect(item.get("rect")),
        }
        enabled = item.get("enabled")
        if enabled is not None:
            entry["enabled"] = enabled
        summary.append(entry)
    return summary


def summarize_named_button_candidates(
    candidates: list[dict[str, Any]],
    *,
    limit: int = 6,
) -> dict[str, Any]:
    ordered = sorted(
        candidates,
        key=lambda item: (
            -float(item.get("match_score", float("-inf"))),
            0 if item.get("exact_match") else 1,
            -int(bool(item.get("enabled"))),
        ),
    )
    top_candidates: list[dict[str, Any]] = []
    send_like_names: list[str] = []

    for item in ordered[:limit]:
        name = str(item.get("name", ""))[:120]
        looks_send_like = bool(item.get("looks_send_like"))
        if looks_send_like and name and name not in send_like_names:
            send_like_names.append(name)
        top_candidates.append(
            {
                "name": name,
                "rect": _compact_rect(item.get("rect")),
                "enabled": item.get("enabled"),
                "match_score": round(float(item.get("match_score", 0.0)), 1),
                "exact_match": bool(item.get("exact_match")),
                "looks_send_like": looks_send_like,
            }
        )

    return {
        "top_candidates": top_candidates,
        "exact_match_found": any(bool(item.get("exact_match")) for item in candidates),
        "send_like_button_detected": any(bool(item.get("looks_send_like")) for item in candidates),
        "send_like_button_with_different_label": any(
            bool(item.get("looks_send_like")) and not bool(item.get("exact_match")) for item in candidates
        ),
        "send_like_button_names": send_like_names[:3],
    }


def summarize_edit_candidates(
    candidates: list[dict[str, Any]],
    *,
    threshold: float = 0.0,
    limit: int = 5,
) -> dict[str, Any]:
    ordered = sorted(candidates, key=lambda item: -float(item.get("score", float("-inf"))))
    top_candidates: list[dict[str, Any]] = []

    for item in ordered[:limit]:
        score = float(item.get("score", 0.0))
        top_candidates.append(
            {
                "name": str(item.get("name", ""))[:120],
                "rect": _compact_rect(item.get("rect")),
                "enabled": item.get("enabled"),
                "score": round(score, 1),
                "above_threshold": score >= threshold,
            }
        )

    best_score = float(ordered[0].get("score", float("-inf"))) if ordered else None
    return {
        "top_candidates": top_candidates,
        "best_score": round(best_score, 1) if best_score is not None else None,
        "composer_candidate_below_threshold": bool(ordered) and best_score is not None and best_score < threshold,
    }


def _compact_rect(rect: Any) -> dict[str, int]:
    if isinstance(rect, tuple) and len(rect) == 4:
        left, top, right, bottom = rect
        return {
            "left": int(left),
            "top": int(top),
            "right": int(right),
            "bottom": int(bottom),
        }
    return {"left": 0, "top": 0, "right": 0, "bottom": 0}


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


def _load_gui_dependencies():
    global pyautogui, Desktop
    if pyautogui is not None and Desktop is not None:
        return pyautogui, Desktop
    import pyautogui as pyautogui_module
    from pywinauto import Desktop as desktop_cls

    pyautogui = pyautogui_module
    Desktop = desktop_cls
    return pyautogui, Desktop
