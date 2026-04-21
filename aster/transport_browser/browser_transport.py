from __future__ import annotations

import atexit
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
    metadata: dict[str, str]


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
    ) -> BrowserResult:
        self._log(
            "generate_start",
            {
                "prompt_length": len(prompt),
                "timeout_sec": timeout_sec,
                "launch_timeout_sec": launch_timeout_sec,
            },
        )
        self._activity(
            "browser_generate_start",
            "Starting a browser ChatGPT run.",
            "Aster will attach to ChatGPT, place the structured prompt, send it, and wait for a parseable reply.",
            details={"prompt_length": len(prompt)},
        )
        prompt_anchor = build_turn_anchor(prompt)
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
                )
                parsed = extract_structured_block(reply)
                self._log(
                    "generate_reply_captured",
                    {
                        "reply_length": len(reply),
                        "parsed_length": len(parsed),
                        "reply_preview": reply[:300],
                    },
                )
                if not parsed.strip():
                    raise RuntimeError("Browser mode could not capture a final ChatGPT response")
                self._activity(
                    "browser_reply_ready",
                    "Captured a reply from ChatGPT and extracted the structured block.",
                    "Aster can now parse the response into exact file operations.",
                    status="success",
                    details={"reply_length": len(reply), "parsed_length": len(parsed)},
                )
                return BrowserResult(
                    raw_text=parsed,
                    metadata={"window_title": target.title},
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

            if send_attempt > 0 and self._reply_started(before_lines, lines, target, ui_state=ui_state):
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

            if send_attempt > 0 and self._reply_started(before_lines, lines, target, ui_state=ui_state):
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

    def _reply_started(self, before_lines, after_lines, target, ui_state: dict[str, Any] | None = None) -> bool:
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
        if self._click_named_button(target, "send prompt"):
            return
        self._focus_window(target)
        x = int(target.left + target.width * 0.95)
        y = int(target.top + target.height * 0.93)
        self._log("send_button_coordinate_click", {"x": x, "y": y})
        pyautogui.click(x, y)

    def _click_line(self, target, line) -> None:
        self._focus_window(target)
        x = int(target.left + line.center[0])
        y = int(target.top + line.center[1])
        pyautogui.click(x, y)

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
        try:
            window = Desktop(backend="uia").window(handle=target.handle)
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
                        if self._invoke_button(ctrl):
                            self._log("button_invoke", {"phrase": phrase, "name": name, "method": "invoke"})
                            return True
                        ctrl.click_input()
                        self._log("button_invoke", {"phrase": phrase, "name": name, "method": "click_input"})
                        return True
                    except Exception as exc:
                        self._log_uia_diagnostic(
                            "named_control_click_failed",
                            {
                                "phrase": phrase,
                                "name": name,
                                "control_types": list(control_types or ()),
                                "error": str(exc),
                                "uia_control_diagnostics": self._build_uia_control_diagnostics(target, phrase=phrase),
                            },
                        )
                        return False
        except Exception as exc:
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
        self._log_uia_diagnostic(
            "named_control_search_failed",
            {
                "phrase": phrase,
                "control_types": list(control_types or ()),
                "uia_control_diagnostics": self._build_uia_control_diagnostics(target, phrase=phrase),
            },
        )
        return False

    def _populate_prompt_directly(self, target, prompt: str, *, prompt_anchor=None) -> bool:
        composer = self._find_composer_edit(target)
        if composer is None:
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
                    method()
                    if self._wait_for_prompt_inserted(target, prompt, seconds=6.0, prompt_anchor=prompt_anchor):
                        self._log("composer_populated", {"method": method_name, "prompt_length": len(prompt)})
                        return True
                except Exception as exc:
                    self._log(
                        "composer_population_method_failed",
                        {"method": method_name, "error": str(exc), "ui_state": self._ui_state(target)},
                    )
        except Exception as exc:
            self._log("composer_population_failed", {"error": str(exc), "ui_state": self._ui_state(target)})
            return False
        self._log("composer_population_failed", {"ui_state": self._ui_state(target)})
        return False

    def _paste_prompt_with_click(self, target, prompt: str, click_point: tuple[int, int], prompt_anchor=None) -> bool:
        self._focus_window(target)
        pyautogui.click(click_point[0], click_point[1])
        time.sleep(0.25)
        self._clear_composer(target)
        self._paste_prompt_via_clipboard(target, prompt, click_point=click_point)
        return self._wait_for_prompt_inserted(target, prompt, seconds=6.0, prompt_anchor=prompt_anchor)

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
        composer = self._find_composer_line(lines)
        if composer is not None:
            self._click_line(target, composer)
            time.sleep(0.2)
            return
        self._focus_window(target)
        pyautogui.click(int(target.left + target.width * 0.55), int(target.top + target.height * 0.88))
        time.sleep(0.2)

    def _attempt_send(self, target, attempt_index: int) -> None:
        strategy = attempt_index % 5
        self._focus_window(target)
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

    def _wait_for_reply_start(self, target, before_lines, seconds: float) -> bool:
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            self._tick_runtime_log_heartbeat("wait_for_reply_start")
            image = self._capture.capture_region(target.left, target.top, target.width, target.height)
            lines = self._ocr.extract(image)
            ui_state = self._ui_state(target)
            if self._reply_started(before_lines, lines, target, ui_state=ui_state):
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
            recovery_handlers = build_recovery_handlers(
                rescan=lambda: None,
                refocus_composer=lambda: self._focus_composer(target, []),
                # Dedicated thread reopen/reattach is still pending, so conservative recovery
                # currently falls back to direct ChatGPT navigation inside the same window.
                reopen_thread=lambda: self._navigate_browser_to_chatgpt(target, chatgpt_url),
                reload_page=lambda: self._navigate_browser_to_chatgpt(target, chatgpt_url),
                reopen_chatgpt=lambda: self._navigate_browser_to_chatgpt(target, chatgpt_url),
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
    ) -> str:
        started_at = time.monotonic()
        deadline = time.monotonic() + timeout_sec
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

        while time.monotonic() < deadline:
            self._tick_runtime_log_heartbeat("capture_reply_text")
            time.sleep(2.0 if attempt_index else 1.2)
            ui_state = self._ui_state(target)
            last_ui_state = ui_state
            image = self._capture.capture_region(target.left, target.top, target.width, target.height)
            lines = self._ocr.extract(image)
            self._raise_for_browser_error(lines, ui_state, stage="reply_capture")
            ocr_text, uia_text = self._capture_visible_reply_sources(target, before_lines, prompt, lines=lines)
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
                if acceptance.accepted:
                    self._log(
                        "reply_candidate_accepted",
                        {
                            "source": source,
                            "score": score,
                            "length": len(candidate),
                            **self._reply_acceptance_log_payload(acceptance),
                        },
                    )
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
                        max_steps=10,
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
                            self._log(
                                "reply_scrolled_capture",
                                {
                                    "length": len(scrolled),
                                    "preview": scrolled[:240],
                                    "structured_completion_score": structured_completion_score(scrolled_progress),
                                    "structured_completion_progress": scrolled_progress,
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
                            if scroll_acceptance.accepted:
                                self._log(
                                    "reply_candidate_accepted",
                                    {
                                        "source": "scrolled",
                                        "score": score,
                                        "length": len(candidate),
                                        **self._reply_acceptance_log_payload(scroll_acceptance),
                                    },
                                )
                                return candidate
                    scanned_with_scroll = True
            previous_candidate_text = current_best[1] if current_best is not None else ""
            previous_ocr_text = ocr_text
            previous_uia_text = uia_text
            attempt_index += 1

        timeout_candidate = (best_source or "best", best_text, best_score) if best_text else None
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
        if last_wait_diagnostics is not None:
            self._log(
                "reply_wait_timeout",
                {
                    **last_wait_diagnostics,
                    **self._reply_acceptance_log_payload(timeout_acceptance),
                    "elapsed_sec": round(time.monotonic() - started_at, 1),
                    "best_overall_source": best_source or None,
                    "best_overall_score": round(best_score, 1) if best_text else None,
                    "best_overall_length": len(best_text),
                },
            )
        if timeout_candidate is not None and timeout_acceptance.accepted:
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
        if fallback_candidate is not None and fallback_acceptance.accepted:
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
            return fallback_block
        return ""

    def _capture_visible_reply_sources(self, target, before_lines, prompt: str, lines=None) -> tuple[str, str]:
        after_lines = lines
        if after_lines is None:
            image = self._capture.capture_region(target.left, target.top, target.width, target.height)
            after_lines = self._ocr.extract(image)
        ocr_text = extract_reply_from_ocr_lines_for_policy(
            before_lines,
            after_lines,
            target_width=target.width,
            target_height=target.height,
            prompt=prompt,
            policy=REPLY_TRACKER_POLICY,
        )
        uia_text = self._read_visible_reply_text(target)
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
            )
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
                        "matched_current_blob_only": assessment["matched_current_blob_only"],
                        "continuity_against_seed": assessment["continuity_against_seed"],
                        "continuity_against_trusted_lineage": assessment["continuity_against_trusted_lineage"],
                        "continuity_against_current": assessment["continuity_against_current"],
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
                        "matched_current_blob_only": assessment["matched_current_blob_only"],
                        "continuity_against_seed": assessment["continuity_against_seed"],
                        "continuity_against_trusted_lineage": assessment["continuity_against_trusted_lineage"],
                        "continuity_against_current": assessment["continuity_against_current"],
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
        self._focus_reply_area(target)
        for _ in range(3):
            pyautogui.press("end")
            time.sleep(0.25)
            pyautogui.press("pagedown")
            time.sleep(0.25)

    def _scroll_reply_up(self, target) -> None:
        self._focus_reply_area(target)
        pyautogui.press("pageup")
        time.sleep(0.25)

    def _focus_reply_area(self, target) -> None:
        self._focus_window(target)
        pyautogui.click(int(target.left + target.width * 0.70), int(target.top + target.height * 0.42))
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
        try:
            window = Desktop(backend="uia").window(handle=target.handle)
        except Exception:
            return ""

        selected: list[tuple[int, int, str]] = []
        seen: set[str] = set()
        min_left = target.left + int(target.width * 0.18)
        min_top = target.top + 110
        max_bottom = target.top + target.height - 70

        for ctrl in window.descendants():
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
