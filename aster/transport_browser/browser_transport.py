from __future__ import annotations

import json
import re
import sys
import time
import traceback
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aster.audit_logger import AuditLogger


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
)

pyautogui = None
Desktop = None


@dataclass(slots=True)
class BrowserResult:
    raw_text: str
    metadata: dict[str, str]


class BrowserChatGPTTransport:
    """Experimental browser-mode adapter."""

    def __init__(self, logger: AuditLogger | None = None) -> None:
        self._executor = None
        self._capture = None
        self._ocr = None
        self._logger = logger
        self._last_window_state: str = ""

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
        try:
            self._ensure_runtime()
            target = self._ensure_chatgpt_window(chatgpt_url=chatgpt_url, launch_timeout_sec=launch_timeout_sec)
            self._prepare_chatgpt_window(
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
            populated_directly = self._populate_prompt_directly(target, prompt)
            self._log("populate_prompt_result", {"direct_uia_write": populated_directly})
            if not populated_directly:
                click_point = self._executor.choose_chatgpt_composer_point(target, before_lines)
                self._log("fallback_clipboard_send_start", {"click_point": click_point})
                self._activity(
                    "browser_prompt_fallback",
                    "Direct composer entry failed, so Aster is falling back to paste automation.",
                    "This keeps the run moving even when the browser blocks direct UI automation text entry.",
                )
                populated_directly = self._paste_prompt_with_click(target, prompt, click_point)
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
            reply = self._capture_reply_text(target, before_lines, prompt, timeout_sec=timeout_sec)
            parsed = self.extract_structured_block(reply)
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
        except Exception as exc:
            self._log(
                "generate_failed",
                {
                    "error": str(exc),
                    "traceback": "".join(traceback.format_exception_only(type(exc), exc)).strip(),
                },
            )
            self._activity(
                "browser_failed",
                "Browser mode failed before a valid patch block was captured.",
                "The ChatGPT page did not reach a clean sent-and-replied state that Aster could parse.",
                status="error",
                details={"error": str(exc)},
            )
            raise

    def _stabilize_and_send(self, target, before_lines, prompt: str) -> None:
        deadline = time.monotonic() + 35.0
        showed_text = False
        send_attempt = 0
        while time.monotonic() < deadline:
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
        candidate = self._executor.read_chatgpt_browser_reply(
            target,
            self._capture,
            self._ocr,
            before_lines,
            "",
            timeout_sec=1.0,
        )
        if candidate.strip():
            if self._reply_detection_blocked(state):
                self._log(
                    "reply_detection_suppressed",
                    {
                        "reason": "unsent_composer_state",
                        "candidate_preview": candidate[:160],
                        "ui_state": state,
                    },
                )
                return False
            if self._score_reply_candidate(candidate) < 20.0 and not self._looks_like_substantive_reply_candidate(candidate):
                self._log(
                    "reply_detection_suppressed",
                    {
                        "reason": "low_signal_candidate",
                        "candidate_preview": candidate[:160],
                        "score": self._score_reply_candidate(candidate),
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
            if self._reply_detection_blocked(state):
                self._log(
                    "reply_detection_suppressed",
                    {
                        "reason": "unsent_composer_state",
                        "line_preview": lowered[:160],
                        "ui_state": state,
                    },
                )
                return False
            if self._score_reply_candidate(lowered) < 20.0:
                self._log(
                    "reply_detection_suppressed",
                    {
                        "reason": "low_signal_line",
                        "line_preview": lowered[:160],
                        "score": self._score_reply_candidate(lowered),
                        "ui_state": state,
                    },
                )
                return False
            return True
        return False

    @staticmethod
    def _reply_detection_blocked(ui_state: dict[str, Any]) -> bool:
        if ui_state.get("show_in_text_field_present"):
            return True
        return ui_state.get("send_prompt_enabled") is True

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
        self._executor.focus_window_target(target)
        x = int(target.left + target.width * 0.95)
        y = int(target.top + target.height * 0.93)
        self._log("send_button_coordinate_click", {"x": x, "y": y})
        pyautogui.click(x, y)

    def _click_line(self, target, line) -> None:
        self._executor.focus_window_target(target)
        x = int(target.left + line.center[0])
        y = int(target.top + line.center[1])
        pyautogui.click(x, y)

    def _click_named_button(self, target, phrase: str) -> bool:
        try:
            window = Desktop(backend="uia").window(handle=target.handle)
            for ctrl in window.descendants(control_type="Button"):
                name = (ctrl.window_text() or "").strip().lower()
                if phrase in name:
                    if self._invoke_button(ctrl):
                        self._log("button_invoke", {"phrase": phrase, "name": name, "method": "invoke"})
                        return True
                    ctrl.click_input()
                    self._log("button_invoke", {"phrase": phrase, "name": name, "method": "click_input"})
                    return True
        except Exception:
            return False
        return False

    def _populate_prompt_directly(self, target, prompt: str) -> bool:
        composer = self._find_composer_edit(target)
        if composer is None:
            self._log("composer_edit_not_found", {"ui_state": self._ui_state(target)})
            return False
        try:
            self._executor.focus_window_target(target)
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
                    if self._wait_for_prompt_inserted(target, prompt, seconds=6.0):
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

    def _paste_prompt_with_click(self, target, prompt: str, click_point: tuple[int, int]) -> bool:
        self._executor.focus_window_target(target)
        pyautogui.click(click_point[0], click_point[1])
        time.sleep(0.25)
        self._clear_composer(target)
        self._paste_prompt_via_clipboard(target, prompt, click_point=click_point)
        return self._wait_for_prompt_inserted(target, prompt, seconds=6.0)

    def _paste_prompt_via_clipboard(self, target, prompt: str, click_point: tuple[int, int] | None) -> None:
        previous_clipboard = ""
        try:
            import pyperclip

            previous_clipboard = pyperclip.paste()
            pyperclip.copy(prompt)
            time.sleep(0.15)
            self._executor.focus_window_target(target)
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
        self._executor.focus_window_target(target)
        pyautogui.hotkey("ctrl", "a")
        time.sleep(0.05)
        pyautogui.press("backspace")
        time.sleep(0.1)

    def _wait_for_prompt_inserted(self, target, prompt: str, seconds: float) -> bool:
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            image = self._capture.capture_region(target.left, target.top, target.width, target.height)
            lines = self._ocr.extract(image)
            ui_state = self._ui_state(target)
            if self._prompt_inserted(lines, prompt, ui_state):
                self._log(
                    "prompt_inserted_confirmed",
                    {"ui_state": ui_state, "ocr_preview": [line.text[:120] for line in lines[:6]]},
                )
                return True
            time.sleep(0.5)
        self._log("prompt_inserted_missing", {"ui_state": self._ui_state(target)})
        return False

    def _prompt_inserted(self, lines, prompt: str, ui_state: dict[str, Any]) -> bool:
        visible_text = "\n".join(line.text.lower() for line in lines)
        prompt_words = [word for word in re.findall(r"[a-z0-9]{4,}", prompt.lower())[:12]]
        matched = sum(1 for word in prompt_words if word in visible_text)
        if ui_state.get("show_in_text_field_present"):
            return True
        if ui_state.get("send_prompt_present") and ui_state.get("send_prompt_enabled") is True:
            return True
        if matched >= 2:
            return True
        if "system" in visible_text and matched >= 1:
            return True
        return False

    def _find_composer_edit(self, target):
        try:
            window = Desktop(backend="uia").window(handle=target.handle)
            best = None
            best_bottom = -1
            for ctrl in window.descendants(control_type="Edit"):
                try:
                    name = (ctrl.window_text() or "").strip().lower()
                    rect = ctrl.rectangle()
                except Exception:
                    continue
                if "chatgpt.com" in name:
                    continue
                if rect.bottom > best_bottom:
                    best = ctrl
                    best_bottom = rect.bottom
            return best
        except Exception:
            return None

    def _focus_composer(self, target, lines) -> None:
        composer = self._find_composer_line(lines)
        if composer is not None:
            self._click_line(target, composer)
            time.sleep(0.2)
            return
        self._executor.focus_window_target(target)
        pyautogui.click(int(target.left + target.width * 0.55), int(target.top + target.height * 0.88))
        time.sleep(0.2)

    def _attempt_send(self, target, attempt_index: int) -> None:
        strategy = attempt_index % 5
        self._executor.focus_window_target(target)
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
        try:
            target = self._executor.find_chatgpt_browser_window()
            self._log("chatgpt_window_found", {"title": target.title, "handle": target.handle})
            self._activity(
                "browser_window_found",
                "Found an existing ChatGPT browser window.",
                "Reusing an open signed-in session is faster and avoids reopening the site.",
                details={"window_title": target.title},
            )
            return target
        except Exception:
            self._log("chatgpt_window_missing_opening_browser", {"url": chatgpt_url})
            self._activity(
                "browser_window_open",
                "No ChatGPT window was open, so Aster is launching one now.",
                "Browser mode needs a live ChatGPT tab before it can continue.",
                details={"url": chatgpt_url},
            )
            webbrowser.open(chatgpt_url, new=2)
            deadline = time.monotonic() + max(5.0, launch_timeout_sec)
            while time.monotonic() < deadline:
                try:
                    target = self._executor.find_chatgpt_browser_window()
                    self._log("chatgpt_window_found_after_launch", {"title": target.title, "handle": target.handle})
                    self._activity(
                        "browser_window_found",
                        "The newly opened ChatGPT browser window is ready.",
                        "Aster can now continue with prompt entry and send attempts.",
                        status="success",
                        details={"window_title": target.title},
                    )
                    return target
                except Exception:
                    time.sleep(1.5)
            raise RuntimeError(
                "ChatGPT browser window was not found after launch. Open ChatGPT, sign in if needed, then try again."
            )

    def _prepare_chatgpt_window(self, target, chatgpt_url: str, timeout_sec: float) -> None:
        self._activity(
            "browser_window_reset",
            "Resetting the attached browser tab to a fresh ChatGPT page.",
            "Starting from a clean ChatGPT page avoids stale thread content and wrong-tab captures during browser runs.",
            details={"url": chatgpt_url},
        )
        self._navigate_browser_to_chatgpt(target, chatgpt_url)
        if self._wait_for_chatgpt_ready(target, timeout_sec=timeout_sec):
            return
        image = self._capture.capture_region(target.left, target.top, target.width, target.height)
        lines = self._ocr.extract(image)
        ui_state = self._ui_state(target)
        self._log(
            "chatgpt_page_not_ready",
            {
                "ui_state": ui_state,
                "ocr_preview": [line.text[:120] for line in lines[:8]],
            },
        )
        raise RuntimeError(
            "Attached browser window did not reach a usable ChatGPT page. "
            "Open ChatGPT in the active browser window and make sure the composer is visible."
        )

    def _navigate_browser_to_chatgpt(self, target, chatgpt_url: str) -> None:
        self._executor.focus_window_target(target)
        pyautogui.hotkey("ctrl", "l")
        time.sleep(0.15)
        self._paste_prompt_via_clipboard(target, chatgpt_url, click_point=None)
        time.sleep(0.15)
        pyautogui.press("enter")
        time.sleep(0.8)

    def _wait_for_chatgpt_ready(self, target, timeout_sec: float) -> bool:
        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline:
            image = self._capture.capture_region(target.left, target.top, target.width, target.height)
            lines = self._ocr.extract(image)
            ui_state = self._ui_state(target)
            visible_text = "\n".join(line.text.lower() for line in lines[:20])
            loading = "loading" in visible_text
            if self._looks_like_chatgpt_page(lines, ui_state) and not loading:
                self._log(
                    "window_ready_confirmed",
                    {
                        "ui_state": ui_state,
                        "ocr_preview": [line.text[:120] for line in lines[:6]],
                    },
                )
                return True
            time.sleep(0.6)
        self._log("window_ready_timeout", {"ui_state": self._ui_state(target)})
        return False

    @staticmethod
    def _looks_like_chatgpt_page(lines, ui_state: dict[str, Any]) -> bool:
        visible_text = "\n".join(line.text.lower() for line in lines[:24])
        if any(hint in visible_text for hint in CHATGPT_PAGE_HINTS):
            return True
        return any(
            (
                ui_state.get("send_prompt_present"),
                ui_state.get("show_in_text_field_present"),
                ui_state.get("stop_streaming_present"),
            )
        )

    def _capture_reply_text(self, target, before_lines, prompt: str, timeout_sec: float) -> str:
        deadline = time.monotonic() + timeout_sec
        best_text = ""
        best_score = float("-inf")
        last_structured = ""
        stable_structured_hits = 0
        attempt_index = 0
        scanned_with_scroll = False

        while time.monotonic() < deadline:
            time.sleep(2.0 if attempt_index else 1.2)
            ui_state = self._ui_state(target)
            image = self._capture.capture_region(target.left, target.top, target.width, target.height)
            lines = self._ocr.extract(image)
            self._raise_for_browser_error(lines, ui_state, stage="reply_capture")
            ocr_text, uia_text = self._capture_visible_reply_sources(target, before_lines, prompt, lines=lines)
            current_best = self._choose_best_reply_candidate(
                {
                    "ocr": ocr_text,
                    "uia": uia_text,
                }
            )
            if current_best is not None:
                source, candidate, score = current_best
                if score > best_score or (score == best_score and len(candidate) > len(best_text)):
                    best_text = candidate
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
                if self._looks_like_patch_plan_json(candidate):
                    if candidate == last_structured:
                        stable_structured_hits += 1
                    else:
                        last_structured = candidate
                        stable_structured_hits = 0
                    if stable_structured_hits >= 1 and not self._response_still_streaming(ui_state):
                        return candidate
                if (
                    not scanned_with_scroll
                    and not self._response_still_streaming(ui_state)
                    and score >= 180.0
                    and self._reply_looks_incomplete(candidate)
                ):
                    scrolled = self._capture_reply_text_by_scrolling(target, before_lines, prompt, max_steps=10)
                    if scrolled.strip():
                        self._log(
                            "reply_scrolled_capture",
                            {
                                "length": len(scrolled),
                                "preview": scrolled[:240],
                            },
                        )
                        best_from_scroll = self._choose_best_reply_candidate({"scrolled": scrolled})
                        if best_from_scroll is not None:
                            _, candidate, score = best_from_scroll
                            if score > best_score or (score == best_score and len(candidate) > len(best_text)):
                                best_text = candidate
                                best_score = score
                            if self._looks_like_patch_plan_json(candidate):
                                return candidate
                    scanned_with_scroll = True
            attempt_index += 1

        if self._looks_like_patch_plan_json(best_text):
            return best_text

        fallback = self._executor.read_chatgpt_browser_reply(
            target,
            self._capture,
            self._ocr,
            before_lines,
            prompt,
            timeout_sec=min(8.0, max(2.0, timeout_sec / 4.0)),
        )
        fallback_block = self.extract_structured_block(fallback)
        if self._looks_like_patch_plan_json(fallback_block):
            self._log(
                "reply_candidate_selected",
                {
                    "source": "screen_reader_fallback",
                    "score": self._score_reply_candidate(fallback_block),
                    "length": len(fallback_block),
                    "preview": fallback_block[:240],
                },
            )
            return fallback_block
        return best_text or fallback or ""

    def _choose_best_reply_candidate(self, sources: dict[str, str]) -> tuple[str, str, float] | None:
        best: tuple[str, str, float] | None = None
        for source, text in sources.items():
            raw = text.strip()
            if not raw:
                continue
            variants = [raw]
            block = self.extract_structured_block(raw)
            if block and block != raw:
                variants.insert(0, block)
            for candidate in variants:
                score = self._score_reply_candidate(candidate)
                if best is None or score > best[2] or (score == best[2] and len(candidate) > len(best[1])):
                    best = (source, candidate, score)
        return best

    def _capture_visible_reply_sources(self, target, before_lines, prompt: str, lines=None) -> tuple[str, str]:
        after_lines = lines
        if after_lines is None:
            image = self._capture.capture_region(target.left, target.top, target.width, target.height)
            after_lines = self._ocr.extract(image)
        ocr_text = self._extract_reply_from_ocr_lines(before_lines, after_lines, target, prompt)
        uia_text = self._read_visible_reply_text(target)
        return ocr_text, uia_text

    def _capture_reply_text_by_scrolling(self, target, before_lines, prompt: str, max_steps: int) -> str:
        self._activity(
            "browser_scroll_read",
            "Scrolling through the ChatGPT thread to collect the full reply.",
            "Long patch plans may span multiple viewports, so Aster needs to read more than the currently visible slice.",
        )
        self._scroll_reply_to_bottom(target)
        segments: list[str] = []
        repeated = 0
        previous_signature = ""

        for step in range(max_steps):
            segment = self._capture_visible_reply_segment(target, before_lines, prompt)
            if self._segment_looks_like_prompt_echo(segment):
                self._log(
                    "reply_scroll_prompt_boundary",
                    {
                        "step": step,
                        "preview": segment[:220],
                    },
                )
                if segments:
                    break
                repeated += 1
                continue
            signature = _normalize(segment)
            if not segment.strip():
                repeated += 1
            elif signature == previous_signature:
                repeated += 1
            else:
                repeated = 0
                previous_signature = signature
                segments.append(segment)
                self._log(
                    "reply_scroll_segment",
                    {
                        "step": step,
                        "length": len(segment),
                        "preview": segment[:220],
                    },
                )

            merged = self._merge_text_segments(list(reversed(segments)))
            if self._looks_like_patch_plan_json(merged):
                self._scroll_reply_to_bottom(target)
                return merged
            if repeated >= 2:
                break
            self._scroll_reply_up(target)

        self._scroll_reply_to_bottom(target)
        return self._merge_text_segments(list(reversed(segments)))

    def _capture_visible_reply_segment(self, target, before_lines, prompt: str) -> str:
        ocr_text, uia_text = self._capture_visible_reply_sources(target, before_lines, prompt)
        return self._clean_captured_segment(self._merge_text_segments([uia_text, ocr_text]))

    def _scroll_reply_to_bottom(self, target) -> None:
        self._focus_reply_area(target)
        for _ in range(3):
            pyautogui.press("end")
            time.sleep(0.25)
            pyautogui.press("pagedown")
            time.sleep(0.25)

    def _scroll_reply_up(self, target) -> None:
        self._focus_reply_area(target)
        for _ in range(2):
            pyautogui.press("pageup")
            time.sleep(0.25)

    def _focus_reply_area(self, target) -> None:
        self._executor.focus_window_target(target)
        pyautogui.click(int(target.left + target.width * 0.70), int(target.top + target.height * 0.42))
        time.sleep(0.2)

    @staticmethod
    def _merge_text_segments(segments: list[str]) -> str:
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

    @staticmethod
    def _reply_looks_incomplete(text: str) -> bool:
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

    @staticmethod
    def _segment_looks_like_prompt_echo(text: str) -> bool:
        lowered = _normalize(text)
        if not lowered:
            return False
        hits = sum(1 for marker in PROMPT_ECHO_MARKERS if marker in lowered)
        return hits >= 2 or ("your message" in lowered and "path:" in lowered)

    @staticmethod
    def _clean_captured_segment(text: str) -> str:
        kept: list[str] = []
        for raw_line in text.splitlines():
            line = raw_line.strip()
            lowered = _normalize(line)
            if not line:
                continue
            if any(marker in lowered for marker in PROMPT_ECHO_MARKERS):
                continue
            if lowered in {"share", "stop streaming", "stop generating"}:
                continue
            kept.append(line)
        return "\n".join(kept)

    @staticmethod
    def _response_still_streaming(ui_state: dict[str, Any]) -> bool:
        return bool(ui_state.get("stop_streaming_present"))

    def _extract_reply_from_ocr_lines(self, before_lines, after_lines, target, prompt: str) -> str:
        before_seen = {_normalize(line.text) for line in before_lines}
        prompt_words = {word for word in re.findall(r"[a-z0-9]{4,}", prompt.lower())}
        selected: list[tuple[int, int, str]] = []
        seen: set[str] = set()

        for line in after_lines:
            text = " ".join(line.text.split())
            lowered = _normalize(text)
            if not text:
                continue
            if line.bbox[1] < 110:
                continue
            if line.bbox[3] > max(0, target.height - 70):
                continue
            if line.center[0] < target.width * 0.18:
                continue
            if lowered in seen:
                continue
            if any(noise in lowered for noise in BROWSER_REPLY_NOISE):
                continue
            if any(marker in lowered for marker in PROMPT_ECHO_MARKERS):
                continue
            overlap = sum(1 for word in prompt_words if word in lowered)
            if prompt_words and overlap >= max(6, len(prompt_words) // 2) and "{" not in text and '"' not in text:
                continue
            if lowered in before_seen and "{" not in text and '"' not in text:
                continue
            selected.append((line.bbox[1], line.bbox[0], text))
            seen.add(lowered)

        if not selected:
            return ""
        selected.sort(key=lambda item: (item[0], item[1]))
        return "\n".join(text for _, _, text in selected)

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

    @classmethod
    def _looks_like_substantive_reply_candidate(cls, text: str) -> bool:
        cleaned = text.strip()
        if len(cleaned) < 80:
            return False
        lowered = _normalize(cleaned)
        if cls._segment_looks_like_prompt_echo(cleaned):
            return False
        if any(noise in lowered for noise in BROWSER_REPLY_NOISE):
            return False
        if any(hint in lowered for hint in INPUT_TOO_LARGE_HINTS):
            return False
        if any(bad in lowered for bad in (
            "ask gemini",
            "github",
            "context omitted for browser size safety",
            "included_files",
            "omitted_files",
            "return promptpackage",
            "def _normalize",
            "self.log(""activity""",
        )):
            return False
        if "aster patch begin" in lowered or "aster_patch_begin" in lowered:
            return True
        if cls._looks_like_patch_plan_json(cleaned):
            return True
        if '"operations"' in cleaned and "{" in cleaned and "}" in cleaned:
            return True
        return False

    @classmethod
    def _score_reply_candidate(cls, text: str) -> float:
        cleaned = text.strip()
        if not cleaned:
            return float("-inf")
        lowered = _normalize(cleaned)
        score = min(120.0, len(cleaned) / 30.0)
        if "aster patch begin" in lowered or "aster_patch_begin" in lowered:
            score += 180.0
        if '"operations"' in cleaned:
            score += 80.0
        if any(op in cleaned for op in ("CREATE FILE", "EDIT FILE", "REPLACE FILE", "RUN COMMANDS", "NEED THESE FILES FIRST")):
            score += 60.0
        if cls._looks_like_patch_plan_json(cleaned):
            score += 220.0
        for noise in (
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
            "self.log(""activity""",
        ):
            if noise in lowered:
                score -= 180.0
        if "thought for" in lowered:
            score -= 25.0
        if any(hint in lowered for hint in STOP_STREAMING_HINTS):
            score -= 40.0
        if not (
            "aster patch begin" in lowered
            or "aster_patch_begin" in lowered
            or cls._looks_like_patch_plan_json(cleaned)
            or ('"operations"' in cleaned and "{" in cleaned and "}" in cleaned)
        ):
            score -= 200.0
        return score

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
    def _looks_like_patch_plan_json(text: str) -> bool:
        candidate = BrowserChatGPTTransport.extract_structured_block(text)
        try:
            data = json.loads(candidate)
        except Exception:
            return False
        if not isinstance(data, dict):
            return False
        operations = data.get("operations")
        return isinstance(operations, list) and bool(operations)

    def _log_window_state(self, target, lines, prompt: str, ui_state: dict[str, Any] | None = None) -> None:
        ui_state = ui_state or self._ui_state(target)
        visible_text = "\n".join(line.text.lower() for line in lines[:12])
        prompt_words = [word for word in re.findall(r"[a-z0-9]{4,}", prompt.lower())[:8]]
        payload = {
            "ui_state": ui_state,
            "ocr_line_count": len(lines),
            "show_in_text_field_visible": "show in text field" in visible_text,
            "prompt_words_visible": [word for word in prompt_words if word in visible_text],
            "ocr_preview": [line.text[:120] for line in lines[:5]],
        }
        signature = str(payload)
        if signature != self._last_window_state:
            self._last_window_state = signature
            self._log("window_state", payload)

    def _ui_state(self, target) -> dict[str, Any]:
        state: dict[str, Any] = {
            "window_title": getattr(target, "title", ""),
            "send_prompt_present": False,
            "send_prompt_enabled": None,
            "show_in_text_field_present": False,
            "stop_streaming_present": False,
            "composer_edit_length": None,
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
                    state["composer_edit_length"] = len((composer.window_text() or "").strip())
                except Exception:
                    state["composer_edit_length"] = "unknown"
        except Exception as exc:
            state["inspection_error"] = str(exc)
        return state


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

