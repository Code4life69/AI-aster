from __future__ import annotations

import re
import sys
import time
import webbrowser
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class BrowserResult:
    raw_text: str
    metadata: dict[str, str]


class BrowserChatGPTTransport:
    """Experimental browser-mode adapter."""

    def __init__(self) -> None:
        self._executor = None
        self._capture = None
        self._ocr = None

    def _ensure_runtime(self) -> None:
        if self._executor is not None:
            return
        try:
            from screen_reader.automation import ActionExecutor
            from screen_reader.capture import ScreenCaptureService
            from screen_reader.ocr import OCREngine
        except Exception as exc:
            sibling = Path("C:/Screen Reader")
            if sibling.exists() and str(sibling) not in sys.path:
                sys.path.insert(0, str(sibling))
            try:
                from screen_reader.automation import ActionExecutor
                from screen_reader.capture import ScreenCaptureService
                from screen_reader.ocr import OCREngine
            except Exception:
                raise RuntimeError(
                    "Browser mode requires the Screen Reader modules and optional automation dependencies."
                ) from exc
        self._executor = ActionExecutor()
        self._capture = ScreenCaptureService()
        self._ocr = OCREngine()

    def generate(
        self,
        prompt: str,
        timeout_sec: float = 90.0,
        chatgpt_url: str = "https://chatgpt.com/",
        launch_timeout_sec: float = 45.0,
    ) -> BrowserResult:
        self._ensure_runtime()
        target = self._ensure_chatgpt_window(chatgpt_url=chatgpt_url, launch_timeout_sec=launch_timeout_sec)
        frame = self._capture.capture_region(target.left, target.top, target.width, target.height)
        before_lines = self._ocr.extract(frame.image)
        click_point = self._executor.choose_chatgpt_composer_point(target, before_lines)
        self._executor.send_prompt_to_chatgpt_browser(target, prompt, click_point)
        time.sleep(1.0)
        reply = self._executor.read_chatgpt_browser_reply(
            target,
            self._capture,
            self._ocr,
            before_lines,
            prompt,
            timeout_sec=timeout_sec,
        )
        parsed = self.extract_structured_block(reply)
        if not parsed.strip():
            raise RuntimeError("Browser mode could not capture a final ChatGPT response")
        return BrowserResult(
            raw_text=parsed,
            metadata={"window_title": target.title},
        )

    def _ensure_chatgpt_window(self, chatgpt_url: str, launch_timeout_sec: float):
        try:
            return self._executor.find_chatgpt_browser_window()
        except Exception:
            webbrowser.open(chatgpt_url, new=2)
            deadline = time.monotonic() + max(5.0, launch_timeout_sec)
            while time.monotonic() < deadline:
                try:
                    return self._executor.find_chatgpt_browser_window()
                except Exception:
                    time.sleep(1.5)
            raise RuntimeError(
                "ChatGPT browser window was not found after launch. Open ChatGPT, sign in if needed, then try again."
            )

    @staticmethod
    def extract_structured_block(raw_text: str) -> str:
        text = raw_text.strip()
        if not text:
            return ""
        fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
        if fenced:
            return fenced[-1].strip()
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return text[start : end + 1].strip()
        return text
