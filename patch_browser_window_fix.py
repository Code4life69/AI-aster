from pathlib import Path
import re

path = Path(r"C:\Ai Asistant\aster\transport_browser\browser_transport.py")
text = path.read_text(encoding="utf-8")

new_ensure = r'''
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
'''

text, count = re.subn(
    r"(?s)    def _ensure_chatgpt_window\(self, chatgpt_url: str, launch_timeout_sec: float\):.*?(?=\n    def _prepare_chatgpt_window)",
    new_ensure.rstrip() + "\n",
    text,
    count=1,
)

if count != 1:
    raise SystemExit("Failed to replace _ensure_chatgpt_window")

old_prepare_snippet = """        if self._wait_for_chatgpt_ready(target, timeout_sec=timeout_sec):
            return
        image = self._capture.capture_region(target.left, target.top, target.width, target.height)
        lines = self._ocr.extract(image)
        ui_state = self._ui_state(target)
"""

new_prepare_snippet = """        ready = self._wait_for_chatgpt_ready(target, timeout_sec=timeout_sec)
        image = self._capture.capture_region(target.left, target.top, target.width, target.height)
        lines = self._ocr.extract(image)
        ui_state = self._ui_state(target)
        visible_text = " ".join(line.text.lower() for line in lines[:20])
        wrong_page_markers = (
            "ask gemini",
            "github",
            "youtube",
            "pull request",
            "issues",
            "commit",
        )
        if any(marker in visible_text for marker in wrong_page_markers):
            self._log(
                "wrong_page_detected",
                {
                    "ui_state": ui_state,
                    "ocr_preview": [line.text[:120] for line in lines[:8]],
                },
            )
            raise RuntimeError(
                "Attached browser window does not look like a clean ChatGPT page. "
                "Wrong-page markers were visible in OCR."
            )
        if ready:
            return
"""

if old_prepare_snippet not in text:
    raise SystemExit("Failed to patch _prepare_chatgpt_window readiness block")

text = text.replace(old_prepare_snippet, new_prepare_snippet, 1)

path.write_text(text, encoding="utf-8", newline="\n")
print("patched", path)
