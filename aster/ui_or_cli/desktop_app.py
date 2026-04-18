from __future__ import annotations

import json
import os
import threading
import traceback
import tkinter as tk
from datetime import datetime, timezone
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from aster.config import load_config
from aster.orchestrator import AsterOrchestrator


try:
    GUI_TIMEZONE = ZoneInfo("America/Chicago")
except ZoneInfoNotFoundError:
    GUI_TIMEZONE = datetime.now().astimezone().tzinfo or timezone.utc


def _format_gui_timestamp(raw_ts: object) -> str:
    text = str(raw_ts or "").strip()
    if not text:
        return "--:--"
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return text
    localized = parsed.astimezone(GUI_TIMEZONE)
    return f"{localized.strftime('%I:%M:%S %p').lstrip('0')} CT"


def _compact_text(value: object, *, limit: int = 180) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _format_detail_summary(details: object) -> str:
    if not isinstance(details, dict) or not details:
        return ""
    parts: list[str] = []
    for key, value in details.items():
        parts.append(f"{key}={_compact_text(value, limit=80)}")
    return ", ".join(parts)


def _format_ui_state_summary(ui_state: object) -> str:
    if not isinstance(ui_state, dict) or not ui_state:
        return ""
    parts: list[str] = []
    title = _compact_text(ui_state.get("window_title", ""), limit=60)
    if title:
        parts.append(f"title={title}")
    if "send_prompt_present" in ui_state or "send_prompt_enabled" in ui_state:
        parts.append(
            "send="
            f"{ui_state.get('send_prompt_present')}/"
            f"{ui_state.get('send_prompt_enabled')}"
        )
    if "show_in_text_field_present" in ui_state:
        parts.append(f"show_in_text={ui_state.get('show_in_text_field_present')}")
    if "stop_streaming_present" in ui_state:
        parts.append(f"stop={ui_state.get('stop_streaming_present')}")
    if "composer_edit_length" in ui_state:
        parts.append(f"composer={ui_state.get('composer_edit_length')}")
    return ", ".join(parts)


def _format_activity_entry(event: dict[str, object]) -> str | None:
    if event.get("kind") != "activity":
        return None
    payload = event.get("payload", {})
    if not isinstance(payload, dict):
        return None
    message = _compact_text(payload.get("message", ""), limit=220)
    if not message:
        return None
    status = str(payload.get("status", "info")).strip().lower()
    stamp = _format_gui_timestamp(event.get("ts"))
    headline = f"[{stamp}] {message}"
    if status and status != "info":
        headline += f" [{status}]"
    lines = [headline]
    why = _compact_text(payload.get("why", ""), limit=240)
    if why:
        lines.append(f"Why: {why}")
    details = _format_detail_summary(payload.get("details"))
    if details:
        lines.append(f"Details: {details}")
    return "\n".join(lines)


def _format_trace_entry(event: dict[str, object]) -> str | None:
    kind = str(event.get("kind", "")).strip()
    if not kind or kind == "activity":
        return None
    payload = event.get("payload", {})
    if not isinstance(payload, dict):
        payload = {}
    stamp = _format_gui_timestamp(event.get("ts"))

    if kind == "browser_transport":
        event_name = str(payload.get("event", "browser_transport")).strip() or "browser_transport"
        lines = [f"[{stamp}] {event_name}"]
        details: list[str] = []
        title = payload.get("title") or payload.get("target_title")
        if title:
            details.append(f"title={_compact_text(title, limit=60)}")
        for key in ("attempt_index", "strategy", "prompt_length", "reply_length", "parsed_length", "length", "source"):
            if key in payload:
                details.append(f"{key}={_compact_text(payload.get(key), limit=60)}")
        score = payload.get("score")
        if isinstance(score, (int, float)):
            details.append(f"score={score:.1f}")
        ui_state = _format_ui_state_summary(payload.get("ui_state"))
        if ui_state:
            details.append(ui_state)
        preview = payload.get("preview")
        if not preview:
            ocr_preview = payload.get("ocr_preview")
            if isinstance(ocr_preview, list) and ocr_preview:
                preview = " | ".join(str(item) for item in ocr_preview[:6])
        if details:
            lines.append("Details: " + ", ".join(details))
        if preview:
            lines.append("Preview: " + _compact_text(preview, limit=260))
        return "\n".join(lines)

    if kind == "prompt_sent":
        lines = [f"[{stamp}] prompt_sent"]
        details = [
            f"mode={_compact_text(payload.get('mode'), limit=30)}",
            f"attempt_index={_compact_text(payload.get('attempt_index'), limit=12)}",
            f"approx_chars={_compact_text(payload.get('approx_chars'), limit=12)}",
            f"included_files={len(payload.get('included_files', [])) if isinstance(payload.get('included_files'), list) else 0}",
            f"omitted_files={len(payload.get('omitted_files', [])) if isinstance(payload.get('omitted_files'), list) else 0}",
        ]
        lines.append("Details: " + ", ".join(details))
        goal = payload.get("goal")
        if goal:
            lines.append("Goal: " + _compact_text(goal, limit=220))
        return "\n".join(lines)

    if kind == "prompt_retry":
        lines = [f"[{stamp}] prompt_retry"]
        details = [
            f"mode={_compact_text(payload.get('mode'), limit=30)}",
            f"approx_chars={_compact_text(payload.get('approx_chars'), limit=12)}",
        ]
        lines.append("Details: " + ", ".join(details))
        preview = payload.get("prior_response_preview")
        if preview:
            lines.append("Preview: " + _compact_text(preview, limit=260))
        return "\n".join(lines)

    if kind == "browser_result":
        return f"[{stamp}] browser_result\nDetails: {_format_detail_summary(payload)}"

    if kind in {"apply_operation", "backup_created", "git_remote_set"}:
        details = _format_detail_summary(payload)
        return f"[{stamp}] {kind}" + (f"\nDetails: {details}" if details else "")

    details = _format_detail_summary(payload)
    return f"[{stamp}] {kind}" + (f"\nDetails: {details}" if details else "")


class DesktopApp:
    def __init__(self) -> None:
        self.config = load_config(Path.cwd())
        self.root = tk.Tk()
        self.root.title("Aster Coding Orchestrator")
        self.root.geometry("1120x780")
        self.project_root = tk.StringVar(value=str(Path.cwd()))
        self.mode = tk.StringVar(value=self.config.default_mode)
        self.status = tk.StringVar(value="Ready")
        self.remote_url = tk.StringVar(value="https://github.com/Code4life69/AI-aster.git")
        self.last_result = None
        self.last_apply_results: list[str] = []
        self._worker_thread: threading.Thread | None = None
        self._worker_action = ""
        self._worker_result = None
        self._worker_error = ""
        self._build()

    def _build(self) -> None:
        top = ttk.Frame(self.root, padding=12)
        top.pack(fill="x")
        ttk.Label(top, text="Project Root").grid(row=0, column=0, sticky="w")
        ttk.Entry(top, textvariable=self.project_root, width=90).grid(row=0, column=1, sticky="ew", padx=8)
        ttk.Button(top, text="Browse", command=self._browse).grid(row=0, column=2)
        ttk.Label(top, text="GitHub Remote").grid(row=1, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(top, textvariable=self.remote_url, width=90).grid(row=1, column=1, sticky="ew", padx=8, pady=(8, 0))
        ttk.Button(top, text="Connect Repo", command=self._connect_repo).grid(row=1, column=2, pady=(8, 0))
        top.columnconfigure(1, weight=1)

        mode_row = ttk.Frame(self.root, padding=(12, 0, 12, 0))
        mode_row.pack(fill="x")
        ttk.Radiobutton(mode_row, text="Browser", variable=self.mode, value="browser").pack(side="left")
        ttk.Radiobutton(mode_row, text="API", variable=self.mode, value="api").pack(side="left", padx=8)
        ttk.Label(mode_row, textvariable=self.status).pack(side="right")

        body = ttk.Frame(self.root, padding=(12, 8, 12, 12))
        body.pack(fill="both", expand=True)
        self.prompt_box = tk.Text(body, height=7, wrap="word")
        self.prompt_box.pack(fill="x")
        self.prompt_box.insert("1.0", "Describe the change you want.")

        button_row = ttk.Frame(body)
        button_row.pack(fill="x", pady=8)
        self.generate_button = ttk.Button(button_row, text="Generate Plan", command=self._generate)
        self.generate_button.pack(side="left")
        self.apply_button = ttk.Button(button_row, text="Apply All", command=self._apply_all)
        self.apply_button.pack(side="left", padx=8)
        self.refresh_button = ttk.Button(button_row, text="Refresh Logs", command=self._refresh_views)
        self.refresh_button.pack(side="left")
        self.open_audit_button = ttk.Button(button_row, text="Open .aster Folder", command=self._open_audit_folder)
        self.open_audit_button.pack(side="left", padx=8)

        panes = ttk.Panedwindow(body, orient="horizontal")
        panes.pack(fill="both", expand=True)

        left = ttk.Frame(panes)
        right = ttk.Frame(panes)
        panes.add(left, weight=3)
        panes.add(right, weight=2)

        ttk.Label(left, text="Plan And Raw Response").pack(anchor="w")
        self.preview = tk.Text(left, wrap="none")
        self.preview.pack(fill="both", expand=True)

        ttk.Label(right, text="Active Reasoning (Central Time)").pack(anchor="w")
        self.activity_view = tk.Text(right, wrap="word", height=16)
        self.activity_view.pack(fill="x")

        ttk.Label(right, text="Live Trace And Launch Logs (Central Time)").pack(anchor="w", pady=(8, 0))
        self.audit_view = tk.Text(right, wrap="word")
        self.audit_view.pack(fill="both", expand=True)
        self._refresh_views()
        self.root.after(1200, self._auto_refresh_views)

    def _browse(self) -> None:
        selected = filedialog.askdirectory(initialdir=self.project_root.get())
        if selected:
            self.project_root.set(selected)

    def _connect_repo(self) -> None:
        config = load_config(Path(self.project_root.get()))
        orchestrator = AsterOrchestrator(config)
        result = orchestrator.connect_remote(self.remote_url.get().strip())
        self._refresh_views()
        messagebox.showinfo("Aster", result)

    def _generate(self) -> None:
        goal = self.prompt_box.get("1.0", "end").strip()
        if not goal:
            messagebox.showerror("Aster", "Enter a request first.")
            return
        if self._is_busy():
            messagebox.showerror("Aster", "Aster is already working. Wait for the current run to finish.")
            return
        config = load_config(Path(self.project_root.get()))
        if self.mode.get() == "api" and not config.api_mode_enabled:
            messagebox.showerror("Aster", "API mode is disabled. Browser mode is the default free path.")
            self.status.set("Ready")
            return
        self.last_apply_results = []
        self._set_busy(True, "Starting plan generation...")
        self._start_worker(self._generate_worker, self.project_root.get(), self.mode.get(), goal)

    def _apply_all(self) -> None:
        if self.last_result is None:
            messagebox.showerror("Aster", "Generate a plan first.")
            return
        if self._is_busy():
            messagebox.showerror("Aster", "Aster is already working. Wait for the current run to finish.")
            return
        self._set_busy(True, "Applying approved changes...")
        self._start_worker(self._apply_worker, self.project_root.get(), self.last_result.plan)

    def _render_result(self) -> None:
        if self.last_result is None:
            return
        sections: list[str] = []
        if self.last_result.sync_log:
            sections.append("Git sync:\n" + "\n".join(self.last_result.sync_log))
        if self.last_result.warnings:
            sections.append("Warnings:\n" + "\n".join(self.last_result.warnings))
        sections.append("Patch preview:\n" + self.last_result.preview)
        sections.append("Raw browser/API response:\n" + self.last_result.raw_response)
        if self.last_apply_results:
            sections.append("Apply results:\n" + "\n".join(self.last_apply_results))
        self.preview.delete("1.0", "end")
        self.preview.insert("end", "\n\n".join(sections))

    def _refresh_views(self) -> None:
        self._refresh_activity()
        self._refresh_logs()

    def _refresh_activity(self) -> None:
        audit_file = Path(self.project_root.get()) / ".aster" / "audit.log.jsonl"
        entries: list[str] = []
        latest_message = ""
        for raw_line in self._tail_lines(audit_file, max_lines=240, max_chars=120000):
            try:
                event = json.loads(raw_line)
            except json.JSONDecodeError:
                continue
            entry = _format_activity_entry(event)
            if not entry:
                continue
            entries.append(entry)
            payload = event.get("payload", {})
            if isinstance(payload, dict):
                latest_message = str(payload.get("message", "")).strip() or latest_message
        self.activity_view.delete("1.0", "end")
        self.activity_view.insert("end", "\n\n".join(entries[-18:]) if entries else "No activity yet.")
        if self._is_busy() and latest_message:
            self.status.set(latest_message[:90])

    def _refresh_logs(self) -> None:
        audit_file = Path(self.project_root.get()) / ".aster" / "audit.log.jsonl"
        launch_out = Path(self.project_root.get()) / ".aster" / "last_launch_stdout.log"
        launch_err = Path(self.project_root.get()) / ".aster" / "last_launch_stderr.log"
        chunks: list[str] = []
        if audit_file.exists():
            trace_entries: list[str] = []
            for raw_line in self._tail_lines(audit_file, max_lines=220, max_chars=120000):
                try:
                    event = json.loads(raw_line)
                except json.JSONDecodeError:
                    continue
                formatted = _format_trace_entry(event)
                if formatted:
                    trace_entries.append(formatted)
            chunks.append("Audit Trace\n" + ("\n\n".join(trace_entries[-28:]) if trace_entries else "No trace events yet."))
        if launch_out.exists():
            chunks.append("last_launch_stdout.log\n" + self._tail_text(launch_out, max_chars=3000))
        if launch_err.exists():
            chunks.append("last_launch_stderr.log\n" + self._tail_text(launch_err, max_chars=3000))
        self.audit_view.delete("1.0", "end")
        self.audit_view.insert("end", "\n\n".join(chunks) if chunks else "No logs yet.")

    def _open_audit_folder(self) -> None:
        folder = Path(self.project_root.get()) / ".aster"
        folder.mkdir(parents=True, exist_ok=True)
        os.startfile(str(folder))

    def _start_worker(self, target, *args) -> None:
        self._worker_result = None
        self._worker_error = ""
        self._worker_action = target.__name__
        self._worker_thread = threading.Thread(target=self._run_worker, args=(target, *args), daemon=True)
        self._worker_thread.start()
        self.root.after(200, self._poll_worker)

    def _run_worker(self, target, *args) -> None:
        try:
            self._worker_result = target(*args)
        except Exception:
            self._worker_error = traceback.format_exc()

    def _generate_worker(self, project_root: str, mode: str, goal: str):
        config = load_config(Path(project_root))
        orchestrator = AsterOrchestrator(config)
        return orchestrator.plan(goal, mode=mode)

    def _apply_worker(self, project_root: str, plan):
        config = load_config(Path(project_root))
        orchestrator = AsterOrchestrator(config)
        return orchestrator.apply(plan, dry_run=False)

    def _poll_worker(self) -> None:
        self._refresh_views()
        if self._worker_thread is not None and self._worker_thread.is_alive():
            self.root.after(400, self._poll_worker)
            return
        self._finish_worker()

    def _auto_refresh_views(self) -> None:
        if not self._is_busy():
            self._refresh_views()
        self.root.after(1200, self._auto_refresh_views)

    def _finish_worker(self) -> None:
        self._set_busy(False)
        if self._worker_error:
            error_text = self._worker_error.strip()
            if self._worker_action == "_generate_worker":
                self.preview.delete("1.0", "end")
                self.preview.insert("end", f"Error:\n{error_text}")
                self.status.set("Error")
            else:
                self.preview.insert("end", f"\n\nApply error:\n{error_text}")
                self.status.set("Apply failed")
        elif self._worker_action == "_generate_worker":
            self.last_result = self._worker_result
            self._render_result()
            self.status.set("Plan ready")
        elif self._worker_action == "_apply_worker":
            self.last_apply_results = list(self._worker_result or [])
            self._render_result()
            self.status.set("Applied")
        self._refresh_views()

    def _set_busy(self, busy: bool, message: str = "Ready") -> None:
        for button in (self.generate_button, self.apply_button, self.open_audit_button):
            if busy:
                button.state(["disabled"])
            else:
                button.state(["!disabled"])
        self.status.set(message)
        self.root.update_idletasks()

    def _is_busy(self) -> bool:
        return self._worker_thread is not None and self._worker_thread.is_alive()

    @staticmethod
    def _tail_text(path: Path, max_chars: int) -> str:
        if not path.exists():
            return ""
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - max_chars))
            return handle.read().decode("utf-8", errors="ignore")[-max_chars:]

    @classmethod
    def _tail_lines(cls, path: Path, max_lines: int, max_chars: int) -> list[str]:
        text = cls._tail_text(path, max_chars=max_chars)
        return text.splitlines()[-max_lines:]

    def run(self) -> None:
        self.root.mainloop()
