from __future__ import annotations

import json
import os
import threading
import traceback
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from aster.config import load_config
from aster.orchestrator import AsterOrchestrator


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

        ttk.Label(right, text="Active Reasoning").pack(anchor="w")
        self.activity_view = tk.Text(right, wrap="word", height=16)
        self.activity_view.pack(fill="x")

        ttk.Label(right, text="Audit Log Tail").pack(anchor="w", pady=(8, 0))
        self.audit_view = tk.Text(right, wrap="word")
        self.audit_view.pack(fill="both", expand=True)
        self._refresh_views()

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
            if event.get("kind") != "activity":
                continue
            payload = event.get("payload", {})
            ts = str(event.get("ts", ""))[11:19]
            message = str(payload.get("message", "")).strip()
            why = str(payload.get("why", "")).strip()
            details = payload.get("details")
            if not message:
                continue
            block = f"[{ts}] {message}"
            if why:
                block += f"\nWhy: {why}"
            if isinstance(details, dict) and details:
                summary = ", ".join(f"{key}={value}" for key, value in details.items())
                block += f"\nDetails: {summary}"
            entries.append(block)
            latest_message = message
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
            chunks.append("audit.log.jsonl\n" + "\n".join(self._tail_lines(audit_file, max_lines=40, max_chars=80000)))
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
