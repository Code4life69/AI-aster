from __future__ import annotations

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
        ttk.Button(button_row, text="Generate Plan", command=self._generate).pack(side="left")
        ttk.Button(button_row, text="Apply All", command=self._apply_all).pack(side="left", padx=8)

        self.preview = tk.Text(body, wrap="none")
        self.preview.pack(fill="both", expand=True)

    def _browse(self) -> None:
        selected = filedialog.askdirectory(initialdir=self.project_root.get())
        if selected:
            self.project_root.set(selected)

    def _connect_repo(self) -> None:
        config = load_config(Path(self.project_root.get()))
        orchestrator = AsterOrchestrator(config)
        result = orchestrator.connect_remote(self.remote_url.get().strip())
        messagebox.showinfo("Aster", result)

    def _generate(self) -> None:
        goal = self.prompt_box.get("1.0", "end").strip()
        if not goal:
            messagebox.showerror("Aster", "Enter a request first.")
            return
        self.status.set("Collecting context and generating plan...")
        self.root.update_idletasks()
        config = load_config(Path(self.project_root.get()))
        if self.mode.get() == "api" and not config.api_mode_enabled:
            messagebox.showerror("Aster", "API mode is disabled. Browser mode is the default free path.")
            self.status.set("Ready")
            return
        orchestrator = AsterOrchestrator(config)
        self.last_result = orchestrator.plan(goal, mode=self.mode.get())
        self.preview.delete("1.0", "end")
        if self.last_result.sync_log:
            self.preview.insert("end", "Git sync:\n" + "\n".join(self.last_result.sync_log) + "\n\n")
        if self.last_result.warnings:
            self.preview.insert("end", "Warnings:\n" + "\n".join(self.last_result.warnings) + "\n\n")
        self.preview.insert("end", self.last_result.preview)
        self.status.set("Plan ready")

    def _apply_all(self) -> None:
        if self.last_result is None:
            messagebox.showerror("Aster", "Generate a plan first.")
            return
        config = load_config(Path(self.project_root.get()))
        orchestrator = AsterOrchestrator(config)
        results = orchestrator.apply(self.last_result.plan, dry_run=False)
        self.preview.insert("end", "\n\nApply results:\n" + "\n".join(results))
        self.status.set("Applied")

    def run(self) -> None:
        self.root.mainloop()
