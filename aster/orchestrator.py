from __future__ import annotations

from dataclasses import dataclass

from aster.audit_logger import AuditLogger
from aster.config import AsterConfig
from aster.context_collector import CollectedContext, ContextCollector
from aster.diff_preview import build_plan_preview
from aster.git_sync import GitSync
from aster.patch_executor import PatchApplier
from aster.prompt_builder import PATCH_PLAN_SCHEMA, PromptBuilder
from aster.response_parser import ParsedPlan, ResponseParser
from aster.safety_guard import SafetyGuard
from aster.session import SessionStore
from aster.transport_api import OpenAIResponsesTransport
from aster.transport_browser import BrowserChatGPTTransport


@dataclass(slots=True)
class OrchestrationResult:
    context: CollectedContext
    plan: ParsedPlan
    preview: str
    warnings: list[str]
    raw_response: str
    sync_log: list[str]


class AsterOrchestrator:
    def __init__(self, config: AsterConfig) -> None:
        self.config = config
        log_root = config.log_dir or (config.project_root / ".aster")
        self.logger = AuditLogger(log_root)
        self.sessions = SessionStore(log_root, limit=config.history_limit)
        self.collector = ContextCollector(
            ignore_patterns=config.ignore_patterns,
            max_file_bytes=config.max_file_bytes,
            max_total_prompt_bytes=config.max_total_prompt_bytes,
        )
        self.prompt_builder = PromptBuilder()
        self.parser = ResponseParser()
        self.guard = SafetyGuard(config.project_root, config.approval_required_for_destructive)
        self.applier = PatchApplier(config.project_root, self.logger, git_integration=config.git_integration)
        self.git = GitSync(config.project_root, remote_name=config.git_remote_name)
        self.api_transport = OpenAIResponsesTransport(config.preferred_model, config.openai_api_key_env)
        self.browser_transport = BrowserChatGPTTransport()

    def plan(self, goal: str, mode: str | None = None) -> OrchestrationResult:
        resolved_mode = mode or self.config.default_mode
        sync_log = self.git.sync_pull() if self.config.sync_with_remote else []
        history = self.sessions.load()
        context = self.collector.collect(self.config.project_root, goal)
        messages = self.prompt_builder.build(goal, context, history)
        self.logger.log("prompt_sent", {"mode": resolved_mode, "messages": messages, "goal": goal, "sync_log": sync_log})
        raw = self._generate(resolved_mode, messages)
        plan = self._parse_with_retry(goal, context, history, resolved_mode, raw)
        warnings = self.guard.validate(plan)
        preview = build_plan_preview(self.config.project_root, plan)
        self.sessions.append("user", goal)
        self.sessions.append("assistant", preview[:4000])
        return OrchestrationResult(
            context=context,
            plan=plan,
            preview=preview,
            warnings=warnings,
            raw_response=raw,
            sync_log=sync_log,
        )

    def apply(self, plan: ParsedPlan, selected_indices: list[int] | None = None, dry_run: bool | None = None) -> list[str]:
        effective_dry_run = self.config.dry_run if dry_run is None else dry_run
        results = self.applier.apply(plan, selected_indices=selected_indices, dry_run=effective_dry_run)
        if not effective_dry_run and self.config.git_integration and self.config.auto_commit_and_push:
            commit_message = self._build_commit_message(plan)
            results.append(self.git.ensure_branch("main"))
            results.extend(self.git.commit_all_if_needed(commit_message))
            if self.config.sync_with_remote:
                results.extend(self.git.sync_push())
        return results

    def connect_remote(self, url: str) -> str:
        result = self.git.set_remote(url)
        self.logger.log("git_remote_set", {"url": url, "result": result})
        return result

    def _generate(self, mode: str, messages: list[dict[str, str]]) -> str:
        if mode == "browser":
            if not self.config.browser_mode_enabled:
                raise RuntimeError("Browser mode is disabled in configuration.")
            prompt = "\n\n".join(f"{item['role'].upper()}:\n{item['content']}" for item in messages)
            result = self.browser_transport.generate(
                prompt,
                chatgpt_url=self.config.chatgpt_url,
                launch_timeout_sec=float(self.config.browser_launch_timeout_seconds),
            )
            self.logger.log("browser_result", result.metadata)
            return result.raw_text
        if not self.config.api_mode_enabled:
            raise RuntimeError("API mode is disabled. Use browser mode or enable API mode in config.")
        return self.api_transport.generate(messages, PATCH_PLAN_SCHEMA)

    def _parse_with_retry(self, goal: str, context: CollectedContext, history, mode: str, raw: str) -> ParsedPlan:
        try:
            return self.parser.parse(raw)
        except Exception:
            messages = self.prompt_builder.build_retry(goal, context, history, raw)
            self.logger.log("prompt_retry", {"mode": mode, "messages": messages})
            retried_raw = self._generate(mode, messages)
            return self.parser.parse(retried_raw)

    def _build_commit_message(self, plan: ParsedPlan) -> str:
        summary = " ".join(plan.summary.split()).strip()
        summary = summary[:72] if summary else "apply patch plan"
        return f"{self.config.auto_commit_message_prefix}: {summary}"
