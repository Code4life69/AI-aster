from __future__ import annotations

from dataclasses import dataclass

from aster.audit_logger import AuditLogger
from aster.browser_core import extract_structured_block
from aster.config import AsterConfig
from aster.context_collector import CollectedContext, ContextCollector
from aster.diff_preview import build_plan_preview
from aster.git_sync import GitSync
from aster.patch_executor import PatchApplier
from aster.prompt_builder import PATCH_PLAN_SCHEMA, PromptBuilder
from aster.response_parser import ParsedPlan, ResponseParser
from aster.runtime_log_heartbeat import RuntimeLogHeartbeatPusher
from aster.safety_guard import SafetyGuard
from aster.session import SessionStore
from aster.transport_api import OpenAIResponsesTransport
from aster.transport_browser import BrowserChatGPTTransport
from aster.visual_action_memory import VisualActionDebugSession, VisualRegionMemoryStore


@dataclass(slots=True)
class OrchestrationResult:
    context: CollectedContext
    plan: ParsedPlan
    preview: str
    warnings: list[str]
    raw_response: str
    sync_log: list[str]


class AsterOrchestrator:
    RUNTIME_LOG_PATHS = (
        ".aster/audit.log.jsonl",
        ".aster/last_launch_stdout.log",
        ".aster/last_launch_stderr.log",
    )

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
        self.guard = SafetyGuard(
            config.project_root,
            config.approval_required_for_destructive,
            config.approval_required_for_commands,
        )
        self.applier = PatchApplier(config.project_root, self.logger, git_integration=config.git_integration)
        self.git = GitSync(config.project_root, remote_name=config.git_remote_name)
        self.api_transport = OpenAIResponsesTransport(config.preferred_model, config.openai_api_key_env)
        self.runtime_log_heartbeat = RuntimeLogHeartbeatPusher(
            tracked_paths=self.RUNTIME_LOG_PATHS,
            enabled=config.runtime_log_heartbeat_push_enabled,
            interval_seconds=config.runtime_log_heartbeat_interval_seconds,
            branch_only=config.runtime_log_heartbeat_branch_only,
            sync_runtime_logs=lambda message: self.sync_runtime_logs(message),
            current_branch=lambda: self.git.current_branch(),
            log_event=lambda event, payload: self.logger.log("browser_transport", {"event": event, **payload}),
        )
        visual_trace_dir = (
            config.visual_action_trace_dir
            if config.visual_action_trace_dir.is_absolute()
            else config.project_root / config.visual_action_trace_dir
        )
        visual_memory_path = (
            config.visual_action_memory_path
            if config.visual_action_memory_path.is_absolute()
            else config.project_root / config.visual_action_memory_path
        )
        visual_action_debug = VisualActionDebugSession(
            root_dir=visual_trace_dir,
            enabled=config.visual_action_trace_enabled,
            checkpoint_interval_seconds=config.visual_action_trace_checkpoint_interval_seconds,
            max_checkpoints_per_key=config.visual_action_trace_max_checkpoints_per_key,
        )
        visual_region_memory = VisualRegionMemoryStore(visual_memory_path)
        if config.visual_action_memory_enabled:
            visual_region_memory.load()
        thread_registry_path = (
            config.thread_registry_path
            if config.thread_registry_path.is_absolute()
            else config.project_root / config.thread_registry_path
        )
        self.browser_transport = BrowserChatGPTTransport(
            self.logger,
            strategy=config.browser_strategy,
            thread_reuse_enabled=config.thread_reuse_enabled,
            thread_registry_path=thread_registry_path,
            verification_level=config.verification_level,
            log_screenshots=config.log_screenshots,
            max_recovery_attempts=config.max_recovery_attempts,
            runtime_log_heartbeat=self.runtime_log_heartbeat,
            visual_action_debug=visual_action_debug,
            visual_region_memory=visual_region_memory,
            visual_action_memory_enabled=config.visual_action_memory_enabled,
        )
        self._last_generation_metadata: dict[str, object] = {}

    def plan(self, goal: str, mode: str | None = None) -> OrchestrationResult:
        resolved_mode = mode or self.config.default_mode
        self._activity(
            "plan_start",
            f"Starting a {resolved_mode} planning run.",
            "Aster needs to collect context and build an exact patch request before asking ChatGPT.",
        )
        if self.config.sync_with_remote:
            self._activity(
                "git_sync",
                "Syncing the local project with GitHub.",
                "Planning against stale files can produce bad edits or conflicts.",
            )
        sync_log = self.git.sync_pull() if self.config.sync_with_remote else []
        self._activity(
            "history_load",
            "Loading recent conversation history.",
            "Follow-up requests should preserve recent context and constraints.",
        )
        history = self.sessions.load()
        self._activity(
            "context_collect",
            "Collecting relevant project files.",
            "Aster sends only the files most likely to matter instead of the whole drive.",
        )
        context = self.collector.collect(self.config.project_root, goal)
        self._activity(
            "context_ready",
            f"Collected {len(context.relevant_files)} relevant files and skipped {len(context.skipped_files)}.",
            "This keeps the prompt focused and avoids overflowing ChatGPT with low-value context.",
            details={"relevant_files": len(context.relevant_files), "skipped_files": len(context.skipped_files)},
        )
        self._activity(
            "prompt_build",
            "Building the structured ChatGPT request.",
            "The browser prompt has to be deterministic so Aster can parse file operations back out.",
        )
        raw, _prompt_package = self._generate_with_prompt_retries(
            goal,
            context,
            history,
            resolved_mode,
            sync_log=sync_log,
        )
        self._activity(
            "response_parse",
            "Parsing ChatGPT's response into a patch plan.",
            "Aster rejects vague advice and only accepts concrete operations it can preview or apply.",
        )
        plan = self._parse_with_retry(goal, context, history, resolved_mode, raw)
        self._activity(
            "safety_check",
            "Running safety checks on the proposed operations.",
            "Aster verifies paths stay inside the project and flags destructive changes.",
        )
        warnings = self.guard.validate(plan)
        self._activity(
            "preview_build",
            "Building the diff preview.",
            "You need to see the exact file changes before deciding whether to apply them.",
        )
        preview = build_plan_preview(self.config.project_root, plan)
        self._activity(
            "history_save",
            "Saving this exchange into session history.",
            "Future follow-up requests can build on what just happened.",
        )
        self.sessions.append("user", goal)
        self.sessions.append("assistant", preview[:4000])
        self._activity(
            "plan_ready",
            f"Plan ready with {len(plan.operations)} operations.",
            "You can review the preview, inspect the raw response, and apply only if it looks correct.",
            status="success",
            details={"operations": len(plan.operations), "warnings": len(warnings)},
        )
        runtime_sync_log: list[str] = []
        if self.config.git_integration and self.config.auto_commit_and_push and self.config.push_runtime_logs_after_plan:
            self._activity(
                "git_runtime_sync",
                "Committing and pushing the latest runtime logs to GitHub.",
                "This keeps the remote repo updated with the newest audit and launch logs after each planning run.",
            )
            runtime_sync_log = self.sync_runtime_logs("sync runtime logs after plan")
        return OrchestrationResult(
            context=context,
            plan=plan,
            preview=preview,
            warnings=warnings,
            raw_response=raw,
            sync_log=sync_log + runtime_sync_log,
        )

    def apply(self, plan: ParsedPlan, selected_indices: list[int] | None = None, dry_run: bool | None = None) -> list[str]:
        effective_dry_run = self.config.dry_run if dry_run is None else dry_run
        selected_operations = self._selected_operations(plan, selected_indices)
        self._activity(
            "apply_start",
            "Applying the selected patch operations.",
            "Aster is about to back up files and perform the approved local changes.",
            details={"dry_run": effective_dry_run},
        )
        results = self.applier.apply(plan, selected_indices=selected_indices, dry_run=effective_dry_run)
        if not effective_dry_run and self.config.git_integration and self.config.auto_commit_and_push:
            commit_message = self._build_commit_message(plan)
            details = {"commit_message": commit_message, "operations": len(selected_operations)}
            if any(op.command_like or op.destructive for op in selected_operations):
                details["warning"] = "Selected operations include commands or destructive file changes."
            self._activity(
                "git_commit_push",
                "Committing and pushing the applied changes to GitHub.",
                "This keeps the remote repo and tracked logs updated after every approved apply run.",
                details=details,
            )
            results.extend(self._sync_repo_state(commit_message))
        self._activity(
            "apply_done",
            "Apply run finished.",
            "Local changes, backups, and optional Git operations are complete.",
            status="success",
        )
        return results

    def connect_remote(self, url: str) -> str:
        self._activity(
            "git_remote_connect",
            "Connecting this project folder to the configured GitHub remote.",
            "Aster needs a remote to keep local edits and GitHub in sync.",
            details={"url": url},
        )
        result = self.git.set_remote(url)
        self.logger.log("git_remote_set", {"url": url, "result": result})
        self._activity(
            "git_remote_connected",
            "GitHub remote connection updated.",
            "Future apply runs can now commit and push to that repo.",
            status="success",
        )
        return result

    def sync_runtime_logs(self, message: str = "Aster runtime sync") -> list[str]:
        return self._sync_repo_state(message, paths=list(self.RUNTIME_LOG_PATHS))

    def _generate(
        self,
        mode: str,
        messages: list[dict[str, str]],
        *,
        retry_attempt: bool = False,
        retry_prompt_mode: str = "",
        retry_prompt_reason: str = "",
    ) -> str:
        if mode == "browser":
            if not self.config.browser_mode_enabled:
                raise RuntimeError("Browser mode is disabled in configuration.")
            prompt = "\n\n".join(f"{item['role'].upper()}:\n{item['content']}" for item in messages)
            result = self.browser_transport.generate(
                prompt,
                chatgpt_url=self.config.chatgpt_url,
                launch_timeout_sec=float(self.config.browser_launch_timeout_seconds),
                retry_attempt=retry_attempt,
                retry_prompt_mode=retry_prompt_mode,
                retry_reason=retry_prompt_reason,
            )
            self._last_generation_metadata = dict(result.metadata)
            self.logger.log("browser_result", result.metadata)
            return result.raw_text
        self._last_generation_metadata = {}
        if not self.config.api_mode_enabled:
            raise RuntimeError("API mode is disabled. Use browser mode or enable API mode in config.")
        self._activity(
            "api_request",
            "Sending the structured request to the OpenAI API.",
            "API mode returns a machine-readable response directly without browser automation.",
        )
        return self.api_transport.generate(messages, PATCH_PLAN_SCHEMA)

    def _parse_with_retry(self, goal: str, context: CollectedContext, history, mode: str, raw: str) -> ParsedPlan:
        try:
            return self.parser.parse(raw)
        except Exception as original_parse_error:
            retry_seed_metadata = self._resolve_retry_seed_metadata(mode, raw)
            prior_text = raw if retry_seed_metadata["retry_seed_valid"] and raw.strip() else None
            self._activity(
                "retry_request",
                "The first response was not machine-parseable, so Aster is retrying with stricter instructions.",
                "Aster only accepts exact file operations and will reprompt once if ChatGPT replies vaguely.",
                status="warning",
            )
            retried_raw, prompt_package = self._generate_with_prompt_retries(
                goal,
                context,
                history,
                mode,
                prior_text=prior_text,
                retry_reason=str(retry_seed_metadata["retry_seed_validity_reason"]),
                retry_seed_used=prior_text is not None,
            )
            self.logger.log(
                "prompt_retry",
                {
                    "mode": mode,
                    "prior_response_preview": raw[:500],
                    "retry_seed_used": prior_text is not None,
                    "retry_seed_valid": retry_seed_metadata["retry_seed_valid"],
                    "retry_seed_validity_reason": retry_seed_metadata["retry_seed_validity_reason"],
                    "retry_seed_validity_reason_source": retry_seed_metadata["retry_seed_validity_reason_source"],
                    "retry_prior_response_length": len(prior_text or ""),
                    "retry_prompt_mode": prompt_package.retry_prompt_mode,
                    "retry_prompt_strategy": prompt_package.retry_prompt_strategy,
                    "retry_prompt_reason": prompt_package.retry_prompt_reason,
                    "retry_prompt_length": prompt_package.retry_prompt_length,
                    "retry_prompt_compacted_relative_to_original": prompt_package.retry_prompt_compacted_relative_to_original,
                    "retry_output_strict_mode": bool(prompt_package.retry_prompt_mode),
                    **self._summarize_prompt_package(prompt_package),
                },
            )
            try:
                return self.parser.parse(retried_raw)
            except Exception as retry_parse_error:
                retry_parse_failure_kind = self._classify_retry_parse_failure_text(retried_raw)
                retry_attempt_failure_reason = str(self._last_generation_metadata.get("retry_attempt_failure_reason", ""))
                self.logger.log(
                    "retry_parse_failure",
                    {
                        "mode": mode,
                        "retry_seed_used": prior_text is not None,
                        "retry_prior_response_length": len(prior_text or ""),
                        "retry_seed_valid": retry_seed_metadata["retry_seed_valid"],
                        "retry_seed_validity_reason": retry_seed_metadata["retry_seed_validity_reason"],
                        "retry_seed_validity_reason_source": retry_seed_metadata["retry_seed_validity_reason_source"],
                        "original_parse_failure_reason": self._format_parse_failure_reason(original_parse_error),
                        "original_parse_text_source": "original_text",
                        "retry_parse_failure_reason": self._format_parse_failure_reason(retry_parse_error),
                        "retry_parse_text_source": "retry_text",
                        "retry_parse_failure_kind": retry_parse_failure_kind,
                        "retry_response_length": len(retried_raw),
                        "retry_output_strict_mode": self._last_generation_metadata.get("retry_output_strict_mode", False),
                        "retry_attempt_capture_mode": self._last_generation_metadata.get("retry_attempt_capture_mode", ""),
                        "retry_attempt_acceptance_tier": self._last_generation_metadata.get(
                            "retry_attempt_acceptance_tier",
                            "",
                        ),
                        "retry_attempt_failure_reason": retry_attempt_failure_reason,
                        "retry_attempt_structured_block_found": self._last_generation_metadata.get(
                            "retry_attempt_structured_block_found",
                            False,
                        ),
                        "retry_attempt_parseable": self._last_generation_metadata.get("retry_attempt_parseable", False),
                        "retry_attempt_prose_contamination": self._last_generation_metadata.get(
                            "retry_attempt_prose_contamination",
                            False,
                        ),
                        "retry_attempt_wrapper_only": self._last_generation_metadata.get(
                            "retry_attempt_wrapper_only",
                            False,
                        ),
                        "retry_attempt_block_count": self._last_generation_metadata.get("retry_attempt_block_count", 0),
                        "retry_attempt_exact_block_only": self._last_generation_metadata.get(
                            "retry_attempt_exact_block_only",
                            False,
                        ),
                        "retry_attempt_extra_text_detected": self._last_generation_metadata.get(
                            "retry_attempt_extra_text_detected",
                            False,
                        ),
                        "retry_attempt_json_object_count": self._last_generation_metadata.get(
                            "retry_attempt_json_object_count",
                            0,
                        ),
                        "retry_attempt_selected_block_index": self._last_generation_metadata.get(
                            "retry_attempt_selected_block_index",
                            None,
                        ),
                        "retry_attempt_block_selection_reason": self._last_generation_metadata.get(
                            "retry_attempt_block_selection_reason",
                            "",
                        ),
                        "retry_attempt_multiple_blocks_ambiguous": self._last_generation_metadata.get(
                            "retry_attempt_multiple_blocks_ambiguous",
                            False,
                        ),
                        "retry_attempt_multiple_blocks_recovered": self._last_generation_metadata.get(
                            "retry_attempt_multiple_blocks_recovered",
                            False,
                        ),
                        "retry_attempt_block_relationship": self._last_generation_metadata.get(
                            "retry_attempt_block_relationship",
                            "",
                        ),
                        "retry_attempt_block_forensics": self._last_generation_metadata.get(
                            "retry_attempt_block_forensics",
                            [],
                        ),
                        "retry_attempt_fragment_repair_pattern_matched": self._last_generation_metadata.get(
                            "retry_attempt_fragment_repair_pattern_matched",
                            False,
                        ),
                        "retry_attempt_fragment_repair_attempted": self._last_generation_metadata.get(
                            "retry_attempt_fragment_repair_attempted",
                            False,
                        ),
                        "retry_attempt_fragment_repair_succeeded": self._last_generation_metadata.get(
                            "retry_attempt_fragment_repair_succeeded",
                            False,
                        ),
                        "retry_attempt_fragment_repair_reason": self._last_generation_metadata.get(
                            "retry_attempt_fragment_repair_reason",
                            "",
                        ),
                        "retry_attempt_repaired_from_block_index": self._last_generation_metadata.get(
                            "retry_attempt_repaired_from_block_index",
                            None,
                        ),
                        "retry_attempt_discarded_wrapper_only_block_index": self._last_generation_metadata.get(
                            "retry_attempt_discarded_wrapper_only_block_index",
                            None,
                        ),
                    },
                )
                if mode == "browser":
                    seed_state = (
                        "after reusing a preserved retry seed"
                        if prior_text is not None
                        else "without reusing any preserved retry seed"
                    )
                    failure_label = retry_attempt_failure_reason or retry_parse_failure_kind
                    raise RuntimeError(
                        "Browser retry response was not machine-parseable. "
                        f"Aster retried {seed_state}, but the follow-up response still could not be parsed "
                        f"({failure_label})."
                    ) from retry_parse_error
                raise

    def _resolve_retry_seed_metadata(self, mode: str, raw: str) -> dict[str, object]:
        retry_seed_valid = bool(raw.strip())
        retry_seed_validity_reason = "raw_response_present" if retry_seed_valid else "empty_response"
        retry_seed_validity_reason_source = "raw_response_fallback"
        if mode == "browser" and self._last_generation_metadata:
            retry_seed_valid = bool(self._last_generation_metadata.get("retry_seed_valid", retry_seed_valid))
            retry_seed_validity_reason = str(
                self._last_generation_metadata.get("retry_seed_validity_reason", retry_seed_validity_reason)
            )
            retry_seed_validity_reason_source = str(
                self._last_generation_metadata.get(
                    "retry_seed_validity_reason_source",
                    "browser_result_metadata",
                )
            )
        return {
            "retry_seed_valid": retry_seed_valid,
            "retry_seed_validity_reason": retry_seed_validity_reason,
            "retry_seed_validity_reason_source": retry_seed_validity_reason_source,
        }

    @staticmethod
    def _format_parse_failure_reason(exc: Exception) -> str:
        message = str(exc).strip()
        return type(exc).__name__ if not message else f"{type(exc).__name__}: {message}"

    def _build_commit_message(self, plan: ParsedPlan) -> str:
        summary = " ".join(plan.summary.split()).strip()
        summary = summary[:72] if summary else "apply patch plan"
        return f"{self.config.auto_commit_message_prefix}: {summary}"

    def _sync_repo_state(self, commit_message: str, paths: list[str] | None = None) -> list[str]:
        if paths is None:
            results = self.git.commit_all_if_needed(commit_message, exclude_paths=list(self.RUNTIME_LOG_PATHS))
        else:
            results = self.git.commit_paths_if_needed(commit_message, paths)
        if self.config.sync_with_remote and self._results_include_successful_commit(results):
            results.extend(self.git.sync_push())
        return results

    @staticmethod
    def _results_include_successful_commit(results: list[str]) -> bool:
        return any("git commit -m " in item and "-> 0:" in item for item in results)

    def _generate_with_prompt_retries(
        self,
        goal: str,
        context: CollectedContext,
        history,
        mode: str,
        *,
        sync_log: list[str] | None = None,
        prior_text: str | None = None,
        retry_reason: str = "",
        retry_seed_used: bool = False,
    ):
        budgets = self._prompt_budgets(mode)
        for attempt_index, budget in enumerate(budgets, start=1):
            prompt_package = self._build_prompt_package(
                goal,
                context,
                history,
                mode=mode,
                max_chars=budget,
                prior_text=prior_text,
                retry_reason=retry_reason,
                retry_seed_used=retry_seed_used,
            )
            if mode == "browser":
                self._activity(
                    "prompt_compact",
                    f"Built a browser-sized prompt of about {prompt_package.approx_chars} characters.",
                    "Browser mode has a lower input limit than the API, so Aster trims context before sending it.",
                    details={
                        "attempt_index": attempt_index,
                        "included_files": len(prompt_package.included_files),
                        "omitted_files": len(prompt_package.omitted_files),
                        "approx_chars": prompt_package.approx_chars,
                        "compacted": prompt_package.compacted,
                        "budget": budget,
                    },
                )
            self.logger.log(
                "prompt_sent",
                {
                    "mode": mode,
                    "goal": goal,
                    "sync_log": sync_log or [],
                    "attempt_index": attempt_index,
                    "retry_prompt": bool(retry_reason) or prior_text is not None,
                    "retry_prompt_mode": prompt_package.retry_prompt_mode,
                    "retry_prompt_strategy": prompt_package.retry_prompt_strategy,
                    "retry_prompt_reason": prompt_package.retry_prompt_reason,
                    "retry_prompt_length": prompt_package.retry_prompt_length,
                    "retry_prompt_compacted_relative_to_original": prompt_package.retry_prompt_compacted_relative_to_original,
                    **self._summarize_prompt_package(prompt_package),
                },
            )
            self._activity(
                "transport_generate",
                f"Sending the request through {mode} mode.",
                "This is where Aster asks ChatGPT for exact file operations.",
                details={"attempt_index": attempt_index},
            )
            try:
                return self._generate(
                    mode,
                    prompt_package.messages,
                    retry_attempt=bool(prompt_package.retry_prompt_mode),
                    retry_prompt_mode=prompt_package.retry_prompt_mode,
                    retry_prompt_reason=prompt_package.retry_prompt_reason,
                ), prompt_package
            except RuntimeError as exc:
                if mode != "browser" or not self._is_prompt_too_large_error(exc) or attempt_index >= len(budgets):
                    raise
                next_budget = budgets[attempt_index]
                self._activity(
                    "browser_prompt_retry",
                    "ChatGPT rejected the browser prompt as too large, so Aster is rebuilding a smaller request.",
                    "Aster can often recover automatically by omitting more context before resending the browser request.",
                    status="warning",
                    details={"attempt_index": attempt_index + 1, "next_budget": next_budget},
                )
                self.logger.log(
                    "browser_prompt_resize_retry",
                    {
                        "failed_budget": budget,
                        "next_budget": next_budget,
                        "attempt_index": attempt_index,
                    },
                )
        raise RuntimeError("Unable to generate a prompt for the selected mode.")

    def _build_prompt_package(
        self,
        goal: str,
        context: CollectedContext,
        history,
        *,
        mode: str,
        max_chars: int,
        prior_text: str | None = None,
        retry_reason: str = "",
        retry_seed_used: bool = False,
    ):
        if prior_text is None and not retry_reason:
            return self.prompt_builder.build(
                goal,
                context,
                history,
                mode=mode,
                max_chars=max_chars,
            )
        return self.prompt_builder.build_retry(
            goal,
            context,
            history,
            prior_text,
            mode=mode,
            max_chars=max_chars,
            retry_reason=retry_reason,
            retry_seed_used=retry_seed_used,
        )

    def _prompt_budgets(self, mode: str) -> list[int]:
        if mode != "browser":
            return [self.config.max_total_prompt_bytes]
        base = min(int(self.config.max_total_prompt_bytes), 30_000)
        budgets: list[int] = []
        for candidate in (base, int(base * 0.78), int(base * 0.6), int(base * 0.42), 8_000):
            budget = max(8_000, candidate)
            if budget not in budgets:
                budgets.append(budget)
        return budgets

    @staticmethod
    def _is_prompt_too_large_error(exc: RuntimeError) -> bool:
        lowered = str(exc).lower()
        return "too large" in lowered or "message too long" in lowered

    @staticmethod
    def _selected_operations(plan: ParsedPlan, selected_indices: list[int] | None) -> list:
        indices = selected_indices or list(range(1, len(plan.operations) + 1))
        return [plan.operations[index - 1] for index in indices]

    def _activity(
        self,
        step: str,
        message: str,
        why: str,
        *,
        status: str = "info",
        details: dict[str, object] | None = None,
    ) -> None:
        self.logger.activity(step, message, why, status=status, details=details)

    @staticmethod
    def _summarize_prompt_package(prompt_package) -> dict[str, object]:
        user_message = next((item["content"] for item in prompt_package.messages if item["role"] == "user"), "")
        return {
            "message_count": len(prompt_package.messages),
            "approx_chars": prompt_package.approx_chars,
            "included_files": list(prompt_package.included_files),
            "omitted_files": list(prompt_package.omitted_files),
            "compacted": prompt_package.compacted,
            "user_preview": user_message[:1200],
        }

    @staticmethod
    def _classify_retry_parse_failure_text(raw_text: str) -> str:
        cleaned = raw_text.strip()
        if not cleaned:
            return "empty_text"
        block_count = cleaned.upper().count("ASTER_PATCH_BEGIN")
        if block_count > 1:
            return "multiple_retry_blocks"
        extracted = extract_structured_block(raw_text).strip()
        outside = cleaned
        if extracted and extracted in outside:
            outside = outside.replace(extracted, " ", 1)
        outside = outside.replace("ASTER_PATCH_BEGIN", " ").replace("ASTER_PATCH_END", " ")
        extra_text_detected = bool(" ".join(outside.split()))
        if extracted and extra_text_detected:
            return "prose_contamination"
        if extracted:
            return "malformed_structured_text"
        lowered = cleaned.lower()
        contamination_markers = (
            "user goal:",
            "project summary:",
            "relevant file tree:",
            "relevant file contents:",
            "conversation history:",
            "constraints:",
            "context omitted for browser size safety",
            "previous response was rejected",
            "retry mode for browser output",
        )
        if any(marker in lowered for marker in contamination_markers):
            return "prose_contamination"
        return "non_json_text"
