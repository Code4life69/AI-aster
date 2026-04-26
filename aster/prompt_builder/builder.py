from __future__ import annotations

from dataclasses import dataclass

from aster.context_collector.collector import CollectedContext, ContextFile
from aster.session import SessionTurn


PATCH_PLAN_SCHEMA: dict[str, object] = {
    "name": "patch_plan",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["summary", "operations", "notes"],
        "properties": {
            "summary": {"type": "string"},
            "notes": {"type": "array", "items": {"type": "string"}},
            "operations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["type", "path", "reason"],
                    "properties": {
                        "type": {
                            "type": "string",
                            "enum": [
                                "CREATE FILE",
                                "REPLACE FILE",
                                "EDIT FILE",
                                "RENAME FILE",
                                "MOVE FILE",
                                "DELETE FILE",
                                "INSTALL DEPENDENCIES",
                                "RUN COMMANDS",
                                "NEED THESE FILES FIRST",
                            ],
                        },
                        "path": {"type": "string"},
                        "new_path": {"type": "string"},
                        "reason": {"type": "string"},
                        "content": {"type": "string"},
                        "diff_hint": {"type": "string"},
                        "commands": {"type": "array", "items": {"type": "string"}},
                        "packages": {"type": "array", "items": {"type": "string"}},
                    },
                },
            },
        },
    },
}

SYSTEM_MESSAGE = (
    "You are a coding orchestrator backend. Return JSON only. "
    "Do not provide explanations outside the schema. "
    "Every change must be represented as a concrete machine-parseable operation."
)

BROWSER_FINAL_OUTPUT_RULES = (
    "Final output rules for browser mode:\n"
    "- Put ASTER_PATCH_BEGIN on its own line.\n"
    "- Then output exactly one JSON object.\n"
    "- Then put ASTER_PATCH_END on its own line.\n"
    "- Do not write any other text outside those markers.\n"
)

BROWSER_PROMPT_CHAR_LIMIT = 30_000
BROWSER_FILE_CHAR_LIMIT = 3_500
BROWSER_HISTORY_CHAR_LIMIT = 1_500
BROWSER_FILE_LIST_LIMIT = 24


@dataclass(slots=True)
class PromptPackage:
    messages: list[dict[str, str]]
    approx_chars: int
    included_files: list[str]
    omitted_files: list[str]
    compacted: bool
    retry_prompt_mode: str = ""
    retry_prompt_strategy: str = ""
    retry_prompt_reason: str = ""
    retry_prompt_length: int = 0
    retry_prompt_compacted_relative_to_original: bool = False


class PromptBuilder:
    def build(
        self,
        goal: str,
        context: CollectedContext,
        history: list[SessionTurn],
        *,
        mode: str = "api",
        max_chars: int | None = None,
    ) -> PromptPackage:
        if mode == "browser":
            return self._build_browser(goal, context, history, max_chars=max_chars or BROWSER_PROMPT_CHAR_LIMIT)
        messages = [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": self._build_standard_user_message(goal, context, history)},
        ]
        return PromptPackage(
            messages=messages,
            approx_chars=self.estimate_rendered_length(messages),
            included_files=[item.path for item in context.relevant_files],
            omitted_files=list(context.skipped_files),
            compacted=False,
        )

    def build_retry(
        self,
        goal: str,
        context: CollectedContext,
        history: list[SessionTurn],
        prior_text: str | None,
        *,
        mode: str = "api",
        max_chars: int | None = None,
        retry_reason: str = "",
        retry_seed_used: bool = True,
    ) -> PromptPackage:
        if mode == "browser":
            return self._build_browser_retry(
                goal,
                context,
                history,
                prior_text,
                max_chars=max_chars,
                retry_reason=retry_reason,
                retry_seed_used=retry_seed_used,
            )
        retry_item = {
            "role": "user",
            "content": self._build_retry_message(
                prior_text or "",
                retry_reason=retry_reason,
                retry_seed_used=retry_seed_used,
            ),
        }
        package = self.build(goal, context, history, mode=mode, max_chars=max_chars)
        messages = [*package.messages, retry_item]
        return PromptPackage(
            messages=messages,
            approx_chars=self.estimate_rendered_length(messages),
            included_files=package.included_files,
            omitted_files=package.omitted_files,
            compacted=package.compacted,
            retry_prompt_mode="seeded_retry" if retry_seed_used and prior_text else "no_seed_retry",
            retry_prompt_strategy="strict_retry_prompt",
            retry_prompt_reason=retry_reason,
            retry_prompt_length=len(retry_item["content"]),
            retry_prompt_compacted_relative_to_original=package.compacted,
        )

    def _build_browser_retry(
        self,
        goal: str,
        context: CollectedContext,
        history: list[SessionTurn],
        prior_text: str | None,
        *,
        max_chars: int | None,
        retry_reason: str,
        retry_seed_used: bool,
    ) -> PromptPackage:
        base_limit = max_chars or BROWSER_PROMPT_CHAR_LIMIT
        retry_message = self._build_browser_retry_message(
            prior_text,
            retry_reason=retry_reason,
            retry_seed_used=retry_seed_used,
            max_chars=min(4_000, max(900, base_limit // 3)),
        )
        retry_item = {"role": "user", "content": retry_message}
        retry_rendered = self.estimate_rendered_length([retry_item])
        package_budget = max(
            1_600,
            min(base_limit - retry_rendered - 2, int(base_limit * (0.5 if not retry_seed_used else 0.74))),
        )
        package = self.build(goal, context, history, mode="browser", max_chars=package_budget)
        available_retry_chars = max(200, base_limit - self.estimate_rendered_length(package.messages) - len("USER:\n"))
        retry_item = {
            "role": "user",
            "content": self._build_browser_retry_message(
                prior_text,
                retry_reason=retry_reason,
                retry_seed_used=retry_seed_used,
                max_chars=available_retry_chars,
            ),
        }
        messages = [*package.messages, retry_item]
        if self.estimate_rendered_length(messages) > base_limit:
            package_budget = max(1_200, base_limit - self.estimate_rendered_length([retry_item]) - 2)
            package = self.build(goal, context, history, mode="browser", max_chars=package_budget)
            messages = [*package.messages, retry_item]
        messages, retry_prompt_length = self._shrink_messages_to_limit(
            package.messages,
            prior_text,
            base_limit=base_limit,
            retry_reason=retry_reason,
            retry_seed_used=retry_seed_used,
        )
        return PromptPackage(
            messages=messages,
            approx_chars=self.estimate_rendered_length(messages),
            included_files=package.included_files,
            omitted_files=package.omitted_files,
            compacted=True,
            retry_prompt_mode=(
                "browser_seeded_retry"
                if retry_seed_used and prior_text
                else "browser_structured_output_only_retry"
            ),
            retry_prompt_strategy=(
                "browser_retry_with_prior_excerpt"
                if retry_seed_used and prior_text
                else "browser_retry_structured_output_only"
            ),
            retry_prompt_reason=retry_reason,
            retry_prompt_length=retry_prompt_length,
            retry_prompt_compacted_relative_to_original=True,
        )

    @staticmethod
    def estimate_rendered_length(messages: list[dict[str, str]]) -> int:
        return len("\n\n".join(f"{item['role'].upper()}:\n{item['content']}" for item in messages))

    def _build_standard_user_message(
        self,
        goal: str,
        context: CollectedContext,
        history: list[SessionTurn],
    ) -> str:
        history_block = "\n".join(f"{item.role.upper()}: {item.content}" for item in history[-6:])
        files_block = "\n\n".join(
            f"PATH: {item.path}\nREASON: {item.reason}\nCONTENT:\n{item.content}" for item in context.relevant_files
        )
        return (
            "User goal:\n"
            f"{goal}\n\n"
            "Project summary:\n"
            f"{context.project_summary}\n\n"
            "Relevant file tree:\n"
            f"{context.file_tree}\n\n"
            "Relevant file contents:\n"
            f"{files_block}\n\n"
            "Conversation history:\n"
            f"{history_block or '[none]'}\n\n"
            "Constraints:\n"
            "- Output exact file-level operations only.\n"
            "- Do not expose secrets.\n"
            "- Do not modify unrelated files.\n"
            "- If context is insufficient, return NEED THESE FILES FIRST operations.\n"
            "- For EDIT FILE or REPLACE FILE, include the full intended file content.\n"
            "- For RUN COMMANDS or INSTALL DEPENDENCIES, keep commands minimal and deterministic.\n"
        )

    def _build_browser(
        self,
        goal: str,
        context: CollectedContext,
        history: list[SessionTurn],
        *,
        max_chars: int,
    ) -> PromptPackage:
        browser_limit = min(max_chars, BROWSER_PROMPT_CHAR_LIMIT)
        history_block = self._compact_history(history)
        selected_files: list[ContextFile] = []
        omitted_files: list[str] = []

        for index, item in enumerate(context.relevant_files):
            candidate_files = [*selected_files, item]
            candidate_omitted = [rest.path for rest in context.relevant_files[index + 1 :]] + list(context.skipped_files)
            messages = self._build_browser_messages(goal, context, history_block, candidate_files, candidate_omitted)
            if self.estimate_rendered_length(messages) <= browser_limit:
                selected_files = candidate_files
                omitted_files = candidate_omitted
            else:
                omitted_files = [candidate.path for candidate in context.relevant_files[index:]] + list(context.skipped_files)
                break
        else:
            omitted_files = list(context.skipped_files)

        messages = self._build_browser_messages(goal, context, history_block, selected_files, omitted_files)
        if self.estimate_rendered_length(messages) > browser_limit:
            trimmed_selected = selected_files[:]
            while trimmed_selected and self.estimate_rendered_length(messages) > browser_limit:
                trimmed_selected.pop()
                omitted_files = [item.path for item in context.relevant_files if item.path not in {entry.path for entry in trimmed_selected}]
                omitted_files.extend(path for path in context.skipped_files if path not in omitted_files)
                messages = self._build_browser_messages(goal, context, history_block, trimmed_selected, omitted_files)
            selected_files = trimmed_selected

        return PromptPackage(
            messages=messages,
            approx_chars=self.estimate_rendered_length(messages),
            included_files=[item.path for item in selected_files],
            omitted_files=omitted_files,
            compacted=bool(omitted_files) or len(selected_files) < len(context.relevant_files),
        )

    def _build_browser_messages(
        self,
        goal: str,
        context: CollectedContext,
        history_block: str,
        selected_files: list[ContextFile],
        omitted_files: list[str],
    ) -> list[dict[str, str]]:
        file_list = self._build_browser_file_list(context.relevant_files, omitted_files)
        contents_block = (
            "\n\n".join(self._format_browser_context_file(item) for item in selected_files)
            if selected_files
            else "[none included yet; if more code is needed, return NEED THESE FILES FIRST]"
        )
        omitted_summary = self._format_omitted_summary(omitted_files)
        user_message = (
            "User goal:\n"
            f"{goal}\n\n"
            "Project summary:\n"
            f"{context.project_summary}\n\n"
            "Relevant file tree:\n"
            f"{file_list}\n\n"
            "Relevant file contents:\n"
            f"{contents_block}\n\n"
            f"{omitted_summary}\n\n"
            "Conversation history:\n"
            f"{history_block}\n\n"
            "Constraints:\n"
            "- Output exact file-level operations only.\n"
            "- Browser mode has a smaller prompt budget than the API.\n"
            "- Use only the provided context.\n"
            "- If more context is needed, return NEED THESE FILES FIRST operations.\n"
            "- For EDIT FILE or REPLACE FILE, include the full intended file content.\n"
            "- For RUN COMMANDS or INSTALL DEPENDENCIES, keep commands minimal and deterministic.\n\n"
            f"{BROWSER_FINAL_OUTPUT_RULES}"
        )
        return [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": user_message},
        ]

    def _build_browser_file_list(self, relevant_files: list[ContextFile], omitted_files: list[str]) -> str:
        lines = [f"- {item.path} ({item.reason})" for item in relevant_files[:BROWSER_FILE_LIST_LIMIT]]
        hidden_count = max(0, len(relevant_files) - BROWSER_FILE_LIST_LIMIT)
        if hidden_count:
            lines.append(f"- ... {hidden_count} more relevant files not listed")
        if omitted_files:
            lines.append(f"- Browser prompt omitted file contents for {len(omitted_files)} paths")
        return "\n".join(lines) if lines else "[none]"

    @staticmethod
    def _format_browser_context_file(item: ContextFile) -> str:
        content = item.content
        if len(content) > BROWSER_FILE_CHAR_LIMIT:
            content = content[:BROWSER_FILE_CHAR_LIMIT] + "\n...[TRUNCATED FOR BROWSER PROMPT]..."
        return f"PATH: {item.path}\nREASON: {item.reason}\nCONTENT:\n{content}"

    @staticmethod
    def _format_omitted_summary(omitted_files: list[str]) -> str:
        if not omitted_files:
            return "All selected context fit within the browser prompt budget."
        preview = ", ".join(omitted_files[:8])
        suffix = "" if len(omitted_files) <= 8 else f", and {len(omitted_files) - 8} more"
        return (
            "Context omitted for browser size safety:\n"
            f"- {len(omitted_files)} file paths were left out or truncated from the browser request.\n"
            f"- First omitted paths: {preview}{suffix}\n"
            "- If this is not enough, return NEED THESE FILES FIRST operations."
        )

    @staticmethod
    def _compact_history(history: list[SessionTurn]) -> str:
        if not history:
            return "[none]"
        entries: list[str] = []
        used = 0
        for item in history[-4:]:
            chunk = f"{item.role.upper()}: {item.content[:600]}"
            if used + len(chunk) > BROWSER_HISTORY_CHAR_LIMIT:
                remaining = max(0, BROWSER_HISTORY_CHAR_LIMIT - used - len(item.role) - 4)
                chunk = f"{item.role.upper()}: {item.content[:remaining]}...[TRUNCATED]"
            entries.append(chunk)
            used += len(chunk)
            if used >= BROWSER_HISTORY_CHAR_LIMIT:
                break
        return "\n".join(entries) or "[none]"

    @staticmethod
    def _retry_reason_instruction(retry_reason: str) -> str:
        if retry_reason == "unbalanced_structure":
            return (
                "The previous browser reply looked truncated. "
                "Return one complete structured response with balanced braces/brackets and a finished operations array."
            )
        if retry_reason == "missing_required_schema_keys":
            return (
                "The previous browser reply was missing required schema keys. "
                "Include summary, notes, and operations, and ensure every operation includes type, path, and reason."
            )
        if retry_reason == "prompt_or_preamble_contamination":
            return (
                "The previous browser reply included prompt or preamble contamination. "
                "Return only the final structured response, with no copied prompt text."
            )
        if retry_reason in {
            "multiple_retry_blocks",
            "ambiguous_multiple_retry_blocks",
            "fragmented_multi_block_retry_output",
            "wrapper_only_multi_block_retry_output",
            "no_single_retry_block_selected",
        }:
            return (
                "The previous browser reply returned more than one ASTER block or fragmented wrappers. "
                "Return exactly one final ASTER block, never a draft plus revised copy, and never repeat the block."
            )
        if retry_reason == "empty_response":
            return "The previous browser reply did not produce usable structured output. Return the final structured response only."
        return "The previous browser reply was not machine-parseable. Return one complete structured response only."

    def _build_retry_message(
        self,
        prior_text: str,
        *,
        max_chars: int = 4_000,
        retry_reason: str = "",
        retry_seed_used: bool = True,
    ) -> str:
        header = (
            "Your previous response was rejected because it was not machine-parseable.\n"
            f"{self._retry_reason_instruction(retry_reason)}\n"
            "Return JSON only with one or more exact operations.\n"
        )
        if not retry_seed_used or not prior_text.strip():
            return header
        header += "Previous response:\n"
        available = max(0, max_chars - len(header))
        excerpt = prior_text[:available]
        if available < len(prior_text):
            excerpt = excerpt.rstrip() + "...[TRUNCATED]"
        return header + excerpt

    def _build_browser_retry_message(
        self,
        prior_text: str | None,
        *,
        retry_reason: str,
        retry_seed_used: bool,
        max_chars: int,
    ) -> str:
        lines = [
            "Retry mode for browser output:",
            "- The previous browser response was invalid or incomplete.",
            f"- Failure mode: {retry_reason or 'not_machine_parseable'}.",
            f"- {self._retry_reason_instruction(retry_reason)}",
            "- Return only one final ASTER_PATCH_BEGIN / ASTER_PATCH_END block.",
            "- Inside the markers, output exactly one JSON object.",
            "- Exactly one block is allowed. Any text before or after the block will fail validation.",
            "- Never repeat the block, provide alternatives, or output multiple versions.",
            '- Required top-level keys: "summary", "notes", "operations".',
            "- Do not add commentary, explanations, prose, markdown fences, or code outside the structured block.",
            "- Do not repeat prompt text, context listings, or browser instructions.",
            "- If the provided context is insufficient, return NEED THESE FILES FIRST operations inside the JSON object.",
        ]
        if retry_seed_used and prior_text and prior_text.strip():
            lines.extend(
                [
                    "",
                    "Rejected prior response excerpt:",
                    prior_text,
                ]
            )
        raw = "\n".join(lines)
        if len(raw) <= max_chars:
            return raw
        if retry_seed_used and prior_text and prior_text.strip():
            prefix = "\n".join(lines[:-1])
            excerpt_header = "\nRejected prior response excerpt:\n"
            available = max(0, max_chars - len(prefix) - len(excerpt_header))
            excerpt = prior_text[:available]
            if available < len(prior_text):
                excerpt = excerpt.rstrip() + "...[TRUNCATED]"
            return prefix + excerpt_header + excerpt
        return raw[:max_chars].rstrip()

    def _shrink_messages_to_limit(
        self,
        base_messages: list[dict[str, str]],
        prior_text: str | None,
        *,
        base_limit: int,
        retry_reason: str,
        retry_seed_used: bool,
    ) -> tuple[list[dict[str, str]], int]:
        retry_limit = max(200, base_limit)
        retry_message = self._build_browser_retry_message(
            prior_text,
            retry_reason=retry_reason,
            retry_seed_used=retry_seed_used,
            max_chars=retry_limit,
        )
        messages = [*base_messages, {"role": "user", "content": retry_message}]
        while self.estimate_rendered_length(messages) > base_limit and retry_limit > 200:
            excess = self.estimate_rendered_length(messages) - base_limit
            next_limit = max(200, retry_limit - excess - 24)
            if next_limit >= retry_limit:
                break
            retry_limit = next_limit
            retry_message = self._build_browser_retry_message(
                prior_text,
                retry_reason=retry_reason,
                retry_seed_used=retry_seed_used,
                max_chars=retry_limit,
            )
            messages = [
                *base_messages,
                {"role": "user", "content": retry_message},
            ]
        return messages, len(retry_message)
