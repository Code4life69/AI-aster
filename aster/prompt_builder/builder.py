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
        prior_text: str,
        *,
        mode: str = "api",
        max_chars: int | None = None,
    ) -> PromptPackage:
        if mode == "browser":
            return self._build_browser_retry(goal, context, history, prior_text, max_chars=max_chars)
        retry_item = {
            "role": "user",
            "content": self._build_retry_message(prior_text),
        }
        package = self.build(goal, context, history, mode=mode, max_chars=max_chars)
        messages = [*package.messages, retry_item]
        return PromptPackage(
            messages=messages,
            approx_chars=self.estimate_rendered_length(messages),
            included_files=package.included_files,
            omitted_files=package.omitted_files,
            compacted=package.compacted,
        )

    def _build_browser_retry(
        self,
        goal: str,
        context: CollectedContext,
        history: list[SessionTurn],
        prior_text: str,
        *,
        max_chars: int | None,
    ) -> PromptPackage:
        base_limit = max_chars or BROWSER_PROMPT_CHAR_LIMIT
        retry_message = self._build_retry_message(prior_text, max_chars=min(4_000, max(600, base_limit // 3)))
        retry_item = {"role": "user", "content": retry_message}
        retry_rendered = self.estimate_rendered_length([retry_item])
        package_budget = max(2_000, base_limit - retry_rendered - 2)
        package = self.build(goal, context, history, mode="browser", max_chars=package_budget)
        available_retry_chars = max(200, base_limit - self.estimate_rendered_length(package.messages) - len("USER:\n"))
        retry_item = {
            "role": "user",
            "content": self._build_retry_message(prior_text, max_chars=available_retry_chars),
        }
        messages = [*package.messages, retry_item]
        if self.estimate_rendered_length(messages) > base_limit:
            package_budget = max(1_200, base_limit - self.estimate_rendered_length([retry_item]) - 2)
            package = self.build(goal, context, history, mode="browser", max_chars=package_budget)
            messages = [*package.messages, retry_item]
        messages = self._shrink_messages_to_limit(
            package.messages,
            prior_text,
            base_limit=base_limit,
        )
        return PromptPackage(
            messages=messages,
            approx_chars=self.estimate_rendered_length(messages),
            included_files=package.included_files,
            omitted_files=package.omitted_files,
            compacted=True,
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
    def _build_retry_message(prior_text: str, max_chars: int = 4_000) -> str:
        header = (
            "Your previous response was rejected because it was vague or not actionable.\n"
            "Return JSON only with one or more exact operations.\n"
            "Previous response:\n"
        )
        available = max(0, max_chars - len(header))
        excerpt = prior_text[:available]
        if available < len(prior_text):
            excerpt = excerpt.rstrip() + "...[TRUNCATED]"
        return header + excerpt

    def _shrink_messages_to_limit(
        self,
        base_messages: list[dict[str, str]],
        prior_text: str,
        *,
        base_limit: int,
    ) -> list[dict[str, str]]:
        retry_limit = max(200, base_limit)
        messages = [*base_messages, {"role": "user", "content": self._build_retry_message(prior_text, max_chars=retry_limit)}]
        while self.estimate_rendered_length(messages) > base_limit and retry_limit > 200:
            excess = self.estimate_rendered_length(messages) - base_limit
            next_limit = max(200, retry_limit - excess - 24)
            if next_limit >= retry_limit:
                break
            retry_limit = next_limit
            messages = [
                *base_messages,
                {"role": "user", "content": self._build_retry_message(prior_text, max_chars=retry_limit)},
            ]
        return messages
