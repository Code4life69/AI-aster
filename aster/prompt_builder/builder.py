from __future__ import annotations

from aster.context_collector import CollectedContext
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
                                "NEED THESE FILES FIRST"
                            ]
                        },
                        "path": {"type": "string"},
                        "new_path": {"type": "string"},
                        "reason": {"type": "string"},
                        "content": {"type": "string"},
                        "diff_hint": {"type": "string"},
                        "commands": {"type": "array", "items": {"type": "string"}},
                        "packages": {"type": "array", "items": {"type": "string"}}
                    }
                }
            }
        }
    }
}


class PromptBuilder:
    def build(self, goal: str, context: CollectedContext, history: list[SessionTurn]) -> list[dict[str, str]]:
        history_block = "\n".join(f"{item.role.upper()}: {item.content}" for item in history[-6:])
        files_block = "\n\n".join(
            f"PATH: {item.path}\nREASON: {item.reason}\nCONTENT:\n{item.content}" for item in context.relevant_files
        )
        user_message = (
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
            "- Prefer API mode first.\n"
            "- Keep browser mode isolated behind an adapter.\n"
            "- Do not modify unrelated files.\n"
            "- If context is insufficient, return NEED THESE FILES FIRST operations.\n"
            "- For EDIT FILE or REPLACE FILE, include the full intended file content.\n"
            "- For RUN COMMANDS or INSTALL DEPENDENCIES, keep commands minimal and deterministic.\n"
        )
        system_message = (
            "You are a coding orchestrator backend. Return JSON only. "
            "Do not provide explanations outside the schema. "
            "Every change must be represented as a concrete machine-parseable operation."
        )
        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]

    def build_retry(
        self,
        goal: str,
        context: CollectedContext,
        history: list[SessionTurn],
        prior_text: str,
    ) -> list[dict[str, str]]:
        messages = self.build(goal, context, history)
        messages.append(
            {
                "role": "user",
                "content": (
                    "Your previous response was rejected because it was vague or not actionable.\n"
                    "Return JSON only with one or more exact operations.\n"
                    "Previous response:\n"
                    f"{prior_text[:4000]}"
                ),
            }
        )
        return messages
