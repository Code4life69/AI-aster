from __future__ import annotations

import json
import re
from dataclasses import dataclass, field


ALLOWED_OPERATION_TYPES = {
    "CREATE FILE",
    "REPLACE FILE",
    "EDIT FILE",
    "RENAME FILE",
    "MOVE FILE",
    "DELETE FILE",
    "INSTALL DEPENDENCIES",
    "RUN COMMANDS",
    "NEED THESE FILES FIRST",
}
CONTENT_OPERATION_TYPES = {"CREATE FILE", "REPLACE FILE", "EDIT FILE"}
RELOCATION_OPERATION_TYPES = {"RENAME FILE", "MOVE FILE"}
COMMAND_OPERATION_TYPES = {"INSTALL DEPENDENCIES", "RUN COMMANDS"}


@dataclass(slots=True)
class PatchOperation:
    type: str
    path: str
    reason: str
    content: str = ""
    new_path: str = ""
    diff_hint: str = ""
    commands: list[str] = field(default_factory=list)
    packages: list[str] = field(default_factory=list)

    @property
    def destructive(self) -> bool:
        return self.type in {"DELETE FILE", "RENAME FILE", "MOVE FILE"}

    @property
    def command_like(self) -> bool:
        return self.type in COMMAND_OPERATION_TYPES


@dataclass(slots=True)
class ParsedPlan:
    summary: str
    notes: list[str]
    operations: list[PatchOperation]

    def requires_more_files(self) -> bool:
        return any(item.type == "NEED THESE FILES FIRST" for item in self.operations)

    def has_command_operations(self) -> bool:
        return any(item.command_like for item in self.operations)

    def has_destructive_operations(self) -> bool:
        return any(item.destructive for item in self.operations)


class ResponseParser:
    def parse(self, raw_text: str) -> ParsedPlan:
        data = json.loads(self._extract_json(raw_text))
        if not isinstance(data, dict):
            raise ValueError("Response must be a JSON object")
        summary = self._require_string(data.get("summary", ""), "summary", allow_empty=True)
        notes = self._require_string_list(data.get("notes", []), "notes", allow_empty=True)
        operations = data.get("operations", [])
        if not isinstance(operations, list):
            raise ValueError("operations must be a list")
        ops = [self._parse_operation(item, index=index) for index, item in enumerate(operations, start=1)]
        if not ops:
            raise ValueError("Response contained no operations")
        return ParsedPlan(summary=summary, notes=notes, operations=ops)

    def _parse_operation(self, item: object, *, index: int) -> PatchOperation:
        if not isinstance(item, dict):
            raise ValueError(f"Operation {index} must be an object")
        op_type = self._require_string(item.get("type"), f"operations[{index}].type")
        if op_type not in ALLOWED_OPERATION_TYPES:
            raise ValueError(f"Unsupported operation type at index {index}: {op_type}")
        path = self._require_string(item.get("path"), f"operations[{index}].path")
        reason = self._require_string(item.get("reason"), f"operations[{index}].reason")
        content = self._require_string(
            item.get("content", ""),
            f"operations[{index}].content",
            allow_empty=True,
        )
        new_path = self._require_string(
            item.get("new_path", ""),
            f"operations[{index}].new_path",
            allow_empty=True,
        )
        diff_hint = self._require_string(
            item.get("diff_hint", ""),
            f"operations[{index}].diff_hint",
            allow_empty=True,
        )
        commands = self._require_string_list(
            item.get("commands", []),
            f"operations[{index}].commands",
            allow_empty=True,
        )
        packages = self._require_string_list(
            item.get("packages", []),
            f"operations[{index}].packages",
            allow_empty=True,
        )

        if op_type in CONTENT_OPERATION_TYPES and "content" not in item:
            raise ValueError(f"{op_type} requires content at operation {index}")
        if op_type in RELOCATION_OPERATION_TYPES and not new_path:
            raise ValueError(f"{op_type} requires new_path at operation {index}")
        if op_type == "RUN COMMANDS" and not commands:
            raise ValueError(f"RUN COMMANDS requires at least one command at operation {index}")
        if op_type == "INSTALL DEPENDENCIES" and not packages:
            raise ValueError(f"INSTALL DEPENDENCIES requires at least one package at operation {index}")

        return PatchOperation(
            type=op_type,
            path=path,
            reason=reason,
            content=content,
            new_path=new_path,
            diff_hint=diff_hint,
            commands=commands,
            packages=packages,
        )

    @staticmethod
    def _require_string(value: object, field_name: str, *, allow_empty: bool = False) -> str:
        if not isinstance(value, str):
            raise ValueError(f"{field_name} must be a string")
        cleaned = value.strip()
        if not cleaned and not allow_empty:
            raise ValueError(f"{field_name} must not be empty")
        return cleaned if not allow_empty else value

    @staticmethod
    def _require_string_list(value: object, field_name: str, *, allow_empty: bool = False) -> list[str]:
        if not isinstance(value, list):
            raise ValueError(f"{field_name} must be a list of strings")
        cleaned: list[str] = []
        for index, item in enumerate(value, start=1):
            if not isinstance(item, str):
                raise ValueError(f"{field_name}[{index}] must be a string")
            text = item.strip()
            if not text and not allow_empty:
                raise ValueError(f"{field_name}[{index}] must not be empty")
            if text:
                cleaned.append(text)
        return cleaned

    @staticmethod
    def _extract_json(raw_text: str) -> str:
        text = raw_text.strip()
        if not text:
            return text
        marker_match = re.search(
            r"ASTER[_ ]PATCH[_ ]BEGIN\s*(\{.*?\})\s*ASTER[_ ]PATCH[_ ]END",
            text,
            flags=re.DOTALL | re.IGNORECASE,
        )
        if marker_match:
            return marker_match.group(1).strip()
        fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
        if fenced:
            return fenced[-1].strip()
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return text[start : end + 1].strip()
        return text
