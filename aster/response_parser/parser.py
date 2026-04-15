from __future__ import annotations

import json
from dataclasses import dataclass, field


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


@dataclass(slots=True)
class ParsedPlan:
    summary: str
    notes: list[str]
    operations: list[PatchOperation]

    def requires_more_files(self) -> bool:
        return any(item.type == "NEED THESE FILES FIRST" for item in self.operations)


class ResponseParser:
    def parse(self, raw_text: str) -> ParsedPlan:
        data = json.loads(raw_text)
        ops = [
            PatchOperation(
                type=item["type"],
                path=item["path"],
                reason=item["reason"],
                content=item.get("content", ""),
                new_path=item.get("new_path", ""),
                diff_hint=item.get("diff_hint", ""),
                commands=list(item.get("commands", [])),
                packages=list(item.get("packages", [])),
            )
            for item in data.get("operations", [])
        ]
        if not ops:
            raise ValueError("Response contained no operations")
        return ParsedPlan(summary=data.get("summary", ""), notes=list(data.get("notes", [])), operations=ops)
