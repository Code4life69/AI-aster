from __future__ import annotations

import difflib
from pathlib import Path

from aster.response_parser import ParsedPlan, PatchOperation


def build_plan_preview(project_root: Path, plan: ParsedPlan) -> str:
    sections = [f"Summary: {plan.summary}"]
    for index, op in enumerate(plan.operations, start=1):
        sections.append(f"\n[{index}] {op.type} {op.path}")
        sections.append(f"Reason: {op.reason}")
        preview = _operation_preview(project_root, op)
        if preview:
            sections.append(preview)
    if plan.notes:
        sections.append("\nNotes:")
        sections.extend(f"- {note}" for note in plan.notes)
    return "\n".join(sections)


def _operation_preview(project_root: Path, op: PatchOperation) -> str:
    path = project_root / op.path
    if op.type in {"CREATE FILE", "REPLACE FILE", "EDIT FILE"}:
        before = path.read_text(encoding="utf-8") if path.exists() else ""
        after = op.content
        diff = difflib.unified_diff(
            before.splitlines(),
            after.splitlines(),
            fromfile=f"a/{op.path}",
            tofile=f"b/{op.path}",
            lineterm="",
        )
        return "\n".join(diff) or "[no textual diff]"
    if op.type in {"RENAME FILE", "MOVE FILE"}:
        return f"{op.path} -> {op.new_path}"
    if op.type == "DELETE FILE":
        return "[file will be deleted]"
    if op.type == "INSTALL DEPENDENCIES":
        return "\n".join(op.packages or [])
    if op.type == "RUN COMMANDS":
        return "\n".join(op.commands or [])
    return op.diff_hint
