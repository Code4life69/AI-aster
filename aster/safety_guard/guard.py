from __future__ import annotations

from pathlib import Path

from aster.response_parser import ParsedPlan


class SafetyGuard:
    def __init__(self, project_root: Path, require_confirmation_for_destructive: bool = True) -> None:
        self.project_root = project_root.resolve()
        self.require_confirmation_for_destructive = require_confirmation_for_destructive

    def validate(self, plan: ParsedPlan) -> list[str]:
        warnings: list[str] = []
        for op in plan.operations:
            if op.path:
                self._ensure_within_project(op.path)
            if op.new_path:
                self._ensure_within_project(op.new_path)
            if op.destructive and self.require_confirmation_for_destructive:
                warnings.append(f"Confirmation required: {op.type} {op.path}")
        return warnings

    def _ensure_within_project(self, raw_path: str) -> None:
        target = (self.project_root / raw_path).resolve()
        if self.project_root not in target.parents and target != self.project_root:
            raise ValueError(f"Operation points outside project root: {raw_path}")
