from pathlib import Path

import pytest

from aster.response_parser import ResponseParser
from aster.safety_guard import SafetyGuard


def test_parser_reads_operations() -> None:
    raw = """
    {
      "summary": "Update one file",
      "notes": ["safe"],
      "operations": [
        {
          "type": "EDIT FILE",
          "path": "src/app.py",
          "reason": "fix bug",
          "content": "print('fixed')"
        }
      ]
    }
    """
    plan = ResponseParser().parse(raw)
    assert plan.summary == "Update one file"
    assert plan.operations[0].path == "src/app.py"


def test_guard_blocks_escape_from_project(tmp_path: Path) -> None:
    raw = """
    {
      "summary": "Bad plan",
      "notes": [],
      "operations": [
        {
          "type": "EDIT FILE",
          "path": "../outside.py",
          "reason": "bad",
          "content": "x=1"
        }
      ]
    }
    """
    plan = ResponseParser().parse(raw)
    guard = SafetyGuard(tmp_path)
    try:
        guard.validate(plan)
    except ValueError as exc:
        assert "outside project root" in str(exc)
    else:
        raise AssertionError("Expected guard to reject plan")


def test_parser_rejects_empty_command_list() -> None:
    raw = """
    {
      "summary": "Bad command plan",
      "notes": [],
      "operations": [
        {
          "type": "RUN COMMANDS",
          "path": ".",
          "reason": "verify",
          "commands": []
        }
      ]
    }
    """
    with pytest.raises(ValueError, match="RUN COMMANDS requires at least one command"):
        ResponseParser().parse(raw)


def test_parser_requires_new_path_for_move() -> None:
    raw = """
    {
      "summary": "Bad move plan",
      "notes": [],
      "operations": [
        {
          "type": "MOVE FILE",
          "path": "src/app.py",
          "reason": "reorganize"
        }
      ]
    }
    """
    with pytest.raises(ValueError, match="MOVE FILE requires new_path"):
        ResponseParser().parse(raw)


def test_guard_warns_for_command_operations(tmp_path: Path) -> None:
    raw = """
    {
      "summary": "Verify tests",
      "notes": [],
      "operations": [
        {
          "type": "RUN COMMANDS",
          "path": ".",
          "reason": "verify",
          "commands": ["git status --short"]
        }
      ]
    }
    """
    plan = ResponseParser().parse(raw)
    warnings = SafetyGuard(tmp_path).validate(plan)
    assert any("Command approval required" in item for item in warnings)
