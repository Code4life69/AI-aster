from pathlib import Path

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
