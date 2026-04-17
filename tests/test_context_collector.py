from pathlib import Path

from aster.context_collector import ContextCollector


def test_context_collector_ignores_env_and_keeps_source(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("OPENAI_API_KEY=secret-value", encoding="utf-8")
    (tmp_path / "app.py").write_text("print('hello')", encoding="utf-8")
    (tmp_path / "README.md").write_text("# Demo", encoding="utf-8")

    collector = ContextCollector(ignore_patterns=[".env"], max_file_bytes=10_000, max_total_prompt_bytes=50_000)
    context = collector.collect(tmp_path, "fix the app")

    paths = [item.path for item in context.relevant_files]
    assert ".env" not in paths
    assert "app.py" in paths
    assert "README.md" in paths


def test_context_collector_ignores_nested_cache_and_aster_dirs(tmp_path: Path) -> None:
    cache_dir = tmp_path / "pkg" / "__pycache__"
    cache_dir.mkdir(parents=True)
    (cache_dir / "module.cpython-311.pyc").write_bytes(b"\x00\x01\x02")
    audit_dir = tmp_path / ".aster"
    audit_dir.mkdir()
    (audit_dir / "audit.log.jsonl").write_text('{"kind":"prompt_sent"}', encoding="utf-8")
    (tmp_path / "pkg" / "module.py").write_text("print('ok')", encoding="utf-8")

    collector = ContextCollector(ignore_patterns=[".aster/", "__pycache__/"], max_file_bytes=10_000, max_total_prompt_bytes=50_000)
    context = collector.collect(tmp_path, "update the module")

    assert ".aster/audit.log.jsonl" not in context.file_tree
    assert "__pycache__" not in context.file_tree
    assert "pkg/module.py" in context.file_tree
