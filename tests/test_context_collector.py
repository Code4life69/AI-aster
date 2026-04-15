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
