from pathlib import Path

from aster.context_collector.collector import CollectedContext, ContextFile
from aster.prompt_builder import PromptBuilder
from aster.session import SessionTurn


def test_browser_prompt_is_compacted_below_limit() -> None:
    context = CollectedContext(
        project_root=Path("C:/demo"),
        project_summary="Project root: C:/demo. Visible files: 20. Dominant file types: .py:20.",
        file_tree="demo/\n" + "\n".join(f"- file_{index}.py" for index in range(20)),
        relevant_files=[
            ContextFile(
                path=f"src/file_{index}.py",
                reason="source_or_related_file",
                content=("print('line')\n" * 800) + f"# file {index}",
            )
            for index in range(12)
        ],
        skipped_files=["logs/runtime.log"],
    )
    history = [SessionTurn(role="user", content="previous request" * 100)]

    package = PromptBuilder().build(
        "make me a calculator app",
        context,
        history,
        mode="browser",
        max_chars=8_000,
    )

    assert package.approx_chars <= 8_000
    assert package.compacted is True
    assert package.included_files
    assert "ASTER_PATCH_BEGIN" in package.messages[-1]["content"]
    assert "NEED THESE FILES FIRST" in package.messages[-1]["content"]
