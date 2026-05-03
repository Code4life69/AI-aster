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


def test_browser_retry_prompt_stays_within_limit() -> None:
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
    history = [SessionTurn(role="user", content="previous request " * 200)]
    prior_text = "non-parseable response " * 500

    package = PromptBuilder().build_retry(
        "make me a calculator app",
        context,
        history,
        prior_text,
        mode="browser",
        max_chars=8_000,
        retry_reason="unbalanced_structure",
        retry_seed_used=True,
    )

    assert package.approx_chars <= 8_000
    assert package.compacted is True
    assert package.messages[-1]["role"] == "user"
    assert "Rejected prior response excerpt:" in package.messages[-1]["content"]
    assert package.retry_prompt_mode == "browser_seeded_retry"
    assert package.retry_prompt_strategy == "browser_retry_with_prior_excerpt"
    assert package.retry_prompt_reason == "unbalanced_structure"
    assert package.retry_prompt_length > 0


def test_browser_retry_without_seed_uses_stricter_prompt_variant() -> None:
    context = CollectedContext(
        project_root=Path("C:/demo"),
        project_summary="Project root: C:/demo. Visible files: 10. Dominant file types: .py:10.",
        file_tree="demo/\n- app.py",
        relevant_files=[
            ContextFile(
                path="app.py",
                reason="source_or_related_file",
                content="print('hello')\n" * 200,
            )
        ],
        skipped_files=["logs/runtime.log"],
    )

    package = PromptBuilder().build_retry(
        "make me a calculator app",
        context,
        [],
        None,
        mode="browser",
        max_chars=5_000,
        retry_reason="missing_required_schema_keys",
        retry_seed_used=False,
    )

    retry_message = package.messages[-1]["content"]
    assert package.approx_chars <= 5_000
    assert package.retry_prompt_mode == "browser_structured_output_only_retry"
    assert package.retry_prompt_strategy == "browser_retry_structured_output_only"
    assert package.retry_prompt_reason == "missing_required_schema_keys"
    assert "Rejected prior response excerpt:" not in retry_message
    assert "Required top-level keys" in retry_message
    assert "Return only one final ASTER_PATCH_BEGIN / ASTER_PATCH_END block." in retry_message
    assert "Any text before or after the block will fail validation." in retry_message
    assert "Never repeat the block" in retry_message


def test_browser_retry_prompt_varies_by_retry_reason() -> None:
    builder = PromptBuilder()
    context = CollectedContext(
        project_root=Path("C:/demo"),
        project_summary="Project root: C:/demo.",
        file_tree="demo/\n- app.py",
        relevant_files=[ContextFile(path="app.py", reason="source_or_related_file", content="print('ok')")],
        skipped_files=[],
    )

    missing_keys = builder.build_retry(
        "make me a calculator app",
        context,
        [],
        None,
        mode="browser",
        max_chars=5_000,
        retry_reason="missing_required_schema_keys",
        retry_seed_used=False,
    )
    unbalanced = builder.build_retry(
        "make me a calculator app",
        context,
        [],
        None,
        mode="browser",
        max_chars=5_000,
        retry_reason="unbalanced_structure",
        retry_seed_used=False,
    )
    multi_block = builder.build_retry(
        "make me a calculator app",
        context,
        [],
        None,
        mode="browser",
        max_chars=5_000,
        retry_reason="multiple_retry_blocks",
        retry_seed_used=False,
    )

    assert "missing required schema keys" in missing_keys.messages[-1]["content"].lower()
    assert "balanced braces/brackets" in unbalanced.messages[-1]["content"].lower()
    assert "returned more than one aster block" in multi_block.messages[-1]["content"].lower()
    assert "never repeat the block" in multi_block.messages[-1]["content"].lower()


def test_browser_wrapper_only_followup_prompt_is_more_explicit() -> None:
    builder = PromptBuilder()
    context = CollectedContext(
        project_root=Path("C:/demo"),
        project_summary="Project root: C:/demo.",
        file_tree="demo/\n- app.py",
        relevant_files=[ContextFile(path="app.py", reason="source_or_related_file", content="print('ok')")],
        skipped_files=[],
    )

    package = builder.build_retry(
        "make me a calculator app",
        context,
        [],
        None,
        mode="browser",
        max_chars=5_000,
        retry_reason="wrapper_only_multi_block_retry_output",
        retry_seed_used=False,
        wrapper_followup=True,
    )

    retry_message = package.messages[-1]["content"].lower()
    assert package.retry_prompt_mode == "browser_wrapper_only_retry_followup"
    assert package.retry_prompt_strategy == "browser_retry_wrapper_only_corrective"
    assert "empty aster wrapper blocks" in retry_message
    assert "must not be empty" in retry_message
    assert "never output an empty wrapper" in retry_message


def test_browser_prose_followup_prompt_is_more_explicit() -> None:
    builder = PromptBuilder()
    context = CollectedContext(
        project_root=Path("C:/demo"),
        project_summary="Project root: C:/demo.",
        file_tree="demo/\n- app.py",
        relevant_files=[ContextFile(path="app.py", reason="source_or_related_file", content="print('ok')")],
        skipped_files=[],
    )

    package = builder.build_retry(
        "make me a calculator app",
        context,
        [],
        None,
        mode="browser",
        max_chars=5_000,
        retry_reason="prose_contaminated_retry_response",
        retry_seed_used=False,
        prose_followup=True,
    )

    retry_message = package.messages[-1]["content"].lower()
    assert package.retry_prompt_mode == "browser_prose_retry_followup"
    assert package.retry_prompt_strategy == "browser_retry_prose_corrective"
    assert "wrapped the answer in extra prose" in retry_message
    assert "no text is allowed before or after the block" in retry_message
    assert "never explain" in retry_message
