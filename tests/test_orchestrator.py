import json
from pathlib import Path

import pytest

from aster.config.models import AsterConfig
from aster.context_collector import CollectedContext
from aster.orchestrator import AsterOrchestrator
from aster.response_parser import ParsedPlan, PatchOperation


class _FakeApplier:
    def apply(self, plan, selected_indices=None, dry_run=True):
        return ["applied"]


class _FakeGit:
    def __init__(self) -> None:
        self.commit_messages: list[str] = []
        self.commit_exclude_paths: list[list[str]] = []
        self.runtime_commit_messages: list[str] = []
        self.runtime_commit_paths: list[list[str]] = []
        self.push_calls = 0
        self.pull_calls = 0
        self.branch = "feature/debug"

    def sync_pull(self) -> list[str]:
        self.pull_calls += 1
        return []

    def commit_all_if_needed(self, message: str, exclude_paths: list[str] | None = None) -> list[str]:
        self.commit_messages.append(message)
        self.commit_exclude_paths.append(list(exclude_paths or []))
        return [
            "git add -A -- . -> 0: ",
            f"git commit -m {message} -> 0: committed",
        ]

    def commit_paths_if_needed(self, message: str, paths: list[str]) -> list[str]:
        self.runtime_commit_messages.append(message)
        self.runtime_commit_paths.append(list(paths))
        joined_paths = " ".join(paths)
        return [
            f"git add -A -- {joined_paths} -> 0: ",
            f"git commit -m {message} -> 0: committed",
        ]

    def sync_push(self) -> list[str]:
        self.push_calls += 1
        return ["push"]

    def current_branch(self) -> str:
        return self.branch


class _FakeBrowserResult:
    def __init__(self, raw_text: str, metadata: dict[str, object] | None = None) -> None:
        self.raw_text = raw_text
        self.metadata = {"window_title": "Fake ChatGPT", **(metadata or {})}


class _RetryingBrowserTransport:
    def __init__(self) -> None:
        self.calls = 0
        self.prompts: list[str] = []

    def generate(self, prompt: str, **kwargs) -> _FakeBrowserResult:
        self.calls += 1
        self.prompts.append(prompt)
        if self.calls == 1:
            raise RuntimeError("ChatGPT browser rejected the prompt as too large.")
        return _FakeBrowserResult(
            '{"summary":"ok","notes":[],"operations":[{"type":"NEED THESE FILES FIRST","path":"README.md","reason":"need context"}]}'
        )


class _BrowserTransport:
    def __init__(self, raw_text: str) -> None:
        self.raw_text = raw_text

    def generate(self, prompt: str, **kwargs) -> _FakeBrowserResult:
        return _FakeBrowserResult(self.raw_text)


class _CollectingLogger:
    def __init__(self) -> None:
        self.entries: list[tuple[str, dict[str, object]]] = []

    def log(self, namespace: str, payload: dict[str, object]) -> None:
        self.entries.append((namespace, payload))

    def activity(self, *args, **kwargs) -> None:
        return None


def _config(tmp_path: Path) -> AsterConfig:
    return AsterConfig(
        project_root=tmp_path,
        log_dir=tmp_path / ".aster",
        sync_with_remote=True,
    )


def test_apply_pushes_command_operations_after_approval(tmp_path: Path) -> None:
    config = _config(tmp_path)
    orchestrator = AsterOrchestrator(config)
    orchestrator.applier = _FakeApplier()
    orchestrator.git = _FakeGit()
    plan = ParsedPlan(
        summary="Run verification",
        notes=[],
        operations=[
            PatchOperation(
                type="RUN COMMANDS",
                path=".",
                reason="verify",
                commands=["git status --short"],
            )
        ],
    )

    results = orchestrator.apply(plan, dry_run=False)

    assert orchestrator.git.commit_messages
    assert orchestrator.git.commit_exclude_paths[-1] == list(AsterOrchestrator.RUNTIME_LOG_PATHS)
    assert orchestrator.git.push_calls == 1
    assert any("push" in item for item in results)


def test_apply_pushes_safe_file_only_plans(tmp_path: Path) -> None:
    config = _config(tmp_path)
    orchestrator = AsterOrchestrator(config)
    orchestrator.applier = _FakeApplier()
    orchestrator.git = _FakeGit()
    plan = ParsedPlan(
        summary="Edit app",
        notes=[],
        operations=[
            PatchOperation(
                type="EDIT FILE",
                path="app.py",
                reason="fix",
                content="print('ok')",
            )
        ],
    )

    orchestrator.apply(plan, dry_run=False)

    assert orchestrator.git.commit_messages
    assert orchestrator.git.commit_exclude_paths[-1] == list(AsterOrchestrator.RUNTIME_LOG_PATHS)
    assert orchestrator.git.push_calls == 1


def test_browser_plan_retries_with_smaller_prompt_after_size_error(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("# Demo", encoding="utf-8")
    for index in range(12):
        (tmp_path / f"module_{index}.py").write_text(("print('hello')\n" * 350), encoding="utf-8")
    config = _config(tmp_path)
    config.sync_with_remote = False
    orchestrator = AsterOrchestrator(config)
    browser = _RetryingBrowserTransport()
    orchestrator.browser_transport = browser

    result = orchestrator.plan("build a calculator app", mode="browser")

    assert browser.calls == 2
    assert len(browser.prompts[1]) < len(browser.prompts[0])
    assert result.plan.requires_more_files() is True


def test_plan_pushes_runtime_logs_after_generation(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("# Demo", encoding="utf-8")
    (tmp_path / "app.py").write_text("print('hello')", encoding="utf-8")
    config = _config(tmp_path)
    orchestrator = AsterOrchestrator(config)
    orchestrator.git = _FakeGit()
    orchestrator.browser_transport = _BrowserTransport(
        '{"summary":"ok","notes":[],"operations":[{"type":"NEED THESE FILES FIRST","path":"README.md","reason":"need context"}]}'
    )

    result = orchestrator.plan("inspect the logs", mode="browser")

    assert result.plan.requires_more_files() is True
    assert orchestrator.git.pull_calls == 1
    assert orchestrator.git.runtime_commit_messages[-1] == "sync runtime logs after plan"
    assert orchestrator.git.runtime_commit_paths[-1] == list(AsterOrchestrator.RUNTIME_LOG_PATHS)
    assert orchestrator.git.push_calls == 1


def test_sync_runtime_logs_skips_push_when_no_runtime_log_changes_exist(tmp_path: Path) -> None:
    class _NoChangeGit(_FakeGit):
        def commit_paths_if_needed(self, message: str, paths: list[str]) -> list[str]:
            self.runtime_commit_messages.append(message)
            self.runtime_commit_paths.append(list(paths))
            return ["Git commit skipped: no changes detected in selected paths."]

    config = _config(tmp_path)
    orchestrator = AsterOrchestrator(config)
    orchestrator.git = _NoChangeGit()

    results = orchestrator.sync_runtime_logs("debug heartbeat runtime logs")

    assert orchestrator.git.runtime_commit_messages[-1] == "debug heartbeat runtime logs"
    assert orchestrator.git.push_calls == 0
    assert results == ["Git commit skipped: no changes detected in selected paths."]


def test_orchestrator_wires_runtime_log_heartbeat_into_browser_transport(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config.runtime_log_heartbeat_push_enabled = True
    config.runtime_log_heartbeat_interval_seconds = 45
    config.runtime_log_heartbeat_branch_only = False

    orchestrator = AsterOrchestrator(config)

    assert orchestrator.runtime_log_heartbeat.enabled is True
    assert orchestrator.runtime_log_heartbeat.interval_seconds == 45
    assert orchestrator.runtime_log_heartbeat.branch_only is False
    assert orchestrator.browser_transport._runtime_log_heartbeat is orchestrator.runtime_log_heartbeat


def test_parse_with_retry_omits_invalid_browser_retry_seed(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config.sync_with_remote = False
    orchestrator = AsterOrchestrator(config)
    logger = _CollectingLogger()
    orchestrator.logger = logger
    orchestrator._last_generation_metadata = {
        "retry_seed_valid": False,
        "retry_seed_validity_reason": "unbalanced_structure",
        "retry_seed_validity_reason_source": "best_salvageable_candidate",
    }
    calls: list[str | None] = []

    def _parse(raw_text: str):
        if raw_text == "":
            raise ValueError("not parseable")
        return ParsedPlan(
            summary="ok",
            notes=[],
            operations=[
                PatchOperation(
                    type="NEED THESE FILES FIRST",
                    path="README.md",
                    reason="need context",
                )
            ],
        )

    orchestrator.parser.parse = _parse

    def _retry_generate(*_args, **kwargs):
        calls.append(kwargs.get("prior_text"))
        calls.append(kwargs.get("retry_reason"))
        calls.append(kwargs.get("retry_seed_used"))

        class _PromptPackage:
            messages = [{"role": "user", "content": "retry"}]
            approx_chars = 5
            included_files = []
            omitted_files = []
            compacted = False
            retry_prompt_mode = "browser_structured_output_only_retry"
            retry_prompt_strategy = "browser_retry_structured_output_only"
            retry_prompt_reason = "unbalanced_structure"
            retry_prompt_length = 5
            retry_prompt_compacted_relative_to_original = True

        return (
            '{"summary":"ok","notes":[],"operations":[{"type":"NEED THESE FILES FIRST","path":"README.md","reason":"need context"}]}',
            _PromptPackage(),
        )

    orchestrator._generate_with_prompt_retries = _retry_generate
    context = CollectedContext(
        project_root=tmp_path,
        project_summary="demo",
        file_tree="demo/",
        relevant_files=[],
        skipped_files=[],
    )

    plan = orchestrator._parse_with_retry("build app", context, [], "browser", "")

    assert calls == [None, "unbalanced_structure", False]
    assert plan.requires_more_files() is True
    prompt_retry_payload = next(payload for namespace, payload in logger.entries if namespace == "prompt_retry")
    assert prompt_retry_payload["retry_seed_used"] is False
    assert prompt_retry_payload["retry_seed_validity_reason"] == "unbalanced_structure"
    assert prompt_retry_payload["retry_seed_validity_reason_source"] == "best_salvageable_candidate"
    assert prompt_retry_payload["retry_prior_response_length"] == 0
    assert prompt_retry_payload["retry_prompt_mode"] == "browser_structured_output_only_retry"
    assert prompt_retry_payload["retry_prompt_strategy"] == "browser_retry_structured_output_only"
    assert prompt_retry_payload["retry_prompt_reason"] == "unbalanced_structure"


def test_parse_with_retry_wraps_non_parseable_browser_retry_response(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config.sync_with_remote = False
    orchestrator = AsterOrchestrator(config)
    logger = _CollectingLogger()
    orchestrator.logger = logger
    orchestrator._last_generation_metadata = {
        "retry_seed_valid": False,
        "retry_seed_validity_reason": "missing_required_schema_keys",
        "retry_seed_validity_reason_source": "best_salvageable_candidate",
    }
    retry_calls: list[str | None] = []

    def _parse(raw_text: str):
        raise json.JSONDecodeError("bad json", raw_text or "", 0)

    orchestrator.parser.parse = _parse

    def _retry_generate(*_args, **kwargs):
        retry_calls.append(kwargs.get("prior_text"))
        retry_calls.append(kwargs.get("retry_reason"))
        retry_calls.append(kwargs.get("retry_seed_used"))

        class _PromptPackage:
            messages = [{"role": "user", "content": "retry"}]
            approx_chars = 5
            included_files = []
            omitted_files = []
            compacted = False
            retry_prompt_mode = "browser_structured_output_only_retry"
            retry_prompt_strategy = "browser_retry_structured_output_only"
            retry_prompt_reason = "missing_required_schema_keys"
            retry_prompt_length = 5
            retry_prompt_compacted_relative_to_original = True

        return ("assistant reply without any structured payload", _PromptPackage())

    orchestrator._generate_with_prompt_retries = _retry_generate
    context = CollectedContext(
        project_root=tmp_path,
        project_summary="demo",
        file_tree="demo/",
        relevant_files=[],
        skipped_files=[],
    )

    with pytest.raises(RuntimeError, match="Browser retry response was not machine-parseable"):
        orchestrator._parse_with_retry("build app", context, [], "browser", "")

    assert retry_calls == [None, "missing_required_schema_keys", False]
    retry_parse_failure = next(payload for namespace, payload in logger.entries if namespace == "retry_parse_failure")
    assert retry_parse_failure["retry_seed_used"] is False
    assert retry_parse_failure["retry_prior_response_length"] == 0
    assert retry_parse_failure["retry_seed_validity_reason"] == "missing_required_schema_keys"
    assert retry_parse_failure["retry_seed_validity_reason_source"] == "best_salvageable_candidate"
    assert retry_parse_failure["retry_parse_text_source"] == "retry_text"
    assert retry_parse_failure["retry_parse_failure_kind"] == "malformed_structured_text"


def test_parse_with_retry_uses_retry_attempt_failure_reason_in_browser_error(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config.sync_with_remote = False
    orchestrator = AsterOrchestrator(config)
    orchestrator._last_generation_metadata = {
        "retry_seed_valid": False,
        "retry_seed_validity_reason": "unbalanced_structure",
        "retry_seed_validity_reason_source": "best_salvageable_candidate",
    }

    def _parse(raw_text: str):
        raise json.JSONDecodeError("bad json", raw_text or "", 0)

    orchestrator.parser.parse = _parse

    def _retry_generate(*_args, **_kwargs):
        orchestrator._last_generation_metadata = {
            "retry_attempt_capture_mode": "structured_block_first",
            "retry_attempt_acceptance_tier": "blocked_or_ambiguous",
            "retry_attempt_failure_reason": "wrapper_without_valid_json",
            "retry_attempt_structured_block_found": True,
            "retry_attempt_parseable": False,
            "retry_attempt_prose_contamination": False,
            "retry_attempt_wrapper_only": True,
        }

        class _PromptPackage:
            messages = [{"role": "user", "content": "retry"}]
            approx_chars = 5
            included_files = []
            omitted_files = []
            compacted = False
            retry_prompt_mode = "browser_structured_output_only_retry"
            retry_prompt_strategy = "browser_retry_structured_output_only"
            retry_prompt_reason = "unbalanced_structure"
            retry_prompt_length = 5
            retry_prompt_compacted_relative_to_original = True

        return ("ASTER_PATCH_BEGIN\n{\"summary\":\"bad\"\nASTER_PATCH_END", _PromptPackage())

    orchestrator._generate_with_prompt_retries = _retry_generate
    context = CollectedContext(
        project_root=tmp_path,
        project_summary="demo",
        file_tree="demo/",
        relevant_files=[],
        skipped_files=[],
    )

    with pytest.raises(RuntimeError, match="wrapper_without_valid_json"):
        orchestrator._parse_with_retry("build app", context, [], "browser", "")


def test_parse_with_retry_uses_fragmented_multi_block_reason_in_browser_error(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config.sync_with_remote = False
    orchestrator = AsterOrchestrator(config)
    orchestrator._last_generation_metadata = {
        "retry_seed_valid": False,
        "retry_seed_validity_reason": "unbalanced_structure",
        "retry_seed_validity_reason_source": "best_salvageable_candidate",
    }

    def _parse(raw_text: str):
        raise json.JSONDecodeError("bad json", raw_text or "", 0)

    orchestrator.parser.parse = _parse

    def _retry_generate(*_args, **_kwargs):
        orchestrator._last_generation_metadata = {
            "retry_attempt_capture_mode": "structured_block_first",
            "retry_attempt_acceptance_tier": "blocked_or_ambiguous",
            "retry_attempt_failure_reason": "fragmented_multi_block_retry_output",
            "retry_attempt_structured_block_found": True,
            "retry_attempt_parseable": False,
            "retry_attempt_prose_contamination": False,
            "retry_attempt_wrapper_only": False,
            "retry_attempt_block_count": 2,
            "retry_attempt_selected_block_index": None,
            "retry_attempt_block_selection_reason": "fragmented_multi_block_fragments",
            "retry_attempt_multiple_blocks_ambiguous": False,
            "retry_attempt_multiple_blocks_recovered": False,
            "retry_attempt_block_relationship": "fragmented_blocks",
            "retry_attempt_block_forensics": [
                {
                    "index": 1,
                    "block_hash": "abc123",
                    "preview": "ASTER_PATCH_BEGIN {\"summary\":\"bad\"",
                    "raw_length": 80,
                    "extracted_length": 30,
                    "has_end_marker": False,
                    "parseable": False,
                    "payload_state": "partial_payload",
                    "block_kind": "fragment_without_end_marker",
                    "schema_hits": 2,
                    "brace_balance": 1,
                    "bracket_balance": 0,
                }
            ],
        }

        class _PromptPackage:
            messages = [{"role": "user", "content": "retry"}]
            approx_chars = 5
            included_files = []
            omitted_files = []
            compacted = False
            retry_prompt_mode = "browser_structured_output_only_retry"
            retry_prompt_strategy = "browser_retry_structured_output_only"
            retry_prompt_reason = "unbalanced_structure"
            retry_prompt_length = 5
            retry_prompt_compacted_relative_to_original = True

        return ("ASTER_PATCH_BEGIN\n{\"summary\":\"bad\"", _PromptPackage())

    orchestrator._generate_with_prompt_retries = _retry_generate
    context = CollectedContext(
        project_root=tmp_path,
        project_summary="demo",
        file_tree="demo/",
        relevant_files=[],
        skipped_files=[],
    )

    with pytest.raises(RuntimeError, match="fragmented_multi_block_retry_output"):
        orchestrator._parse_with_retry("build app", context, [], "browser", "")
