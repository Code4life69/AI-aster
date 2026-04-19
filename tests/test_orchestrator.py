from pathlib import Path

from aster.config.models import AsterConfig
from aster.orchestrator import AsterOrchestrator
from aster.response_parser import ParsedPlan, PatchOperation


class _FakeApplier:
    def apply(self, plan, selected_indices=None, dry_run=True):
        return ["applied"]


class _FakeGit:
    def __init__(self) -> None:
        self.commit_messages: list[str] = []
        self.runtime_commit_messages: list[str] = []
        self.runtime_commit_paths: list[list[str]] = []
        self.push_calls = 0
        self.pull_calls = 0

    def sync_pull(self) -> list[str]:
        self.pull_calls += 1
        return []

    def commit_all_if_needed(self, message: str) -> list[str]:
        self.commit_messages.append(message)
        return [f"commit {message}"]

    def commit_paths_if_needed(self, message: str, paths: list[str]) -> list[str]:
        self.runtime_commit_messages.append(message)
        self.runtime_commit_paths.append(list(paths))
        return [f"commit selected {message}"]

    def sync_push(self) -> list[str]:
        self.push_calls += 1
        return ["push"]


class _FakeBrowserResult:
    def __init__(self, raw_text: str) -> None:
        self.raw_text = raw_text
        self.metadata = {"window_title": "Fake ChatGPT"}


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
