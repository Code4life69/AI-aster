from pathlib import Path

from aster.visual_action_memory import (
    NormalizedRegion,
    VisualActionDebugSession,
    VisualRegionMemoryStore,
    build_visual_manifest_entry,
    build_visual_trace_paths,
    decay_confidence,
    denormalize_region,
    normalize_region,
)


def test_normalize_and_denormalize_region_round_trip() -> None:
    window_rect = (100, 200, 1100, 1000)
    rect = (250, 520, 950, 820)

    normalized = normalize_region(rect, window_rect)
    restored = denormalize_region(normalized, window_rect)

    assert normalized.left == 0.15
    assert normalized.top == 0.4
    assert restored == rect


def test_decay_confidence_reduces_score_over_time() -> None:
    assert decay_confidence(0.8, elapsed_seconds=0.0) == 0.8
    assert round(decay_confidence(0.8, elapsed_seconds=7 * 24 * 60 * 60), 3) == 0.4


def test_visual_memory_store_load_save_and_select_hint(tmp_path: Path) -> None:
    path = tmp_path / ".aster" / "visual_memory.json"
    store = VisualRegionMemoryStore(path)
    store.remember_success(
        intent="composer_area",
        region=NormalizedRegion(0.2, 0.7, 0.6, 0.15),
        window_title_pattern="ChatGPT",
        page_state="chatgpt_composer_visible",
        action_type="composer_focus",
        uia_hints=("Ask anything",),
        ocr_hints=("Ask anything",),
        now=1000.0,
    )
    store.save()

    reloaded = VisualRegionMemoryStore(path)
    reloaded.load()
    hint = reloaded.select_hint(
        intent="composer_area",
        window_title="ChatGPT - Google Chrome",
        page_state="chatgpt_composer_visible",
        uia_hints=("Ask anything",),
        ocr_hints=("Ask anything",),
        now=1005.0,
    )

    assert hint is not None
    assert hint.title_match is True
    assert hint.page_state_match is True
    assert hint.score > hint.confidence


def test_visual_memory_store_invalidate_and_merge_success(tmp_path: Path) -> None:
    path = tmp_path / "memory.json"
    store = VisualRegionMemoryStore(path)
    store.remember_success(
        intent="reply_region",
        region=NormalizedRegion(0.4, 0.2, 0.45, 0.5),
        window_title_pattern="ChatGPT",
        page_state="chatgpt_visible",
        action_type="reply_focus",
        now=10.0,
    )
    original_left = store.entries[0].region.left
    updated = store.remember_success(
        intent="reply_region",
        region=NormalizedRegion(0.42, 0.22, 0.4, 0.48),
        window_title_pattern="ChatGPT",
        page_state="chatgpt_visible",
        action_type="reply_capture_region",
        now=20.0,
    )
    confidence_before_invalidation = updated.confidence
    invalidated = store.invalidate(
        intent="reply_region",
        window_title_pattern="ChatGPT",
        page_state="chatgpt_visible",
        reason_weight=0.2,
        now=30.0,
    )

    assert updated.success_count == 2
    assert "reply_focus" in updated.successful_action_types
    assert "reply_capture_region" in updated.successful_action_types
    assert updated.region.left > original_left
    assert invalidated is not None
    assert invalidated.confidence < confidence_before_invalidation
    assert invalidated.failure_count == 1


def test_visual_memory_select_hint_prefers_matching_title_and_state(tmp_path: Path) -> None:
    store = VisualRegionMemoryStore(tmp_path / "memory.json")
    store.entries = [
        store.remember_success(
            intent="send_button_area",
            region=NormalizedRegion(0.9, 0.86, 0.05, 0.06),
            window_title_pattern="ChatGPT",
            page_state="chatgpt_composer_visible",
            action_type="click_action",
            now=100.0,
        ),
        store.remember_success(
            intent="send_button_area",
            region=NormalizedRegion(0.1, 0.1, 0.05, 0.06),
            window_title_pattern="Some Other App",
            page_state="browser_unknown",
            action_type="click_action",
            now=100.0,
        ),
    ]

    hint = store.select_hint(
        intent="send_button_area",
        window_title="ChatGPT - Google Chrome",
        page_state="chatgpt_composer_visible",
        now=110.0,
    )

    assert hint is not None
    assert hint.region.left > 0.5


def test_visual_trace_paths_and_manifest_entry() -> None:
    step = build_visual_trace_paths(Path(".aster/visual_trace"), "run123", "0001_click")
    entry = build_visual_manifest_entry(
        run_id="run123",
        step_id="0001_click",
        timestamp="2026-04-21T12:00:00Z",
        stage="post",
        action_type="click_action",
        target_intent="composer_area",
        window_title="ChatGPT - Google Chrome",
        cursor_before=(10, 20),
        cursor_during=(30, 40),
        cursor_after=(50, 60),
        target_rect=(100, 200, 300, 400),
        candidate_rects=[(100, 200, 300, 400)],
        ui_summary={"send_prompt_present": True},
        ocr_summary=["Ask anything"],
        page_summary={"looks_like_chatgpt": True},
        screenshot_paths={"pre": "pre.png", "during": "during.png", "post": "post.png"},
        action_outcome="completed",
        confidence_before=0.5,
        confidence_after=0.9,
        used_remembered_region=True,
        updated_remembered_region=True,
    )

    assert step.pre_path.name == "0001_click_pre.png"
    assert step.post_path.name == "0001_click_post.png"
    assert entry["target_intent"] == "composer_area"
    assert entry["used_remembered_region"] is True
    assert entry["updated_remembered_region"] is True


def test_visual_action_debug_session_checkpoint_and_manifest(tmp_path: Path) -> None:
    session = VisualActionDebugSession(
        root_dir=tmp_path / ".aster" / "visual_trace",
        enabled=True,
        checkpoint_interval_seconds=10,
        max_checkpoints_per_key=2,
    )
    run_id = session.start_run(run_id="run456")
    step = session.next_step("scroll_action")

    saved = session.save_stage_image(step, "pre", b"fake-image-bytes")
    session.append_manifest_entry(step, {"run_id": run_id, "step_id": step.step_id})

    assert saved is not None
    assert saved.exists()
    assert session.should_capture_checkpoint("reply_capture", now=100.0) is True
    assert session.should_capture_checkpoint("reply_capture", now=105.0) is False
    assert session.should_capture_checkpoint("reply_capture", now=111.0) is True
    assert session.should_capture_checkpoint("reply_capture", now=122.0) is False
    assert step.manifest_path.read_text(encoding="utf-8").strip().startswith("{")
