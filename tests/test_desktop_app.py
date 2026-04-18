from aster.ui_or_cli.desktop_app import _format_activity_entry, _format_gui_timestamp, _format_trace_entry


def test_format_gui_timestamp_converts_utc_to_central() -> None:
    assert _format_gui_timestamp("2026-04-18T00:09:25.448403+00:00") == "7:09:25 PM CT"


def test_format_activity_entry_uses_central_time_and_details() -> None:
    event = {
        "ts": "2026-04-18T00:06:55.023827+00:00",
        "kind": "activity",
        "payload": {
            "message": "Starting a browser planning run.",
            "why": "Aster needs to collect context first.",
            "status": "info",
            "details": {"attempt_index": 1, "approx_chars": 27469},
        },
    }

    formatted = _format_activity_entry(event)

    assert formatted is not None
    assert "[7:06:55 PM CT] Starting a browser planning run." in formatted
    assert "Why: Aster needs to collect context first." in formatted
    assert "Details: attempt_index=1, approx_chars=27469" in formatted


def test_format_trace_entry_summarizes_browser_transport() -> None:
    event = {
        "ts": "2026-04-18T00:09:04.310155+00:00",
        "kind": "browser_transport",
        "payload": {
            "event": "window_ready_confirmed",
            "ui_state": {
                "window_title": "Calculator GUI Creation - Google Chrome",
                "send_prompt_present": False,
                "send_prompt_enabled": None,
                "show_in_text_field_present": False,
                "stop_streaming_present": False,
                "composer_edit_length": 12,
            },
            "ocr_preview": ["No LeftX", "Github|X", "WeightX", "Calculat", "#movie"],
        },
    }

    formatted = _format_trace_entry(event)

    assert formatted is not None
    assert "[7:09:04 PM CT] window_ready_confirmed" in formatted
    assert "title=Calculator GUI Creation - Google Chrome" in formatted
    assert "send=False/None" in formatted
    assert "Preview: No LeftX | Github|X | WeightX | Calculat | #movie" in formatted
