"""Unit tests for the NDJSON event models, TurnEvents, and NdjsonEncoder.

Covers Req 4.4, 4.8, 7.1, 7.4, 7.9, 9.4, 11.6, 12.4. Every secret is an obviously fake
placeholder.
"""

import base64
import json
import math
from datetime import date
from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError

from friday.data.dataset import Dataset
from friday.data.stats import STATS_LABELS, UNDEFINED_TEXT, describe
from friday.errors import AwsCredentialError, TTSError
from friday.events import (
    AUDIO_MIME,
    PNG_DATA_URL_PREFIX,
    Audio,
    ChartEvent,
    NdjsonEncoder,
    StatusEvent,
    TurnEvents,
    audio_error_kind,
    csv_url,
)
from friday.redact import REDACTED, Redactor

SESSION = "123e4567-e89b-42d3-a456-426614174000"
FAKE_SECRET = "FAKEsecretVALUEforTESTSonly0123456789xyz"  # noqa: S105 - fake, 40 chars
FAKE_KEY_ID = "AKIAFAKEFAKEFAKEFAKE"  # AWS key-ID shape, fake


def _dataset(values: list[float], *, derived_from: str | None = None) -> Dataset:
    dates = tuple(date(2020 + i // 12, i % 12 + 1, 1) for i in range(len(values)))
    return Dataset(
        dataset_id="ds-1",
        indicator="inflation",
        value_column="inflation",
        series_id="CPIAUCSL",
        dates=dates,
        values=np.array(values, dtype=np.float64),
        transformation="yoy",
        derived_from=derived_from,
    )


def _strict_loads(line: bytes) -> dict[str, Any]:
    """Parse one line, failing on NaN/Infinity tokens (not valid strict JSON)."""

    def reject(token: str) -> None:
        raise AssertionError(f"non-strict JSON token {token}")

    assert line.endswith(b"\n")
    assert line.count(b"\n") == 1
    return json.loads(line, parse_constant=reject)


@pytest.fixture
def encoder() -> NdjsonEncoder:
    return NdjsonEncoder(Redactor([FAKE_SECRET]))


# --- seq -----------------------------------------------------------------------


def test_seq_is_consecutive_from_one_across_event_types() -> None:
    turn = TurnEvents(SESSION)
    ds = _dataset([1.0, 2.0])
    events = [
        turn.status("tool_start", "fetch_data", "Fetching inflation data, Boss."),
        turn.dataset_preview(ds),
        turn.status("tool_done", "fetch_data", "Done."),
        turn.stats_table(ds, describe(ds)),
        turn.chart("chart-1", b"png", "alt"),
        turn.final("ok", "Here you go, Boss.", "Here you go, Boss."),
    ]
    assert [e.seq for e in events] == [1, 2, 3, 4, 5, 6]
    assert turn.last_seq == 6


def test_each_turn_has_its_own_counter() -> None:
    first, second = TurnEvents(SESSION), TurnEvents(SESSION)
    first.status("tool_start", "fetch_data", "a")
    first.status("tool_done", "fetch_data", "b")
    assert second.final("ok", "Boss.", "Boss.").seq == 1
    assert TurnEvents(SESSION).last_seq == 0


@pytest.mark.parametrize("seq", [0, -1, 1.0, math.inf, math.nan, True])
def test_seq_must_be_a_positive_int(seq: object) -> None:
    with pytest.raises(ValidationError):
        StatusEvent.model_validate({"seq": seq, "phase": "tool_start", "tool": "x", "text": "t"})


# --- status and final ----------------------------------------------------------


def test_status_event_wire_shape_with_audio(encoder: NdjsonEncoder) -> None:
    event = TurnEvents(SESSION).status(
        "tool_start", "fetch_data", "Fetching inflation data, Boss.", audio=b"\xff\xfbmp3"
    )
    assert _strict_loads(encoder.encode(event)) == {
        "type": "status",
        "seq": 1,
        "phase": "tool_start",
        "tool": "fetch_data",
        "text": "Fetching inflation data, Boss.",
        "audio": {"mime": AUDIO_MIME, "b64": base64.b64encode(b"\xff\xfbmp3").decode()},
        "audio_error": None,
    }


def test_status_event_without_audio_reports_the_error(encoder: NdjsonEncoder) -> None:
    event = TurnEvents(SESSION).status("tool_error", "plot_data", "x", audio_error="tts_timeout")
    wire = _strict_loads(encoder.encode(event))
    assert wire["audio"] is None
    assert wire["audio_error"] == "tts_timeout"


def test_audio_and_audio_error_are_mutually_exclusive() -> None:
    turn = TurnEvents(SESSION)
    with pytest.raises(ValidationError):
        turn.status("tool_done", "fetch_data", "Done.", audio=b"x", audio_error="tts_failed")
    with pytest.raises(ValidationError):
        turn.final("ok", "t", "s", audio=b"x", audio_error="tts_failed")


def test_final_event_wire_shape(encoder: NdjsonEncoder) -> None:
    turn = TurnEvents(SESSION)
    turn.status("tool_start", "fetch_data", "a")
    event = turn.final("aws_credentials", "Boss, refresh .env.", "")
    assert _strict_loads(encoder.encode(event)) == {
        "type": "final",
        "seq": 2,
        "outcome": "aws_credentials",
        "text": "Boss, refresh .env.",
        "spoken_text": "",
        "audio": None,
        "audio_error": None,
    }


def test_unknown_phase_outcome_and_audio_error_are_rejected() -> None:
    base = {"seq": 1, "tool": "x", "text": "t"}
    with pytest.raises(ValidationError):
        StatusEvent.model_validate({**base, "phase": "tool_maybe"})
    with pytest.raises(ValidationError):
        StatusEvent.model_validate({**base, "phase": "tool_done", "audio_error": "boom"})
    with pytest.raises(ValidationError):
        TurnEvents(SESSION).final("finished", "t", "s")  # type: ignore[arg-type]


def test_audio_error_kind_uses_the_error_kind() -> None:
    assert audio_error_kind(TTSError("tts_failed", "x")) == "tts_failed"
    assert audio_error_kind(TTSError("tts_timeout", "x")) == "tts_timeout"
    assert audio_error_kind(AwsCredentialError()) == "aws_credentials"


def test_events_are_frozen() -> None:
    event = TurnEvents(SESSION).status("tool_start", "fetch_data", "t")
    with pytest.raises(ValidationError):
        event.text = "changed"  # type: ignore[misc]


# --- dataset_preview -------------------------------------------------------------


def test_dataset_preview_wire_shape(encoder: NdjsonEncoder) -> None:
    values = [1.26583, math.nan, *[float(i) for i in range(10)]]
    event = TurnEvents(SESSION).dataset_preview(_dataset(values, derived_from="ds-0"))
    wire = _strict_loads(encoder.encode(event))
    assert wire["type"] == "dataset_preview"
    assert wire["dataset_id"] == "ds-1"
    assert wire["indicator"] == "inflation"
    assert wire["series_id"] == "CPIAUCSL"
    assert wire["row_count"] == 12
    assert (wire["first_date"], wire["last_date"]) == ("2020-01-01", "2020-12-01")
    assert wire["columns"] == ["date", "inflation"]
    assert len(wire["rows"]) == 10
    assert wire["rows"][:2] == [["2020-01-01", "1.2658"], ["2020-02-01", ""]]
    assert wire["csv_url"] == f"/api/datasets/ds-1.csv?session={SESSION}"
    assert wire["derived_from"] == "ds-0"
    assert wire["empty"] is False


def test_empty_dataset_preview_is_flagged(encoder: NdjsonEncoder) -> None:
    wire = _strict_loads(encoder.encode(TurnEvents(SESSION).dataset_preview(_dataset([]))))
    assert wire["empty"] is True
    assert wire["rows"] == []
    assert wire["row_count"] == 0
    assert wire["first_date"] is None
    assert wire["last_date"] is None
    assert wire["derived_from"] is None


def test_non_finite_values_encode_as_strict_json(encoder: NdjsonEncoder) -> None:
    ds = _dataset([math.inf, -math.inf, math.nan, -5e-324])
    turn = TurnEvents(SESSION)
    preview = _strict_loads(encoder.encode(turn.dataset_preview(ds)))
    assert [value for _, value in preview["rows"]] == ["inf", "-inf", "", "0"]
    stats = _strict_loads(encoder.encode(turn.stats_table(ds, describe(ds))))
    assert stats["rows"][2] == {"label": "Mean", "value": UNDEFINED_TEXT}


def test_csv_url_quotes_its_parts() -> None:
    assert csv_url("ds 1/x", "a&b") == "/api/datasets/ds%201%2Fx.csv?session=a%26b"


# --- stats_table and chart -------------------------------------------------------


def test_stats_table_has_the_eleven_formatted_rows(encoder: NdjsonEncoder) -> None:
    ds = _dataset([1.0, 2.0, math.nan, 4.0])
    wire = _strict_loads(encoder.encode(TurnEvents(SESSION).stats_table(ds, describe(ds))))
    assert wire["type"] == "stats_table"
    assert (wire["dataset_id"], wire["indicator"]) == ("ds-1", "inflation")
    assert [row["label"] for row in wire["rows"]] == list(STATS_LABELS)
    assert wire["rows"][0] == {"label": "Count", "value": "3"}
    assert wire["rows"][1] == {"label": "Missing values", "value": "1"}
    assert wire["rows"][2] == {"label": "Mean", "value": "2.33"}
    assert wire["rows"][-1] == {"label": "Last date", "value": "2020-04-01"}


def test_chart_event_carries_a_png_data_url(encoder: NdjsonEncoder) -> None:
    png = b"\x89PNG\r\n\x1a\nrest"
    alt = "Line chart of inflation from 2020-01-01 to 2020-04-01"
    wire = _strict_loads(encoder.encode(TurnEvents(SESSION).chart("chart-1", png, alt)))
    assert set(wire) == {"type", "seq", "chart_id", "image", "alt"}
    assert wire["chart_id"] == "chart-1"
    assert wire["alt"] == alt
    assert wire["image"].startswith(PNG_DATA_URL_PREFIX)
    assert base64.b64decode(wire["image"].removeprefix(PNG_DATA_URL_PREFIX)) == png


# --- encoder: redaction and line format --------------------------------------------


def test_credentials_are_redacted_in_every_text_field(encoder: NdjsonEncoder) -> None:
    turn = TurnEvents(SESSION)
    fragment = FAKE_SECRET[5:20]
    events = [
        turn.status("tool_error", "fetch_data", f"failed with {FAKE_SECRET}"),
        turn.final("ok", f"text {fragment}", f"spoken {FAKE_KEY_ID}"),
        turn.chart("chart-1", b"png", f"alt api_key={FAKE_SECRET[:8]}x"),
    ]
    lines = b"".join(encoder.encode(event) for event in events).decode()
    assert FAKE_SECRET not in lines
    assert fragment not in lines
    assert FAKE_KEY_ID not in lines
    assert lines.count(REDACTED) == 4
    for line in lines.splitlines(keepends=True):
        _strict_loads(line.encode())


def test_redaction_keeps_the_line_valid_json_around_escapes(encoder: NdjsonEncoder) -> None:
    event = TurnEvents(SESSION).final("ok", 'see api_key=abc"\\ and\nmore', "x")
    wire = _strict_loads(encoder.encode(event))
    assert wire["text"] == f'see api_key={REDACTED}"\\ and\nmore'


def test_base64_payloads_are_not_redacted(encoder: NdjsonEncoder) -> None:
    # Bytes whose base64 contains an AWS key-ID shape and a secret window.
    payload = base64.b64decode(FAKE_KEY_ID + FAKE_SECRET)
    turn = TurnEvents(SESSION)
    chart = _strict_loads(encoder.encode(turn.chart("chart-1", payload, "alt")))
    status = _strict_loads(encoder.encode(turn.status("tool_done", "x", "Done.", audio=payload)))
    assert base64.b64decode(chart["image"].removeprefix(PNG_DATA_URL_PREFIX)) == payload
    assert base64.b64decode(status["audio"]["b64"]) == payload


def test_lines_are_ascii_and_single_line(encoder: NdjsonEncoder) -> None:
    text = "Boss\u2028line\nbreak \u00e9 \ud800"
    line = encoder.encode(TurnEvents(SESSION).final("ok", text, text))
    line.decode("ascii")
    assert _strict_loads(line)["text"] == text


def test_audio_model_round_trips_mp3_bytes() -> None:
    audio = Audio.from_mp3(b"\x00\x01mp3")
    assert audio.mime == AUDIO_MIME
    assert base64.b64decode(audio.b64) == b"\x00\x01mp3"


def test_chart_event_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        ChartEvent.model_validate(
            {"seq": 1, "chart_id": "c", "image": "i", "alt": "a", "extra": True}
        )
