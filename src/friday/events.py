"""NDJSON events streamed by ``POST /api/chat`` and their encoder (Req 4.4, 4.8, 7.1, 12.4).

Each line of the stream is one event object with a ``type`` and a per-turn ``seq``
(1, 2, 3, ... in emission order). The five event models below are the single definition
of the wire schema; the field names follow the design's "NDJSON event schema" exactly.
:class:`TurnEvents` is the per-turn factory: it owns the ``seq`` counter (no global state)
and builds every event from domain objects, so display formatting comes from
``data.dataset.preview_rows`` and ``data.stats.format_stats_rows`` only.

Strict JSON. No event field is a float. Display numbers are pre-formatted strings
(a Missing_Value is ``""``, an infinite YoY value is ``"inf"``/``"-inf"`` from
``format_display_value``, an undefined statistic is ``"undefined"``), and the only numbers
are the ints ``seq`` and ``row_count``, validated in strict mode. The encoder also uses
``allow_nan=False``, so a ``NaN``/``Infinity`` token can never be written.

Redaction. :class:`NdjsonEncoder` passes every string in the line through the
:class:`~friday.redact.Redactor` before serialization. Field names are fixed identifiers
and the remaining content is digits and JSON punctuation, so this masks everything a
whole-line pass would, while guaranteeing the line stays valid JSON (a whole-line pass can
swallow the backslash of a JSON escape, e.g. after ``api_key=``). The base64 payloads
(``audio.b64`` and the chart ``image`` data URL) are exempt: they encode MP3/PNG bytes
from Polly and the chart renderer, never text, and the AWS key-ID pattern would otherwise
match inside about one in a million charts and corrupt the image.

This module is application-layer code: stdlib, pydantic, and the domain layer only.
"""

import base64
import json
from typing import Annotated, Final, Literal, cast
from urllib.parse import quote, urlencode

from pydantic import BaseModel, ConfigDict, Field, model_validator

from friday.data.dataset import DATE_COLUMN, Dataset, PreviewRow, preview_rows
from friday.data.stats import Stats, format_stats_rows
from friday.errors import AwsCredentialError, TTSError
from friday.redact import Redactor

AUDIO_MIME: Final = "audio/mpeg"
"""MIME type of every audio clip (Polly MP3)."""

PNG_DATA_URL_PREFIX: Final = "data:image/png;base64,"
"""Prefix of the chart ``image`` data URL."""

DATASET_CSV_PATH: Final = "/api/datasets/{dataset_id}.csv"
"""Route of a CSV_Export download; ``server.py`` serves the same pattern (Req 7.4)."""

BINARY_FIELDS: Final = frozenset({"b64", "image"})
"""Field names that hold base64 payloads and are exempt from redaction."""

Phase = Literal["tool_start", "tool_done", "tool_error"]
"""Status line phase: before a Tool runs, after success, or after a failure."""

AudioErrorKind = Literal["tts_failed", "tts_timeout", "aws_credentials"]
"""Why an event carries no audio (Req 4.8); ``kind`` codes from ``errors.py``."""

Outcome = Literal["ok", "llm_unavailable", "aws_credentials", "tool_limit", "no_data"]
"""How a turn ended (``final.outcome``)."""

Seq = Annotated[int, Field(ge=1)]
"""Per-turn event sequence number, starting at 1."""

_AUDIO_ERROR_KINDS: Final[frozenset[str]] = frozenset(
    {"tts_failed", "tts_timeout", "aws_credentials"}
)


class _Frozen(BaseModel):
    """Immutable, strict, closed models: one definition for schema and validation."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True, allow_inf_nan=False)


class Audio(_Frozen):
    """One base64-encoded MP3 clip."""

    mime: Literal["audio/mpeg"] = AUDIO_MIME
    b64: str

    @classmethod
    def from_mp3(cls, data: bytes) -> "Audio":
        """Wrap raw MP3 bytes from the TTS client."""
        return cls(b64=base64.b64encode(data).decode("ascii"))


def _check_audio(audio: Audio | None, audio_error: AudioErrorKind | None) -> None:
    """An event has audio or an audio error, never both (Req 4.8)."""
    if audio is not None and audio_error is not None:
        raise ValueError("an event cannot carry both audio and audio_error")


class StatusEvent(_Frozen):
    """A spoken status line for a Tool call (Req 4.4, 4.5, 4.11)."""

    type: Literal["status"] = "status"
    seq: Seq
    phase: Phase
    tool: str = Field(min_length=1)
    text: str
    audio: Audio | None = None
    audio_error: AudioErrorKind | None = None

    @model_validator(mode="after")
    def _audio_xor_error(self) -> "StatusEvent":
        _check_audio(self.audio, self.audio_error)
        return self


class DatasetPreviewEvent(_Frozen):
    """A Preview_Table with its CSV_Export link (Req 7.1-7.4, 7.7, 7.9)."""

    type: Literal["dataset_preview"] = "dataset_preview"
    seq: Seq
    dataset_id: str
    indicator: str
    series_id: str
    row_count: int = Field(ge=0)
    first_date: str | None
    last_date: str | None
    columns: tuple[str, str]
    rows: tuple[PreviewRow, ...]
    csv_url: str
    derived_from: str | None
    empty: bool


class StatsCell(_Frozen):
    """One stats-table row: label and display value."""

    label: str
    value: str


class StatsTableEvent(_Frozen):
    """The 11 Descriptive_Statistics rows of a Dataset (Req 9.4)."""

    type: Literal["stats_table"] = "stats_table"
    seq: Seq
    dataset_id: str
    indicator: str
    rows: tuple[StatsCell, ...]


class ChartEvent(_Frozen):
    """An inline Chart_Image with alt text (Req 11.6)."""

    type: Literal["chart"] = "chart"
    seq: Seq
    chart_id: str
    image: str
    alt: str


class FinalEvent(_Frozen):
    """The Friday response; always the last event of a turn."""

    type: Literal["final"] = "final"
    seq: Seq
    outcome: Outcome
    text: str
    spoken_text: str
    audio: Audio | None = None
    audio_error: AudioErrorKind | None = None

    @model_validator(mode="after")
    def _audio_xor_error(self) -> "FinalEvent":
        _check_audio(self.audio, self.audio_error)
        return self


Event = StatusEvent | DatasetPreviewEvent | StatsTableEvent | ChartEvent | FinalEvent
"""Any NDJSON event."""


def audio_error_kind(error: TTSError | AwsCredentialError) -> AudioErrorKind:
    """The ``audio_error`` code for a failed synthesis (the error's ``kind``)."""
    if error.kind not in _AUDIO_ERROR_KINDS:
        raise ValueError(f"not an audio error kind: {error.kind!r}")
    return cast(AudioErrorKind, error.kind)


def csv_url(dataset_id: str, session_id: str) -> str:
    """The CSV_Export download URL for a Dataset in a Session (Req 7.4)."""
    path = DATASET_CSV_PATH.format(dataset_id=quote(dataset_id, safe=""))
    return f"{path}?{urlencode({'session': session_id})}"


class TurnEvents:
    """Builds one turn's events with consecutive ``seq`` numbers starting at 1.

    Create one per turn. Build each event right before queueing it, so ``seq`` follows
    emission order.
    """

    def __init__(self, session_id: str) -> None:
        """Start a turn of ``session_id`` (used for CSV_Export links)."""
        self._session_id = session_id
        self._seq = 0

    @property
    def last_seq(self) -> int:
        """The ``seq`` of the most recently built event (0 before the first)."""
        return self._seq

    def _next(self) -> int:
        self._seq += 1
        return self._seq

    def status(
        self,
        phase: Phase,
        tool: str,
        text: str,
        *,
        audio: bytes | None = None,
        audio_error: AudioErrorKind | None = None,
    ) -> StatusEvent:
        """A status line, with its MP3 audio or the reason it has none."""
        return StatusEvent(
            seq=self._next(),
            phase=phase,
            tool=tool,
            text=text,
            audio=_audio(audio),
            audio_error=audio_error,
        )

    def dataset_preview(self, ds: Dataset) -> DatasetPreviewEvent:
        """The Preview_Table of ``ds``; ``empty`` when it has no rows (Req 7.9)."""
        empty = ds.row_count == 0
        return DatasetPreviewEvent(
            seq=self._next(),
            dataset_id=ds.dataset_id,
            indicator=ds.indicator,
            series_id=ds.series_id,
            row_count=ds.row_count,
            first_date=None if empty else ds.dates[0].isoformat(),
            last_date=None if empty else ds.dates[-1].isoformat(),
            columns=(DATE_COLUMN, ds.value_column),
            rows=preview_rows(ds),
            csv_url=csv_url(ds.dataset_id, self._session_id),
            derived_from=ds.derived_from,
            empty=empty,
        )

    def stats_table(self, ds: Dataset, stats: Stats) -> StatsTableEvent:
        """The stats table of ``ds`` in the Descriptive_Statistics order (Req 9.4)."""
        rows = tuple(StatsCell(label=r.label, value=r.value) for r in format_stats_rows(stats))
        return StatsTableEvent(
            seq=self._next(), dataset_id=ds.dataset_id, indicator=ds.indicator, rows=rows
        )

    def chart(self, chart_id: str, png: bytes, alt: str) -> ChartEvent:
        """An inline PNG chart as a data URL (Req 11.6)."""
        image = PNG_DATA_URL_PREFIX + base64.b64encode(png).decode("ascii")
        return ChartEvent(seq=self._next(), chart_id=chart_id, image=image, alt=alt)

    def final(
        self,
        outcome: Outcome,
        text: str,
        spoken_text: str,
        *,
        audio: bytes | None = None,
        audio_error: AudioErrorKind | None = None,
    ) -> FinalEvent:
        """The Friday response that ends the turn."""
        return FinalEvent(
            seq=self._next(),
            outcome=outcome,
            text=text,
            spoken_text=spoken_text,
            audio=_audio(audio),
            audio_error=audio_error,
        )


def _audio(data: bytes | None) -> Audio | None:
    return None if data is None else Audio.from_mp3(data)


class NdjsonEncoder:
    """Encodes events as redacted, strict-JSON NDJSON lines (Req 12.4)."""

    def __init__(self, redactor: Redactor) -> None:
        """Use ``redactor`` (the Container's single instance) for every line."""
        self._redactor = redactor

    def encode(self, event: Event) -> bytes:
        """One ASCII-only JSON line ending in ``\\n``; every text value is redacted.

        ``ensure_ascii`` escapes non-ASCII (including lone surrogates) as ``\\uXXXX``, so
        the line always encodes and a raw U+2028 never reaches the browser.
        """
        tree = self._redact(event.model_dump(mode="json"))
        line = json.dumps(tree, allow_nan=False, ensure_ascii=True, separators=(",", ":"))
        return (line + "\n").encode("ascii")

    def _redact(self, value: object) -> object:
        """Redact every string in a JSON tree, except the base64 payload fields."""
        if isinstance(value, str):
            return self._redactor.redact(value)
        if isinstance(value, dict):
            items = cast(dict[str, object], value).items()
            return {k: v if k in BINARY_FIELDS else self._redact(v) for k, v in items}
        if isinstance(value, list):
            return [self._redact(item) for item in cast(list[object], value)]
        return value
