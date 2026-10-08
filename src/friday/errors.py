"""The ``FridayError`` hierarchy (single source of error kinds).

Every error carries a ``kind`` string that is reused verbatim as the wire code in
tool results (``error.kind``), NDJSON events (``outcome`` / ``audio_error``), and
HTTP error bodies (``error.code``). This module is pure: no SDK imports.

``tool_limit`` and ``no_data`` are normal turn outcomes, not exceptions.
"""

from collections.abc import Sequence
from typing import ClassVar, Literal

ConfigKind = Literal["config_missing", "indicator_map_invalid", "port_invalid"]
TTSKind = Literal["tts_failed", "tts_timeout"]
STTKind = Literal["stt_failed", "stt_timeout", "stt_empty"]
RequestKind = Literal[
    "invalid_text", "request_too_large", "audio_too_large", "dataset_not_found", "invalid_session"
]
ToolValidationKind = Literal["unknown_tool", "invalid_args"]
FredKind = Literal["http_error", "timeout", "no_observations"]
PortCause = Literal["invalid value", "port in use", "bind not permitted"]

JsonDict = dict[str, object]


class FridayError(Exception):
    """Base class for every Friday error. ``kind`` is the wire code."""

    KINDS: ClassVar[frozenset[str]] = frozenset()

    def __init__(self, kind: str, message: str) -> None:
        if kind not in self.KINDS:
            raise ValueError(f"{type(self).__name__} does not accept kind {kind!r}")
        super().__init__(message)
        self.kind: str = kind
        self.message: str = message


# --- Startup / configuration -----------------------------------------------


class ConfigError(FridayError):
    """Startup configuration failure; the CLI exits non-zero before serving."""

    KINDS = frozenset({"config_missing", "indicator_map_invalid", "port_invalid"})

    def __init__(self, kind: ConfigKind, message: str, *, missing: Sequence[str] = ()) -> None:
        super().__init__(kind, message)
        self.missing: tuple[str, ...] = tuple(missing)

    @classmethod
    def missing_vars(cls, names: Sequence[str]) -> "ConfigError":
        """Build the single error naming every missing required variable (Req 12.6)."""
        listed = ", ".join(names)
        return cls(
            "config_missing",
            f"Missing required environment variables: {listed}",
            missing=names,
        )

    @classmethod
    def indicator_map(cls, path: str, cause: str) -> "ConfigError":
        """Build an invalid Indicator_Map error naming the path and cause (Req 12.7)."""
        return cls("indicator_map_invalid", f"Invalid Indicator_Map file {path}: {cause}")


class PortError(ConfigError):
    """Invalid, busy, or forbidden Backend port (Req 13.5)."""

    def __init__(self, value: str, cause: PortCause) -> None:
        super().__init__("port_invalid", f"FRIDAY_PORT {value!r}: {cause}")
        self.value: str = value
        self.cause: PortCause = cause


# --- AWS / speech / LLM ----------------------------------------------------


class AwsCredentialError(FridayError):
    """Expired, invalid, or insufficiently permitted AWS credentials (Req 5.12)."""

    KINDS = frozenset({"aws_credentials"})

    def __init__(self, message: str = "AWS credentials have expired or lack access.") -> None:
        super().__init__("aws_credentials", message)


class LLMUnavailableError(FridayError):
    """Any Bedrock failure other than a credential error, including timeouts (Req 5.7)."""

    KINDS = frozenset({"llm_unavailable"})

    def __init__(self, message: str = "The language model is unavailable.") -> None:
        super().__init__("llm_unavailable", message)


class TTSError(FridayError):
    """Polly failure or timeout (Req 4.8)."""

    KINDS = frozenset({"tts_failed", "tts_timeout"})

    def __init__(self, kind: TTSKind, message: str) -> None:
        super().__init__(kind, message)


class STTError(FridayError):
    """Transcribe failure, timeout, or empty transcript (Req 3.6)."""

    KINDS = frozenset({"stt_failed", "stt_timeout", "stt_empty"})

    def __init__(self, kind: STTKind, message: str) -> None:
        super().__init__(kind, message)


class RequestError(FridayError):
    """Invalid HTTP request from the Browser_UI.

    ``invalid_session`` means the session ID (``X-Friday-Session`` header or ``session``
    query parameter) is missing or is not a UUID.
    """

    KINDS = frozenset(
        {
            "invalid_text",
            "request_too_large",
            "audio_too_large",
            "dataset_not_found",
            "invalid_session",
        }
    )

    def __init__(self, kind: RequestKind, message: str) -> None:
        super().__init__(kind, message)

    @classmethod
    def invalid_session(cls) -> "RequestError":
        """Error for a session ID that is not a UUID string (the raw value is not echoed)."""
        return cls("invalid_session", "session ID must be a UUID")


# --- Tools -----------------------------------------------------------------


class ToolError(FridayError):
    """A Tool failure, returned to the LLM as ``{"ok": false, "tool", "error"}``."""

    def details(self) -> JsonDict:
        """Extra fields merged into the ``error`` object of the tool result."""
        return {}

    def to_result(self, tool: str) -> JsonDict:
        """Render the tool result sent back to the LLM (Req 5.8, 5.11)."""
        error: JsonDict = {"kind": self.kind, "message": self.message, **self.details()}
        return {"ok": False, "tool": tool, "error": error}


class ToolValidationError(ToolError):
    """Unknown tool name or schema-invalid arguments (Req 5.8, 10.10)."""

    KINDS = frozenset({"unknown_tool", "invalid_args"})

    def __init__(
        self, kind: ToolValidationKind, message: str, *, name: str, field: str | None = None
    ) -> None:
        super().__init__(kind, message)
        self.name: str = name
        self.field: str | None = field

    @classmethod
    def unknown_tool(cls, name: str, supported: Sequence[str]) -> "ToolValidationError":
        """Error naming the unknown tool and the supported ones."""
        listed = ", ".join(supported)
        return cls("unknown_tool", f"unknown tool '{name}'; supported: {listed}", name=name)

    @classmethod
    def invalid_args(cls, name: str, field: str, reason: str) -> "ToolValidationError":
        """Error naming the offending argument."""
        return cls("invalid_args", f"invalid argument '{field}': {reason}", name=name, field=field)

    def details(self) -> JsonDict:
        return {"field": self.field} if self.field is not None else {}


class UnknownIndicator(ToolError):
    """The requested Indicator matches no Indicator_Map entry (Req 6.9)."""

    KINDS = frozenset({"unknown_indicator"})

    def __init__(self, query: str, supported: Sequence[str]) -> None:
        listed = ", ".join(supported)
        super().__init__("unknown_indicator", f"unknown indicator '{query}'; supported: {listed}")
        self.query: str = query
        self.supported: tuple[str, ...] = tuple(supported)

    def details(self) -> JsonDict:
        return {"supported": list(self.supported)}


class DateRangeError(ToolError):
    """Malformed date, non-calendar date, or start later than end (Req 6.12, 11.9)."""

    KINDS = frozenset({"invalid_date"})

    def __init__(self, field: str, message: str) -> None:
        super().__init__("invalid_date", message)
        self.field: str = field

    def details(self) -> JsonDict:
        return {"field": self.field}


class FredError(ToolError):
    """FRED HTTP error, timeout, or zero observations (Req 6.10)."""

    KINDS = frozenset({"http_error", "timeout", "no_observations"})

    def __init__(self, kind: FredKind, message: str) -> None:
        super().__init__(kind, message)


class RangeTooShortForYoY(ToolError):
    """The YoY_Transformation left zero rows (Req 6.13)."""

    KINDS = frozenset({"range_too_short_for_yoy"})

    def __init__(
        self,
        message: str = "the date range is too short for the year-over-year transformation; "
        "it needs more than 12 monthly observations",
    ) -> None:
        super().__init__("range_too_short_for_yoy", message)


class DatasetNotFound(ToolError):
    """A Dataset_ID that is not present in the Session (Req 9.10, 10.9, 11.8)."""

    KINDS = frozenset({"dataset_not_found"})

    def __init__(self, dataset_id: str) -> None:
        super().__init__("dataset_not_found", f"dataset '{dataset_id}' was not found")
        self.dataset_id: str = dataset_id

    def details(self) -> JsonDict:
        return {"dataset_id": self.dataset_id}


class ChartRenderError(ToolError):
    """Chart rendering failed for a reason other than invalid arguments (Req 11.10)."""

    KINDS = frozenset({"chart_render_failed"})

    def __init__(self, message: str = "the chart could not be rendered") -> None:
        super().__init__("chart_render_failed", message)


class ToolFailedError(ToolError):
    """An unexpected exception inside a Tool (Req 5.11).

    The message is fixed and names only the Tool, so exception text (which may quote
    inputs or internals) never reaches the LLM or the browser.
    """

    KINDS = frozenset({"tool_failed"})

    def __init__(self, tool: str) -> None:
        super().__init__("tool_failed", f"the {tool} tool failed unexpectedly")
