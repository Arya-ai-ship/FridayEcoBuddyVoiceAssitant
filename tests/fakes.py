"""Test doubles for every external service Friday talks to (no network, no credentials).

- ``ScriptedChatClient`` is a real MAF chat client (same layer order as ``BedrockChatClient``,
  minus telemetry) whose model responses are replayed from a script. It records the
  messages, instructions, and tools of every model call, and can raise or hang on any call.
- ``FakeTTS``, ``FakeSTT``, and ``FakeFred`` satisfy the ``friday.ports`` Protocols and raise
  only ``FridayError`` subclasses, like the real adapters.

Model failures are raised raw, the way ``BedrockChatClient`` re-raises botocore errors, so
the chat middleware's mapping through ``friday.aws_errors`` is exercised for real.
"""

import asyncio
from collections.abc import AsyncIterable, Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Literal

from agent_framework import (
    BaseChatClient,
    ChatAndFunctionMiddlewareTypes,
    ChatMiddlewareLayer,
    ChatResponse,
    ChatResponseUpdate,
    Content,
    FinishReasonLiteral,
    FunctionInvocationConfiguration,
    FunctionInvocationLayer,
    Message,
    ResponseStream,
)
from agent_framework.exceptions import ChatClientException
from botocore.exceptions import ClientError

from friday.constants import FRED_TIMEOUT_S, STT_SAMPLE_RATE_HZ, STT_TIMEOUT_S, TTS_TIMEOUT_S
from friday.data.fetch import Observation
from friday.errors import AwsCredentialError, FredError, FredKind, STTError, TTSError

# --- Error builders -----------------------------------------------------------


def aws_client_error(
    code: str = "ExpiredTokenException",
    status: int = 403,
    message: str = "The security token included in the request is expired",
    operation: str = "Converse",
) -> ClientError:
    """A realistic botocore ``ClientError`` (defaults: expired session token, HTTP 403)."""
    return ClientError(
        {
            "Error": {"Code": code, "Message": message},
            "ResponseMetadata": {
                "RequestId": "fake-request",
                "HostId": "",
                "HTTPStatusCode": status,
                "HTTPHeaders": {},
                "RetryAttempts": 0,
            },
        },
        operation,
    )


def wrapped_in_maf(inner: BaseException, message: str = "Bedrock call failed") -> BaseException:
    """``inner`` as the ``__cause__`` of a MAF ``ChatClientException`` (``raise ... from``)."""
    wrapper = ChatClientException(message, log_level=None)
    wrapper.__cause__ = inner
    return wrapper


def credential_error(*, wrapped: bool = False) -> BaseException:
    """An ``ExpiredTokenException`` ``ClientError``, bare or wrapped by a MAF exception."""
    inner = aws_client_error()
    return wrapped_in_maf(inner) if wrapped else inner


# --- Scripted chat client -------------------------------------------------------


@dataclass(frozen=True)
class Call:
    """One function call in a scripted model response.

    ``arguments`` may be a mapping or a raw string (for example invalid JSON).
    ``call_id`` defaults to a unique ``call_<n>`` assigned by the client.
    """

    name: str
    arguments: Mapping[str, Any] | str | None = None
    call_id: str | None = None


@dataclass(frozen=True)
class Reply:
    """A scripted model response: zero or more function calls and/or final text."""

    calls: tuple[Call, ...] = ()
    text: str | None = None


@dataclass(frozen=True)
class Raise:
    """Raise ``error`` from this model call."""

    error: BaseException


@dataclass(frozen=True)
class Hang:
    """Block this model call until it is cancelled (e.g. by a timeout) or released.

    When the client's ``release`` event is set, the call continues with the next step.
    """


Step = Reply | Raise | Hang


def text(value: str) -> Reply:
    """A final-text model response."""
    return Reply(text=value)


def calls(*items: Call, text: str | None = None) -> Reply:
    """A model response with one or several function calls."""
    return Reply(calls=tuple(items), text=text)


@dataclass(frozen=True)
class ModelCall:
    """What the model received on one call."""

    messages: tuple[Message, ...]
    instructions: str | None
    tools: tuple[Any, ...]
    options: Mapping[str, Any]

    @property
    def tool_names(self) -> tuple[str, ...]:
        """Names of the tools offered on this call, in order."""
        return tuple(str(getattr(tool, "name", tool)) for tool in self.tools)


class ScriptExhaustedError(AssertionError):
    """The client was called more times than the script has steps."""


class ScriptedChatClient(
    FunctionInvocationLayer[Any], ChatMiddlewareLayer[Any], BaseChatClient[Any]
):
    """MAF chat client that replays ``script`` one step per model call."""

    OTEL_PROVIDER_NAME = "friday.scripted"

    def __init__(
        self,
        script: Iterable[Step] = (),
        *,
        function_invocation_configuration: FunctionInvocationConfiguration | None = None,
        middleware: Sequence[ChatAndFunctionMiddlewareTypes] | None = None,
    ) -> None:
        super().__init__(
            function_invocation_configuration=function_invocation_configuration,
            middleware=middleware,
        )
        self.script: list[Step] = list(script)
        self.calls: list[ModelCall] = []
        self.hang_started = asyncio.Event()
        self.release = asyncio.Event()
        self._next_call_id = 0

    def add(self, *steps: Step) -> None:
        """Append steps to the script (e.g. before the next turn)."""
        self.script.extend(steps)

    @property
    def remaining(self) -> int:
        """Script steps not yet consumed."""
        return len(self.script)

    def _inner_get_response(
        self,
        *,
        messages: Sequence[Message],
        stream: bool,
        options: Mapping[str, Any],
        **kwargs: Any,
    ) -> Awaitable[ChatResponse] | ResponseStream[ChatResponseUpdate, ChatResponse]:
        self.calls.append(
            ModelCall(
                messages=tuple(messages),
                instructions=options.get("instructions"),
                tools=tuple(options.get("tools") or ()),
                options=dict(options),
            )
        )
        if not stream:
            return self._respond()

        async def _stream() -> AsyncIterable[ChatResponseUpdate]:
            reply = await self._next_reply()
            yield ChatResponseUpdate(
                role="assistant",
                contents=self._contents(reply),
                finish_reason=_finish_reason(reply),
            )

        return self._build_response_stream(_stream())

    async def _respond(self) -> ChatResponse:
        reply = await self._next_reply()
        return ChatResponse(
            messages=[Message(role="assistant", contents=self._contents(reply))],
            response_id=f"resp_{len(self.calls)}",
            finish_reason=_finish_reason(reply),
        )

    async def _next_reply(self) -> Reply:
        """Consume steps until a ``Reply``: raise for ``Raise``, wait for ``Hang``."""
        while True:
            if not self.script:
                raise ScriptExhaustedError(f"no scripted response for model call {len(self.calls)}")
            step = self.script.pop(0)
            match step:
                case Raise(error=error):
                    raise error
                case Hang():
                    self.hang_started.set()
                    await self.release.wait()
                case Reply():
                    return step

    def _contents(self, reply: Reply) -> list[Content]:
        contents: list[Content] = [
            Content.from_function_call(
                call.call_id or self._new_call_id(), call.name, arguments=call.arguments
            )
            for call in reply.calls
        ]
        if reply.text is not None:
            contents.append(Content.from_text(reply.text))
        return contents

    def _new_call_id(self) -> str:
        self._next_call_id += 1
        return f"call_{self._next_call_id}"


def _finish_reason(reply: Reply) -> FinishReasonLiteral:
    return "tool_calls" if reply.calls else "stop"


# --- Speech fakes -------------------------------------------------------------------

TTSFailure = Literal["failed", "timeout", "credentials"]
STTFailure = Literal["failed", "timeout", "credentials", "empty"]


def _tts_error(failure: TTSFailure) -> AwsCredentialError | TTSError:
    if failure == "credentials":
        return AwsCredentialError()
    if failure == "timeout":
        return TTSError("tts_timeout", f"speech synthesis timed out after {TTS_TIMEOUT_S:g} s")
    return TTSError("tts_failed", "speech synthesis failed")


def _stt_error(failure: STTFailure) -> AwsCredentialError | STTError:
    if failure == "credentials":
        return AwsCredentialError()
    if failure == "timeout":
        return STTError("stt_timeout", f"transcription timed out after {STT_TIMEOUT_S:g} s")
    if failure == "empty":
        return STTError("stt_empty", "the transcript was empty")
    return STTError("stt_failed", "transcription failed")


@dataclass
class FakeTTS:
    """``TTSClient`` fake. Records texts; fails every call, or those ``fail_when`` selects."""

    failure: TTSFailure | None = None
    fail_when: Callable[[str], bool] | None = None
    calls: list[str] = field(default_factory=list[str])

    async def synthesize(self, text: str, timeout_s: float = TTS_TIMEOUT_S) -> bytes:
        """Return fake MP3 bytes (``ID3`` + the UTF-8 text) or raise the configured error."""
        self.calls.append(text)
        if self.failure is not None and (self.fail_when is None or self.fail_when(text)):
            raise _tts_error(self.failure)
        return b"ID3" + text.encode()


@dataclass(frozen=True)
class STTCall:
    """One recorded ``transcribe`` call."""

    pcm16: bytes
    sample_rate: int
    timeout_s: float


@dataclass
class FakeSTT:
    """``STTClient`` fake returning ``transcript`` or raising the configured error."""

    transcript: str = "show me inflation"
    failure: STTFailure | None = None
    calls: list[STTCall] = field(default_factory=list[STTCall])

    async def transcribe(
        self,
        pcm16: bytes,
        sample_rate: int = STT_SAMPLE_RATE_HZ,
        timeout_s: float = STT_TIMEOUT_S,
    ) -> str:
        """Record the call and return the transcript (or raise)."""
        self.calls.append(STTCall(pcm16, sample_rate, timeout_s))
        if self.failure is not None:
            raise _stt_error(self.failure)
        return self.transcript


# --- FRED fake --------------------------------------------------------------------------


def monthly_observations(
    start_year: int = 2015, end_year: int = 2024, base: float = 100.0, step: float = 0.5
) -> tuple[Observation, ...]:
    """Deterministic monthly observations on the first of each month."""
    rows: list[Observation] = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            value = base + step * len(rows)
            rows.append({"date": date(year, month, 1).isoformat(), "value": f"{value:g}"})
    return tuple(rows)


@dataclass(frozen=True)
class FredCall:
    """One recorded ``observations`` call."""

    series_id: str
    start: date | None
    end: date | None


@dataclass
class FakeFred:
    """``FredSource`` fake serving ``series`` (or ``default``) filtered to the date range.

    ``failure`` raises that ``FredError`` kind for every series, or only for the IDs in
    ``fail_series`` when given. An empty filtered range raises ``no_observations``.
    """

    series: Mapping[str, Sequence[Observation]] = field(
        default_factory=dict[str, Sequence[Observation]]
    )
    default: Sequence[Observation] = field(default_factory=monthly_observations)
    failure: FredKind | None = None
    fail_series: frozenset[str] | None = None
    calls: list[FredCall] = field(default_factory=list[FredCall])

    async def observations(
        self, series_id: str, start: date | None, end: date | None
    ) -> Sequence[Observation]:
        """Record the call, then return the in-range observations or raise."""
        self.calls.append(FredCall(series_id, start, end))
        if self.failure is not None and (self.fail_series is None or series_id in self.fail_series):
            raise _fred_error(self.failure, series_id)
        rows = tuple(
            row
            for row in self.series.get(series_id, self.default)
            if (start is None or row["date"] >= start.isoformat())
            and (end is None or row["date"] <= end.isoformat())
        )
        if not rows:
            raise _fred_error("no_observations", series_id)
        return rows


def _fred_error(kind: FredKind, series_id: str) -> FredError:
    messages: dict[FredKind, str] = {
        "http_error": f"FRED returned HTTP 500 for series {series_id}",
        "timeout": f"FRED request for series {series_id} timed out after {FRED_TIMEOUT_S:g} s",
        "no_observations": f"FRED returned no observations for series {series_id}",
    }
    return FredError(kind, messages[kind])
