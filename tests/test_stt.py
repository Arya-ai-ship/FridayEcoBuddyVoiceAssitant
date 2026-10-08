"""Tests for ``speech/stt.py`` ``TranscribeSTT`` (no network, placeholder credentials).

The streaming client is replaced through the ``client_factory`` parameter by a fake that
records the request, the chunks, and the close, and replays scripted SDK result events.
Pacing uses a recording ``sleep`` so the tests run instantly.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass, field, replace

import pytest
from aws_sdk_transcribe_streaming.models import (
    Alternative,
    AudioStream,
    AudioStreamAudioEvent,
    BadRequestException,
    LimitExceededException,
    MediaEncoding,
    Result,
    StartStreamTranscriptionInput,
    Transcript,
    TranscriptEvent,
    TranscriptResultStream,
    TranscriptResultStreamLimitExceededException,
    TranscriptResultStreamTranscriptEvent,
    TranscriptResultStreamUnknown,
)
from smithy_aws_core.identity import AWSIdentityProperties
from smithy_core.exceptions import CallError, SmithyIdentityError
from smithy_http.aio.crt import AWSCRTHTTPClient

from friday.config import Settings, secret_values
from friday.constants import AWS_MAX_ATTEMPTS, STT_CHUNK_MS, STT_PACE_FACTOR, STT_SAMPLE_RATE_HZ
from friday.errors import AwsCredentialError, STTError
from friday.ports import STTClient
from friday.redact import Redactor
from friday.speech.stt import (
    StreamingClient,
    TranscribeSTT,
    audio_chunks,
    chunk_size,
    transcribe_config,
)

FAKE_KEY_ID = "AKIAFAKEFAKEFAKEFAKE"
FAKE_SECRET = "FAKEsecretVALUEforTESTSonly0123456789xyz"  # noqa: S105 - fake placeholder
FAKE_TOKEN = "FAKEsessionTOKENforTESTSonly0123456789"  # noqa: S105 - fake placeholder
REGION = "us-east-2"
LANGUAGE = "en-GB"
CHUNK_BYTES = STT_SAMPLE_RATE_HZ * STT_CHUNK_MS // 1000 * 2  # 3200 bytes per 100 ms


@pytest.fixture
def settings() -> Settings:
    return Settings(
        aws_access_key_id=FAKE_KEY_ID,
        aws_secret_access_key=FAKE_SECRET,
        aws_session_token=FAKE_TOKEN,
        aws_region=REGION,
        fred_api_key="fakefredkey000000000000000000000",
        bedrock_model_id="us.openai.gpt-5.6-terra",
        transcribe_language_code=LANGUAGE,
    )


@pytest.fixture
def redactor(settings: Settings) -> Redactor:
    return Redactor(secret_values(settings))


# --- Fake stream -----------------------------------------------------------------------


def transcript_event(*results: tuple[str | None, bool]) -> TranscriptResultStreamTranscriptEvent:
    """A result event; each ``(text, is_partial)`` is one result (``None``: no alternatives)."""
    return TranscriptResultStreamTranscriptEvent(
        value=TranscriptEvent(
            transcript=Transcript(
                results=[
                    Result(
                        is_partial=partial,
                        alternatives=None if text is None else [Alternative(transcript=text)],
                    )
                    for text, partial in results
                ]
            )
        )
    )


@dataclass
class FakeStream:
    """Duplex stream fake: publisher and receiver in one object."""

    events: Sequence[TranscriptResultStream] = ()
    error: BaseException | None = None  # raised by receive() after the events
    wait_for_input_close: bool = True
    hang: bool = False
    log: list[str] = field(default_factory=list[str])
    chunks: list[bytes] = field(default_factory=list[bytes])
    _input_closed: asyncio.Event = field(default_factory=asyncio.Event)
    _pending: list[TranscriptResultStream] = field(default_factory=list[TranscriptResultStream])

    def __post_init__(self) -> None:
        self._pending = list(self.events)

    @property
    def input_stream(self) -> FakeStream:
        return self

    async def await_output(self) -> tuple[object, FakeStream]:
        return None, self

    async def send(self, event: AudioStream) -> None:
        assert not self._input_closed.is_set(), "send after close"
        assert isinstance(event, AudioStreamAudioEvent)
        chunk = event.value.audio_chunk
        assert chunk is not None
        self.chunks.append(chunk)
        self.log.append(f"send:{len(chunk)}")

    async def receive(self) -> TranscriptResultStream | None:
        if self.hang:
            await asyncio.Event().wait()
        if self.wait_for_input_close:
            await self._input_closed.wait()  # Transcribe finalizes after the end frame.
        if self._pending:
            return self._pending.pop(0)
        if self.error is not None:
            raise self.error
        return None

    async def close(self) -> None:
        if self._input_closed.is_set():
            self.log.append("close:again")
        else:
            self.log.append("close:input")
        self._input_closed.set()


@dataclass
class FakeClient:
    """``StreamingClient`` fake handing out ``stream`` (or raising ``start_error``)."""

    stream: FakeStream = field(default_factory=FakeStream)
    start_error: BaseException | None = None
    requests: list[StartStreamTranscriptionInput] = field(
        default_factory=list[StartStreamTranscriptionInput]
    )
    closed: bool = False

    async def start_stream_transcription(self, input: StartStreamTranscriptionInput) -> FakeStream:
        self.requests.append(input)
        if self.start_error is not None:
            raise self.start_error
        return self.stream

    async def close(self) -> None:
        self.closed = True


@dataclass
class Harness:
    stt: TranscribeSTT
    client: FakeClient
    sleeps: list[float]
    factory_calls: list[int]


def make_stt(client: FakeClient, redactor: Redactor) -> Harness:
    sleeps: list[float] = []
    factory_calls: list[int] = []

    async def factory() -> StreamingClient:
        factory_calls.append(1)
        return client

    async def sleep(delay: float) -> None:
        sleeps.append(delay)

    stt = TranscribeSTT(factory, language_code=LANGUAGE, redactor=redactor, sleep=sleep)
    return Harness(stt, client, sleeps, factory_calls)


def audio(n_bytes: int) -> bytes:
    return bytes(i % 251 for i in range(n_bytes))


# --- Streaming and results -------------------------------------------------------------


async def test_streams_paced_chunks_then_joins_final_results(redactor: Redactor) -> None:
    events = [
        transcript_event(("show", True)),
        transcript_event(("Show me", False)),
        TranscriptResultStreamUnknown(tag="SomethingNew"),
        transcript_event(("inflat", True), (None, False), ("  inflation data. ", False)),
    ]
    h = make_stt(FakeClient(FakeStream(events=events)), redactor)
    pcm = audio(3 * CHUNK_BYTES + CHUNK_BYTES // 2)

    text = await h.stt.transcribe(pcm)

    assert text == "Show me inflation data."
    stream = h.client.stream
    assert [len(c) for c in stream.chunks] == [CHUNK_BYTES] * 3 + [CHUNK_BYTES // 2]
    assert b"".join(stream.chunks) == pcm  # in order, nothing dropped or duplicated
    assert stream.log[:5] == [*(f"send:{len(c)}" for c in stream.chunks), "close:input"]
    assert h.sleeps == [STT_CHUNK_MS / 1000 / STT_PACE_FACTOR] * 3  # 25 ms between chunks
    (request,) = h.client.requests
    assert request.language_code == LANGUAGE
    assert request.media_encoding is MediaEncoding.PCM
    assert request.media_sample_rate_hertz == STT_SAMPLE_RATE_HZ


async def test_sample_rate_sets_request_and_chunk_size(redactor: Redactor) -> None:
    h = make_stt(FakeClient(FakeStream(events=[transcript_event(("hi", False))])), redactor)
    await h.stt.transcribe(audio(chunk_size(8000) * 2), sample_rate=8000)
    assert h.client.requests[0].media_sample_rate_hertz == 8000
    assert [len(c) for c in h.client.stream.chunks] == [1600, 1600]


def test_audio_chunks_cover_the_audio_in_whole_samples() -> None:
    assert chunk_size(STT_SAMPLE_RATE_HZ) == CHUNK_BYTES
    assert audio_chunks(b"", STT_SAMPLE_RATE_HZ) == []
    pcm = audio(CHUNK_BYTES + 2)
    assert audio_chunks(pcm, STT_SAMPLE_RATE_HZ) == [pcm[:CHUNK_BYTES], pcm[CHUNK_BYTES:]]


async def test_client_is_created_once_and_closed(redactor: Redactor) -> None:
    h = make_stt(FakeClient(), redactor)
    for _ in range(2):
        h.client.stream = FakeStream(events=[transcript_event(("again", False))])
        assert await h.stt.transcribe(audio(CHUNK_BYTES)) == "again"
    assert h.factory_calls == [1]
    await h.stt.close()
    assert h.client.closed


@pytest.mark.parametrize(
    "events",
    [
        [],
        [transcript_event(("partial only", True))],
        [transcript_event(("   ", False), (None, False), ("", False))],
    ],
    ids=["no-results", "partials-only", "whitespace"],
)
async def test_empty_transcript_is_stt_empty(
    redactor: Redactor, events: list[TranscriptResultStream]
) -> None:
    h = make_stt(FakeClient(FakeStream(events=events)), redactor)
    with pytest.raises(STTError) as info:
        await h.stt.transcribe(audio(CHUNK_BYTES))
    assert info.value.kind == "stt_empty"


async def test_empty_audio_is_stt_empty_without_a_call(redactor: Redactor) -> None:
    h = make_stt(FakeClient(), redactor)
    with pytest.raises(STTError) as info:
        await h.stt.transcribe(b"")
    assert info.value.kind == "stt_empty"
    assert h.factory_calls == []


# --- Failures --------------------------------------------------------------------------


async def test_timeout_is_stt_timeout_and_closes_the_stream(redactor: Redactor) -> None:
    h = make_stt(FakeClient(FakeStream(hang=True)), redactor)
    with pytest.raises(STTError) as info:
        await h.stt.transcribe(audio(CHUNK_BYTES), timeout_s=0.05)
    assert info.value.kind == "stt_timeout"
    assert "close:input" in h.client.stream.log


@pytest.mark.parametrize(
    "stream",
    [
        FakeStream(error=BadRequestException(message="bad sample rate")),
        FakeStream(
            events=[
                TranscriptResultStreamLimitExceededException(
                    value=LimitExceededException(message="too many streams")
                )
            ]
        ),
        FakeStream(error=OSError("Failed to read from stream.")),
    ],
    ids=["raised-modeled-error", "error-event", "transport"],
)
async def test_service_failures_are_stt_failed(redactor: Redactor, stream: FakeStream) -> None:
    h = make_stt(FakeClient(stream), redactor)
    with pytest.raises(STTError) as info:
        await h.stt.transcribe(audio(CHUNK_BYTES))
    assert info.value.kind == "stt_failed"


@pytest.mark.parametrize(
    "client",
    [
        # Unmodeled 403 as smithy reports it, echoing the key ID and secret.
        FakeClient(
            FakeStream(
                error=CallError(
                    "Unknown error for operation"
                    " com.amazonaws.transcribestreaming#StartStreamTranscription"
                    " - status: 403 - id: com.amazonaws.transcribestreaming#"
                    f"UnrecognizedClientException ({FAKE_KEY_ID} {FAKE_SECRET})"
                )
            )
        ),
        FakeClient(start_error=SmithyIdentityError(f"no credentials for {FAKE_SECRET}")),
    ],
    ids=["unrecognized-client-403", "identity-error"],
)
async def test_credential_failures_are_redacted_aws_errors(
    redactor: Redactor, client: FakeClient
) -> None:
    h = make_stt(client, redactor)
    with pytest.raises(AwsCredentialError) as info:
        await h.stt.transcribe(audio(CHUNK_BYTES))
    assert info.value.kind == "aws_credentials"
    for secret in (FAKE_KEY_ID, FAKE_SECRET, FAKE_SECRET[:8]):
        assert secret not in info.value.message


def test_rejects_non_positive_pacing(redactor: Redactor) -> None:
    async def factory() -> StreamingClient:
        raise AssertionError("not called")

    with pytest.raises(ValueError, match="pace_factor"):
        TranscribeSTT(factory, language_code=LANGUAGE, redactor=redactor, pace_factor=0)
    with pytest.raises(ValueError, match="chunk_ms"):
        TranscribeSTT(factory, language_code=LANGUAGE, redactor=redactor, chunk_ms=0)


# --- Real client config: Settings only -------------------------------------------------


@pytest.fixture
def conflicting_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """AWS_* values that must not influence the Transcribe config (placeholders only)."""
    values = {
        "AWS_ACCESS_KEY_ID": "AKIAENVENVENVENVENV1",
        "AWS_SECRET_ACCESS_KEY": "envSecretFromAwsVariable0000000000000000",
        "AWS_SESSION_TOKEN": "envTokenFromAwsVariable00000000000000",
        "AWS_REGION": "ap-south-1",
        "AWS_DEFAULT_REGION": "ap-south-1",
        "AWS_PROFILE": "friday-test-profile-that-does-not-exist",
        "AWS_CONFIG_FILE": "/nonexistent/friday/config",
        "AWS_SHARED_CREDENTIALS_FILE": "/nonexistent/friday/credentials",
        "AWS_MAX_ATTEMPTS": "9",
        "AWS_ENDPOINT_URL": "https://endpoint-from-env.invalid",
        "AWS_ENDPOINT_URL_TRANSCRIBE_STREAMING": "https://endpoint-from-env.invalid",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)


@pytest.mark.usefixtures("conflicting_env")
@pytest.mark.parametrize("token", [FAKE_TOKEN, None])
async def test_config_uses_only_settings(settings: Settings, token: str | None) -> None:
    config = await transcribe_config(replace(settings, aws_session_token=token))
    try:
        assert config.region == REGION
        assert config.max_attempts == AWS_MAX_ATTEMPTS
        assert config.endpoint_uri is None
        assert isinstance(config.transport, AWSCRTHTTPClient)
        resolver = config.aws_credentials_identity_resolver
        assert resolver is not None
        identity = await resolver.get_identity(properties=AWSIdentityProperties())
        assert (identity.access_key_id, identity.secret_access_key, identity.session_token) == (
            FAKE_KEY_ID,
            FAKE_SECRET,
            token,
        )
        assert FAKE_SECRET not in repr(config)
    finally:
        await config.transport.close()  # type: ignore[union-attr]


def test_from_settings_satisfies_the_port(settings: Settings, redactor: Redactor) -> None:
    assert isinstance(TranscribeSTT.from_settings(settings, redactor), STTClient)
