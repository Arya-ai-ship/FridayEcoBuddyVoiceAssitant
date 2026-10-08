"""Amazon Transcribe streaming speech-to-text adapter (Req 3.3, 3.6, 5.12, 12.11).

:class:`TranscribeSTT` implements the :class:`~friday.ports.STTClient` Protocol. It streams
the recorded PCM16 audio, held only in memory, to Transcribe in ``STT_CHUNK_MS`` chunks
paced at ``STT_PACE_FACTOR`` times real time, closes the input stream, and joins the
non-partial results. The whole call runs under ``asyncio.wait_for(timeout_s)``. Every
failure is mapped by :func:`~friday.aws_errors.to_stt_error` to ``AwsCredentialError`` or
``STTError`` (``stt_timeout``/``stt_failed``) with a redacted message; an empty or
whitespace-only transcript raises ``STTError("stt_empty")``.

SDK API used (``aws-sdk-transcribe-streaming==0.11.0``, smithy-based, ``awscrt`` extra):

- ``AsyncTranscribeStreamingConfig.resolve(profile=, fs=, config_file_path=,
  credentials_file_path=, region=, aws_access_key_id=, aws_secret_access_key=,
  aws_session_token=, retry_mode=, max_attempts=, endpoint_uri=, sdk_ua_app_id=,
  transport=)``. Passing the key and secret makes ``resolve`` wire a
  ``StaticCredentialsResolver``, so the client never builds its default ``IdentityChain``
  (environment variables, shared files, IMDS). Every other field that has an
  environment/config-file resolver is overridden too, and the explicit ``profile`` is
  validated against an in-memory :class:`_NoSharedConfig` file system, so neither
  ``AWS_PROFILE``, ``AWS_CONFIG_FILE``/``AWS_SHARED_CREDENTIALS_FILE``, ``AWS_REGION``,
  ``AWS_ENDPOINT_URL*``, nor any file on disk is consulted.
- ``transport=AWSCRTHTTPClient()`` (``smithy_http.aio.crt``): the default aiohttp
  transport cannot do the bidirectional HTTP/2 stream.
- ``AsyncTranscribeStreamingClient(config=).start_stream_transcription(input=
  StartStreamTranscriptionInput(language_code=LanguageCode(..), media_encoding=
  MediaEncoding.PCM, media_sample_rate_hertz=..))`` returns a ``DuplexEventStream``.
- ``stream.input_stream.send(AudioStreamAudioEvent(value=AudioEvent(audio_chunk=..)))``,
  then ``stream.input_stream.close()`` (sends the signed empty end frame).
- ``stream.await_output()`` returns ``(StartStreamTranscriptionOutput, receiver)``;
  ``receiver.receive()`` yields ``TranscriptResultStreamTranscriptEvent`` values
  (``.value.transcript.results[i].is_partial`` / ``.alternatives[0].transcript``) until
  ``None``. Modeled error events (``BadRequestException`` and so on) are raised by
  ``receive()``; any that arrive as values are raised here.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Awaitable, Callable, Sequence
from contextlib import suppress
from typing import Protocol

from aws_sdk_transcribe_streaming.client import AsyncTranscribeStreamingClient
from aws_sdk_transcribe_streaming.config import AsyncTranscribeStreamingConfig
from aws_sdk_transcribe_streaming.models import (
    AudioEvent,
    AudioStream,
    AudioStreamAudioEvent,
    LanguageCode,
    MediaEncoding,
    StartStreamTranscriptionInput,
    TranscriptEvent,
    TranscriptResultStream,
    TranscriptResultStreamTranscriptEvent,
    TranscriptResultStreamUnknown,
)
from smithy_http.aio.crt import AWSCRTHTTPClient

from friday.aws_errors import to_stt_error
from friday.config import Settings
from friday.constants import (
    AWS_MAX_ATTEMPTS,
    STT_CHUNK_MS,
    STT_CLOSE_TIMEOUT_S,
    STT_EDGE_PAD_MS,
    STT_PACE_FACTOR,
    STT_SAMPLE_RATE_HZ,
    STT_SILENCE_LEVEL,
    STT_TIMEOUT_S,
)
from friday.errors import FridayError, STTError
from friday.redact import Redactor

BYTES_PER_SAMPLE = 2
"""PCM16 mono: two bytes per sample."""

_PROFILE = "default"
"""Explicit profile name, so ``AWS_PROFILE`` is never consulted."""

_RETRY_MODE = "standard"
"""The only retry mode smithy accepts as an override; ``max_attempts=1`` disables retries."""


# --- Structural types for the subset of the SDK stream that is used --------------------


class AudioPublisher(Protocol):
    """Input side of the duplex stream."""

    async def send(self, event: AudioStream) -> None: ...

    async def close(self) -> None: ...


class ResultReceiver(Protocol):
    """Output side of the duplex stream."""

    async def receive(self) -> TranscriptResultStream | None: ...

    async def close(self) -> None: ...


class TranscriptionStream(Protocol):
    """A started ``StartStreamTranscription`` duplex stream."""

    @property
    def input_stream(self) -> AudioPublisher: ...

    async def await_output(self) -> tuple[object, ResultReceiver]: ...


class StreamingClient(Protocol):
    """The part of ``AsyncTranscribeStreamingClient`` the adapter uses (fakeable in tests)."""

    async def start_stream_transcription(
        self, input: StartStreamTranscriptionInput
    ) -> TranscriptionStream: ...

    async def close(self) -> None: ...


ClientFactory = Callable[[], Awaitable[StreamingClient]]
Sleep = Callable[[float], Awaitable[None]]


# --- Client construction (explicit Settings credentials only) --------------------------


class _NoSharedConfig:
    """smithy ``FileSystem`` that never touches disk: one empty ``[default]`` profile."""

    async def read_file(self, path: str) -> str | None:
        """Return an empty default profile for every requested config path."""
        return f"[{_PROFILE}]\n"


async def transcribe_config(settings: Settings) -> AsyncTranscribeStreamingConfig:
    """Resolve the streaming client config from the explicit Settings values only.

    Credentials (with the session token only when set), region, retry settings, and
    endpoint are all explicit overrides, so nothing comes from the environment or from
    shared AWS config files (Req 12.11). The config owns a CRT transport; closing the
    client built from it closes the transport.
    """
    return await AsyncTranscribeStreamingConfig.resolve(
        profile=_PROFILE,
        fs=_NoSharedConfig(),
        config_file_path=os.devnull,
        credentials_file_path=os.devnull,
        region=settings.aws_region,
        aws_access_key_id=settings.aws_access_key_id,
        aws_secret_access_key=settings.aws_secret_access_key,
        aws_session_token=settings.aws_session_token,
        retry_mode=_RETRY_MODE,
        max_attempts=AWS_MAX_ATTEMPTS,
        endpoint_uri=None,
        sdk_ua_app_id=None,
        transport=AWSCRTHTTPClient(),
    )


async def make_transcribe_client(settings: Settings) -> AsyncTranscribeStreamingClient:
    """Build a Transcribe streaming client from :func:`transcribe_config`."""
    return AsyncTranscribeStreamingClient(config=await transcribe_config(settings))


# --- Pure helpers ----------------------------------------------------------------------


def chunk_size(sample_rate: int, chunk_ms: int = STT_CHUNK_MS) -> int:
    """Bytes in one ``chunk_ms`` chunk of PCM16 mono audio (whole samples, at least one)."""
    samples = max(1, sample_rate * chunk_ms // 1000)
    return samples * BYTES_PER_SAMPLE


def audio_chunks(pcm16: bytes, sample_rate: int, chunk_ms: int = STT_CHUNK_MS) -> list[bytes]:
    """Split audio into consecutive chunks of ``chunk_ms``; the last one may be shorter."""
    size = chunk_size(sample_rate, chunk_ms)
    return [pcm16[i : i + size] for i in range(0, len(pcm16), size)]


def final_transcripts(event: TranscriptEvent) -> list[str]:
    """Stripped, non-empty first-alternative texts of the non-partial results in ``event``."""
    transcript = event.transcript
    texts: list[str] = []
    for result in (transcript.results if transcript else None) or ():
        if result.is_partial or not result.alternatives:
            continue
        text = (result.alternatives[0].transcript or "").strip()
        if text:
            texts.append(text)
    return texts


def join_transcripts(parts: Sequence[str]) -> str:
    """Join final result texts with single spaces."""
    return " ".join(part.strip() for part in parts if part.strip())


def trim_silence(
    pcm16: bytes,
    sample_rate: int,
    level: int = STT_SILENCE_LEVEL,
    pad_ms: int = STT_EDGE_PAD_MS,
) -> bytes:
    """Drop leading and trailing near-silence, keeping ``pad_ms`` of audio at each edge.

    The browser ends a recording after a 2 s pause, so the tail is mostly silence; since
    audio is streamed in real time, trimming it cuts that wait from every request. Audio
    with no sample at or above ``level`` is returned unchanged (Transcribe then reports
    an empty transcript). Inner pauses are never touched.
    """
    samples = memoryview(pcm16[: len(pcm16) - len(pcm16) % BYTES_PER_SAMPLE]).cast("h")
    count = len(samples)
    # Scan inward from each edge only, so the cost is the silence, not the whole clip.
    start = next((i for i in range(count) if abs(samples[i]) >= level), None)
    if start is None:
        return pcm16
    end = next(i for i in range(count - 1, start - 1, -1) if abs(samples[i]) >= level)
    pad = sample_rate * pad_ms // 1000
    first = max(0, start - pad)
    last = min(count, end + 1 + pad)
    return pcm16[first * BYTES_PER_SAMPLE : last * BYTES_PER_SAMPLE]


# --- Adapter ---------------------------------------------------------------------------


class TranscribeSTT:
    """``STTClient`` backed by Amazon Transcribe ``StartStreamTranscription``."""

    def __init__(
        self,
        client_factory: ClientFactory,
        *,
        language_code: str,
        redactor: Redactor,
        sleep: Sleep = asyncio.sleep,
        pace_factor: float = STT_PACE_FACTOR,
        chunk_ms: int = STT_CHUNK_MS,
    ) -> None:
        """Wrap a client factory (called once, lazily) and the streaming parameters.

        ``sleep`` and ``pace_factor`` exist so tests can stream without real delays.
        """
        if pace_factor <= 0:
            raise ValueError("pace_factor must be positive")
        if chunk_ms <= 0:
            raise ValueError("chunk_ms must be positive")
        self._client_factory = client_factory
        self._language_code = language_code
        self._redactor = redactor
        self._sleep = sleep
        self._pace_factor = pace_factor
        self._chunk_ms = chunk_ms
        self._client: StreamingClient | None = None
        self._client_lock = asyncio.Lock()

    @classmethod
    def from_settings(cls, settings: Settings, redactor: Redactor) -> TranscribeSTT:
        """Build the adapter with the configured language code (called by ``wiring.py``)."""

        async def factory() -> StreamingClient:
            return await make_transcribe_client(settings)

        return cls(factory, language_code=settings.transcribe_language_code, redactor=redactor)

    async def transcribe(
        self,
        pcm16: bytes,
        sample_rate: int = STT_SAMPLE_RATE_HZ,
        timeout_s: float = STT_TIMEOUT_S,
    ) -> str:
        """Return the joined final transcript of mono PCM16 audio within ``timeout_s``.

        Raises:
            AwsCredentialError: the AWS credentials are expired, invalid, or lack access.
            STTError: ``stt_timeout`` when ``timeout_s`` elapses; ``stt_empty`` when the
                audio or the transcript is empty; ``stt_failed`` for any other failure.
        """
        if not pcm16:
            raise STTError("stt_empty", "No audio was recorded.")
        pcm16 = trim_silence(pcm16, sample_rate)
        # Real-time streaming takes as long as the audio itself, so the budget is the
        # response timeout plus the time needed to send the clip.
        send_s = len(pcm16) / BYTES_PER_SAMPLE / sample_rate / self._pace_factor
        try:
            text = await asyncio.wait_for(self._stream(pcm16, sample_rate), timeout_s + send_s)
        except FridayError:
            raise
        except Exception as exc:  # CancelledError is a BaseException and propagates.
            raise to_stt_error(exc, self._redactor) from exc
        if not text.strip():
            raise STTError("stt_empty", "The transcript was empty.")
        return text

    async def close(self) -> None:
        """Close the streaming client, if one was created."""
        client, self._client = self._client, None
        if client is not None:
            await client.close()

    async def _get_client(self) -> StreamingClient:
        """Create the streaming client on first use and reuse it afterwards."""
        async with self._client_lock:
            if self._client is None:
                self._client = await self._client_factory()
            return self._client

    def _request(self, sample_rate: int) -> StartStreamTranscriptionInput:
        """The ``StartStreamTranscription`` request for PCM audio in the configured language."""
        return StartStreamTranscriptionInput(
            language_code=LanguageCode(self._language_code),
            media_encoding=MediaEncoding.PCM,
            media_sample_rate_hertz=sample_rate,
        )

    async def _stream(self, pcm16: bytes, sample_rate: int) -> str:
        """Send the audio and collect the final results concurrently."""
        client = await self._get_client()
        stream = await client.start_stream_transcription(input=self._request(sample_rate))
        receiver: ResultReceiver | None = None
        try:
            async with asyncio.TaskGroup() as tasks:
                tasks.create_task(self._send_audio(stream.input_stream, pcm16, sample_rate))
                _, receiver = await stream.await_output()
                parts = await _receive_final(receiver)
        finally:
            # Each close is bounded: on a stalled stream close() waits for buffered audio
            # to drain, which never happens, and would otherwise block the request (and
            # its timeout's cancellation) forever.
            if receiver is not None:
                with suppress(Exception):
                    await asyncio.wait_for(receiver.close(), STT_CLOSE_TIMEOUT_S)
            with suppress(Exception):  # Idempotent; already closed on the normal path.
                await asyncio.wait_for(stream.input_stream.close(), STT_CLOSE_TIMEOUT_S)
        return join_transcripts(parts)

    async def _send_audio(self, publisher: AudioPublisher, pcm16: bytes, sample_rate: int) -> None:
        """Send the audio chunks at ``pace_factor`` times real time, then close the input."""
        delay = self._chunk_ms / 1000 / self._pace_factor
        for index, chunk in enumerate(audio_chunks(pcm16, sample_rate, self._chunk_ms)):
            if index:
                await self._sleep(delay)
            await publisher.send(AudioStreamAudioEvent(value=AudioEvent(audio_chunk=chunk)))
        await publisher.close()


async def _receive_final(receiver: ResultReceiver) -> list[str]:
    """Read result events until the stream ends; keep only non-partial transcripts."""
    parts: list[str] = []
    while (event := await receiver.receive()) is not None:
        if isinstance(event, TranscriptResultStreamTranscriptEvent):
            parts.extend(final_transcripts(event.value))
        elif not isinstance(event, TranscriptResultStreamUnknown):
            raise event.value  # A modeled service error delivered as an event.
    return parts
