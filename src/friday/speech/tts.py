"""Amazon Polly text-to-speech adapter (Req 4.6, 4.8, 5.12, 12.11).

:class:`PollyTTS` implements the :class:`~friday.ports.TTSClient` Protocol. It calls
``synthesize_speech`` with the configured voice and engine and ``OutputFormat="mp3"``.
The blocking boto3 call, including reading and closing the audio stream, runs in a
worker thread through ``asyncio.to_thread`` under ``asyncio.wait_for``, so the
coroutine returns within ``timeout_s`` even if the thread is still waiting on the
network. Every failure is mapped by :func:`~friday.aws_errors.to_tts_error` to
``AwsCredentialError`` or ``TTSError`` (``tts_timeout``/``tts_failed``) with a
redacted message, so no SDK exception escapes the port.

:func:`make_polly_client` builds the client from :func:`isolated_boto3_session`, so
only the Settings credentials, region, and the default AWS endpoint are used.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, cast

from botocore.config import Config

from friday.aws_errors import to_tts_error
from friday.aws_session import isolated_boto3_session
from friday.config import Settings
from friday.constants import AWS_MAX_ATTEMPTS, TTS_CONNECT_TIMEOUT_S, TTS_TIMEOUT_S
from friday.redact import Redactor

if TYPE_CHECKING:  # boto3-stubs is a dev-only dependency.
    from mypy_boto3_polly import PollyClient
    from mypy_boto3_polly.literals import EngineType, VoiceIdType

POLLY_SERVICE = "polly"
"""boto3 service name of Amazon Polly."""

OUTPUT_FORMAT = "mp3"
"""Audio format returned to the Browser_UI."""


def polly_config() -> Config:
    """botocore config for Polly: timeouts inside the TTS budget, a single attempt.

    ``total_max_attempts`` is used because ``max_attempts=1`` still allows one retry
    (spike finding 7). ``asyncio.wait_for`` in :meth:`PollyTTS.synthesize` stays the
    authoritative limit.
    """
    return Config(
        read_timeout=TTS_TIMEOUT_S,
        connect_timeout=TTS_CONNECT_TIMEOUT_S,
        retries={"total_max_attempts": AWS_MAX_ATTEMPTS},
    )


def make_polly_client(settings: Settings) -> PollyClient:
    """Build a Polly client from the explicit Settings credentials and region only."""
    # Only the bedrock-runtime/polly/sts stubs are installed; the other overloads are unknown.
    return isolated_boto3_session(settings).client(  # pyright: ignore[reportUnknownMemberType]
        POLLY_SERVICE,
        region_name=settings.aws_region,
        config=polly_config(),
    )


class PollyTTS:
    """``TTSClient`` backed by Amazon Polly ``synthesize_speech``."""

    def __init__(
        self, client: PollyClient, *, voice_id: str, engine: str, redactor: Redactor
    ) -> None:
        """Wrap a Polly ``client`` (real, Stubber-activated, or fake) and a Redactor."""
        self._client = client
        self._voice_id = voice_id
        self._engine = engine
        self._redactor = redactor

    @classmethod
    def from_settings(cls, settings: Settings, redactor: Redactor) -> PollyTTS:
        """Build the adapter with the configured voice and engine (called by ``wiring.py``)."""
        return cls(
            make_polly_client(settings),
            voice_id=settings.polly_voice_id,
            engine=settings.polly_engine,
            redactor=redactor,
        )

    async def synthesize(self, text: str, timeout_s: float = TTS_TIMEOUT_S) -> bytes:
        """Return MP3 audio for ``text`` within ``timeout_s`` seconds.

        Raises:
            AwsCredentialError: the AWS credentials are expired, invalid, or lack access.
            TTSError: ``tts_timeout`` when ``timeout_s`` elapses (or botocore times out);
                ``tts_failed`` for any other failure.
        """
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(self._synthesize_blocking, text), timeout_s
            )
        except Exception as exc:  # CancelledError is a BaseException and propagates.
            raise to_tts_error(exc, self._redactor) from exc

    def _synthesize_blocking(self, text: str) -> bytes:
        """Call Polly and read the whole audio stream, closing it (runs in a worker thread)."""
        response = self._client.synthesize_speech(
            Text=text,
            # Settings hold plain strings; Polly validates the voice and engine names.
            VoiceId=cast("VoiceIdType", self._voice_id),
            Engine=cast("EngineType", self._engine),
            OutputFormat=OUTPUT_FORMAT,
        )
        stream = response["AudioStream"]
        try:
            return stream.read()
        finally:
            stream.close()
