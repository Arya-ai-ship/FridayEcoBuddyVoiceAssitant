"""Ports: the Protocols the application layer uses to reach external services.

The Agent, the Friday middleware, and the Tool registry depend only on these Protocols.
The concrete adapters (``speech.tts.PollyTTS``, ``speech.stt.TranscribeSTT``,
``data.fred.FredClient``) and the test fakes are constructed in ``wiring.py`` (or injected
through ``build_container``). This module imports no SDK: stdlib and the domain layer only.

The LLM port is not defined here. It is MAF's chat-client base abstraction, which both the
``BedrockChatClient`` adapter and the scripted test client implement.

Every failure an implementation raises is a ``FridayError`` subclass from ``errors.py``, so
callers map ``error.kind`` straight to the wire code. SDK exceptions never escape a port.
"""

from collections.abc import Sequence
from datetime import date
from typing import Protocol, runtime_checkable

from friday.constants import STT_SAMPLE_RATE_HZ, STT_TIMEOUT_S, TTS_TIMEOUT_S
from friday.data.fetch import Observation


@runtime_checkable
class TTSClient(Protocol):
    """Text-to-speech for status lines and Friday responses (Req 4.6, 4.8)."""

    async def synthesize(self, text: str, timeout_s: float = TTS_TIMEOUT_S) -> bytes:
        """Return MP3 audio for ``text``, spoken with the configured voice and engine.

        The whole call, including any network round trip, finishes within ``timeout_s``
        seconds or fails.

        Raises:
            AwsCredentialError: the AWS credentials are expired, invalid, or lack access.
            TTSError: ``tts_timeout`` when ``timeout_s`` elapses; ``tts_failed`` for any
                other synthesis failure.
        """
        ...


@runtime_checkable
class STTClient(Protocol):
    """Speech-to-text for recorded Mic_Button audio (Req 3.3, 3.6)."""

    async def transcribe(
        self,
        pcm16: bytes,
        sample_rate: int = STT_SAMPLE_RATE_HZ,
        timeout_s: float = STT_TIMEOUT_S,
    ) -> str:
        """Return the final transcript of mono 16-bit little-endian PCM audio.

        The audio is held only in memory. The non-partial results are joined into one
        non-empty string, and the whole call finishes within ``timeout_s`` seconds or fails.

        Raises:
            AwsCredentialError: the AWS credentials are expired, invalid, or lack access.
            STTError: ``stt_timeout`` when ``timeout_s`` elapses; ``stt_empty`` when the
                transcript is empty or whitespace only; ``stt_failed`` for any other failure.
        """
        ...


@runtime_checkable
class FredSource(Protocol):
    """Source of FRED series observations for the Fetch_Tool (Req 6.2, 6.10, 6.15)."""

    async def observations(
        self, series_id: str, start: date | None, end: date | None
    ) -> Sequence[Observation]:
        """Return the observations of one FRED series between ``start`` and ``end``.

        Each observation is a ``{"date": "YYYY-MM-DD", "value": "<number>" | "."}``
        mapping, the shape ``data.fetch.build_dataset`` consumes. ``None`` leaves that side
        of the range unbounded. Exactly one series is requested per call, and the result
        is never empty.

        Raises:
            FredError: ``http_error`` for a non-success response or transport failure;
                ``timeout`` when the request exceeds the FRED timeout; ``no_observations``
                when the series has no observations in the range. Messages never contain
                the API key.
        """
        ...
