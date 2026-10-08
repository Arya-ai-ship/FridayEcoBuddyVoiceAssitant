"""Unit tests for the Polly and Transcribe adapters (Req 3.3, 3.6, 4.6, 4.8).

Polly is driven through ``botocore.stub.Stubber`` so the configured voice, engine, and
output format reach ``synthesize_speech`` without a network call. Transcribe is driven
through a fake streaming client that satisfies the ``StreamingClient`` Protocol, so the
request's language code, the joined non-partial transcript, the empty-transcript case,
and the timeout and credential error mappings are all checked offline.
"""

# pyright: reportMissingTypeStubs=false

from __future__ import annotations

import asyncio
from collections.abc import Sequence

import pytest
from aws_sdk_transcribe_streaming.models import (
    Alternative,
    Result,
    StartStreamTranscriptionInput,
    Transcript,
    TranscriptEvent,
    TranscriptResultStream,
    TranscriptResultStreamTranscriptEvent,
)
from botocore.exceptions import ClientError
from botocore.stub import Stubber

from friday.config import Settings
from friday.errors import AwsCredentialError, STTError, TTSError
from friday.redact import Redactor
from friday.speech.stt import StreamingClient, TranscribeSTT, TranscriptionStream
from friday.speech.tts import PollyTTS, make_polly_client

pytestmark = pytest.mark.asyncio

FAKE_KEY_ID = "AKIAFAKEFAKEFAKEFAKE"
FAKE_SECRET = "FAKEsecretVALUEforTESTSonly0123456789xyz"  # noqa: S105 - fake placeholder
FAKE_TOKEN = "FAKEsessionTOKENforTESTSonly0123456789"  # noqa: S105 - fake placeholder
VOICE = "Matthew"
ENGINE = "generative"
LANGUAGE = "en-GB"
AUDIO = b"\x00\x01" * 1600  # 100 ms of PCM16 mono at 16 kHz


def _settings() -> Settings:
    return Settings(
        aws_access_key_id=FAKE_KEY_ID,
        aws_secret_access_key=FAKE_SECRET,
        aws_session_token=FAKE_TOKEN,
        aws_region="us-east-2",
        fred_api_key="fakefredkey000000000000000000000",
        bedrock_model_id="us.openai.gpt-5.6-terra",
        polly_voice_id=VOICE,
        polly_engine=ENGINE,
        transcribe_language_code=LANGUAGE,
    )


@pytest.fixture
def redactor() -> Redactor:
    return Redactor((FAKE_KEY_ID, FAKE_SECRET, FAKE_TOKEN))


# --- Polly (Req 4.6, 4.8) --------------------------------------------------------


class _AudioStream:
    """A minimal Polly ``AudioStream``: readable once, closeable, records the close."""

    def __init__(self, data: bytes) -> None:
        self._data = data
        self.closed = False

    def read(self) -> bytes:
        return self._data

    def close(self) -> None:
        self.closed = True


async def test_polly_called_with_configured_voice_engine_and_format(redactor: Redactor) -> None:
    client = make_polly_client(_settings())
    audio = _AudioStream(b"mp3-bytes")
    with Stubber(client) as stubber:
        stubber.add_response(
            "synthesize_speech",
            {"AudioStream": audio, "ContentType": "audio/mpeg", "RequestCharacters": 5},
            {"Text": "Hello", "VoiceId": VOICE, "Engine": ENGINE, "OutputFormat": "mp3"},
        )
        tts = PollyTTS(client, voice_id=VOICE, engine=ENGINE, redactor=redactor)
        result = await tts.synthesize("Hello")
    assert result == b"mp3-bytes"
    assert audio.closed  # the adapter reads then closes the stream
    stubber.assert_no_pending_responses()


async def test_polly_credential_error_maps_to_aws_credentials(redactor: Redactor) -> None:
    client = make_polly_client(_settings())
    with Stubber(client) as stubber:
        stubber.add_client_error(
            "synthesize_speech", service_error_code="ExpiredTokenException", http_status_code=403
        )
        tts = PollyTTS(client, voice_id=VOICE, engine=ENGINE, redactor=redactor)
        with pytest.raises(AwsCredentialError) as info:
            await tts.synthesize("Hello")
    assert info.value.kind == "aws_credentials"


async def test_polly_other_error_maps_to_tts_failed(redactor: Redactor) -> None:
    client = make_polly_client(_settings())
    with Stubber(client) as stubber:
        stubber.add_client_error(
            "synthesize_speech", service_error_code="ServiceFailure", http_status_code=500
        )
        tts = PollyTTS(client, voice_id=VOICE, engine=ENGINE, redactor=redactor)
        with pytest.raises(TTSError) as info:
            await tts.synthesize("Hello")
    assert info.value.kind == "tts_failed"


async def test_polly_timeout_maps_to_tts_timeout(redactor: Redactor) -> None:
    class SlowPolly:
        def synthesize_speech(self, **_kwargs: object) -> dict[str, object]:
            import time

            time.sleep(0.5)
            return {"AudioStream": _AudioStream(b"")}

    tts = PollyTTS(SlowPolly(), voice_id=VOICE, engine=ENGINE, redactor=redactor)  # type: ignore[arg-type]
    with pytest.raises(TTSError) as info:
        await tts.synthesize("Hello", timeout_s=0.01)
    assert info.value.kind == "tts_timeout"


async def test_polly_message_is_redacted(redactor: Redactor) -> None:
    client = make_polly_client(_settings())
    with Stubber(client) as stubber:
        stubber.add_client_error(
            "synthesize_speech",
            service_error_code="ServiceFailure",
            service_message=f"boom token={FAKE_TOKEN}",
            http_status_code=500,
        )
        tts = PollyTTS(client, voice_id=VOICE, engine=ENGINE, redactor=redactor)
        with pytest.raises(TTSError) as info:
            await tts.synthesize("Hello")
    assert FAKE_TOKEN not in info.value.message


# --- Transcribe (Req 3.3, 3.6) ---------------------------------------------------


def _transcript_event(text: str, *, is_partial: bool) -> TranscriptResultStreamTranscriptEvent:
    result = Result(is_partial=is_partial, alternatives=[Alternative(transcript=text)])
    return TranscriptResultStreamTranscriptEvent(
        value=TranscriptEvent(transcript=Transcript(results=[result]))
    )


class _FakeReceiver:
    """Yields a scripted list of result-stream events, then ``None``."""

    def __init__(self, events: Sequence[TranscriptResultStream]) -> None:
        self._events = list(events)
        self.closed = False

    async def receive(self) -> TranscriptResultStream | None:
        return self._events.pop(0) if self._events else None

    async def close(self) -> None:
        self.closed = True


class _FakePublisher:
    def __init__(self) -> None:
        self.sent = 0
        self.closed = False

    async def send(self, _event: object) -> None:
        self.sent += 1

    async def close(self) -> None:
        self.closed = True


class _FakeStream:
    def __init__(self, events: Sequence[TranscriptResultStream]) -> None:
        self.input_stream = _FakePublisher()
        self._receiver = _FakeReceiver(events)

    async def await_output(self) -> tuple[object, _FakeReceiver]:
        return object(), self._receiver


class _FakeClient:
    """A fake ``StreamingClient`` that records the request and returns a scripted stream."""

    def __init__(self, events: Sequence[TranscriptResultStream]) -> None:
        self._events = events
        self.requests: list[StartStreamTranscriptionInput] = []

    async def start_stream_transcription(
        self, input: StartStreamTranscriptionInput
    ) -> TranscriptionStream:
        self.requests.append(input)
        return _FakeStream(self._events)  # type: ignore[return-value]

    async def close(self) -> None:
        pass


def _stt(client: StreamingClient, redactor: Redactor) -> TranscribeSTT:
    async def factory() -> StreamingClient:
        return client

    async def no_sleep(_seconds: float) -> None:
        return None

    return TranscribeSTT(factory, language_code=LANGUAGE, redactor=redactor, sleep=no_sleep)


async def test_transcribe_uses_language_code_and_joins_final_results(redactor: Redactor) -> None:
    events = [
        _transcript_event("partial", is_partial=True),
        _transcript_event("Hello Boss", is_partial=False),
        _transcript_event("how are you", is_partial=False),
    ]
    client = _FakeClient(events)
    text = await _stt(client, redactor).transcribe(AUDIO)
    assert text == "Hello Boss how are you"
    language_code = client.requests[0].language_code
    assert language_code is not None and str(language_code.value) == LANGUAGE
    assert client.requests[0].media_sample_rate_hertz == 16000


async def test_transcribe_empty_transcript_raises_stt_empty(redactor: Redactor) -> None:
    client = _FakeClient([_transcript_event("   ", is_partial=False)])
    with pytest.raises(STTError) as info:
        await _stt(client, redactor).transcribe(AUDIO)
    assert info.value.kind == "stt_empty"


async def test_transcribe_no_audio_raises_stt_empty(redactor: Redactor) -> None:
    client = _FakeClient([])
    with pytest.raises(STTError) as info:
        await _stt(client, redactor).transcribe(b"")
    assert info.value.kind == "stt_empty"


async def test_transcribe_timeout_maps_to_stt_timeout(redactor: Redactor) -> None:
    class HangingClient:
        async def start_stream_transcription(
            self, input: StartStreamTranscriptionInput
        ) -> TranscriptionStream:
            await asyncio.sleep(10)
            raise AssertionError("unreachable")

        async def close(self) -> None:
            pass

    with pytest.raises(STTError) as info:
        await _stt(HangingClient(), redactor).transcribe(AUDIO, timeout_s=0.01)  # type: ignore[arg-type]
    assert info.value.kind == "stt_timeout"


async def test_transcribe_credential_error_maps_to_aws_credentials(redactor: Redactor) -> None:
    error = ClientError(
        {"Error": {"Code": "AccessDeniedException", "Message": "denied"}},
        "StartStreamTranscription",
    )

    class FailingClient:
        async def start_stream_transcription(
            self, input: StartStreamTranscriptionInput
        ) -> TranscriptionStream:
            raise error

        async def close(self) -> None:
            pass

    with pytest.raises(AwsCredentialError) as info:
        await _stt(FailingClient(), redactor).transcribe(AUDIO)  # type: ignore[arg-type]
    assert info.value.kind == "aws_credentials"


async def test_transcribe_other_error_maps_to_stt_failed(redactor: Redactor) -> None:
    class FailingClient:
        async def start_stream_transcription(
            self, input: StartStreamTranscriptionInput
        ) -> TranscriptionStream:
            raise RuntimeError(f"boom token={FAKE_TOKEN}")

        async def close(self) -> None:
            pass

    with pytest.raises(STTError) as info:
        await _stt(FailingClient(), redactor).transcribe(AUDIO)  # type: ignore[arg-type]
    assert info.value.kind == "stt_failed"
    assert FAKE_TOKEN not in info.value.message
