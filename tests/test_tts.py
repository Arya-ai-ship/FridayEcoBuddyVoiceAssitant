"""Tests for ``speech/tts.py`` ``PollyTTS`` (no network: Stubber or a fake client).

The real Polly client is built from placeholder credentials and only ever driven
through ``botocore.stub.Stubber``, which answers before any request is signed or sent.
"""

from __future__ import annotations

import io
import threading
from collections.abc import Iterator
from typing import Any, cast

import pytest
from botocore.response import StreamingBody
from botocore.stub import Stubber

from friday.config import Settings
from friday.constants import AWS_MAX_ATTEMPTS, TTS_CONNECT_TIMEOUT_S, TTS_TIMEOUT_S
from friday.errors import AwsCredentialError, TTSError
from friday.ports import TTSClient
from friday.redact import Redactor
from friday.speech.tts import PollyTTS, make_polly_client

FAKE_KEY_ID = "AKIAFAKEFAKEFAKEFAKE"
FAKE_SECRET = "FAKEsecretVALUEforTESTSonly0123456789xyz"  # noqa: S105 - fake placeholder
FAKE_TOKEN = "FAKEsessionTOKENforTESTSonly0123456789"  # noqa: S105 - fake placeholder
REGION = "us-east-2"
VOICE = "Matthew"
ENGINE = "standard"
AUDIO = b"ID3\x04fake-mp3-bytes"


@pytest.fixture
def settings() -> Settings:
    return Settings(
        aws_access_key_id=FAKE_KEY_ID,
        aws_secret_access_key=FAKE_SECRET,
        aws_session_token=FAKE_TOKEN,
        aws_region=REGION,
        fred_api_key="fakefredkey000000000000000000000",
        bedrock_model_id="us.openai.gpt-5.6-terra",
        polly_voice_id=VOICE,
        polly_engine=ENGINE,
    )


@pytest.fixture
def redactor() -> Redactor:
    return Redactor([FAKE_KEY_ID, FAKE_SECRET, FAKE_TOKEN])


@pytest.fixture
def stubbed(settings: Settings, redactor: Redactor) -> Iterator[tuple[PollyTTS, Stubber]]:
    """A ``PollyTTS`` over a real Polly client with an active Stubber."""
    client = make_polly_client(settings)
    with Stubber(client) as stubber:
        yield PollyTTS(client, voice_id=VOICE, engine=ENGINE, redactor=redactor), stubber
        stubber.assert_no_pending_responses()


def _expected(text: str) -> dict[str, str]:
    return {"Text": text, "VoiceId": VOICE, "Engine": ENGINE, "OutputFormat": "mp3"}


def _assert_redacted(message: str) -> None:
    for secret in (FAKE_SECRET, FAKE_TOKEN):
        assert secret[:12] not in message
        assert secret[-12:] not in message
    assert FAKE_KEY_ID not in message


# --- Client construction -----------------------------------------------------


def test_client_config_region_and_attempts(settings: Settings) -> None:
    client = make_polly_client(settings)
    config = client.meta.config
    assert client.meta.region_name == REGION
    assert client.meta.service_model.service_name == "polly"
    assert config.read_timeout == TTS_TIMEOUT_S
    assert config.connect_timeout == TTS_CONNECT_TIMEOUT_S <= TTS_TIMEOUT_S
    assert config.retries is not None
    assert config.retries["total_max_attempts"] == AWS_MAX_ATTEMPTS
    assert "max_attempts" not in config.retries


def test_from_settings_satisfies_the_port(settings: Settings, redactor: Redactor) -> None:
    assert isinstance(PollyTTS.from_settings(settings, redactor), TTSClient)


# --- Synthesis -----------------------------------------------------------------


async def test_success_returns_audio_with_configured_voice_engine_format(
    stubbed: tuple[PollyTTS, Stubber],
) -> None:
    tts, stubber = stubbed
    stubber.add_response(
        "synthesize_speech",
        {
            "AudioStream": StreamingBody(io.BytesIO(AUDIO), len(AUDIO)),
            "ContentType": "audio/mpeg",
            "RequestCharacters": 6,
        },
        expected_params=_expected("Hello."),
    )
    assert await tts.synthesize("Hello.") == AUDIO


@pytest.mark.parametrize("code", ["ExpiredTokenException", "UnrecognizedClientException"])
async def test_credential_client_error_maps_to_aws_credential_error(
    stubbed: tuple[PollyTTS, Stubber], code: str
) -> None:
    tts, stubber = stubbed
    stubber.add_client_error(
        "synthesize_speech",
        service_error_code=code,
        service_message=f"token {FAKE_TOKEN} for {FAKE_KEY_ID} is invalid",
        http_status_code=403,
        expected_params=_expected("Hi."),
    )
    with pytest.raises(AwsCredentialError) as info:
        await tts.synthesize("Hi.")
    assert info.value.kind == "aws_credentials"
    _assert_redacted(info.value.message)


async def test_generic_client_error_maps_to_tts_failed(stubbed: tuple[PollyTTS, Stubber]) -> None:
    tts, stubber = stubbed
    stubber.add_client_error(
        "synthesize_speech",
        service_error_code="ServiceFailureException",
        service_message=f"internal failure near {FAKE_SECRET}",
        http_status_code=500,
    )
    with pytest.raises(TTSError) as info:
        await tts.synthesize("Hi.")
    assert info.value.kind == "tts_failed"
    assert "ServiceFailureException" in info.value.message
    _assert_redacted(info.value.message)


class _Stream:
    """Minimal ``StreamingBody`` stand-in that records ``close()``."""

    def __init__(self, data: bytes) -> None:
        self.data = data
        self.closed = False

    def read(self) -> bytes:
        return self.data

    def close(self) -> None:
        self.closed = True


class _FakePolly:
    """Fake Polly client: blocks until released, then returns audio or raises."""

    def __init__(self, *, release: threading.Event | None = None, error: Exception | None = None):
        self.release = release
        self.error = error
        self.stream = _Stream(AUDIO)
        self.calls: list[dict[str, Any]] = []

    def synthesize_speech(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        if self.release is not None:
            self.release.wait(timeout=5)
        if self.error is not None:
            raise self.error
        return {"AudioStream": self.stream}


def _tts(fake: _FakePolly, redactor: Redactor) -> PollyTTS:
    return PollyTTS(cast(Any, fake), voice_id=VOICE, engine=ENGINE, redactor=redactor)


async def test_audio_stream_is_read_fully_and_closed(redactor: Redactor) -> None:
    fake = _FakePolly()
    assert await _tts(fake, redactor).synthesize("Done.") == AUDIO
    assert fake.stream.closed
    assert fake.calls == [_expected("Done.")]


async def test_slow_call_maps_to_tts_timeout(redactor: Redactor) -> None:
    release = threading.Event()
    fake = _FakePolly(release=release)
    try:
        with pytest.raises(TTSError) as info:
            await _tts(fake, redactor).synthesize("Pulling inflation.", timeout_s=0.01)
    finally:
        release.set()  # let the worker thread finish so the loop shuts down promptly
    assert info.value.kind == "tts_timeout"


async def test_unexpected_exception_maps_to_tts_failed_redacted(redactor: Redactor) -> None:
    fake = _FakePolly(error=RuntimeError(f"socket closed while sending {FAKE_SECRET}"))
    with pytest.raises(TTSError) as info:
        await _tts(fake, redactor).synthesize("Hi.")
    assert info.value.kind == "tts_failed"
    _assert_redacted(info.value.message)
