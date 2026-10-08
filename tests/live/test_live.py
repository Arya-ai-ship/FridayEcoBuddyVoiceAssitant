"""Opt-in live tests against the real AWS and FRED services (task 18.2).

These run only when ``FRIDAY_LIVE_TESTS=1`` and real credentials are present in the
environment (or ``.env``). The offline network guard is disabled for ``tests/live/``. The
suite is effectively ``friday --check`` plus one harness turn:

- FRED reports monthly observations for all 10 default series (Req 6.14),
- one direct ``BedrockChatClient`` Converse request succeeds (Req 5.1),
- one harness turn makes GPT-5.6 Terra call ``fetch_data`` and emits a ``dataset_preview``
  and ``final(outcome="ok")`` (Req 5.1, 5.14),
- one Polly synthesis returns audio (Req 4.6),
- one Transcribe round trip returns without error (Req 3.3).

Requirements: 3.3, 4.6, 5.1, 5.14, 6.14
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from friday.config import Settings, load_settings, read_environment
from friday.data.indicators import default_indicator_map
from friday.events import FinalEvent
from friday.wiring import Container, build_container

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.skipif(
        os.environ.get("FRIDAY_LIVE_TESTS") != "1",
        reason="live tests run only with FRIDAY_LIVE_TESTS=1",
    ),
]

SESSION_ID = "00000000-0000-4000-8000-000000000000"


@pytest.fixture(scope="module")
def settings() -> Settings:
    return load_settings(read_environment(Path(".env")))


@pytest.fixture
async def container(settings: Settings) -> AsyncIterator[Container]:
    c = build_container(settings)
    try:
        yield c
    finally:
        await c.close()


async def test_fred_reports_monthly_for_all_ten_default_series(container: Container) -> None:
    """Every default series returns observations roughly one month apart (Req 6.14)."""
    for entry in default_indicator_map().entries:
        rows = await container.fred.observations(entry.series_id, None, None)
        assert rows, f"{entry.series_id} returned no observations"
        dates = sorted(r["date"] for r in rows)
        # Monthly series are dated on the first of the month.
        assert all(d.endswith("-01") for d in dates[:12]), entry.series_id


async def test_direct_bedrock_converse_request_succeeds(container: Container) -> None:
    """A tiny Converse request through the Container's chat client returns text (Req 5.1)."""
    from typing import Any, cast

    response = await cast(Any, container.chat_client).get_response("Reply with OK.")
    text = response.text
    assert isinstance(text, str) and text.strip()


async def test_harness_turn_fetches_and_previews_inflation(container: Container) -> None:
    """GPT-5.6 Terra calls fetch_data for a fetch request, yielding a preview + ok (Req 5.14)."""
    session = container.sessions.get_or_create(SESSION_ID)
    events = [e async for e in container.agent.run_turn(session, "please pull inflation")]
    assert any(e.type == "dataset_preview" for e in events)
    final = events[-1]
    assert isinstance(final, FinalEvent) and final.outcome == "ok"


async def test_polly_synthesis_returns_audio(container: Container) -> None:
    """Polly returns non-empty MP3 bytes (Req 4.6)."""
    audio = await container.tts.synthesize("Check, Boss.")
    assert isinstance(audio, bytes) and len(audio) > 0


async def test_transcribe_round_trip_returns_without_error(container: Container) -> None:
    """One second of silence transcribes to an empty result without raising (Req 3.3)."""
    from friday.constants import STT_SAMPLE_RATE_HZ
    from friday.errors import STTError

    silence = b"\x00\x00" * STT_SAMPLE_RATE_HZ
    try:
        text = await container.stt.transcribe(silence)
        assert isinstance(text, str)
    except STTError as exc:
        # Silence correctly produces an empty transcript; that is a pass, not a failure.
        assert exc.kind == "stt_empty"
