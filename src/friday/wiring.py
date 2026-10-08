"""Composition root: build the ``Container`` of long-lived objects (Req 5.2, 5.14, 6.1, 12.11).

``build_container`` is the only place concrete adapters are constructed. It builds the MAF
Bedrock chat client (unless overridden), the Friday middleware with ``map_llm_error``, the
shared harness agent, the ``SessionStore`` over ``harness.create_session``, the Friday
``Agent`` facade, one ``Redactor``, the ``ChartRenderer``, a shared ``httpx.AsyncClient``,
the Polly/Transcribe/STS boto3 clients, and the Indicator_Map. The ``chat_client``, ``tts``,
``stt``, and ``fred`` keyword overrides let tests inject fakes; otherwise the real adapters
are built. Nothing here is a module-level singleton; the ``Container`` owns every resource
and ``close()`` releases them on shutdown.

This module and ``llm/bedrock.py`` are the only modules that import the MAF Bedrock package.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Final, cast

import httpx
from agent_framework import Agent as HarnessAgent
from agent_framework import AgentSession, BaseChatClient, InMemoryHistoryProvider

from friday.agent.harness import build_harness_agent
from friday.agent.loop import Agent
from friday.agent.middleware import FridayMiddleware
from friday.agent.tools import ToolRegistry
from friday.aws_session import isolated_boto3_session
from friday.config import Settings
from friday.constants import CLIENT_TIMEOUT_S
from friday.data.fred import FredClient
from friday.data.indicators import IndicatorMap, load_indicator_map
from friday.data.plot import ChartRenderer
from friday.llm.bedrock import make_chat_client, make_llm_error_mapper
from friday.ports import FredSource, STTClient, TTSClient
from friday.redact import Redactor
from friday.session import SessionStore
from friday.speech.stt import TranscribeSTT
from friday.speech.tts import PollyTTS

_log: Final = logging.getLogger(__name__)
STS_SERVICE: Final = "sts"
"""boto3 service name for the ``--check`` credential probe."""


@dataclass(frozen=True)
class Container:
    """Every long-lived object one running Friday process needs; closed on shutdown."""

    settings: Settings
    redactor: Redactor
    indicators: IndicatorMap
    chat_client: BaseChatClient
    tools: ToolRegistry
    renderer: ChartRenderer
    agent: Agent
    sessions: SessionStore
    tts: TTSClient
    stt: STTClient
    fred: FredSource
    http: httpx.AsyncClient
    sts_client: object

    async def close(self) -> None:
        """Release the HTTP client, the Transcribe stream client, and the chat client."""
        await _safe_close(self.http.aclose())
        close = getattr(self.stt, "close", None)
        if callable(close):
            await _safe_close(close())
        chat_close = getattr(self.chat_client, "close", None)
        if callable(chat_close):
            await _safe_close(chat_close())


async def _safe_close(awaitable: object) -> None:
    """Await a close coroutine, swallowing any error so shutdown always completes."""
    if awaitable is None:
        return
    try:
        await awaitable  # type: ignore[misc]
    except Exception as exc:  # noqa: BLE001 - shutdown must not raise
        _log.warning("Error during Container shutdown: %s", type(exc).__name__)


def build_container(
    settings: Settings,
    *,
    chat_client: BaseChatClient | None = None,
    tts: TTSClient | None = None,
    stt: STTClient | None = None,
    fred: FredSource | None = None,
) -> Container:
    """Assemble the ``Container``; ``None`` arguments fall back to the real adapters."""
    redactor = Redactor(settings_secrets(settings))
    indicators = load_indicator_map(settings.indicator_map_path)
    renderer = ChartRenderer()
    http = httpx.AsyncClient(timeout=CLIENT_TIMEOUT_S, follow_redirects=False)

    session = isolated_boto3_session(settings)
    sts_client = session.client(STS_SERVICE, region_name=settings.aws_region)  # type: ignore[call-overload]

    client: BaseChatClient = (
        chat_client if chat_client is not None else cast(BaseChatClient, make_chat_client(settings))
    )
    fred_source: FredSource = fred if fred is not None else FredClient.from_settings(settings, http)
    tts_client: TTSClient = tts if tts is not None else PollyTTS.from_settings(settings, redactor)
    stt_client: STTClient = (
        stt if stt is not None else TranscribeSTT.from_settings(settings, redactor)
    )

    tools = ToolRegistry(fred_source, renderer, indicators)
    history = InMemoryHistoryProvider()
    middleware = FridayMiddleware.build(tools, tts_client, make_llm_error_mapper(redactor))
    harness: HarnessAgent = build_harness_agent(client, tools, middleware, history)
    agent = Agent(harness, tts_client, history, redactor)
    sessions = SessionStore(lambda: _new_agent_session(harness))

    return Container(
        settings=settings,
        redactor=redactor,
        indicators=indicators,
        chat_client=client,
        tools=tools,
        renderer=renderer,
        agent=agent,
        sessions=sessions,
        tts=tts_client,
        stt=stt_client,
        fred=fred_source,
        http=http,
        sts_client=sts_client,
    )


def _new_agent_session(harness: HarnessAgent) -> AgentSession:
    """Create a fresh ``AgentSession`` through the shared harness agent."""
    return harness.create_session()


def settings_secrets(settings: Settings) -> list[str]:
    """Credential values the Redactor must mask (never logged or returned)."""
    from friday.config import secret_values

    return secret_values(settings)
