"""Tests for the test doubles in ``fakes.py`` and the autouse network guard.

The harness tests run the real MAF agent (``create_harness_agent`` with every harness
feature disabled) on ``ScriptedChatClient``; the network guard is active throughout.
"""

import asyncio
import socket
from datetime import date
from typing import Any

import pytest
from agent_framework import Agent, FunctionTool, Message, create_harness_agent
from agent_framework.exceptions import ChatClientException
from botocore.exceptions import ClientError

from fakes import (
    Call,
    FakeFred,
    FakeSTT,
    FakeTTS,
    Hang,
    Raise,
    ScriptedChatClient,
    ScriptExhaustedError,
    calls,
    credential_error,
    text,
)
from friday.aws_errors import ErrorKind, classify_aws_error
from friday.constants import MAX_TOOL_CALLS
from friday.errors import AwsCredentialError, FredError, STTError, TTSError
from friday.ports import FredSource, STTClient, TTSClient

LIMITS: Any = {
    "max_iterations": MAX_TOOL_CALLS + 1,
    "max_consecutive_errors_per_request": MAX_TOOL_CALLS + 1,
    "allow_concurrent_invocation": False,
}
CITY_SCHEMA = {
    "type": "object",
    "properties": {"city": {"type": "string"}},
    "required": ["city"],
}


def make_agent(client: ScriptedChatClient, seen: list[str]) -> Agent[Any]:
    def lookup(city: str) -> str:
        seen.append(city)
        return f"sunny in {city}"

    tool = FunctionTool(
        name="lookup",
        description="Look up the weather.",
        func=lookup,
        input_model=CITY_SCHEMA,
        approval_mode="never_require",
    )
    return create_harness_agent(
        client,
        harness_instructions="HARNESS",
        agent_instructions="AGENT",
        tools=[tool],
        disable_todo=True,
        disable_mode=True,
        disable_file_memory=True,
        disable_web_search=True,
        disable_tool_auto_approval=True,
        disable_compaction=True,
    )


def function_results(message: Message) -> list[str]:
    return [str(c.result) for c in message.contents if c.type == "function_result"]


async def test_scripted_function_call_runs_on_real_harness(network_guard: Any) -> None:
    client = ScriptedChatClient(
        [calls(Call("lookup", {"city": "Austin"})), text("It is sunny, Boss.")],
        function_invocation_configuration=LIMITS,
    )
    seen: list[str] = []
    agent = make_agent(client, seen)

    response = await agent.run("weather?", session=agent.create_session())

    assert response.text == "It is sunny, Boss."
    assert seen == ["Austin"]
    assert client.remaining == 0
    assert [c.tool_names for c in client.calls] == [("lookup",), ("lookup",)]
    assert all(c.instructions == "HARNESS\n\nAGENT" for c in client.calls)
    assert client.calls[0].messages[-1].text == "weather?"
    assert function_results(client.calls[1].messages[-1]) == ["sunny in Austin"]
    assert network_guard.attempts == []


async def test_several_calls_in_one_response_run_in_order() -> None:
    client = ScriptedChatClient(
        [calls(Call("lookup", {"city": "Austin"}), Call("lookup", {"city": "Boston"})), text("ok")],
        function_invocation_configuration=LIMITS,
    )
    seen: list[str] = []
    response = await make_agent(client, seen).run("two", session=None)
    assert response.text == "ok"
    assert seen == ["Austin", "Boston"]
    assert function_results(client.calls[1].messages[-1]) == ["sunny in Austin", "sunny in Boston"]


async def test_streaming_run_replays_script() -> None:
    client = ScriptedChatClient(
        [calls(Call("lookup", {"city": "Austin"})), text("streamed")],
        function_invocation_configuration=LIMITS,
    )
    seen: list[str] = []
    stream = make_agent(client, seen).run("hi", session=None, stream=True)
    response = await stream.get_final_response()
    assert response.text == "streamed"
    assert seen == ["Austin"]


@pytest.mark.parametrize("wrapped", [False, True])
async def test_credential_error_propagates_raw(wrapped: bool) -> None:
    client = ScriptedChatClient(
        [calls(Call("lookup", {"city": "Austin"})), Raise(credential_error(wrapped=wrapped))]
    )
    expected = ChatClientException if wrapped else ClientError
    with pytest.raises(expected) as info:
        await make_agent(client, []).run("hi", session=None)
    assert classify_aws_error(info.value) is ErrorKind.CREDENTIAL
    assert len(client.calls) == 2


async def test_hang_is_cancellable_and_recorded() -> None:
    client = ScriptedChatClient([Hang()])
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(make_agent(client, []).run("hi", session=None), timeout=0.05)
    assert client.hang_started.is_set()
    assert len(client.calls) == 1


async def test_released_hang_continues_and_exhaustion_raises() -> None:
    client = ScriptedChatClient([Hang(), text("late")])
    client.release.set()
    response = await make_agent(client, []).run("hi", session=None)
    assert response.text == "late"
    with pytest.raises(ScriptExhaustedError):
        await make_agent(client, []).run("again", session=None)


def test_fakes_satisfy_ports() -> None:
    assert isinstance(FakeTTS(), TTSClient)
    assert isinstance(FakeSTT(), STTClient)
    assert isinstance(FakeFred(), FredSource)


async def test_fake_tts_and_stt_failures() -> None:
    tts = FakeTTS(failure="timeout", fail_when=lambda t: t.startswith("Pulling"))
    assert await tts.synthesize("Done.") == b"ID3Done."
    with pytest.raises(TTSError) as tts_info:
        await tts.synthesize("Pulling inflation")
    assert tts_info.value.kind == "tts_timeout"
    with pytest.raises(AwsCredentialError):
        await FakeTTS(failure="credentials").synthesize("x")
    assert tts.calls == ["Done.", "Pulling inflation"]

    stt = FakeSTT(transcript="pull GDP")
    assert await stt.transcribe(b"\x00\x00") == "pull GDP"
    assert stt.calls[0].pcm16 == b"\x00\x00"
    with pytest.raises(STTError) as stt_info:
        await FakeSTT(failure="empty").transcribe(b"")
    assert stt_info.value.kind == "stt_empty"


async def test_fake_fred_filters_records_and_fails() -> None:
    fred = FakeFred()
    rows = await fred.observations("UNRATE", date(2024, 1, 1), date(2024, 3, 1))
    assert [r["date"] for r in rows] == ["2024-01-01", "2024-02-01", "2024-03-01"]
    with pytest.raises(FredError) as empty:
        await fred.observations("UNRATE", date(1990, 1, 1), date(1990, 12, 1))
    assert empty.value.kind == "no_observations"
    assert [c.series_id for c in fred.calls] == ["UNRATE", "UNRATE"]

    failing = FakeFred(failure="http_error", fail_series=frozenset({"GDP"}))
    assert await failing.observations("UNRATE", None, None)
    with pytest.raises(FredError) as http:
        await failing.observations("GDP", None, None)
    assert http.value.kind == "http_error"


@pytest.mark.parametrize("address", [("192.0.2.1", 80), ("example.com", 443), ("2001:db8::1", 80)])
def test_guard_blocks_non_loopback_connect(network_guard: Any, address: tuple[str, int]) -> None:
    family = socket.AF_INET6 if ":" in address[0] else socket.AF_INET
    with socket.socket(family, socket.SOCK_STREAM) as sock:
        with pytest.raises(RuntimeError, match="blocked non-loopback"):
            sock.connect(address)
        with pytest.raises(RuntimeError, match="blocked non-loopback"):
            sock.connect_ex(address)
    assert len(network_guard.attempts) == 2
    network_guard.attempts.clear()  # acknowledged, so teardown does not fail the test


def test_guard_allows_loopback(network_guard: Any) -> None:
    with (
        socket.create_server(("127.0.0.1", 0)) as server,
        socket.create_connection(server.getsockname(), timeout=1),
    ):
        pass
    assert network_guard.attempts == []
