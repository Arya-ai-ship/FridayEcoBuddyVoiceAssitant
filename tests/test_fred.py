"""Unit tests for ``data.fred.FredClient`` over ``httpx.MockTransport`` (no network).

Covers the request shape (one series, ``file_type=json``, optional bounds, timeout), the
response shaping, each ``FredError`` kind, and that the API key never appears in errors
or log lines (Req 6.2, 6.10, 6.15, 12.4).
"""

from __future__ import annotations

import asyncio
import logging
import math
from collections.abc import AsyncIterator, Callable, Coroutine
from datetime import date
from typing import Any

import httpx
import pytest

from friday.config import Settings
from friday.constants import FRED_TIMEOUT_S
from friday.data.fetch import build_dataset
from friday.data.fred import FRED_OBSERVATIONS_URL, FredClient, parse_observations
from friday.data.indicators import IndicatorEntry
from friday.errors import FredError
from friday.ports import FredSource

API_KEY = "placeholder-fred-key-0123456789"  # not a real credential
PLACEHOLDER = "placeholder"  # not a real credential
SERIES = "UNRATE"

Handler = Callable[[httpx.Request], Coroutine[Any, Any, httpx.Response]]


def _ok_body(*observations: dict[str, Any]) -> dict[str, Any]:
    return {"count": len(observations), "observations": list(observations)}


def _obs(day: str, value: str) -> dict[str, Any]:
    return {
        "realtime_start": "2024-01-01",
        "realtime_end": "2024-01-01",
        "date": day,
        "value": value,
    }


class Recorder:
    """Async MockTransport handler that records requests and returns a fixed response."""

    def __init__(self, respond: Callable[[httpx.Request], httpx.Response]) -> None:
        self.requests: list[httpx.Request] = []
        self._respond = respond

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self._respond(request)


@pytest.fixture
async def make_client() -> AsyncIterator[Callable[..., FredClient]]:
    clients: list[httpx.AsyncClient] = []

    def build(handler: Handler, **kwargs: Any) -> FredClient:
        http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        clients.append(http)
        return FredClient(http, api_key=API_KEY, **kwargs)

    yield build
    for http in clients:
        await http.aclose()


def _friday_log(caplog: pytest.LogCaptureFixture) -> str:
    """Rendered ``friday.*`` log records (httpx's own request log is silenced by the CLI)."""
    lines = [r.getMessage() for r in caplog.records if r.name.startswith("friday")]
    assert lines, "expected the client to log the failure"
    return "\n".join(lines)


def _assert_no_key(text: str) -> None:
    assert API_KEY not in text
    assert API_KEY[:8] not in text
    assert "api_key" not in text
    assert "stlouisfed" not in text


async def test_satisfies_fred_source_protocol(make_client: Callable[..., FredClient]) -> None:
    client = make_client(Recorder(lambda _r: httpx.Response(200, json=_ok_body())))
    assert isinstance(client, FredSource)
    _assert_no_key(repr(client))


async def test_request_has_one_series_json_bounds_and_timeout(
    make_client: Callable[..., FredClient],
) -> None:
    body = _ok_body(_obs("2024-01-01", "3.7"), _obs("2024-02-01", "."))
    recorder = Recorder(lambda _r: httpx.Response(200, json=body))
    client = make_client(recorder)

    result = await client.observations(SERIES, date(2024, 1, 1), date(2024, 2, 29))

    assert [dict(o) for o in result] == [
        {"date": "2024-01-01", "value": "3.7"},
        {"date": "2024-02-01", "value": "."},
    ]
    (request,) = recorder.requests
    assert request.method == "GET"
    assert str(request.url.copy_with(query=None)) == FRED_OBSERVATIONS_URL
    assert dict(request.url.params) == {
        "series_id": SERIES,
        "api_key": API_KEY,
        "file_type": "json",
        "observation_start": "2024-01-01",
        "observation_end": "2024-02-29",
    }
    assert request.extensions["timeout"] == httpx.Timeout(FRED_TIMEOUT_S).as_dict()


async def test_unbounded_request_omits_bounds(make_client: Callable[..., FredClient]) -> None:
    recorder = Recorder(lambda _r: httpx.Response(200, json=_ok_body(_obs("2024-01-01", "1"))))
    await make_client(recorder).observations(SERIES, None, None)
    params = recorder.requests[0].url.params
    assert "observation_start" not in params
    assert "observation_end" not in params


async def test_result_is_read_only(make_client: Callable[..., FredClient]) -> None:
    client = make_client(
        Recorder(lambda _r: httpx.Response(200, json=_ok_body(_obs("2024-01-01", "1"))))
    )
    (obs,) = await client.observations(SERIES, None, None)
    with pytest.raises(TypeError):
        obs["value"] = "2"  # type: ignore[index]


@pytest.mark.parametrize("status", [400, 404, 429, 500, 503])
async def test_error_status_is_http_error(
    make_client: Callable[..., FredClient], status: int, caplog: pytest.LogCaptureFixture
) -> None:
    body = {"error_code": status, "error_message": f"Bad Request. api_key={API_KEY} rejected"}
    client = make_client(Recorder(lambda _r: httpx.Response(status, json=body)))
    with caplog.at_level(logging.DEBUG), pytest.raises(FredError) as info:
        await client.observations(SERIES, None, None)
    assert info.value.kind == "http_error"
    assert SERIES in info.value.message
    assert str(status) in info.value.message
    _assert_no_key(str(info.value))
    _assert_no_key(_friday_log(caplog))


async def test_transport_error_is_http_error(
    make_client: Callable[..., FredClient], caplog: pytest.LogCaptureFixture
) -> None:
    async def fail(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(f"cannot reach {request.url}", request=request)

    with caplog.at_level(logging.DEBUG), pytest.raises(FredError) as info:
        await make_client(fail).observations(SERIES, None, None)
    assert info.value.kind == "http_error"
    assert SERIES in info.value.message
    assert info.value.__suppress_context__
    _assert_no_key(str(info.value))
    _assert_no_key(_friday_log(caplog))


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(200, content=b"<html>not json</html>"),
        httpx.Response(200, json=["not", "an", "object"]),
        httpx.Response(200, json={"count": 0}),
        httpx.Response(200, json={"observations": [{"date": "2024-01-01"}]}),
        httpx.Response(200, json={"observations": [{"date": "20240101", "value": "1"}]}),
        httpx.Response(200, json={"observations": [{"date": "2024-01-01", "value": "n/a"}]}),
        httpx.Response(200, json={"observations": ["2024-01-01"]}),
    ],
    ids=[
        "not-json",
        "array",
        "no-observations-key",
        "no-value",
        "bad-date",
        "bad-value",
        "entry-not-object",
    ],
)
async def test_malformed_body_is_http_error(
    make_client: Callable[..., FredClient], response: httpx.Response
) -> None:
    with pytest.raises(FredError) as info:
        await make_client(Recorder(lambda _r: response)).observations(SERIES, None, None)
    assert info.value.kind == "http_error"
    assert SERIES in info.value.message
    _assert_no_key(str(info.value))


async def test_error_body_with_success_status_is_http_error(
    make_client: Callable[..., FredClient], caplog: pytest.LogCaptureFixture
) -> None:
    body = {"error_code": 400, "error_message": f"Bad Request. api_key={API_KEY} rejected"}
    client = make_client(Recorder(lambda _r: httpx.Response(200, json=body)))
    with caplog.at_level(logging.DEBUG), pytest.raises(FredError) as info:
        await client.observations(SERIES, None, None)
    assert info.value.kind == "http_error"
    assert "Bad Request" not in info.value.message
    _assert_no_key(str(info.value))
    _assert_no_key(_friday_log(caplog))


@pytest.mark.parametrize("status", [301, 302, 307, 308])
async def test_redirect_is_http_error_and_not_followed(
    status: int, caplog: pytest.LogCaptureFixture
) -> None:
    recorder = Recorder(
        lambda _r: httpx.Response(status, headers={"Location": "https://elsewhere.test/obs"})
    )
    # The Container owns the client's redirect policy; the FRED client must not follow anyway.
    transport = httpx.MockTransport(recorder)
    async with httpx.AsyncClient(transport=transport, follow_redirects=True) as http:
        with caplog.at_level(logging.DEBUG), pytest.raises(FredError) as info:
            await FredClient(http, api_key=API_KEY).observations(SERIES, None, None)
    assert info.value.kind == "http_error"
    assert str(status) in info.value.message
    assert [r.url.host for r in recorder.requests] == ["api.stlouisfed.org"]
    _assert_no_key(str(info.value))
    _assert_no_key(_friday_log(caplog))


@pytest.mark.parametrize(
    "exc_type", [httpx.ConnectTimeout, httpx.ReadTimeout, httpx.WriteTimeout, httpx.PoolTimeout]
)
async def test_httpx_timeout_is_timeout(
    make_client: Callable[..., FredClient],
    exc_type: type[httpx.TimeoutException],
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def slow(request: httpx.Request) -> httpx.Response:
        raise exc_type(f"timed out on {request.url}", request=request)

    with caplog.at_level(logging.DEBUG), pytest.raises(FredError) as info:
        await make_client(slow).observations(SERIES, None, None)
    assert info.value.kind == "timeout"
    assert SERIES in info.value.message
    assert info.value.__suppress_context__
    _assert_no_key(str(info.value))
    _assert_no_key(_friday_log(caplog))


async def test_overall_deadline_is_timeout(make_client: Callable[..., FredClient]) -> None:
    async def hang(_request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(10)
        return httpx.Response(200, json=_ok_body())

    with pytest.raises(FredError) as info:
        await make_client(hang, timeout_s=0.05).observations(SERIES, None, None)
    assert info.value.kind == "timeout"
    _assert_no_key(str(info.value))


async def test_zero_observations_is_no_observations(
    make_client: Callable[..., FredClient],
) -> None:
    client = make_client(Recorder(lambda _r: httpx.Response(200, json=_ok_body())))
    with pytest.raises(FredError) as info:
        await client.observations(SERIES, date(2030, 1, 1), None)
    assert info.value.kind == "no_observations"
    assert SERIES in info.value.message


async def test_missing_placeholder_becomes_nan_in_dataset(
    make_client: Callable[..., FredClient],
) -> None:
    body = _ok_body(_obs("2024-02-01", "."), _obs("2024-01-01", "3.7"), _obs("2024-03-01", "3.9"))
    client = make_client(Recorder(lambda _r: httpx.Response(200, json=body)))
    entry = IndicatorEntry(name="Unemployment Rate", aliases=(), series_id=SERIES)

    observations = await client.observations(SERIES, None, None)
    dataset = build_dataset(observations, entry, None, None, dataset_id="ds-1")

    assert dataset.dates == (date(2024, 1, 1), date(2024, 2, 1), date(2024, 3, 1))
    assert dataset.values[0] == pytest.approx(3.7)
    assert math.isnan(dataset.values[1])
    assert dataset.values[2] == pytest.approx(3.9)


async def test_large_response_keeps_every_observation_in_order(
    make_client: Callable[..., FredClient],
) -> None:
    # ~80 years of monthly data, served newest first: the client must not reorder or drop.
    months = [f"{1945 + i // 12}-{i % 12 + 1:02d}-01" for i in range(960)][::-1]
    body = _ok_body(*(_obs(day, "." if i % 7 == 0 else f"{i}.5") for i, day in enumerate(months)))
    client = make_client(Recorder(lambda _r: httpx.Response(200, json=body)))

    result = await client.observations(SERIES, None, None)

    assert [o["date"] for o in result] == months
    assert [o["value"] for o in result] == [o["value"] for o in body["observations"]]


def test_parse_observations_drops_extra_keys() -> None:
    (obs,) = parse_observations({"observations": [_obs("2024-01-01", "1.5")]})
    assert dict(obs) == {"date": "2024-01-01", "value": "1.5"}


async def test_from_settings_uses_settings_key() -> None:
    recorder = Recorder(lambda _r: httpx.Response(200, json=_ok_body(_obs("2024-01-01", "1"))))
    settings = Settings(
        aws_access_key_id=PLACEHOLDER,
        aws_secret_access_key=PLACEHOLDER,
        aws_session_token=None,
        aws_region="us-east-1",
        fred_api_key=API_KEY,
        bedrock_model_id="placeholder-model",
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(recorder)) as http:
        await FredClient.from_settings(settings, http).observations(SERIES, None, None)
    assert recorder.requests[0].url.params["api_key"] == API_KEY


async def test_rejects_blank_key() -> None:
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _r: httpx.Response(200))
    ) as http:
        with pytest.raises(ValueError):
            FredClient(http, api_key="  ")


@pytest.mark.parametrize("timeout_s", [0, -1, -0.5])
async def test_rejects_non_positive_timeout(timeout_s: float) -> None:
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _r: httpx.Response(200))
    ) as http:
        with pytest.raises(ValueError):
            FredClient(http, api_key=API_KEY, timeout_s=timeout_s)
