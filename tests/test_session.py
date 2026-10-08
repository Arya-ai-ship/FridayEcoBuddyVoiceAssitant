"""Unit tests for ``session.py``: Session, SessionStore, and session ID validation (task 9.2)."""

import asyncio
import uuid
from datetime import date

import numpy as np
import pytest
from agent_framework import AgentSession

from friday.constants import SESSION_MAX_IDLE_S
from friday.data.dataset import Dataset, SessionView
from friday.errors import FridayError, RequestError
from friday.session import ChartRecord, Session, SessionStore, parse_session_id

SID = "0dbe73b6-fd72-489e-9b98-728259f091c8"
OTHER_SID = "6f1c2a3b-4d5e-4f60-8a7b-9c0d1e2f3a4b"


class FakeClock:
    """Manually advanced monotonic clock."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class CountingFactory:
    """``AgentSession`` factory that records every session it creates."""

    def __init__(self) -> None:
        self.created: list[AgentSession] = []

    def __call__(self) -> AgentSession:
        agent_session = AgentSession()
        self.created.append(agent_session)
        return agent_session


def make_dataset(dataset_id: str, values: list[float] | None = None) -> Dataset:
    values = [1.0, 2.0] if values is None else values
    dates = tuple(date(2000, i + 1, 1) for i in range(len(values)))
    return Dataset(
        dataset_id=dataset_id,
        indicator="inflation",
        value_column="inflation",
        series_id="CPIAUCSL",
        dates=dates,
        values=np.array(values, dtype=np.float64),
        transformation="yoy",
    )


def make_chart(chart_id: str) -> ChartRecord:
    return ChartRecord(
        chart_id=chart_id,
        png=b"\x89PNG\r\n\x1a\n",
        dataset_ids=("ds-1",),
        title="Inflation",
        alt_text="Line chart of inflation from 2000-01-01 to 2000-02-01",
        first_date=date(2000, 1, 1),
        last_date=date(2000, 2, 1),
    )


def make_session() -> Session:
    return Session(SID, AgentSession(), last_seen=0.0)


# --- Session -------------------------------------------------------------------


def test_new_session_is_empty() -> None:
    session = make_session()
    assert session.datasets == {}
    assert session.charts == {}
    assert session.fred_cache == {}
    assert session.latest_dataset() is None
    assert not session.lock.locked()


def test_next_id_counts_per_prefix_and_never_repeats() -> None:
    session = make_session()
    ids = [session.next_id("ds"), session.next_id("chart"), session.next_id("ds")]
    assert ids == ["ds-1", "chart-1", "ds-2"]
    assert len({session.next_id("ds") for _ in range(50)}) == 50


def test_datasets_keep_insertion_order_and_latest_is_last_added() -> None:
    session = make_session()
    for _ in range(3):
        session.add_dataset(make_dataset(session.next_id("ds")))
    assert list(session.datasets) == ["ds-1", "ds-2", "ds-3"]
    latest = session.latest_dataset()
    assert latest is not None
    assert latest.dataset_id == "ds-3"


def test_add_dataset_rejects_a_reused_id() -> None:
    session = make_session()
    session.add_dataset(make_dataset("ds-1"))
    with pytest.raises(ValueError, match="ds-1"):
        session.add_dataset(make_dataset("ds-1"))
    assert len(session.datasets) == 1


def test_charts_store_png_and_metadata() -> None:
    session = make_session()
    chart = make_chart(session.next_id("chart"))
    session.add_chart(chart)
    assert session.charts == {"chart-1": chart}
    assert session.charts["chart-1"].png.startswith(b"\x89PNG")
    with pytest.raises(ValueError, match="chart-1"):
        session.add_chart(make_chart("chart-1"))


def test_session_satisfies_session_view() -> None:
    session = make_session()
    session.add_dataset(make_dataset("ds-1", [1.0, float("nan")]))
    view: SessionView = session
    assert list(view.datasets) == ["ds-1"]
    latest = view.latest_dataset()
    assert latest is not None
    assert latest.missing_count == 1


async def test_lock_runs_turns_one_at_a_time() -> None:
    session = make_session()
    order: list[str] = []

    async def turn(name: str) -> None:
        async with session.lock:
            order.append(f"{name}-start")
            await asyncio.sleep(0.01)
            order.append(f"{name}-end")

    await asyncio.gather(turn("a"), turn("b"))
    assert order == ["a-start", "a-end", "b-start", "b-end"]


# --- Session ID validation -----------------------------------------------------


@pytest.mark.parametrize(
    "raw", [SID, SID.upper(), str(uuid.uuid4()), "00000000-0000-0000-0000-000000000000"]
)
def test_parse_session_id_accepts_hyphenated_uuids(raw: str) -> None:
    assert parse_session_id(raw) == raw.lower()


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "not-a-uuid",
        SID.replace("-", ""),
        "{" + SID + "}",
        "urn:uuid:" + SID,
        SID + " ",
        " " + SID[1:],
        SID[:-1] + "g",
        "../../etc/passwd" + "x" * 20,
    ],
)
def test_parse_session_id_rejects_other_strings(raw: str) -> None:
    with pytest.raises(RequestError) as info:
        parse_session_id(raw)
    assert info.value.kind == "invalid_session"
    assert isinstance(info.value, FridayError)
    assert info.value.message == "session ID must be a UUID"  # the raw value is not echoed


# --- SessionStore --------------------------------------------------------------


def test_get_or_create_creates_once_per_id_through_the_factory() -> None:
    factory = CountingFactory()
    store = SessionStore(factory)
    first = store.get_or_create(SID)
    again = store.get_or_create(SID.upper())
    other = store.get_or_create(OTHER_SID)
    assert first is again
    assert first is not other
    assert first.session_id == SID
    assert [first.agent_session, other.agent_session] == factory.created
    assert set(store.sessions) == {SID, OTHER_SID}


def test_get_or_create_rejects_an_invalid_id_without_creating() -> None:
    factory = CountingFactory()
    store = SessionStore(factory)
    with pytest.raises(RequestError, match="UUID"):
        store.get_or_create("abc")
    assert factory.created == []
    assert dict(store.sessions) == {}


def test_end_discards_the_session_and_ignores_unknown_or_invalid_ids() -> None:
    factory = CountingFactory()
    store = SessionStore(factory)
    first = store.get_or_create(SID)
    store.end(OTHER_SID)
    store.end("garbage")
    assert SID in store.sessions
    store.end(SID)
    assert SID not in store.sessions
    replacement = store.get_or_create(SID)
    assert replacement is not first
    assert replacement.datasets == {}
    assert len(factory.created) == 2


def test_evict_idle_drops_only_sessions_idle_past_the_limit() -> None:
    clock = FakeClock()
    store = SessionStore(CountingFactory(), clock=clock)
    store.get_or_create(SID)
    clock.now += 100.0
    store.get_or_create(OTHER_SID)
    clock.now += SESSION_MAX_IDLE_S - 50.0  # SID idle for limit + 50 s, OTHER for limit - 50 s
    store.evict_idle()
    assert set(store.sessions) == {OTHER_SID}


def test_evict_idle_keeps_a_session_at_exactly_the_limit_and_refreshes_on_use() -> None:
    clock = FakeClock()
    store = SessionStore(CountingFactory(), clock=clock)
    store.get_or_create(SID)
    clock.now += 10.0
    store.evict_idle(max_idle_s=10.0)
    assert SID in store.sessions
    store.get_or_create(SID)  # refreshes last_seen
    clock.now += 10.0
    store.evict_idle(max_idle_s=10.0)
    assert SID in store.sessions
    clock.now += 0.5
    store.evict_idle(max_idle_s=10.0)
    assert SID not in store.sessions


async def test_evict_idle_keeps_a_session_with_a_running_turn() -> None:
    clock = FakeClock()
    store = SessionStore(CountingFactory(), clock=clock)
    session = store.get_or_create(SID)
    async with session.lock:
        clock.now += SESSION_MAX_IDLE_S + 1.0
        store.evict_idle()
        assert SID in store.sessions
    store.evict_idle()
    assert SID not in store.sessions
