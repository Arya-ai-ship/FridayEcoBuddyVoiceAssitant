"""Per-browser Sessions and the in-memory SessionStore (Req 5.1, 6.8, 8.4).

A ``Session`` holds everything one browser tab accumulates: the MAF ``AgentSession`` (the
conversation history the harness persists), the Datasets and charts the Tools produced,
and the per-Session FRED cache. Turns within a Session run one at a time under its
``asyncio.Lock``.

The ``SessionStore`` creates each ``AgentSession`` through an injected factory (in
production ``harness_agent.create_session``, built in ``wiring.py``) and takes an injectable
clock for idle eviction, so it holds no global state.
"""

import asyncio
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date

from agent_framework import AgentSession

from friday.constants import SESSION_MAX_IDLE_S
from friday.data.dataset import Dataset
from friday.data.fetch import Observation
from friday.errors import RequestError

FredCacheKey = tuple[str, date | None, date | None]
"""FRED cache key: ``(series_id, start, end)``; ``None`` is an open side (Req 6.15)."""

Clock = Callable[[], float]
"""Monotonic clock in seconds, injectable for idle-eviction tests."""


@dataclass(frozen=True, eq=False)
class ChartRecord:
    """One rendered chart: the PNG bytes plus what the Tool result and event report."""

    chart_id: str
    png: bytes
    dataset_ids: tuple[str, ...]
    title: str
    alt_text: str
    first_date: date
    last_date: date


@dataclass(eq=False)
class Session:
    """State of one browser Session. Datasets and charts are only ever added, never removed.

    ``datasets`` and ``charts`` are insertion-ordered, so the last entry is the most recent.
    IDs come from ``next_id`` and are never reused within the Session (Req 6.8).
    """

    session_id: str
    agent_session: AgentSession
    last_seen: float
    datasets: dict[str, Dataset] = field(default_factory=dict[str, Dataset])
    charts: dict[str, ChartRecord] = field(default_factory=dict[str, ChartRecord])
    fred_cache: dict[FredCacheKey, Sequence[Observation]] = field(
        default_factory=dict[FredCacheKey, Sequence[Observation]]
    )
    counters: dict[str, int] = field(default_factory=dict[str, int])
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def next_id(self, prefix: str) -> str:
        """Return a new ``"{prefix}-{n}"`` ID; ``n`` counts up from 1 per prefix."""
        count = self.counters.get(prefix, 0) + 1
        self.counters[prefix] = count
        return f"{prefix}-{count}"

    def add_dataset(self, dataset: Dataset) -> None:
        """Store ``dataset`` as the most recent Dataset under its own ``dataset_id``.

        Raises:
            ValueError: a Dataset with that ID is already stored.
        """
        if dataset.dataset_id in self.datasets:
            raise ValueError(f"dataset ID {dataset.dataset_id!r} is already in use")
        self.datasets[dataset.dataset_id] = dataset

    def add_chart(self, chart: ChartRecord) -> None:
        """Store ``chart`` under its own ``chart_id``.

        Raises:
            ValueError: a chart with that ID is already stored.
        """
        if chart.chart_id in self.charts:
            raise ValueError(f"chart ID {chart.chart_id!r} is already in use")
        self.charts[chart.chart_id] = chart

    def latest_dataset(self) -> Dataset | None:
        """The most recently inserted Dataset, or ``None`` when there is none (Req 8.4)."""
        return next(reversed(self.datasets.values()), None)


def parse_session_id(raw: str) -> str:
    """Return the canonical (lowercase, hyphenated) form of a UUID session ID.

    Only the 36-character hyphenated form that ``crypto.randomUUID()`` produces is
    accepted, in either case.

    Raises:
        RequestError: ``invalid_session`` when ``raw`` is not such a UUID string.
    """
    if len(raw) != 36:
        raise RequestError.invalid_session()
    try:
        canonical = str(uuid.UUID(raw))
    except ValueError:
        raise RequestError.invalid_session() from None
    if canonical != raw.lower():
        raise RequestError.invalid_session()
    return canonical


class SessionStore:
    """In-memory Sessions keyed by canonical UUID session ID."""

    def __init__(
        self, new_agent_session: Callable[[], AgentSession], *, clock: Clock = time.monotonic
    ) -> None:
        """Create an empty store.

        Args:
            new_agent_session: factory for each Session's MAF ``AgentSession``
                (``harness_agent.create_session`` in production).
            clock: monotonic time source in seconds, used for idle eviction.
        """
        self._new_agent_session = new_agent_session
        self._clock = clock
        self._sessions: dict[str, Session] = {}

    @property
    def sessions(self) -> Mapping[str, Session]:
        """Read-only view of the live Sessions by session ID."""
        return self._sessions

    def get_or_create(self, session_id: str) -> Session:
        """Return the Session for ``session_id``, creating it if needed, and mark it used.

        Raises:
            RequestError: ``invalid_session`` when ``session_id`` is not a UUID string.
        """
        key = parse_session_id(session_id)
        now = self._clock()
        session = self._sessions.get(key)
        if session is None:
            session = Session(key, self._new_agent_session(), last_seen=now)
            self._sessions[key] = session
        session.last_seen = now
        return session

    def end(self, session_id: str) -> None:
        """Discard a Session. Unknown or malformed IDs are ignored (beacons are best effort)."""
        try:
            key = parse_session_id(session_id)
        except RequestError:
            return
        self._sessions.pop(key, None)

    def evict_idle(self, max_idle_s: float = SESSION_MAX_IDLE_S) -> None:
        """Discard Sessions unused for more than ``max_idle_s`` seconds.

        A Session whose lock is held (a turn is running) is kept.
        """
        now = self._clock()
        idle = [
            key
            for key, session in self._sessions.items()
            if now - session.last_seen > max_idle_s and not session.lock.locked()
        ]
        for key in idle:
            del self._sessions[key]
