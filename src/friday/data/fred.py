"""FRED API adapter: ``FredClient`` implements the ``FredSource`` port (Req 6.2, 6.10, 6.15).

One call requests exactly one series from ``fred/series/observations`` with
``file_type=json`` and the optional ``observation_start``/``observation_end`` bounds, within
:data:`~friday.constants.FRED_TIMEOUT_S`. The ``httpx.AsyncClient`` is owned by the
Container and injected here; this module creates no client and holds no global state.

Every failure becomes a :class:`~friday.errors.FredError` (``http_error``, ``timeout``, or
``no_observations``) whose message names the series ID and, for an error status, the HTTP
status. Messages never contain the API key or a request URL, and they pass through a
:class:`~friday.redact.Redactor` for the key as defense in depth (Req 12.4). httpx
exceptions are not chained into the raised error, because their text and ``request``
carry the full URL including ``api_key``.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping, Sequence
from datetime import date
from types import MappingProxyType
from typing import Final, cast

import httpx

from friday.config import Settings
from friday.constants import FRED_TIMEOUT_S
from friday.data.dates import parse_date
from friday.data.fetch import FRED_MISSING, Observation
from friday.errors import DateRangeError, FredError, FredKind
from friday.redact import Redactor

FRED_OBSERVATIONS_URL: Final = "https://api.stlouisfed.org/fred/series/observations"
"""FRED series observations endpoint (one series per request)."""

_log: Final = logging.getLogger(__name__)


class _MalformedResponse(ValueError):
    """The FRED response body is not the expected observations JSON."""


class FredClient:
    """``FredSource`` backed by the FRED JSON API over an injected ``httpx.AsyncClient``."""

    def __init__(
        self,
        http: httpx.AsyncClient,
        *,
        api_key: str,
        timeout_s: float = FRED_TIMEOUT_S,
    ) -> None:
        """Wrap ``http`` (owned and closed by the Container) with the FRED ``api_key``."""
        if not api_key.strip():
            raise ValueError("api_key must not be blank")
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        self._http = http
        self._api_key = api_key
        self._timeout_s = timeout_s
        self._redactor = Redactor([api_key])

    @classmethod
    def from_settings(cls, settings: Settings, http: httpx.AsyncClient) -> FredClient:
        """Build a client with the FRED API key from explicitly passed ``settings``."""
        return cls(http, api_key=settings.fred_api_key)

    def __repr__(self) -> str:
        """Represent the client without the API key."""
        return f"FredClient(timeout_s={self._timeout_s})"

    async def observations(
        self, series_id: str, start: date | None, end: date | None
    ) -> Sequence[Observation]:
        """Return the ``{"date", "value"}`` observations of ``series_id`` in the range.

        Raises:
            FredError: ``http_error`` for a non-2xx status, a transport failure, or a
                malformed body; ``timeout`` when the request exceeds the timeout;
                ``no_observations`` when the series has no observations in the range.
        """
        response = await self._get(series_id, start, end)
        if not response.is_success:
            raise self._error(
                "http_error",
                f"FRED request for series '{series_id}' failed with HTTP {response.status_code}",
            )
        try:
            observations = parse_observations(response.json())
        except ValueError:  # includes json.JSONDecodeError and _MalformedResponse
            raise self._error(
                "http_error",
                f"FRED returned a malformed response for series '{series_id}' "
                f"(HTTP {response.status_code})",
            ) from None
        if not observations:
            raise self._error(
                "no_observations",
                f"FRED returned no observations for series '{series_id}' in the requested range",
            )
        return observations

    async def _get(self, series_id: str, start: date | None, end: date | None) -> httpx.Response:
        """Send the request, bounding the whole exchange by the timeout."""
        try:
            async with asyncio.timeout(self._timeout_s):
                return await self._http.get(
                    FRED_OBSERVATIONS_URL,
                    params=self._params(series_id, start, end),
                    timeout=httpx.Timeout(self._timeout_s),
                    # Never follow a redirect: the request URL carries the api_key, and a
                    # 3xx is reported as http_error whatever the injected client's policy.
                    follow_redirects=False,
                )
        except (TimeoutError, httpx.TimeoutException):
            raise self._error(
                "timeout",
                f"FRED request for series '{series_id}' timed out after {self._timeout_s:g} s",
            ) from None
        except httpx.HTTPError as exc:
            # The exception text may contain the URL; report only its type.
            raise self._error(
                "http_error",
                f"FRED request for series '{series_id}' failed ({type(exc).__name__})",
            ) from None

    def _params(self, series_id: str, start: date | None, end: date | None) -> dict[str, str]:
        """Query parameters for one series request."""
        params = {"series_id": series_id, "api_key": self._api_key, "file_type": "json"}
        if start is not None:
            params["observation_start"] = start.isoformat()
        if end is not None:
            params["observation_end"] = end.isoformat()
        return params

    def _error(self, kind: FredKind, message: str) -> FredError:
        """Build a redacted ``FredError`` and log it (the message has no key or URL)."""
        safe = self._redactor.redact(message)
        _log.warning("FRED %s: %s", kind, safe)
        return FredError(kind, safe)


def parse_observations(body: object) -> tuple[Observation, ...]:
    """Shape a FRED JSON body into read-only ``{"date", "value"}`` observations.

    Keeps only the ``date`` and ``value`` keys of each entry. Raises ``ValueError`` when the
    body is not an object with an ``observations`` list of entries whose ``date`` is an ISO
    calendar date and whose ``value`` is a number string or FRED's ``.`` placeholder.
    """
    if not isinstance(body, Mapping):
        raise _MalformedResponse("body is not a JSON object")
    raw = cast("Mapping[str, object]", body).get("observations")
    if not isinstance(raw, list):
        raise _MalformedResponse("'observations' is not a list")
    return tuple(_parse_entry(entry) for entry in cast("list[object]", raw))


def _parse_entry(entry: object) -> Observation:
    """Validate one observation entry and keep only its date and value."""
    if not isinstance(entry, Mapping):
        raise _MalformedResponse("observation is not a JSON object")
    fields = cast("Mapping[str, object]", entry)
    day = fields.get("date")
    value = fields.get("value")
    if not isinstance(day, str) or not isinstance(value, str):
        raise _MalformedResponse("observation date or value is not a string")
    try:
        parse_date(day, "date")
    except DateRangeError:
        raise _MalformedResponse("observation date is not YYYY-MM-DD") from None
    if value != FRED_MISSING:
        float(value)  # raises ValueError for a non-numeric value
    return MappingProxyType({"date": day, "value": value})
