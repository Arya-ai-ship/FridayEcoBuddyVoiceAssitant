"""Shared pytest fixtures.

The autouse ``network_guard`` fails any socket connection to a non-loopback address during
offline tests, catching unexpected egress from MAF telemetry, boto3, or httpx. Tests under
``tests/live/`` (opt-in, real services) are exempt.

pytest's default ``prepend`` import mode puts ``tests/`` on ``sys.path``, so test modules
import helpers as ``from strategies import ...`` and ``from fakes import ...``.
"""

import ipaddress
import socket
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

LIVE_DIR = Path(__file__).parent / "live"


class NetworkBlockedError(RuntimeError):
    """A non-loopback connection was attempted in an offline test.

    Not an ``OSError``, so client libraries cannot mistake it for a retryable network error.
    """


def is_loopback(address: Any) -> bool:
    """True for AF_UNIX paths and IPv4/IPv6 loopback addresses (or ``localhost``)."""
    if isinstance(address, str | bytes):
        return True  # AF_UNIX path (e.g. the event loop's self-pipe)
    host = str(address[0]).split("%", 1)[0]
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False  # a hostname other than localhost would resolve off-host


class NetworkGuard:
    """Records blocked connection attempts; tests may inspect and clear ``attempts``."""

    def __init__(self) -> None:
        self.attempts: list[Any] = []

    def check(self, address: Any) -> None:
        """Raise ``NetworkBlockedError`` for a non-loopback ``address``."""
        if not is_loopback(address):
            self.attempts.append(address)
            raise NetworkBlockedError(f"blocked non-loopback connection to {address!r}")


@pytest.fixture(autouse=True)
def network_guard(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> Iterator[NetworkGuard]:
    """Fail any non-loopback ``socket.connect``/``connect_ex`` (skipped for ``tests/live/``)."""
    guard = NetworkGuard()
    if LIVE_DIR in request.path.parents:
        yield guard
        return

    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex

    def connect(self: socket.socket, address: Any) -> None:
        guard.check(address)
        real_connect(self, address)

    def connect_ex(self: socket.socket, address: Any) -> int:
        guard.check(address)
        return real_connect_ex(self, address)

    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(socket.socket, "connect_ex", connect_ex)
    yield guard
    if guard.attempts:
        # Fails the test even if a library swallowed the NetworkBlockedError.
        pytest.fail(f"non-loopback connections attempted: {guard.attempts!r}")
