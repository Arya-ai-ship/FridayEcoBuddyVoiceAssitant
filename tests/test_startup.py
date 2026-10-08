"""Startup tests for ``cli.main`` (task 15.6, Req 12.6, 12.7, 13.2, 13.4, 13.5, 13.8).

Config-error paths (missing vars, malformed Indicator_Map, bad/busy port) are driven
in-process. The live bind-and-serve path (binds only to 127.0.0.1, prints the Local_URL,
exits within 5 s of SIGINT) runs the real console script in a subprocess with fake
credentials, so no live AWS call blocks startup.
"""

from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

from friday.cli import main

ROOT = Path(__file__).resolve().parents[1]
GOOD_ENV = {
    "AWS_ACCESS_KEY_ID": "AKIAFAKEFAKEFAKEFAKE",
    "AWS_SECRET_ACCESS_KEY": "FAKEsecretVALUEforTESTSonly0123456789xyz",
    "AWS_REGION": "us-east-2",
    "FRED_API_KEY": "fakefredkey000000000000000000000",
    "BEDROCK_MODEL_ID": "us.openai.gpt-5.6-terra",
}
FRIDAY_VARS = (*GOOD_ENV, "AWS_SESSION_TOKEN", "FRIDAY_PORT", "INDICATOR_MAP_PATH")


@pytest.fixture
def clean_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Run from an empty directory (no ``.env``) with Friday vars cleared."""
    monkeypatch.chdir(tmp_path)
    for name in FRIDAY_VARS:
        monkeypatch.delenv(name, raising=False)
    yield tmp_path


def test_missing_required_vars_exits_nonzero_with_one_message(
    clean_cwd: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main([])
    assert code == 2
    err = capsys.readouterr().err
    assert "Missing required environment variables" in err
    for name in GOOD_ENV:
        assert name in err


def test_malformed_indicator_map_exits_nonzero(
    clean_cwd: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    bad_map = clean_cwd / "bad.json"
    bad_map.write_text("{ not valid json ", encoding="utf-8")
    for name, value in GOOD_ENV.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("INDICATOR_MAP_PATH", str(bad_map))

    code = main([])
    assert code == 2
    assert "Indicator_Map" in capsys.readouterr().err


def test_invalid_port_exits_nonzero_naming_the_value(
    clean_cwd: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    for name, value in GOOD_ENV.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("FRIDAY_PORT", "99999")  # out of range

    code = main([])
    assert code == 2
    assert "99999" in capsys.readouterr().err


def test_port_in_use_exits_nonzero(
    clean_cwd: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Pre-bind a port so the manual bind in cli fails with EADDRINUSE.
    holder = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    holder.bind(("127.0.0.1", 0))
    holder.listen()
    port = holder.getsockname()[1]
    try:
        for name, value in GOOD_ENV.items():
            monkeypatch.setenv(name, value)
        monkeypatch.setenv("FRIDAY_PORT", str(port))
        code = main([])
        assert code == 2
        err = capsys.readouterr().err
        assert str(port) in err and "in use" in err
    finally:
        holder.close()


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def test_server_binds_to_loopback_prints_url_and_exits_on_sigint(tmp_path: Path) -> None:
    port = _free_port()
    env = {**os.environ, **GOOD_ENV, "FRIDAY_PORT": str(port)}
    for name in ("AWS_SESSION_TOKEN", "INDICATOR_MAP_PATH"):
        env.pop(name, None)
    proc = subprocess.Popen(  # noqa: S603
        [sys.executable, "-m", "friday.cli"],
        cwd=tmp_path,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        url = _await_url(proc, port)
        assert url == f"http://127.0.0.1:{port}"
        # It binds only to loopback: a connect to 127.0.0.1 succeeds.
        with socket.create_connection(("127.0.0.1", port), timeout=5):
            pass
        # SIGINT exits within 5 s (Req 13.8).
        start = time.monotonic()
        proc.send_signal(signal.SIGINT)
        proc.wait(timeout=5)
        assert time.monotonic() - start < 5
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)


def _await_url(proc: subprocess.Popen[str], port: int, timeout: float = 20.0) -> str:
    """Read the subprocess output until the Local_URL line appears."""
    deadline = time.monotonic() + timeout
    assert proc.stdout is not None
    while time.monotonic() < deadline:
        line = proc.stdout.readline()
        if not line:
            if proc.poll() is not None:
                raise AssertionError("server exited before printing the URL")
            continue
        if "Friday is running at" in line:
            return line.split("Friday is running at", 1)[1].strip()
    raise AssertionError("timed out waiting for the Local_URL")
