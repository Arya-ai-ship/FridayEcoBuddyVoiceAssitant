"""Property test for Settings merge and required-variable reporting (design: Property 25).

Every value is a Hypothesis-generated fake. The ``.env`` file is written to a fresh
``tempfile`` directory per example; the project's real ``.env`` is never read.
"""

import re
import string
import tempfile
from pathlib import Path
from typing import Final

import pytest
from hypothesis import event, given, settings
from hypothesis import strategies as st

from friday.config import ALL_VARS, REQUIRED, load_settings, merge_environment, read_environment
from friday.errors import ConfigError

DotenvMap = dict[str, str | None]
ProcessMap = dict[str, str]

CREDENTIALS: Final = (
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "FRED_API_KEY",
)

# Characters that survive a single-quoted ``.env`` value unchanged (no quotes,
# backslashes, ``$``, ``#``, or newlines), so the same value can sit in both mappings.
_SAFE: Final = string.ascii_letters + string.digits + " -_./:+=@,;!?%&()[]{}<>~^|*"
_FILE_BLANK: Final = " \t"
# Unicode whitespace that str.strip() removes; process values are not file-bound.
_PROCESS_BLANK: Final = " \t\n\r\x0b\x0c\u00a0\u2003\u3000"
_NOISE: Final = ("PATH", "HOME", "FRIDAY_UNRELATED")


def _resolved(value: str | None) -> str | None:
    """Oracle: a value counts as set only when it has a non-whitespace character."""
    if value is None or value.strip() == "":
        return None
    return value.strip()


def _cores(name: str) -> st.SearchStrategy[str]:
    """Non-blank values for ``name``.

    Credentials always contain a digit, so a leak can't hide behind a coincidental match
    with the (digit-free) variable names and wording of the error message.
    """
    if name == "FRIDAY_PORT":
        return st.integers(1, 65535).map(str)  # keep load_settings off the port error path
    if name in CREDENTIALS:
        return st.tuples(
            st.text(_SAFE, max_size=12),
            st.sampled_from(string.digits),
            st.text(_SAFE, min_size=7, max_size=24),
        ).map("".join)
    return st.text(_SAFE, min_size=1, max_size=20).filter(lambda s: s.strip() != "")


def _padded(core: st.SearchStrategy[str], blank: str) -> st.SearchStrategy[str]:
    pad = st.text(st.sampled_from(blank), max_size=3)
    return st.tuples(pad, core, pad).map("".join)


@st.composite
def environments(draw: st.DrawFn) -> tuple[DotenvMap, ProcessMap]:
    """A ``.env`` mapping and a process environment over every setting plus noise.

    Each setting is independently absent, blank (empty or whitespace-only), or set
    (padded with whitespace) in each source. The ``.env`` side can also hold a bare
    ``NAME`` line, which python-dotenv reads as ``None``.
    """
    dotenv: DotenvMap = {}
    process: ProcessMap = {}
    for name in (*ALL_VARS, *_NOISE):
        core = _cores(name)
        # "set" is listed twice so that all five required settings resolve often enough
        # (roughly 1 example in 6) to exercise the success path as well.
        file_kind = draw(st.sampled_from(("absent", "blank", "set", "set", "bare")))
        if file_kind == "bare":
            dotenv[name] = None
        elif file_kind == "blank":
            dotenv[name] = draw(st.text(st.sampled_from(_FILE_BLANK), max_size=3))
        elif file_kind == "set":
            dotenv[name] = draw(_padded(core, _FILE_BLANK))
        proc_kind = draw(st.sampled_from(("absent", "blank", "set", "set")))
        if proc_kind == "blank":
            process[name] = draw(st.text(st.sampled_from(_PROCESS_BLANK), max_size=3))
        elif proc_kind == "set":
            process[name] = draw(_padded(core, _PROCESS_BLANK))
    return dotenv, process


def _dotenv_text(dotenv: DotenvMap) -> str:
    lines = [name if value is None else f"{name}='{value}'" for name, value in dotenv.items()]
    return "\n".join(lines) + "\n"


@settings(max_examples=200, deadline=None)
@given(environments())
def test_settings_merge_and_required_variable_reporting(
    case: tuple[DotenvMap, ProcessMap],
) -> None:
    """Feature: friday-voice-data-assistant, Property 25: Settings merge and
    required-variable reporting.

    **Validates: Requirements 12.2, 12.6**
    """
    dotenv, process = case
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / ".env"
        path.write_text(_dotenv_text(dotenv), encoding="utf-8")
        merged = read_environment(path, environ=process)
    assert merged == merge_environment(dotenv, process)

    # Process value wins when non-blank; otherwise the .env value is used.
    expected: dict[str, str | None] = {}
    for name in ALL_VARS:
        proc_value, file_value = process.get(name), dotenv.get(name)
        if _resolved(proc_value) is not None:
            assert merged[name] == proc_value
            expected[name] = _resolved(proc_value)
        elif file_value is not None:
            assert merged[name] == file_value
            expected[name] = _resolved(file_value)
        else:
            assert _resolved(merged.get(name)) is None
            expected[name] = None

    missing = {name for name in REQUIRED if expected[name] is None}
    event("all required set" if not missing else "some required missing")
    if not missing:
        resolved = load_settings(merged)
        for name in REQUIRED:
            assert getattr(resolved, name.lower()) == expected[name]
        return

    with pytest.raises(ConfigError) as info:
        load_settings(merged)
    err = info.value
    assert err.kind == "config_missing"
    assert len(err.missing) == len(missing)
    assert set(err.missing) == missing
    named = set(re.findall(r"[A-Z][A-Z0-9_]*", err.message)) & set(ALL_VARS)
    assert named == missing

    secrets = [
        cleaned
        for name in CREDENTIALS
        for raw in (process.get(name), dotenv.get(name))
        if (cleaned := _resolved(raw)) is not None
    ]
    for secret in secrets:
        assert secret not in err.message
        assert secret not in str(err)
