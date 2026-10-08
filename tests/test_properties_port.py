"""Property test for ``FRIDAY_PORT`` parsing (design: Correctness Property 26)."""

from typing import Final

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from friday.config import parse_port
from friday.constants import DEFAULT_PORT, MAX_PORT, MIN_PORT
from friday.errors import PortError

_ASCII_DIGITS: Final = "0123456789"

# Characters for which str.isspace() is true, well beyond the ASCII set.
_WHITESPACE: Final = (
    " \t\n\r\x0b\x0c\x1c\x1d\x1e\x1f\x85\u00a0\u1680\u2000\u2003\u2028\u2029\u202f\u205f\u3000"
)

# Python's int() rejects strings longer than this by default (sys.int_info).
_INT_STR_DIGIT_LIMIT: Final = 4300


def _expected(raw: str | None) -> int | None:
    """The design's statement, written independently of ``parse_port``.

    Returns the port to expect, or ``None`` when a ``PortError`` is expected. A
    "decimal integer" is one or more ASCII digits with optional surrounding
    whitespace; signs, underscores, and non-ASCII digits are not accepted.
    """
    if raw is None or raw.isspace() or raw == "":
        return DEFAULT_PORT
    core = raw.strip()
    if any(c not in _ASCII_DIGITS for c in core):
        return None
    significant = core.lstrip("0")
    if len(significant) > len(str(MAX_PORT)):
        return None
    value = int(significant or "0")
    return value if MIN_PORT <= value <= MAX_PORT else None


def _whitespace(max_size: int) -> st.SearchStrategy[str]:
    # Built from a list rather than st.text(sampled_from(...)): Hypothesis 6.168's
    # shrinker crashes ("48 is not in list") on that form when a failure is found.
    return st.lists(st.sampled_from(_WHITESPACE), max_size=max_size).map("".join)


def _padding() -> st.SearchStrategy[str]:
    return _whitespace(3)


@st.composite
def _padded(draw: st.DrawFn, core: str) -> str:
    return draw(_padding()) + core + draw(_padding())


def _blank() -> st.SearchStrategy[str | None]:
    return st.none() | _whitespace(6)


def _in_range() -> st.SearchStrategy[str]:
    zeros = st.integers(0, 3).map(lambda n: "0" * n)
    return st.builds(lambda z, p: f"{z}{p}", zeros, st.integers(MIN_PORT, MAX_PORT))


def _out_of_range() -> st.SearchStrategy[str]:
    small = st.integers(max_value=MIN_PORT - 1)
    large = st.integers(min_value=MAX_PORT + 1)
    return (small | large).map(str)


def _huge_digit_strings() -> st.SearchStrategy[str]:
    """Digit strings past int()'s length limit: huge values, or long zero padding."""
    length = st.integers(_INT_STR_DIGIT_LIMIT - 5, _INT_STR_DIGIT_LIMIT + 200)
    huge = st.builds(lambda d, n: d * n, st.sampled_from("123456789"), length)
    zero_padded = st.builds(lambda n, p: "0" * n + str(p), length, st.integers(MIN_PORT, MAX_PORT))
    return huge | zero_padded


def _signed() -> st.SearchStrategy[str]:
    return st.builds(
        lambda sign, n: f"{sign}{n}", st.sampled_from("+-"), st.integers(0, MAX_PORT + 10)
    )


def _non_ascii_digits() -> st.SearchStrategy[str]:
    """A valid port with some digits swapped for other Unicode decimal digits (e.g. ٨, ８)."""
    other_digit = st.characters(categories=("Nd",)).filter(lambda c: c not in _ASCII_DIGITS)

    @st.composite
    def build(draw: st.DrawFn) -> str:
        chars = list(str(draw(st.integers(MIN_PORT, MAX_PORT))))
        idx = draw(st.lists(st.integers(0, len(chars) - 1), min_size=1, unique=True))
        for i in idx:
            chars[i] = draw(other_digit)
        return "".join(chars)

    return build()


def _numeric_lookalikes() -> st.SearchStrategy[str]:
    """Forms int() or float() might accept but a decimal port is not."""
    port = st.integers(MIN_PORT, MAX_PORT)
    return st.one_of(
        port.map(lambda p: f"{p}.0"),
        port.map(hex),
        port.filter(lambda p: p >= 10).map(lambda p: f"{str(p)[0]}_{str(p)[1:]}"),
        port.map(lambda p: f"{p} {p}"),
        port.map(lambda p: f"{p}e0"),
    )


def port_inputs() -> st.SearchStrategy[str | None]:
    """Blank, valid, and invalid ``FRIDAY_PORT`` values, mixed with arbitrary text."""
    cores = st.one_of(
        _in_range(),
        _out_of_range(),
        _signed(),
        _non_ascii_digits(),
        _numeric_lookalikes(),
        _huge_digit_strings(),
        st.text(max_size=12),
    )
    return st.one_of(_blank(), cores.flatmap(_padded), st.text())


@settings(max_examples=300)
@given(port_inputs())
def test_port_parsing(raw: str | None) -> None:
    """Feature: friday-voice-data-assistant, Property 26: Port parsing.

    **Validates: Requirements 13.3, 13.5**
    """
    expected = _expected(raw)

    if expected is not None:
        assert parse_port(raw) == expected
        return

    assert raw is not None
    with pytest.raises(PortError) as info:
        parse_port(raw)
    assert info.value.kind == "port_invalid"
    assert info.value.value == raw
    assert info.value.cause == "invalid value"
    assert repr(raw) in str(info.value)
    assert "invalid value" in str(info.value)
