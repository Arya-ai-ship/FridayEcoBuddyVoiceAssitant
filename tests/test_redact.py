"""Unit tests for the Redactor and RedactingFilter (Req 5.13, 12.4).

Every secret below is an obviously fake placeholder.
"""

import io
import logging

import pytest

from friday.redact import REDACTED, RedactingFilter, Redactor

FAKE_SECRET = "FAKEsecretVALUEforTESTSonly0123456789xyz"  # noqa: S105 - fake, 40 chars
FAKE_FRED_KEY = "fakefredkey000000000000000000000"  # 32 chars
FAKE_KEY_ID = "AKIAFAKEFAKEFAKEFAKE"  # matches the AWS key-ID shape


@pytest.fixture
def redactor() -> Redactor:
    return Redactor([FAKE_SECRET, FAKE_FRED_KEY, "", "   "])


def test_full_secret_is_masked(redactor: Redactor) -> None:
    out = redactor.redact(f"error: bad secret {FAKE_SECRET} rejected")
    assert out == f"error: bad secret {REDACTED} rejected"


def test_partial_fragment_of_eight_chars_is_masked(redactor: Redactor) -> None:
    fragment = FAKE_SECRET[10:18]
    out = redactor.redact(f"token prefix={fragment}!")
    assert fragment not in out
    assert out == f"token prefix={REDACTED}!"


def test_fragment_shorter_than_minimum_is_kept(redactor: Redactor) -> None:
    fragment = FAKE_SECRET[10:17]  # 7 chars
    assert redactor.redact(f"x {fragment} y") == f"x {fragment} y"


def test_maximal_span_is_masked_as_one(redactor: Redactor) -> None:
    # Secret fragment glued to its own extension: the whole covered span becomes one marker.
    text = "a" + FAKE_SECRET[:12] + FAKE_FRED_KEY[:9] + "b"
    assert redactor.redact(text) == f"a{REDACTED}b"


def test_text_without_secrets_is_unchanged(redactor: Redactor) -> None:
    text = "Fetching inflation data, Boss. 937 rows from 1948-01-01."
    assert redactor.redact(text) == text
    assert redactor.redact("") == ""


def test_aws_key_id_shape_is_masked_without_configuration() -> None:
    out = Redactor([]).redact(f"UnrecognizedClientException for {FAKE_KEY_ID}.")
    assert FAKE_KEY_ID not in out
    assert out == f"UnrecognizedClientException for {REDACTED}."


def test_api_key_query_value_is_masked() -> None:
    url = "https://api.example.invalid/fred/series/observations?series_id=UNRATE&api_key=abc123&x=1"
    out = Redactor([]).redact(url)
    assert out.endswith(f"series_id=UNRATE&api_key={REDACTED}&x=1")
    assert "abc123" not in out


def test_short_secret_is_masked_only_in_full() -> None:
    r = Redactor(["shrt"])
    assert r.redact("a shrt b") == f"a {REDACTED} b"
    assert r.redact("a shr b") == "a shr b"


def test_invalid_min_fragment_rejected() -> None:
    with pytest.raises(ValueError, match="min_fragment"):
        Redactor([FAKE_SECRET], min_fragment=0)


def _logger_with_filter(redactor: Redactor) -> tuple[logging.Logger, io.StringIO]:
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.addFilter(RedactingFilter(redactor))
    logger = logging.getLogger(f"friday.test_redact.{id(stream)}")
    logger.handlers = [handler]
    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    return logger, stream


def test_filter_redacts_message_and_args(redactor: Redactor) -> None:
    logger, stream = _logger_with_filter(redactor)
    logger.info("calling FRED with key %s and %s", FAKE_FRED_KEY, FAKE_KEY_ID)
    output = stream.getvalue()
    assert FAKE_FRED_KEY[:8] not in output
    assert FAKE_KEY_ID not in output
    assert output.count(REDACTED) == 2


def test_filter_redacts_child_logger_records_on_handler(redactor: Redactor) -> None:
    logger, stream = _logger_with_filter(redactor)
    child = logger.getChild("child")
    child.warning("leak %s", FAKE_SECRET)
    assert FAKE_SECRET[:8] not in stream.getvalue()
    assert REDACTED in stream.getvalue()


def test_filter_redacts_traceback(redactor: Redactor) -> None:
    logger, stream = _logger_with_filter(redactor)
    try:
        raise RuntimeError(f"signature mismatch for {FAKE_SECRET}")
    except RuntimeError:
        logger.exception("request failed")
    output = stream.getvalue()
    assert "RuntimeError" in output
    assert FAKE_SECRET[:8] not in output
    assert REDACTED in output


def test_filter_keeps_record_with_mismatched_args(redactor: Redactor) -> None:
    record = logging.LogRecord("x", logging.INFO, __file__, 1, "%s %s", (FAKE_SECRET,), None)
    assert RedactingFilter(redactor).filter(record) is True
    assert FAKE_SECRET[:8] not in record.getMessage()
