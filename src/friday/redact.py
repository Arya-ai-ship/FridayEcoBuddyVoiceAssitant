"""Credential redaction for every string that leaves the process (Req 5.13, 12.4).

The :class:`Redactor` masks Credentials "in full or in part": any fragment of
``min_fragment`` (default :data:`~friday.constants.REDACT_MIN_FRAGMENT`) or more
characters of a configured secret is replaced, together with the rest of the
maximal span it belongs to, by :data:`REDACTED`. As defense in depth it also masks
AWS access-key-ID shapes and ``api_key=`` query values, even when they are not
among the configured secrets.

:class:`RedactingFilter` applies a Redactor to log records. Attach it to the log
*handler* (not a logger) so records from every child logger pass through it.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from typing import Final

from friday.constants import REDACT_MIN_FRAGMENT

REDACTED: Final = "[REDACTED]"
"""Replacement text for every masked span."""

_AWS_KEY_ID: Final = re.compile(r"(?:AKIA|ASIA)[A-Z0-9]{16}")
"""AWS access key ID shape (long-term ``AKIA`` and temporary ``ASIA`` keys)."""

_API_KEY_PARAM: Final = re.compile(r"(api_key=)[^&\s\"'#<>]+", re.IGNORECASE)
"""An ``api_key`` query parameter value, as used by the FRED API."""


class Redactor:
    """Masks configured Credential values and well-known secret shapes in text."""

    def __init__(self, secrets: Iterable[str], min_fragment: int = REDACT_MIN_FRAGMENT) -> None:
        """Index every ``min_fragment``-character window of each non-blank secret.

        Secrets shorter than ``min_fragment`` are masked only where they occur in full.
        """
        if min_fragment < 1:
            raise ValueError("min_fragment must be at least 1")
        self._k = min_fragment
        windows: set[str] = set()
        short: set[str] = set()
        for secret in secrets:
            if not secret.strip():
                continue
            if len(secret) < min_fragment:
                short.add(secret)
                continue
            windows.update(
                secret[i : i + min_fragment] for i in range(len(secret) - min_fragment + 1)
            )
        self._windows: frozenset[str] = frozenset(windows)
        # Longest first so a short secret that contains another is masked whole.
        self._short: tuple[str, ...] = tuple(sorted(short, key=len, reverse=True))

    def redact(self, text: str) -> str:
        """Return ``text`` with every Credential fragment and secret shape masked.

        Text that contains no Credential fragment and no secret shape is returned unchanged.
        """
        text = self._mask_fragments(text)
        text = _AWS_KEY_ID.sub(REDACTED, text)
        return _API_KEY_PARAM.sub(lambda m: m.group(1) + REDACTED, text)

    def _mask_fragments(self, text: str) -> str:
        """Replace each maximal span covered by secret windows with :data:`REDACTED`."""
        k = self._k
        covered = [False] * len(text)
        if self._windows and len(text) >= k:
            for i in range(len(text) - k + 1):
                if text[i : i + k] in self._windows:
                    covered[i : i + k] = [True] * k
        for secret in self._short:
            start = text.find(secret)
            while start != -1:
                covered[start : start + len(secret)] = [True] * len(secret)
                start = text.find(secret, start + 1)
        if not any(covered):
            return text
        parts: list[str] = []
        i = 0
        while i < len(text):
            if covered[i]:
                while i < len(text) and covered[i]:
                    i += 1
                parts.append(REDACTED)
            else:
                j = i
                while j < len(text) and not covered[j]:
                    j += 1
                parts.append(text[i:j])
                i = j
        return "".join(parts)


class RedactingFilter(logging.Filter):
    """Logging filter that redacts the rendered message, traceback, and stack info.

    The message is rendered with its arguments first (``record.getMessage()``), then
    redacted into ``record.msg`` with ``record.args`` cleared, so a secret passed as a
    format argument is masked too. The filter never drops a record.
    """

    def __init__(self, redactor: Redactor) -> None:
        """Wrap ``redactor``; the filter applies to records of every logger name."""
        super().__init__()
        self._redactor = redactor

    def filter(self, record: logging.LogRecord) -> bool:
        """Rewrite ``record`` in place with redacted text and keep it."""
        try:
            message = record.getMessage()
        except (TypeError, ValueError):
            # Mismatched format arguments: redact the parts separately instead of failing.
            message = f"{record.msg} {record.args!r}"
        record.msg = self._redactor.redact(message)
        record.args = ()
        if record.exc_info and not record.exc_text:
            record.exc_text = logging.Formatter().formatException(record.exc_info)
        if record.exc_text:
            record.exc_text = self._redactor.redact(record.exc_text)
        if record.stack_info:
            record.stack_info = self._redactor.redact(record.stack_info)
        return True
