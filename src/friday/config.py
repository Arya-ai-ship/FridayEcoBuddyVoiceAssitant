"""Startup configuration: environment merging, Settings, and port parsing.

``read_environment`` is the only function in the package that reads the process
environment or the ``.env`` file. It never mutates ``os.environ``. Everything else
here is pure, and ``Settings`` is passed explicitly to the rest of the Backend.

Importing this module has no side effects, so any layer may import ``Settings``.
"""

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

from dotenv import dotenv_values

from friday.constants import DEFAULT_PORT, MAX_PORT, MIN_PORT
from friday.errors import ConfigError, PortError

# --- Variable names (Req 12.1) ---------------------------------------------
REQUIRED: Final[tuple[str, ...]] = (
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_REGION",
    "FRED_API_KEY",
    "BEDROCK_MODEL_ID",
)
"""Required settings; any that are unset or blank stop startup (Req 12.6)."""

OPTIONAL: Final[tuple[str, ...]] = (
    "AWS_SESSION_TOKEN",
    "POLLY_VOICE_ID",
    "POLLY_ENGINE",
    "TRANSCRIBE_LANGUAGE_CODE",
    "INDICATOR_MAP_PATH",
    "FRIDAY_PORT",
)
"""Optional settings; blank or unset means the documented default (Req 12.3)."""

ALL_VARS: Final[tuple[str, ...]] = REQUIRED + OPTIONAL

# --- Defaults for optional settings (mirrors .env.example) ------------------
DEFAULT_POLLY_VOICE_ID: Final = "Joanna"
DEFAULT_POLLY_ENGINE: Final = "neural"
DEFAULT_TRANSCRIBE_LANGUAGE_CODE: Final = "en-US"

_PORT_PATTERN: Final = re.compile(r"[0-9]+")


@dataclass(frozen=True)
class Settings:
    """Resolved Backend configuration. Credential fields are hidden from ``repr``."""

    aws_access_key_id: str = field(repr=False)
    aws_secret_access_key: str = field(repr=False)
    aws_session_token: str | None = field(repr=False)
    aws_region: str
    fred_api_key: str = field(repr=False)
    bedrock_model_id: str
    polly_voice_id: str = DEFAULT_POLLY_VOICE_ID
    polly_engine: str = DEFAULT_POLLY_ENGINE
    transcribe_language_code: str = DEFAULT_TRANSCRIBE_LANGUAGE_CODE
    indicator_map_path: str | None = None
    port: int = DEFAULT_PORT


def _clean(value: str | None) -> str | None:
    """Strip a raw value; empty or whitespace-only counts as unset."""
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def merge_environment(
    dotenv: Mapping[str, str | None], process: Mapping[str, str]
) -> dict[str, str]:
    """Overlay the process environment on ``.env`` values for the known settings.

    A process value wins when it is non-blank; otherwise the ``.env`` value is kept
    (Req 12.2). Only the variables in ``ALL_VARS`` are returned.
    """
    merged: dict[str, str] = {}
    for name in ALL_VARS:
        process_value = process.get(name)
        dotenv_value = dotenv.get(name)
        if process_value is not None and _clean(process_value) is not None:
            merged[name] = process_value
        elif dotenv_value is not None:
            merged[name] = dotenv_value
        elif process_value is not None:
            merged[name] = process_value
    return merged


def read_environment(dotenv_path: Path, environ: Mapping[str, str] | None = None) -> dict[str, str]:
    """Read ``.env`` (if present) overlaid with the process environment.

    A missing ``.env`` file is allowed (Req 12.10). ``os.environ`` is never mutated.
    ``environ`` defaults to ``os.environ`` and exists so callers can inject a mapping.
    """
    process: Mapping[str, str] = os.environ if environ is None else environ
    dotenv: dict[str, str | None] = (
        dict(dotenv_values(dotenv_path, interpolate=False)) if dotenv_path.is_file() else {}
    )
    return merge_environment(dotenv, process)


def parse_port(raw: str | None) -> int:
    """Parse ``FRIDAY_PORT``: blank or unset means 8000; else an integer 1..65535.

    Raises ``PortError(value, "invalid value")`` otherwise (Req 13.3, 13.5).
    """
    cleaned = _clean(raw)
    if cleaned is None:
        return DEFAULT_PORT
    if _PORT_PATTERN.fullmatch(cleaned) is None:
        raise PortError(str(raw), "invalid value")
    # Drop leading zeros and bound the length first: int() rejects >4300-digit strings.
    significant = cleaned.lstrip("0")
    if len(significant) > len(str(MAX_PORT)):
        raise PortError(str(raw), "invalid value")
    port = int(significant or "0")
    if not MIN_PORT <= port <= MAX_PORT:
        raise PortError(str(raw), "invalid value")
    return port


def load_settings(env: Mapping[str, str]) -> Settings:
    """Build ``Settings`` from a merged environment mapping (pure).

    Every missing or blank required variable is reported in one ``ConfigError``
    that names them and contains no Credential value (Req 12.6). Optional settings
    fall back to their defaults (Req 12.3). An invalid port raises ``PortError``.
    """
    values = {name: _clean(env.get(name)) for name in ALL_VARS}
    required = {name: value for name in REQUIRED if (value := values[name]) is not None}
    missing = [name for name in REQUIRED if name not in required]
    if missing:
        raise ConfigError.missing_vars(missing)

    return Settings(
        aws_access_key_id=required["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=required["AWS_SECRET_ACCESS_KEY"],
        aws_session_token=values["AWS_SESSION_TOKEN"],
        aws_region=required["AWS_REGION"],
        fred_api_key=required["FRED_API_KEY"],
        bedrock_model_id=required["BEDROCK_MODEL_ID"],
        polly_voice_id=values["POLLY_VOICE_ID"] or DEFAULT_POLLY_VOICE_ID,
        polly_engine=values["POLLY_ENGINE"] or DEFAULT_POLLY_ENGINE,
        transcribe_language_code=values["TRANSCRIBE_LANGUAGE_CODE"]
        or DEFAULT_TRANSCRIBE_LANGUAGE_CODE,
        indicator_map_path=values["INDICATOR_MAP_PATH"],
        port=parse_port(env.get("FRIDAY_PORT")),
    )


def secret_values(settings: Settings) -> list[str]:
    """Return every Credential value, for the Redactor (Req 5.13, 12.4)."""
    secrets = [
        settings.aws_access_key_id,
        settings.aws_secret_access_key,
        settings.aws_session_token,
        settings.fred_api_key,
    ]
    return [s for s in secrets if s]
