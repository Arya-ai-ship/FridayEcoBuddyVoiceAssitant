"""Unit tests for friday.config (tasks 2.1 and 2.6; Req 12.2, 12.3, 12.6, 12.10, 13.3).

Every value here is an obvious fake. ``.env`` files are written under ``tmp_path``
and the process environment is injected as a mapping, so the real ``.env`` and the
developer's shell environment are never read.
"""

import os
from pathlib import Path

import pytest
from dotenv import dotenv_values

from friday.config import (
    DEFAULT_POLLY_ENGINE,
    DEFAULT_POLLY_VOICE_ID,
    DEFAULT_TRANSCRIBE_LANGUAGE_CODE,
    OPTIONAL,
    REQUIRED,
    Settings,
    load_settings,
    merge_environment,
    parse_port,
    read_environment,
    secret_values,
)
from friday.constants import DEFAULT_PORT
from friday.errors import ConfigError, PortError

ENV_EXAMPLE = Path(__file__).resolve().parent.parent / ".env.example"

FAKE_REQUIRED = {
    "AWS_ACCESS_KEY_ID": "FAKE-ACCESS-KEY-ID-0000",
    "AWS_SECRET_ACCESS_KEY": "fake-secret-access-key-placeholder",
    "AWS_REGION": "us-east-2",
    "FRED_API_KEY": "fake-fred-api-key-placeholder",
    "BEDROCK_MODEL_ID": "us.openai.gpt-5.6-terra",
}
FAKE_OPTIONAL_CREDENTIAL = {"AWS_SESSION_TOKEN": "fake-session-token-placeholder"}
FAKE_SESSION_TOKEN = FAKE_OPTIONAL_CREDENTIAL["AWS_SESSION_TOKEN"]
FAKE_CREDENTIALS = (
    FAKE_REQUIRED["AWS_ACCESS_KEY_ID"],
    FAKE_REQUIRED["AWS_SECRET_ACCESS_KEY"],
    FAKE_REQUIRED["FRED_API_KEY"],
    FAKE_SESSION_TOKEN,
)


def write_dotenv(directory: Path, values: dict[str, str]) -> Path:
    """Write ``values`` as a ``.env`` file in ``directory`` and return its path."""
    path = directory / ".env"
    path.write_text("".join(f"{name}={value}\n" for name, value in values.items()))
    return path


def settings_from(dotenv_path: Path, environ: dict[str, str]) -> Settings:
    """Run the startup path: ``read_environment`` then ``load_settings``."""
    return load_settings(read_environment(dotenv_path, environ=environ))


def assert_optional_defaults(s: Settings) -> None:
    """Every optional setting holds its documented default (Req 12.3)."""
    assert s.aws_session_token is None
    assert s.polly_voice_id == DEFAULT_POLLY_VOICE_ID
    assert s.polly_engine == DEFAULT_POLLY_ENGINE
    assert s.transcribe_language_code == DEFAULT_TRANSCRIBE_LANGUAGE_CODE
    assert s.indicator_map_path is None
    assert s.port == DEFAULT_PORT


# --- Optional defaults (Req 12.3) -------------------------------------------


def test_unset_optionals_use_defaults(tmp_path: Path) -> None:
    assert_optional_defaults(settings_from(write_dotenv(tmp_path, FAKE_REQUIRED), {}))


def test_blank_optionals_use_defaults(tmp_path: Path) -> None:
    blanks = {name: "  " for name in OPTIONAL}
    dotenv = write_dotenv(tmp_path, {**FAKE_REQUIRED, **{name: "" for name in OPTIONAL}})
    assert_optional_defaults(settings_from(dotenv, blanks))


def test_defaults_match_env_example() -> None:
    """The code defaults equal the defaults shown in ``.env.example`` (placeholders only)."""
    example = dotenv_values(ENV_EXAMPLE, interpolate=False)
    shown = {name: value or "" for name, value in example.items() if name in OPTIONAL}
    assert set(shown) == set(OPTIONAL)
    assert load_settings({**FAKE_REQUIRED, **shown}) == load_settings(FAKE_REQUIRED)


def test_set_optionals_override_defaults(tmp_path: Path) -> None:
    map_path = str(tmp_path / "fake-indicators.json")
    dotenv = write_dotenv(
        tmp_path,
        {
            **FAKE_REQUIRED,
            "AWS_SESSION_TOKEN": FAKE_SESSION_TOKEN,
            "POLLY_VOICE_ID": "FakeVoice",
            "POLLY_ENGINE": "standard",
            "TRANSCRIBE_LANGUAGE_CODE": "en-GB",
            "INDICATOR_MAP_PATH": map_path,
            "FRIDAY_PORT": "8123",
        },
    )
    s = settings_from(dotenv, {})
    assert s.aws_session_token == FAKE_SESSION_TOKEN
    assert (s.polly_voice_id, s.polly_engine) == ("FakeVoice", "standard")
    assert s.transcribe_language_code == "en-GB"
    assert s.indicator_map_path == map_path
    assert s.port == 8123


# --- No .env file (Req 12.10) -----------------------------------------------


def test_no_dotenv_reads_process_env_only(tmp_path: Path) -> None:
    missing = tmp_path / ".env"
    assert not missing.exists()
    s = settings_from(missing, {**FAKE_REQUIRED, "POLLY_VOICE_ID": "FakeVoice"})
    assert s.aws_region == FAKE_REQUIRED["AWS_REGION"]
    assert s.polly_voice_id == "FakeVoice"
    assert s.port == DEFAULT_PORT


def test_no_dotenv_ignores_unrelated_process_vars(tmp_path: Path) -> None:
    env = read_environment(tmp_path / ".env", environ={"AWS_REGION": "eu-west-1", "HOME": "/x"})
    assert env == {"AWS_REGION": "eu-west-1"}


def test_no_dotenv_and_empty_process_env_reports_every_required(tmp_path: Path) -> None:
    with pytest.raises(ConfigError) as info:
        settings_from(tmp_path / ".env", {})
    assert info.value.missing == tuple(REQUIRED)


# --- Process environment wins over .env (Req 12.2) --------------------------


def test_process_env_wins_over_dotenv(tmp_path: Path) -> None:
    dotenv = write_dotenv(
        tmp_path, {**FAKE_REQUIRED, "AWS_REGION": "us-west-2", "POLLY_VOICE_ID": "DotenvVoice"}
    )
    s = settings_from(dotenv, {"AWS_REGION": "eu-west-1", "POLLY_VOICE_ID": "ProcessVoice"})
    assert s.aws_region == "eu-west-1"
    assert s.polly_voice_id == "ProcessVoice"
    assert s.fred_api_key == FAKE_REQUIRED["FRED_API_KEY"]


def test_blank_process_value_keeps_dotenv_value(tmp_path: Path) -> None:
    dotenv = write_dotenv(tmp_path, {**FAKE_REQUIRED, "POLLY_ENGINE": "standard"})
    s = settings_from(dotenv, {"AWS_REGION": "   ", "POLLY_ENGINE": ""})
    assert s.aws_region == FAKE_REQUIRED["AWS_REGION"]
    assert s.polly_engine == "standard"


def test_process_value_fills_variable_absent_from_dotenv(tmp_path: Path) -> None:
    partial = {k: v for k, v in FAKE_REQUIRED.items() if k != "BEDROCK_MODEL_ID"}
    s = settings_from(write_dotenv(tmp_path, partial), {"BEDROCK_MODEL_ID": "fake.model-id"})
    assert s.bedrock_model_id == "fake.model-id"


def test_merge_environment_precedence() -> None:
    merged = merge_environment(
        {"AWS_REGION": "us-west-2", "POLLY_ENGINE": "standard", "FRIDAY_PORT": None},
        {"AWS_REGION": "eu-west-1", "POLLY_ENGINE": "  ", "UNRELATED": "x"},
    )
    assert merged == {"AWS_REGION": "eu-west-1", "POLLY_ENGINE": "standard"}


def test_default_environ_is_os_environ_and_not_mutated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AWS_REGION", "eu-west-1")
    monkeypatch.delenv("POLLY_VOICE_ID", raising=False)
    dotenv = write_dotenv(tmp_path, {"AWS_REGION": "us-west-2", "POLLY_VOICE_ID": "FakeVoice"})
    env = read_environment(dotenv)
    assert env["AWS_REGION"] == "eu-west-1"
    assert env["POLLY_VOICE_ID"] == "FakeVoice"
    assert "POLLY_VOICE_ID" not in os.environ


# --- Missing required variables (Req 12.6) ----------------------------------


def test_missing_vars_named_in_one_message_without_credentials(tmp_path: Path) -> None:
    """Credentials present in .env and the process env never leak into the error."""
    dotenv = write_dotenv(
        tmp_path,
        {
            "AWS_ACCESS_KEY_ID": FAKE_REQUIRED["AWS_ACCESS_KEY_ID"],
            "AWS_SESSION_TOKEN": FAKE_SESSION_TOKEN,
            "AWS_REGION": "   ",
        },
    )
    environ = {
        "AWS_SECRET_ACCESS_KEY": FAKE_REQUIRED["AWS_SECRET_ACCESS_KEY"],
        "FRED_API_KEY": "",
    }
    with pytest.raises(ConfigError) as info:
        settings_from(dotenv, environ)
    err = info.value
    assert err.kind == "config_missing"
    assert err.missing == ("AWS_REGION", "FRED_API_KEY", "BEDROCK_MODEL_ID")
    text = f"{err.message} {err} {err!r}"
    for name in err.missing:
        assert name in err.message
    for name in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"):
        assert name not in err.message
    for value in FAKE_CREDENTIALS:
        assert value not in text


def test_whitespace_only_required_counts_as_missing() -> None:
    with pytest.raises(ConfigError) as info:
        load_settings({**FAKE_REQUIRED, "AWS_SECRET_ACCESS_KEY": " \t "})
    assert info.value.missing == ("AWS_SECRET_ACCESS_KEY",)


# --- Credential hiding and Redactor inputs (Req 12.4) -----------------------


def test_settings_repr_hides_credentials() -> None:
    s = load_settings({**FAKE_REQUIRED, "AWS_SESSION_TOKEN": FAKE_SESSION_TOKEN})
    text = repr(s)
    for secret in secret_values(s):
        assert secret not in text
    assert "us-east-2" in text


def test_secret_values_includes_token_only_when_set() -> None:
    assert len(secret_values(load_settings(FAKE_REQUIRED))) == 3
    s = load_settings({**FAKE_REQUIRED, "AWS_SESSION_TOKEN": FAKE_SESSION_TOKEN})
    assert sorted(secret_values(s)) == sorted(FAKE_CREDENTIALS)


# --- Port parsing examples (Req 13.3, 13.5) ---------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [(None, 8000), ("", 8000), ("  ", 8000), ("1", 1), ("65535", 65535), (" 8080 ", 8080)],
)
def test_parse_port_valid(raw: str | None, expected: int) -> None:
    assert parse_port(raw) == expected


@pytest.mark.parametrize("raw", ["0", "65536", "-1", "+80", "80.0", "abc", "1_000", "٨٠"])
def test_parse_port_invalid(raw: str) -> None:
    with pytest.raises(PortError) as info:
        parse_port(raw)
    assert info.value.cause == "invalid value"
    assert raw in info.value.message


def test_invalid_port_in_dotenv_raises_port_error(tmp_path: Path) -> None:
    dotenv = write_dotenv(tmp_path, {**FAKE_REQUIRED, "FRIDAY_PORT": "99999"})
    with pytest.raises(PortError):
        settings_from(dotenv, {})
