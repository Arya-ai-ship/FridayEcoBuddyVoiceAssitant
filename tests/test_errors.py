"""Unit tests for the FridayError hierarchy and its wire codes."""

import pytest

from friday.errors import (
    AwsCredentialError,
    ChartRenderError,
    ConfigError,
    DatasetNotFound,
    DateRangeError,
    FredError,
    FridayError,
    LLMUnavailableError,
    PortError,
    RangeTooShortForYoY,
    RequestError,
    STTError,
    ToolError,
    ToolFailedError,
    ToolValidationError,
    TTSError,
    UnknownIndicator,
)

# Wire codes per class, exactly as listed in the design's error hierarchy.
DESIGN_KINDS: dict[type[FridayError], set[str]] = {
    ConfigError: {"config_missing", "indicator_map_invalid", "port_invalid"},
    AwsCredentialError: {"aws_credentials"},
    LLMUnavailableError: {"llm_unavailable"},
    TTSError: {"tts_failed", "tts_timeout"},
    STTError: {"stt_failed", "stt_timeout", "stt_empty"},
    RequestError: {
        "invalid_text",
        "request_too_large",
        "audio_too_large",
        "dataset_not_found",
        "invalid_session",
    },
    ToolValidationError: {"unknown_tool", "invalid_args"},
    UnknownIndicator: {"unknown_indicator"},
    DateRangeError: {"invalid_date"},
    FredError: {"http_error", "timeout", "no_observations"},
    RangeTooShortForYoY: {"range_too_short_for_yoy"},
    DatasetNotFound: {"dataset_not_found"},
    ChartRenderError: {"chart_render_failed"},
    ToolFailedError: {"tool_failed"},
}

TOOL_ERRORS = [
    ToolValidationError,
    UnknownIndicator,
    DateRangeError,
    FredError,
    RangeTooShortForYoY,
    DatasetNotFound,
    ChartRenderError,
    ToolFailedError,
]


@pytest.mark.parametrize(("cls", "kinds"), DESIGN_KINDS.items())
def test_kinds_match_design(cls: type[FridayError], kinds: set[str]) -> None:
    assert set(cls.KINDS) == kinds
    assert issubclass(cls, FridayError)


@pytest.mark.parametrize("cls", TOOL_ERRORS)
def test_tool_errors_subclass_tool_error(cls: type[FridayError]) -> None:
    assert issubclass(cls, ToolError)


def test_port_error_is_config_error_naming_value_and_cause() -> None:
    err = PortError("99999", "invalid value")
    assert isinstance(err, ConfigError)
    assert err.kind == "port_invalid"
    assert "99999" in err.message and "invalid value" in err.message


def test_unknown_kind_is_rejected() -> None:
    with pytest.raises(ValueError, match="does not accept kind"):
        TTSError("stt_failed", "x")  # pyright: ignore[reportArgumentType]


def test_missing_vars_names_every_variable_in_one_message() -> None:
    err = ConfigError.missing_vars(["AWS_REGION", "FRED_API_KEY"])
    assert err.kind == "config_missing"
    assert err.missing == ("AWS_REGION", "FRED_API_KEY")
    assert "AWS_REGION" in str(err) and "FRED_API_KEY" in str(err)


def test_indicator_map_error_names_path_and_cause() -> None:
    err = ConfigError.indicator_map("config/map.json", "entry 'cpi' has no series_id")
    assert err.kind == "indicator_map_invalid"
    assert "config/map.json" in err.message and "cpi" in err.message


def test_tool_result_shape() -> None:
    result = DatasetNotFound("ds-9").to_result("describe_data")
    assert result == {
        "ok": False,
        "tool": "describe_data",
        "error": {
            "kind": "dataset_not_found",
            "message": "dataset 'ds-9' was not found",
            "dataset_id": "ds-9",
        },
    }


def test_unknown_indicator_lists_every_supported_name() -> None:
    err = UnknownIndicator("gdp", ["cpi", "inflation"])
    result = err.to_result("fetch_data")
    error = result["error"]
    assert isinstance(error, dict)
    assert error["supported"] == ["cpi", "inflation"]
    assert "cpi" in err.message and "inflation" in err.message


def test_validation_error_names_field() -> None:
    err = ToolValidationError.invalid_args("fill_missing", "method", "unsupported")
    assert err.kind == "invalid_args"
    assert err.field == "method"
    assert "method" in err.message
    unknown = ToolValidationError.unknown_tool("delete_all", ["fetch_data"])
    assert unknown.kind == "unknown_tool" and "delete_all" in unknown.message


def test_tool_failed_message_names_only_the_tool() -> None:
    result = ToolFailedError("plot_data").to_result("plot_data")
    assert result["error"] == {
        "kind": "tool_failed",
        "message": "the plot_data tool failed unexpectedly",
    }
