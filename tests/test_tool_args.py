"""Unit tests for tool argument models, specs, and validation (Req 5.4, 5.8, 10.10, 11.4, 11.8)."""

import json

import pytest
from pydantic import ValidationError

from friday.agent.templates import TOOL_NAMES
from friday.agent.tool_args import (
    ARGUMENTS_FIELD,
    METHOD_FIELD,
    TOOL_MODELS,
    DescribeArgs,
    FetchArgs,
    FillArgs,
    PlotArgs,
    ToolSpec,
    tool_specs,
    validate_tool_args,
)
from friday.agent.tools import ToolRegistry
from friday.data.dates import END_FIELD, START_FIELD
from friday.data.plot import IDS_FIELD, PLOT_TOOL, TITLE_FIELD
from friday.errors import ToolValidationError

DATE = {"type": "string", "pattern": "^[0-9]{4}-[0-9]{2}-[0-9]{2}$"}

# The documented schemas (design: "Tool argument schemas (sent to the LLM)").
DOCUMENTED_SCHEMAS: dict[str, dict[str, object]] = {
    "fetch_data": {
        "type": "object",
        "additionalProperties": False,
        "required": ["indicator"],
        "properties": {
            "indicator": {"type": "string", "minLength": 1, "maxLength": 100},
            "start_date": DATE,
            "end_date": DATE,
        },
    },
    "describe_data": {
        "type": "object",
        "additionalProperties": False,
        "properties": {"dataset_id": {"type": "string"}},
    },
    "fill_missing": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "dataset_id": {"type": "string"},
            "method": {"type": "string", "enum": ["forward_fill", "linear_interpolation"]},
        },
    },
    "plot_data": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "dataset_ids": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
                "maxItems": 5,
                "uniqueItems": True,
            },
            "title": {"type": "string", "maxLength": 100},
            "start_date": DATE,
            "end_date": DATE,
        },
    },
}


def _rejected(name: str, raw: object) -> ToolValidationError:
    with pytest.raises(ToolValidationError) as info:
        validate_tool_args(name, raw)
    return info.value


# --- Specs -------------------------------------------------------------------


def test_exactly_the_four_tools_in_order() -> None:
    assert tuple(TOOL_MODELS) == TOOL_NAMES
    assert [spec.name for spec in ToolRegistry.specs()] == list(TOOL_NAMES)


@pytest.mark.parametrize("spec", tool_specs(), ids=lambda s: s.name)
def test_spec_schema_matches_design(spec: ToolSpec) -> None:
    assert spec.input_schema == DOCUMENTED_SCHEMAS[spec.name]
    assert spec.description == TOOL_MODELS[spec.name].description


def test_schemas_are_flat_for_bedrock() -> None:
    text = json.dumps([spec.input_schema for spec in tool_specs()])
    for token in ("$defs", "$ref", "anyOf", "null", "default"):
        assert token not in text


def test_field_names_match_shared_constants() -> None:
    assert {START_FIELD, END_FIELD} <= set(FetchArgs.model_fields)
    assert {IDS_FIELD, TITLE_FIELD, START_FIELD, END_FIELD} == set(PlotArgs.model_fields)
    assert METHOD_FIELD in FillArgs.model_fields
    assert PlotArgs.tool == PLOT_TOOL


# --- Valid arguments ---------------------------------------------------------


def test_valid_arguments_parse_to_models() -> None:
    fetch = validate_tool_args("fetch_data", {"indicator": "cpi", "start_date": "2020-01-01"})
    assert fetch == FetchArgs(indicator="cpi", start_date="2020-01-01")
    assert validate_tool_args("describe_data", {}) == DescribeArgs()
    plot = validate_tool_args("plot_data", {"dataset_ids": ["ds-1", "ds-2"], "title": "T"})
    assert isinstance(plot, PlotArgs)
    assert plot.dataset_ids == ("ds-1", "ds-2")


def test_fill_method_defaults_to_forward_fill_even_for_null() -> None:
    assert validate_tool_args("fill_missing", {}) == FillArgs(method="forward_fill")
    assert validate_tool_args("fill_missing", {"method": None}) == FillArgs()
    args = validate_tool_args("fill_missing", {"method": "linear_interpolation"})
    assert args == FillArgs(method="linear_interpolation")


@pytest.mark.parametrize("raw", [None, "", "  ", "{}"])
def test_no_arguments(raw: object) -> None:
    assert validate_tool_args("describe_data", raw) == DescribeArgs()


def test_json_text_arguments() -> None:
    assert ToolRegistry.validate("fetch_data", '{"indicator": "gdp"}') == FetchArgs(indicator="gdp")


def test_explicit_null_means_omitted() -> None:
    args = validate_tool_args("plot_data", {"dataset_ids": None, "title": None})
    assert args == PlotArgs()


def test_models_are_frozen() -> None:
    args = FetchArgs(indicator="cpi")
    with pytest.raises(ValidationError):
        args.indicator = "gdp"  # type: ignore[misc]


# --- Rejections ----------------------------------------------------------------


def test_unknown_tool_is_named() -> None:
    err = _rejected("web_search", {})
    assert err.kind == "unknown_tool"
    assert err.name == "web_search"
    assert err.message == (
        "unknown tool 'web_search'; supported: fetch_data, describe_data, fill_missing, plot_data"
    )


def test_invalid_method_message() -> None:
    err = _rejected("fill_missing", {"method": "x"})
    assert err.kind == "invalid_args"
    assert err.field == "method"
    assert err.message == "invalid method 'x'; supported: forward_fill, linear_interpolation"
    assert err.to_result("fill_missing")["error"] == {
        "kind": "invalid_args",
        "message": err.message,
        "field": "method",
    }


def test_long_values_are_truncated_in_messages() -> None:
    err = _rejected("fill_missing", {"method": "m" * 500})
    assert len(err.message) < 120


@pytest.mark.parametrize(
    ("name", "raw", "field", "reason"),
    [
        ("fetch_data", {}, "indicator", "is required"),
        ("fetch_data", {"indicator": ""}, "indicator", "must not be empty"),
        ("fetch_data", {"indicator": "x" * 101}, "indicator", "at most 100 characters, got 101"),
        ("fetch_data", {"indicator": 5}, "indicator", "must be a string"),
        (
            "fetch_data",
            {"indicator": "cpi", "start_date": "2020/01/01"},
            "start_date",
            "YYYY-MM-DD",
        ),
        ("fetch_data", {"indicator": "cpi", "end_date": "20200101"}, "end_date", "YYYY-MM-DD"),
        ("fetch_data", {"indicator": "cpi", "series_id": "X"}, "series_id", "allowed: indicator"),
        ("describe_data", {"dataset_id": 3}, "dataset_id", "must be a string"),
        ("plot_data", {"dataset_ids": []}, "dataset_ids", "must not be empty"),
        ("plot_data", {"dataset_ids": list("abcdef")}, "dataset_ids", "at most 5 items, got 6"),
        ("plot_data", {"dataset_ids": ["a", "a"]}, "dataset_ids", "'a' is listed more than once"),
        ("plot_data", {"dataset_ids": "ds-1"}, "dataset_ids", "must be an array of strings"),
        ("plot_data", {"dataset_ids": ["a", 1]}, "dataset_ids", "item 1 must be a string"),
        ("plot_data", {"title": "t" * 101}, "title", "at most 100 characters, got 101"),
        ("plot_data", {"code": "import os"}, "code", "is not accepted"),
    ],
)
def test_invalid_arguments_name_the_field(name: str, raw: object, field: str, reason: str) -> None:
    err = _rejected(name, raw)
    assert err.kind == "invalid_args"
    assert err.name == name
    assert err.field == field
    assert err.message.startswith(f"invalid argument '{field}': ")
    assert reason in err.message


@pytest.mark.parametrize(
    "raw",
    [{"raw": '{"indicator": "cpi"'}, "{not json", "[1, 2]", ["indicator"], 7],
)
def test_malformed_arguments_are_rejected(raw: object) -> None:
    err = _rejected("fetch_data", raw)
    assert err.kind == "invalid_args"
    assert err.field == ARGUMENTS_FIELD
    assert err.message.startswith(f"invalid argument '{ARGUMENTS_FIELD}': ")
