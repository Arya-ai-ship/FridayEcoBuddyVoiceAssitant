"""Tool argument models, LLM tool specs, and argument validation (Req 5.4, 5.8, 10.10, 11.4).

Each of the four Tools has one frozen Pydantic model with ``extra="forbid"``. That single
definition gives both the JSON schema sent to the LLM (``tool_specs``) and the validator
(``validate_tool_args``), so the two cannot drift apart. Limits come from
``friday.constants``, tool names from ``agent.templates``, Fill_Methods from ``data.fill``,
and the date shape from ``data.dates``.

Schema flattening: ``model_json_schema()`` would render an optional ``X | None`` field as
``anyOf: [X, {"type": "null"}]`` with ``default``/``title`` keys. ``input_schema`` flattens
that to just ``X`` (an optional argument is simply omitted) and drops the model-level
``title``/``description``, so each schema is exactly the documented one. No
model nests another and the Fill_Method ``Literal`` renders inline as an ``enum``, so the
schemas have no ``$defs``/``$ref``, which keeps them portable across Bedrock Converse
models. Explicit ``null`` values are still accepted and mean "omitted".

Validation errors are translated from Pydantic's wording into short Friday messages that
name the offending argument (Req 5.8). The pattern only checks the date shape; the Tool
checks that the date exists on the calendar (``parse_date_range``).
"""

import json
from collections.abc import Mapping, Sized
from dataclasses import dataclass
from types import MappingProxyType
from typing import Annotated, ClassVar, Final, cast

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from friday.agent.templates import DESCRIBE, FETCH, FILL, PLOT, TOOL_NAMES, ToolName
from friday.constants import MAX_ECHO_CHARS, MAX_INDICATOR_CHARS, MAX_PLOT_SERIES, MAX_TITLE_CHARS
from friday.data.dates import ISO_DATE_PATTERN
from friday.data.fill import DEFAULT_FILL_METHOD, FillMethod, invalid_method_message
from friday.errors import ToolValidationError

DATE_PATTERN: Final = f"^{ISO_DATE_PATTERN}$"
"""Schema pattern for ``start_date``/``end_date`` (shape only)."""
DATE_FORMAT: Final = "YYYY-MM-DD"
"""How the date shape is described in error messages."""

ARGUMENTS_FIELD: Final = "arguments"
"""Field named when the arguments as a whole are not a JSON object."""
RAW_KEY: Final = "raw"
"""MAF passes arguments that are not valid JSON as ``{"raw": "<text>"}`` (spike item 4)."""
METHOD_FIELD: Final = "method"
"""The Fill_Tool argument with its own error wording (Req 10.10)."""

IsoDate = Annotated[str, Field(pattern=DATE_PATTERN)]


@dataclass(frozen=True)
class ToolSpec:
    """One Tool as the LLM sees it: name, description, and JSON schema of its arguments."""

    name: ToolName
    description: str
    input_schema: dict[str, object]


class ToolArgsModel(BaseModel):
    """Base of the four argument models: immutable, unknown arguments rejected."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    tool: ClassVar[ToolName]
    description: ClassVar[str]


class FetchArgs(ToolArgsModel):
    """``fetch_data`` arguments."""

    tool: ClassVar[ToolName] = FETCH
    description: ClassVar[str] = (
        "Fetch a supported US economic indicator from FRED. Returns a dataset_id."
    )

    indicator: Annotated[str, Field(min_length=1, max_length=MAX_INDICATOR_CHARS)]
    start_date: IsoDate | None = None
    end_date: IsoDate | None = None


class DescribeArgs(ToolArgsModel):
    """``describe_data`` arguments."""

    tool: ClassVar[ToolName] = DESCRIBE
    description: ClassVar[str] = (
        "Descriptive statistics for a dataset. Omit dataset_id to use the most recent dataset."
    )

    dataset_id: str | None = None


class FillArgs(ToolArgsModel):
    """``fill_missing`` arguments; ``method`` defaults to ``forward_fill`` (Req 10.2)."""

    tool: ClassVar[ToolName] = FILL
    description: ClassVar[str] = (
        "Fill missing values into a NEW dataset. Default method forward_fill."
    )

    dataset_id: str | None = None
    method: FillMethod = DEFAULT_FILL_METHOD

    @field_validator(METHOD_FIELD, mode="before")
    @classmethod
    def _null_means_default(cls, value: object) -> object:
        return DEFAULT_FILL_METHOD if value is None else value


class PlotArgs(ToolArgsModel):
    """``plot_data`` arguments (Req 11.4): 1-5 distinct IDs, a short title, a date range."""

    tool: ClassVar[ToolName] = PLOT
    description: ClassVar[str] = (
        "Line chart of 1-5 datasets on one axes. Omit dataset_ids to plot the most recent dataset."
    )

    dataset_ids: (
        Annotated[
            tuple[str, ...],
            Field(
                min_length=1, max_length=MAX_PLOT_SERIES, json_schema_extra={"uniqueItems": True}
            ),
        ]
        | None
    ) = None
    title: Annotated[str, Field(max_length=MAX_TITLE_CHARS)] | None = None
    start_date: IsoDate | None = None
    end_date: IsoDate | None = None

    @field_validator("dataset_ids")
    @classmethod
    def _distinct_ids(cls, value: tuple[str, ...] | None) -> tuple[str, ...] | None:
        seen: set[str] = set()
        for dataset_id in value or ():
            if dataset_id in seen:
                raise ValueError(f"dataset '{dataset_id}' is listed more than once")
            seen.add(dataset_id)
        return value


ToolArgs = FetchArgs | DescribeArgs | FillArgs | PlotArgs
"""Validated arguments of any Friday Tool; ``args.tool`` names the Tool."""

TOOL_MODELS: Final[Mapping[str, type[ToolArgs]]] = MappingProxyType(
    {model.tool: model for model in (FetchArgs, DescribeArgs, FillArgs, PlotArgs)}
)
"""Argument model per Tool name, in ``TOOL_NAMES`` order (Req 5.4)."""


_DROPPED_KEYS: Final = frozenset({"title", "description", "default"})
"""Schema annotations Pydantic adds that the documented schemas do not have."""
_NULL_BRANCH: Final = {"type": "null"}


def input_schema(model: type[ToolArgsModel]) -> dict[str, object]:
    """The JSON schema of ``model``'s arguments as sent to the LLM (flattened, see above)."""
    schema = model.model_json_schema()
    properties = cast("dict[str, dict[str, object]]", schema["properties"])
    flat = {key: value for key, value in schema.items() if key not in _DROPPED_KEYS}
    flat["properties"] = {name: _flat_property(prop) for name, prop in properties.items()}
    return flat


def _flat_property(prop: Mapping[str, object]) -> dict[str, object]:
    """Replace ``anyOf: [X, null]`` with ``X`` and drop ``title``/``default``."""
    flat = {key: value for key, value in prop.items() if key not in _DROPPED_KEYS}
    branches = cast("list[dict[str, object]]", flat.get("anyOf", []))
    concrete = [branch for branch in branches if branch != _NULL_BRANCH]
    if len(branches) == 2 and len(concrete) == 1:
        del flat["anyOf"]
        flat.update(concrete[0])
    return flat


def tool_specs() -> list[ToolSpec]:
    """The four Tool specs, built fresh from the argument models (Req 5.4)."""
    return [ToolSpec(m.tool, m.description, input_schema(m)) for m in TOOL_MODELS.values()]


def validate_tool_args(name: str, raw_args: object) -> ToolArgs:
    """Validate a Tool call from the LLM and return its typed arguments.

    ``raw_args`` is a mapping, JSON object text, or ``None`` (no arguments). Raises
    ``ToolValidationError``: ``unknown_tool`` naming the Tool and the supported ones, or
    ``invalid_args`` naming the first offending argument (Req 5.8, 10.10, 11.8).
    """
    model = TOOL_MODELS.get(name)
    if model is None:
        raise ToolValidationError.unknown_tool(name, TOOL_NAMES)
    arguments = _as_mapping(name, raw_args)
    try:
        return model.model_validate(arguments)
    except ValidationError as exc:
        raise _translate(name, model, exc) from exc


def _as_mapping(name: str, raw_args: object) -> Mapping[str, object]:
    """Normalise ``raw_args`` to a mapping, rejecting text that is not a JSON object."""
    if raw_args is None or (isinstance(raw_args, str) and not raw_args.strip()):
        return {}
    if isinstance(raw_args, str):
        try:
            raw_args = json.loads(raw_args)
        except json.JSONDecodeError:
            raise _not_json(name) from None
    if not isinstance(raw_args, Mapping):
        raise ToolValidationError.invalid_args(name, ARGUMENTS_FIELD, "must be a JSON object")
    arguments = cast("Mapping[str, object]", raw_args)
    if set(arguments) == {RAW_KEY} and isinstance(arguments[RAW_KEY], str):
        raise _not_json(name)
    return arguments


def _not_json(name: str) -> ToolValidationError:
    return ToolValidationError.invalid_args(name, ARGUMENTS_FIELD, "must be valid JSON")


@dataclass(frozen=True)
class _Issue:
    """The parts of one Pydantic error that the translation uses."""

    kind: str
    loc: tuple[int | str, ...]
    value: object
    ctx: Mapping[str, object]
    msg: str

    @classmethod
    def first(cls, exc: ValidationError) -> "_Issue":
        error = exc.errors()[0]
        return cls(error["type"], error["loc"], error["input"], error.get("ctx", {}), error["msg"])


def _translate(name: str, model: type[ToolArgsModel], exc: ValidationError) -> ToolValidationError:
    """Turn the first Pydantic error into a ``ToolValidationError`` naming the argument."""
    issue = _Issue.first(exc)
    field = str(issue.loc[0]) if issue.loc else ARGUMENTS_FIELD
    if field == METHOD_FIELD and issue.kind == "literal_error":
        message = invalid_method_message(_shown(issue.value))
        return ToolValidationError("invalid_args", message, name=name, field=field)
    return ToolValidationError.invalid_args(name, field, _reason(issue, model))


def _reason(issue: _Issue, model: type[ToolArgsModel]) -> str:
    """Concise reason for one Pydantic error (the field itself is named by the caller)."""
    ctx, value = issue.ctx, issue.value
    item = next((f"item {part} " for part in issue.loc[1:] if isinstance(part, int)), "")
    match issue.kind:
        case "missing":
            return "is required"
        case "extra_forbidden":
            return f"is not accepted; allowed: {', '.join(model.model_fields)}"
        case "string_type":
            return f"{item}must be a string"
        case "string_too_short" | "too_short" if ctx.get("min_length") == 1:
            return "must not be empty"
        case "string_too_long":
            return f"must be at most {ctx.get('max_length')} characters, got {_length(value)}"
        case "too_long":
            return f"must list at most {ctx.get('max_length')} items, got {_length(value)}"
        case "string_pattern_mismatch":
            return f"must be a date in {DATE_FORMAT} format, got '{_shown(value)}'"
        case "tuple_type" | "list_type":
            return "must be an array of strings"
        case "value_error" if "error" in ctx:
            return str(ctx["error"])  # a ValueError raised by one of the models' validators
        case _:
            return issue.msg


def _length(value: object) -> int | str:
    return len(value) if isinstance(value, Sized) else "?"


def _shown(value: object) -> str:
    """``value`` as text for an error message, cut to ``MAX_ECHO_CHARS``."""
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    return text if len(text) <= MAX_ECHO_CHARS else f"{text[: MAX_ECHO_CHARS - 1]}…"
