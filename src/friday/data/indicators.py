"""Indicator_Map: maps Indicator names and aliases to FRED series IDs (Req 6.1, 6.2, 6.9, 12.7).

The map is loaded once at startup, either from ``INDICATOR_MAP_PATH`` or from the packaged
``default_indicators.json`` (the Default_Indicator_Map). Matching ignores letter case and
leading/trailing whitespace. This module is pure apart from reading the map file.
"""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from types import MappingProxyType
from typing import Final, Literal, cast

from friday.errors import ConfigError, UnknownIndicator

Transformation = Literal["yoy"]

DEFAULT_MAP_RESOURCE: Final = "default_indicators.json"
"""File name of the packaged Default_Indicator_Map inside ``friday.data``."""

_TRANSFORMATIONS: Final[frozenset[str]] = frozenset({"yoy"})


def normalize(text: str) -> str:
    """Normalize an Indicator name, alias, or query for matching (Req 6.2)."""
    return text.strip().casefold()


@dataclass(frozen=True)
class IndicatorEntry:
    """One Indicator: canonical name, aliases, FRED series ID, and optional Transformation."""

    name: str
    aliases: tuple[str, ...]
    series_id: str
    transformation: Transformation | None = None

    def match_keys(self) -> tuple[str, ...]:
        """Normalized name followed by normalized aliases, without duplicates."""
        return tuple(dict.fromkeys(normalize(t) for t in (self.name, *self.aliases)))


@dataclass(frozen=True)
class IndicatorMap:
    """An immutable Indicator_Map whose normalized names and aliases are unique."""

    entries: tuple[IndicatorEntry, ...]
    _index: Mapping[str, IndicatorEntry] = field(
        init=False, repr=False, compare=False, default_factory=lambda: MappingProxyType({})
    )

    def __post_init__(self) -> None:
        """Build the normalized lookup index; raise ``ValueError`` on any key collision."""
        index: dict[str, IndicatorEntry] = {}
        for entry in self.entries:
            for key in entry.match_keys():
                other = index.get(key)
                if other is not None:
                    raise ValueError(f"'{key}' is used by both '{other.name}' and '{entry.name}'")
                index[key] = entry
        object.__setattr__(self, "_index", MappingProxyType(index))

    @property
    def names(self) -> tuple[str, ...]:
        """Canonical Indicator names in map order."""
        return tuple(entry.name for entry in self.entries)

    def resolve(self, query: str) -> IndicatorEntry:
        """Return the entry whose name or alias matches ``query`` (Req 6.2).

        Raises ``UnknownIndicator`` listing every supported Indicator name (Req 6.9).
        """
        entry = self._index.get(normalize(query))
        if entry is None:
            raise UnknownIndicator(query, self.names)
        return entry


# --- Loading ---------------------------------------------------------------


def load_indicator_map(path: str | Path | None) -> IndicatorMap:
    """Load the Indicator_Map from ``path``, or the Default_Indicator_Map when ``path`` is None.

    Raises ``ConfigError`` (kind ``indicator_map_invalid``) naming the path and the cause
    when the file is missing, unreadable, not valid JSON, or not a valid Indicator_Map
    (Req 6.1, 12.7).
    """
    if path is None:
        source = f"<packaged {DEFAULT_MAP_RESOURCE}>"
        text = resources.files("friday.data").joinpath(DEFAULT_MAP_RESOURCE).read_text("utf-8")
        return parse_indicator_map(text, source)
    file_path = Path(path)
    try:
        text = file_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise ConfigError.indicator_map(str(file_path), "file does not exist") from None
    except (OSError, UnicodeDecodeError) as exc:
        cause = f"file cannot be read ({type(exc).__name__})"
        raise ConfigError.indicator_map(str(file_path), cause) from None
    return parse_indicator_map(text, str(file_path))


def default_indicator_map() -> IndicatorMap:
    """Return the packaged Default_Indicator_Map of 11 Indicators (Req 6.14)."""
    return load_indicator_map(None)


def parse_indicator_map(text: str, source: str) -> IndicatorMap:
    """Parse Indicator_Map JSON text. ``source`` names the file in error messages."""
    try:
        data: object = json.loads(text)
    except json.JSONDecodeError as exc:
        cause = f"invalid JSON ({exc.msg} at line {exc.lineno})"
        raise ConfigError.indicator_map(source, cause) from None
    if not isinstance(data, Mapping):
        raise ConfigError.indicator_map(source, "top level must be an object with 'indicators'")
    raw_entries = cast(Mapping[str, object], data).get("indicators")
    if not isinstance(raw_entries, list) or not raw_entries:
        raise ConfigError.indicator_map(source, "'indicators' must be a non-empty list")
    entries = tuple(
        _parse_entry(raw, position, source)
        for position, raw in enumerate(cast(list[object], raw_entries), start=1)
    )
    try:
        return IndicatorMap(entries)
    except ValueError as exc:
        raise ConfigError.indicator_map(source, f"duplicate name or alias: {exc}") from None


def _parse_entry(raw: object, position: int, source: str) -> IndicatorEntry:
    """Validate one raw entry; errors name the Indicator (or its position if unnamed)."""
    if not isinstance(raw, Mapping):
        raise ConfigError.indicator_map(source, f"entry {position} must be an object")
    fields = cast(Mapping[str, object], raw)
    name = fields.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ConfigError.indicator_map(source, f"entry {position} has no Indicator name")
    label = f"indicator '{name.strip()}'"
    series_id = fields.get("series_id")
    if not isinstance(series_id, str) or not series_id.strip():
        raise ConfigError.indicator_map(source, f"{label} has no FRED series ID")
    return IndicatorEntry(
        name=name.strip(),
        aliases=_parse_aliases(fields.get("aliases", []), label, source),
        series_id=series_id.strip(),
        transformation=_parse_transformation(fields.get("transformation"), label, source),
    )


def _parse_aliases(raw: object, label: str, source: str) -> tuple[str, ...]:
    if not isinstance(raw, list):
        raise ConfigError.indicator_map(source, f"{label}: 'aliases' must be a list of strings")
    items = cast(Sequence[object], raw)
    if not all(isinstance(a, str) and a.strip() for a in items):
        raise ConfigError.indicator_map(source, f"{label}: every alias must be a non-empty string")
    return tuple(cast(str, a).strip() for a in items)


def _parse_transformation(raw: object, label: str, source: str) -> Transformation | None:
    if raw is None:
        return None
    if isinstance(raw, str) and raw in _TRANSFORMATIONS:
        return cast(Transformation, raw)
    raise ConfigError.indicator_map(
        source, f'{label}: unsupported transformation {raw!r} (use null or "yoy")'
    )
