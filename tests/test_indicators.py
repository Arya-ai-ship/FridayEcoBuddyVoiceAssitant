"""Unit tests for the Indicator_Map: default map, resolution examples, and loading.

Resolution across arbitrary maps is covered by the property test in
``test_properties_indicators.py``; this file covers the packaged Default_Indicator_Map
(Req 6.1, 6.14) and every malformed-file failure of the loader (Req 12.7).
"""

import json
import os
from pathlib import Path
from typing import Final

import pytest

from friday.data.indicators import (
    DEFAULT_MAP_RESOURCE,
    IndicatorEntry,
    default_indicator_map,
    load_indicator_map,
)
from friday.errors import ConfigError, UnknownIndicator

# The Default_Indicator_Map table from the requirements Glossary, row for row:
# name -> (aliases, FRED series ID, Transformation).
GLOSSARY: Final[dict[str, tuple[tuple[str, ...], str, str | None]]] = {
    "cpi": (("consumer price index", "cpi-u", "headline cpi"), "CPIAUCSL", None),
    "inflation": (("inflation rate", "cpi inflation", "yoy inflation"), "CPIAUCSL", "yoy"),
    "core cpi": (("core consumer price index", "cpi less food and energy"), "CPILFESL", None),
    "unemployment rate": (("unemployment", "jobless rate"), "UNRATE", None),
    "nonfarm payrolls": (("payrolls", "nfp", "total nonfarm employment"), "PAYEMS", None),
    "fed funds rate": (
        ("federal funds rate", "fed funds", "effective federal funds rate"),
        "FEDFUNDS",
        None,
    ),
    "10-year treasury yield": (
        ("10 year treasury yield", "10-year yield", "10y yield", "ten-year treasury yield"),
        "GS10",
        None,
    ),
    "housing prices": (
        ("home prices", "house prices", "case-shiller", "case-shiller index"),
        "CSUSHPINSA",
        None,
    ),
    "housing starts": (("new housing starts", "housing units started"), "HOUST", None),
    "industrial production": (("industrial production index", "ip index"), "INDPRO", None),
}


def _write(tmp_path: Path, data: object) -> Path:
    path = tmp_path / "map.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _entry(name: str, *aliases: str, series_id: str = "S") -> dict[str, object]:
    return {"name": name, "aliases": list(aliases), "series_id": series_id}


def _load_error(path: Path) -> ConfigError:
    """Load ``path``, expecting the loader's ConfigError; check kind and that it names the path."""
    with pytest.raises(ConfigError) as info:
        load_indicator_map(path)
    error = info.value
    assert error.kind == "indicator_map_invalid"
    assert error.message.startswith(f"Invalid Indicator_Map file {path}: ")
    return error


# --- Default_Indicator_Map (Req 6.1, 6.14) ----------------------------------


def test_default_map_matches_the_glossary_table_exactly() -> None:
    imap = default_indicator_map()
    assert imap.names == tuple(GLOSSARY)
    got = {e.name: (e.aliases, e.series_id, e.transformation) for e in imap.entries}
    assert got == GLOSSARY


def test_default_map_only_inflation_uses_yoy() -> None:
    yoy = [e.name for e in default_indicator_map().entries if e.transformation is not None]
    assert yoy == ["inflation"]


def test_unset_path_loads_the_packaged_default_map(tmp_path: Path) -> None:
    from_none = load_indicator_map(None)
    assert from_none == default_indicator_map()
    assert len(from_none.entries) == 10
    # The packaged resource loaded as an explicit file yields the same map.
    packaged = Path(__file__).parents[1] / "src" / "friday" / "data" / DEFAULT_MAP_RESOURCE
    assert load_indicator_map(packaged) == from_none
    assert load_indicator_map(str(packaged)) == from_none


def test_every_default_alias_belongs_to_exactly_one_indicator() -> None:
    imap = default_indicator_map()
    for entry in imap.entries:
        for key in (entry.name, *entry.aliases):
            assert imap.resolve(key) is entry


# --- Resolution examples (Req 6.2, 6.9) -------------------------------------


@pytest.mark.parametrize(
    ("query", "name"),
    [
        ("Inflation", "inflation"),
        ("  YOY Inflation \n", "inflation"),
        ("CPI-U", "cpi"),
        ("case-shiller", "housing prices"),
        ("10Y Yield", "10-year treasury yield"),
        ("nfp", "nonfarm payrolls"),
    ],
)
def test_resolve_ignores_case_and_whitespace(query: str, name: str) -> None:
    assert default_indicator_map().resolve(query).name == name


def test_resolve_unknown_lists_every_supported_name() -> None:
    imap = default_indicator_map()
    with pytest.raises(UnknownIndicator) as info:
        imap.resolve("gdp")
    assert info.value.supported == imap.names
    assert all(name in info.value.message for name in GLOSSARY)


# --- Loading a custom file (Req 6.1) ----------------------------------------


def test_load_custom_file_strips_fields(tmp_path: Path) -> None:
    data = {"indicators": [{"name": " GDP ", "aliases": [" output "], "series_id": " GDP "}]}
    imap = load_indicator_map(_write(tmp_path, data))
    assert imap.entries == (IndicatorEntry("GDP", ("output",), "GDP", None),)
    assert imap.resolve("OUTPUT").series_id == "GDP"


def test_repeated_alias_within_one_entry_is_not_a_collision(tmp_path: Path) -> None:
    imap = load_indicator_map(_write(tmp_path, {"indicators": [_entry("gdp", "Output", "output")]}))
    assert imap.resolve("output").name == "gdp"


# --- Malformed files (Req 12.7): every error names the path and the cause ----


def test_missing_file(tmp_path: Path) -> None:
    error = _load_error(tmp_path / "nope.json")
    assert error.message.endswith(": file does not exist")


def test_directory_is_unreadable(tmp_path: Path) -> None:
    error = _load_error(tmp_path)
    assert "file cannot be read (IsADirectoryError)" in error.message


@pytest.mark.skipif(
    not hasattr(os, "geteuid") or os.geteuid() == 0, reason="root ignores file permissions"
)
def test_file_without_read_permission_is_unreadable(tmp_path: Path) -> None:
    path = _write(tmp_path, {"indicators": [_entry("gdp")]})
    path.chmod(0)
    try:
        error = _load_error(path)
    finally:
        path.chmod(0o600)
    assert "file cannot be read (PermissionError)" in error.message


def test_non_utf8_file_is_unreadable(tmp_path: Path) -> None:
    path = tmp_path / "latin1.json"
    path.write_bytes(b'{"indicators": [{"name": "caf\xe9", "series_id": "X"}]}')
    error = _load_error(path)
    assert "file cannot be read (UnicodeDecodeError)" in error.message


@pytest.mark.parametrize("text", ["{not json", "", '{"indicators": [}'])
def test_bad_json(tmp_path: Path, text: str) -> None:
    path = tmp_path / "bad.json"
    path.write_text(text, encoding="utf-8")
    error = _load_error(path)
    assert "invalid JSON (" in error.message
    assert "at line 1)" in error.message


@pytest.mark.parametrize(
    ("data", "cause"),
    [
        ([_entry("gdp")], "top level must be an object with 'indicators'"),
        ({"series": []}, "'indicators' must be a non-empty list"),
        ({"indicators": []}, "'indicators' must be a non-empty list"),
        ({"indicators": {"gdp": "GDP"}}, "'indicators' must be a non-empty list"),
        ({"indicators": [_entry("gdp"), "GDP"]}, "entry 2 must be an object"),
        ({"indicators": [{"series_id": "GDP"}]}, "entry 1 has no Indicator name"),
        ({"indicators": [{"name": "  ", "series_id": "GDP"}]}, "entry 1 has no Indicator name"),
        (
            {"indicators": [{"name": "gdp", "aliases": "output", "series_id": "GDP"}]},
            "indicator 'gdp': 'aliases' must be a list of strings",
        ),
        (
            {"indicators": [{"name": "gdp", "aliases": ["output", " "], "series_id": "GDP"}]},
            "indicator 'gdp': every alias must be a non-empty string",
        ),
        (
            {"indicators": [{"name": "gdp", "series_id": "GDP", "transformation": "mom"}]},
            "indicator 'gdp': unsupported transformation 'mom'",
        ),
    ],
)
def test_not_an_indicator_map(tmp_path: Path, data: object, cause: str) -> None:
    assert cause in _load_error(_write(tmp_path, data)).message


@pytest.mark.parametrize("series_id", ["", "   ", None, 42])
def test_empty_series_id_names_the_indicator(tmp_path: Path, series_id: object) -> None:
    data = {"indicators": [_entry("cpi"), {"name": " Real GDP ", "series_id": series_id}]}
    error = _load_error(_write(tmp_path, data))
    assert error.message.endswith(": indicator 'Real GDP' has no FRED series ID")


def test_missing_series_id_names_the_indicator(tmp_path: Path) -> None:
    error = _load_error(_write(tmp_path, {"indicators": [{"name": "gdp", "aliases": ["output"]}]}))
    assert error.message.endswith(": indicator 'gdp' has no FRED series ID")


@pytest.mark.parametrize(
    ("entries", "key", "first", "second"),
    [
        # alias vs. alias of another entry, equal only after case and whitespace normalization
        ([_entry("a", "Shared"), _entry("b", " shared\t")], "shared", "a", "b"),
        # name vs. another entry's alias
        ([_entry("GDP"), _entry("output", " gdp ")], "gdp", "GDP", "output"),
        # alias vs. a later entry's name
        ([_entry("cpi", "Inflation"), _entry("INFLATION ")], "inflation", "cpi", "INFLATION"),
        # name vs. name
        ([_entry("cpi"), _entry(" CPI")], "cpi", "cpi", "CPI"),
    ],
)
def test_normalized_collisions_name_both_indicators(
    tmp_path: Path, entries: list[dict[str, object]], key: str, first: str, second: str
) -> None:
    error = _load_error(_write(tmp_path, {"indicators": entries}))
    expected = f"duplicate name or alias: '{key}' is used by both '{first}' and '{second}'"
    assert error.message.endswith(expected)
