"""Packaging checks: src layout, console script, and pinned dependencies (Req 13.1)."""

import re
import tomllib
from pathlib import Path

import friday
import friday.cli

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"
PINNED = re.compile(r"^[A-Za-z0-9_.\-]+(\[[A-Za-z0-9_,\-]+\])?==[0-9][A-Za-z0-9.\-+]*$")


def _pyproject() -> dict[str, object]:
    return tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))


def test_package_is_importable_from_src_layout() -> None:
    assert Path(friday.__file__).parent.parent.name == "src"
    assert callable(friday.cli.main)


def test_console_script_points_at_cli_main() -> None:
    project = _pyproject()["project"]
    assert isinstance(project, dict)
    assert project["scripts"] == {"friday": "friday.cli:main"}
    assert project["requires-python"] == ">=3.12"


def test_every_dependency_is_pinned_exactly() -> None:
    data = _pyproject()
    project = data["project"]
    groups = data["dependency-groups"]
    assert isinstance(project, dict) and isinstance(groups, dict)
    deps = [*project["dependencies"], *groups["dev"]]
    unpinned = [d for d in deps if not PINNED.match(d)]
    assert deps
    assert unpinned == []
