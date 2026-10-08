"""Architecture guard-rails for ``src/friday`` (design: Engineering Standards).

Parses every module with ``ast`` (nothing is imported or executed) and fails on:

- an import edge the layering table forbids (internal layers and third-party packages),
- an import of a restricted dotted prefix (the MAF Bedrock package) outside its owners,
- environment access (``os.environ``, ``os.getenv``, ``dotenv``) outside ``config.py``,
- a ``print(`` call outside ``cli.py``,
- a module that is not assigned to a layer yet (classify it in ``LAYER_OF`` below).
"""

from __future__ import annotations

import ast
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "friday"
ROOT_PACKAGE = "friday"

DOMAIN = "domain"
CONFIG = "config"
PORTS = "ports"
APPLICATION = "application"
ADAPTERS = "adapters"
COMPOSITION = "composition"
PACKAGE = "package"  # package ``__init__`` modules

# Module (dotted, relative to ``friday``) -> layer. A trailing ``.*`` matches a whole
# subpackage. Exact entries win over wildcards.
LAYER_OF: dict[str, str] = {
    "errors": DOMAIN,
    "constants": DOMAIN,
    "style": DOMAIN,
    "redact": DOMAIN,
    "agent.narration": DOMAIN,
    "agent.templates": DOMAIN,
    "data.*": DOMAIN,
    "config": CONFIG,
    "ports": PORTS,
    "agent.loop": APPLICATION,
    "agent.harness": APPLICATION,
    "agent.middleware": APPLICATION,
    "agent.tools": APPLICATION,
    "agent.tool_args": APPLICATION,
    "agent.tool_runs": APPLICATION,
    "agent.prompts": APPLICATION,
    "session": APPLICATION,
    "events": APPLICATION,
    "llm.bedrock": ADAPTERS,
    "speech.*": ADAPTERS,
    "data.fred": ADAPTERS,
    "aws_errors": ADAPTERS,
    "aws_session": ADAPTERS,
    "wiring": COMPOSITION,
    "server": COMPOSITION,
    "cli": COMPOSITION,
    "check": COMPOSITION,
}

ALL_LAYERS = frozenset({DOMAIN, CONFIG, PORTS, APPLICATION, ADAPTERS, COMPOSITION, PACKAGE})

# Internal layers each layer may import. ``config`` is importable everywhere because
# it is side-effect free on import and holds the frozen ``Settings`` type.
ALLOWED_INTERNAL: dict[str, frozenset[str]] = {
    DOMAIN: frozenset({DOMAIN, CONFIG, PACKAGE}),
    CONFIG: frozenset({DOMAIN, PACKAGE}),
    PORTS: frozenset({DOMAIN, CONFIG, PORTS, PACKAGE}),
    APPLICATION: frozenset({DOMAIN, CONFIG, PORTS, APPLICATION, PACKAGE}),
    ADAPTERS: frozenset({DOMAIN, CONFIG, PORTS, ADAPTERS, PACKAGE}),
    COMPOSITION: ALL_LAYERS,
    PACKAGE: frozenset({DOMAIN, CONFIG, PORTS, PACKAGE}),
}

# Third-party top-level packages each layer may import (stdlib is always allowed).
# ``None`` means unrestricted.
PURE_THIRD_PARTY = frozenset({"numpy", "pydantic"})
ALLOWED_THIRD_PARTY: dict[str, frozenset[str] | None] = {
    DOMAIN: PURE_THIRD_PARTY,
    CONFIG: frozenset({"dotenv"}),
    PORTS: PURE_THIRD_PARTY,
    APPLICATION: PURE_THIRD_PARTY | {"agent_framework"},  # MAF core
    ADAPTERS: None,
    COMPOSITION: None,
    PACKAGE: PURE_THIRD_PARTY,
}
# matplotlib is allowed only in the chart module of the domain layer.
EXTRA_THIRD_PARTY: dict[str, frozenset[str]] = {"data.plot": frozenset({"matplotlib"})}

# Dotted import prefixes restricted to specific modules, whatever their layer allows.
# The MAF Bedrock package is ``agent_framework_bedrock``; ``agent_framework.amazon``
# re-exports it lazily (spike item 1), so it is banned even though ``agent_framework``
# itself is allowed in the application layer.
BEDROCK_IMPORTERS = frozenset({"llm.bedrock", "wiring"})
RESTRICTED_PREFIXES: dict[str, frozenset[str]] = {
    "agent_framework_bedrock": BEDROCK_IMPORTERS,
    "agent_framework.amazon": BEDROCK_IMPORTERS,
}

ENV_MODULE = "config"
PRINT_MODULE = "cli"
ENV_ATTRS = frozenset({"environ", "environb", "getenv", "getenvb", "putenv", "unsetenv"})
ENV_PACKAGES = frozenset({"dotenv"})


@dataclass(frozen=True)
class Module:
    """A parsed source module: dotted name relative to ``friday`` and its AST."""

    name: str  # "" for friday/__init__.py, "data" for friday/data/__init__.py
    is_package: bool
    tree: ast.Module


def module_name(path: Path, src: Path = SRC) -> tuple[str, bool]:
    """Return the dotted name (relative to the root package) and whether it is a package."""
    parts = list(path.relative_to(src).with_suffix("").parts)
    is_package = parts[-1] == "__init__"
    if is_package:
        parts = parts[:-1]
    return ".".join(parts), is_package


def load_modules(src: Path = SRC) -> list[Module]:
    """Parse every ``.py`` file under ``src``."""
    modules: list[Module] = []
    for path in sorted(src.rglob("*.py")):
        name, is_package = module_name(path, src)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        modules.append(Module(name, is_package, tree))
    return modules


def layer_of(name: str, is_package: bool) -> str | None:
    """Return the layer of a module, or ``None`` if it is unclassified."""
    if is_package:
        return PACKAGE
    if name in LAYER_OF:
        return LAYER_OF[name]
    for pattern, layer in LAYER_OF.items():
        if pattern.endswith(".*") and name.startswith(pattern[:-1]):
            return layer
    return None


def _resolve_relative(module: Module, level: int, target: str | None) -> str:
    """Resolve a relative import to an absolute dotted name under ``friday``."""
    base = [ROOT_PACKAGE, *module.name.split(".")] if module.name else [ROOT_PACKAGE]
    if not module.is_package:
        base = base[:-1]
    if level > 1:
        base = base[: len(base) - (level - 1)]
    return ".".join([*base, *([target] if target else [])])


def imported_names(module: Module) -> list[str]:
    """Absolute dotted names imported by a module, including ``from x import y`` as ``x.y``.

    ``x.y`` is later narrowed to ``x`` when ``y`` is not a module.
    """
    names: list[str] = []
    for node in ast.walk(module.tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = (
                _resolve_relative(module, node.level, node.module)
                if node.level
                else node.module or ""
            )
            names.append(base)
            names.extend(f"{base}.{alias.name}" for alias in node.names if alias.name != "*")
    return names


def _internal_target(dotted: str, known: dict[str, bool]) -> str | None:
    """Map an absolute ``friday.*`` name to the longest known module name, if any."""
    parts = dotted.split(".")[1:]
    while True:
        candidate = ".".join(parts)
        if candidate in known:
            return candidate
        if not parts:
            return None
        parts = parts[:-1]


def restricted_prefix_violations(module: Module) -> list[str]:
    """Return imports matching a ``RESTRICTED_PREFIXES`` entry the module may not use."""
    problems: list[str] = []
    for dotted in imported_names(module):
        for prefix, importers in RESTRICTED_PREFIXES.items():
            matches = dotted == prefix or dotted.startswith(f"{prefix}.")
            if matches and module.name not in importers:
                problems.append(
                    f"friday.{module.name} imports '{dotted}' "
                    f"(only {', '.join(sorted(importers))} may import '{prefix}')"
                )
    return problems


def import_violations(module: Module, known: dict[str, bool]) -> list[str]:
    """Return the forbidden import edges of one module."""
    layer = layer_of(module.name, module.is_package)
    if layer is None:
        return [f"friday.{module.name}: module is not assigned to a layer in LAYER_OF"]
    allowed_tp = ALLOWED_THIRD_PARTY[layer]
    if allowed_tp is not None:
        allowed_tp = allowed_tp | EXTRA_THIRD_PARTY.get(module.name, frozenset())
    problems = restricted_prefix_violations(module)
    for dotted in imported_names(module):
        top = dotted.split(".")[0]
        if top == ROOT_PACKAGE:
            target = _internal_target(dotted, known)
            if target is None:
                continue  # imports a not-yet-existing module; checked once it exists
            target_layer = layer_of(target, known[target])
            if target_layer is not None and target_layer not in ALLOWED_INTERNAL[layer]:
                problems.append(
                    f"friday.{module.name} ({layer}) imports friday.{target} ({target_layer})"
                )
        elif top in sys.stdlib_module_names:
            continue
        elif allowed_tp is not None and top not in allowed_tp:
            problems.append(f"friday.{module.name} ({layer}) imports third-party '{top}'")
    return sorted(set(problems))


def env_violations(module: Module) -> list[str]:
    """Return environment accesses (``os.environ``/``getenv``, dotenv) in one module."""
    os_aliases = {"os"}
    problems: list[str] = []
    for node in ast.walk(module.tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "os":
                    os_aliases.add(alias.asname or "os")
                if alias.name.split(".")[0] in ENV_PACKAGES:
                    problems.append(f"line {node.lineno}: imports {alias.name}")
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            if node.module.split(".")[0] in ENV_PACKAGES:
                problems.append(f"line {node.lineno}: imports from {node.module}")
            if node.module == "os":
                problems.extend(
                    f"line {node.lineno}: imports os.{a.name}"
                    for a in node.names
                    if a.name in ENV_ATTRS
                )
    for node in ast.walk(module.tree):
        if (
            isinstance(node, ast.Attribute)
            and node.attr in ENV_ATTRS
            and isinstance(node.value, ast.Name)
            and node.value.id in os_aliases
        ):
            problems.append(f"line {node.lineno}: uses {node.value.id}.{node.attr}")
    return problems


def print_violations(module: Module) -> list[str]:
    """Return ``print(`` calls in one module."""
    return [
        f"line {node.lineno}: print("
        for node in ast.walk(module.tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "print"
    ]


# --- Checks against the real source tree -------------------------------------

MODULES = load_modules()
KNOWN = {m.name: m.is_package for m in MODULES}


def test_source_tree_is_found() -> None:
    assert "constants" in KNOWN
    assert "" in KNOWN  # friday/__init__.py


@pytest.mark.parametrize("module", MODULES, ids=lambda m: f"friday.{m.name}".rstrip("."))
def test_imports_respect_layering(module: Module) -> None:
    assert import_violations(module, KNOWN) == []


@pytest.mark.parametrize(
    "module",
    [m for m in MODULES if m.name != ENV_MODULE],
    ids=lambda m: f"friday.{m.name}".rstrip("."),
)
def test_no_environment_access_outside_config(module: Module) -> None:
    assert env_violations(module) == []


@pytest.mark.parametrize(
    "module",
    [m for m in MODULES if m.name != PRINT_MODULE],
    ids=lambda m: f"friday.{m.name}".rstrip("."),
)
def test_no_print_outside_cli(module: Module) -> None:
    assert print_violations(module) == []


# --- Checks that the checker itself catches violations -----------------------


def _module(name: str, source: str, is_package: bool = False) -> Module:
    return Module(name, is_package, ast.parse(source))


SAMPLE_KNOWN = {
    "": True,
    "agent": True,
    "agent.loop": False,
    "agent.middleware": False,
    "agent.narration": False,
    "aws_errors": False,
    "aws_session": False,
    "data": True,
    "data.fetch": False,
    "data.fred": False,
    "data.plot": False,
    "errors": False,
    "config": False,
    "llm": True,
    "llm.bedrock": False,
    "ports": False,
    "server": False,
    "speech": True,
    "speech.polly": False,
    "wiring": False,
}


@pytest.mark.parametrize(
    ("name", "source"),
    [
        ("data.fetch", "import boto3"),
        ("data.fetch", "import httpx"),
        ("agent.narration", "from fastapi import FastAPI"),
        ("data.fetch", "import matplotlib.pyplot"),
        ("data.fetch", "from friday.agent import loop"),
        ("data.fetch", "from ..wiring import build_container"),
        ("ports", "import botocore"),
        ("agent.loop", "from friday.data.fred import FredClient"),
        ("data.fred", "from friday.agent.loop import Agent"),
        ("data.fred", "import friday.wiring"),
        ("config", "from friday.ports import TTSClient"),
        ("unknown_module", "import json"),
        # MAF: core is application-only; the Bedrock package only in llm.bedrock/wiring.
        ("data.fetch", "import agent_framework"),
        ("data.fetch", "from agent_framework import Agent"),
        ("agent.loop", "from agent_framework_bedrock import BedrockChatClient"),
        ("agent.loop", "import agent_framework_bedrock"),
        ("agent.loop", "from agent_framework.amazon import BedrockChatClient"),
        ("agent.loop", "from agent_framework import amazon"),
        ("agent.loop", "import agent_framework.amazon"),
        ("speech.polly", "from agent_framework_bedrock import BedrockChatClient"),
        ("server", "from agent_framework_bedrock import BedrockChatClient"),
        ("agent.middleware", "from friday.aws_errors import classify_aws_error"),
        ("agent.middleware", "from ..aws_errors import classify_aws_error"),
        ("agent.middleware", "import botocore"),
        ("agent.middleware", "from botocore.exceptions import ClientError"),
        ("agent.middleware", "from friday.aws_session import isolated_boto3_session"),
        ("llm.bedrock", "from friday.agent.middleware import FridayMiddleware"),
    ],
)
def test_checker_flags_forbidden_imports(name: str, source: str) -> None:
    assert import_violations(_module(name, source), SAMPLE_KNOWN) != []


@pytest.mark.parametrize(
    ("name", "source"),
    [
        ("data.fetch", "import numpy as np\nfrom friday.errors import FridayError"),
        ("data.fetch", "from ..errors import FridayError\nfrom . import plot"),
        ("data.plot", "from matplotlib.figure import Figure"),
        ("agent.loop", "from friday.ports import TTSClient\nfrom friday.config import Settings"),
        ("data.fred", "import httpx\nfrom friday.errors import FridayError"),
        ("wiring", "import boto3\nfrom friday.data.fred import FredClient"),
        ("config", "from dotenv import dotenv_values"),
        ("agent.loop", "import agent_framework"),
        ("agent.loop", "from agent_framework import Agent, create_harness_agent"),
        ("agent.middleware", "from agent_framework import FunctionMiddleware"),
        ("llm.bedrock", "from agent_framework_bedrock import BedrockChatClient"),
        ("llm.bedrock", "import boto3\nfrom friday.aws_errors import classify_aws_error"),
        ("llm.bedrock", "from friday.aws_session import isolated_boto3_session"),
        ("speech.polly", "from ..aws_session import isolated_boto3_session"),
        ("wiring", "from agent_framework_bedrock import BedrockChatClient"),
        ("wiring", "from agent_framework.amazon import BedrockChatClient"),
    ],
)
def test_checker_allows_permitted_imports(name: str, source: str) -> None:
    assert import_violations(_module(name, source), SAMPLE_KNOWN) == []


@pytest.mark.parametrize(
    "source",
    [
        "import os\nx = os.environ['A']",
        "import os\nx = os.getenv('A')",
        "import os as o\nx = o.environ.get('A')",
        "from os import environ",
        "from dotenv import load_dotenv",
        "import dotenv",
    ],
)
def test_checker_flags_environment_access(source: str) -> None:
    assert env_violations(_module("server", source)) != []


def test_checker_allows_other_os_use() -> None:
    assert env_violations(_module("server", "import os\np = os.path.join('a', 'b')")) == []


def test_checker_flags_print() -> None:
    assert print_violations(_module("server", "print('hi')")) != []
    assert print_violations(_module("server", "log.info('print(')")) == []
