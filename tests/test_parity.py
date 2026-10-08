"""Parity between the Python sources of truth and their browser mirrors.

- ``static/js/constants.js`` must match ``friday/constants.py`` (same names and values).
- ``static/styles.css`` ``:root`` color tokens must match ``friday/style.UI_TOKENS``,
  and no literal hex color may appear outside ``:root``.
"""

import re
from pathlib import Path

from friday import constants, style

STATIC = Path(__file__).resolve().parents[1] / "src" / "friday" / "static"
CONSTANTS_JS = STATIC / "js" / "constants.js"
STYLES_CSS = STATIC / "styles.css"

# Browser-facing limits that constants.js must define (design: js/constants.js).
BROWSER_CONSTANTS = frozenset(
    {"MAX_TEXT_CHARS", "CLIENT_TIMEOUT_S", "MIC_MAX_S", "MIC_SILENCE_S", "STT_SAMPLE_RATE_HZ"}
)

JS_EXPORT = re.compile(r"^export\s+const\s+([A-Za-z_$][\w$]*)\s*=\s*(.+?);?\s*$", re.MULTILINE)
JS_NUMBER = re.compile(r"^-?\d[\d_]*(\.\d+)?$")
CSS_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)
CSS_ROOT = re.compile(r"(?<![\w-]):root\s*\{(?P<body>[^}]*)\}")
CSS_PROPERTY = re.compile(r"(--[\w-]+)\s*:\s*([^;]+);")
HEX_COLOR = re.compile(r"#[0-9A-Fa-f]{3,8}\b")


def js_constants(text: str) -> dict[str, float]:
    """Parse ``export const NAME = <number>;`` lines; fail on any non-literal export."""
    values: dict[str, float] = {}
    for name, raw in JS_EXPORT.findall(text):
        literal = raw.strip()
        assert JS_NUMBER.match(literal), f"{name} must be a plain numeric literal, got {literal!r}"
        assert name not in values, f"{name} is exported twice"
        values[name] = float(literal.replace("_", ""))
    return values


def css_root_tokens(text: str) -> dict[str, str]:
    """Return the custom properties declared in every top-level ``:root`` block."""
    css = CSS_COMMENT.sub("", text)
    tokens: dict[str, str] = {}
    for block in CSS_ROOT.finditer(css):
        for name, value in CSS_PROPERTY.findall(block.group("body")):
            tokens[name] = value.strip()
    return tokens


def css_outside_root(text: str) -> str:
    """Return the stylesheet with comments and ``:root`` blocks removed."""
    return CSS_ROOT.sub("", CSS_COMMENT.sub("", text))


def test_constants_js_defines_browser_limits() -> None:
    values = js_constants(CONSTANTS_JS.read_text(encoding="utf-8"))
    assert BROWSER_CONSTANTS.issubset(values)


def test_constants_js_matches_constants_py() -> None:
    values = js_constants(CONSTANTS_JS.read_text(encoding="utf-8"))
    for name, value in values.items():
        assert hasattr(constants, name), f"{name} in constants.js has no constants.py source"
        assert getattr(constants, name) == value, f"{name}: constants.js={value}"


def test_styles_root_tokens_match_style_py() -> None:
    tokens = css_root_tokens(STYLES_CSS.read_text(encoding="utf-8"))
    for name, value in style.UI_TOKENS.items():
        assert name in tokens, f"{name} missing from styles.css :root"
        assert tokens[name].upper() == value.upper(), f"{name}: styles.css={tokens[name]}"


def test_styles_root_has_no_extra_color_tokens() -> None:
    tokens = css_root_tokens(STYLES_CSS.read_text(encoding="utf-8"))
    colors = {name for name, value in tokens.items() if HEX_COLOR.search(value)}
    assert colors <= set(style.UI_TOKENS), "add new colors to style.UI_TOKENS first"


def test_styles_use_tokens_instead_of_literal_colors() -> None:
    rest = css_outside_root(STYLES_CSS.read_text(encoding="utf-8"))
    assert HEX_COLOR.findall(rest) == []


def test_parsers_reject_drift() -> None:
    assert js_constants("export const A = 1_000;\n") == {"A": 1000.0}
    assert css_root_tokens(":root { --bg: #000000; }\n.x { --bg: #FFFFFF; }") == {"--bg": "#000000"}
    assert HEX_COLOR.findall(css_outside_root(":root{--a:#111111;}\n.x{color:#222222;}")) == [
        "#222222"
    ]
