"""WCAG luminance and contrast checks for the UI and chart style tokens.

Covers Requirements 14.3 (Dark_Theme with Neon_Accent colors), 14.4 (text contrast
at least 4.5:1), 14.7 (dark chart background with readable text), and 14.8 (distinct
Neon_Accent line colors).
"""

import colorsys

import pytest

from friday import style

MAX_BG_LUMINANCE = 0.05
MIN_TEXT_CONTRAST = 4.5


def _channel(value: int) -> float:
    """Linearize one 8-bit sRGB channel (WCAG 2.x definition)."""
    c = value / 255
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def _rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.removeprefix("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def relative_luminance(hex_color: str) -> float:
    """WCAG 2.x relative luminance of a ``#RRGGBB`` color."""
    r, g, b = (_channel(v) for v in _rgb(hex_color))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast_ratio(a: str, b: str) -> float:
    """WCAG 2.x contrast ratio between two ``#RRGGBB`` colors."""
    hi, lo = sorted((relative_luminance(a), relative_luminance(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def _hue_deg(hex_color: str) -> float:
    r, g, b = (v / 255 for v in _rgb(hex_color))
    return colorsys.rgb_to_hsv(r, g, b)[0] * 360


def _saturation(hex_color: str) -> float:
    r, g, b = (v / 255 for v in _rgb(hex_color))
    return colorsys.rgb_to_hsv(r, g, b)[1]


# --- Helper sanity checks against known WCAG reference values ---------------


def test_luminance_reference_values() -> None:
    assert relative_luminance("#000000") == pytest.approx(0.0)
    assert relative_luminance("#FFFFFF") == pytest.approx(1.0)
    assert contrast_ratio("#000000", "#FFFFFF") == pytest.approx(21.0)
    assert contrast_ratio("#777777", "#FFFFFF") == pytest.approx(4.48, abs=0.01)


# --- Backgrounds (Req 14.3, 14.7) -------------------------------------------


@pytest.mark.parametrize("bg", [style.UI_BG, style.CHART_BG], ids=["page", "chart"])
def test_backgrounds_are_dark(bg: str) -> None:
    assert relative_luminance(bg) <= MAX_BG_LUMINANCE


# --- Text contrast (Req 14.4, 14.7) ------------------------------------------

UI_TEXT_TOKENS = {
    "text": style.UI_TEXT,
    "cyan": style.CYAN,  # neon labels, links, and focus ring
    "violet": style.VIOLET,
}


@pytest.mark.parametrize("fg", UI_TEXT_TOKENS.values(), ids=UI_TEXT_TOKENS.keys())
def test_ui_text_contrast(fg: str) -> None:
    assert contrast_ratio(fg, style.UI_BG) >= MIN_TEXT_CONTRAST


def test_chart_text_contrast() -> None:
    assert contrast_ratio(style.CHART_TEXT, style.CHART_BG) >= MIN_TEXT_CONTRAST


def test_ui_tokens_use_the_checked_values() -> None:
    assert style.UI_TOKENS["--bg"] == style.UI_BG
    assert style.UI_TOKENS["--text"] == style.UI_TEXT


# --- Neon_Accent palette (Req 14.3, 14.8) ------------------------------------


def test_palette_has_five_pairwise_distinct_colors() -> None:
    palette = [c.upper() for c in style.NEON_PALETTE]
    assert len(palette) == 5
    assert len(set(palette)) == 5, "palette colors must be pairwise distinct"


def test_palette_colors_are_saturated_and_visible_on_chart() -> None:
    for color in style.NEON_PALETTE:
        assert _saturation(color) >= 0.4, color
        assert contrast_ratio(color, style.CHART_BG) >= 3.0, color


def test_palette_includes_cyan_and_violet() -> None:
    hues = [_hue_deg(c) for c in style.NEON_PALETTE]
    assert any(170 <= h <= 200 for h in hues), "no cyan/teal color in palette"
    assert any(250 <= h <= 290 for h in hues), "no violet color in palette"
    assert 170 <= _hue_deg(style.CYAN) <= 200
    assert 250 <= _hue_deg(style.VIOLET) <= 290
