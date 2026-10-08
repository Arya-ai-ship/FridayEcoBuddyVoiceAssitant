"""Checks that constants and style values match the design."""

import re

from friday import constants as c
from friday import style


def test_limits_match_design() -> None:
    assert c.MAX_TEXT_CHARS == 2000
    assert c.MAX_TOOL_CALLS == 8
    assert c.LLM_TIMEOUT_S == 60
    assert c.STT_TIMEOUT_S == 30
    assert c.TTS_TIMEOUT_S == 15
    assert c.FRED_TIMEOUT_S == 15
    assert c.CLIENT_TIMEOUT_S == 60
    assert c.MIC_MAX_S == 60
    assert c.MAX_SPOKEN_WORDS == 60
    assert c.MAX_DONE_WORDS == 10
    assert c.PREVIEW_ROWS == 10
    assert c.MAX_PLOT_SERIES == 5
    assert c.MAX_TITLE_CHARS == 100
    assert c.MAX_CHAT_BODY_BYTES == 16 * 1024
    assert c.MAX_AUDIO_BYTES == 2_500_000
    assert c.DEFAULT_PORT == 8000
    assert c.BIND_HOST == "127.0.0.1"


def test_audio_limit_covers_mic_cap() -> None:
    bytes_per_s = c.STT_SAMPLE_RATE_HZ * 2  # PCM16 mono
    assert c.MIC_MAX_S * bytes_per_s <= c.MAX_AUDIO_BYTES


def test_palette_has_one_color_per_plot_series() -> None:
    assert len(style.NEON_PALETTE) == c.MAX_PLOT_SERIES


def test_style_values_are_hex_colors() -> None:
    hex_color = re.compile(r"^#[0-9A-F]{6}$")
    colors = [
        *style.NEON_PALETTE,
        *style.UI_TOKENS.values(),
        *style.REDUCED_MOTION_STATE_COLORS.values(),
        style.CHART_BG,
        style.CHART_TEXT,
        style.CHART_GRID,
    ]
    assert all(hex_color.match(color) for color in colors)
    assert set(style.REDUCED_MOTION_STATE_COLORS) == {"idle", "listening", "working", "speaking"}
