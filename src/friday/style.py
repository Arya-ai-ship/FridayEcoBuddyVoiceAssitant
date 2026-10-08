"""Colors and chart style (single source of truth).

UI tokens are mirrored as ``:root`` custom properties in ``static/styles.css`` and
checked by ``tests/test_parity.py``. ``data/plot.py`` reads the chart style from here.
"""

from types import MappingProxyType
from typing import Final

# --- Neon_Accent colors ----------------------------------------------------
CYAN: Final = "#00E5FF"
VIOLET: Final = "#B388FF"
MAGENTA: Final = "#FF4FD8"
LIME: Final = "#B2FF59"
AMBER: Final = "#FFD740"

NEON_PALETTE: Final[tuple[str, ...]] = (CYAN, VIOLET, MAGENTA, LIME, AMBER)
"""Line colors assigned to chart series by index (Req 14.8)."""

# --- Browser_UI (Dark_Theme) -----------------------------------------------
UI_BG: Final = "#0A0E17"
"""Page background; WCAG relative luminance about 0.004 (Req 14.3)."""

UI_TEXT: Final = "#E6F1FF"
"""Primary text; contrast about 17:1 on ``UI_BG`` (Req 14.4)."""

ORB_IDLE: Final = "#5B6B8C"
"""Reduced-motion static orb color for ``idle`` (Req 14.6)."""

REDUCED_MOTION_STATE_COLORS: Final = MappingProxyType(
    {
        "idle": ORB_IDLE,
        "listening": CYAN,
        "working": AMBER,
        "speaking": VIOLET,
    }
)
"""One distinct static orb color per Assistant_State (Req 14.6)."""

UI_TOKENS: Final = MappingProxyType(
    {
        "--bg": UI_BG,
        "--text": UI_TEXT,
        "--cyan": CYAN,
        "--violet": VIOLET,
        "--magenta": MAGENTA,
        "--amber": AMBER,
        "--orb-idle": REDUCED_MOTION_STATE_COLORS["idle"],
        "--orb-listening": REDUCED_MOTION_STATE_COLORS["listening"],
        "--orb-working": REDUCED_MOTION_STATE_COLORS["working"],
        "--orb-speaking": REDUCED_MOTION_STATE_COLORS["speaking"],
    }
)
"""CSS custom properties that ``styles.css`` must declare in ``:root`` with these values."""

# --- Chart style (Plot_Tool fixed styling, Req 11.4, 14.7) ------------------
CHART_BG: Final = "#0B0F1A"
"""Chart background; WCAG relative luminance about 0.005."""

CHART_TEXT: Final = "#E6F1FF"
"""Axes, labels, ticks, title, and legend text."""

CHART_GRID: Final = "#1E2A44"
"""Grid line color."""

CHART_LINE_WIDTH: Final = 1.8
"""Line width (points) of every chart series."""

CHART_WIDTH_IN: Final = 10.0
CHART_HEIGHT_IN: Final = 5.5
CHART_DPI: Final = 110
CHART_FORMAT: Final = "png"
