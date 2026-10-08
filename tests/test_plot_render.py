"""Unit tests for ``ChartRenderer`` (Req 11.4, 11.7, 11.10, 14.7).

PNG output is checked with the stdlib only (``struct``/``zlib``): signature, chunk CRCs,
IHDR dimensions, a complete IDAT stream, and the top-left pixel color.
"""

import math
import struct
import threading
import zlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date

import numpy as np
import pytest
from matplotlib.figure import Figure

import friday.data.plot as plot_module
from friday.constants import MAX_PLOT_SERIES
from friday.data.dataset import Dataset
from friday.data.plot import ChartRenderer, ChartSpec, build_chart_spec
from friday.errors import ChartRenderError
from friday.style import CHART_BG, CHART_DPI, CHART_HEIGHT_IN, CHART_WIDTH_IN

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
EXPECTED_WIDTH = round(CHART_WIDTH_IN * CHART_DPI)  # 1100
EXPECTED_HEIGHT = round(CHART_HEIGHT_IN * CHART_DPI)  # 605
CHANNELS_BY_COLOR_TYPE = {2: 3, 6: 4}  # truecolor RGB / RGBA
COLOR_TOLERANCE = 2


@dataclass(frozen=True)
class PngInfo:
    """What the stdlib PNG reader extracts."""

    width: int
    height: int
    color_type: int
    top_left_rgb: tuple[int, int, int]


def read_png(data: bytes) -> PngInfo:
    """Parse ``data`` as an 8-bit truecolor PNG, verifying its structure.

    The first pixel of the first scanline equals its raw (filtered) bytes for every PNG
    filter type, because its left, upper, and upper-left neighbours are all zero.
    """
    assert data.startswith(PNG_SIGNATURE)
    offset = len(PNG_SIGNATURE)
    chunks: list[tuple[bytes, bytes]] = []
    while offset < len(data):
        (length,) = struct.unpack(">I", data[offset : offset + 4])
        chunk_type = data[offset + 4 : offset + 8]
        body = data[offset + 8 : offset + 8 + length]
        (crc,) = struct.unpack(">I", data[offset + 8 + length : offset + 12 + length])
        assert zlib.crc32(chunk_type + body) == crc, f"bad CRC in {chunk_type!r}"
        chunks.append((chunk_type, body))
        offset += 12 + length
    assert offset == len(data)
    assert chunks[0][0] == b"IHDR"
    assert chunks[-1] == (b"IEND", b"")

    width, height, bit_depth, color_type, _, _, interlace = struct.unpack(">IIBBBBB", chunks[0][1])
    assert bit_depth == 8
    assert color_type in CHANNELS_BY_COLOR_TYPE
    assert interlace == 0
    channels = CHANNELS_BY_COLOR_TYPE[color_type]

    raw = zlib.decompress(b"".join(body for kind, body in chunks if kind == b"IDAT"))
    assert len(raw) == height * (1 + width * channels)
    assert raw[0] in range(5)  # a valid filter type for scanline 0
    r, g, b = raw[1], raw[2], raw[3]
    return PngInfo(width=width, height=height, color_type=color_type, top_left_rgb=(r, g, b))


def _hex_rgb(color: str) -> tuple[int, int, int]:
    return int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)


def assert_valid_chart_png(data: bytes) -> None:
    """A well-formed PNG at the fixed size whose corner shows the chart background."""
    info = read_png(data)
    assert (info.width, info.height) == (EXPECTED_WIDTH, EXPECTED_HEIGHT)
    expected = _hex_rgb(CHART_BG)
    assert all(
        abs(got - want) <= COLOR_TOLERANCE
        for got, want in zip(info.top_left_rgb, expected, strict=True)
    ), f"corner pixel {info.top_left_rgb} is not the chart background {expected}"


def _ds(dataset_id: str, indicator: str, values: list[float]) -> Dataset:
    return Dataset(
        dataset_id=dataset_id,
        indicator=indicator,
        value_column=indicator,
        series_id=indicator.upper(),
        dates=tuple(date(2020, month, 1) for month in range(1, len(values) + 1)),
        values=np.array(values, dtype=np.float64),
        transformation=None,
    )


INDICATORS = ("inflation", "unemployment", "gdp", "interest rates", "retail sales")


def _datasets(count: int) -> dict[str, Dataset]:
    datasets = (
        _ds(f"ds-{i + 1}", INDICATORS[i], [1.0 + i, math.nan, 3.0 + i, 4.0 + i, 2.0 + i])
        for i in range(count)
    )
    return {ds.dataset_id: ds for ds in datasets}


def _spec(count: int) -> ChartSpec:
    datasets = _datasets(count)
    return build_chart_spec(datasets, list(datasets))


def test_constants_match_the_documented_size() -> None:
    assert (EXPECTED_WIDTH, EXPECTED_HEIGHT) == (1100, 605)
    assert len(INDICATORS) == MAX_PLOT_SERIES


@pytest.mark.parametrize("count", [1, MAX_PLOT_SERIES])
def test_renders_a_valid_png_at_the_fixed_size(count: int) -> None:
    spec = _spec(count)
    assert spec.show_legend == (count > 1)
    assert_valid_chart_png(ChartRenderer().render(spec))


def test_renders_png_for_a_series_with_gaps() -> None:
    datasets = {"ds-1": _ds("ds-1", "inflation", [1.0, math.nan, 3.0, 4.0])}
    png = ChartRenderer().render(build_chart_spec(datasets, ["ds-1"]))
    assert_valid_chart_png(png)


def test_rendering_leaves_input_datasets_unchanged() -> None:
    datasets = _datasets(MAX_PLOT_SERIES)
    before = {
        dataset_id: (ds.dates, ds.values.copy(), ds.indicator, ds.series_id)
        for dataset_id, ds in datasets.items()
    }
    spec = build_chart_spec(datasets, list(datasets), start="2020-02-01", end="2020-04-01")

    ChartRenderer().render(spec)

    assert list(datasets) == list(before)
    for dataset_id, ds in datasets.items():
        dates, values, indicator, series_id = before[dataset_id]
        assert ds.dates == dates
        assert np.array_equal(ds.values, values, equal_nan=True)
        assert (ds.indicator, ds.series_id) == (indicator, series_id)
        assert not ds.values.flags.writeable


def test_matplotlib_failure_becomes_chart_render_error(monkeypatch: pytest.MonkeyPatch) -> None:
    boom = RuntimeError("agg exploded")

    def failing_savefig(self: Figure, *args: object, **kwargs: object) -> None:
        raise boom

    monkeypatch.setattr(Figure, "savefig", failing_savefig)
    renderer = ChartRenderer()

    with pytest.raises(ChartRenderError) as caught:
        renderer.render(_spec(1))

    assert caught.value.kind == "chart_render_failed"
    assert caught.value.__cause__ is boom
    assert "agg exploded" not in str(caught.value)

    # The lock was released, so the same renderer works once matplotlib recovers.
    monkeypatch.undo()
    assert_valid_chart_png(renderer.render(_spec(1)))


def test_chart_render_error_passes_through_unwrapped(monkeypatch: pytest.MonkeyPatch) -> None:
    original = ChartRenderError("custom")

    def failing_render(spec: ChartSpec) -> bytes:
        raise original

    monkeypatch.setattr(plot_module, "_render_png", failing_render)

    with pytest.raises(ChartRenderError) as caught:
        ChartRenderer().render(_spec(1))

    assert caught.value is original


def test_concurrent_renders_are_serialized_and_valid(monkeypatch: pytest.MonkeyPatch) -> None:
    real_render = plot_module._render_png  # pyright: ignore[reportPrivateUsage]
    guard = threading.Lock()
    active = 0
    peak = 0

    def tracking_render(spec: ChartSpec) -> bytes:
        nonlocal active, peak
        with guard:
            active += 1
            peak = max(peak, active)
        try:
            return real_render(spec)
        finally:
            with guard:
                active -= 1

    monkeypatch.setattr(plot_module, "_render_png", tracking_render)
    renderer = ChartRenderer()
    specs = [_spec(1 + i % MAX_PLOT_SERIES) for i in range(6)]

    with ThreadPoolExecutor(max_workers=len(specs)) as pool:
        results = list(pool.map(renderer.render, specs))

    assert peak == 1
    assert len(results) == len(specs)
    for png in results:
        assert_valid_chart_png(png)
