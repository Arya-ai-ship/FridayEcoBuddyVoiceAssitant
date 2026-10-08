"""Tests for the ``ports`` Protocols: structural conformance, defaults, and data shape."""

import inspect
from collections.abc import Callable, Sequence
from datetime import date

import numpy as np
import pytest

from friday.constants import STT_SAMPLE_RATE_HZ, STT_TIMEOUT_S, TTS_TIMEOUT_S
from friday.data.fetch import Observation, build_dataset
from friday.data.indicators import IndicatorEntry
from friday.ports import FredSource, STTClient, TTSClient


class _TTS:
    async def synthesize(self, text: str, timeout_s: float = TTS_TIMEOUT_S) -> bytes:
        return text.encode()


class _STT:
    async def transcribe(
        self,
        pcm16: bytes,
        sample_rate: int = STT_SAMPLE_RATE_HZ,
        timeout_s: float = STT_TIMEOUT_S,
    ) -> str:
        return "please pull inflation"


class _Fred:
    async def observations(
        self, series_id: str, start: date | None, end: date | None
    ) -> Sequence[Observation]:
        return [{"date": "2024-02-01", "value": "."}, {"date": "2024-01-01", "value": "3.7"}]


class _Nothing:
    pass


@pytest.mark.parametrize(
    ("instance", "port"),
    [(_TTS(), TTSClient), (_STT(), STTClient), (_Fred(), FredSource)],
)
def test_conforming_classes_satisfy_their_port(instance: object, port: type) -> None:
    assert isinstance(instance, port)
    assert not isinstance(_Nothing(), port)


def test_ports_do_not_cross_match() -> None:
    assert not isinstance(_TTS(), STTClient)
    assert not isinstance(_STT(), FredSource)
    assert not isinstance(_Fred(), TTSClient)


@pytest.mark.parametrize(
    ("method", "defaults"),
    [
        (TTSClient.synthesize, {"timeout_s": TTS_TIMEOUT_S}),
        (STTClient.transcribe, {"sample_rate": STT_SAMPLE_RATE_HZ, "timeout_s": STT_TIMEOUT_S}),
    ],
)
def test_default_limits_come_from_constants(
    method: Callable[..., object], defaults: dict[str, float]
) -> None:
    params = inspect.signature(method).parameters
    assert {name: params[name].default for name in defaults} == defaults


async def test_fred_source_output_feeds_build_dataset() -> None:
    source: FredSource = _Fred()
    rows = await source.observations("UNRATE", None, None)
    entry = IndicatorEntry(name="unemployment rate", aliases=(), series_id="UNRATE")
    dataset = build_dataset(rows, entry, None, None, dataset_id="ds_1")
    assert dataset.dates == (date(2024, 1, 1), date(2024, 2, 1))
    assert dataset.values[0] == 3.7
    assert np.isnan(dataset.values[1])
