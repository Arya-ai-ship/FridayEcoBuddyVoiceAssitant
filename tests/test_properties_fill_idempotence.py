"""Property test for Fill idempotence (task 6.8, Req 10.8, 10.11).

Values are compared bit-for-bit: NaN (Missing) positions must match, and every non-missing
value must have the same IEEE-754 bit pattern (so ``-0.0`` vs ``0.0`` or a one-ulp drift
counts as a change).
"""

import math
from dataclasses import replace

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from friday.data.dataset import Dataset, FloatArray
from friday.data.fill import FILL_METHODS, FillMethod, fill
from strategies import datasets, finite_floats


@st.composite
def complete_datasets(draw: st.DrawFn) -> Dataset:
    """A ``datasets()`` draw with every Missing_Value replaced by a finite float."""
    ds = draw(datasets())
    values = [draw(finite_floats()) if math.isnan(v) else v for v in ds.values.tolist()]
    return replace(ds, values=np.array(values, dtype=np.float64))


def assert_same_values(actual: FloatArray, expected: FloatArray) -> None:
    """``actual`` and ``expected`` agree on Missing positions and bit-for-bit elsewhere."""
    missing = np.isnan(expected)
    np.testing.assert_array_equal(np.isnan(actual), missing)
    np.testing.assert_array_equal(
        actual.view(np.uint64)[~missing], expected.view(np.uint64)[~missing]
    )


@settings(max_examples=200)
@given(ds=datasets() | complete_datasets(), method=st.sampled_from(FILL_METHODS))
def test_fill_is_idempotent(ds: Dataset, method: FillMethod) -> None:
    """Feature: friday-voice-data-assistant, Property 15: Fill is idempotent.

    For any Dataset and Fill_Method, applying ``fill`` to ``fill(ds, m)`` with the same
    method gives identical dates and values and reports ``filled == 0``. In particular, a
    Dataset with no Missing values is returned unchanged with ``filled == unfilled == 0``.

    **Validates: Requirements 10.8, 10.11**
    """
    once = fill(ds, method, dataset_id="ds-once")
    twice = fill(once.dataset, method, dataset_id="ds-twice")

    assert twice.filled == 0
    assert twice.dataset.dates == once.dataset.dates
    assert_same_values(twice.dataset.values, once.dataset.values)

    if ds.missing_count == 0:
        assert once.filled == 0
        assert once.unfilled == 0
        assert once.dataset.dates == ds.dates
        assert_same_values(once.dataset.values, ds.values)
