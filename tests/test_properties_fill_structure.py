"""Property test for Fill structure and gap accounting (task 6.7, Req 10.1, 10.6, 10.7).

Values are compared bit-for-bit (``uint64`` views) so a source value that is rewritten with
a different sign of zero or a rounded neighbour counts as changed. Datasets use the full
finite float range, since ``fill`` must not alter any non-missing value regardless of size.
"""

from dataclasses import fields

import numpy as np
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from friday.data.dataset import Dataset
from friday.data.fill import FILL_METHODS, FillMethod, fill
from strategies import datasets


def _snapshot(ds: Dataset) -> dict[str, object]:
    """Every field of ``ds``, with ``values`` as raw bytes so NaNs compare equal."""
    snap: dict[str, object] = {f.name: getattr(ds, f.name) for f in fields(ds)}
    snap["values"] = ds.values.tobytes()
    return snap


@settings(max_examples=200)
@given(
    source=datasets(),
    method=st.sampled_from(FILL_METHODS),
    new_id=st.text("abcdefghijklmnopqrstuvwxyz0123456789-", min_size=1, max_size=12),
)
def test_fill_preserves_structure_and_accounts_for_every_gap(
    source: Dataset, method: FillMethod, new_id: str
) -> None:
    """Feature: friday-voice-data-assistant, Property 14: Fill preserves structure and
    accounts for every gap.

    For any Dataset and Fill_Method, the filled Dataset has the same row count, the same
    dates in the same order, and the same ``value_column``. Every non-missing source value
    is unchanged. ``filled + unfilled`` equals the source's Missing count, and ``unfilled``
    equals the result's Missing count. The result is a new Dataset (new ID, derived from
    the source, Fill_Method recorded) and the source Dataset is left unchanged.

    **Validates: Requirements 10.1, 10.6, 10.7**
    """
    assume(new_id != source.dataset_id)
    before = _snapshot(source)

    result, filled, unfilled = fill(source, method, dataset_id=new_id)

    # Structure (Req 10.1, 10.7).
    assert result is not source
    assert result.row_count == source.row_count
    assert result.dates == source.dates
    assert result.value_column == source.value_column
    assert result.indicator == source.indicator
    assert result.series_id == source.series_id
    assert result.transformation == source.transformation
    assert result.dataset_id == new_id
    assert result.derived_from == source.dataset_id
    assert result.fill_method == method
    assert not result.values.flags.writeable

    # Every non-missing source value is unchanged, bit-for-bit (Req 10.7).
    present = ~np.isnan(source.values)
    np.testing.assert_array_equal(
        result.values.view(np.uint64)[present], source.values.view(np.uint64)[present]
    )

    # Gap accounting (Req 10.6).
    assert filled >= 0
    assert unfilled >= 0
    assert filled + unfilled == source.missing_count
    assert unfilled == result.missing_count

    # Source Dataset and its ID unchanged (Req 10.1).
    assert _snapshot(source) == before
