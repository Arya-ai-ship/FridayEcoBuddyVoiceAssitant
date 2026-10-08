"""Property test for the CSV_Export round trip (task 4.3, Req 7.5, 7.6)."""

import numpy as np
from hypothesis import given, settings

from friday.data.dataset import DATE_COLUMN, Dataset, parse_csv, to_csv
from strategies import datasets


@settings(max_examples=200)
@given(datasets())
def test_csv_export_round_trips(ds: Dataset) -> None:
    """Feature: friday-voice-data-assistant, Property 9: CSV export round-trips.

    For any Dataset, including Missing values, extreme floats, and values with many
    significant digits, ``parse_csv(to_csv(ds))`` returns the same column names, row count,
    dates, and values. Values are compared bit-for-bit, and Missing positions match. The
    header row is ``date,<value_column>``.

    **Validates: Requirements 7.5, 7.6**
    """
    text = to_csv(ds)
    table = parse_csv(text)

    assert table.columns == (DATE_COLUMN, ds.value_column)
    assert len(text.splitlines()) == ds.row_count + 1  # header plus one line per row
    assert len(table.dates) == ds.row_count
    assert table.values.shape == (ds.row_count,)
    assert table.dates == ds.dates

    missing = np.isnan(ds.values)
    np.testing.assert_array_equal(np.isnan(table.values), missing)
    # Bit-for-bit on non-missing values: distinguishes -0.0 from 0.0 and catches any rounding.
    original_bits = ds.values.view(np.uint64)[~missing]
    parsed_bits = table.values.view(np.uint64)[~missing]
    np.testing.assert_array_equal(parsed_bits, original_bits)
