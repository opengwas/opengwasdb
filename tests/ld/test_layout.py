"""The record of the chromosome ranges in `variants.attrs` (ld-store §2): its keys are layout's."""

from __future__ import annotations

from ogquery.ld.store.layout import chromosomes_listed

from opengwasdb.ld.store.layout import chromosomes_record


def test_the_record_lists_each_range_with_the_spec_keys_in_order() -> None:
    record = chromosomes_record({"1": (0, 2), "22": (2, 9)})
    assert record == [
        {"name": "1", "row_start": 0, "row_stop": 2},
        {"name": "22", "row_start": 2, "row_stop": 9},
    ]
    assert chromosomes_listed(record) == [("1", 0, 2), ("22", 2, 9)]
