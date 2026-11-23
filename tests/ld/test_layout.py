"""The record of the chromosome ranges in `variants.attrs` (ld-store §2): its keys are layout's."""

from __future__ import annotations

import pytest

from opengwasdb.ld.store.layout import chromosomes_listed, chromosomes_record


def test_the_record_lists_each_range_with_the_spec_keys_in_order() -> None:
    record = chromosomes_record({"1": (0, 2), "22": (2, 9)})
    assert record == [
        {"name": "1", "row_start": 0, "row_stop": 2},
        {"name": "22", "row_start": 2, "row_stop": 9},
    ]
    assert chromosomes_listed(record) == [("1", 0, 2), ("22", 2, 9)]


def test_a_repeated_chromosome_is_kept_for_a_rule_to_see() -> None:
    record = [
        {"name": "1", "row_start": 0, "row_stop": 2},
        {"name": "1", "row_start": 2, "row_stop": 3},
    ]
    assert chromosomes_listed(record) == [("1", 0, 2), ("1", 2, 3)]


@pytest.mark.parametrize(
    "record",
    [
        None,
        {"name": "1"},
        [{"name": "1", "row_start": 0}],
        [{"name": "1", "row_start": 0, "row_stop": 2, "extra": 1}],
        [{"name": 1, "row_start": 0, "row_stop": 2}],
        [{"name": "1", "row_start": 0.0, "row_stop": 2}],
        [{"name": "1", "row_start": True, "row_stop": 2}],
        [{"name": "1", "row_start": 3, "row_stop": 2}],
        [{"name": "1", "row_start": -1, "row_stop": 2}],
        ["1"],
    ],
)
def test_a_malformed_record_is_refused(record: object) -> None:
    with pytest.raises(ValueError, match="chromosomes"):
        chromosomes_listed(record)
