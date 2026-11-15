"""A rejected cell is two logged sources, not two lost cells (#357)."""

from __future__ import annotations

import gzip

import pytest

from benchmarks.attribute_epic307_comparison import source_conflicts


def _dropped(path, count):
    with gzip.open(path, "wt") as stream:
        stream.write("source_key\treason\n")
        stream.write("7:1:A:G\tduplicate_conflicting\n" * count)
        stream.write("7:2:A:C\tlift_off_primary\n")


def test_conflicting_pairs_are_counted_per_analysis_not_per_source(tmp_path):
    _dropped(tmp_path / "0.tsv.gz", 2)
    _dropped(tmp_path / "1.tsv.gz", 2)
    assert source_conflicts(tmp_path) == {"7:1:A:G": 2}


def test_a_triplet_is_not_silently_treated_as_one_and_a_half_cells(tmp_path):
    _dropped(tmp_path / "0.tsv.gz", 3)
    with pytest.raises(ValueError, match="not a pair"):
        source_conflicts(tmp_path)


def test_an_empty_admission_record_does_not_confirm_the_explanation(tmp_path):
    _dropped(tmp_path / "0.tsv.gz", 0)
    with pytest.raises(ValueError, match="no conflicting source pairs"):
        source_conflicts(tmp_path)
