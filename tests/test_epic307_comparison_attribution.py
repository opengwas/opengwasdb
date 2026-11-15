"""A rejected cell is two logged sources, not two lost cells (#357)."""

from __future__ import annotations

import gzip
import json

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


def test_attribution_reads_the_producers_named_codes(tmp_path, monkeypatch):
    import numpy as np
    from dense_sources import build_dense_from_records

    from benchmarks.attribute_epic307_comparison import attribute
    from benchmarks.compare_store_releases import ROW_CODES
    from opengwasdb.layouts.dense.build import NormalisedAssociation
    from opengwasdb.store.open import open_store
    from opengwasdb.variants import CanonicalVariant

    store = tmp_path / "store.opengwasdb"
    build_dense_from_records(
        [
            NormalisedAssociation(
                analysis_id="a0",
                variant=CanonicalVariant("7", 1, "A", "G"),
                z=2.0,
                se=0.125,
                eaf=0.25,
            )
        ],
        store,
        store_id="codes",
        release_id="codes",
        reference_assembly="GRCh38",
    )
    root = open_store(store).arrays(mode="r+")
    if "eaf_baseline" not in root:
        root.create_array("eaf_baseline", data=np.array([0.25], dtype=np.float32))
    dropped = tmp_path / "OGS-00009/store.opengwasdb.admission/dropped"
    dropped.mkdir(parents=True)
    _dropped(dropped / "0.tsv.gz", 2)
    base = tmp_path / "offclock/compare/observed-vs-published"
    base.parent.mkdir(parents=True)
    base.with_suffix(".json").write_text(
        json.dumps(
            {
                "left": str(store),
                "right": str(store),
                "respelled": {"rows": 0},
                "matched_by_alid": {
                    "planes": {
                        "z": {"only_right": 1, "examples": [{"left_row": 0, "cells_in_row": 1}]},
                        "se": {"se_log_bound": 0.004, "max_abs_log_ratio": 0.001},
                        "eaf": {"value_differs": 1},
                    }
                },
            }
        )
    )
    base.with_suffix(".unmatched.tsv").write_text(
        "kind\tchromosome\tsource_alid\nright_only\t7_alt\t7:2:A:C\n"
    )
    monkeypatch.setitem(ROW_CODES, "presence", 64)
    monkeypatch.setitem(ROW_CODES, "eaf_value", 128)
    np.savez(
        base.with_suffix(".codes.npz"),
        matched_left_rows=np.array([0]),
        matched_right_rows=np.array([0]),
        matched_codes=np.array([192], dtype=np.uint8),
    )

    result = attribute(tmp_path)

    assert result["duplicate_conflicting_357"]["matches_presence_differences"]
    assert [row["left_row"] for row in result["eaf_after_conflict_drops"]["rows"]] == [0]


def test_an_empty_admission_record_does_not_confirm_the_explanation(tmp_path):
    _dropped(tmp_path / "0.tsv.gz", 0)
    with pytest.raises(ValueError, match="no conflicting source pairs"):
        source_conflicts(tmp_path)
