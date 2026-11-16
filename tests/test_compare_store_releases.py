"""Tests for `compare_store_releases.discover_arrays` across both Zarr formats (#312).

The OGS-00009/OGS-00010 comparison sets a format 0.2.0 Store beside a 0.1.0 one;
a walker that only knew `zarr.json` would describe the 0.1.0 Store as holding no
arrays at all, and a structure table built on it would silently be empty.
"""

from __future__ import annotations

import pytest
import zarr

from benchmarks.compare_store_releases import discover_arrays


def test_discover_arrays_reads_a_v2_and_a_sharded_v3_array(tmp_path):
    zarr.create_array(
        store=tmp_path / "v2" / "data.zarr" / "z", shape=(20, 10), chunks=(10, 5),
        dtype="int16", zarr_format=2,
    )
    zarr.create_array(
        store=tmp_path / "v3" / "data.zarr" / "z", shape=(20, 10), chunks=(10, 5),
        shards=(20, 10), dtype="int16", zarr_format=3,
        compressors=zarr.codecs.BloscCodec(cname="zstd", clevel=3, shuffle="bitshuffle"),
    )

    v2 = discover_arrays(tmp_path / "v2")
    v3 = discover_arrays(tmp_path / "v3")

    assert list(v2) == ["data.zarr/z"], "the v2 array must be found at all"
    assert v2["data.zarr/z"]["zarr_format"] == 2
    assert v2["data.zarr/z"]["inner_chunk"] == [10, 5]
    assert v2["data.zarr/z"]["shard"] is None
    assert v2["data.zarr/z"]["compressor"]["sharding"] is None
    assert v3["data.zarr/z"]["zarr_format"] == 3
    assert v3["data.zarr/z"]["shard"] == [20, 10]
    assert v3["data.zarr/z"]["compressor"]["sharding"] == "sharding_indexed"


# --- the aligned comparison (#322) --------------------------------------------

#: The left release's rows: (position, effect allele, other allele, z by Analysis).
_LEFT_ROWS = (
    (1, "A", "G", (0.5, -0.5)),
    (2, "C", "T", (1.0, 2.0)),
    (3, "AT", "A", (1.5, -2.5)),
    (4, "G", "C", (0.25, 0.75)),
    (5, "T", "A", (-1.0, 3.0)),
)


def _release(directory, rows, *, format_version=None, reference_assembly="GRCh38"):
    import json

    from dense_sources import build_dense_from_records

    from opengwasdb.layouts.dense.build import NormalisedAssociation
    from opengwasdb.variants import CanonicalVariant

    records = [
        NormalisedAssociation(
            analysis_id=f"a{column}",
            variant=CanonicalVariant("1", position, effect, other),
            z=zs[column],
            se=0.125 * (column + 1),
            eaf=eaf,
        )
        for position, effect, other, zs, eaf in rows
        for column in range(2)
    ]
    store = directory / "store.opengwasdb"
    build_dense_from_records(
        records, store, store_id="aligned", release_id="aligned-v1",
        reference_assembly=reference_assembly,
    )
    if format_version is not None:
        # The V3 arrays are shared by 0.2.1 and 0.2.2; these historical rows
        # explicitly use sorted indels, whose semantics the old stamp declares.
        manifest = store / "manifest.json"
        payload = json.loads(manifest.read_text())
        payload["format_version"] = format_version
        manifest.write_text(json.dumps(payload))
    return store


def _alids(store):
    from opengwasdb.variants import VariantAxis

    axis = VariantAxis(store)
    try:
        return [axis.by_index(row).alid for row in range(axis.n_variants)]
    finally:
        axis.close()


@pytest.mark.parametrize("legacy_format", ["0.2.0", "0.2.1"])
def test_the_aligned_comparison_accounts_for_every_row_by_kind(tmp_path, legacy_format):
    """A dropped row, an indel spelled the other way round with its statistics flipped,
    and one changed Z: each is found as what it is, and nothing else differs."""
    from benchmarks.compare_store_releases import ROW_CODES, compare_aligned

    left_rows = [(*row, 0.25) for row in _LEFT_ROWS]
    right_rows = [
        (1, "A", "G", (0.5, -0.5), 0.25),
        (2, "C", "T", (1.0, 4.0), 0.25),
        (3, "A", "AT", (-1.5, 2.5), 0.75),
        (5, "T", "A", (-1.0, 3.0), 0.25),
    ]
    left = _release(tmp_path / "left", left_rows)
    right = _release(tmp_path / "right", right_rows, format_version=legacy_format)
    assert "1:3:AT:A" in _alids(left) and "1:4:G:C" in _alids(left)
    assert "1:3:A:AT" in _alids(right) and "1:3:AT:A" not in _alids(right)
    assert len(_alids(right)) == len(_alids(left)) - 1, "the right release must drop a row"

    report, table, codes = compare_aligned(left, right, n_workers=1, band_rows=2)

    assert report["matched_by_alid"]["rows"] == 3
    assert report["left_only"]["rows"] == 1 and report["right_only"]["rows"] == 0
    assert [(kind, entry["alid"]) for kind, _, _, entry, _, _ in table if kind == "left_only"] == [
        ("left_only", "1:4:G:C")
    ]
    assert report["respelled"]["rows"] == 1 and report["respelled"]["flipped"] == 1
    assert report["respelled"]["rows_differing"] == 0, report["respelled"]
    matched = report["matched_by_alid"]
    assert matched["planes"]["z"]["value_differs"] == 1, matched["planes"]["z"]
    assert matched["planes"]["eaf"]["value_differs"] == 0
    assert matched["planes"]["se"]["value_differs_beyond_codec"] == 0
    assert matched["rows_by_code"] == {"z_value": 1}
    changed = codes["matched_left_rows"][codes["matched_codes"] == ROW_CODES["z_value"]]
    assert _alids(left)[int(changed[0])] == "1:2:C:T"


@pytest.mark.parametrize("reverse", [False, True])
def test_the_aligned_comparison_sees_a_respelled_indel_whose_statistics_did_not_flip(
    tmp_path, reverse
):
    """The pairing's orientation is what makes the flipped indel agree: an indel
    respelled with its statistics left as they were differs in Z and EAF."""
    from benchmarks.compare_store_releases import compare_aligned

    left = _release(tmp_path / "left", [(*row, 0.25) for row in _LEFT_ROWS])
    right_rows = [(*row, 0.25) for row in _LEFT_ROWS]
    right_rows[2] = (3, "A", "AT", (1.5, -2.5), 0.25)
    right = _release(tmp_path / "right", right_rows, format_version="0.2.1")
    if reverse:
        left, right = right, left

    report, _, _ = compare_aligned(left, right, n_workers=1)

    respelled = report["respelled"]
    assert respelled["rows"] == 1 and respelled["flipped"] == 1
    assert respelled["planes"]["z"]["value_differs"] == 2
    assert respelled["planes"]["eaf"]["value_differs"] == 2
    assert respelled["rows_by_code"] == {"z_value+eaf_value": 1}


def test_reference_first_releases_keep_insertion_and_deletion_as_distinct_variants(tmp_path):
    from benchmarks.compare_store_releases import compare_aligned
    from opengwasdb.model.manifest import StoreManifest

    common = [
        (1, "A", "G", (0.5, -0.5), 0.25),
        (3, "AT", "A", (1.5, -2.5), 0.25),
        (3, "A", "AT", (-1.5, 2.5), 0.75),
    ]
    left = _release(tmp_path / "left", [*common, (4, "AT", "A", (1.5, -2.5), 0.25)])
    right = _release(tmp_path / "right", [*common, (4, "A", "AT", (-1.5, 2.5), 0.75)])
    assert StoreManifest.load(left).format_version == "0.2.2"
    assert StoreManifest.load(right).format_version == "0.2.2"
    assert {"1:3:AT:A", "1:3:A:AT"} <= set(_alids(left)) & set(_alids(right))
    assert set(_alids(left)) - set(_alids(right)) == {"1:4:AT:A"}
    assert set(_alids(right)) - set(_alids(left)) == {"1:4:A:AT"}

    report, table, _ = compare_aligned(left, right)

    assert report["matched_by_alid"]["rows"] == 3
    assert report["respelled"]["rows"] == 0
    assert report["left_only"]["rows"] == report["right_only"]["rows"] == 1
    assert {(kind, row["alid"]) for kind, _, _, row, _, _ in table} == {
        ("left_only", "1:4:AT:A"), ("right_only", "1:4:A:AT"),
    }


def test_cross_assembly_alignment_refuses_equal_coordinate_strings(tmp_path):
    from benchmarks.compare_store_releases import compare_aligned
    from opengwasdb.model.manifest import StoreManifest

    rows = [(1, "A", "G", (0.5, -0.5), 0.25)]
    left = _release(tmp_path / "left", rows)
    right = _release(tmp_path / "right", rows, reference_assembly="GRCh37")
    assert _alids(left) == _alids(right) == ["1:1:A:G"]
    assert StoreManifest.load(left).reference_assembly == "GRCh38"
    assert StoreManifest.load(right).reference_assembly == "GRCh37"

    with pytest.raises(SystemExit, match="Reference Assembly.*GRCh38.*GRCh37"):
        compare_aligned(left, right)


def test_an_se_difference_is_judged_against_both_codecs_bound():
    """One source SE coded by two residual models can decode a step apart in log space,
    `float16`-rounded on one side, or further apart: the three are counted apart."""
    import numpy as np

    from benchmarks.compare_store_releases import (
        ROW_CODES,
        _plane_difference,
        se_log_half_step,
    )

    bound = 2 * se_log_half_step({"se": {"kind": "int8_residual", "residual_range": 0.5}})
    assert bound == 0.5 / 127
    source = np.float32(0.0371)
    a = np.array([[source], [source], [source], [source]], dtype=np.float32)
    b = np.array(
        [
            [source],
            [np.float32(source * np.exp(0.99 * bound))],
            [np.float32(source).astype(np.float16).astype(np.float32)],
            [np.float32(source * np.exp(1.5 * bound))],
        ],
        dtype=np.float32,
    )
    assert b[2, 0] != source, "the float16 rounding must change the value"
    rows = np.arange(4)

    record, codes = _plane_difference("se", a, b, rows, rows, bound)

    assert record["value_differs"] == 3
    assert record["value_differs_within_codec"] == 1
    assert record["value_differs_by_float16_rounding"] == 1
    assert record["value_differs_beyond_codec"] == 1
    assert codes.tolist() == [
        0,
        ROW_CODES["se_within_codec"],
        ROW_CODES["se_float16_rounding"],
        ROW_CODES["se_value"],
    ]
