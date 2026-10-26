"""Tests for `compare_store_releases.discover_arrays` across both Zarr formats (#312).

The OGS-00009/OGS-00010 comparison sets a format 0.2.0 Store beside a 0.1.0 one;
a walker that only knew `zarr.json` would describe the 0.1.0 Store as holding no
arrays at all, and a structure table built on it would silently be empty.
"""

from __future__ import annotations

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
