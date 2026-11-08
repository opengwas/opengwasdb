"""A Dense build's inline Top-Hit Index equals a standalone rebuild (issue #312).

`build-dense-vcf` harvests the index's candidate cells while it writes the ``z``
plane, and hands them to `write_top_hit_indexes_for_store`. `build-dense-top-hits`
re-derives the same index afterwards, by scanning the stored matrix, and the
Store Release Build runs it after every Dense build. The two must agree tier by
tier, array by array and in the index's own attributes, or that step is not
repetition and cannot be dropped (opengwas/opengwasdb-stores#357).

The fixture is a real `build-dense-vcf` release whose hits fall in every tier and
in three row bands of both paths -- three row chunks of the inline gathers, three
bands of the narrowed scan below -- over residual-coded ``se`` and ``eaf`` planes,
the layout where the inline side holds the source's own values and the scan side
only ever sees the plane.
"""
from __future__ import annotations

import logging
import shutil
from pathlib import Path

import numpy as np
from residual_fixtures import (
    MAF_VCF_N_VARIANTS,
    maf_vcf_rows,
    write_gwas_vcf_with_eaf,
    write_hg38_manifest,
)

from benchmarks.compare_store_releases import compare_values, discover_arrays
from opengwasdb.encoding.banding import BYTES_PER_CELL
from opengwasdb.layouts.dense.build_vcf import build_dense_from_vcf_manifest
from opengwasdb.layouts.dense.top_hits import build_top_hit_indexes
from opengwasdb.model.manifest import StoreManifest
from opengwasdb.store.open import open_store

ANALYSES = tuple(f"t{column}" for column in range(7))

TIER_KEYS = ("p_5e_04", "p_5e_06", "p_5e_08")

#: The variant rows that carry a hit, one per row chunk of `z` at the
#: `chunk_shape` below (100 rows), and one per band of `SCAN_BAND_ROWS`.
TIER_ROWS = (0, 250, 550)

#: The z magnitude each of those rows carries, and the step across Analyses so
#: the hit arrays are not the same value repeated. The three magnitudes fall in
#: the loosest tier only, the middle one and looser, and all three; every one
#: sits clear of its tier's |z| cutoff (3.481 / 4.565 / 5.451) by more than the
#: stored plane's fixed-point quantisation, so no cell can cross a boundary.
TIER_MAGNITUDES = {0: 6.0, 250: 4.7, 550: 3.6}
TIER_STEPS = {0: 0.05, 250: 0.02, 550: 0.04}

#: Rows per band for the narrowed standalone scan: several bands over this
#: fixture, where the default one-GiB budget would give the old single band. It
#: is the budget that is narrowed, so the band is the one `build_top_hit_indexes`
#: sizes for itself; `band_rows` rounds it to whole rows chunks of `z`.
SCAN_BAND_ROWS = 200
SCAN_WORKING_BYTES = SCAN_BAND_ROWS * BYTES_PER_CELL * len(ANALYSES)


def _designed_z(row: int, column: int) -> float:
    """The z a fixture cell holds: a designed hit at its tier row, else null."""
    magnitude = TIER_MAGNITUDES.get(row, 1.0) + column * TIER_STEPS.get(row, 0.0)
    return magnitude if column % 2 == 0 else -magnitude


def _build_inline_store(directory: Path) -> Path:
    """A Dense Store Release built by `build-dense-vcf`, so its index is inline."""
    directory.mkdir(parents=True, exist_ok=True)
    vcfs = {
        analysis_id: write_gwas_vcf_with_eaf(
            directory / f"{analysis_id}.vcf", maf_vcf_rows(column, _designed_z)
        )
        for column, analysis_id in enumerate(ANALYSES)
    }
    store = directory / "inline.opengwasdb"
    build_dense_from_vcf_manifest(
        write_hg38_manifest(directory / "manifest.tsv", vcfs),
        store,
        store_id="inline-vs-standalone",
        release_id="r1",
        chunk_shape=(100, 2),
    )
    return store


def _tier_arrays(root: Path) -> dict[str, Path]:
    """Every Top-Hit Index array of one release, by its path within `root`."""
    prefix = "data.zarr/top_hits/"
    return {
        relative: root / relative
        for relative in discover_arrays(root)
        if relative.startswith(prefix)
    }


def _bands(values: np.ndarray, band_size: int) -> set[int]:
    """Which bands of `band_size` the hit positions at `values` fall in."""
    return {int(value) // band_size for value in values}


def _messages(caplog) -> list[str]:
    return [record.getMessage() for record in caplog.records]


def _assert_index_matches(inline: Path, rebuilt: Path) -> None:
    """Every tier array and the index's own attributes, both sides."""
    inline_arrays = _tier_arrays(inline)
    rebuilt_arrays = _tier_arrays(rebuilt)
    assert inline_arrays, "fixture is meaningful only if the store has an index"
    assert set(inline_arrays) == set(rebuilt_arrays), (
        f"only inline: {sorted(set(inline_arrays) - set(rebuilt_arrays))}; "
        f"only rebuilt: {sorted(set(rebuilt_arrays) - set(inline_arrays))}"
    )
    for relative in sorted(inline_arrays):
        report = compare_values(inline_arrays[relative], rebuilt_arrays[relative])
        assert report["comparable"], (relative, report)
        assert report["differing_cells"] == 0, (relative, report)

    inline_root = open_store(inline).arrays(mode="r")
    rebuilt_root = open_store(rebuilt).arrays(mode="r")
    assert dict(inline_root["top_hits"].attrs) == dict(rebuilt_root["top_hits"].attrs)
    assert "thresholds" in inline_root["top_hits"].attrs
    for key in TIER_KEYS:
        attrs = dict(inline_root[f"top_hits/{key}"].attrs)
        assert attrs == dict(rebuilt_root[f"top_hits/{key}"].attrs), key
        assert "threshold" in attrs, (key, attrs)
        assert attrs.get("order") == "analysis_index,variant_index", (key, attrs)


def test_a_dense_builds_inline_top_hit_index_equals_a_standalone_rebuild(
    tmp_path, caplog
) -> None:
    with caplog.at_level(logging.INFO, logger="opengwasdb.layouts.dense.top_hits"):
        inline = _build_inline_store(tmp_path)
    inline_messages = _messages(caplog)
    assert any(
        message.startswith("Collecting top-hit EAF in variant-row order")
        for message in inline_messages
    ), "the build must have written the index from its own harvest"
    assert not any("Top-hit index: scanning" in message for message in inline_messages)

    root = open_store(inline).arrays(mode="r")
    counts = {key: int(root[f"top_hits/{key}/variant_index"].shape[0]) for key in TIER_KEYS}
    assert all(counts.values()), f"every tier needs hits, got {counts}"
    assert counts["p_5e_08"] < counts["p_5e_06"] < counts["p_5e_04"], (
        f"the hits must span every tier, got {counts}"
    )
    assert all("eaf" in root[f"top_hits/{key}"] for key in TIER_KEYS), (
        "every tier must carry a frequency, so the frequency gather is compared too"
    )
    encoding = StoreManifest.load(inline).encoding
    assert encoding.se.is_residual, "the fixture must make the inline side re-read se"
    assert encoding.eaf.kind == "int8_residual", encoding.eaf.to_manifest()

    loosest = root["top_hits/p_5e_04"]
    hit_rows = np.asarray(loosest["variant_index"][:])
    hit_columns = np.asarray(loosest["analysis_index"][:])
    chunk_rows, column_band = int(root["z"].chunks[0]), int(root["z"].shards[1])
    assert len(_bands(hit_rows, chunk_rows)) > 1, (
        f"hits must fall in more than one row chunk of {chunk_rows} rows"
    )
    assert len(_bands(hit_rows, SCAN_BAND_ROWS)) > 1, (
        f"hits must fall in more than one scan band of {SCAN_BAND_ROWS} rows"
    )
    assert set(hit_columns.tolist()) == set(range(len(ANALYSES))), (
        "every Analysis must contribute a hit"
    )
    # The inline harvest writes a column band per `z` shard, and the decided
    # shard spans the whole Analysis axis below 1025 Analyses (ADR 0058), so this
    # one-band fact is read from the store rather than assumed. A release with
    # more Analyses than that is where the harvest spans bands -- OGS-00009 and
    # the other full-size Dense releases do -- and its fixture costs about 150 s.
    assert column_band >= len(ANALYSES), (column_band, len(ANALYSES))
    assert len(_bands(hit_columns, column_band)) == 1, column_band

    standalone = Path(shutil.copytree(inline, tmp_path / "standalone.opengwasdb"))
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="opengwasdb.layouts.dense.top_hits"):
        build_top_hit_indexes(standalone)
    assert any(
        message.startswith("Top-hit index: scanning") for message in _messages(caplog)
    ), "the rebuild must have taken the scan path, not the harvest"
    _assert_index_matches(inline, standalone)

    narrow = Path(shutil.copytree(inline, tmp_path / "narrow.opengwasdb"))
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="opengwasdb.layouts.dense.top_hits"):
        build_top_hit_indexes(narrow, working_bytes=SCAN_WORKING_BYTES)
    assert any(
        f"{SCAN_BAND_ROWS} rows per band" in message for message in _messages(caplog)
    ), "the budget must be what narrows the scan's band"
    bands = -(-MAF_VCF_N_VARIANTS // SCAN_BAND_ROWS)
    assert any(
        message.startswith(f"Top-hit scan: {bands}/{bands}") for message in _messages(caplog)
    ), f"the rebuild must have scanned in {bands} bands of {SCAN_BAND_ROWS} rows"
    _assert_index_matches(inline, narrow)
