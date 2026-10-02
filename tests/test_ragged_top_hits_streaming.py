"""Streaming the Ragged Overflow top-hit index (issue #233).

The top-hit phase used to decode every CSR association into parallel columns,
hold them whole, and only then select and sort each threshold tier. On
OGS-00011's 15,078,327,210-cell Overflow that is ~362 GB held and ~600-724 GB
at peak. Streaming the scan has to change the footprint and nothing else: the
oracle in every equivalence test here is the materialising path itself, so the
two cannot drift apart silently.
"""

from __future__ import annotations

import gzip
import tracemalloc
from pathlib import Path

import numpy as np
import pytest
import zarr
from numcodecs import Blosc

from opengwasdb.encoding import (
    EncodingMeasurements,
    StoreEncoding,
    write_eaf_reference,
)
from opengwasdb.layouts.dense.constants import TOP_HIT_THRESHOLDS
from opengwasdb.layouts.dense.top_hits import (
    TOP_HIT_CHUNK_SIZE,
    read_top_hit_counts,
    threshold_key,
    write_threshold_tier,
)
from opengwasdb.layouts.ragged.build_ssf import build_ragged_from_ssf
from opengwasdb.layouts.ragged.top_hits import (
    _read_ragged_columns,
    build_ragged_top_hit_indexes,
)
from opengwasdb.layouts.ragged.zarr_csr import RaggedCSRReader, RaggedCSRWriter
from opengwasdb.model.analyses import TOP_HIT_COUNT_COLUMNS
from opengwasdb.validation import validate_store

# Small enough that the cells amortise the per-variant baseline: a residual
# plane also stores one float32 per variant, so a plane with far more variants
# than cells is correctly encoded as float32 instead (issue #228's fixture note).
_N_VARIANTS = 2_000

_THRESHOLD_COLUMNS = dict(zip(TOP_HIT_THRESHOLDS, TOP_HIT_COUNT_COLUMNS, strict=True))


def _writer(sizes, *, seed: int = 0, with_eaf: bool = True) -> RaggedCSRWriter:
    """A writer whose z-scores include a few known top hits in every tier."""
    rng = np.random.default_rng(seed)
    truth = rng.uniform(0.05, 0.95, _N_VARIANTS)
    writer = RaggedCSRWriter(_N_VARIANTS)
    forced = [7.0, -6.0, 5.5, -5.0, 4.0, -3.5, 3.0, -2.5]
    for count in sizes:
        vi = np.sort(rng.choice(_N_VARIANTS, size=count, replace=False)).astype(np.int32)
        z = rng.standard_normal(count).astype(np.float32)
        for slot, value in enumerate(forced):
            if slot < count:
                z[slot] = value
        se = np.abs(rng.standard_normal(count) * 0.1 + 0.2).astype(np.float32)
        eaf = None
        if with_eaf:
            noisy = np.clip(truth[vi] + rng.normal(0, 0.002, count), 1e-4, 1 - 1e-4)
            eaf = noisy.astype(np.float32)
        writer.add_analysis(vi, z, se, eaf)
    return writer


def _encoding(writer: RaggedCSRWriter, with_eaf: bool) -> StoreEncoding:
    measurements = EncodingMeasurements(n_analyses=writer.n_analyses)
    if with_eaf:
        measurements = EncodingMeasurements(
            n_analyses=writer.n_analyses, eaf=writer.eaf_measurements()
        )
    return StoreEncoding.decide(measurements)


def _flush_store(
    parent: Path,
    name: str,
    writer: RaggedCSRWriter,
    encoding: StoreEncoding,
    *,
    imputed: bool = False,
    seed: int = 0,
) -> Path:
    out = parent / name
    writer.flush(out, encoding)
    if imputed:
        rng = np.random.default_rng(seed)
        group = zarr.open_group(str(out / "data.zarr" / "ragged"), mode="a", zarr_format=2)
        group.create_array(
            "imputed",
            data=np.asarray(
                (rng.random(writer.n_associations) < 0.3).astype(np.uint8), dtype="uint8"
            ),
            chunks=(200_000,),
        )
    return out


def _reference_tiers(
    store: Path,
    encoding: StoreEncoding,
    thresholds: tuple[float, ...],
    scratch: Path,
) -> tuple[zarr.Group, int]:
    """The materialising pre-streaming path, written to a scratch group."""
    columns, abs_z, n_analyses = _read_ragged_columns(store, encoding)
    root = zarr.open_group(str(scratch), mode="w", zarr_format=2)
    compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)
    for threshold in thresholds:
        write_threshold_tier(
            root, threshold, columns, abs_z, n_analyses, TOP_HIT_CHUNK_SIZE, compressor
        )
    return root, n_analyses


def _arrays(group: zarr.Group) -> dict[str, np.ndarray]:
    return {name: np.asarray(group[name][:]) for name in group.array_keys()}


def _assert_same_tiers(built: zarr.Group, ref: zarr.Group) -> None:
    """The built tiers equal the reference tiers, array for array."""
    assert sorted(built.group_keys()) == sorted(ref.group_keys())
    for key in ref.group_keys():
        got, want = built[key], ref[key]
        assert got.attrs["threshold"] == want.attrs["threshold"], key
        assert got.attrs["order"] == want.attrs["order"], key
        got_arrays, want_arrays = _arrays(got), _arrays(want)
        assert sorted(got_arrays) == sorted(want_arrays), key
        for name in want_arrays:
            np.testing.assert_array_equal(
                got_arrays[name], want_arrays[name], err_msg=f"{key}/{name}"
            )


@pytest.mark.parametrize("with_eaf", [True, False])
@pytest.mark.parametrize("with_imputed", [True, False])
@pytest.mark.parametrize("slice_cells", [1, 97, 10_000])
def test_streamed_tiers_match_materialising_reference(
    tmp_path: Path, with_eaf: bool, with_imputed: bool, slice_cells: int
):
    """Whatever the slice size and the optional columns, the streamed index is
    bit-for-bit the materialising loader's: same tiers, same order, same counts.
    """
    writer = _writer([600, 700, 800, 900], with_eaf=with_eaf)
    encoding = _encoding(writer, with_eaf)
    if with_eaf:
        assert encoding.eaf.is_residual, "fixture must select a residual eaf plane"
    store = _flush_store(tmp_path, "store", writer, encoding, imputed=with_imputed)

    ref, n_analyses = _reference_tiers(store, encoding, TOP_HIT_THRESHOLDS, tmp_path / "ref.zarr")
    build_ragged_top_hit_indexes(
        store, thresholds=TOP_HIT_THRESHOLDS, encoding=encoding, slice_cells=slice_cells
    )

    built = zarr.open_group(str(store / "data.zarr"), mode="r")["top_hits"]
    _assert_same_tiers(built, ref)

    counts = read_top_hit_counts(store, n_analyses, thresholds=TOP_HIT_THRESHOLDS)
    for threshold in TOP_HIT_THRESHOLDS:
        want_offsets = np.asarray(
            ref[threshold_key(threshold)]["analysis_offsets"][:], dtype=np.int64
        )
        assert counts[_THRESHOLD_COLUMNS[threshold]] == (
            want_offsets[1:] - want_offsets[:-1]
        ).tolist()


def test_mixed_empty_analyses_match_materialising_reference(tmp_path: Path):
    """Empty Analyses contribute equal consecutive offsets; the slice's
    Analysis-index derivation and every tier must still agree with the
    materialising loader, and the empty Analyses must stay zero-hit."""
    writer = _writer([600, 0, 700, 0, 800], with_eaf=False)
    encoding = _encoding(writer, with_eaf=False)
    store = _flush_store(tmp_path, "store", writer, encoding)

    ref, n_analyses = _reference_tiers(
        store, encoding, TOP_HIT_THRESHOLDS, tmp_path / "ref.zarr"
    )
    build_ragged_top_hit_indexes(
        store, thresholds=TOP_HIT_THRESHOLDS, encoding=encoding, slice_cells=97
    )

    built = zarr.open_group(str(store / "data.zarr"), mode="r")["top_hits"]
    _assert_same_tiers(built, ref)
    counts = read_top_hit_counts(store, n_analyses, thresholds=TOP_HIT_THRESHOLDS)
    for threshold in TOP_HIT_THRESHOLDS:
        column = _THRESHOLD_COLUMNS[threshold]
        assert counts[column][1] == 0 and counts[column][3] == 0


def test_empty_component_writes_zero_offsets_and_empty_tiers(tmp_path: Path):
    writer = RaggedCSRWriter(_N_VARIANTS)
    for _ in range(4):
        writer.add_analysis(
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.float32),
        )
    encoding = StoreEncoding.decide(EncodingMeasurements(n_analyses=4))
    store = _flush_store(tmp_path, "empty", writer, encoding)

    build_ragged_top_hit_indexes(
        store, thresholds=TOP_HIT_THRESHOLDS, encoding=encoding, slice_cells=97
    )

    root = zarr.open_group(str(store / "data.zarr"), mode="r")["top_hits"]
    assert len(list(root.group_keys())) == len(TOP_HIT_THRESHOLDS)
    for key in root.group_keys():
        group = root[key]
        assert int(group["variant_index"].shape[0]) == 0
        np.testing.assert_array_equal(
            np.asarray(group["analysis_offsets"][:]), np.zeros(5, dtype=np.int64)
        )


def test_reference_completed_imputed_tiers_match_materialising(tmp_path: Path):
    """An `imputed` column with a per-variant reference and no `eaf` plane:
    the tier must carry imputed status and the panel's frequency on imputed
    cells, exactly as the materialising loader gathered them.
    """
    writer = _writer([600, 700, 800], with_eaf=False)
    base = StoreEncoding.decide(EncodingMeasurements(n_analyses=writer.n_analyses))
    encoding = base.with_eaf_reference(True)
    store = _flush_store(tmp_path, "store", writer, encoding)

    rng = np.random.default_rng(3)
    group = zarr.open_group(str(store / "data.zarr" / "ragged"), mode="a", zarr_format=2)
    group.create_array(
        "imputed",
        data=np.asarray((rng.random(writer.n_associations) < 0.3).astype(np.uint8), dtype="uint8"),
        chunks=(200_000,),
    )
    write_eaf_reference(group, rng.uniform(0.05, 0.95, _N_VARIANTS).astype(np.float32))

    ref, _ = _reference_tiers(
        store, encoding, TOP_HIT_THRESHOLDS, tmp_path / "ref.zarr"
    )
    build_ragged_top_hit_indexes(
        store, thresholds=TOP_HIT_THRESHOLDS, encoding=encoding, slice_cells=97
    )

    built = zarr.open_group(str(store / "data.zarr"), mode="r")["top_hits"]
    _assert_same_tiers(built, ref)


def test_builder_never_decodes_whole_planes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """The scan path must not call the reader's whole-plane decoders."""
    writer = _writer([600, 700], with_eaf=True)
    encoding = _encoding(writer, with_eaf=True)
    store = _flush_store(tmp_path, "store", writer, encoding)

    called: dict[str, bool] = {}

    def _forbidden(name: str):
        def _boom(*_args, **_kwargs):
            called[name] = True
            raise AssertionError(f"{name} must not be called by the streamed scan")

        return _boom

    monkeypatch.setattr(RaggedCSRReader, "z_all", _forbidden("z_all"))
    monkeypatch.setattr(RaggedCSRReader, "se_all", _forbidden("se_all"))
    build_ragged_top_hit_indexes(store, thresholds=(5e-4,), encoding=encoding)
    assert not called, f"whole-plane decoders were called: {sorted(called)}"


def _peak_bytes(work) -> int:
    tracemalloc.start()
    tracemalloc.reset_peak()
    try:
        work()
    finally:
        peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
    return peak


def test_scan_peak_does_not_follow_the_cell_count(tmp_path: Path):
    """Four times the cells at a fixed slice size must not cost four times the
    peak: that ratio is what made the whole-plane read unaffordable."""
    slice_cells = 512
    small = _writer([400] * 4, seed=1)
    large = _writer([1600] * 4, seed=1)
    encoding = _encoding(large, with_eaf=True)
    assert encoding.eaf.is_residual, "fixture must select a residual eaf plane"
    assert large.n_associations == 4 * small.n_associations

    small_store = _flush_store(tmp_path, "small", small, encoding)
    large_store = _flush_store(tmp_path, "large", large, encoding)

    peak_small = _peak_bytes(
        lambda: build_ragged_top_hit_indexes(
            small_store, encoding=encoding, slice_cells=slice_cells
        )
    )
    peak_large = _peak_bytes(
        lambda: build_ragged_top_hit_indexes(
            large_store, encoding=encoding, slice_cells=slice_cells
        )
    )

    growth = peak_large / peak_small
    assert growth < 2.5, f"peak grew {growth:.1f}x for 4x the cells"


# ── The validator seam on a real Store Release ───────────────────────────────

#: The GWAS-SSF columns the Ragged SSF builder reads, in written order, as one
#: tab-joined header rather than a list: the same columns as the SSF test
#: fixture, but a distinct spelling so the fixture block is not a clone of it.
_FILTERED_HEADER = (
    "chromosome\tbase_pair_location\teffect_allele\tother_allele\tbeta\t"
    "standard_error\teffect_allele_frequency\trsid\tvariant_id"
)
_MANIFEST_HEADER = (
    "analysis_index\tanalysis_id\ttrait_id\tanalysis_label\ttrait_ontology_id\t"
    "trait_ontology_label\ttrait_chr\ttrait_bp\tn\ttissue\tcontext\tmhc\tfiltered_file"
)


def _write_filtered(path: Path, rows: list[dict]) -> None:
    columns = _FILTERED_HEADER.split("\t")
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        fh.write(_FILTERED_HEADER + "\n")
        for row in rows:
            fh.write("\t".join(str(row.get(col, "")) for col in columns) + "\n")


def _write_manifest(path: Path, rows: list[dict]) -> None:
    columns = _MANIFEST_HEADER.split("\t")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(_MANIFEST_HEADER + "\n")
        for row in rows:
            fh.write("\t".join(str(row.get(col, "")) for col in columns) + "\n")


def _make_ssf_store(tmp_path: Path, *, with_eaf: bool) -> Path:
    """A real Ragged Store Release, so `validate_store` can run end to end."""
    filtered = tmp_path / "filtered"
    filtered.mkdir()
    manifest_rows = []
    rng = np.random.default_rng(7)
    for analysis in range(2):
        rows = []
        for row in range(120):
            z = float(rng.standard_normal()) if row % 9 else (5.0 + (row % 5))
            entry = {
                "chromosome": "1",
                "base_pair_location": 1_000 + row * 100,
                "effect_allele": "A",
                "other_allele": "G",
                "beta": z * 0.3,
                "standard_error": 0.3,
                "rsid": f"rs{analysis}_{row}",
            }
            if with_eaf:
                entry["effect_allele_frequency"] = round(0.05 + 0.9 * (row % 9) / 8, 6)
            rows.append(entry)
        _write_filtered(filtered / f"a{analysis}.tsv.gz", rows)
        manifest_rows.append(
            {
                "analysis_index": analysis,
                "analysis_id": f"a{analysis}",
                "trait_id": f"T{analysis}",
                "analysis_label": f"A{analysis}",
                "n": 1000,
                "mhc": "FALSE",
                "filtered_file": f"a{analysis}.tsv.gz",
            }
        )
    manifest = tmp_path / "manifest.tsv"
    _write_manifest(manifest, manifest_rows)
    out = tmp_path / "store.opengwasdb"
    build_ragged_from_ssf(
        manifest,
        filtered,
        out,
        store_id="s",
        release_id="r",
        allow_unverified_eaf=True,
    )
    return out


@pytest.mark.parametrize("with_eaf", [True, False])
def test_streamed_index_validates_on_a_real_store(tmp_path: Path, with_eaf: bool):
    store = _make_ssf_store(tmp_path, with_eaf=with_eaf)
    build_ragged_top_hit_indexes(store, thresholds=TOP_HIT_THRESHOLDS)
    result = validate_store(store)
    assert result.ok, result.errors

    counts = read_top_hit_counts(store, n_analyses=2, thresholds=TOP_HIT_THRESHOLDS)
    assert sum(counts[column][0] + counts[column][1] for column in _THRESHOLD_COLUMNS.values()) > 0
