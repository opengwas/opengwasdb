"""Streaming the Ragged Overflow Component's cells (issue #228).

The joint SE fit and the CSR flush both held a whole Overflow plane in memory,
which is ~1.3 TB and ~1.1 TB respectively on OGS-00011's 15,078,327,210 cells.
Streaming them has to change the footprint and nothing else: the oracle in
every equivalence test here is the materialising path itself, so the two cannot
drift apart silently.
"""

from __future__ import annotations

import tracemalloc
from pathlib import Path

import numpy as np
import pytest
import zarr

import opengwasdb.store.arrays as store_arrays
from opengwasdb.encoding import EncodingMeasurements, OverflowCells, StoreEncoding
from opengwasdb.encoding.codec import StoreCodec
from opengwasdb.encoding.plan import EafEncoding, SeEncoding, ZEncoding
from opengwasdb.encoding.se import OverflowCellBatches, optimise_dense_se_joint
from opengwasdb.layouts.ragged.zarr_csr import RaggedCSRWriter

# Small enough that the cells amortise the per-variant baseline: a residual
# plane also stores one float32 per variant, so a plane with far more
# variants than cells is correctly encoded as float32 instead.
_N_VARIANTS = 2_000


def _writer(sizes, *, seed=0, without_eaf=()):
    """A writer holding one Analysis per entry of `sizes`, sorted by variant.

    Each variant has one true frequency that every Analysis reports with a
    little noise, because that is what makes a per-variant baseline worth
    storing: independent frequencies per Analysis have nothing for a residual
    to be small against, and the encoding tree correctly picks `float32`.

    `without_eaf` names Analyses whose source reported no frequency, which the
    writer stores as all-NaN (ADR 0036) -- the mix a real release carries.
    """
    rng = np.random.default_rng(seed)
    truth = rng.uniform(0.05, 0.95, _N_VARIANTS)
    writer = RaggedCSRWriter(_N_VARIANTS)
    for i, count in enumerate(sizes):
        vi = np.sort(rng.choice(_N_VARIANTS, size=count, replace=False)).astype(np.int32)
        z = rng.standard_normal(count).astype(np.float32)
        se = np.abs(rng.standard_normal(count) * 0.1 + 0.2).astype(np.float32)
        noisy = np.clip(truth[vi] + rng.normal(0, 0.002, count), 1e-4, 1 - 1e-4)
        eaf = None if i in without_eaf else noisy.astype(np.float32)
        writer.add_analysis(vi, z, se, eaf)
    return writer


def _residual_encoding(writer, n_analyses):
    encoding = StoreEncoding.decide(
        EncodingMeasurements(n_analyses=n_analyses, eaf=writer.eaf_measurements())
    )
    assert encoding.eaf.is_residual, "fixture must select a residual EAF plane to be meaningful"
    return encoding


# ── Seam: the streamed cell source ───────────────────────────────────────────


@pytest.mark.parametrize("cell_budget", [1, 10, 500, 10_000])
def test_streamed_batches_reconstruct_the_materialised_cells(tmp_path, cell_budget):
    """Concatenating the batches must give back exactly what `se_fit_inputs`
    builds in one piece -- same values, same order, same dtypes.

    The batches read their frequencies back from the written `eaf` plane, so
    the plane has to be written first (issue #232); agreement with
    `se_fit_inputs`' in-memory round trip is what shows the plane encodes a
    per-cell function of the value and nothing batch-dependent."""
    sizes = [1500, 1, 0, 1800, 900, 1900]
    writer = _writer(sizes, without_eaf=(2, 4))
    encoding = _residual_encoding(writer, len(sizes))
    writer.write_eaf_plane(tmp_path / "streamed", encoding)

    whole = writer.se_fit_inputs(encoding)
    batches = list(writer.se_fit_batches(cell_budget=cell_budget))

    for field in ("se_values", "eaf_values", "analysis_indices"):
        streamed = (
            np.concatenate([getattr(b, field) for b in batches])
            if batches
            else np.empty(0, dtype=getattr(whole, field).dtype)
        )
        expected = getattr(whole, field)
        assert streamed.dtype == expected.dtype, field
        np.testing.assert_array_equal(streamed, expected, err_msg=field)


# ── Seam: the joint SE fit ───────────────────────────────────────────────────


def _dense_group(path, n_analyses=2, n_rows=600):
    """A Dense Component whose SE really does follow the log model, so the
    joint selection has a residual plane worth choosing."""
    group = zarr.open_group(str(path), mode="w", zarr_format=2)
    eaf = np.repeat(np.linspace(0.05, 0.95, n_rows, dtype=np.float32)[:, None], n_analyses, axis=1)
    coefficients = np.array([[-3.0, -0.5], [-2.7, -0.45]], dtype=np.float32)[:n_analyses]
    se = np.exp(
        coefficients[None, :, 0]
        + coefficients[None, :, 1] * np.log(2 * eaf * (1 - eaf))
        + 0.1 * np.sin(np.arange(n_rows)[:, None] * 0.1)
    ).astype(np.float32)
    group.create_array("eaf", data=np.asarray(eaf, dtype="float32"), chunks=(100, n_analyses))
    group.create_array("se", data=np.asarray(se, dtype="float16"), chunks=(100, n_analyses))
    group.create_array(
        "z", data=np.asarray(np.ones_like(eaf), dtype="float16"), chunks=(100, n_analyses)
    )
    return group, eaf, se


def _grouped_overflow_cells(eaf, se, n_analyses):
    """Overflow cells laid out the way a CSR holds them: Analysis by Analysis,
    not interleaved, which is the ordering the batching relies on."""
    return OverflowCells(
        se_values=np.concatenate([se[:, a] for a in range(n_analyses)]).ravel(),
        eaf_values=np.concatenate([eaf[:, a] for a in range(n_analyses)]).ravel(),
        analysis_indices=np.concatenate(
            [np.full(se.shape[0], a, dtype=np.int64) for a in range(n_analyses)]
        ),
        n_analyses=n_analyses,
    )


def test_streamed_overflow_selects_the_same_plan_as_one_bundle(tmp_path):
    """The optimiser must not be able to tell how its Overflow cells arrived:
    same chosen encoding, same coefficients, bit for bit."""
    preliminary = StoreEncoding(
        z=ZEncoding("float16"), se=SeEncoding("float16"), eaf=EafEncoding("float32")
    )
    group_a, eaf, se = _dense_group(tmp_path / "a.zarr")
    group_b, _, _ = _dense_group(tmp_path / "b.zarr")
    cells = _grouped_overflow_cells(eaf, se, 2)

    def batches():
        half = se.shape[0]
        for a in range(2):
            lo, hi = a * half, (a + 1) * half
            yield OverflowCells(
                se_values=cells.se_values[lo:hi],
                eaf_values=cells.eaf_values[lo:hi],
                analysis_indices=cells.analysis_indices[lo:hi],
                n_analyses=2,
            )

    whole_plan, whole_coef = optimise_dense_se_joint(
        group_a, preliminary, overflow=cells, overflow_chunk=200
    )
    streamed_plan, streamed_coef = optimise_dense_se_joint(
        group_b,
        preliminary,
        overflow=OverflowCellBatches(
            n_analyses=2, analysis_batches=batches, chunk_batches=lambda _n: batches()
        ),
        overflow_chunk=200,
    )

    assert whole_plan.se.is_residual, "fixture must select a residual SE plane to be meaningful"
    assert streamed_plan == whole_plan
    np.testing.assert_array_equal(streamed_coef, whole_coef)


# ── Seam: chunk-aligned batches for the byte measurement ─────────────────────


@pytest.mark.parametrize("multiple", [1, 7, 64, 1000])
def test_chunk_aligned_batches_end_on_multiples_and_reconstruct_the_cells(tmp_path, multiple):
    """The byte measurement charges chunk by chunk and pads only the plane's
    final edge chunk, so every batch but the last must end on a chunk boundary
    -- a batch that ended anywhere else would change the selected SE range."""
    sizes = [1500, 1, 0, 1800, 900, 1900]
    writer = _writer(sizes, without_eaf=(2, 4))
    encoding = _residual_encoding(writer, len(sizes))
    writer.write_eaf_plane(tmp_path / "streamed", encoding)

    whole = writer.se_fit_inputs(encoding)
    batches = list(writer.se_fit_chunk_batches(multiple, cell_budget=3 * multiple))

    for batch in batches[:-1]:
        assert len(batch.se_values) % multiple == 0, "interior batch must end on a chunk boundary"
    for field in ("se_values", "eaf_values", "analysis_indices"):
        streamed = np.concatenate([getattr(b, field) for b in batches])
        np.testing.assert_array_equal(streamed, getattr(whole, field), err_msg=field)


# ── Seam: the CSR flush ──────────────────────────────────────────────────────


def _model_writer(sizes, *, seed=0):
    """A writer whose SE follows the log-SE model, so a residual SE plane is
    worth choosing and the streamed SE write is actually exercised."""
    rng = np.random.default_rng(seed)
    truth = rng.uniform(0.05, 0.95, _N_VARIANTS)
    coefficients = np.array([-3.0, -0.5])
    writer = RaggedCSRWriter(_N_VARIANTS)
    for count in sizes:
        vi = np.sort(rng.choice(_N_VARIANTS, size=count, replace=False)).astype(np.int32)
        eaf = np.clip(truth[vi] + rng.normal(0, 0.002, count), 1e-4, 1 - 1e-4)
        x = np.log(2 * eaf * (1 - eaf))
        se = np.exp(coefficients[0] + coefficients[1] * x + rng.normal(0, 0.01, count))
        writer.add_analysis(
            vi,
            rng.standard_normal(count).astype(np.float32),
            se.astype(np.float32),
            eaf.astype(np.float32),
        )
    return writer


def _residual_se_encoding(writer, n_analyses):
    eaf_measured = writer.eaf_measurements()
    preliminary = StoreEncoding.decide(
        EncodingMeasurements(n_analyses=n_analyses, eaf=eaf_measured)
    )
    final = StoreEncoding.decide(
        EncodingMeasurements(
            n_analyses=n_analyses,
            eaf=eaf_measured,
            se=writer.se_measurements(preliminary),
        )
    )
    assert final.eaf.is_residual, "fixture must select a residual EAF plane"
    assert final.se.is_residual, "fixture must select a residual SE plane"
    return final


def _stored(path):
    """Every array a flushed CSR wrote, by name, for comparison."""
    root = zarr.open_group(str(Path(path) / "data.zarr" / "ragged"), mode="r")
    out = {}

    def walk(node, prefix=""):
        for name in node.array_keys():
            out[f"{prefix}{name}"] = np.asarray(node[name][:])
        for name in node.group_keys():
            walk(node[name], f"{prefix}{name}/")

    walk(root)
    return out


@pytest.mark.parametrize("region_cells", [200, 997, 4096])
def test_flush_writes_the_same_arrays_whatever_the_region_size(
    tmp_path, region_cells, monkeypatch
):
    """One region is the pre-streaming path, which the round-trip tests already
    pin as correct, so array-for-array agreement with a cut-up flush is what
    shows the streamed write changed the footprint and nothing else.

    #249 raises every region to a whole shard (50,000,000 elements in
    production), which for a fixture this size made the whole array one region
    and left the streaming path untested.  The shard and inner chunk are
    monkeypatched small so the cut flush really is several regions, and the
    region count is asserted before anything is compared (#249 review round 1).
    """
    monkeypatch.setattr(store_arrays, "RAGGED_SEQUENCE_SHARD_ELEMENTS", 2000)
    monkeypatch.setattr(store_arrays, "ASSOCIATION_SEQUENCE_CHUNK", 500)
    # Every Analysis carries enough cells to fit: `fit_se` declares the whole
    # plane ineligible unless every Analysis's coefficients come out finite, so
    # a one-cell Analysis would silently take the float16 branch and leave the
    # streamed SE write untested. The empty and single-cell Analyses are covered
    # by the cell-source tests above.
    sizes = [1500, 1800, 900, 1900]
    writer = _model_writer(sizes)
    encoding = _residual_se_encoding(writer, len(sizes))

    regions = list(writer._flat_regions(writer.n_associations, region_cells))
    assert len(regions) > 1, regions

    writer.flush(tmp_path / "whole", encoding, region_cells=1 << 30)
    # `region_cells` bigger than the shard must still be a whole number of shards.
    writer.flush(tmp_path / "cut", encoding, region_cells=region_cells)

    whole, cut = _stored(tmp_path / "whole"), _stored(tmp_path / "cut")
    assert sorted(cut) == sorted(whole)
    for name in whole:
        np.testing.assert_array_equal(cut[name], whole[name], err_msg=name)


def _peak_bytes(work) -> int:
    tracemalloc.start()
    tracemalloc.reset_peak()
    try:
        work()
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return peak


def test_flush_peak_memory_does_not_follow_the_cell_count(tmp_path, monkeypatch):
    """Four times the cells at a fixed region size must not cost four times the
    peak: that ratio is what made the OGS-00011 Overflow's flush unaffordable at
    a measured 72.9 bytes a cell.

    #249 raises the region to a whole shard, so a fixture shorter than a shard
    was one region and the bounded region was not exercised.  The shard and inner
    chunk are monkeypatched small so the step is a bounded whole shard (#249
    review round 1).
    """
    monkeypatch.setattr(store_arrays, "RAGGED_SEQUENCE_SHARD_ELEMENTS", 500)
    monkeypatch.setattr(store_arrays, "ASSOCIATION_SEQUENCE_CHUNK", 100)
    region = 500
    small = _model_writer([500] * 4, seed=1)
    large = _model_writer([2000] * 4, seed=1)
    # One plan for both, chosen from the larger plane: the variable under test is
    # the cell count, not the encoding.
    encoding = _residual_se_encoding(large, 4)
    assert large.n_associations == 4 * small.n_associations
    assert len(list(large._flat_regions(large.n_associations, region))) > 1

    peak_small = _peak_bytes(lambda: small.flush(tmp_path / "small", encoding, region_cells=region))
    peak_large = _peak_bytes(lambda: large.flush(tmp_path / "large", encoding, region_cells=region))

    growth = peak_large / peak_small
    assert growth < 2.5, f"peak grew {growth:.1f}x for 4x the cells"


# ── Seam: the eaf write, and the fit reading it back (issue #232) ────────────


def test_write_eaf_plane_then_flush_se_leaves_one_complete_group(tmp_path):
    """The eaf half is written first and the SE half is added to the same group.

    A group carrying only the frequency half is not a finished component: it
    has no `completion_state`, which only `flush_se` writes, so a build stopped
    between the write and the flush cannot be mistaken for one. The two halves
    together must store exactly what one `flush` stores, array for array.
    """
    sizes = [1500, 1800, 900, 1900]
    writer = _model_writer(sizes)
    encoding = _residual_se_encoding(writer, len(sizes))
    store = tmp_path / "split"
    writer.write_eaf_plane(store, encoding, region_cells=512)

    half = zarr.open_group(str(store / "data.zarr" / "ragged"), mode="r")
    assert "eaf" in half and "se" not in half
    assert "completion_state" not in half.attrs

    writer.flush_se(store, encoding, region_cells=512)
    whole = zarr.open_group(str(store / "data.zarr" / "ragged"), mode="r")
    assert "se" in whole
    assert whole.attrs["completion_state"] == "observed_only"

    one = _model_writer(sizes)
    one.flush(tmp_path / "one", encoding, region_cells=512)
    split_arrays, one_arrays = _stored(store), _stored(tmp_path / "one")
    assert sorted(split_arrays) == sorted(one_arrays)
    for name in one_arrays:
        np.testing.assert_array_equal(split_arrays[name], one_arrays[name], err_msg=name)


def test_fit_reads_the_plane_without_materialising_it(tmp_path):
    """Four times the cells at the same largest Analysis must not cost four
    times the fit's peak: the frequencies come back a batch at a time from the
    written plane, never as one whole-plane array (issue #232)."""
    budget = 512
    small, large = _model_writer([2000] * 2, seed=1), _model_writer([2000] * 8, seed=1)
    encoding = _residual_se_encoding(large, 8)
    assert large.n_associations == 4 * small.n_associations

    def fit_peak(writer, name):
        writer.write_eaf_plane(tmp_path / name, encoding, region_cells=budget)

        def consume() -> None:
            for _batch in writer.se_fit_batches(cell_budget=budget):
                pass

        return _peak_bytes(consume)

    growth = fit_peak(large, "large") / fit_peak(small, "small")
    assert growth < 2.0, f"fit peak grew {growth:.1f}x for 4x the cells"


def test_se_fit_reads_the_eaf_plane_back_instead_of_re_encoding(tmp_path, monkeypatch):
    """The fit must not encode a cell the write already encoded (issue #232).

    One `encode_eaf` per cell per build, at `write_eaf_plane`: a fit that
    re-encoded would run the round trip twice over the whole component, which is
    the duplication this ticket removes. `decode_eaf` must still run -- the fit
    predicts from what a reader decodes.
    """
    calls = {"encode": 0, "decode": 0}
    encode, decode = StoreCodec.encode_eaf, StoreCodec.decode_eaf

    def counted_encode(self, *args, **kwargs):
        calls["encode"] += 1
        return encode(self, *args, **kwargs)

    def counted_decode(self, *args, **kwargs):
        calls["decode"] += 1
        return decode(self, *args, **kwargs)

    monkeypatch.setattr(StoreCodec, "encode_eaf", counted_encode)
    monkeypatch.setattr(StoreCodec, "decode_eaf", counted_decode)

    writer = _model_writer([1500, 1800, 900, 1900])
    encoding = _residual_se_encoding(writer, 4)
    writer.write_eaf_plane(tmp_path / "store", encoding)
    written_encodes = calls["encode"]
    assert written_encodes >= 1, "the write must encode the plane"

    for _batch in writer.se_fit_batches(cell_budget=1024):
        pass
    assert calls["encode"] == written_encodes, "the fit re-encoded a cell it should read back"
    assert calls["decode"] > 0, "the fit must still decode what it reads"
