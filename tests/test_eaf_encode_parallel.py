"""The residual `eaf` encode's row bands run across the ordered pool (#320).

The encode transposes the staged `float32` grid into three outputs: the
per-variant baseline, the `int8` residual plane and the exception table. The
first two are per row, so a band's worker can compute them alone; the third is
one table sorted by flat position, so its bytes are the order the bands'
entries were concatenated in. The fixture spans three row bands and puts
exception cells in two of them, and the parallel build holds the first band's
worker until the others have finished -- so a parent that assembled the table
in completion order cannot produce the serial bytes.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import pytest
from fork_order import CompletionOrder
from residual_fixtures import write_gwas_vcf_with_eaf, write_hg38_manifest
from store_assertions import assert_same_band_arrays

import opengwasdb.layouts.dense.build_vcf as dense_build
from opengwasdb.encoding import (
    EafEncoding,
    EafExceptionTable,
    SeEncoding,
    StoreCodec,
    StoreEncoding,
    ZEncoding,
    eaf_exception_table,
)
from opengwasdb.layouts.dense.build_vcf import (
    _EAF_STAGING,
    _encode_residual_dense_eaf,
    build_dense_from_vcf_manifest,
)
from opengwasdb.store.open import OpenGWASDBStore, open_store

_N_VARIANTS = 600
_ANALYSES = ("a", "b", "c")
#: Rows per band. Deliberately shorter than a shard, which a fixture-sized
#: variant axis clips to 600 rows: the production rule (`_eaf_row_band`, pinned
#: by `test_shard_aligned_writes.py`) would put this whole axis in one band.
_BAND_ROWS = 200
_N_BANDS = _N_VARIANTS // _BAND_ROWS
#: Rows whose frequency the second Analysis reports nowhere near the others',
#: by a residual no candidate range covers -- one in each row band.
_SHIFTED = {10: 0.90, 250: 0.02, 410: 0.93, 555: 0.03}


def _frequencies() -> np.ndarray:
    """The fixture's frequency grid: three Analyses agreeing, bar `_SHIFTED`."""
    grid = np.repeat(
        np.linspace(0.05, 0.95, _N_VARIANTS)[:, None], len(_ANALYSES), axis=1
    )
    for row, frequency in _SHIFTED.items():
        grid[row, 1] = frequency
    return grid.astype(np.float32)


def _rows(frequencies: np.ndarray) -> list[str]:
    """One Analysis's GWAS-VCF body: its frequency column, and an ordinary z."""
    lines = []
    for row in range(_N_VARIANTS):
        frequency = float(frequencies[row])
        z = 6.0 if row % 7 == 0 else 1.0
        lines.append(
            f"1\t{(row + 1) * 1000}\t.\tA\tG\t.\tPASS\t.\tES:SE:EZ:AF\t"
            f"{z * 0.4:.6f}:0.4:{z}:{frequency:.6f}\n"
        )
    return lines


def _manifest(tmp_path: Path) -> Path:
    grid = _frequencies()
    vcfs = {
        name: write_gwas_vcf_with_eaf(
            tmp_path / f"{name}.vcf", _rows(grid[:, column])
        )
        for column, name in enumerate(_ANALYSES)
    }
    return write_hg38_manifest(tmp_path / "manifest.tsv", vcfs)


def _build(manifest: Path, out: Path, n_workers: int) -> Path:
    """Build the fixture's manifest, into a release the caller names."""
    built = build_dense_from_vcf_manifest(
        manifest, out, store_id="s", release_id="r", n_workers=n_workers,
        allow_unverified_eaf=True,
    )
    return built.output_path


@pytest.fixture(autouse=True)
def three_bands(monkeypatch: pytest.MonkeyPatch) -> None:
    """Shorten the row band so a fixture-sized axis has several of them."""
    monkeypatch.setattr(
        dense_build, "_eaf_row_band", lambda shard_rows, n_analyses: _BAND_ROWS
    )


# The pool's own finishing order, made with the fork start method so a worker
# inherits it exactly as it inherits the encode's shared context.
_ORDER = CompletionOrder(_N_BANDS)
_REAL_ENCODE_BAND = dense_build._encode_eaf_band


def _encode_band_finishing_last(context: object, r0: int) -> object:
    """The build's own band worker, with the first band held until the end."""
    band = r0 // _BAND_ROWS
    if band == 0:
        _ORDER.wait_for_the_rest()
    result = _REAL_ENCODE_BAND(context, r0)
    _ORDER.claim(band)
    return result


def _exception_bands(root: object) -> np.ndarray:
    """Which row band each exception cell of a built store's table falls in."""
    index = np.asarray(root["eaf_exception_index"][:])
    return (index // (_BAND_ROWS * len(_ANALYSES))).astype(np.int64)


def test_the_eaf_encode_is_identical_across_worker_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _manifest(tmp_path)
    serial = _build(manifest, tmp_path / "serial.opengwasdb", n_workers=1)

    _ORDER.reset()
    monkeypatch.setattr(dense_build, "_encode_eaf_band", _encode_band_finishing_last)
    parallel = _build(manifest, tmp_path / "parallel.opengwasdb", n_workers=_N_BANDS)

    rs = open_store(serial).arrays(mode="r")
    rp = open_store(parallel).arrays(mode="r")

    # The fixture only means anything if it is residual-coded and its exception
    # cells straddle more than one band: one band's run cannot be misordered.
    assert rs["eaf"].dtype == np.dtype("int8"), "fixture did not residual-code eaf"
    assert _N_BANDS >= 3
    bands = _exception_bands(rs)
    assert len(bands) >= 2 and len(np.unique(bands)) >= 2, (
        f"fixture has {len(bands)} exception cell(s) in bands {sorted(set(bands.tolist()))}; "
        "ordering needs at least one in two bands"
    )
    # ...and if the first band did not really finish last, input order and
    # completion order coincide and nothing below is under test.
    finished = _ORDER.order()
    assert sorted(finished) == list(range(_N_BANDS))
    assert finished[-1] == 0, f"bands finished {finished}; the first was to be last"

    assert _EAF_STAGING not in rs and _EAF_STAGING not in rp
    assert_same_band_arrays(rs, rp)


def test_the_exception_table_is_sensitive_to_band_order(tmp_path: Path) -> None:
    """So the equality test above can fail: a table's bytes are band order.

    Assembling the same bands' exception runs the way a completion-ordered
    parent would -- last band first -- gives a different index, and the table's
    own rule (sorted, unique) refuses to hold it.
    """
    root = open_store(_build(_manifest(tmp_path), tmp_path / "store.opengwasdb", 1))
    arrays = root.arrays(mode="r")
    index = np.asarray(arrays["eaf_exception_index"][:], dtype=np.int64)
    value = np.asarray(arrays["eaf_exception_value"][:], dtype=np.float32)
    span = _BAND_ROWS * len(_ANALYSES)

    def runs() -> list[EafExceptionTable]:
        in_band = index // span
        return [
            EafExceptionTable(index=index[in_band == band], value=value[in_band == band])
            for band in range(_N_BANDS)
        ]

    assert sum(1 for run in runs() if len(run)) >= 2, "one band's run cannot be reordered"
    ordered = eaf_exception_table(runs())
    np.testing.assert_array_equal(ordered.index, index)
    np.testing.assert_array_equal(ordered.value, value)

    completion = list(reversed(runs()))
    assert not np.array_equal(
        np.concatenate([run.index for run in completion]), ordered.index
    )
    with pytest.raises(ValueError, match="sorted and unique"):
        eaf_exception_table(completion)


def _residual_codec() -> StoreCodec:
    return StoreCodec(
        StoreEncoding(
            z=ZEncoding("float16"),
            se=SeEncoding("float16"),
            eaf=EafEncoding("int8_residual", residual_range=0.5),
        )
    )


def test_a_failed_band_leaves_the_staging_plane_in_place(tmp_path: Path) -> None:
    """A resume must be able to tell that the encode never finished.

    `1.5` is not a frequency, so the last band's worker fails after the earlier
    bands were encoded and written; the staged grid must still be there.
    """
    grid = _frequencies()
    grid[-1, 2] = 1.5
    store = tmp_path / "staged.opengwasdb"
    chunks = (100, len(_ANALYSES))

    with OpenGWASDBStore.staging(store) as staged:
        source = dense_build._create_eaf_array(
            staged, _N_VARIANTS, len(_ANALYSES), chunks, name=_EAF_STAGING
        )
        source[:] = grid
        with pytest.raises(ValueError, match="not a frequency"):
            _encode_residual_dense_eaf(
                staged,
                _N_VARIANTS,
                len(_ANALYSES),
                chunks,
                _residual_codec(),
                time.monotonic(),
                n_workers=_N_BANDS,
            )
        root = staged.arrays(mode="r")
        assert _EAF_STAGING in root, "the failed encode deleted the staged grid"
        assert "eaf_exception_index" not in root, "the exception table was written anyway"


# ── The memory budget that bounds the pool (#320) ────────────────────────────
#
# The cap is arithmetic between a reading of the host's memory and a band's
# size, so it is checked where it is decided rather than through a build: the
# store a capped run writes is the uncapped run's byte for byte, which
# `test_the_eaf_encode_is_identical_across_worker_counts` already asserts.


def _one_band() -> int:
    """Bytes one worker holds for a fixture band, from the empirical model."""
    return dense_build._eaf_band_bytes(len(_ANALYSES), _BAND_ROWS)


def test_a_small_budget_lowers_the_worker_count(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A host that can hold one band and a half gets one worker, not a crash."""
    with caplog.at_level(logging.INFO, logger="opengwasdb.layouts.dense.build_vcf"):
        monkeypatch.setattr(dense_build, "available_memory_bytes", lambda: 6 * _one_band())
        workers = dense_build._eaf_encode_pool(len(_ANALYSES), _BAND_ROWS, requested=3)

    assert workers == 1
    messages = [record.getMessage() for record in caplog.records]
    assert any("1 of 3 worker(s)" in message for message in messages), messages
    assert any(f"{_one_band() >> 20} MiB" in message for message in messages), messages


def test_a_budget_below_one_band_warns_and_still_runs_one_worker(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The overrun the cap exists for: less than a band's worth of the quarter.

    One worker is still one band, so the encode runs -- but it runs into a
    budget it does not fit, and that is a warning, not a line of progress.
    """
    with caplog.at_level(logging.INFO, logger="opengwasdb.layouts.dense.build_vcf"):
        monkeypatch.setattr(dense_build, "available_memory_bytes", lambda: 2 * _one_band())
        workers = dense_build._eaf_encode_pool(len(_ANALYSES), _BAND_ROWS, requested=3)

    assert workers == 1, "the floor is one worker, never zero"
    warnings = [
        record.getMessage() for record in caplog.records if record.levelno == logging.WARNING
    ]
    assert any("one band" in message for message in warnings), warnings


def test_an_unreadable_meminfo_runs_one_worker_and_says_why(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(dense_build, "available_memory_bytes", lambda: None)
    with caplog.at_level(logging.INFO, logger="opengwasdb.layouts.dense.build_vcf"):
        workers = dense_build._eaf_encode_pool(len(_ANALYSES), _BAND_ROWS, requested=3)

    assert workers == 1
    warnings = [
        record.getMessage() for record in caplog.records if record.levelno == logging.WARNING
    ]
    assert any("meminfo" in message and "one worker" in message for message in warnings), warnings


def test_an_unreadable_meminfo_is_silent_at_one_worker(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """At `--n-workers 1` the reading changes nothing, so it must not warn."""
    monkeypatch.setattr(dense_build, "available_memory_bytes", lambda: None)
    with caplog.at_level(logging.INFO, logger="opengwasdb.layouts.dense.build_vcf"):
        workers = dense_build._eaf_encode_pool(len(_ANALYSES), _BAND_ROWS, requested=1)

    assert workers == 1
    assert [r for r in caplog.records if r.levelno == logging.WARNING] == []


class _Plane:
    """A stand-in for the `eaf` array the parent writes bands into."""

    def __init__(self, shape: tuple[int, int]) -> None:
        self.codes = np.zeros(shape, dtype=np.int8)

    def __setitem__(self, key: slice, value: np.ndarray) -> None:
        self.codes[key] = value


class _Grid:
    """A stand-in for the staged `float32` grid: plausible frequencies, band at a time."""

    def __init__(self, n_analyses: int) -> None:
        self.n_analyses = n_analyses

    def __getitem__(self, key: slice) -> np.ndarray:
        rows = np.linspace(0.05, 0.95, key.stop - key.start)[:, None]
        return np.repeat(rows, self.n_analyses, axis=1).astype(np.float32)


def _fake_context() -> object:
    """A context over the stand-in grid: no store, three bands, three Analyses."""
    return dense_build._EafBandContext(
        staging=_Grid(len(_ANALYSES)),
        codec=_residual_codec(),
        n_analyses=len(_ANALYSES),
        band_rows=_BAND_ROWS,
        n_variants=_N_VARIANTS,
    )


def test_the_pool_is_bounded_to_one_band_per_worker(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    """The parent's window is the effective worker count, so it holds one band a worker."""
    seen: list[tuple[int, int | None]] = []
    real = dense_build.ordered_map
    unset = object()

    def spy(fn, items, n_workers, max_in_flight=None, *, shared=unset):
        seen.append((n_workers, max_in_flight))
        if shared is unset:
            return real(fn, items, n_workers, max_in_flight)
        return real(fn, items, n_workers, max_in_flight, shared=shared)

    # Two and a half bands of budget: two workers fit, and no more.
    monkeypatch.setattr(dense_build, "available_memory_bytes", lambda: 10 * _one_band())
    monkeypatch.setattr(dense_build, "ordered_map", spy)
    plane = _Plane((_N_VARIANTS, len(_ANALYSES)))
    baseline = np.full(_N_VARIANTS, np.nan, dtype=np.float32)

    tables = dense_build._write_eaf_bands(plane, baseline, _fake_context(), 0.0, n_workers=3)

    assert len(tables) == _N_BANDS
    assert np.isfinite(baseline).all(), "the bands' baselines were not written"
    bounded = [call for call in seen if call[1] is not None]
    assert bounded == [(2, 2)], f"the encode's pool was not bounded: {bounded}"


def test_a_band_of_the_wrong_length_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    """The parent knows where a band ends, so a short result cannot land beside it."""
    real = dense_build._encode_eaf_band

    def truncating(context: object, r0: int) -> object:
        band = real(context, r0)
        return dense_build._EncodedEafBand(
            start=band.start,
            codes=band.codes,
            baseline=band.baseline[:-1],
            exceptions=band.exceptions,
        )

    monkeypatch.setattr(dense_build, "available_memory_bytes", lambda: 10**15)
    monkeypatch.setattr(dense_build, "_encode_eaf_band", truncating)
    plane = _Plane((_N_VARIANTS, len(_ANALYSES)))
    with pytest.raises(ValueError, match="rows"):
        dense_build._write_eaf_bands(
            plane,
            np.full(_N_VARIANTS, np.nan, dtype=np.float32),
            _fake_context(),
            0.0,
            n_workers=1,
        )
