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

import opengwasdb.encoding.memory as memory
import opengwasdb.layouts.dense.build_vcf as dense_build
from opengwasdb.encoding import StoreCodec, eaf_exception_table
from opengwasdb.layouts.dense.build_vcf import (
    _EAF_STAGING,
    _encode_residual_dense_eaf,
    build_dense_from_vcf_manifest,
)
from opengwasdb.store.open import OpenGWASDBStore, open_store
from opengwasdb.store_reader.encoding.codec import EafExceptionTable
from opengwasdb.store_reader.encoding.plan import EafEncoding, SeEncoding, StoreEncoding, ZEncoding

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


def _one_worker() -> int:
    """Bytes one worker costs the budget for a fixture band: the fork's floor and
    the band's cells, and the two encoded bands the parent may hold for it."""
    return memory.worker_bytes(
        dense_build._eaf_band_bytes(len(_ANALYSES), _BAND_ROWS)
    ) + memory.RESULTS_HELD_PER_WORKER * dense_build._eaf_band_result_bytes(
        len(_ANALYSES), _BAND_ROWS
    )


def _host(monkeypatch: pytest.MonkeyPatch, available: int | None, parent: int) -> None:
    """A host with `available` bytes available, and a parent holding `parent`."""
    monkeypatch.setattr(memory, "available_memory_bytes", lambda path=memory.MEMINFO: available)
    monkeypatch.setattr(memory, "resident_bytes", lambda path=memory.SELF_STATUS: parent)


def _encode_pool(caplog: pytest.LogCaptureFixture, requested: int) -> int:
    with caplog.at_level(logging.INFO, logger=memory.log.name):
        return dense_build._eaf_encode_pool(_fake_context(), requested)


def test_the_parent_and_the_fork_s_floor_count_against_the_pool(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The encode is sized as every pool is (`pool_workers`): a budget of four
    workers, half of it held by the parent, runs two of the three asked for."""
    _host(monkeypatch, memory.MEMORY_BUDGET_DIVISOR * 4 * _one_worker(), 2 * _one_worker())

    assert _encode_pool(caplog, 3) == 2
    assert "Encode eaf: 2 of 3 worker(s)" in caplog.text
    assert f"200-row band of {len(_ANALYSES)} analyses" in caplog.text


def test_no_more_workers_than_bands(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _host(monkeypatch, 10**15, 0)

    assert _encode_pool(caplog, 16) == _N_BANDS
    assert f"{_N_BANDS} of {_N_BANDS} worker(s)" in caplog.text


def test_a_budget_below_one_band_warns_and_still_runs_one_worker(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The overrun the cap exists for: less than a worker's worth of the quarter.

    One worker is still one band, so the encode runs -- but it runs into a
    budget it does not fit, and that is a warning, not a line of progress.
    """
    _host(monkeypatch, memory.MEMORY_BUDGET_DIVISOR * _one_worker() // 2, 0)

    assert _encode_pool(caplog, 3) == 1, "the floor is one worker, never zero"
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("which will overrun it" in message for message in warnings), warnings


def test_an_unreadable_meminfo_runs_one_worker_and_says_why(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _host(monkeypatch, None, 0)

    assert _encode_pool(caplog, 3) == 1
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("meminfo" in message and "one worker" in message for message in warnings), warnings


def test_an_unreadable_meminfo_is_silent_at_one_worker(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """At `--n-workers 1` the reading changes nothing, so it must not warn."""
    _host(monkeypatch, None, 0)

    assert _encode_pool(caplog, 1) == 1
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


def test_the_bands_run_on_the_pool_the_budget_sized(monkeypatch: pytest.MonkeyPatch) -> None:
    """The band loop forks the workers the budget allows, not the ones asked for."""
    seen: list[int] = []
    real = dense_build.ordered_map

    def spy(fn, items, n_workers, **kwargs):
        seen.append(n_workers)
        return real(fn, items, n_workers, **kwargs)

    _host(monkeypatch, memory.MEMORY_BUDGET_DIVISOR * 2 * _one_worker(), 0)
    monkeypatch.setattr(dense_build, "ordered_map", spy)
    plane = _Plane((_N_VARIANTS, len(_ANALYSES)))
    baseline = np.full(_N_VARIANTS, np.nan, dtype=np.float32)

    tables = dense_build._write_eaf_bands(plane, baseline, _fake_context(), 0.0, n_workers=3)

    assert len(tables) == _N_BANDS
    assert np.isfinite(baseline).all(), "the bands' baselines were not written"
    assert seen == [2], f"the encode did not run on the pool the budget sized: {seen}"


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
