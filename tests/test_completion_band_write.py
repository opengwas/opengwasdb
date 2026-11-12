"""Dense Reference Completion's band write, spread over worker processes (#321).

After the LD-block phase, completion writes the completed planes one row band
at a time: a band reads its source rows once, takes its own fills, and writes
every plane in the destination's encoding. A band is whole Dense shards, so
``n_workers`` runs bands in a fork pool, each worker writing the shards its
band owns -- and, as for the LD blocks (ADR 0023), the worker count must change
nothing in the Store Release.

The fixture narrows the Dense shard and the band to 20 rows, so its 120
variants make six bands, and leaves cells unobserved across the axis, so fills
land in most bands. Its source codes SE as residuals against its EAF and holds
an entry in every side table (Z overflow, EAF and SE exceptions), so each plane
and table the band write produces is compared. Each test checks that before
relying on it.

"Identical" is what the release comparison tool compares: every array's
metadata (on-disk size included) and decoded values, `analyses.tsv` and
`variants.tsv.gz`, with the completion quality rows and the manifest but for
when it was written. Stored bytes are not compared: a chunk the parent
compresses with Blosc threads writes its blocks in the order they finish (ADR
0056), so two runs of the same code already differ there.
"""

from __future__ import annotations

import json
import multiprocessing
import os
import sqlite3
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from dense_sources import build_dense_from_records
from fork_order import CompletionOrder
from ld_blocks import write_ld_block
from zarr.storage import LocalStore

import opengwasdb.layouts.dense.complete as complete_module
import opengwasdb.store.arrays as arrays_module
from benchmarks.compare_store_releases import compare_releases
from opengwasdb.completion.checkpoint import checkpoint_dir_for
from opengwasdb.layouts.dense.build import NormalisedAssociation
from opengwasdb.layouts.dense.complete import complete_dense_store, resume_dense_completion
from opengwasdb.model.manifest import StoreManifest
from opengwasdb.store.arrays import open_group
from opengwasdb.variants import CanonicalVariant

N_VARIANTS = 120
ANALYSES = ("a", "b", "c")
BAND_ROWS = 20
N_BANDS = N_VARIANTS // BAND_ROWS
N_WORKERS = 3

#: The rows each Analysis leaves unobserved: spread across the axis, so fills
#: land in most bands.
UNOBSERVED = {"a": {7, 61}, "b": {15, 33, 70, 101}, "c": {44, 88, 118}}


def _records() -> list[NormalisedAssociation]:
    """Three Analyses whose SE tracks its MAF-predicted value, so the source
    codes it as residuals, with one cell for each side table: an EAF far from its
    variant's baseline, an SE far from its prediction, and a Z beyond the
    fixed-point range."""
    frequencies = np.linspace(0.05, 0.95, N_VARIANTS, dtype=np.float32)
    records = []
    for col, analysis_id in enumerate(ANALYSES):
        for row in range(N_VARIANTS):
            if row in UNOBSERVED[analysis_id]:
                continue
            eaf = 0.002 if (analysis_id, row) == ("c", 20) else float(frequencies[row])
            se = float(
                np.exp(
                    (-3.0 + col * 0.2)
                    - 0.5 * np.log(2 * eaf * (1 - eaf))
                    + 0.12 * np.sin(row * (0.07 + col * 0.01))
                )
            )
            if (analysis_id, row) == ("b", 50):
                se *= 40.0
            z = 8.0 if row % 40 == 0 else 1.0 + 0.01 * row
            if (analysis_id, row) == ("a", 90):
                z = 120.0
            records.append(
                NormalisedAssociation(
                    analysis_id,
                    CanonicalVariant("1", (row + 1) * 1000, "A", "G"),
                    z=z,
                    se=se,
                    eaf=eaf,
                )
            )
    return records


def _ld_panel(root: Path) -> Path:
    """Three LD blocks of 40 variants, every one of them in the source."""
    frequencies = np.linspace(0.05, 0.95, N_VARIANTS, dtype=np.float32)
    for block in range(3):
        rows = range(block * 40, (block + 1) * 40)
        write_ld_block(
            root / "EUR" / "1",
            f"{rows[0] * 1000 + 1000}-{rows[-1] * 1000 + 1000}",
            [(f"1:{(r + 1) * 1000}:A:G", float(frequencies[r]), (r + 1) * 1000) for r in rows],
            seed=block,
        )
    return root


def _narrow_bands(patch: pytest.MonkeyPatch) -> None:
    """A 20-row Dense shard, so the completion band is 20 rows too."""
    patch.setattr(arrays_module, "DENSE_SHARD_SHAPE", (BAND_ROWS, 2))
    patch.setattr(arrays_module, "DENSE_CHUNK_SHAPE", (10, 1))
    patch.setattr(complete_module, "DEFAULT_CHUNK_SHAPE", (10, 1))
    patch.setattr(complete_module, "_BAND_ROWS", BAND_ROWS)


@pytest.fixture(scope="module")
def inputs(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    root = tmp_path_factory.mktemp("band-write")
    source = root / "obs.opengwasdb"
    build_dense_from_records(
        _records(),
        source,
        store_id="s",
        release_id="obs",
        reference_assembly="GRCh38",
        chunk_shape=(20, 1),
    )
    return source, _ld_panel(root / "ld_panel")


def _complete(source: Path, panel: Path, dst: Path, n_workers: int) -> Path:
    complete_dense_store(
        source, dst, panel, ancestry="EUR", min_cor=0.0, release_id="comp", n_workers=n_workers
    )
    return dst


@pytest.fixture(scope="module")
def serial_release(inputs: tuple[Path, Path], tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The release completed on one worker: the serial path the others must equal."""
    source, panel = inputs
    with pytest.MonkeyPatch.context() as patch:
        _narrow_bands(patch)
        dst = tmp_path_factory.mktemp("serial") / "serial.opengwasdb"
        return _complete(source, panel, dst, n_workers=1)


def _assert_fixture_is_meaningful(source: Path, release: Path) -> None:
    """Several bands, fills in more than one, and every side table in use."""
    root = open_group(release / "data.zarr", "r")
    assert tuple(int(size) for size in root["z"].shards)[0] == BAND_ROWS
    assert complete_module.completion_band_rows(BAND_ROWS) == BAND_ROWS
    n_bands = -(-int(root["z"].shape[0]) // BAND_ROWS)
    assert n_bands == N_BANDS >= 3
    imputed_rows = np.flatnonzero(np.asarray(root["imputed"][:]).any(axis=1))
    assert len(set((imputed_rows // BAND_ROWS).tolist())) >= 2, imputed_rows
    plan = StoreManifest.load(source).encoding
    assert plan.se.is_residual and plan.eaf.is_residual
    assert StoreManifest.load(release).encoding.eaf.reference
    for table in ("z_overflow_index", "eaf_exception_index", "se_exception_index"):
        assert int(root[table].shape[0]) >= 1, table


@dataclass(frozen=True)
class _Inputs:
    """A completion's source and panel, and the release one worker makes of them."""

    source: Path
    panel: Path
    serial: Path

    def complete(self, dst: Path, n_workers: int) -> Path:
        return _complete(self.source, self.panel, dst, n_workers)


@pytest.fixture
def completion(
    inputs: tuple[Path, Path], serial_release: Path, monkeypatch: pytest.MonkeyPatch
) -> _Inputs:
    """The fixture, checked, with the bands narrowed for the test's own completion."""
    source, panel = inputs
    _narrow_bands(monkeypatch)
    _assert_fixture_is_meaningful(source, serial_release)
    return _Inputs(source, panel, serial_release)


def _quality_rows(release: Path) -> list[tuple[Any, ...]]:
    with sqlite3.connect(release / "index.sqlite") as db:
        return db.execute("SELECT * FROM completion_quality ORDER BY rowid").fetchall()


def _manifest_without_timestamp(release: Path) -> dict[str, Any]:
    manifest = json.loads((release / "manifest.json").read_text(encoding="utf-8"))
    manifest.pop("created_at")
    return manifest


def _assert_same_release(left: Path, right: Path) -> None:
    """The release comparison tool finds nothing, and neither does the rest."""
    report = compare_releases(left, right)
    assert report["arrays_only_left"] == [] and report["arrays_only_right"] == []
    assert report["arrays_with_metadata_differences"] == []
    assert report["arrays_with_value_differences"] == []
    assert report["analyses_tsv"]["differing_cells_by_column"] == {}
    assert report["variants_tsv_gz"]["identical"]
    assert report["encoding_plan"]["left"] == report["encoding_plan"]["right"]
    assert _quality_rows(left) == _quality_rows(right)
    assert _manifest_without_timestamp(left) == _manifest_without_timestamp(right)


# ── one worker against several, a band finishing out of order ─────────────

#: Bands 0 to 2: the first is held until the other two have finished.
_ORDER = CompletionOrder(3)
#: The band write's worker this module wraps; ``None`` where there is none.
_REAL_COMPLETE_BAND = getattr(complete_module, "_complete_band", None)


def _first_band_finishing_last(plan: Any, band: int) -> Any:
    """The band write's own worker, the first band held until bands 1 and 2 finish."""
    if band == 0:
        _ORDER.wait_for_the_rest()
    assert _REAL_COMPLETE_BAND is not None
    outcome = _REAL_COMPLETE_BAND(plan, band)
    if band < 3:
        _ORDER.claim(band)
    return outcome


def test_a_release_completed_on_three_workers_is_the_one_completed_on_one(
    completion: _Inputs, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The first band is held until two others have finished, so a write that
    took the pool's bands in the order they finished would put one band's
    tallies or side-table entries under another's rows."""
    _ORDER.reset()
    monkeypatch.setattr(
        complete_module, "_complete_band", _first_band_finishing_last, raising=False
    )

    parallel = completion.complete(tmp_path / "parallel.opengwasdb", n_workers=N_WORKERS)

    finished = _ORDER.order()
    assert sorted(finished) == [0, 1, 2], "the bands did not run on a pool"
    assert finished[-1] == 0
    _assert_same_release(completion.serial, parallel)


# ── an interrupted band write ──────────────────────────────────────────────

_FORK = multiprocessing.get_context("fork")
_BANDS_WRITTEN = _FORK.Value("i", 0)
_CRASHED_IN_PID = _FORK.Value("i", 0)
_CRASHING_BAND = 3


def _band_crashing_after_two_others(plan: Any, band: int) -> Any:
    """The band write's worker, failing on one band once two others are written."""
    if band == _CRASHING_BAND:
        deadline = time.monotonic() + CompletionOrder.WAIT_SECONDS
        while _BANDS_WRITTEN.value < 2:
            if time.monotonic() > deadline:
                raise AssertionError("no two bands were written beside the failing one")
            os.sched_yield()
        _CRASHED_IN_PID.value = os.getpid()
        raise RuntimeError("simulated crash in the band write")
    assert _REAL_COMPLETE_BAND is not None
    outcome = _REAL_COMPLETE_BAND(plan, band)
    with _BANDS_WRITTEN.get_lock():
        _BANDS_WRITTEN.value += 1
    return outcome


def test_a_band_write_interrupted_after_some_bands_resumes_to_the_same_release(
    completion: _Inputs, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A worker fails once other bands are already in the planes: the staged
    release goes, every LD block's checkpoint stays, and the resumed completion,
    which writes every band again, is the release a single worker writes."""
    dst = tmp_path / "resumed.opengwasdb"
    _BANDS_WRITTEN.value = 0
    _CRASHED_IN_PID.value = 0
    with monkeypatch.context() as crash:
        crash.setattr(
            complete_module, "_complete_band", _band_crashing_after_two_others, raising=False
        )
        with pytest.raises(RuntimeError, match="simulated crash in the band write"):
            completion.complete(dst, n_workers=N_WORKERS)

    assert _BANDS_WRITTEN.value >= 2
    assert _CRASHED_IN_PID.value not in (0, os.getpid()), "the band ran in the parent"
    checkpoint_dir = checkpoint_dir_for(dst)
    assert len(list((checkpoint_dir / "blocks").glob("*.npz"))) == 3
    assert not dst.exists()
    assert list(dst.parent.glob(f".{dst.name}.tmp*")) == []

    resume_dense_completion(checkpoint_dir, n_workers=N_WORKERS)

    assert not checkpoint_dir.exists()
    _assert_same_release(completion.serial, dst)


# ── scratch disk ───────────────────────────────────────────────────────────


def _tree_bytes(path: Path, skip: str | None = None) -> int:
    """Apparent bytes of every file under `path`, but a top-level `skip` entry."""
    total = 0
    for dirpath, dirnames, filenames in os.walk(path):
        if skip is not None and Path(dirpath) == path:
            dirnames[:] = [name for name in dirnames if name != skip]
        for filename in filenames:
            try:
                total += os.lstat(os.path.join(dirpath, filename)).st_size
            except FileNotFoundError:
                pass
    return total


class _ScratchPeak:
    """What a completion holds on disk, read after every array write any of its
    processes makes, so the peak is the one the write reached, not a sample.

    Three figures: everything (the LD-block checkpoints and the staged release),
    the staged release's planes alone (`data.zarr` but for the Top-Hit Index,
    written after the band write), and the fill shards, so a test can see that
    a reading was taken while the band write still had them. Kept in shared
    memory made with the fork start method, so a band worker's writes count.
    """

    def __init__(self, work: Path, name: str) -> None:
        self._work, self._name = work, name
        self._lock = _FORK.Lock()
        self._total = _FORK.Value("q", 0, lock=False)
        self._planes = _FORK.Value("q", 0, lock=False)
        self._fills = _FORK.Value("q", 0, lock=False)

    def read(self) -> None:
        checkpoint = _tree_bytes(self._work / f".{self._name}.checkpoint")
        staged = [p for p in self._work.iterdir() if p.name.startswith(f".{self._name}.tmp")]
        total = checkpoint + sum(_tree_bytes(s) for s in staged)
        planes = sum(_tree_bytes(s / "data.zarr", skip="top_hits") for s in staged)
        fills = sum(_tree_bytes(s / "fill_shards") for s in staged)
        with self._lock:
            self._total.value = max(self._total.value, total)
            self._planes.value = max(self._planes.value, planes)
            self._fills.value = max(self._fills.value, fills)

    def peaks(self) -> tuple[int, int, int]:
        """`(everything, planes, fill shards)`, each its largest reading."""
        return self._total.value, self._planes.value, self._fills.value


def _reading_after_every_write(monkeypatch: pytest.MonkeyPatch, peak: _ScratchPeak) -> None:
    original: Callable[..., Any] = LocalStore._set

    async def measured(self: LocalStore, key: str, value: Any, exclusive: bool = False) -> None:
        await original(self, key, value, exclusive=exclusive)
        peak.read()

    monkeypatch.setattr(LocalStore, "_set", measured)


def _planes_bytes(release: Path) -> int:
    return _tree_bytes(release / "data.zarr", skip="top_hits")


def test_the_band_write_holds_no_plane_the_release_does_not(
    completion: _Inputs, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Peak scratch disk is the LD blocks' checkpoints and the fill shards beside
    the release's own planes, at one worker and at three.

    The serial write ADR 0023 describes staged SE as a `float32` plane beside
    them, and at its end an `int8` copy too, while the checkpoints and every
    fill shard were still on disk. At no write may the staged planes hold more
    than the release they become, and three workers hold no more than one.
    """
    peaks = {}
    for n_workers in (1, N_WORKERS):
        work = tmp_path / f"work-{n_workers}"
        work.mkdir()
        peak = _ScratchPeak(work, "dst.opengwasdb")
        with monkeypatch.context() as patch:
            _reading_after_every_write(patch, peak)
            release = completion.complete(work / "dst.opengwasdb", n_workers=n_workers)
        total, planes, fills = peak.peaks()
        assert fills > 0, "no reading was taken while the fill shards were on disk"
        final = _planes_bytes(release)
        assert final == _planes_bytes(completion.serial)
        assert planes <= final, (
            f"at {n_workers} worker(s) the staged planes held {planes:,} bytes; "
            f"the release holds {final:,}"
        )
        peaks[n_workers] = total
    assert peaks[N_WORKERS] <= peaks[1], peaks
