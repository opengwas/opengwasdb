"""Ragged Reference Completion's write, spread over worker processes (#321).

After the LD-block phase, Ragged completion assembles each Analysis's
completed rows -- its observed rows read from the source, its fills, its
missing block positions -- and the writer lays them end to end in Analysis
order. The Analyses are assembled in a fork pool, as Dense writes its bands
there, and the worker count must change nothing in the Store Release.

The writer then writes the sequence a region of whole shards at a time, and
the regions run in the same pool, each worker writing its own shards.

The fixture is a residual-SE source of four Analyses over one LD block, each
leaving different block positions unobserved, and a panel that adds positions
the source never held, so every Analysis gains fills. Each test checks that
before relying on it. "Identical" is `completion_releases.assert_same_release`'s.
"""

from __future__ import annotations

import multiprocessing
import os
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import zarr
from completion_releases import CompletionInputs, assert_same_release
from fork_order import CompletionOrder
from pool_budget import pin_pool_units
from test_ragged_residual_completion import (
    _FREQS,
    _TRAIT_BP,
    N_SOURCE,
    _expected_se,
    _source_bp,
    _source_z,
    _write_filtered_ssf,
    write_eaf_carrying_panel,
)

import opengwasdb.encoding.memory as memory
import opengwasdb.layouts.ragged.complete as complete_module
import opengwasdb.layouts.ragged.zarr_csr as zarr_csr
import opengwasdb.store.arrays as store_arrays
from opengwasdb.completion.checkpoint import checkpoint_dir_for
from opengwasdb.encoding.plan import EafEncoding, SeEncoding, StoreEncoding, ZEncoding
from opengwasdb.layouts.ragged.build_ssf import build_ragged_from_ssf
from opengwasdb.layouts.ragged.complete import complete_ragged_store, resume_ragged_completion
from opengwasdb.layouts.ragged.zarr_csr import RAGGED_ZARR_PATH, RaggedCSRWriter
from opengwasdb.model.manifest import StoreManifest

N_WORKERS = 3

#: Each Analysis's SE model offset, and the block positions it leaves unobserved.
ANALYSES = {
    "a": (0.0, frozenset()),
    "b": (0.2, frozenset({36, 37, 38, 39})),
    "c": (0.1, frozenset({5, 17, 30})),
    "d": (0.3, frozenset({1, 2, 3, 25})),
}
#: The source row of each Analysis whose SE is a full log-unit off its model,
#: which only an exact exception holds.
EXACT = 12
#: Frequencies far from their variant's baseline, which only an EAF exception
#: holds: one in the first Analysis, and so in the sequence's first region, and
#: one in a later region.
EAF_OUTLIERS = {("a", 3), ("c", 7)}


def _build_source(root: Path) -> Path:
    filtered = root / "filtered"
    filtered.mkdir(parents=True)
    lines = [
        "analysis_index\tanalysis_id\ttrait_id\ttrait_chr\ttrait_bp\tn\tfiltered_file\t"
        "source_genome_build"
    ]
    for index, (analysis_id, (offset, unobserved)) in enumerate(ANALYSES.items()):
        se = _expected_se(_FREQS, offset)
        se[EXACT] *= float(np.exp(1.0))
        rows = [
            (
                _source_bp(k),
                _source_z(k) * float(se[k]),
                float(se[k]),
                0.002 if (analysis_id, k) in EAF_OUTLIERS else float(_FREQS[k]),
            )
            for k in range(N_SOURCE)
            if k not in unobserved
        ]
        _write_filtered_ssf(filtered / f"{analysis_id}.tsv.gz", rows)
        lines.append(
            f"{index}\t{analysis_id}\tT{index}\t1\t{_TRAIT_BP}\t50000\t{analysis_id}.tsv.gz\t"
            "GRCh38"
        )
    (root / "manifest.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    store = root / "obs.opengwasdb"
    build_ragged_from_ssf(
        root / "manifest.tsv", filtered, store, store_id="parallel-rag", release_id="obs",
        allow_unverified_eaf=True, association_coverage="cis",
    )
    return store


def _complete(source: Path, panel: Path, dst: Path, n_workers: int) -> Path:
    complete_ragged_store(
        source, dst, panel, ancestry="EUR", cis_window_bp=1_000_000, min_cor=0.0,
        release_id="comp", n_workers=n_workers,
    )
    return dst


def _assert_fixture_is_meaningful(inputs: CompletionInputs) -> None:
    """Fills in at least three Analyses, a residual SE plane with exceptions."""
    assert StoreManifest.load(inputs.source).encoding.se.is_residual
    group = zarr.open_group(str(inputs.serial / RAGGED_ZARR_PATH), mode="r")
    offsets = np.asarray(group["offsets"][:])
    imputed = np.asarray(group["imputed"][:])
    filled = [
        ai for ai in range(len(offsets) - 1) if imputed[offsets[ai] : offsets[ai + 1]].any()
    ]
    assert len(filled) >= 3, filled
    assert int(group["se_exception_index"].shape[0]) >= 1


@pytest.fixture(autouse=True)
def small_units(monkeypatch: pytest.MonkeyPatch) -> None:
    pin_pool_units(monkeypatch)


@pytest.fixture(scope="module")
def completion(tmp_path_factory: pytest.TempPathFactory) -> CompletionInputs:
    root = tmp_path_factory.mktemp("ragged-parallel-write")
    source = _build_source(root / "source")
    panel = write_eaf_carrying_panel(root / "ld_panel")
    serial = _complete(source, panel, root / "serial.opengwasdb", 1)
    inputs = CompletionInputs(source, panel, serial, _complete)
    _assert_fixture_is_meaningful(inputs)
    return inputs


# ── one worker against several, an Analysis finishing out of order ────────

#: Analyses 0 to 2: the first is held until the other two are assembled.
_ORDER = CompletionOrder(3)
#: The assembly's worker this module wraps; ``None`` where there is none.
_REAL_COMPLETE = getattr(complete_module, "_complete_analysis", None)


def _first_analysis_finishing_last(assembly: Any, item: tuple[int, Any]) -> Any:
    ai = item[0]
    if ai == 0:
        _ORDER.wait_for_the_rest()
    assert _REAL_COMPLETE is not None
    completed = _REAL_COMPLETE(assembly, item)
    if ai < 3:
        _ORDER.claim(ai)
    return completed


def test_a_release_completed_on_three_workers_is_the_one_completed_on_one(
    completion: CompletionInputs, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The first Analysis is held until two others are assembled, so a writer
    fed in the order the pool finished would put one Analysis's rows under
    another's offsets."""
    _ORDER.reset()
    monkeypatch.setattr(
        complete_module, "_complete_analysis", _first_analysis_finishing_last, raising=False
    )

    parallel = completion.complete(tmp_path / "parallel.opengwasdb", n_workers=N_WORKERS)

    finished = _ORDER.order()
    assert sorted(finished) == [0, 1, 2], "the Analyses were not assembled on a pool"
    assert finished[-1] == 0
    assert_same_release(completion.serial, parallel)


# ── an interrupted write ───────────────────────────────────────────────────

_FORK = multiprocessing.get_context("fork")
_ASSEMBLED = _FORK.Value("i", 0)
_CRASHED_IN_PID = _FORK.Value("i", 0)
_CRASHING_ANALYSIS = 2


def _analysis_crashing_after_two_others(assembly: Any, item: tuple[int, Any]) -> Any:
    if item[0] == _CRASHING_ANALYSIS:
        deadline = time.monotonic() + CompletionOrder.WAIT_SECONDS
        while _ASSEMBLED.value < 2:
            if time.monotonic() > deadline:
                raise AssertionError("no two Analyses were assembled beside the failing one")
            os.sched_yield()
        _CRASHED_IN_PID.value = os.getpid()
        raise RuntimeError("simulated crash in the assembly")
    assert _REAL_COMPLETE is not None
    completed = _REAL_COMPLETE(assembly, item)
    with _ASSEMBLED.get_lock():
        _ASSEMBLED.value += 1
    return completed


def test_a_write_interrupted_after_some_analyses_resumes_to_the_same_release(
    completion: CompletionInputs, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A worker fails once other Analyses are assembled: the staged release
    goes, every LD block's checkpoint stays, and the resumed completion, which
    assembles and writes every Analysis again, is the release one worker writes."""
    dst = tmp_path / "resumed.opengwasdb"
    _ASSEMBLED.value = 0
    _CRASHED_IN_PID.value = 0
    with monkeypatch.context() as crash:
        crash.setattr(
            complete_module, "_complete_analysis", _analysis_crashing_after_two_others,
            raising=False,
        )
        with pytest.raises(RuntimeError, match="simulated crash in the assembly"):
            completion.complete(dst, n_workers=N_WORKERS)

    assert _ASSEMBLED.value >= 2
    assert _CRASHED_IN_PID.value not in (0, os.getpid()), "the Analysis ran in the parent"
    _assert_resumes_to(dst, completion.serial)


def _assert_resumes_to(dst: Path, serial: Path) -> None:
    """The failed completion left its LD blocks' checkpoints and no staged
    release, and resuming them writes the release one worker writes."""
    checkpoint_dir = checkpoint_dir_for(dst)
    assert list((checkpoint_dir / "blocks").glob("*.npz"))
    assert not dst.exists()
    assert list(dst.parent.glob(f".{dst.name}.tmp*")) == []

    resume_ragged_completion(checkpoint_dir, n_workers=N_WORKERS)

    assert not checkpoint_dir.exists()
    assert_same_release(serial, dst)


# ── the sequence written a region at a time, a region finishing out of order ──

#: A sequence shard of 500 cells, so the fixture's sequence is several regions.
SHARD_CELLS = 500


def _narrow_shards(patch: pytest.MonkeyPatch) -> None:
    """500-cell shards, each its own region: a region is otherwise at least
    `DEFAULT_FLUSH_REGION_CELLS`, which holds the whole fixture."""
    patch.setattr(store_arrays, "RAGGED_SEQUENCE_SHARD_ELEMENTS", SHARD_CELLS)
    patch.setattr(store_arrays, "ASSOCIATION_SEQUENCE_CHUNK", 100)
    patch.setattr(zarr_csr, "sequence_region_step", lambda _total, _cells: SHARD_CELLS)


@pytest.fixture(scope="module")
def narrow_serial(completion: CompletionInputs, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The release one worker writes with 500-cell shards: one region a shard."""
    with pytest.MonkeyPatch.context() as patch:
        _narrow_shards(patch)
        release = completion.complete(
            tmp_path_factory.mktemp("narrow") / "serial.opengwasdb", n_workers=1
        )
    group = zarr.open_group(str(release / RAGGED_ZARR_PATH), mode="r")
    assert tuple(group["z"].shards) == (SHARD_CELLS,)
    assert int(group["z"].shape[0]) > 3 * SHARD_CELLS, "fixture: at least four regions"
    for table in ("eaf_exception_index", "se_exception_index"):
        positions = np.asarray(group[table][:])
        assert (positions < SHARD_CELLS).any() and (positions >= SHARD_CELLS).any(), table
    return release


_REGIONS = CompletionOrder(3)
_REAL_WRITE_REGION = getattr(zarr_csr, "_write_region", None)


def _first_region_finishing_last(plan: Any, region: tuple[int, int]) -> Any:
    index = region[0] // SHARD_CELLS
    if index == 0:
        _REGIONS.wait_for_the_rest()
    assert _REAL_WRITE_REGION is not None
    outcome = _REAL_WRITE_REGION(plan, region)
    if index < 3:
        _REGIONS.claim(index)
    return outcome


def test_a_sequence_written_a_region_a_worker_is_the_one_one_worker_writes(
    completion: CompletionInputs,
    narrow_serial: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The first region is held until two others are written, so a writer that
    took the regions' side-table entries or exception counts in the order the
    pool finished would file them under another region's cells."""
    _narrow_shards(monkeypatch)
    _REGIONS.reset()
    monkeypatch.setattr(zarr_csr, "_write_region", _first_region_finishing_last, raising=False)

    parallel = completion.complete(tmp_path / "parallel.opengwasdb", n_workers=N_WORKERS)

    finished = _REGIONS.order()
    assert sorted(finished) == [0, 1, 2], "the regions were not written on a pool"
    assert finished[-1] == 0
    assert_same_release(narrow_serial, parallel)


# ── refusals, and the region pool's size ──────────────────────────────────


def test_a_completed_analysis_out_of_place_is_refused(
    completion: CompletionInputs, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An Analysis's rows that name another Analysis are refused, not appended
    under the offsets of the one the writer expects next."""
    real = complete_module._complete_analysis

    def mislabelled(assembly: Any, item: tuple[int, Any]) -> Any:
        ai, completed = real(assembly, item)
        return ai + 1, completed

    monkeypatch.setattr(complete_module, "_complete_analysis", mislabelled)
    with pytest.raises(RuntimeError, match="Analysis 1 arrived where Analysis 0 belongs"):
        completion.complete(tmp_path / "refused.opengwasdb", n_workers=1)


def test_a_region_outcome_out_of_place_is_refused(
    completion: CompletionInputs, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A region's table runs and counts that name another region are refused."""
    _narrow_shards(monkeypatch)
    real = zarr_csr._write_region

    def mislabelled(plan: Any, region: tuple[int, int]) -> Any:
        return replace(real(plan, region), lo=region[0] + 1)

    monkeypatch.setattr(zarr_csr, "_write_region", mislabelled)
    with pytest.raises(RuntimeError, match="region 1 arrived where region 0 belongs"):
        completion.complete(tmp_path / "refused.opengwasdb", n_workers=1)


def test_the_region_pool_is_no_larger_than_its_regions_or_the_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """At most one worker a region, and no more than the memory budget holds:
    16 GiB available is 4 GiB, and a parent of 1 GiB leaves 3 GiB for workers at
    the pinned 16 MiB plus 120 bytes a cell of a 2^23-cell region, each with
    the two results the parent may hold for it."""
    monkeypatch.setattr(memory, "resident_bytes", lambda path=memory.SELF_STATUS: 1024**3)
    monkeypatch.setattr(
        memory, "available_memory_bytes", lambda path=memory.MEMINFO: 16 * 1024**3
    )
    regions = [(0, 500), (500, 1000), (1000, 1200)]
    assert zarr_csr._region_workers(8, regions, 4) == 3
    assert zarr_csr._region_workers(2, regions, 4) == 2
    assert zarr_csr._region_workers(8, regions[:1], 4) == 1
    cells = 1 << 23
    big = [(k * cells, (k + 1) * cells) for k in range(8)]
    per_region = (
        zarr_csr._REGION_FIXED_BYTES
        + zarr_csr._REGION_BYTES_PER_CELL * cells
        + memory.RESULTS_HELD_PER_WORKER
        * (zarr_csr._REGION_RESULT_BYTES_PER_ANALYSIS * 4 + cells // 40)
    )
    assert zarr_csr._region_workers(8, big, 4) == min(8, (3 * 1024**3) // per_region)


def test_a_flush_of_regions_refuses_a_residual_plan_with_no_model(tmp_path: Path) -> None:
    """Regions code SE against a model they are given; a writer that would fit
    its own needs the whole frequency plane first, which `flush` writes."""
    writer = RaggedCSRWriter(4)
    writer.add_analysis(
        np.arange(4), np.ones(4), np.full(4, 0.1), eaf=np.full(4, 0.3, np.float32)
    )
    plan = StoreEncoding(
        z=ZEncoding("float16"), se=SeEncoding("int8_residual", 0.5), eaf=EafEncoding("float32")
    )
    with pytest.raises(ValueError, match="codes SE against a given model"):
        writer.flush_regions(tmp_path / "refused.opengwasdb", plan, n_workers=2)


_REGIONS_WRITTEN = _FORK.Value("i", 0)
_CRASHED_REGION_PID = _FORK.Value("i", 0)
_CRASHING_REGION = 2


def _region_crashing_after_two_others(plan: Any, region: tuple[int, int]) -> Any:
    """The region writer, failing on one region once two others are in the planes."""
    if region[0] // SHARD_CELLS == _CRASHING_REGION:
        deadline = time.monotonic() + CompletionOrder.WAIT_SECONDS
        while _REGIONS_WRITTEN.value < 2:
            if time.monotonic() > deadline:
                raise AssertionError("no two regions were written beside the failing one")
            os.sched_yield()
        _CRASHED_REGION_PID.value = os.getpid()
        raise RuntimeError("simulated crash in the CSR write")
    assert _REAL_WRITE_REGION is not None
    outcome = _REAL_WRITE_REGION(plan, region)
    with _REGIONS_WRITTEN.get_lock():
        _REGIONS_WRITTEN.value += 1
    return outcome


def test_a_sequence_write_interrupted_after_some_regions_resumes_to_the_same_release(
    completion: CompletionInputs,
    narrow_serial: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A worker fails once two regions' planes are on disk: the staged release
    goes with them, the LD blocks' checkpoints stay, and the resumed completion
    is the release one worker writes."""
    _narrow_shards(monkeypatch)
    dst = tmp_path / "resumed.opengwasdb"
    _REGIONS_WRITTEN.value = 0
    _CRASHED_REGION_PID.value = 0
    with monkeypatch.context() as crash:
        crash.setattr(zarr_csr, "_write_region", _region_crashing_after_two_others)
        with pytest.raises(RuntimeError, match="simulated crash in the CSR write"):
            completion.complete(dst, n_workers=N_WORKERS)

    assert _REGIONS_WRITTEN.value >= 2
    assert _CRASHED_REGION_PID.value not in (0, os.getpid()), "the region ran in the parent"
    _assert_resumes_to(dst, narrow_serial)
