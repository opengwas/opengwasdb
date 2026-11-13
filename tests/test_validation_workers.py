"""`validate --n-workers N` spreads each Dense grid's band pass across forked workers (#315).

The parent reads the first band itself, so what the folds load on their first block
is loaded once and inherited, then hands the rest to the ordered fork pool, one band
a task, and adds each band's outcomes in band order. The report is the one a single
process makes, rides included: the subset digests, the EAF exception table's spans,
the Top-Hit folds. This file holds the pool's own behaviour and the Hybrid cases;
`test_validation_workers_dense` and `_tables` run the other cases of
`worker_passes`.
"""

from __future__ import annotations

import errno
import json
import multiprocessing
import os
import signal
import subprocess
import sys
from pathlib import Path
from shutil import copytree

import numpy as np
import pytest
import zarr
from dense_sources import build_dense_from_records
from eaf_tables import table_records
from pool_budget import FIXTURE_UNIT_BYTES
from worker_passes import (
    COMPLETED,
    MIN_BANDS,
    RICH,
    WORKERS,
    Case,
    _bands,
    _failed,
    _Finished,
    _grid,
    _narrow,
    _reads_bands,
    _report,
    assert_the_same_report,
    cases_of,
    subset_cell,
    worker_floor_pinned,
)

from opengwasdb.encoding import banding, memory
from opengwasdb.encoding.codec import SE_EXCEPTION
from opengwasdb.layouts.dense import indexed_subsets
from opengwasdb.store import arrays, open_store
from opengwasdb.store.arrays import open_group
from opengwasdb.validation import run, scans, validate_store
from opengwasdb.validation.rule import Kind, ValidationOptions
from opengwasdb.validation.validate import release_targets

_pinned = worker_floor_pinned()
case = cases_of("hybrid_seed")


def test_the_report_is_the_same_at_one_and_several_workers(case: tuple[Path, Case]) -> None:
    assert_the_same_report(*case)


# ── The reduction is in band order ───────────────────────────────────────────


def test_bands_are_added_in_band_order_whatever_order_workers_finish_in(
    tmp_path: Path, small_seed: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two bands cannot be read, each in its own words; the first of them in band order
    is made to finish after the second. The report names the first: added in the
    order the workers finished, it would name the second."""
    store = copytree(small_seed, tmp_path / "store.opengwasdb")
    n_bands = 2 * WORKERS + 1
    rows = int(_grid(store)["z"].shape[0]) // n_bands + 1
    _narrow(monkeypatch, store, rows)
    assert _bands(store) == [n_bands], "every band past the parent's must be in flight at once"
    unreadable = frozenset({2, 5})
    _reads_bands(monkeypatch, rows, _Finished(n_bands), unreadable=unreadable)
    serial = _report(store, 1)
    finished = _Finished(n_bands)
    _reads_bands(monkeypatch, rows, finished, slow=2, after=5, unreadable=unreadable)

    spread = _report(store, WORKERS)

    done = list(finished.order[: finished.count.value])
    assert done.index(5) < done.index(2), f"band 2 must finish after band 5: {done}"
    assert {pid for pid in finished.pids if pid} - {os.getpid()}, "workers must read bands"
    [message] = [f[3] for f in serial if f[0] == "associations.decodable"]
    assert message == "validation failed: band 2's chunks cannot be read"
    assert spread == serial


def test_a_subset_cell_a_later_worker_reads_is_still_found(
    tmp_path: Path, rich_seed: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The worker that takes the first band it is handed is held until the band with
    the corrupt cell has been read, so a second worker reads it."""
    store = copytree(rich_seed, tmp_path / "store.opengwasdb")
    subset_cell("z", 42, 1)(store)
    _narrow(monkeypatch, store, RICH)
    n_bands = _bands(store)[0]
    corrupt = 42 // RICH
    assert 1 < corrupt <= 2 * WORKERS, "the corrupt band must be in flight with the first"
    serial = _report(store, 1)
    finished = _Finished(n_bands)
    _reads_bands(monkeypatch, RICH, finished, slow=1, after=corrupt)

    report = _report(store, WORKERS)

    assert finished.pids[corrupt] not in (0, os.getpid(), finished.pids[1])
    assert _failed(report) == {"subsets.indexed"}
    assert report == serial


def test_the_run_s_digest_key_is_the_workers_key(
    tmp_path: Path, completed_seed: Path, monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The key is drawn once, where the pass is driven, and every worker digests under
    it: a worker drawing its own would disagree with the index's digests, and every
    unit would be compared cell by cell."""
    store = copytree(completed_seed, tmp_path / "store.opengwasdb")
    _narrow(monkeypatch, store, COMPLETED)
    monkeypatch.setattr(indexed_subsets, "_physical_write_unit", lambda _array: 20)
    monkeypatch.setattr(indexed_subsets, "EXHAUSTIVE_VALIDATION_BAND_CELLS", 20)
    assert min(_bands(store)) >= MIN_BANDS

    with caplog.at_level("INFO", logger=indexed_subsets.__name__):
        result = validate_store(store, n_workers=WORKERS)

    assert result.ok, result.errors
    lines = [r.getMessage() for r in caplog.records if "compared cell by cell" in r.getMessage()]
    assert lines, "the comparison must say how many units it compared cell by cell"
    vouched = "0 of 8 units compared cell by cell (0 whose digests disagreed, 0 the pass "
    assert all(vouched in line for line in lines), lines


def test_a_worker_that_dies_fails_the_run_naming_its_band(
    tmp_path: Path, rich_seed: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A worker killed mid-band -- for memory, say -- fails the run: no Store finding is
    made of a read that never finished, and the band it was reading is named."""
    store = copytree(rich_seed, tmp_path / "store.opengwasdb")
    _narrow(monkeypatch, store, RICH)
    assert min(_bands(store)) >= MIN_BANDS
    read, driver = scans.GridSplit.read, os.getpid()

    def dies(self: scans.GridSplit, unit: scans.GridBand) -> scans.Block:
        if unit.r0 == RICH and os.getpid() != driver:
            os._exit(1)
        return read(self, unit)

    monkeypatch.setattr(scans.GridSplit, "read", dies)

    awaited = r"waiting for the result of band 1 \(rows 20 to 40\)"
    with pytest.raises(run.WorkerLost, match=awaited):
        validate_store(store, n_workers=WORKERS)


def test_a_fork_the_host_refuses_fails_the_run(
    tmp_path: Path, rich_seed: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A host out of processes refuses the pool's fork: the run fails, naming the
    band it was about to hand out; no Store finding is made of a read never begun."""
    store = copytree(rich_seed, tmp_path / "store.opengwasdb")
    _narrow(monkeypatch, store, RICH)
    assert min(_bands(store)) >= MIN_BANDS

    def refused() -> int:
        raise BlockingIOError(errno.EAGAIN, "Resource temporarily unavailable")

    monkeypatch.setattr(os, "fork", refused)

    awaited = r"pool failed \(BlockingIOError.*waiting for the result of band 1 \(rows 20 to 40\)"
    with pytest.raises(run.PoolFailure, match=awaited):
        validate_store(store, n_workers=WORKERS)


class _Reads:
    """Every read of a Store array, by the process that read it and the one that
    opened the handle it read through, across the fork pool."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        context = multiprocessing.get_context("fork")
        self.in_workers = context.Value("i", 0)
        self.foreign = context.Value("i", 0)
        self.driver = os.getpid()
        opened_in: dict[int, int] = {}
        get, read = zarr.Group.__getitem__, zarr.Array.__getitem__

        def opened(group: zarr.Group, key: str) -> object:
            member = get(group, key)
            opened_in[id(member)] = os.getpid()
            return member

        def counted(array: zarr.Array, selection: object) -> object:
            if os.getpid() != self.driver:
                with self.in_workers.get_lock():
                    self.in_workers.value += 1
                if opened_in.get(id(array), os.getpid()) != os.getpid():
                    with self.foreign.get_lock():
                        self.foreign.value += 1
            return read(array, selection)

        monkeypatch.setattr(zarr.Group, "__getitem__", opened)
        monkeypatch.setattr(zarr.Array, "__getitem__", counted)


def test_a_worker_reads_only_through_handles_it_opened(
    tmp_path: Path, rich_seed: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The planes, and the side tables the codecs look exceptions up in: a worker
    reads every array through a handle it opened after the fork. The rich release's
    SE exception sits in its second band, which a worker reads."""
    store = copytree(rich_seed, tmp_path / "store.opengwasdb")
    _narrow(monkeypatch, store, RICH)
    se = _grid(store)["se"]
    assert (np.asarray(se[RICH : 2 * RICH]) == SE_EXCEPTION).any(), "band 1 must look one up"
    reads = _Reads(monkeypatch)

    validate_store(store, n_workers=WORKERS)

    assert reads.in_workers.value > 0, "workers must read the planes"
    assert reads.foreign.value == 0, f"{reads.foreign.value} reads through the driver's handles"


def test_no_workers_is_refused_before_anything_is_read(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="n_workers must be at least 1, got 0"):
        validate_store(tmp_path / "absent.opengwasdb", n_workers=0)


# ── The pool is sized by the host's memory ───────────────────────────────────


def _split(store: Path) -> scans.GridSplit:
    [target] = [
        t for t in release_targets(open_store(store), ValidationOptions()) if t.kind is Kind.GRID
    ]
    return scans.GridSplit(target, scans._grid_limits(target))


@pytest.mark.parametrize(("room", "workers"), [(2.5, 2), (None, 1)], ids=["two-fit", "unread"])
def test_the_host_s_memory_caps_the_pool_and_not_the_report(
    tmp_path: Path, rich_seed: Path, monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture, room: float | None, workers: int,
) -> None:
    """Each worker holds a band and its floor beside the parent, and the parent the
    outcomes in flight: a quarter of the host's available memory holds what it
    holds of them, and an unreadable host gets one worker. The report does not move."""
    store = copytree(rich_seed, tmp_path / "store.opengwasdb")
    subset_cell("z", 200, 1)(store)
    _narrow(monkeypatch, store, RICH)
    split = _split(store)
    per_worker = split.unit_bytes + memory.RESULTS_HELD_PER_WORKER * split.result_bytes
    assert split.unit_bytes > FIXTURE_UNIT_BYTES, "a worker must hold its band"
    available = None if room is None else int(memory.MEMORY_BUDGET_DIVISOR * room * per_worker)
    monkeypatch.setattr(memory, "available_memory_bytes", lambda: available)
    monkeypatch.setattr(memory, "resident_bytes", lambda: 0)

    with caplog.at_level("INFO", logger=memory.__name__):
        spread = _report(store, WORKERS)

    sized = [r.getMessage() for r in caplog.records if "worker(s);" in r.getMessage()]
    assert any(f": {workers} of {WORKERS} worker(s); a band is" in line for line in sized), sized
    assert spread == _report(store, 1)


# ── Forked workers read multi-shard bands (ADR 0056) ─────────────────────────

#: Shards a band reads, side by side in one shard row, as OGS-00009's bands cross
#: its shard columns: 2,000 rows by one Analysis, of inner chunks 100 by one.
PROBE_SHARD = (2_000, 1)
PROBE_INNER = (100, 1)
PROBE_ROWS = 400
#: Long enough for a cold interpreter to import, validate twice and fork; a worker
#: that hangs never finishes, so the margin hides nothing.
PROBE_TIMEOUT_S = 120

_PROBE = """
import json
import logging
import sys
from pathlib import Path

from opengwasdb.store import arrays
from opengwasdb.encoding import banding
from opengwasdb.validation import scans, validate_store

# The pool's sizing line goes to stderr, where the test reads how many workers forked.
logging.basicConfig(level=logging.INFO, stream=sys.stderr)
arrays.DENSE_SHARD_SHAPE = tuple(json.loads(sys.argv[2]))
banding.DECODED_BAND_BYTES = int(sys.argv[3])
# A worker's floor at a fixture's size here too, so a small host still forks them all.
scans.WORKER_FLOOR_BYTES = int(sys.argv[5])
store = Path(sys.argv[1])
reports = [
    [[f.rule, f.severity.value, f.target, f.message] for f in validate_store(
        store, n_workers=n
    ).findings]
    for n in (1, int(sys.argv[4]))
]
print(json.dumps(reports))
"""


def test_forked_workers_read_bands_of_several_shards_without_hanging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """In a fresh interpreter, as ADR 0056's fork test runs: the parent reads its band
    through the codec pipeline before the pool forks, and every worker's first read
    spans as many shards as the parent's did. With the pipeline's own thread pool
    (`codec_pipeline.max_workers` above one) that read waits on threads the fork did
    not copy, and never returns; at the seam's one worker it finishes."""
    monkeypatch.setattr(arrays, "DENSE_SHARD_SHAPE", PROBE_SHARD)
    store = tmp_path / "store.opengwasdb"
    build_dense_from_records(
        table_records(4_000), store, store_id="probe", release_id="probe-v1",
        reference_assembly="GRCh38", chunk_shape=PROBE_INNER,
    )
    z = open_group(store / "data.zarr")["z"]
    n_variants, width = (int(size) for size in z.shape)
    assert tuple(z.shards) == PROBE_SHARD and tuple(z.chunks) == PROBE_INNER
    assert width > PROBE_SHARD[1], "a band must read several shards side by side"
    assert PROBE_SHARD[0] % PROBE_ROWS == 0, "no band may cross a shard row"
    assert n_variants // PROBE_ROWS >= MIN_BANDS
    budget = banding.BYTES_PER_CELL * width * PROBE_ROWS
    script = tmp_path / "probe.py"
    script.write_text(_PROBE, encoding="utf-8")
    floor = str(FIXTURE_UNIT_BYTES)
    argv = [str(store), json.dumps(PROBE_SHARD), str(budget), str(WORKERS), floor]
    # A session of its own, so a timeout kills the probe's forked workers too.
    probe = subprocess.Popen(
        [sys.executable, str(script), *argv],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True,
    )
    try:
        stdout, stderr = probe.communicate(timeout=PROBE_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        os.killpg(probe.pid, signal.SIGKILL)
        probe.communicate()
        pytest.fail(f"validation at {WORKERS} workers did not finish: a forked reader hung")
    assert probe.returncode == 0, stderr
    assert f": {WORKERS} of {WORKERS} worker(s);" in stderr, "the probe must fork its workers"
    serial, spread = json.loads(stdout.strip().splitlines()[-1])
    assert spread == serial
